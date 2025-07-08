# main.py

import asyncio
import httpx
import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Dict, Optional

# --- 1. پیکربندی و خواندن متغیرهای محیطی ---
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8')
    
    # Basalam Tokens
    basalam_upload_token: str
    revision_api_token: str

    # SafeImage Parameters with default values
    SAFEIMAGE_SIGMA_BLUR: int = 25
    SAFEIMAGE_FEATHER: int = 5
    SAFEIMAGE_ONLY_CLOTHES: bool = False
    SAFEIMAGE_BLUR_FACE: bool = True
    SAFEIMAGE_BLUR_HAIR: bool = True
    SAFEIMAGE_BLUR_ARMS: bool = True
    SAFEIMAGE_BLUR_LEGS: bool = True
    SAFEIMAGE_BLUR_TORSO: bool = False
    SAFEIMAGE_SHOW_MASK: bool = False


settings = Settings()

# --- 2. مدل‌های داده (Pydantic) برای ورودی و خروجی API ---
class UploadedImageInfo(BaseModel):
    id: str
    url: str

class ImageProcessRequest(BaseModel):
    photo_links: List[str] = Field(..., description="لیستی از لینک‌های تصاویری که باید پردازش شوند.")

class ImageProcessResponse(BaseModel):
    processed_images: List[Dict] = Field(..., description="لیست تصاویر پردازش و آپلود شده (شامل خطاها).")

# --- 3. ساخت نمونه اصلی برنامه FastAPI ---
app = FastAPI(
    title="Basalam Image Processing Service",
    description="سرویسی برای بررسی، سانسور و آپلود تصاویر در باسلام.",
    version="3.4.0" # Version with safe censorship-failure handling
)

# --- 4. توابع کمکی برای ارتباط با سرویس‌های دیگر ---

async def check_images_revision(session: httpx.AsyncClient, image_urls: List[str]) -> List[Dict]:
    """لیستی از تصاویر را برای بررسی به سرویس باسلام ارسال می‌کند."""
    url = "https://revision.basalam.com/api_v1.0/validation/image/hijab-detector/bulk"
    headers = {"api-token": settings.revision_api_token}
    payload = {"images": [{"file_id": index, "url": img_url} for index, img_url in enumerate(image_urls)]}
    
    try:
        response = await session.post(url, headers=headers, json=payload, timeout=30.0)
        response.raise_for_status()
        return response.json()
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"سرویس بررسی تصویر در دسترس نیست: {e}")

async def censor_image(session: httpx.AsyncClient, image_data: bytes, content_type: str) -> bytes:
    """Sends the inappropriate image to the censorship service for processing."""
    url = "https://safeimage.adminai.ir/process"
    form_data = {
        "sigma_blur": str(settings.SAFEIMAGE_SIGMA_BLUR),
        "feather": str(settings.SAFEIMAGE_FEATHER),
        "only_clothes": str(settings.SAFEIMAGE_ONLY_CLOTHES).lower(),
        "blur_face": str(settings.SAFEIMAGE_BLUR_FACE).lower(),
        "blur_hair": str(settings.SAFEIMAGE_BLUR_HAIR).lower(),
        "blur_arms": str(settings.SAFEIMAGE_BLUR_ARMS).lower(),
        "blur_legs": str(settings.SAFEIMAGE_BLUR_LEGS).lower(),
        "blur_torso": str(settings.SAFEIMAGE_BLUR_TORSO).lower(),
        "show_mask": str(settings.SAFEIMAGE_SHOW_MASK).lower(),
    }
    files = {"image": ("image.jpg", image_data, content_type)}

    try:
        response = await session.post(url, data=form_data, files=files, timeout=45.0)
        response.raise_for_status()
        result_data = response.json()
        if result_data.get("success") is True and "blurred_image" in result_data:
            base64_image_str = result_data["blurred_image"].split(',')[-1]
            return base64.b64decode(base64_image_str + '==')
        detail = result_data.get("message", "Censorship service returned an invalid format.")
        raise HTTPException(status_code=500, detail=detail)
    except (httpx.RequestError, httpx.HTTPStatusError, base64.BinasciiError) as e:
        raise HTTPException(status_code=502, detail=f"Censorship service error: {e}")

async def upload_image_to_basalam(session: httpx.AsyncClient, image_data: bytes) -> Dict[str, str]:
    """تصویر نهایی را آپلود کرده و یک دیکشنری حاوی ID و URL برمی‌گرداند."""
    url = "https://uploadio.basalam.com/v3/files"
    headers = {"authorization": f"{settings.basalam_upload_token}"}
    files = {"file": ("image.jpg", image_data, "image/jpeg")}
    form_data = {"file_type": "product.photo"}

    try:
        response = await session.post(url, headers=headers, files=files, data=form_data, timeout=45.0)
        response.raise_for_status()
        response_json = response.json()
        return {"id": str(response_json["id"]), "url": response_json["urls"]["primary"]}
    except (httpx.RequestError, KeyError, IndexError) as e:
        raise HTTPException(status_code=503, detail=f"Basalam upload service error: {e}")

# ✨ --- تابع اصلی پردازش تصویر (اصلاح شده) --- ✨
async def process_single_image(session: httpx.AsyncClient, image_url: str, is_forbidden: bool) -> Optional[Dict[str, str]]:
    """
    Processes a single image. If it's forbidden and censorship fails, returns None.
    """
    try:
        download_response = await session.get(image_url, timeout=30.0)
        download_response.raise_for_status()
        image_data = download_response.content
        content_type = download_response.headers.get('content-type', 'image/jpeg')

        if is_forbidden:
            try:
                final_image_data = await censor_image(session, image_data, content_type)
                print(f"INFO: Successfully censored image: {image_url}")
            except HTTPException as e:
                # اگر سانسور خطا داد، تصویر را نادیده بگیر
                print(f"WARNING: Censorship failed for forbidden image {image_url}. Skipping. Reason: {e.detail}")
                return None
        else:
            final_image_data = image_data

        return await upload_image_to_basalam(session, final_image_data)

    except httpx.RequestError as e:
        print(f"ERROR: Download failed for {image_url}. Skipping. Reason: {e}")
        return {"error": f"FAILED_DOWNLOAD:{image_url}", "details": str(e)} # Or return None if you want to skip download errors too
    except Exception as e:
        print(f"ERROR: An unexpected error occurred for {image_url}. Skipping. Reason: {e}")
        return {"error": f"UNEXPECTED_ERROR:{image_url}", "details": str(e)}

# ✨ --- اندپوینت اصلی (اصلاح شده) --- ✨
@app.post("/process-images", response_model=ImageProcessResponse, summary="پردازش و آپلود دسته‌ای تصاویر")
async def process_images_endpoint(request: ImageProcessRequest):
    """
    این اندپوینت لیستی از لینک‌های تصاویر را دریافت کرده، آن‌ها را بررسی و در صورت نیاز سانسور می‌کند،
    سپس در باسلام آپلود کرده و در نهایت لیستی از آبجکت‌های حاوی ID و URL تصاویر نهایی را برمی‌گرداند.
    تصاویری که نیاز به سانسور داشته باشند و در این فرآیند شکست بخورند، از لیست نهایی حذف می‌شوند.
    """
    if not request.photo_links:
        return {"processed_images": []}

    async with httpx.AsyncClient() as session:
        revision_results = await check_images_revision(session, request.photo_links)
        id_to_url_map = {index: url for index, url in enumerate(request.photo_links)}
        url_to_forbidden_status = {
            id_to_url_map.get(item.get('file_id')): item.get('is_forbidden', True)
            for item in revision_results if item.get('file_id') in id_to_url_map
        }
        
        tasks = [
            process_single_image(session, url, url_to_forbidden_status.get(url, True))
            for url in request.photo_links
        ]
        
        results_with_none = await asyncio.gather(*tasks)
        
        # فیلتر کردن نتایج None تا در خروجی نهایی قرار نگیرند
        final_results = [result for result in results_with_none if result is not None]

    return {"processed_images": final_results}