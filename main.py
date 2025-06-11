import asyncio
import httpx
import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Dict

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
    version="3.2.0" # Version with configurable parameters
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
    """
    Sends the inappropriate image to the censorship service for processing
    using parameters from environment variables.
    """
    url = "https://safeimage.adminai.ir/process"
    
    # Build form data from settings, converting bools to lowercase strings ("true"/"false")
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
            base64_image_str = result_data["blurred_image"]

            if "," in base64_image_str:
                base64_image_str = base64_image_str.split(',')[1]

            missing_padding = len(base64_image_str) % 4
            if missing_padding:
                base64_image_str += '=' * (4 - missing_padding)
            
            try:
                return base64.b64decode(base64_image_str)
            except base64.BinasciiError as e:
                raise HTTPException(status_code=500, detail=f"Failed to decode Base64 string even after fixing: {e}")

        detail = result_data.get("message", "Censorship service returned an invalid format.")
        raise HTTPException(status_code=500, detail=f"{detail} Response: {result_data}")

    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"Censorship service returned an error: {e.response.status_code}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred with the censorship service: {e}")


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
        uploaded_id = response_json.get("id")
        uploaded_url = response_json.get("urls", {}).get("primary")

        if uploaded_id is not None and uploaded_url is not None:
            return {"id": str(uploaded_id), "url": uploaded_url}
        
        raise KeyError("ID or primary URL not found in upload response.")
    except (httpx.RequestError, KeyError, IndexError) as e:
        raise HTTPException(status_code=503, detail=f"Could not connect to or parse response from Basalam upload service: '{e}'")

async def process_single_image(session: httpx.AsyncClient, image_url: str, is_forbidden: bool) -> Dict[str, str]:
    """A complete processing cycle for a single image: download, censor (if needed), and upload."""
    try:
        download_response = await session.get(image_url, timeout=30.0)
        download_response.raise_for_status()
        image_data = download_response.content
        content_type = download_response.headers.get('content-type', 'image/jpeg')

        final_image_data = image_data
        if is_forbidden:
            final_image_data = await censor_image(session, image_data, content_type)

        upload_result = await upload_image_to_basalam(session, final_image_data)
        return upload_result

    except httpx.RequestError as e:
        return {"error": f"FAILED_DOWNLOAD:{image_url}", "details": str(e)}
    except HTTPException as e:
        return {"error": f"PROCESSING_FAILED:{image_url}", "details": e.detail}
    except Exception as e:
        return {"error": f"UNEXPECTED_ERROR:{image_url}", "details": str(e)}

# --- 5. تعریف Endpoint اصلی API ---
@app.post("/process-images", response_model=ImageProcessResponse, summary="پردازش و آپلود دسته‌ای تصاویر")
async def process_images_endpoint(request: ImageProcessRequest):
    """
    این اندپوینت لیستی از لینک‌های تصاویر را دریافت کرده، آن‌ها را بررسی و در صورت نیاز سانسور می‌کند،
    سپس در باسلام آپلود کرده و در نهایت لیستی از آبجکت‌های حاوی ID و URL تصاویر نهایی را برمی‌گرداند.
    """
    if not request.photo_links:
        return {"processed_images": []}

    async with httpx.AsyncClient() as session:
        revision_results = await check_images_revision(session, request.photo_links)
        id_to_url_map = {index: url for index, url in enumerate(request.photo_links)}
        url_to_forbidden_status = {}
        for item in revision_results:
            file_id = item.get('file_id')
            if file_id is not None:
                original_url = id_to_url_map.get(file_id)
                if original_url:
                    url_to_forbidden_status[original_url] = item.get('is_forbidden', True)
        
        tasks = [
            process_single_image(session, url, url_to_forbidden_status.get(url, True))
            for url in request.photo_links
        ]
        
        final_results = await asyncio.gather(*tasks)

    return {"processed_images": final_results}