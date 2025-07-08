# main.py

import asyncio
import httpx
import base64
from functools import wraps
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Dict, Optional, Any, Callable, Coroutine

# --- 0. ثابت‌های جدید برای تلاش مجدد ---
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 1

# ✨ --- دکوریتور جدید برای تلاش مجدد --- ✨
def async_retry(max_retries: int = MAX_RETRIES, delay: int = RETRY_DELAY_SECONDS):
    """
    A decorator to retry an async function if it raises an exception.
    """
    def decorator(func: Callable[..., Coroutine[Any, Any, Any]]):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except (httpx.RequestError, httpx.HTTPStatusError, HTTPException) as e:
                    if attempt == max_retries - 1:
                        print(f"ERROR: All {max_retries} retries failed for {func.__name__}. Re-raising exception.")
                        raise e  # Re-raise the last exception after all retries fail
                    
                    print(f"WARNING: Attempt {attempt + 1}/{max_retries} failed for {func.__name__}. Retrying...")
                    await asyncio.sleep(delay)
        return wrapper
    return decorator


# --- 1. پیکربندی و خواندن متغیرهای محیطی ---
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8')
    basalam_upload_token: str
    revision_api_token: str
    # ... سایر تنظیمات
    
settings = Settings()

# --- 2. مدل‌های داده (Pydantic) ---
class ImageProcessRequest(BaseModel):
    photo_links: List[str]

class ImageProcessResponse(BaseModel):
    processed_images: List[Dict]

# --- 3. ساخت نمونه اصلی برنامه FastAPI ---
app = FastAPI(
    title="Fully Resilient Basalam Image Processor",
    description="سرویسی مقاوم برای پردازش تصویر با مکانیزم تلاش مجدد برای تمام سرویس‌های خارجی.",
    version="4.0.0" # Version with Global Retry Decorator
)


# --- 4. توابع کمکی (با دکوریتور تلاش مجدد) ---

@async_retry()
async def download_image(session: httpx.AsyncClient, image_url: str) -> tuple[bytes, Optional[str]]:
    """Downloads an image and returns its content and content-type."""
    response = await session.get(image_url, timeout=30.0)
    response.raise_for_status()
    image_data = response.content
    content_type = response.headers.get('content-type', 'image/jpeg')
    return image_data, content_type

@async_retry()
async def check_images_revision(session: httpx.AsyncClient, image_urls: List[str]) -> List[Dict]:
    """لیستی از تصاویر را برای بررسی به سرویس باسلام ارسال می‌کند."""
    url = "https://revision.basalam.com/api_v1.0/validation/image/hijab-detector/bulk"
    headers = {"api-token": settings.revision_api_token}
    payload = {"images": [{"file_id": index, "url": img_url} for index, img_url in enumerate(image_urls)]}
    response = await session.post(url, headers=headers, json=payload, timeout=30.0)
    response.raise_for_status()
    return response.json()

@async_retry()
async def censor_image(session: httpx.AsyncClient, image_data: bytes, content_type: Optional[str]) -> bytes:
    """Sends an image to the censorship service."""
    # ... (کد داخلی این تابع بدون تغییر باقی می‌ماند)
    url = "https://safeimage.adminai.ir/process"
    form_data = { "sigma_blur": "25" } # Simplified for brevity
    files = {"image": ("image.jpg", image_data, content_type)}
    response = await session.post(url, data=form_data, files=files, timeout=45.0)
    response.raise_for_status()
    result_data = response.json()
    if result_data.get("success") is True and "blurred_image" in result_data:
        base64_image_str = result_data["blurred_image"].split(',')[-1]
        return base64.b64decode(base64_image_str + '==')
    raise HTTPException(status_code=500, detail="Censorship failed with valid response.")


@async_retry()
async def upload_image_to_basalam(session: httpx.AsyncClient, image_data: bytes) -> Dict[str, str]:
    """تصویر نهایی را آپلود می‌کند."""
    # ... (کد داخلی این تابع بدون تغییر باقی می‌ماند)
    url = "https://uploadio.basalam.com/v3/files"
    headers = {"authorization": f"{settings.basalam_upload_token}"}
    files = {"file": ("image.jpg", image_data, "image/jpeg")}
    form_data = {"file_type": "product.photo"}
    response = await session.post(url, headers=headers, files=files, data=form_data, timeout=45.0)
    response.raise_for_status()
    response_json = response.json()
    return {"id": str(response_json["id"]), "url": response_json["urls"]["primary"]}


# ✨ --- تابع اصلی پردازش (ساده‌تر شده) --- ✨
async def process_single_image(session: httpx.AsyncClient, image_url: str, is_forbidden: bool) -> Optional[Dict[str, str]]:
    """Processes one image using resilient, retrying helper functions."""
    try:
        image_data, content_type = await download_image(session, image_url)
        final_image_data = image_data

        if is_forbidden:
            try:
                # تلاش مجدد برای سانسور توسط دکوریتور انجام می‌شود
                final_image_data = await censor_image(session, image_data, content_type)
            except (httpx.RequestError, httpx.HTTPStatusError, HTTPException) as e:
                # اگر سانسور بعد از همه تلاش‌ها شکست خورد، تصویر را نادیده بگیر
                print(f"ERROR: Censorship ultimately failed for {image_url}. Skipping. Reason: {e}")
                return None
        
        # تلاش مجدد برای آپلود توسط دکوریتور انجام می‌شود
        return await upload_image_to_basalam(session, final_image_data)

    except Exception as e:
        # این خطاها اکنون فقط برای شکست‌های نهایی (پس از همه تلاش‌ها) رخ می‌دهند
        print(f"ERROR: A final error occurred for {image_url}. Skipping. Reason: {e}")
        return {"error": f"PROCESSING_FAILED:{image_url}", "details": str(e)}


# --- 5. تعریف Endpoint اصلی API (بدون تغییر) ---
@app.post("/process-images", response_model=ImageProcessResponse)
async def process_images_endpoint(request: ImageProcessRequest):
    # ... (کد داخلی این تابع بدون تغییر باقی می‌ماند)
    if not request.photo_links:
        return {"processed_images": []}

    async with httpx.AsyncClient() as session:
        try:
            # تلاش مجدد برای این تابع هم اعمال می‌شود
            revision_results = await check_images_revision(session, request.photo_links)
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Revision service is down. Cannot proceed. Error: {e}")

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
        final_results = [result for result in results_with_none if result is not None]

    return {"processed_images": final_results}