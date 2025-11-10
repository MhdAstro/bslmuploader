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

# Censorship retry settings (less retries since it's slow)
CENSOR_MAX_RETRIES = 1  # Only try once since SafeImage is slow
CENSOR_TIMEOUT = 90  # 90 second timeout for censorship
MAX_CENSORSHIP_ATTEMPTS = 3  # Maximum attempts to censor and verify

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

    # SafeImage censorship parameters (optimized for speed)
    safeimage_sigma_blur: str = "25"
    safeimage_feather: str = "5"
    safeimage_show_mask: str = "false"  # Disabled to reduce response size
    safeimage_only_clothes: str = "false"
    safeimage_blur_face: str = "true"
    safeimage_blur_hair: str = "true"
    safeimage_blur_arms: str = "false"  # Disabled for faster processing
    safeimage_blur_legs: str = "false"  # Disabled for faster processing
    safeimage_blur_torso: str = "false"

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

async def check_images_revision(session: httpx.AsyncClient, image_urls: List[str]) -> List[Dict]:
    """
    Check multiple images with revision service.
    Uses bulk API first, falls back to single API calls if bulk fails (504 timeout).
    """
    bulk_url = "https://revision.basalam.com/api_v1.0/validation/image/hijab-detector/bulk"
    headers = {"api-token": settings.revision_api_token}
    payload = {"images": [{"file_id": index, "url": img_url} for index, img_url in enumerate(image_urls)]}

    try:
        print(f"Checking {len(image_urls)} images with bulk revision API...")
        response = await session.post(bulk_url, headers=headers, json=payload, timeout=30.0)
        response.raise_for_status()
        result = response.json()
        print(f"✓ Bulk API check successful")
        return result

    except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
        # Bulk API failed - fall back to individual single API calls
        print(f"⚠ Bulk API failed ({type(e).__name__}), falling back to single API calls...")

        results = []
        single_url = "https://revision.basalam.com/api_v1.0/validation/image/hijab-detector"

        for index, img_url in enumerate(image_urls):
            try:
                params = {"image_url": img_url}
                response = await session.get(single_url, headers=headers, params=params, timeout=30.0)
                response.raise_for_status()
                result = response.json()

                # Convert single API response to bulk API format
                results.append({
                    "file_id": index,
                    "url": img_url,
                    "is_forbidden": result.get('is_forbidden', True)
                })

            except Exception as single_error:
                print(f"✗ Single API failed for image {index}: {single_error}")
                # Assume forbidden if check fails
                results.append({
                    "file_id": index,
                    "url": img_url,
                    "is_forbidden": True
                })

        print(f"✓ Completed {len(results)} single API checks")
        return results

async def check_single_image_revision_by_url(session: httpx.AsyncClient, image_url: str) -> bool:
    """
    Check a single image with revision service using URL.
    Uses bulk API first, falls back to single API if bulk fails (504 timeout).
    Returns True if forbidden, False if acceptable.
    """
    # Try bulk API first
    bulk_url = "https://revision.basalam.com/api_v1.0/validation/image/hijab-detector/bulk"
    headers = {"api-token": settings.revision_api_token}
    payload = {"images": [{"file_id": 0, "url": image_url}]}

    try:
        print(f"  → Trying bulk revision API...")
        response = await session.post(bulk_url, headers=headers, json=payload, timeout=30.0)
        response.raise_for_status()
        result = response.json()

        if result and len(result) > 0:
            is_forbidden = result[0].get('is_forbidden', True)
            print(f"  → Bulk API result: {'FORBIDDEN' if is_forbidden else 'ACCEPTABLE'}")
            return is_forbidden
        return True  # Default to forbidden if unclear

    except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
        # Bulk API failed (504 timeout or other HTTP error) - fall back to single API
        print(f"  ⚠ Bulk API failed ({type(e).__name__}), trying single API fallback...")

        try:
            # Use single image API as fallback
            single_url = f"https://revision.basalam.com/api_v1.0/validation/image/hijab-detector"
            params = {"image_url": image_url}

            response = await session.get(single_url, headers=headers, params=params, timeout=30.0)
            response.raise_for_status()
            result = response.json()

            # Single API response format: {"is_forbidden": bool, "evaluation_result": {...}}
            is_forbidden = result.get('is_forbidden', True)
            print(f"  ✓ Single API result: {'FORBIDDEN' if is_forbidden else 'ACCEPTABLE'}")
            return is_forbidden

        except Exception as fallback_error:
            print(f"  ✗ Single API also failed: {fallback_error}")
            print(f"  → Assuming FORBIDDEN as safe default")
            return True  # Assume forbidden if both APIs fail

    except Exception as e:
        print(f"  WARNING: Unexpected error in revision check: {e}")
        return True  # Assume forbidden if check fails

@async_retry(max_retries=CENSOR_MAX_RETRIES, delay=2)
async def censor_image(session: httpx.AsyncClient, image_data: bytes, content_type: Optional[str],
                      blur_arms: bool = None, blur_legs: bool = None, blur_torso: bool = None) -> bytes:
    """
    Sends an image to the censorship service with configurable blur parameters.
    Allows overriding specific blur settings for progressive censorship.
    """
    url = "https://safeimage.adminai.ir/process"

    # Use provided parameters or fall back to settings
    use_blur_arms = blur_arms if blur_arms is not None else settings.safeimage_blur_arms == "true"
    use_blur_legs = blur_legs if blur_legs is not None else settings.safeimage_blur_legs == "true"
    use_blur_torso = blur_torso if blur_torso is not None else settings.safeimage_blur_torso == "true"

    # Prepare form data with all SafeImage parameters matching the curl request
    form_data = {
        "sigma_blur": settings.safeimage_sigma_blur,
        "feather": settings.safeimage_feather,
        "show_mask": settings.safeimage_show_mask,
        "only_clothes": settings.safeimage_only_clothes,
        "blur_face": settings.safeimage_blur_face,
        "blur_hair": settings.safeimage_blur_hair,
        "blur_arms": "true" if use_blur_arms else "false",
        "blur_legs": "true" if use_blur_legs else "false",
        "blur_torso": "true" if use_blur_torso else "false",
    }

    # Prepare image file with proper content type
    files = {"image": ("image.jpg", image_data, content_type or "image/jpeg")}

    # Add headers similar to the curl request
    headers = {
        "accept": "*/*",
        "accept-language": "en-US,en;q=0.9,fa;q=0.8",
        "origin": "https://safeimage.adminai.ir",
        "referer": "https://safeimage.adminai.ir/",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }

    # Send request to SafeImage service with extended timeout
    print(f"DEBUG: Sending image to SafeImage (size: {len(image_data)} bytes)...")
    response = await session.post(url, data=form_data, files=files, headers=headers, timeout=CENSOR_TIMEOUT)
    print(f"DEBUG: SafeImage responded with status {response.status_code}")
    response.raise_for_status()

    # Parse response according to the actual structure
    result_data = response.json()

    if result_data.get("success") is True and "blurred_image" in result_data:
        blurred_image = result_data["blurred_image"]

        # Handle base64 with or without data URI prefix
        if blurred_image.startswith("data:"):
            # Remove data URI prefix (e.g., "data:image/jpeg;base64,")
            base64_image_str = blurred_image.split(',', 1)[1]
        else:
            base64_image_str = blurred_image

        # Decode base64 to bytes (handle padding if needed)
        try:
            # Add padding if necessary
            missing_padding = len(base64_image_str) % 4
            if missing_padding:
                base64_image_str += '=' * (4 - missing_padding)
            return base64.b64decode(base64_image_str)
        except Exception as e:
            print(f"ERROR: Failed to decode base64 image from SafeImage. Error: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to decode censored image: {e}")

    # If success is False or blurred_image is missing
    raise HTTPException(status_code=500, detail=f"Censorship service returned unsuccessful response: {result_data}")


async def censor_with_verification(session: httpx.AsyncClient, image_data: bytes, content_type: Optional[str]) -> Optional[Dict[str, str]]:
    """
    Progressive censorship with verification loop.
    Tries multiple censorship levels until the image is acceptable or max attempts reached.

    Censorship Levels:
    - Level 1: Face + Hair only (fast)
    - Level 2: Face + Hair + Arms + Legs (moderate)
    - Level 3: Face + Hair + Arms + Legs + Torso (strict, everything)

    Returns upload result dict {"id": ..., "url": ...} if successful, None if all attempts failed.
    """

    censorship_levels = [
        {"name": "Level 1 (Face + Hair)", "blur_arms": False, "blur_legs": False, "blur_torso": False},
        {"name": "Level 2 (+ Arms + Legs)", "blur_arms": True, "blur_legs": True, "blur_torso": False},
        {"name": "Level 3 (+ Torso - Everything)", "blur_arms": True, "blur_legs": True, "blur_torso": True},
    ]

    for attempt, level in enumerate(censorship_levels, 1):
        try:
            print(f"  Attempt {attempt}/{len(censorship_levels)}: Applying {level['name']}...")

            # Apply censorship with current level
            censored_data = await censor_image(
                session, image_data, content_type,
                blur_arms=level['blur_arms'],
                blur_legs=level['blur_legs'],
                blur_torso=level['blur_torso']
            )

            print(f"  ✓ Censorship applied successfully (size: {len(censored_data)} bytes)")

            # Upload censored image to get URL for verification
            print(f"  → Uploading censored image to get URL for verification...")
            upload_result = await upload_image_to_basalam(session, censored_data)
            censored_url = upload_result["url"]
            print(f"  ✓ Uploaded to: {censored_url}")

            # Verify with revision service using URL (avoids 413 error)
            print(f"  → Checking censored image with revision service...")
            is_forbidden = await check_single_image_revision_by_url(session, censored_url)

            if not is_forbidden:
                print(f"  ✓ SUCCESS! Censored image is now acceptable!")
                return upload_result  # Return the upload result
            else:
                print(f"  ✗ Still forbidden after {level['name']}, trying stricter level...")
                # Note: Previously uploaded image stays on server (acceptable trade-off)

        except Exception as e:
            print(f"  ✗ Censorship {level['name']} failed: {e}")
            if attempt == len(censorship_levels):
                print(f"  All censorship attempts exhausted")
                return None

    print(f"  WARNING: All censorship levels tried, image still forbidden")
    return None

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
async def process_single_image(session: httpx.AsyncClient, image_url: str, is_forbidden: bool, initial_check_failed: bool = False) -> Optional[Dict[str, str]]:
    """Processes one image using resilient, retrying helper functions."""
    try:
        image_data, content_type = await download_image(session, image_url)
        final_image_data = image_data

        # If initial check failed (temporary URLs), verify the original first
        if initial_check_failed:
            print(f"INFO: Initial check failed for {image_url}, uploading to verify...")
            try:
                # Upload original to get permanent URL
                original_upload = await upload_image_to_basalam(session, image_data)
                original_url = original_upload["url"]
                print(f"  ✓ Uploaded original, checking with revision: {original_url}")

                # Check if original is actually forbidden
                is_actually_forbidden = await check_single_image_revision_by_url(session, original_url)

                if not is_actually_forbidden:
                    print(f"  ✓ Original image is ACCEPTABLE! No censorship needed.")
                    return original_upload
                else:
                    print(f"  → Original is FORBIDDEN, will apply censorship...")
                    # Continue to censorship below
                    is_forbidden = True
            except Exception as e:
                print(f"  ⚠ Could not verify original, will apply censorship: {e}")
                is_forbidden = True

        # Apply progressive censorship if image is marked as forbidden
        if is_forbidden:
            try:
                print(f"INFO: Image {image_url} is forbidden. Starting progressive censorship with verification...")

                # Use progressive censorship with verification loop
                # Returns upload result dict if successful, None if all levels failed
                upload_result = await censor_with_verification(session, image_data, content_type)

                if upload_result:
                    print(f"SUCCESS: Image successfully censored, verified, and uploaded for {image_url}")
                    return upload_result  # Already uploaded, return the result
                else:
                    # All censorship attempts failed - upload original as fallback
                    print(f"WARNING: All censorship attempts failed for {image_url}. Uploading original uncensored image as fallback.")
                    # Continue to upload original below

            except (httpx.RequestError, httpx.HTTPStatusError, HTTPException) as e:
                # اگر سانسور شکست خورد، تصویر اصلی را آپلود می‌کنیم (fallback behavior)
                print(f"WARNING: Censorship failed for {image_url}. Uploading original uncensored image as fallback. Reason: {e}")
                # Continue to upload original below

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
        # Try to check images with revision service
        # If URLs are temporary/inaccessible, this will fail and we'll assume all are forbidden
        initial_check_failed = False
        try:
            print(f"Initial revision check for {len(request.photo_links)} images...")
            revision_results = await check_images_revision(session, request.photo_links)

            id_to_url_map = {index: url for index, url in enumerate(request.photo_links)}
            url_to_forbidden_status = {
                id_to_url_map.get(item.get('file_id')): item.get('is_forbidden', True)
                for item in revision_results if item.get('file_id') in id_to_url_map
            }
            print(f"✓ Initial revision check completed")

        except Exception as e:
            # Initial revision check failed (likely temporary URLs not accessible)
            # Assume all images are forbidden - progressive censorship will verify properly
            print(f"⚠ Initial revision check failed: {e}")
            print(f"→ Assuming all images forbidden - will verify after censorship")
            url_to_forbidden_status = {url: True for url in request.photo_links}
            initial_check_failed = True

        tasks = [
            process_single_image(session, url, url_to_forbidden_status.get(url, True), initial_check_failed)
            for url in request.photo_links
        ]
        
        results_with_none = await asyncio.gather(*tasks)
        final_results = [result for result in results_with_none if result is not None]

    return {"processed_images": final_results}