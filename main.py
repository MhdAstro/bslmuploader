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

# Censorship retry settings with exponential backoff
CENSOR_MAX_RETRIES = 2  # 2 retries for 502 errors
CENSOR_RETRY_DELAY = 3  # Initial delay of 3 seconds
CENSOR_TIMEOUT = 90  # 90 second timeout for censorship
MAX_CONCURRENT_CENSORSHIP = 2  # Limit concurrent censorship to avoid overwhelming SafeImage

# ✨ --- دکوریتور جدید برای تلاش مجدد --- ✨
def async_retry(max_retries: int = MAX_RETRIES, delay: int = RETRY_DELAY_SECONDS, exponential_backoff: bool = False):
    """
    A decorator to retry an async function if it raises an exception.
    Supports exponential backoff for services that get overloaded.
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

                    # Calculate delay with exponential backoff if enabled
                    if exponential_backoff:
                        wait_time = delay * (2 ** attempt)  # 5, 10, 20 seconds...
                    else:
                        wait_time = delay

                    print(f"WARNING: Attempt {attempt + 1}/{max_retries} failed for {func.__name__}. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
        return wrapper
    return decorator


# --- 1. پیکربندی و خواندن متغیرهای محیطی ---
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8')
    basalam_upload_token: str
    revision_api_token: str

    # SafeImage censorship parameters (full censorship)
    safeimage_sigma_blur: str = "25"
    safeimage_feather: str = "5"
    safeimage_show_mask: str = "false"  # Disabled to reduce response size
    safeimage_only_clothes: str = "false"
    safeimage_blur_face: str = "true"
    safeimage_blur_hair: str = "true"
    safeimage_blur_arms: str = "true"  # Enabled for full censorship
    safeimage_blur_legs: str = "true"  # Enabled for full censorship
    safeimage_blur_torso: str = "true"  # Enabled for full censorship

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

@async_retry(max_retries=CENSOR_MAX_RETRIES, delay=CENSOR_RETRY_DELAY, exponential_backoff=True)
async def censor_images_batch(session: httpx.AsyncClient, images_data: List[tuple[bytes, Optional[str]]], sigma_blur: Optional[str] = None) -> List[Optional[bytes]]:
    """
    Applies full censorship to multiple images using SafeImage batch service.
    Blurs: face, hair, arms, legs, and torso.
    
    Args:
        images_data: List of tuples (image_bytes, content_type)
        sigma_blur: Optional custom sigma_blur value (overrides settings)
    
    Returns:
        List of censored image bytes (None if censorship failed for that image)
    """
    if not images_data:
        return []
    
    url = "https://safeimage.adminai.ir/process-batch"

    # Prepare form data with all SafeImage parameters (full censorship)
    form_data = {
        "sigma_blur": sigma_blur or settings.safeimage_sigma_blur,
        "feather": settings.safeimage_feather,
        "only_clothes": settings.safeimage_only_clothes,
        "blur_face": settings.safeimage_blur_face,
        "blur_hair": settings.safeimage_blur_hair,
        "blur_arms": settings.safeimage_blur_arms,
        "blur_legs": settings.safeimage_blur_legs,
        "blur_torso": settings.safeimage_blur_torso,
    }

    # Prepare multiple image files
    files = []
    for idx, (image_data, content_type) in enumerate(images_data):
        files.append(("images", (f"image_{idx}.jpg", image_data, content_type or "image/jpeg")))

    # Add headers similar to the curl request
    headers = {
        "accept": "*/*",
        "accept-language": "en-US,en;q=0.9,fa;q=0.8",
        "origin": "https://safeimage.adminai.ir",
        "referer": "https://safeimage.adminai.ir/",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }

    # Send batch request to SafeImage service with extended timeout
    total_size = sum(len(img[0]) for img in images_data)
    print(f"DEBUG: Sending {len(images_data)} images to SafeImage batch API (total size: {total_size} bytes)...")
    response = await session.post(url, data=form_data, files=files, headers=headers, timeout=CENSOR_TIMEOUT)
    print(f"DEBUG: SafeImage batch API responded with status {response.status_code}")
    response.raise_for_status()

    # Parse batch response
    result_data = response.json()

    if not result_data.get("success"):
        raise HTTPException(status_code=500, detail=f"Batch censorship failed: {result_data}")

    # Extract censored images from results
    results = result_data.get("results", [])
    censored_images = []
    
    for idx, result in enumerate(results):
        if not result.get("success"):
            print(f"WARNING: Image {idx} censorship failed in batch")
            censored_images.append(None)
            continue
            
        blurred_image = result.get("blurred_image")
        if not blurred_image:
            print(f"WARNING: No blurred_image for image {idx}")
            censored_images.append(None)
            continue

        # Handle base64 with or without data URI prefix
        if blurred_image.startswith("data:"):
            base64_image_str = blurred_image.split(',', 1)[1]
        else:
            base64_image_str = blurred_image

        # Decode base64 to bytes
        try:
            missing_padding = len(base64_image_str) % 4
            if missing_padding:
                base64_image_str += '=' * (4 - missing_padding)
            censored_images.append(base64.b64decode(base64_image_str))
        except Exception as e:
            print(f"ERROR: Failed to decode base64 for image {idx}: {e}")
            censored_images.append(None)

    print(f"DEBUG: Successfully censored {sum(1 for img in censored_images if img is not None)}/{len(images_data)} images")
    return censored_images


async def upload_images_batch(session: httpx.AsyncClient, images_data: List[tuple[bytes, str]]) -> List[Optional[Dict[str, str]]]:
    """
    Upload multiple images in parallel (no batch API available for upload).
    
    Args:
        images_data: List of tuples (image_bytes, content_type)
    
    Returns:
        List of upload results
    """
    upload_tasks = [
        upload_image_to_basalam(session, img_data, content_type)
        for img_data, content_type in images_data
    ]
    results = await asyncio.gather(*upload_tasks, return_exceptions=True)
    
    # Convert exceptions to None
    return [result if not isinstance(result, Exception) else None for result in results]


async def censor_and_upload_batch(session: httpx.AsyncClient, images_info: List[Dict]) -> List[Optional[Dict[str, str]]]:
    """
    Fully optimized batch processing with minimal requests:
    1. Batch censor Level 1 (sigma=25)
    2. Batch upload all Level 1 censored images
    3. Batch check all with revision
    4. For forbidden: Batch censor Level 2 (sigma=30)
    5. Batch upload Level 2 censored images
    6. Batch check Level 2 with revision
    7. Upload any still forbidden images anyway
    
    Args:
        images_info: List of dicts with 'image_data', 'content_type', 'index'
    
    Returns:
        List of upload results
    """
    if not images_info:
        return []
    
    # === LEVEL 1: Batch Censorship ===
    print(f"  → LEVEL 1: Batch censoring {len(images_info)} images (sigma_blur=25)...")
    images_data = [(info["image_data"], info["content_type"]) for info in images_info]
    level1_censored = await censor_images_batch(session, images_data, sigma_blur="25")
    print(f"  ✓ Level 1 batch censorship completed")

    # Filter out failed censorships
    valid_level1 = [
        (info, censored) for info, censored in zip(images_info, level1_censored)
        if censored is not None
    ]
    
    if not valid_level1:
        print(f"  ✗ All Level 1 censorships failed")
        return [None] * len(images_info)
    
    # === LEVEL 1: Batch Upload ===
    print(f"  → LEVEL 1: Batch uploading {len(valid_level1)} censored images...")
    level1_upload_data = [(censored, "image/png") for _, censored in valid_level1]
    level1_uploads = await upload_images_batch(session, level1_upload_data)
    print(f"  ✓ Level 1 batch upload completed")
    
    # === LEVEL 1: Batch Revision Check ===
    level1_urls = [upload['url'] for upload in level1_uploads if upload is not None]
    print(f"  → LEVEL 1: Batch checking {len(level1_urls)} images with revision...")
    revision_results = await check_images_revision(session, level1_urls)
    print(f"  ✓ Level 1 batch revision check completed")
    
    # Map revision results
    url_to_forbidden = {
        result['url']: result.get('is_forbidden', True)
        for result in revision_results
    }
    
    # Separate acceptable and forbidden
    acceptable_results = []
    forbidden_for_level2 = []
    
    for (info, _), upload in zip(valid_level1, level1_uploads):
        if upload is None:
            continue
            
        is_forbidden = url_to_forbidden.get(upload['url'], True)
        
        if is_forbidden:
            print(f"  [{info['index']}] ⚠ FORBIDDEN after Level 1, needs Level 2")
            forbidden_for_level2.append({
                'info': info,
                'level1_upload': upload
            })
        else:
            print(f"  [{info['index']}] ✓ ACCEPTABLE after Level 1")
            acceptable_results.append((info['index'], upload))
    
    # === LEVEL 2: Process Forbidden Images ===
    if forbidden_for_level2:
        print(f"\n  → LEVEL 2: Batch censoring {len(forbidden_for_level2)} images (sigma_blur=30)...")
        level2_images_data = [
            (item['info']['image_data'], item['info']['content_type'])
            for item in forbidden_for_level2
        ]
        level2_censored = await censor_images_batch(session, level2_images_data, sigma_blur="30")
        print(f"  ✓ Level 2 batch censorship completed")
        
        # === LEVEL 2: Batch Upload ===
        valid_level2 = [
            (item, censored) for item, censored in zip(forbidden_for_level2, level2_censored)
            if censored is not None
        ]
        
        if valid_level2:
            print(f"  → LEVEL 2: Batch uploading {len(valid_level2)} censored images...")
            level2_upload_data = [(censored, "image/png") for _, censored in valid_level2]
            level2_uploads = await upload_images_batch(session, level2_upload_data)
            print(f"  ✓ Level 2 batch upload completed")
            
            # === LEVEL 2: Batch Revision Check ===
            level2_urls = [upload['url'] for upload in level2_uploads if upload is not None]
            print(f"  → LEVEL 2: Batch checking {len(level2_urls)} images with revision...")
            level2_revision = await check_images_revision(session, level2_urls)
            print(f"  ✓ Level 2 batch revision check completed")
            
            level2_url_to_forbidden = {
                result['url']: result.get('is_forbidden', True)
                for result in level2_revision
            }
            
            # Process Level 2 results
            for (item, _), upload in zip(valid_level2, level2_uploads):
                if upload is None:
                    # Use Level 1 result
                    acceptable_results.append((item['info']['index'], item['level1_upload']))
                    continue
                
                is_still_forbidden = level2_url_to_forbidden.get(upload['url'], True)
                
                if is_still_forbidden:
                    print(f"  [{item['info']['index']}] ⚠ STILL FORBIDDEN after Level 2, uploading anyway")
                else:
                    print(f"  [{item['info']['index']}] ✓ ACCEPTABLE after Level 2")
                
                acceptable_results.append((item['info']['index'], upload))
        else:
            # All Level 2 censorships failed, use Level 1 results
            for item in forbidden_for_level2:
                print(f"  [{item['info']['index']}] ⚠ Level 2 failed, using Level 1 result")
                acceptable_results.append((item['info']['index'], item['level1_upload']))
    
    # Build final results array
    results = [None] * len(images_info)
    for idx, upload in acceptable_results:
        result_idx = next(i for i, info in enumerate(images_info) if info['index'] == idx)
        results[result_idx] = upload
    
    return results

@async_retry()
async def upload_image_to_basalam(session: httpx.AsyncClient, image_data: bytes, content_type: str = "image/jpeg") -> Dict[str, str]:
    """
    تصویر نهایی را آپلود می‌کند.
    
    Args:
        image_data: بایت‌های تصویر
        content_type: نوع محتوا (image/jpeg یا image/png)
    """
    url = "https://uploadio.basalam.com/v3/files"
    headers = {"authorization": f"{settings.basalam_upload_token}"}
    
    # Determine file extension based on content type
    file_ext = "png" if "png" in content_type.lower() else "jpg"
    filename = f"image.{file_ext}"
    
    files = {"file": (filename, image_data, content_type)}
    form_data = {"file_type": "product.photo"}
    response = await session.post(url, headers=headers, files=files, data=form_data, timeout=45.0)
    response.raise_for_status()
    response_json = response.json()
    return {"id": str(response_json["id"]), "url": response_json["urls"]["primary"]}


# ✨ --- تابع اصلی پردازش (ساده‌تر شده) --- ✨
async def download_and_upload_original(session: httpx.AsyncClient, image_url: str, index: int) -> Optional[Dict]:
    """Downloads image and uploads original to Basalam to get permanent URL."""
    try:
        print(f"[{index}] Downloading and uploading: {image_url}")
        image_data, content_type = await download_image(session, image_url)
        # Upload with original content type to preserve quality
        upload_result = await upload_image_to_basalam(session, image_data, content_type=content_type)

        print(f"[{index}] ✓ Uploaded: {upload_result['url']} (type: {content_type})")
        return {
            "index": index,
            "original_temp_url": image_url,
            "uploaded_url": upload_result["url"],
            "upload_result": upload_result,
            "image_data": image_data,
            "content_type": content_type
        }
    except Exception as e:
        print(f"[{index}] ✗ Failed to download/upload: {e}")
        return None


async def process_censorship_batch(session: httpx.AsyncClient, forbidden_images: List[Dict]) -> List[Optional[Dict[str, str]]]:
    """
    Apply batch censorship to all forbidden images at once.
    Returns list of upload results (None for failed/forbidden images).
    """
    if not forbidden_images:
        return []
    
    print(f"\n{'='*60}")
    print(f"Processing {len(forbidden_images)} FORBIDDEN images with batch censorship...")
    print(f"{'='*60}")

    try:
        # Apply batch censorship, upload, and verify
        results = await censor_and_upload_batch(session, forbidden_images)
        
        success_count = sum(1 for r in results if r is not None)
        print(f"\n✓ Batch censorship completed: {success_count}/{len(forbidden_images)} images acceptable")
        
        return results

    except Exception as e:
        print(f"\n✗ Batch censorship failed: {e}")
        print(f"→ Skipping all {len(forbidden_images)} forbidden images")
        return [None] * len(forbidden_images)


# --- 5. تعریف Endpoint اصلی API (بدون تغییر) ---
@app.post("/process-images", response_model=ImageProcessResponse)
async def process_images_endpoint(request: ImageProcessRequest):
    """
    New clean flow:
    1. Upload all original images first → get permanent URLs
    2. Check all permanent URLs with revision service
    3. For forbidden images: apply progressive censorship
    4. Return all final results
    """
    if not request.photo_links:
        return {"processed_images": []}

    async with httpx.AsyncClient() as session:
        print(f"\n{'='*60}")
        print(f"STEP 1: Uploading {len(request.photo_links)} original images...")
        print(f"{'='*60}")

        # Step 1: Upload all original images to get permanent URLs
        upload_tasks = [
            download_and_upload_original(session, url, index)
            for index, url in enumerate(request.photo_links)
        ]
        uploaded_images = await asyncio.gather(*upload_tasks)
        uploaded_images = [img for img in uploaded_images if img is not None]

        if not uploaded_images:
            print("✗ No images were successfully uploaded")
            return {"processed_images": []}

        print(f"✓ Successfully uploaded {len(uploaded_images)} images")

        print(f"\n{'='*60}")
        print(f"STEP 2: Checking uploaded images with revision service...")
        print(f"{'='*60}")

        # Step 2: Check all uploaded permanent URLs with revision service
        uploaded_urls = [img["uploaded_url"] for img in uploaded_images]
        try:
            revision_results = await check_images_revision(session, uploaded_urls)

            # Map results to uploaded images
            url_to_forbidden = {}
            for result in revision_results:
                file_id = result.get('file_id')
                if file_id is not None and file_id < len(uploaded_images):
                    url = uploaded_images[file_id]["uploaded_url"]
                    url_to_forbidden[url] = result.get('is_forbidden', True)

            print(f"✓ Revision check completed")

        except Exception as e:
            print(f"⚠ Revision check failed: {e}")
            print(f"→ Assuming all images forbidden for safety")
            url_to_forbidden = {img["uploaded_url"]: True for img in uploaded_images}

        print(f"\n{'='*60}")
        print(f"STEP 3: Processing censorship for forbidden images...")
        print(f"Using BATCH API for all forbidden images at once")
        print(f"{'='*60}")

        # Step 3: Separate acceptable and forbidden images
        acceptable_images = []
        forbidden_images = []
        
        for img in uploaded_images:
            is_forbidden = url_to_forbidden.get(img["uploaded_url"], True)
            if is_forbidden:
                forbidden_images.append(img)
            else:
                acceptable_images.append(img["upload_result"])
        
        print(f"→ {len(acceptable_images)} images are ACCEPTABLE (no censorship needed)")
        print(f"→ {len(forbidden_images)} images are FORBIDDEN (need batch censorship)")

        # Apply batch censorship to all forbidden images at once
        censored_results = await process_censorship_batch(session, forbidden_images)
        
        # Combine acceptable and censored results
        results_with_none = acceptable_images + censored_results

        # Filter out None results (skipped images that were still forbidden after censorship)
        final_results = [result for result in results_with_none if result is not None]

        skipped_count = len(results_with_none) - len(final_results)
        print(f"\n{'='*60}")
        print(f"✓ COMPLETED: {len(final_results)} images processed successfully")
        if skipped_count > 0:
            print(f"⚠ SKIPPED: {skipped_count} images (still forbidden after censorship)")
        print(f"{'='*60}\n")

    return {"processed_images": final_results}