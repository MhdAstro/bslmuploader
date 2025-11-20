#!/usr/bin/env python3
"""Test script for bulk revision API"""

import asyncio
import httpx
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')
    revision_api_token: str


async def test_bulk_revision():
    settings = Settings()

    # Test URLs
    test_urls = [
        "https://cdn.basalam.com/avanak_images/1737367862_62803_r3sma.png",
        "https://cdn.basalam.com/avanak_images/1737367862_62803_r3sma.png"
    ]

    bulk_url = "https://revision.basalam.com/api_v1.0/validation/image/hijab-detector/bulk"
    headers = {
        "api-token": settings.revision_api_token,
        "Content-Type": "application/json"
    }
    payload = {"images": [{"file_id": index, "url": img_url} for index, img_url in enumerate(test_urls)]}

    async with httpx.AsyncClient() as session:
        try:
            print(f"Testing bulk revision API with {len(test_urls)} images...")
            print(f"Payload: {payload}")

            response = await session.post(bulk_url, headers=headers, json=payload, timeout=40.0)
            print(f"Response status: {response.status_code}")
            response.raise_for_status()

            result = response.json()
            print("\nSUCCESS!")
            print(f"Result: {result}")

            for item in result:
                file_id = item.get('file_id')
                is_forbidden = item.get('is_forbidden')
                print(f"  Image {file_id}: {'FORBIDDEN' if is_forbidden else 'ACCEPTABLE'}")

        except Exception as e:
            print(f"\nFAILED: {type(e).__name__}: {str(e)}")
            if hasattr(e, 'response'):
                print(f"Response status: {e.response.status_code}")
                print(f"Response body: {e.response.text}")


if __name__ == "__main__":
    asyncio.run(test_bulk_revision())
