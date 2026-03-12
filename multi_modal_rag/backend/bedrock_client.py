# backend/bedrock_client.py
"""
AWS Bedrock Runtime client initialization.

This module creates a **single, reusable Bedrock Runtime client**
with proper timeouts, retries, and region handling.

Usage:
    from backend.bedrock_client import get_bedrock_client
    bedrock_client = get_bedrock_client()
"""

from __future__ import annotations

import boto3
from botocore.config import Config
from functools import lru_cache

from backend.config import settings


@lru_cache(maxsize=1)
def get_bedrock_client():
    """
    Returns a cached AWS Bedrock Runtime client.

    - Uses region from config (.env / defaults)
    - Configures timeouts & retries
    - Safe to call multiple times (singleton behavior)

    Raises:
        RuntimeError: if client creation fails
    """
    try:
        config = Config(
            region_name=settings.aws_region,
            retries={
                "max_attempts": settings.bedrock_max_attempts,
                "mode": "standard",
            },
            connect_timeout=settings.bedrock_connect_timeout,
            read_timeout=settings.bedrock_read_timeout,
        )

        client = boto3.client(
            service_name="bedrock-runtime",
            config=config,
        )

        return client

    except Exception as exc:
        raise RuntimeError(
            f"❌ Failed to create Bedrock client in region "
            f"'{settings.aws_region}': {exc}"
        ) from exc


def health_check() -> bool:
    """
    Lightweight sanity check to ensure Bedrock client is usable.

    NOTE:
    Bedrock does not provide a simple 'ping' API.
    This function just verifies client creation and config.

    Returns:
        bool: True if client can be created
    """
    try:
        _ = get_bedrock_client()
        return True
    except Exception:
        return False


# Optional: eager validation during import (comment out if undesired)
if __name__ == "__main__":
    if health_check():
        print("✅ Bedrock client ready")
    else:
        print("❌ Bedrock client initialization failed")
