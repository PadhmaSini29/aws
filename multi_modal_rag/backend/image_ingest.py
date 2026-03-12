# backend/image_ingest.py
"""
Image ingestion pipeline (folder-based).

Implements logic from your notebook:
- Read images from a folder
- Encode images to base64
- Summarize images using Claude 3 Sonnet (Bedrock multimodal)
- Store summaries in PGVector with image_base64 metadata

IMPORTANT:
- Images ingested here are explicitly tagged with:
    image_source = "folder"
- This enables STRICT image-folder-only retrieval later.

Supported image types:
- .png
- .jpg
- .jpeg
"""

from __future__ import annotations

import os
import json
import base64
import mimetypes
from typing import List

from langchain_core.documents import Document

from backend.config import settings
from backend.bedrock_client import get_bedrock_client
from backend.vectorstore import add_documents


# -------------------------------------------------
# Helpers
# -------------------------------------------------
def _encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _guess_media_type(path: str) -> str:
    mt, _ = mimetypes.guess_type(path)
    return mt or "image/png"


def _summarize_image_with_claude(
    image_b64: str,
    media_type: str,
    prompt: str,
) -> str:
    """
    Summarize an image using Claude 3 Sonnet (Bedrock multimodal).
    """
    client = get_bedrock_client()

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_b64,
                        },
                    },
                ],
            }
        ],
        "max_tokens": 400,
        "temperature": 0.1,
    }

    response = client.invoke_model(
        modelId=settings.claude_model_id,
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json",
    )

    response_body = json.loads(response["body"].read())
    return response_body["content"][0]["text"]


# -------------------------------------------------
# Main ingestion function
# -------------------------------------------------
def ingest_images_folder(
    image_dir: str,
    *,
    prompt: str | None = None,
) -> None:
    """
    Ingest all images from a folder into the IMAGE collection.

    Images ingested here are:
    - type = "image"
    - image_source = "folder"

    Args:
        image_dir (str): folder containing images
        prompt (str, optional): custom image summary prompt
    """
    if not os.path.isdir(image_dir):
        raise NotADirectoryError(f"Image directory not found: {image_dir}")

    settings.ensure_dirs()
    prompt = prompt or settings.image_summary_prompt

    image_paths = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if not image_paths:
        print("⚠️ No images found in folder")
        return

    image_docs: List[Document] = []

    for path in image_paths:
        image_b64 = _encode_image(path)
        media_type = _guess_media_type(path)

        summary = _summarize_image_with_claude(
            image_b64=image_b64,
            media_type=media_type,
            prompt=prompt,
        )

        image_docs.append(
            Document(
                page_content=summary,
                metadata={
                    "type": settings.type_image,
                    "image_source": "folder",   # 🔥 CRITICAL TAG
                    "source": path,
                    "image_base64": image_b64
                    if settings.store_raw_base64_in_metadata
                    else None,
                },
            )
        )

    # ✅ CORRECT COLLECTION
    add_documents(
        image_docs,
        collection=settings.collection_images,
        
    )

    print(
        f"✅ Images ingested from folder | count: {len(image_docs)}"
    )


# -------------------------------------------------
# Optional CLI test
# -------------------------------------------------
if __name__ == "__main__":
    folder = input("Enter image folder path: ").strip()
    ingest_images_folder(folder)
