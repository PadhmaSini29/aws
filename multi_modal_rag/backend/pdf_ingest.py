# backend/pdf_ingest.py
"""
PDF ingestion pipeline.

Implements the full logic from the notebook:
- Load PDF
- Extract text (LangChain PyPDFLoader)
- Token-based chunking
- Extract embedded images (PdfReader)
- Summarize images using Claude 3 Sonnet (Bedrock)
- Store:
    - text chunks        → TEXT collection
    - image summaries    → PDF IMAGE collection
"""

from __future__ import annotations

import os
import json
import base64
import mimetypes
from typing import List

from pypdf import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter
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
def ingest_pdf(pdf_path: str) -> None:
    """
    Ingest a PDF into the vector stores:
    - text chunks
    - image summaries (charts / figures)
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    settings.ensure_dirs()

    # =================================================
    # 1️⃣ TEXT INGESTION → TEXT COLLECTION
    # =================================================
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = TokenTextSplitter(
        chunk_size=settings.chunk_size_tokens,
        chunk_overlap=settings.chunk_overlap_tokens,
    )

    text_chunks = splitter.split_documents(documents)

    text_docs = [
        Document(
            page_content=chunk.page_content,
            metadata={
                "type": settings.type_text,
                "source": pdf_path,
                "page": chunk.metadata.get("page"),
            },
        )
        for chunk in text_chunks
    ]

    add_documents(
        text_docs,
        collection=settings.collection_text,
    )

    # =================================================
    # 2️⃣ IMAGE INGESTION → PDF IMAGE COLLECTION
    # =================================================
    reader = PdfReader(pdf_path)
    image_docs: List[Document] = []

    for page_idx, page in enumerate(reader.pages):
        if not hasattr(page, "images"):
            continue

        for img in page.images:
            image_path = os.path.join(settings.images_dir, img.name)

            # Save extracted image
            with open(image_path, "wb") as f:
                f.write(img.data)

            image_b64 = _encode_image(image_path)
            media_type = _guess_media_type(image_path)

            summary = _summarize_image_with_claude(
                image_b64=image_b64,
                media_type=media_type,
                prompt=settings.image_summary_prompt,
            )

            image_docs.append(
                Document(
                    page_content=summary,
                    metadata={
                        "type": settings.type_image,
                        "source": pdf_path,
                        "page": page_idx,
                        "image_base64": image_b64
                        if settings.store_raw_base64_in_metadata
                        else None,
                    },
                )
            )

    if image_docs:
        add_documents(
            image_docs,
            collection=settings.collection_pdf_images,
        )

    print(
        f"✅ PDF ingested | text chunks: {len(text_docs)} | images: {len(image_docs)}"
    )


# -------------------------------------------------
# Optional CLI test
# -------------------------------------------------
if __name__ == "__main__":
    path = input("Enter PDF path: ").strip()
    ingest_pdf(path)
