"""
Unified retrieval logic for the Multi-Modal RAG system.

COMPATIBLE with rag_service.ask()

Guarantees:
- AUTO mode mixes modalities intentionally
- TEXT mode → text collections ONLY (no images ever)
- AUDIO mode → audio transcripts only
- VIDEO mode → transcript + frames
- IMAGE-FOLDER mode → ONLY folder images

Claude-ready output:
{
    "texts": [...],
    "images": [...]
}
"""

from __future__ import annotations
from typing import Dict, List, Optional

from backend.config import settings
from backend.vectorstore import similarity_search


# =================================================
# AUTO / MIXED MODE (signature-compatible)
# =================================================
def retrieve_context(
    query: str,
    *,
    k: Optional[int] = None,
    include_images: bool = True,
    include_video_frames: bool = True,
    include_audio: bool = True,
    include_video_transcript: bool = True,
) -> Dict[str, List[str]]:

    k = k or settings.top_k

    texts: List[str] = []
    images: List[str] = []

    # ---------------- TEXT ----------------
    for d in similarity_search(
        query,
        collection=settings.collection_text,
        k=k,
    ):
        texts.append(d.page_content)

    # 🔒 HARD STOP: TEXT-ONLY MODE
    if not include_images and not include_audio and not include_video_frames:
        return {
            "texts": texts,
            "images": [],
        }

    # ---------------- IMAGE FOLDER ----------------
    if include_images:
        for d in similarity_search(
            query,
            collection=settings.collection_images,
            k=k,
        ):
            texts.append(d.page_content)
            img = d.metadata.get("image_base64")
            if img:
                images.append(img)

    # ---------------- PDF IMAGES ----------------
    if include_images:
        for d in similarity_search(
            query,
            collection=settings.collection_pdf_images,
            k=k,
        ):
            texts.append(d.page_content)
            img = d.metadata.get("image_base64")
            if img:
                images.append(img)

    # ---------------- AUDIO ----------------
    if include_audio:
        for d in similarity_search(
            query,
            collection=settings.collection_audio,
            k=k,
        ):
            texts.append(d.page_content)

    # ---------------- VIDEO TRANSCRIPT ----------------
    if include_video_transcript:
        for d in similarity_search(
            "TRANSCRIPT_MARKER",
            collection=settings.collection_video_text,
            k=1,
        ):
            texts.append(d.page_content)

    # ---------------- VIDEO FRAMES ----------------
    if include_video_frames:
        for d in similarity_search(
            query,
            collection=settings.collection_video_frames,
            k=k,
        ):
            texts.append(d.page_content)
            img = d.metadata.get("image_base64")
            if img:
                images.append(img)

    return {
        "texts": texts,
        "images": images[: settings.max_return_images],
    }


# =================================================
# IMAGE-FOLDER ONLY
# =================================================
def retrieve_image_folder_only(
    query: str,
    k: Optional[int] = None,
) -> Dict[str, List[str]]:

    k = k or settings.top_k

    texts: List[str] = []
    images: List[str] = []

    for d in similarity_search(
        query,
        collection=settings.collection_images,
        k=k,
        filter={
            "type": settings.type_image,
            "image_source": "folder",
        },
    ):
        texts.append(d.page_content)
        img = d.metadata.get("image_base64")
        if img:
            images.append(img)

    return {
        "texts": texts,
        "images": images[: settings.max_return_images],
    }


# =================================================
# AUDIO ONLY
# =================================================
def retrieve_audio_only(
    query: str,
    k: int = 3,
) -> Dict[str, List[str]]:

    return {
        "texts": [
            d.page_content
            for d in similarity_search(
                query,
                collection=settings.collection_audio,
                k=k,
            )
        ],
        "images": [],
    }


# =================================================
# VIDEO ONLY (TRANSCRIPT + FRAMES)
# =================================================
def retrieve_video_only(
    query: str,
    k: int = 6,
) -> Dict[str, List[str]]:

    texts: List[str] = []
    images: List[str] = []

    # Transcript (always included)
    for d in similarity_search(
        "TRANSCRIPT_MARKER",
        collection=settings.collection_video_text,
        k=1,
    ):
        texts.append(d.page_content)

    # Frames
    for d in similarity_search(
        query,
        collection=settings.collection_video_frames,
        k=k,
    ):
        texts.append(d.page_content)
        img = d.metadata.get("image_base64")
        if img:
            images.append(img)

    return {
        "texts": texts,
        "images": images[: settings.max_return_images],
    }
