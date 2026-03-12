# backend/audio_ingest.py
"""
Audio ingestion pipeline.

Steps:
- Transcribe audio using Whisper
- Chunk transcript
- Store chunks in PGVector (audio collection)
"""

from __future__ import annotations

import os
from typing import List

import whisper
from langchain_text_splitters import TokenTextSplitter
from langchain_core.documents import Document

from backend.config import settings
from backend.vectorstore import add_documents


# =================================================
# MAIN INGEST FUNCTION
# =================================================
def ingest_audio(audio_path: str) -> None:
    """
    Ingest an audio file:
    - Transcribe using Whisper
    - Chunk transcript
    - Store as audio transcripts
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    settings.ensure_dirs()

    # -------------------------
    # 1️⃣ Transcribe
    # -------------------------
    model = whisper.load_model(settings.whisper_model_name)
    result = model.transcribe(audio_path)

    transcript_text = result.get("text", "").strip()
    if not transcript_text:
        print("⚠️ Empty transcript, skipping audio ingestion")
        return

    # -------------------------
    # 2️⃣ Chunk transcript
    # -------------------------
    splitter = TokenTextSplitter(
        chunk_size=settings.chunk_size_tokens,
        chunk_overlap=settings.chunk_overlap_tokens,
    )

    chunks = splitter.split_text(transcript_text)

    # -------------------------
    # 3️⃣ Create Documents
    # -------------------------
    audio_docs: List[Document] = [
        Document(
            page_content=chunk,
            metadata={
                "type": settings.type_audio_transcript,
                "source": audio_path,
            },
        )
        for chunk in chunks
    ]

    # -------------------------
    # 4️⃣ Store in vector DB
    # -------------------------
    add_documents(
        audio_docs,
        collection=settings.collection_audio,
    )

    print(f"✅ Audio ingested | chunks: {len(audio_docs)}")


# =================================================
# CLI TEST
# =================================================
if __name__ == "__main__":
    path = input("Enter audio file path: ").strip()
    ingest_audio(path)
