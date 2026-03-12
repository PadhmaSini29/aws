"""
Markdown ingestion pipeline.

Stores markdown content as TEXT chunks.
"""

from __future__ import annotations

import os
from pathlib import Path

from langchain_core.documents import Document

from backend.config import settings
from backend.vectorstore import add_documents


def ingest_markdown(md_path: str) -> None:
    if not os.path.exists(md_path):
        raise FileNotFoundError(f"Markdown file not found: {md_path}")

    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        print("⚠️ Markdown empty, skipping")
        return

    doc = Document(
        page_content=content,
        metadata={
            "type": settings.type_markdown,
            "source": str(Path(md_path).resolve()),
        },
    )

    # ✅ CRITICAL FIX (this line was missing)
    add_documents(
        [doc],
        collection=settings.collection_text,
    )

    print(f"✅ Markdown ingested | source: {md_path}")
