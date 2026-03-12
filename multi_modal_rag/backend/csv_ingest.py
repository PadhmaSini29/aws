# backend/csv_ingest.py
"""
CSV ingestion pipeline.

Stores CSV content in the TEXT collection.
"""

from __future__ import annotations

import os
import pandas as pd

from langchain_core.documents import Document

from backend.config import settings
from backend.vectorstore import add_documents


# =================================================
# MAIN INGEST FUNCTION
# =================================================
def ingest_csv(csv_path: str) -> None:
    """
    Ingest a CSV file into the TEXT collection.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    settings.ensure_dirs()

    df = pd.read_csv(csv_path)

    if df.empty:
        print("⚠️ CSV is empty, skipping ingestion")
        return

    csv_text = df.to_string(index=False)

    doc = Document(
        page_content=csv_text,
        metadata={
            "type": settings.type_csv,
            "source": csv_path,
        },
    )

    add_documents(
        [doc],
        collection=settings.collection_text,  # ✅ REQUIRED
    )

    print(f"✅ CSV ingested | rows: {len(df)} | cols: {len(df.columns)}")


# =================================================
# CLI TEST
# =================================================
if __name__ == "__main__":
    path = input("Enter CSV path: ").strip()
    ingest_csv(path)
