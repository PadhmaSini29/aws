# backend/vectorstore.py
"""
PGVector vector store helpers (MULTI-COLLECTION) with HARD DB BOOTSTRAP.

Fixes the persistent error:
    ValueError: Collection not found

Why it happens:
- LangChain PGVector expects a row in `langchain_pg_collection`
- If that row doesn't exist, add_documents() raises "Collection not found"
- Some versions of PGVector/LangChain do NOT reliably create it

This module:
✅ Ensures pgvector extension exists
✅ Ensures required tables exist
✅ Ensures collection row exists (per collection name)
✅ Then uses PGVector normally

Collections (from settings):
- Text             → settings.collection_text
- Image folder     → settings.collection_images
- PDF images       → settings.collection_pdf_images
- Audio transcript → settings.collection_audio
- Video transcript → settings.collection_video_text
- Video frames     → settings.collection_video_frames
"""

from __future__ import annotations

import uuid
from functools import lru_cache
from typing import Iterable, List, Optional

from sqlalchemy import create_engine, text as sql_text
from sqlalchemy.engine import Engine

from langchain_community.vectorstores.pgvector import PGVector
from langchain_core.documents import Document

from backend.config import settings
from backend.embeddings import get_embeddings


# -------------------------------------------------
# SQL bootstrap (tables + collection row)
# -------------------------------------------------
@lru_cache(maxsize=1)
def _engine() -> Engine:
    """
    Create a cached SQLAlchemy engine for bootstrap SQL.
    """
    return create_engine(settings.pg_conn_string, future=True)


def _bootstrap_pgvector_schema(engine: Engine) -> None:
    """
    Ensure pgvector extension + langchain tables exist.
    Safe to call many times.
    """
    with engine.begin() as conn:
        # pgvector extension
        conn.execute(sql_text("CREATE EXTENSION IF NOT EXISTS vector;"))

        # LangChain PGVector tables (default names used by langchain_community)
        conn.execute(
            sql_text(
                """
                CREATE TABLE IF NOT EXISTS langchain_pg_collection (
                    uuid UUID PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL,
                    cmetadata JSONB
                );
                """
            )
        )

        conn.execute(
            sql_text(
                """
                CREATE TABLE IF NOT EXISTS langchain_pg_embedding (
                    uuid UUID PRIMARY KEY,
                    collection_id UUID REFERENCES langchain_pg_collection(uuid) ON DELETE CASCADE,
                    embedding vector,
                    document TEXT,
                    cmetadata JSONB
                );
                """
            )
        )

        # Helpful index
        conn.execute(
            sql_text(
                """
                CREATE INDEX IF NOT EXISTS langchain_pg_embedding_collection_id_idx
                ON langchain_pg_embedding(collection_id);
                """
            )
        )


def _ensure_collection_row(engine: Engine, collection_name: str) -> None:
    """
    Ensure a row exists in langchain_pg_collection for `collection_name`.
    """
    with engine.begin() as conn:
        # Check
        res = conn.execute(
            sql_text("SELECT uuid FROM langchain_pg_collection WHERE name = :name"),
            {"name": collection_name},
        ).fetchone()

        if res is None:
            conn.execute(
                sql_text(
                    """
                    INSERT INTO langchain_pg_collection (uuid, name, cmetadata)
                    VALUES (:uuid, :name, :meta)
                    """
                ),
                {
                    "uuid": str(uuid.uuid4()),
                    "name": collection_name,
                    "meta": "{}",
                },
            )


def ensure_collection(collection_name: str) -> None:
    """
    Public: ensure DB is ready + collection exists.
    """
    eng = _engine()
    _bootstrap_pgvector_schema(eng)
    _ensure_collection_row(eng, collection_name)


# -------------------------------------------------
# Vectorstore factory (per collection)
# -------------------------------------------------
@lru_cache(maxsize=32)
def get_vectorstore(collection: str) -> PGVector:
    """
    Get a PGVector instance for a collection.
    Collection is guaranteed to exist in DB before returning.
    """
    ensure_collection(collection)

    return PGVector(
        connection_string=settings.pg_conn_string,
        collection_name=collection,
        embedding_function=get_embeddings(),
    )


# -------------------------------------------------
# Public APIs
# -------------------------------------------------
def add_documents(docs: Iterable[Document], *, collection: str) -> None:
    docs = list(docs)
    if not docs:
        return

    vs = get_vectorstore(collection)
    vs.add_documents(docs)


def similarity_search(
    query: str,
    *,
    collection: str,
    k: Optional[int] = None,
    filter: Optional[dict] = None,
) -> List[Document]:
    vs = get_vectorstore(collection)
    k = k or settings.top_k

    if filter:
        return vs.similarity_search(query=query, k=k, filter=filter)

    return vs.similarity_search(query=query, k=k)


def delete_collection(collection: str) -> None:
    """
    Delete ONE collection safely.
    This removes the collection row and cascades embeddings.
    """
    ensure_collection(collection)  # ensures tables exist

    eng = _engine()
    with eng.begin() as conn:
        conn.execute(
            sql_text("DELETE FROM langchain_pg_collection WHERE name = :name"),
            {"name": collection},
        )

    # Clear cache so future calls re-create
    try:
        get_vectorstore.cache_clear()
    except Exception:
        pass

    print(f"🗑️ Deleted collection: {collection}")


def delete_all_collections() -> None:
    for c in [
        settings.collection_text,
        settings.collection_images,
        settings.collection_pdf_images,
        settings.collection_audio,
        settings.collection_video_text,
        settings.collection_video_frames,
    ]:
        delete_collection(c)


def bootstrap_all_collections() -> None:
    """
    Ensure ALL collections exist.
    Call once at startup.
    """
    for c in [
        settings.collection_text,
        settings.collection_images,
        settings.collection_pdf_images,
        settings.collection_audio,
        settings.collection_video_text,
        settings.collection_video_frames,
    ]:
        ensure_collection(c)
        print(f"✔ ensured collection: {c}")


def collection_info() -> dict:
    return {
        "pg_conn_string": settings.pg_conn_string,
        "embedding_model": settings.embed_model_id,
        "collections": {
            "text": settings.collection_text,
            "images": settings.collection_images,
            "pdf_images": settings.collection_pdf_images,
            "audio": settings.collection_audio,
            "video_text": settings.collection_video_text,
            "video_frames": settings.collection_video_frames,
        },
    }


if __name__ == "__main__":
    bootstrap_all_collections()
    print("✅ All collections bootstrapped")
    print(collection_info())
