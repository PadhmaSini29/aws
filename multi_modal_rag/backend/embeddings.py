# backend/embeddings.py
"""
Embeddings module.

Provides a single, reusable Titan embedding function using
Amazon Bedrock + LangChain, fully aligned with the notebook logic.

Usage:
    from backend.embeddings import get_embeddings
    embeddings = get_embeddings()
"""

from __future__ import annotations

from functools import lru_cache

from langchain_community.embeddings import BedrockEmbeddings

from backend.bedrock_client import get_bedrock_client
from backend.config import settings


@lru_cache(maxsize=1)
def get_embeddings() -> BedrockEmbeddings:
    """
    Returns a cached BedrockEmbeddings instance using
    Amazon Titan text embeddings.

    - Uses Bedrock Runtime client
    - Uses model ID from config
    - Normalizes embeddings (recommended for cosine similarity)

    Returns:
        BedrockEmbeddings
    """
    client = get_bedrock_client()

    embeddings = BedrockEmbeddings(
        client=client,
        model_id=settings.embed_model_id,
        normalize=True,
    )

    return embeddings


def embedding_health_check() -> int:
    """
    Simple sanity check to verify embeddings are working.

    Returns:
        int: embedding vector dimension

    Raises:
        RuntimeError: if embedding fails
    """
    try:
        embeddings = get_embeddings()
        vec = embeddings.embed_query("health check")
        return len(vec)
    except Exception as exc:
        raise RuntimeError(
            f"❌ Titan embeddings health check failed: {exc}"
        ) from exc


# Optional local test
if __name__ == "__main__":
    dim = embedding_health_check()
    print(f"✅ Titan embeddings OK | dim = {dim}")
