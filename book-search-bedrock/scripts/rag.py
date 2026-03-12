import os
from typing import List, Dict
from dotenv import load_dotenv

from scripts.utils import get_opensearch_client, generate_embedding, claude_generate_answer

load_dotenv()

INDEX_NAME = os.getenv("INDEX_NAME", "books")

def retrieve_top_books(query: str, k: int = 4) -> List[Dict]:
    """
    Hybrid retrieval:
    - keyword match on title/author/description
    - vector kNN on embedding
    Returns top-k sources for citations.
    """
    client = get_opensearch_client()
    vec = generate_embedding(query)

    resp = client.search(
        index=INDEX_NAME,
        body={
            "size": k,
            "query": {
                "bool": {
                    "should": [
                        {"multi_match": {"query": query, "fields": ["title^2", "author", "description"]}},
                        {"knn": {"embedding": {"vector": vec, "k": k}}},
                    ],
                    "minimum_should_match": 1
                }
            }
        }
    )

    hits = resp.get("hits", {}).get("hits", [])
    sources = []
    for h in hits:
        s = h.get("_source", {})
        sources.append({
            "id": h.get("_id"),
            "title": s.get("title", ""),
            "author": s.get("author", ""),
            "description": s.get("description", ""),
            "score": h.get("_score"),
        })
    return sources

def rag_answer(question: str, k: int = 4) -> Dict:
    sources = retrieve_top_books(question, k=k)

    if not sources:
        return {"answer": "I couldn't find any relevant books in the index.", "sources": []}

    answer = claude_generate_answer(question=question, sources=sources)
    return {"answer": answer, "sources": sources}
