import os
import json
import boto3
from dotenv import load_dotenv
from opensearchpy import OpenSearch

load_dotenv()

# ---------- ENV ----------
OPENSEARCH_URL  = os.getenv("OPENSEARCH_URL")
OPENSEARCH_USER = os.getenv("OPENSEARCH_USER")
OPENSEARCH_PASS = os.getenv("OPENSEARCH_PASS")
AWS_REGION      = os.getenv("AWS_REGION")
BEDROCK_MODEL   = os.getenv("BEDROCK_MODEL_ID")

EMBED_DIM = 1536  # Titan Embeddings dimension
import json
import os
import boto3
from typing import List, Dict

def get_bedrock_runtime():
    # Reuse the same runtime client for embeddings + Claude
    region = os.getenv("AWS_REGION", "us-east-1")
    return boto3.client("bedrock-runtime", region_name=region)

def claude_generate_answer(*, question: str, sources: List[Dict], max_tokens: int = 700, temperature: float = 0.2) -> str:
    """
    FULL RAG generation: sends retrieved sources to Claude and asks it to answer with citations [1], [2], etc.
    Uses Anthropic Claude Messages format on Amazon Bedrock.
    """
    model_id = os.getenv("CLAUDE_MODEL_ID")
    if not model_id:
        raise RuntimeError("Missing CLAUDE_MODEL_ID in .env")

    # Build a numbered context block for citations
    context_lines = []
    for i, s in enumerate(sources, start=1):
        title = s.get("title", "Unknown title")
        author = s.get("author", "Unknown author")
        desc = s.get("description", "")
        context_lines.append(f"[{i}] {title} — {author}\n{desc}")

    context_text = "\n\n".join(context_lines)

    system = (
        "You are a helpful assistant. Answer the user's question ONLY using the provided sources. "
        "If the sources do not contain enough info, say you don't know. "
        "Cite sources using [1], [2], etc. Do not invent citations."
    )

    user_text = (
        f"Question:\n{question}\n\n"
        f"Sources:\n{context_text}\n\n"
        "Instructions:\n"
        "- Write a clear, direct answer.\n"
        "- Add citations like [1] or [1][2] after the relevant sentences.\n"
    )

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": user_text}]}
        ],
    }

    brt = get_bedrock_runtime()
    resp = brt.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body),
    )

    payload = json.loads(resp["body"].read())
    # Claude responses are typically in payload["content"] list with text blocks
    parts = payload.get("content", [])
    answer = ""
    for p in parts:
        if p.get("type") == "text":
            answer += p.get("text", "")
    return answer.strip()


# ---------- CLIENTS ----------
def get_opensearch_client():
    return OpenSearch(
        hosts=[OPENSEARCH_URL],
        http_auth=(OPENSEARCH_USER, OPENSEARCH_PASS),
        use_ssl=True,
        verify_certs=True,
        timeout=60,              
        max_retries=3,
        retry_on_timeout=True
)

    

def get_bedrock_client():
    return boto3.client(
        service_name="bedrock-runtime",
        region_name=AWS_REGION
    )

# ---------- EMBEDDINGS ----------
def generate_embedding(text: str) -> list[float]:
    """
    Generate vector embedding using Amazon Bedrock (Titan).
    """
    bedrock = get_bedrock_client()

    response = bedrock.invoke_model(
        modelId=BEDROCK_MODEL,
        body=json.dumps({"inputText": text}),
        contentType="application/json",
        accept="application/json"
    )

    body = json.loads(response["body"].read())
    return body["embedding"]

# ---------- INDEX ----------
def create_books_index(client, index_name: str):
    """
    Create OpenSearch index with vector + text mapping.
    """
    if client.indices.exists(index=index_name):
        client.indices.delete(index=index_name)

    mapping = {
        "settings": {
            "index": {
                "number_of_shards": 2,
                "number_of_replicas": 1
            }
        },
        "mappings": {
            "properties": {
                "title": {"type": "text"},
                "author": {"type": "text"},
                "description": {"type": "text"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": EMBED_DIM
                }
            }
        }
    }

    client.indices.create(index=index_name, body=mapping)
