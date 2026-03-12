import os, json
import numpy as np
import boto3
from dotenv import load_dotenv
from opensearchpy import OpenSearch

load_dotenv()

OPENSEARCH_URL = os.getenv("OPENSEARCH_URL")
OPENSEARCH_USER = os.getenv("OPENSEARCH_USER")
OPENSEARCH_PASS = os.getenv("OPENSEARCH_PASS")
INDEX_NAME = os.getenv("INDEX_NAME")

AWS_REGION = os.getenv("AWS_REGION")
MODEL_ID = os.getenv("BEDROCK_MODEL_ID")

EMBED_DIM = 1536  # Titan embeddings dimension

# OpenSearch client
client = OpenSearch(
    hosts=[OPENSEARCH_URL],
    http_auth=(OPENSEARCH_USER, OPENSEARCH_PASS),
    use_ssl=True,
    verify_certs=True,
)

# Bedrock client
bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)

def get_embedding(text: str) -> list[float]:
    response = bedrock.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps({"inputText": text}),
        contentType="application/json",
        accept="application/json"
    )
    body = json.loads(response["body"].read())
    return body["embedding"]
def recreate_index():

    if client.indices.exists(index=INDEX_NAME):
        client.indices.delete(index=INDEX_NAME)

    body = {
        "settings": {
            "index": {
                "knn": True,                  
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
                    "dimension": 1536
                }
            }
        }
    }

    client.indices.create(index=INDEX_NAME, body=body)
    print(f"✅ Created k-NN index '{INDEX_NAME}'")

def bulk_ingest(path="C:/Users/lgspa/Downloads/AWS/book-search-bedrock/data/books.json"):
    books = json.load(open(path, "r", encoding="utf-8"))
    actions = []

    for b in books:
        text = f"{b['title']} | {b['author']} | {b['description']}"
        b["embedding"] = get_embedding(text)

        actions.append({"index": {"_index": INDEX_NAME, "_id": b["id"]}})
        actions.append(b)

    bulk_body = "\n".join(json.dumps(x) for x in actions) + "\n"

    resp = client.bulk(
        body=bulk_body,
        index=INDEX_NAME,
        params={"refresh": "true"}  # ensures data is visible
)


    if resp.get("errors"):
        print("❌ Bulk ingest errors:", resp)
    else:
        print(f"✅ Ingested {len(books)} books")

if __name__ == "__main__":
    recreate_index()
    bulk_ingest()
