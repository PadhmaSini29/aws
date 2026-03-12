import os, json
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

client = OpenSearch(
    hosts=[OPENSEARCH_URL],
    http_auth=(OPENSEARCH_USER, OPENSEARCH_PASS),
    use_ssl=True,
    verify_certs=True,
)

bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)

def embed_query(q):
    resp = bedrock.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps({"inputText": q}),
        contentType="application/json",
        accept="application/json"
    )
    return json.loads(resp["body"].read())["embedding"]

def show(resp):
    for h in resp["hits"]["hits"]:
        s = h["_source"]
        print(f"- {s['title']} by {s['author']} (score={h['_score']})")

def keyword_search(q):
    resp = client.search(index=INDEX_NAME, body={
        "query": {
            "multi_match": {
                "query": q,
                "fields": ["title^2", "author", "description"]
            }
        }
    })
    print("\n🔎 KEYWORD SEARCH")
    show(resp)

def semantic_search(q, k=3):
    vec = embed_query(q)
    resp = client.search(index=INDEX_NAME, body={
        "size": k,
        "query": {
            "knn": {
                "embedding": {"vector": vec, "k": k}
            }
        }
    })
    print("\n🧠 SEMANTIC SEARCH")
    show(resp)

def hybrid_search(q, k=3):
    vec = embed_query(q)
    resp = client.search(index=INDEX_NAME, body={
        "size": k,
        "query": {
            "bool": {
                "should": [
                    {"multi_match": {
                        "query": q,
                        "fields": ["title^2", "author", "description"]
                    }},
                    {"knn": {"embedding": {"vector": vec, "k": k}}}
                ]
            }
        }
    })
    print("\n✨ HYBRID SEARCH")
    show(resp)

if __name__ == "__main__":
    q = input("Search books: ")
    keyword_search(q)
    semantic_search(q)
    hybrid_search(q)
