import boto3
from typing import Optional

HARDCODED_SCHEMA = """
Tables:

client(
  client_id INTEGER PRIMARY KEY NOT NULL,
  first_name TEXT,
  last_name TEXT,
  age INTEGER,
  risk_tolerance TEXT
)

investment(
  investment_id INTEGER PRIMARY KEY NOT NULL,
  client_id INTEGER,
  asset_type TEXT,
  investment_amount REAL,
  current_value REAL,
  purchase_date VARCHAR,
  FOREIGN KEY (client_id) REFERENCES client(client_id)
)

portfolio_performance(
  client_id INTEGER NOT NULL,
  year INTEGER NOT NULL,
  total_return_percentage REAL,
  benchmark_return REAL,
  PRIMARY KEY (client_id, year),
  FOREIGN KEY (client_id) REFERENCES client(client_id)
)
"""

class SchemaRetriever:
    def __init__(self, region: str, knowledge_base_id: str = ""):
        self.knowledge_base_id = knowledge_base_id.strip()
        self.client = boto3.client("bedrock-agent-runtime", region_name=region)

    def get_schema(self, question: str) -> str:
        if not self.knowledge_base_id:
            return HARDCODED_SCHEMA.strip()

        # Retrieve relevant chunks from KB using the question + schema keywords
        query = f"""You are retrieving DB schema context.
Return info about tables/columns/keys relevant to: {question}
Wealth management schema: client, investment, portfolio_performance.
"""
        resp = self.client.retrieve(
            knowledgeBaseId=self.knowledge_base_id,
            retrievalQuery={"text": query},
            retrievalConfiguration={
                "vectorSearchConfiguration": {"numberOfResults": 5}
            },
        )
        results = resp.get("retrievalResults", [])
        if not results:
            return HARDCODED_SCHEMA.strip()

        # Extract text from KB results (simple join)
        chunks = []
        for r in results:
            content = r.get("content", {})
            txt = content.get("text", "")
            if txt:
                chunks.append(txt)

        merged = "\n\n---\n\n".join(chunks).strip()
        return merged if merged else HARDCODED_SCHEMA.strip()
