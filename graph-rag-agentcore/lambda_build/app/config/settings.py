# app/config/settings.py

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """
    Central configuration for Graph-RAG + AgentCore project.
    Works for:
    - Local development
    - AWS Lambda
    """

    # 🔌 Neo4j
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "12345678")

    # 🧠 Agent / Bedrock
    AWS_REGION: str = os.getenv("AWS_REGION", "us-west-2")
    BEDROCK_MODEL_ID: str = os.getenv(
        "BEDROCK_MODEL_ID",
        "anthropic.claude-3-5-haiku-20241022-v1:0",
    )

    # 🧾 Application
    APP_NAME: str = "graph-rag-agentcore"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # 🧠 Memory
    DEFAULT_SESSION_PREFIX: str = "graph-rag-session"


# Singleton-style import
settings = Settings()
