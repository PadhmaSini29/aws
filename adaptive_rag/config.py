"""
Configuration management for the Adaptive Structured RAG (NL2SQL) Agent
"""

import os
import logging
from typing import Dict, Any


def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure application-wide logging.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def get_config() -> Dict[str, Any]:
    """
    Load configuration from environment variables with safe defaults.
    This function NEVER raises KeyError.
    """

    return {
        # ============================================================
        # AWS / Bedrock Configuration
        # ============================================================
        "aws_region": os.environ.get("AWS_REGION", "us-east-1"),

        # Bedrock model (Nova works without inference profile)
        # To switch back to Claude later, set BEDROCK_MODEL_ID to inference profile ARN
        "bedrock_model_id": os.environ.get(
            "BEDROCK_MODEL_ID",
            "amazon.nova-lite-v1:0"
        ),

        # ============================================================
        # PostgreSQL Configuration (RECOMMENDED)
        # ============================================================
        "pg_host": os.environ.get("PG_HOST", "localhost"),
        "pg_port": int(os.environ.get("PG_PORT", "5432")),
        "pg_database": os.environ.get("PG_DATABASE", "wealthmanagement"),
        "pg_user": os.environ.get("PG_USER", "postgres"),
        "pg_password": os.environ.get("PG_PASSWORD", "postgres"),

        # ============================================================
        # Athena Configuration (OPTIONAL / BLOCKED IN CORP ACCOUNTS)
        # ============================================================
        "athena_database": os.environ.get("ATHENA_DATABASE", ""),
        "athena_output_location": os.environ.get("ATHENA_OUTPUT_LOCATION", ""),

        # ============================================================
        # Bedrock Knowledge Base (OPTIONAL)
        # ============================================================
        "knowledge_base_id": os.environ.get("KNOWLEDGE_BASE_ID", ""),

        # ============================================================
        # Local SQLite Configuration (FALLBACK)
        # ============================================================
        "sqlite_path": os.environ.get(
            "SQLITE_PATH",
            r".\data\wealthmanagement.db"
        ),
    }
