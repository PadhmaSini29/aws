"""
Core NL2SQL Agent with self-correcting Structured RAG logic
"""

import logging
import re
from typing import Dict, Any, Literal

from src.llm import BedrockClaude
from src.tools.knowledge_base_tool import SchemaRetriever
from src.tools.sqllite_tool import run_sqlite_query
from src.tools.postgres_tool import run_postgres_query

logger = logging.getLogger("nl2sql-agent")

SQL_BLOCK_REGEX = re.compile(
    r"```sql\s*(.*?)\s*```",
    re.IGNORECASE | re.DOTALL
)

SYSTEM_PROMPT = """
You are an expert data analyst.

TASK:
Convert the user's natural language question into a SINGLE valid SQL query
based strictly on the provided database schema.

RULES:
- Output ONLY SQL
- No explanations
- Use correct table and column names
- SQLite / PostgreSQL compatible SQL
- Wrap SQL inside ```sql ``` block
"""


def extract_sql(text: str) -> str:
    """
    Extract SQL from ```sql``` fenced block.
    """
    match = SQL_BLOCK_REGEX.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


class NL2SQLAgent:
    def __init__(
        self,
        llm: BedrockClaude,
        schema_retriever: SchemaRetriever,
        cfg: Dict[str, Any],
    ):
        self.llm = llm
        self.schema_retriever = schema_retriever
        self.cfg = cfg

    def answer(
        self,
        question: str,
        engine: Literal["sqlite", "postgres"],
        max_attempts: int = 3
    ) -> Dict[str, Any]:

        schema_context = self.schema_retriever.get_schema(question)
        last_error = ""

        for attempt in range(1, max_attempts + 1):
            logger.info(f"Attempt {attempt} for question: {question}")

            user_prompt = f"""
SCHEMA:
{schema_context}

QUESTION:
{question}
"""

            if last_error:
                user_prompt += f"""
PREVIOUS ERROR:
{last_error}

Fix the SQL and regenerate.
"""

            response = self.llm.chat(
                system=SYSTEM_PROMPT,
                user=user_prompt,
                max_tokens=800
            )

            sql = extract_sql(response)
            logger.info(f"Generated SQL:\n{sql}")

            try:
                if engine == "sqlite":
                    result = run_sqlite_query(
                        self.cfg["sqlite_path"],
                        sql
                    )

                elif engine == "postgres":
                    result = run_postgres_query(
                        host=self.cfg["pg_host"],
                        port=self.cfg["pg_port"],
                        database=self.cfg["pg_database"],
                        user=self.cfg["pg_user"],
                        password=self.cfg["pg_password"],
                        sql=sql
                    )

                else:
                    raise ValueError(f"Unsupported engine: {engine}")

                return {
                    "engine": engine,
                    "sql": sql,
                    "result": result,
                    "attempts": attempt
                }

            except Exception as e:
                last_error = str(e)
                logger.warning(f"SQL execution failed: {last_error}")

        raise RuntimeError(
            f"Failed after {max_attempts} attempts. Last error: {last_error}"
        )


def create_nl2sql_agent(cfg: Dict[str, Any]) -> NL2SQLAgent:
    """
    Factory method to create NL2SQLAgent from config
    """

    llm = BedrockClaude(
        region=cfg["aws_region"],
        model_id=cfg["bedrock_model_id"]
    )

    schema_retriever = SchemaRetriever(
        region=cfg["aws_region"],
        knowledge_base_id=cfg["knowledge_base_id"]
    )

    return NL2SQLAgent(
        llm=llm,
        schema_retriever=schema_retriever,
        cfg=cfg
    )
