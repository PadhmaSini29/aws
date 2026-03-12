"""
Entry point for the Adaptive Structured RAG (NL2SQL) Agent
"""

import argparse
import logging

from config import setup_logging, get_config
from src.agent import create_nl2sql_agent


def main():
    parser = argparse.ArgumentParser(
        description="Adaptive Structured RAG NL2SQL Agent"
    )

    parser.add_argument(
        "--question",
        "-q",
        required=True,
        type=str,
        help="Natural language question to convert to SQL"
    )

    parser.add_argument(
        "--engine",
        "-e",
        choices=["sqlite", "sqllite", "postgres"],
        default="postgres",
        help="Execution engine (sqlite | postgres)"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Starting NL2SQL agent")

    # Load config
    cfg = get_config()

    # Create agent
    agent = create_nl2sql_agent(cfg)

    # Normalize engine name
    engine = "sqlite" if args.engine in ("sqlite", "sqllite") else "postgres"

    # Execute query
    result = agent.answer(
        question=args.question,
        engine=engine
    )

    print("\n==============================")
    print("ENGINE:", result["engine"])
    print("==============================")

    print("\nGENERATED SQL:\n")
    print(result["sql"])

    print("\nRESULT:\n")
    res = result["result"]
    print("Columns:", res.get("columns"))

    for row in res.get("rows", []):
        print(row)

    logger.info("NL2SQL agent finished successfully")


if __name__ == "__main__":
    main()
