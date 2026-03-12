# app/main.py
"""
Local entry point (CLI/testing) for the Graph-RAG + AgentCore Knowledge Assistant.

Run:
  python -m app.main
or (from repo root):
  python app/main.py

Notes:
- Expects Neo4j running locally (bolt://localhost:7687 by default).
- Expects AWS credentials configured if your agent uses Bedrock/AgentCore.
"""

from __future__ import annotations

import json
import os
import uuid

from app.services.rag_service import answer_question


def _print_banner() -> None:
    print("\n=== Graph-RAG + AgentCore Knowledge Assistant (Local) ===")
    print("Type your question and press Enter.")
    print("Commands: ':new' (new session), ':exit' (quit)\n")


def main() -> None:
    # Keep a single session across turns to demonstrate memory
    session_id = os.getenv("SESSION_ID") or f"local-{uuid.uuid4().hex}"

    _print_banner()
    print(f"Session: {session_id}\n")

    while True:
        q = input("You > ").strip()
        if not q:
            continue

        if q.lower() in (":exit", "exit", "quit", ":q"):
            print("Bye 👋")
            return

        if q.lower() in (":new", "new"):
            session_id = f"local-{uuid.uuid4().hex}"
            print(f"\n(New session started) Session: {session_id}\n")
            continue

        try:
            result = answer_question(question=q, session_id=session_id)

            # result can be either a string or a dict, depending on your rag_service
            if isinstance(result, dict):
                print("\nAssistant > " + (result.get("answer") or "").strip())
                # Optional debug fields
                if result.get("debug"):
                    print("\n--- debug ---")
                    print(json.dumps(result["debug"], indent=2))
                    print("-------------\n")
                else:
                    print()
            else:
                print("\nAssistant > " + str(result).strip() + "\n")

        except Exception as e:
            print(f"\n[ERROR] {type(e).__name__}: {e}\n")


if __name__ == "__main__":
    main()
