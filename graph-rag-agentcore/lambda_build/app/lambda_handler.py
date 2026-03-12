# app/lambda_handler.py
"""
AWS Lambda entry for the Graph-RAG + AgentCore Knowledge Assistant.

Supports:
1) API Gateway HTTP API (v2)
   GET /ask?q=...&session_id=...

2) Direct Lambda invoke
   {
     "question": "...",
     "session_id": "..."
   }

Returns JSON:
{
  "answer": "...",
  "session_id": "...",
  "meta": {}
}
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, Optional

from app.services.rag_service import answer_question


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def _get_query_param(event: Dict[str, Any], key: str) -> Optional[str]:
    params = event.get("queryStringParameters") or {}
    if isinstance(params, dict):
        value = params.get(key)
        if value is not None:
            return str(value)
    return None


def _json_response(status_code: int, body: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            # CORS (adjust if you want stricter rules)
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
        },
        "body": json.dumps(body),
    }


# ---------------------------------------------------------
# Lambda entrypoint
# ---------------------------------------------------------

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    # -----------------------------------------------------
    # Handle CORS preflight
    # -----------------------------------------------------
    http_method = (
        event.get("requestContext", {})
        .get("http", {})
        .get("method")
        or event.get("httpMethod")
    )

    if http_method == "OPTIONS":
        return _json_response(200, {"ok": True})

    # -----------------------------------------------------
    # Extract inputs
    # -----------------------------------------------------
    question = _get_query_param(event, "q") or _get_query_param(event, "question")
    session_id = _get_query_param(event, "session_id")

    # Try JSON body (API Gateway / direct invoke)
    if question is None:
        raw_body = event.get("body")
        if raw_body:
            try:
                body = json.loads(raw_body) if isinstance(raw_body, str) else raw_body
                if isinstance(body, dict):
                    question = body.get("q") or body.get("question")
                    session_id = session_id or body.get("session_id")
            except Exception:
                pass  # fall through to validation

    # Direct invoke fallback
    if question is None:
        question = event.get("question") or event.get("q")
        session_id = session_id or event.get("session_id")

    # -----------------------------------------------------
    # Validate
    # -----------------------------------------------------
    if not question or not str(question).strip():
        return _json_response(
            400,
            {
                "error": "Missing required parameter: 'q' (or 'question')",
                "example": {
                    "GET": "/ask?q=What%20is%20Graph%20RAG%3F&session_id=my-session-123",
                    "POST": {
                        "question": "What is Graph RAG?",
                        "session_id": "my-session-123",
                    },
                },
            },
        )

    # Generate session if not provided
    session_id = session_id or f"api-{uuid.uuid4().hex}"

    # -----------------------------------------------------
    # Execute Graph-RAG + Agent
    # -----------------------------------------------------
    try:
        result = answer_question(
            question=str(question).strip(),
            session_id=session_id,
        )

        if isinstance(result, dict):
            answer = str(result.get("answer", "")).strip()
            meta = result.get("meta") or {}
        else:
            answer = str(result).strip()
            meta = {}

        return _json_response(
            200,
            {
                "answer": answer,
                "session_id": session_id,
                "meta": meta,
            },
        )

    except Exception as exc:
        return _json_response(
            500,
            {
                "error": f"{type(exc).__name__}: {exc}",
                "session_id": session_id,
            },
        )
