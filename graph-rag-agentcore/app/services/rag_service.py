# app/services/rag_service.py

from __future__ import annotations

from typing import Dict, Union

from app.agent.agent_core import AgentCore
from app.graph.graph_retriever import retrieve_graph_context

# Initialize once (important for Lambda cold-start optimization)
_AGENT = AgentCore()


def answer_question(
    *,
    question: str,
    session_id: str,
    debug: bool = False,
) -> Union[str, Dict]:
    """
    Orchestrates Graph-RAG → AgentCore execution.

    Flow:
    1. Retrieve graph-based context from Neo4j (Graph RAG)
    2. Inject context into AgentCore
    3. Return grounded answer (with session memory)

    Args:
        question: User question
        session_id: Session ID for agent memory
        debug: If True, return debug information

    Returns:
        Either answer string or structured dict (if debug=True)
    """

    # 1️⃣ Graph RAG: Retrieve relationship-based context
    graph_context = retrieve_graph_context(question=question)

    # 2️⃣ AgentCore execution (memory is session-scoped)
    answer = _AGENT.run(
        question=question,
        graph_context=graph_context,
        session_id=session_id,
    )

    if debug:
        return {
            "answer": answer,
            "debug": {
                "graph_context": graph_context,
                "session_id": session_id,
            },
        }

    return answer
