# app/agent/agent_core.py

from __future__ import annotations

from typing import Optional

from strands import Agent

from app.agent.prompts import SYSTEM_PROMPT, TASK_PROMPT_TEMPLATE


class AgentCore:
    """
    AgentCore wrapper using Strands Agent.

    Responsibilities:
    - Initialize the agent runtime
    - Inject Graph-RAG context into prompts
    - Support session-based execution (session_id)

    NOTE:
    The installed Strands Agent SDK does NOT support `enable_memory`.
    Session handling is still passed via `session_id` for forward compatibility.
    """

    def __init__(self) -> None:
        # Initialize agent WITHOUT enable_memory (not supported in SDK)
        self._agent = Agent(
            system_prompt=SYSTEM_PROMPT
        )

    def run(
        self,
        *,
        question: str,
        graph_context: str,
        session_id: Optional[str] = None,
    ) -> str:
        """
        Execute the agent with Graph-RAG context.

        Args:
            question: User question
            graph_context: Context retrieved from Neo4j graph
            session_id: Session identifier (future memory compatibility)

        Returns:
            Agent response text
        """

        # Build final prompt using graph-derived context
        prompt = TASK_PROMPT_TEMPLATE.format(
            graph_context=graph_context,
            question=question,
        )

        # Execute agent (session_id passed but not enforced by SDK)
        response = self._agent(
            prompt,
            session_id=session_id,
        )

        return str(response)
