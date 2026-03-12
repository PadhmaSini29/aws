import json
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    """
    MCP Gateway → Lambda handler

    Expected MCP event format:
    {
      "toolName": "graph_rag_tool",
      "input": {
        "query": "Explain Graph RAG"
      }
    }
    """

    logger.info("🔹 Raw MCP event: %s", json.dumps(event))

    # MCP always sends tool input inside "input"
    tool_input = event.get("input", {})
    query = tool_input.get("query", "").strip()

    logger.info("🔹 Extracted query: %s", query)

    if not query:
        return {
            "error": "No query provided to Graph RAG tool"
        }

    # --------------------------------------------------
    # STUB Graph RAG response (replace later with Neo4j)
    # --------------------------------------------------
    graph_context = (
        "Graph RAG combines knowledge graphs with retrieval-augmented "
        "generation to improve reasoning over connected data."
    )

    # MCP tools must return plain JSON (NOT statusCode/body)
    return {
        "query": query,
        "graph_context": graph_context,
        "note": "Stub response. Next step: connect Neo4j."
    }
