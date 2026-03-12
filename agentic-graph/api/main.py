from fastapi import FastAPI, Query
from graphs.support_graph import build_support_graph

app = FastAPI(
    title="Agentic Customer Support Engine",
    description="Multi-agent decision system using Strands Agent Graph",
    version="1.0.0"
)

# Build the agent graph once at startup
graph = build_support_graph()


@app.post("/support")
def support(
    query: str = Query(..., description="Customer support query")
):
    """
    Process a customer support query through a multi-agent graph
    and return a decision-ready response.
    """

    # Execute agent graph
    result = graph(query)

    # Safely extract decision agent output
    decision_result = result.results.get("decision")

    final_decision = (
        str(decision_result.result)
        if decision_result and decision_result.result
        else "NO_DECISION_RETURNED"
    )

    return {
        "final_decision": final_decision,
        "execution_order": [node.node_id for node in result.execution_order],
        "token_usage": result.accumulated_usage,
    }
