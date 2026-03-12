# runtime_app.py
import uuid
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


from bedrock_agentcore import BedrockAgentCoreApp
from strands import Agent

app = BedrockAgentCoreApp()

agent = Agent(
    name="graph_rag_agent",
    system_prompt="You are a helpful AI assistant."
)

def handler(event: dict, context=None) -> dict:
    """
    This function is REQUIRED by AgentCore.
    Any exception here = 500 error.
    """
    request_id = str(uuid.uuid4())

    logger.info(f"[REQUEST_START] request_id={request_id}")
    logger.info(f"Input event: {event}")

    # AgentCore may wrap input inside "input"
    payload = event.get("input", event)

    question = payload.get("prompt") or payload.get("question") or ""
    session_id = payload.get("session_id")

    # Absolute safest possible prompt
    prompt = f"Answer the following question:\n{question}"
    logger.info(f"Question received: {question}")
    response = agent(
        prompt,
        session_id=session_id
    )
    logger.info(f"[REQUEST_END] request_id={request_id}")
    return {
        "output": str(response)
    }

# --------------------------------------------------
# 🔑 CRITICAL LINE (MOST PEOPLE MISS THIS)
# --------------------------------------------------
app.entrypoint(handler)

# --------------------------------------------------
# Local container start
# --------------------------------------------------
if __name__ == "__main__":
    app.run()
