import os
from dotenv import load_dotenv
load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
TOOL_MODEL_ID = os.getenv(
    "TOOL_MODEL_ID",
    "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
)
ORCHESTRATOR_MODEL_ID = os.getenv(
    "ORCHESTRATOR_MODEL_ID",
    "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
)
def get_model(model_id: str):
    """
    Returns a BedrockModel if available,
    otherwise falls back to raw model_id string.
    """
    try:
        from strands.models import BedrockModel
        return BedrockModel(
            model_id=model_id,
            max_tokens=2048,
            temperature=0.2
        )
    except Exception:
        return model_id
