from strands import tool
from agents.decision_agent import build_decision_agent


@tool
def decide_action(compliance_result_json: str, risk_result_json: str) -> str:
    """
    Make a final BFSI decision based on compliance and risk outputs.

    Args:
        compliance_result_json: Compliance result JSON
        risk_result_json: Risk assessment JSON

    Returns:
        STRICT JSON final decision
    """
    agent = build_decision_agent()

    prompt = f"""
Compliance result (JSON):
{compliance_result_json}

Risk assessment (JSON):
{risk_result_json}

Make a final decision and return STRICT JSON only.
"""

    response = agent(prompt)
    return str(response)
