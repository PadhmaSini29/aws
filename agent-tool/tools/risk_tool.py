from strands import tool
from agents.risk_assessment_agent import build_risk_assessment_agent


@tool
def assess_risk(customer_json: str, compliance_result_json: str) -> str:
    """
    Assess BFSI risk using customer data and compliance output.

    Args:
        customer_json: Customer profile as JSON string
        compliance_result_json: Compliance result JSON string

    Returns:
        STRICT JSON risk assessment
    """
    agent = build_risk_assessment_agent()

    prompt = f"""
Customer profile (JSON):
{customer_json}

Compliance result (JSON):
{compliance_result_json}

Assess risk and return STRICT JSON only.
"""

    response = agent(prompt)
    return str(response)
