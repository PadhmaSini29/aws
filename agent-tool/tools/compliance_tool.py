from strands import tool
from agents.compliance_agent import build_compliance_agent


@tool
def validate_compliance(customer_json: str, policy_guidance: str) -> str:
    """
    Validate a customer profile against BFSI policy guidance.

    Args:
        customer_json: Customer profile as JSON string
        policy_guidance: Interpreted policy rules or guidance

    Returns:
        STRICT JSON compliance result
    """
    agent = build_compliance_agent()

    prompt = f"""
Customer profile (JSON):
{customer_json}

Policy guidance:
{policy_guidance}

Validate compliance and return STRICT JSON only.
"""

    response = agent(prompt)
    return str(response)
