from strands import tool
from agents.policy_interpretation_agent import build_policy_interpretation_agent


@tool
def interpret_policy(policy_text: str) -> str:
    """
    Interpret BFSI policy text into clear, actionable guidance.

    Args:
        policy_text: Raw or structured policy content

    Returns:
        Plain-English interpretation with do/don't rules
    """
    agent = build_policy_interpretation_agent()
    response = agent(policy_text)
    return str(response)
