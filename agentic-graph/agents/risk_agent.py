from strands import Agent

risk_agent = Agent(
    name="risk_analyzer",
    system_prompt="""
    Assess escalation risk based on:
    - Complaint severity
    - Sentiment
    Return: Low, Medium, High
    """
)
