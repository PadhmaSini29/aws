from strands import Agent

risk_agent = Agent(
    name="risk_agent",
    system_prompt=(
        "You are a FinOps Risk Agent. "
        "Check for risks: performance/SLA impact, security, hidden costs (NAT egress, log spikes). "
        "Flag what must be validated before changes."
    ),
    model="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    tools=[] 
)
