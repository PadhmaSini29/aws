from strands import Agent

research_agent = Agent(
    name="research_agent",
    system_prompt=(
        "You are a FinOps Research Agent. "
        "Identify cost drivers and what they typically mean (e.g., NAT, EC2, logs). "
        "Be factual and practical."
    ),
    model="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    tools=[] 
)
