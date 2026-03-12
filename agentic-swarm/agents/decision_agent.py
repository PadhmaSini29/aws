from strands import Agent

decision_agent = Agent(
    name="decision_agent",
    system_prompt=(
        "You are the FinOps Decision Agent. "
        "Combine inputs from all agents into a decision-ready action plan: "
        "1) What to do today, 2) What to do this week, 3) What to monitor. "
        "Keep it crisp and ordered."
    ),
    model="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    tools=[] 
)
