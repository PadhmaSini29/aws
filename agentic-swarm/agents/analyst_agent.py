from strands import Agent

analyst_agent = Agent(
    name="analyst_agent",
    system_prompt=(
        "You are a FinOps Analyst Agent. "
        "Given cost data, find optimization opportunities and quantify impact when possible. "
        "Suggest concrete actions (rightsizing, schedules, Savings Plans, log retention)."
    ),
    model="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    tools=[] 
)
