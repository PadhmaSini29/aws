from strands import Agent

decision_agent = Agent(
    name="decision_agent",
    system_prompt="""
    Decide final action:
    - AUTO_RESOLVE
    - REQUEST_MORE_INFO
    - ESCALATE_TO_HUMAN
    - PRIORITY_SUPPORT

    Provide reason in 1 sentence.
    """
)
