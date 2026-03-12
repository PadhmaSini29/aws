from strands import Agent

knowledge_agent = Agent(
    name="knowledge_agent",
    system_prompt="""
    Provide short factual answers using internal knowledge.
    If answer is not found, say: NOT_FOUND
    """
)
