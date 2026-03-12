from strands import Agent

sentiment_agent = Agent(
    name="sentiment_analyzer",
    system_prompt="""
    Analyze customer sentiment.
    Return: Positive, Neutral, Frustrated, Angry
    """
)
