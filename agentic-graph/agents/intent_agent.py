from strands import Agent

intent_agent = Agent(
    name="intent_classifier",
    system_prompt="""
    You classify customer support requests.
    Return one of:
    Billing, Technical, Delivery, Complaint, General
    Only return the label.
    """
)
