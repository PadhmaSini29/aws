from strands import Agent

agent = Agent(
    model="amazon.titan-text-lite-v1"
)

response = agent("Explain Agentic AI.")
print(response)
