from strands import Agent
from strands_tools import http_request

dog_breed_helper = Agent(
    model="qwen.qwen3-32b-v1:0",
    system_prompt=(
        "You are a dog breed and research assistant.\n"
        "Always use the http_request tool for online searches.\n"
        "Keep responses short and simple.\n"
    ),
    tools=[http_request]
)

query = """
1. Recommend 2 beginner-friendly dog breeds and explain why.
2. Fetch Wikipedia's summary for 'Dog breed' using this URL:
   https://en.wikipedia.org/api/rest_v1/page/summary/Dog_breed
   Summarize it in 3 bullet points.
"""

response = dog_breed_helper(query)

print("\n=== Agent Response ===\n")
print(response)
