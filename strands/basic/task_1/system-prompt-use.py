from strands import Agent

dog_breed_helper = Agent(
    model="qwen.qwen3-32b-v1:0",
    system_prompt=(
        "You are a dog breed expert who helps new pet parents choose the right breed.\n"
        "Rules:\n"
        "1. Give benefits and challenges.\n"
        "2. Only list 3 breeds.\n"
        "3. Use simple language.\n"
    )
)

query = "I work 9-5 and hike on weekends. Suggest a beginner-friendly dog breed."

response = dog_breed_helper(query)

print("\n=== Agent Response ===\n")
print(response)
