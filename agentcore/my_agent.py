from bedrock_agentcore import BedrockAgentCoreApp
from strands import Agent
import os
from strands_tools import calculator

SYSTEM_PROMPT = (
    "You are a helpful assistant that can perform calculations. "
    "Use the calculate tool for any math problems."
)

MODEL_ID = os.getenv(
    "MODEL_ID",
    "us.anthropic.claude-3-5-haiku-20241022-v1:0"
)

agent = None
app = BedrockAgentCoreApp()

def create_agent(tools):
    global agent
    if agent is None:
        agent = Agent(
            model=MODEL_ID,
            tools=[tools],
            system_prompt=SYSTEM_PROMPT
        )
    return agent

@app.entrypoint
def invoke(payload):
    agent = create_agent(calculator)
    prompt = payload.get("prompt", "Hello!")
    result = agent(prompt)

    return {
        "response": result.message.get("content", [{}])[0].get("text", "")
    }

if __name__ == "__main__":
    app.run()
