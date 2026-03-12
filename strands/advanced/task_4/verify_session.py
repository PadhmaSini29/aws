import os
from pathlib import Path
from dotenv import load_dotenv

from strands import Agent
from strands.models.bedrock import BedrockModel
from strands.session.file_session_manager import FileSessionManager
from strands.agent.conversation_manager import SummarizingConversationManager

load_dotenv()
AWS_REGION = os.getenv("AWS_REGION", "us-west-2")
model = BedrockModel(
    model_id="anthropic.claude-3-5-haiku-20241022-v1:0",
    region_name=AWS_REGION,
    max_tokens=4000,
)

# -----------------------------------------
# Create a sessions directory
# -----------------------------------------
SESSION_DIR = Path("./sessions")
SESSION_DIR.mkdir(exist_ok=True)

SESSION_ID = "demo_session_001"

# -----------------------------------------
# Conversation + Session Managers
# -----------------------------------------
session_manager = FileSessionManager(
    session_id=SESSION_ID,
    storage_dir=str(SESSION_DIR),
)

conversation_manager = SummarizingConversationManager(
    summary_ratio=0.5,             # summarize older messages
    preserve_recent_messages=3     # never summarize last 3 interactions
)

# -----------------------------------------
# System Prompt
# -----------------------------------------
SYSTEM_PROMPT = """
You are a friendly memory-capable assistant.
You remember user details across the entire session.
"""

# -----------------------------------------
# Create the agent with session + conversation management
# -----------------------------------------
agent = Agent(
    system_prompt=SYSTEM_PROMPT,
    model=model,
    conversation_manager=conversation_manager,
    session_manager=session_manager,
)

# -----------------------------------------
# Display session state
# -----------------------------------------
print("\n=== Session Started ===")
print("Session ID:", SESSION_ID)
print("Current stored state:", agent.state.get())

# -----------------------------------------
# Example prompt
# -----------------------------------------
prompt = """
Hi, my name is Alice. I love hiking and reading science fiction novels.
I also enjoy cooking and trying out new recipes.
"""

print("\nUser prompt sent to agent:")
print(prompt)

result = agent(prompt)

print("\n=== Agent Response ===")
print(result.message)

# Ask again in same session (memory test)
follow_up = "Do you remember my name and what I enjoy doing?"
print("\nFollow-up question:", follow_up)

follow_up_result = agent(follow_up)

print("\n=== Agent Follow-Up Response ===")
print(follow_up_result.message)

print("\nSession saved in:", SESSION_DIR)
