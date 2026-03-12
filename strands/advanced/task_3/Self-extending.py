import os
from typing import Optional
from dotenv import load_dotenv
from strands import Agent, tool
from strands.models.bedrock import BedrockModel
from strands_tools import editor as base_editor

load_dotenv()
AWS_REGION = os.getenv("AWS_REGION", "us-west-2")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
    import boto3
    print("ENV credentials not found — loading from AWS CLI profile...")
    session = boto3.Session()
    creds = session.get_credentials().get_frozen_credentials()
    AWS_ACCESS_KEY_ID = creds.access_key
    AWS_SECRET_ACCESS_KEY = creds.secret_key

print("AWS credentials loaded successfully.")

# -----------------------------------------------------------------------------
# 2️⃣ Bedrock Model (clean config)
# -----------------------------------------------------------------------------
model = BedrockModel(
    model_id="anthropic.claude-3-5-haiku-20241022-v1:0",
    max_tokens=4000,
    temperature=0.7,
)

# -----------------------------------------------------------------------------
# 3️⃣ Ensure tools/ exists and is importable
# -----------------------------------------------------------------------------
os.makedirs("tools", exist_ok=True)
init_file = os.path.join("tools", "__init__.py")
if not os.path.exists(init_file):
    with open(init_file, "w") as f:
        f.write("")

# -----------------------------------------------------------------------------
# 4️⃣ Wrapped editor tool that forces paths into ./tools
# -----------------------------------------------------------------------------
@tool
def safe_editor(path: str, content: str, mode: Optional[str] = "write") -> str:
    """
    A safe wrapper around the base editor tool that ensures:
    - All files are written under ./tools/
    - Prevents absolute paths like /tools/... from going somewhere else
    """
    # Normalize path
    normalized = path.replace("\\", "/")

    # Strip leading slashes so "/tools/xxx.py" becomes "tools/xxx.py"
    while normalized.startswith("/"):
        normalized = normalized[1:]

    # If path does not start with "tools/", force it
    if not normalized.startswith("tools/"):
        normalized = f"tools/{normalized}"

    # Now call the original editor with safe normalized path
    return base_editor(path=normalized, content=content, mode=mode)


# -----------------------------------------------------------------------------
# 5️⃣ System Prompt
# -----------------------------------------------------------------------------
SYSTEM_PROMPT = """
You are a self-extending AI agent.

You can:
- Create NEW Python tools in the 'tools/' directory using the 'safe_editor' tool.
- Tools are loaded automatically from the local 'tools' directory.
- ALWAYS use the @tool decorator.

When creating a tool:
- Use paths like 'add_numbers.py' or 'tools/add_numbers.py'
- Do NOT use absolute paths like '/tools/...'
"""

# -----------------------------------------------------------------------------
# 6️⃣ Create Agent
# -----------------------------------------------------------------------------
agent = Agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[safe_editor],
    load_tools_from_directory=True,
)

# -----------------------------------------------------------------------------
# 7️⃣ Interactive Loop
# -----------------------------------------------------------------------------
print("🟦 Self-Extending Agent (Bedrock Edition)")
print("Type a request to create/use tools. Type 'exit' to quit.\n")

while True:
    query = input("\nUser> ").strip()

    if query.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    if not query:
        continue

    result = agent(query)
    print("\nAssistant:\n", result.message)
