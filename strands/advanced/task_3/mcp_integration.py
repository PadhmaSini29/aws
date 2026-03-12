import os
from dotenv import load_dotenv

from strands import Agent
from strands.models.bedrock import BedrockModel
from strands.tools.mcp import MCPClient
from strands_tools import file_read, file_write

from mcp import stdio_client, StdioServerParameters
import boto3

# ---------------------------------------------
# 🔵 Load environment (.env is optional)
# ---------------------------------------------
load_dotenv()

# ---------------------------------------------
# 🔵 Load AWS credentials (with AWS CLI fallback)
# ---------------------------------------------
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN")

# Region priority:
# 1. ENV
# 2. AWS CLI config
# 3. fallback "us-west-2"
session = boto3.Session()
AWS_REGION = os.getenv("AWS_REGION") or session.region_name or "us-west-2"

# Load credentials from AWS CLI if ENV missing
if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
    print("ENV credentials not found — loading from AWS CLI profile...")

    creds = session.get_credentials()
    if creds is None:
        raise ValueError("No AWS credentials found in environment OR AWS CLI.")

    frozen = creds.get_frozen_credentials()
    AWS_ACCESS_KEY_ID = frozen.access_key
    AWS_SECRET_ACCESS_KEY = frozen.secret_key
    AWS_SESSION_TOKEN = frozen.token  # optional

print("AWS credentials loaded successfully.")
print(f"Region: {AWS_REGION}")

# Build env dict safely for MCP server
aws_env = {
    "AWS_ACCESS_KEY_ID": AWS_ACCESS_KEY_ID,
    "AWS_SECRET_ACCESS_KEY": AWS_SECRET_ACCESS_KEY,
    "AWS_REGION": AWS_REGION,
}
if AWS_SESSION_TOKEN:
    aws_env["AWS_SESSION_TOKEN"] = AWS_SESSION_TOKEN

# ---------------------------------------------
# 🔵 Configure Bedrock Model  (correct syntax)
# ---------------------------------------------
model = BedrockModel(
    model_id="anthropic.claude-3-5-haiku-20241022-v1:0",
    region_name=AWS_REGION,     # ✔ valid parameter
    max_tokens=8000
)

# ---------------------------------------------
# 🔵 Path to uvx.exe for Windows
# ---------------------------------------------
UVX_PATH = r"C:\Users\lgspa\.local\bin\uvx.exe"

# ---------------------------------------------
# 🔵 MCP Servers (Windows-compatible)
# ---------------------------------------------
# MUST include "run" before package name on Windows.
# MUST use full uvx.exe path.
# MUST extend timeout from default 30 sec → 60 sec.

aws_docs_mcp_client = MCPClient(
    lambda: stdio_client(
        StdioServerParameters(
            command=UVX_PATH,
            args=["run", "awslabs.aws-documentation-mcp-server@latest"],
            timeout=60
        )
    )
)

aws_pricing_mcp_client = MCPClient(
    lambda: stdio_client(
        StdioServerParameters(
            command=UVX_PATH,
            args=["run", "awslabs.aws-pricing-mcp-server@latest"],
            env=aws_env,
            timeout=60
        )
    )
)

# ---------------------------------------------
# 🔵 System Prompt
# ---------------------------------------------
SYSTEM_PROMPT = """
You are an AWS Solutions Architect with expertise in AWS services.

Use AWS Documentation MCP + AWS Pricing MCP to answer questions.

Workflow:
1. Use get_pricing_service_codes()
2. Use get_pricing("AmazonS3", "us-east-1")
3. Use get_pricing("AmazonCloudFront", "us-east-1")
4. Provide documentation URLs + pricing summary
5. Save output using file_write
"""

prompt = """
Create a summary of hosting a static website on AWS using S3 + CloudFront.
Include documentation URLs and pricing. Save result to static_website_aws.md.
"""

# ---------------------------------------------
# 🔵 Connect MCP + Run Agent
# ---------------------------------------------
with aws_docs_mcp_client, aws_pricing_mcp_client:
    tools = (
        aws_docs_mcp_client.list_tools_sync()
        + aws_pricing_mcp_client.list_tools_sync()
    )

    print(f"\nLoaded {len(tools)} MCP tools:")
    for tool in tools:
        print("-", tool.tool_name)

    aws_agent = Agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tools=[file_read, file_write, tools]
    )

    result = aws_agent(prompt)
    print("\nAgent Output:\n", result.message)

# ---------------------------------------------
# 🔵 Metrics
# ---------------------------------------------
summary = aws_agent.event_loop_metrics.get_summary()

print("\nToken Usage:")
print(summary["accumulated_usage"])

print("\nTool Usage:")
for name, details in summary["tool_usage"].items():
    print(name, details)
