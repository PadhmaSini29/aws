import json
import uuid
import boto3
import os
import sys
def invoke_agent(agent_arn, prompt, region=None, session_id=None):
    """
    Invoke an Amazon Bedrock AgentCore Runtime agent programmatically
    """
    if not region:
        region = agent_arn.split(":")[3]

    client = boto3.client(
        "bedrock-agentcore",
        region_name=region
    )

    if not session_id:
        session_id = str(uuid.uuid4())

    payload = json.dumps({"prompt": prompt}).encode("utf-8")

    try:
        response = client.invoke_agent_runtime(
            agentRuntimeArn=agent_arn,
            runtimeSessionId=session_id,
            payload=payload,
            qualifier="DEFAULT"
        )

        chunks = []
        for chunk in response.get("response", []):
            chunks.append(chunk.decode("utf-8"))

        result = json.loads("".join(chunks))
        return result, session_id

    except Exception as e:
        print(f"❌ Error invoking agent: {e}")
        return None, session_id


def main():

    agent_arn = os.getenv("AGENT_ARN")
    region = os.getenv("AWS_REGION")

    if len(sys.argv) > 1:
        agent_arn = sys.argv[1]
    if len(sys.argv) > 2:
        region = sys.argv[2]

    if not agent_arn:
        print("Usage: python invoke_agent.py <AGENT_ARN> [REGION]")
        print("Or set AGENT_ARN environment variable")
        sys.exit(1)

    if not region:
        region = agent_arn.split(":")[3]

    print(f"\nInvoking AgentCore Runtime")
    print(f"Agent ARN : {agent_arn}")
    print(f"Region   : {region}")
    print("-" * 60)

    test_prompts = [
        "What is 25 + 30?",
        "Now multiply that result by 2",
        "What was the first calculation I asked you to do?",
        "Calculate the square root of 144",
        "Add 10 to the previous result",
        "What is 15 * 8 + 7?",
        "Divide the previous result by 3"
    ]

    session_id = str(uuid.uuid4())
    print(f"Session ID: {session_id}")
    print("=" * 60)

    for i, prompt in enumerate(test_prompts, start=1):
        print(f"\n{i}. Prompt: {prompt}")
        print("-" * 60)

        result, _ = invoke_agent(
            agent_arn=agent_arn,
            prompt=prompt,
            region=region,
            session_id=session_id
        )

        if result and isinstance(result, dict):
            print("Response:")
            print(result.get("response", result))
        else:
            print("❌ No response from agent")

        print("-" * 60)


if __name__ == "__main__":
    main()
