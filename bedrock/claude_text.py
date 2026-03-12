import boto3
import json

client = boto3.client("bedrock-runtime", region_name="us-east-1")

prompt = "Explain machine learning in simple words."

body = {
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 300,
    "temperature": 0.7,
    "messages": [
        {"role": "user", "content": prompt}
    ]
}

response = client.invoke_model(
    modelId="anthropic.claude-3-sonnet-20240229-v1:0",
    body=json.dumps(body),
    accept="application/json",
    contentType="application/json"
)

result = json.loads(response["body"].read().decode())
print("\n===== RESPONSE =====\n")
print(result["content"][0]["text"])
