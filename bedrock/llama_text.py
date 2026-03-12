import boto3
import json

client = boto3.client("bedrock-runtime")

prompt = "Write a poem about technology."

body = {
    "prompt": prompt,
    "max_gen_len": 200,
    "temperature": 0.7,
    "top_p": 0.9
}

response = client.invoke_model(
    modelId="meta.llama3-8b-instruct-v1:0",
    body=json.dumps(body),
    accept="application/json",
    contentType="application/json"
)

result = json.loads(response['body'].read())
print(result["generation"])
