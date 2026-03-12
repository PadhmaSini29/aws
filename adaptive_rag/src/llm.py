import boto3
from botocore.exceptions import ClientError
from typing import Optional

class BedrockClaude:
    def __init__(self, region: str, model_id: str):
        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.model_id = model_id

    def chat(self, system: str, user: str, max_tokens: int = 800) -> str:
        try:
            resp = self.client.converse(
                modelId=self.model_id,
                system=[{"text": system}],
                messages=[{
                    "role": "user",
                    "content": [{"text": user}]
                }],
                inferenceConfig={
                    "maxTokens": max_tokens,
                    "temperature": 0.1
                },
            )
            # Converse returns message blocks; stitch them
            msg = resp["output"]["message"]["content"]
            text_parts = [b.get("text", "") for b in msg if "text" in b]
            return "\n".join(text_parts).strip()
        except ClientError as e:
            raise RuntimeError(f"Bedrock Converse failed: {e}")
