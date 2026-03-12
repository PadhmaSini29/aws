import json
from bedrock_agentcore_starter_toolkit.operations.gateway.client import GatewayClient

GATEWAY_ID = "graph-rag-gateway-zdnnz4v8bn"
GATEWAY_ARN = "arn:aws:bedrock-agentcore:us-east-1:762233739050:gateway/graph-rag-gateway-zdnnz4v8bn"
LAMBDA_ARN = "arn:aws:lambda:us-east-1:762233739050:function:GraphRagGatewayTool"

client = GatewayClient()

response = client.create_mcp_gateway_target(
    name="graph-rag-tools",
    gateway={
        # 🔑 REQUIRED BY SDK (even though undocumented)
        "gatewayId": GATEWAY_ID,

        # 🔑 REQUIRED BY AWS backend
        "gatewayArn": GATEWAY_ARN,
    },
    target_type="lambda",
    target_payload={
        "lambdaArn": LAMBDA_ARN,
        "toolSchema": {
            "inlinePayload": [
                {
                    "name": "graph_rag_tool",
                    "description": "Query Graph RAG system via Lambda",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "User question for Graph RAG"
                            }
                        },
                        "required": ["query"]
                    }
                }
            ]
        }
    }
)

print("✅ MCP Gateway target created successfully")
print(json.dumps(response, indent=2, default=str))
