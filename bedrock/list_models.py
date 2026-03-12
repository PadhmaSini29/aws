import boto3

client = boto3.client("bedrock", region_name="us-west-2")

response = client.list_foundation_models()

for m in response['modelSummaries']:
    print(m['modelId'], " | ", m['modelName'], " | ", m['providerName'])
