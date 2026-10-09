import os
import json
import boto3
from dotenv import load_dotenv

load_dotenv(override=True)

client = boto3.client(
    "bedrock-runtime",
    region_name=os.getenv("AWS_REGION"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)

profile_arn = os.getenv("BEDROCK_INFERENCE_PROFILE_ARN")

body = {
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 100,
    "messages": [{"role": "user", "content": "Say hello in one word."}],
}

response = client.invoke_model(
    modelId=profile_arn,
    body=json.dumps(body),
)

result = json.loads(response["body"].read())
print(result)