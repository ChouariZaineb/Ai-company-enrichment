"""
Wraps AWS Bedrock calls to DeepSeek 3.2, via the org's application inference
profile in eu-north-1 (Stockholm) — a different region from the Claude
profile. Uses Bedrock's Converse API (client.converse), which is a unified
interface across model providers, rather than hand-rolling DeepSeek's raw
request schema.
"""
import os
import boto3
from dotenv import load_dotenv

load_dotenv(override=True)

_client = boto3.client(
    "bedrock-runtime",
    region_name=os.getenv("AWS_REGION_DEEPSEEK"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)

_INFERENCE_PROFILE_ARN = os.getenv("BEDROCK_DEEPSEEK_INFERENCE_PROFILE_ARN")
if not _INFERENCE_PROFILE_ARN:
    print("[WARN] No BEDROCK_DEEPSEEK_INFERENCE_PROFILE_ARN found in .env — DeepSeek calls will fail.")


def call_deepseek_bedrock(prompt: str, max_tokens: int = 3000, temperature: float = 0) -> str:
    """
    Sends a single-turn prompt to DeepSeek 3.2 via Bedrock's Converse API
    and returns the raw text response. Raises on failure — callers handle retries.
    """
    response = _client.converse(
        modelId=_INFERENCE_PROFILE_ARN,
        messages=[
            {"role": "user", "content": [{"text": prompt}]}
        ],
        inferenceConfig={
            "maxTokens": max_tokens,
            "temperature": temperature,
        },
    )
    return response["output"]["message"]["content"][0]["text"].strip()


def call_deepseek_bedrock_with_tools(
    messages: list[dict],
    tools: list[dict] = None,
    system: str = "",
    max_tokens: int = 2000,
    temperature: float = 0,
) -> dict:
    """
    Calls DeepSeek via Bedrock Converse API with optional tool use.
    Returns the full parsed response body.
    """
    request_body = {
        "messages": messages,
        "inferenceConfig": {
            "maxTokens": max_tokens,
            "temperature": temperature,
        },
    }

    if system:
        request_body["system"] = [{"text": system}]

    if tools:
        request_body["toolConfig"] = {
            "tools": tools,
            "toolChoice": {"any": {}},
        }

    response = _client.converse(
        modelId=_INFERENCE_PROFILE_ARN,
        **request_body,
    )

    return response