# src/bedrock_client_sonnet5.py
"""
Same as bedrock_client.py but uses Claude Sonnet 5 via a separate
inference profile ARN. Used exclusively by the multi-agent pipeline.
"""
import os
import json
import boto3
from dotenv import load_dotenv

load_dotenv(override=True)

_client = boto3.client(
    "bedrock-runtime",
    region_name=os.getenv("AWS_REGION"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)

_SONNET5_ARN = os.getenv("BEDROCK_INFERENCE_PROFILE_ARN_sonnet5")
if not _SONNET5_ARN:
    print("[WARN] No BEDROCK_INFERENCE_PROFILE_ARN_sonnet5 in .env — Sonnet 5 calls will fail.")


def call_sonnet5(
    prompt: str,
    max_tokens: int = 3000,
) -> str:
    """Plain text call to Claude Sonnet 5."""
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    response = _client.invoke_model(
        modelId=_SONNET5_ARN,
        body=json.dumps(body),
    )
    result = json.loads(response["body"].read())
    # Claude Sonnet 5 may return multiple content blocks
    for block in result.get("content", []):
        if isinstance(block, dict) and block.get("type") == "text":
            return block.get("text", "").strip()
        if isinstance(block, str):
            return block.strip()
    # fallback
    return str(result.get("content", "")).strip()
def call_sonnet5_with_tools(
    messages: list[dict],
    tools: list[dict],
    system: str = "",
    max_tokens: int = 3000,
    temperature: float = 0,
) -> dict:
    """Tool-use call to Claude Sonnet 5. Returns full parsed response body."""
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "system": system,
        "messages": messages,
        "tools": tools,
        "tool_choice": {"type": "any"},
    }
    response = _client.invoke_model(
        modelId=_SONNET5_ARN,
        body=json.dumps(body),
    )
    return json.loads(response["body"].read())