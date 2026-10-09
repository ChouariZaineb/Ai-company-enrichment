"""
Wraps AWS Bedrock calls to Claude Sonnet 4.6, via the org's application
inference profile. Mirrors gemini_key_manager.py's role: one place that
owns auth + the raw call, so sector/topic matcher modules stay focused
on prompts and parsing.
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

_INFERENCE_PROFILE_ARN = os.getenv("BEDROCK_INFERENCE_PROFILE_ARN")
if not _INFERENCE_PROFILE_ARN:
    print("[WARN] No BEDROCK_INFERENCE_PROFILE_ARN found in .env — Bedrock calls will fail.")


def call_claude_bedrock(prompt: str, max_tokens: int = 1500, temperature: float = 0) -> str:
    """
    Sends a single-turn prompt to Claude Sonnet 4.6 via Bedrock and returns
    the raw text response. Raises on failure — callers handle retries.
    """
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }

    response = _client.invoke_model(
        modelId=_INFERENCE_PROFILE_ARN,
        body=json.dumps(body),
    )

    result = json.loads(response["body"].read())

    return result["content"][0]["text"].strip()

def call_claude_bedrock_with_tools(
    messages: list[dict],
    tools: list[dict],
    system: str = "",
    max_tokens: int = 1500,
    temperature: float = 0,
) -> dict:
    """
    Raw tool-use call to Claude Sonnet 4.6 via Bedrock. `messages` and `tools`
    must already be in Anthropic's native block format. Returns the parsed
    response body - callers extract the tool_use block themselves.
    """
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system,
        "messages": messages,
        "tools": tools,
        "tool_choice": {"type": "any"},  # force a tool call, never free text
    }

    response = _client.invoke_model(
        modelId=_INFERENCE_PROFILE_ARN,
        body=json.dumps(body),
    )
    return json.loads(response["body"].read())