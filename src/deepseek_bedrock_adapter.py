# src/deepseek_bedrock_adapter.py
"""
Implements agent_core.LLMClient for DeepSeek on Bedrock.
Translates the generic agent message history + ToolSpec list into
Bedrock's Converse API format, and the response back into a ToolCall.
"""
from agent_core import ToolSpec, ToolCall
from bedrock_deepseek_client import call_deepseek_bedrock_with_tools


def _tool_specs_to_converse(tools: list[ToolSpec]) -> list[dict]:
    return [
        {
            "toolSpec": {
                "name": t.name,
                "description": t.description,
                "inputSchema": {"json": t.parameters},
            }
        }
        for t in tools
    ]


def _generic_messages_to_converse(messages: list[dict]) -> list[dict]:
    """
    Converts generic agent messages to Bedrock Converse API format.
    Skips system messages (handled separately as system parameter).
    """
    converse_messages = []
    for m in messages:
        role = m["role"]

        if role == "system":
            continue

        elif role == "assistant" and "tool_call" in m:
            call: ToolCall = m["tool_call"]
            converse_messages.append({
                "role": "assistant",
                "content": [{
                    "toolUse": {
                        "toolUseId": call.call_id or "tool-0",
                        "name": call.name,
                        "input": call.arguments,
                    }
                }],
            })

        elif role == "tool_result":
            call: ToolCall = m["tool_call"]
            converse_messages.append({
                "role": "user",
                "content": [{
                    "toolResult": {
                        "toolUseId": call.call_id or "tool-0",
                        "content": [{"text": str(m["content"])}],
                    }
                }],
            })

        else:
            converse_messages.append({
                "role": "user",
                "content": [{"text": m["content"]}],
            })

    return converse_messages


class DeepSeekBedrockLLMClient:
    """agent_core.LLMClient implementation backed by DeepSeek on Bedrock."""

    def call_with_tools(self, messages: list[dict], tools: list[ToolSpec]) -> ToolCall:
        system_prompt = next(
            (m["content"] for m in messages if m["role"] == "system"), ""
        )
        converse_messages = _generic_messages_to_converse(messages)
        converse_tools    = _tool_specs_to_converse(tools)

        response = call_deepseek_bedrock_with_tools(
            messages=converse_messages,
            tools=converse_tools,
            system=system_prompt,
        )

        # Bedrock Converse response: response["output"]["message"]["content"]
        content = response.get("output", {}).get("message", {}).get("content", [])
        for block in content:
            if "toolUse" in block:
                tool_use = block["toolUse"]
                return ToolCall(
                    name=tool_use["name"],
                    arguments=tool_use["input"],
                    call_id=tool_use.get("toolUseId", "tool-0"),
                )

        return ToolCall(name="__no_tool_use_block__", arguments={})