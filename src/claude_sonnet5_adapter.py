# src/claude_sonnet5_adapter.py
"""
LLMClient adapter for Claude Sonnet 5 — same pattern as
claude_bedrock_adapter.py but uses bedrock_client_sonnet5.py.
"""
from agent_core import ToolSpec, ToolCall
from bedrock_client_sonnet5 import call_sonnet5_with_tools


def _tool_specs_to_anthropic(tools: list[ToolSpec]) -> list[dict]:
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.parameters,
        }
        for t in tools
    ]


def _generic_messages_to_anthropic(messages: list[dict]) -> list[dict]:
    anthropic_messages = []
    for m in messages:
        role = m["role"]
        if role == "system":
            continue
        elif role == "assistant" and "tool_call" in m:
            call: ToolCall = m["tool_call"]
            anthropic_messages.append({
                "role": "assistant",
                "content": [{
                    "type": "tool_use",
                    "id": call.call_id,
                    "name": call.name,
                    "input": call.arguments,
                }],
            })
        elif role == "tool_result":
            call: ToolCall = m["tool_call"]
            anthropic_messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": call.call_id,
                    "content": str(m["content"]),
                }],
            })
        else:
            anthropic_messages.append({"role": "user", "content": m["content"]})
    return anthropic_messages


class ClaudeSonnet5LLMClient:
    """agent_core.LLMClient backed by Claude Sonnet 5 on Bedrock."""

    def call_with_tools(self, messages: list[dict], tools: list[ToolSpec]) -> ToolCall:
        system_prompt       = next((m["content"] for m in messages if m["role"] == "system"), "")
        anthropic_messages  = _generic_messages_to_anthropic(messages)
        anthropic_tools     = _tool_specs_to_anthropic(tools)

        body = call_sonnet5_with_tools(
            messages=anthropic_messages,
            tools=anthropic_tools,
            system=system_prompt,
        )

        for block in body.get("content", []):
            if block.get("type") == "tool_use":
                return ToolCall(
                    name=block["name"],
                    arguments=block["input"],
                    call_id=block["id"],
                )

        return ToolCall(name="__no_tool_use_block__", arguments={})