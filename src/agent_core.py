# src/agent_core.py
"""
Generic tool-loop agent scaffold. Model-agnostic: any LLM wrapper that
implements LLMClient.call_with_tools(...) can plug into run_tool_agent().
"""

from dataclasses import dataclass
from typing import Any, Callable, Protocol


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict          # JSON schema for the arguments this tool accepts
    handler: Callable[..., Any]  # the python function that actually executes it


@dataclass
class ToolCall:
    name: str
    arguments: dict
    call_id: str | None = None  # provider-specific id (e.g. Anthropic tool_use_id) - needed to round-trip tool results correctly


@dataclass
class AgentResult:
    status: str                # "success" | "max_steps_exceeded"
    payload: Any = None
    steps_used: int = 0


class LLMClient(Protocol):
    """Every model wrapper (Claude/Gemini/DeepSeek) must expose exactly this."""

    def call_with_tools(self, messages: list[dict], tools: list[ToolSpec]) -> ToolCall:
        """
        Sends messages + tool specs to the model in forced tool-use mode.
        Must ALWAYS return a ToolCall - never free text. The wrapper is
        responsible for enforcing that on the underlying API.
        """
        ...


def run_tool_agent(
    system_prompt: str,
    user_prompt: str,
    tools: list[ToolSpec],
    llm_client: LLMClient,
    final_tool_name: str,
    max_steps: int = 8,
) -> AgentResult:
    """
    Runs the tool-call loop until the model calls `final_tool_name`,
    or until `max_steps` is reached (the recursion limit) - whichever
    comes first. Never loops beyond max_steps, no exceptions.
    """
    tool_map = {t.name: t for t in tools}
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    for step in range(1, max_steps + 1):
        call = llm_client.call_with_tools(messages, tools)

        if call.name not in tool_map:
            # model named a tool that doesn't exist - correct it, don't crash
            messages.append({
                "role": "user",
                "content": f"Unknown tool '{call.name}'. Available: {list(tool_map)}",
            })
            continue

        tool = tool_map[call.name]
        result = tool.handler(**call.arguments)

        if call.name == final_tool_name:
            return AgentResult(status="success", payload=result, steps_used=step)

        messages.append({"role": "assistant", "tool_call": call})
        messages.append({"role": "tool_result", "tool_call": call, "content": result})

    # hit the recursion limit without a final answer - fail safe, don't spin forever
    return AgentResult(status="max_steps_exceeded", payload=None, steps_used=max_steps)