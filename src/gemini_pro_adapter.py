# src/gemini_pro_adapter.py
"""
Implements agent_core.LLMClient for Gemini Pro via the REST API.
Since the existing Gemini Pro integration uses plain httpx REST calls
(not the SDK), we simulate tool-use by prompting the model to return
a structured JSON decision at each step, then routing it ourselves.
The agent_core loop stays unchanged — this adapter translates between
the generic ToolCall format and Gemini's JSON response.
"""
import json
import httpx
from agent_core import ToolSpec, ToolCall
from gemini_key_manager import build_gemini_pro_url


def _build_tool_decision_prompt(messages: list[dict], tools: list[ToolSpec]) -> str:
    """
    Converts the agent message history + available tools into a single
    Gemini prompt that asks the model to decide which tool to call next
    and with what arguments, returning a structured JSON response.
    """
    tools_block = "\n".join(
        f"- {t.name}: {t.description}\n  Parameters: {json.dumps(t.parameters)}"
        for t in tools
    )

    history_lines = []
    for m in messages:
        role = m["role"]
        if role == "system":
            history_lines.append(f"[SYSTEM]\n{m['content']}")
        elif role == "user":
            history_lines.append(f"[USER]\n{m['content']}")
        elif role == "assistant" and "tool_call" in m:
            call = m["tool_call"]
            history_lines.append(
                f"[ASSISTANT called tool: {call.name}]\n"
                f"Arguments: {json.dumps(call.arguments)}"
            )
        elif role == "tool_result":
            call = m["tool_call"]
            history_lines.append(
                f"[TOOL RESULT for {call.name}]\n{m['content']}"
            )

    history_block = "\n\n".join(history_lines)

    tool_names = [t.name for t in tools]

    return f"""{history_block}

---
Available tools:
{tools_block}

Based on the conversation above, decide which tool to call next and with what arguments.
You MUST call one of these tools: {tool_names}
Do NOT respond with free text — respond ONLY with a JSON object in this exact format, no markdown fences:
{{"tool_name": "<tool name>", "arguments": {{<arguments as JSON object>}}}}"""


class GeminiProLLMClient:
    """
    agent_core.LLMClient implementation backed by Gemini Pro REST API.
    Simulates tool-use via structured JSON output — the model decides
    which tool to call and with what arguments in each turn.
    """

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self._call_counter = 0

    def call_with_tools(self, messages: list[dict], tools: list[ToolSpec]) -> ToolCall:
        self._call_counter += 1
        prompt = _build_tool_decision_prompt(messages, tools)

        for attempt in range(1, self.max_retries + 1):
            try:
                response = httpx.post(
                    build_gemini_pro_url(),
                    json={
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {"temperature": 0},
                    },
                    timeout=90,
                )
                response.raise_for_status()
                data     = response.json()
                raw_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()

                # strip markdown fences if present
                if raw_text.startswith("```"):
                    raw_text = raw_text.strip("`")
                    if raw_text.lower().startswith("json"):
                        raw_text = raw_text[4:].strip()

                # find JSON object boundaries
                start = raw_text.find("{")
                end   = raw_text.rfind("}") + 1
                if start == -1 or end == 0:
                    raise ValueError(f"No JSON object found in response: {raw_text[:200]}")

                parsed    = json.loads(raw_text[start:end])
                tool_name = parsed.get("tool_name", "").strip()
                arguments = parsed.get("arguments", {})

                if not tool_name:
                    raise ValueError(f"Empty tool_name in response: {parsed}")

                return ToolCall(
                    name=tool_name,
                    arguments=arguments,
                    call_id=f"gemini-{self._call_counter}",
                )

            except httpx.HTTPStatusError as e:
                if e.response.status_code in (429, 503) and attempt < self.max_retries:
                    print(f"[WARN] Gemini Pro adapter {e.response.status_code}, retrying ({attempt}/{self.max_retries})...")
                    continue
                print(f"[ERROR] Gemini Pro adapter HTTP error: {e}")
                break
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                if attempt < self.max_retries:
                    print(f"[WARN] Gemini Pro adapter parse error ({e}), retrying ({attempt}/{self.max_retries})...")
                    continue
                print(f"[ERROR] Gemini Pro adapter failed to parse response: {e}")
                break
            except Exception as e:
                print(f"[ERROR] Gemini Pro adapter unexpected error: {e}")
                break

        # fallback — return unknown tool name so agent_core's recovery path handles it
        return ToolCall(name="__gemini_adapter_failed__", arguments={})