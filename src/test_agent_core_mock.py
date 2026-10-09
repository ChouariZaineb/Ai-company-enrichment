# src/test_agent_core_mock.py
"""
Quick manual sanity check for run_tool_agent() - no real LLM calls.
Run with: python src/test_agent_core_mock.py
"""

from agent_core import ToolSpec, ToolCall, run_tool_agent


# --- fake tools -------------------------------------------------------

def fake_lookup(query: str):
    return f"3 fake candidates for '{query}'"

def fake_submit(topics: list):
    return {"submitted": topics}

TOOLS = [
    ToolSpec(
        name="lookup_candidates",
        description="fake taxonomy lookup",
        parameters={"query": "str"},
        handler=fake_lookup,
    ),
    ToolSpec(
        name="submit_topics",
        description="fake final submission",
        parameters={"topics": "list"},
        handler=fake_submit,
    ),
]


# --- fake LLM clients ---------------------------------------------------

class SucceedsAfterTwoSteps:
    """Simulates a model that looks up once, then submits."""
    def __init__(self):
        self.calls = 0

    def call_with_tools(self, messages, tools):
        self.calls += 1
        if self.calls == 1:
            return ToolCall(name="lookup_candidates", arguments={"query": "fintech"})
        return ToolCall(name="submit_topics", arguments={"topics": ["Digital Banking"]})


class NeverSubmits:
    """Simulates a model stuck in a loop - should hit the recursion limit."""
    def call_with_tools(self, messages, tools):
        return ToolCall(name="lookup_candidates", arguments={"query": "fintech"})


class HallucinatesToolName:
    """Simulates a model naming a tool that doesn't exist, then recovering."""
    def __init__(self):
        self.calls = 0

    def call_with_tools(self, messages, tools):
        self.calls += 1
        if self.calls == 1:
            return ToolCall(name="does_not_exist", arguments={})
        return ToolCall(name="submit_topics", arguments={"topics": ["Payments"]})


# --- run the three cases -------------------------------------------------

def run_case(label, client):
    result = run_tool_agent(
        system_prompt="You are a test agent.",
        user_prompt="Find topics.",
        tools=TOOLS,
        llm_client=client,
        final_tool_name="submit_topics",
        max_steps=5,
    )
    print(f"\n{label}")
    print(f"  status:     {result.status}")
    print(f"  steps_used: {result.steps_used}")
    print(f"  payload:    {result.payload}")


if __name__ == "__main__":
    run_case("Case 1: succeeds after 2 steps", SucceedsAfterTwoSteps())
    run_case("Case 2: never submits -> should hit recursion limit", NeverSubmits())
    run_case("Case 3: hallucinated tool name -> should recover", HallucinatesToolName())