# src/test_claude_bedrock_adapter_live.py
from agent_core import ToolSpec, run_tool_agent
from claude_bedrock_adapter import ClaudeBedrockLLMClient

def fake_finish(answer: str):
    return {"answer": answer}

TOOLS = [
    ToolSpec(
        name="finish",
        description="Call this with your final one-word answer.",
        parameters={"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]},
        handler=fake_finish,
    ),
]

result = run_tool_agent(
    system_prompt="You must call the finish tool with the answer.",
    user_prompt="What is the capital of France? Call finish with just the city name.",
    tools=TOOLS,
    llm_client=ClaudeBedrockLLMClient(),
    final_tool_name="finish",
    max_steps=3,
)
print(result)