from shared.llm.types import AgentIntent, TokenUsage, ToolRequest, ToolResult


def test_models() -> None:
    # Test ToolRequest
    tr = ToolRequest(tool_name="test_tool", arguments={"a": 1})
    print(f"ToolRequest: {tr}")

    # Test AgentIntent
    intent = AgentIntent(thought="thinking", plan="planning", actions=[tr], final_answer=None)
    print(f"AgentIntent: {intent}")

    # Test ToolResult
    res = ToolResult(
        tool_name="test_tool",
        success=True,
        output="ok",
        latency_ms=10.5,
        token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    print(f"ToolResult: {res}")


if __name__ == "__main__":
    try:
        test_models()
        print("All new models verified successfully!")
    except Exception as e:
        print(f"Verification failed: {e}")
        exit(1)
