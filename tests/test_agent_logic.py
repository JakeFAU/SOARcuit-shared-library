from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from shared.llm.agent import Agent
from shared.llm.tools.base import BaseTool
from shared.llm.types import AgentIntent, ChatMessage, Role, ToolRequest


class MockInput(BaseModel):
    query: str


async def mock_tool_fn(validated_input: MockInput) -> str:
    return "result"


@pytest.mark.anyio
async def test_agent_build_system_prompt():
    llm_service = MagicMock()
    tool = BaseTool(
        name="test_tool",
        description="A test tool",
        function=mock_tool_fn,
        input_model=MockInput,
    )
    agent = Agent(
        name="test_agent",
        base_instructions="You are a test agent.",
        llm_service=llm_service,
        tools=[tool],
    )

    prompt = agent._build_system_prompt()
    assert "You are a test agent." in prompt
    assert "test_tool" in prompt
    assert "A test tool" in prompt
    assert "## OPERATIONAL PROTOCOL" in prompt
    assert "## AVAILABLE TOOLS" in prompt
    assert "## CONSTRAINTS" in prompt


@pytest.mark.anyio
async def test_agent_decide():
    llm_service = MagicMock()
    expected_intent = AgentIntent(
        thought="I should use a tool.",
        plan="Call test_tool",
        actions=[ToolRequest(tool_name="test_tool", arguments={"query": "test"})],
    )
    llm_service.chat_structured = AsyncMock(return_value=expected_intent)

    agent = Agent(
        name="test_agent",
        base_instructions="You are a test agent.",
        llm_service=llm_service,
        model="gpt-4o",
    )

    history = [ChatMessage(role=Role.USER, content="Hello")]
    intent = await agent.decide(history)

    assert intent == expected_intent
    llm_service.chat_structured.assert_called_once()
    _, kwargs = llm_service.chat_structured.call_args
    assert kwargs["response_model"] == AgentIntent
    assert kwargs["model"] == "gpt-4o"
    messages = kwargs["messages"]
    assert messages[0].role == Role.SYSTEM
    assert "You are a test agent." in messages[0].content
    assert messages[1] == history[0]
