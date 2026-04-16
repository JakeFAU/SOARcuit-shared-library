import pytest
from unittest.mock import AsyncMock, MagicMock

from shared.llm.agent import Agent
from shared.llm.dispatcher import ToolDispatcher
from shared.llm.orchestrator import SessionOrchestrator
from shared.llm.types import AgentIntent, ToolRequest, ToolResult


@pytest.mark.anyio
async def test_orchestrator_run_success():
    """Verify that the orchestrator runs through multiple cycles and returns a final answer."""
    # Mock Agent
    agent = MagicMock(spec=Agent)
    agent.name = "test_agent"

    # Turn 1: call a tool
    intent1 = AgentIntent(
        thought="I need to call a tool.",
        plan="Call search_tool",
        actions=[ToolRequest(tool_name="search_tool", arguments={"query": "test"})],
    )
    # Turn 2: final answer
    intent2 = AgentIntent(
        thought="I have the result.",
        plan="Give final answer",
        final_answer="The result is 42",
    )

    agent.decide = AsyncMock(side_effect=[intent1, intent2])

    # Mock Dispatcher
    dispatcher = MagicMock(spec=ToolDispatcher)
    tool_result = ToolResult(tool_name="search_tool", success=True, output="Search result: 42")
    dispatcher.execute = AsyncMock(return_value=[tool_result])

    orchestrator = SessionOrchestrator(agent=agent, dispatcher=dispatcher, max_iterations=5)

    final_answer = await orchestrator.run("What is the result?")

    assert final_answer == "The result is 42"
    assert agent.decide.call_count == 2
    assert dispatcher.execute.call_count == 1


@pytest.mark.anyio
async def test_orchestrator_max_iterations():
    """Verify that the orchestrator stops after reaching max_iterations."""
    agent = MagicMock(spec=Agent)
    agent.name = "test_agent"

    # Always call a tool, never giving a final answer
    intent = AgentIntent(
        thought="I keep thinking.",
        plan="Call tool again",
        actions=[ToolRequest(tool_name="tool", arguments={})],
    )
    agent.decide = AsyncMock(return_value=intent)

    dispatcher = MagicMock(spec=ToolDispatcher)
    dispatcher.execute = AsyncMock(
        return_value=[ToolResult(tool_name="tool", success=True, output="result")]
    )

    orchestrator = SessionOrchestrator(agent=agent, dispatcher=dispatcher, max_iterations=3)

    final_answer = await orchestrator.run("Hello")

    assert final_answer == "I'm sorry, I reached the limit of my reasoning capacity for this request."
    assert agent.decide.call_count == 3


@pytest.mark.anyio
async def test_orchestrator_no_actions_or_final_answer():
    """Verify that the orchestrator breaks if the agent returns no actions and no final answer."""
    agent = MagicMock(spec=Agent)
    agent.name = "test_agent"

    # Agent returns no actions and no final answer (unexpected but possible)
    intent = AgentIntent(
        thought="I am stuck.",
        plan="Do nothing",
        actions=[],
        final_answer=None,
    )
    agent.decide = AsyncMock(return_value=intent)

    dispatcher = MagicMock(spec=ToolDispatcher)

    orchestrator = SessionOrchestrator(agent=agent, dispatcher=dispatcher, max_iterations=5)

    final_answer = await orchestrator.run("Help")

    # The loop breaks, and it falls through to the "failed_max_iterations" return
    # because it didn't return a final_answer within the loop.
    assert final_answer == "I'm sorry, I reached the limit of my reasoning capacity for this request."
    assert agent.decide.call_count == 1
    assert dispatcher.execute.call_count == 0
