import asyncio
import time
from typing import Any

import pytest
from pydantic import BaseModel, Field
from shared.llm.dispatcher import ToolDispatcher
from shared.llm.tools.base import BaseTool
from shared.llm.types import ToolRequest


class MockInput(BaseModel):
    seconds: float = Field(0.1)
    should_fail: bool = Field(False)


async def mock_tool_func(input_data: MockInput) -> str:
    if input_data.should_fail:
        raise ValueError("Intentional failure")
    await asyncio.sleep(input_data.seconds)
    return f"Slept for {input_data.seconds} seconds"


@pytest.fixture
def mock_tool() -> BaseTool:
    return BaseTool(
        name="sleep_tool",
        description="A tool that sleeps",
        function=mock_tool_func,
        input_model=MockInput,
    )


@pytest.fixture
def dispatcher(mock_tool: BaseTool) -> ToolDispatcher:
    return ToolDispatcher(tools=[mock_tool])


@pytest.mark.anyio
async def test_dispatcher_parallel_execution(
    dispatcher: ToolDispatcher, mock_tool: BaseTool
) -> None:
    """Verify that tools are executed in parallel."""
    requests = [
        ToolRequest(tool_name="sleep_tool", arguments={"seconds": 0.2}),
        ToolRequest(tool_name="sleep_tool", arguments={"seconds": 0.2}),
        ToolRequest(tool_name="sleep_tool", arguments={"seconds": 0.2}),
    ]

    start_time = time.perf_counter()
    results = await dispatcher.execute(requests)
    duration = time.perf_counter() - start_time

    assert len(results) == 3
    for r in results:
        assert r.success is True
        assert "Slept for 0.2 seconds" in r.output
        assert r.latency_ms > 0

    # If executed in parallel, total duration should be close to 0.2s, not 0.6s
    assert duration < 0.4  # Generous buffer for CI


@pytest.mark.anyio
async def test_dispatcher_tool_not_found(dispatcher: ToolDispatcher) -> None:
    """Verify error handling when a tool is not found."""
    requests = [ToolRequest(tool_name="unknown_tool", arguments={})]
    results = await dispatcher.execute(requests)

    assert len(results) == 1
    assert results[0].success is False
    assert "not found in registry" in results[0].error


@pytest.mark.anyio
async def test_dispatcher_execution_error(dispatcher: ToolDispatcher) -> None:
    """Verify error handling when a tool execution fails."""
    requests = [ToolRequest(tool_name="sleep_tool", arguments={"should_fail": True})]
    results = await dispatcher.execute(requests)

    assert len(results) == 1
    assert results[0].success is False
    assert "Intentional failure" in results[0].error
    assert results[0].latency_ms > 0


@pytest.mark.anyio
async def test_dispatcher_empty_requests(dispatcher: ToolDispatcher) -> None:
    """Verify that empty requests return empty results."""
    results = await dispatcher.execute([])
    assert results == []


@pytest.mark.anyio
async def test_dispatcher_mixed_results(
    dispatcher: ToolDispatcher, mock_tool: BaseTool
) -> None:
    """Verify handling of mixed success and failure."""
    requests = [
        ToolRequest(tool_name="sleep_tool", arguments={"seconds": 0.1}),
        ToolRequest(tool_name="unknown_tool", arguments={}),
        ToolRequest(tool_name="sleep_tool", arguments={"should_fail": True}),
    ]

    results = await dispatcher.execute(requests)

    assert len(results) == 3
    assert results[0].success is True
    assert results[1].success is False
    assert results[2].success is False
