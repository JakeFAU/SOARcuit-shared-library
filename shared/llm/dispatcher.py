import asyncio
import time

from .tools.base import BaseTool
from .types import ToolRequest, ToolResult


class ToolDispatcher:
    """
    Execution engine for tool calls.
    Handles parallel execution, latency tracking, and error normalization.
    """

    def __init__(self, tools: list[BaseTool]):
        self.tools = {t.name: t for t in tools}

    async def execute(self, requests: list[ToolRequest]) -> list[ToolResult]:
        """Executes multiple tool calls in parallel using asyncio.gather."""
        if not requests:
            return []
        return list(await asyncio.gather(*[self._run_single(req) for req in requests]))

    async def _run_single(self, req: ToolRequest) -> ToolResult:
        """Executes a single tool and captures metrics/errors."""
        tool = self.tools.get(req.tool_name)
        if not tool:
            return ToolResult(
                tool_name=req.tool_name,
                success=False,
                error=f"Tool '{req.tool_name}' not found in registry.",
            )

        start_time = time.perf_counter()
        try:
            output = await tool.execute(**req.arguments)
            latency = (time.perf_counter() - start_time) * 1000

            # Note: Token usage is currently not captured by individual tools,
            # so we use a default TokenUsage for now (handled by ToolResult default).
            return ToolResult(tool_name=tool.name, success=True, output=output, latency_ms=latency)
        except Exception as e:
            latency = (time.perf_counter() - start_time) * 1000
            return ToolResult(
                tool_name=req.tool_name,
                success=False,
                error=str(e),
                latency_ms=latency,
            )
