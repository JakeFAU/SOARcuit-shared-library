import asyncio
import time

from opentelemetry.trace import Status, StatusCode

from shared.infrastructure.logging import get_logger
from shared.observability.tracer import get_tracer

from .tools.base import BaseTool
from .types import ToolRequest, ToolResult

logger = get_logger("dispatcher")
tracer = get_tracer("dispatcher")


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

        with tracer.start_as_current_span("dispatcher.batch_execute") as span:
            span.set_attribute("batch.size", len(requests))
            span.set_attribute("batch.tool_names", [r.tool_name for r in requests])

            logger.info("batch_execution_started", tool_count=len(requests))
            results = await asyncio.gather(*[self._run_single(req) for req in requests])

            success_count = sum(1 for r in results if r.success)
            span.set_attribute("batch.success_count", success_count)
            logger.info(
                "batch_execution_completed", success_count=success_count, total=len(requests)
            )

            return list(results)

    async def _run_single(self, req: ToolRequest) -> ToolResult:
        """Executes a single tool and captures metrics/errors with tracing."""
        with tracer.start_as_current_span(f"tool.{req.tool_name}.execute") as span:
            span.set_attribute("tool.name", req.tool_name)

            tool = self.tools.get(req.tool_name)
            if not tool:
                error_msg = f"Tool '{req.tool_name}' not found in registry."
                span.set_status(Status(StatusCode.ERROR, error_msg))
                logger.error("tool_not_found", tool_name=req.tool_name)
                return ToolResult(
                    tool_name=req.tool_name,
                    success=False,
                    error=error_msg,
                )

            start_time = time.perf_counter()
            try:
                # BaseTool.execute already has internal tracing for the function call
                output = await tool.execute(**req.arguments)
                latency = (time.perf_counter() - start_time) * 1000

                span.set_attribute("tool.latency_ms", latency)
                span.set_status(Status(StatusCode.OK))

                return ToolResult(
                    tool_name=tool.name, success=True, output=output, latency_ms=latency
                )
            except Exception as e:
                latency = (time.perf_counter() - start_time) * 1000
                error_str = str(e)

                span.set_status(Status(StatusCode.ERROR, error_str))
                span.record_exception(e)
                logger.exception("tool_execution_failed", tool_name=req.tool_name, error=error_str)

                return ToolResult(
                    tool_name=req.tool_name,
                    success=False,
                    error=error_str,
                    latency_ms=latency,
                )
