from __future__ import annotations

import asyncio
import os
import time
import tracemalloc
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, TypeVar
from uuid import UUID

import psutil

from .models import ActionStepMeasurement
from .naming import CanonicalName

T = TypeVar("T")


class Profiler:
    """
    Async-friendly profiling helper for measuring action steps.
    """

    def __init__(
        self,
        step_name: str,
        decision_id: UUID,
        action_run_id: UUID,
        parent_action_run_id: UUID | None = None,
        meme_id: UUID | None = None,
        source_event_id: str | None = None,
        step_index: int = 0,
        queue_wait_ms: float = 0.0,
        metadata: dict[str, Any] | None = None,
        measure_memory: bool = False,
    ):
        self.step_name = step_name
        self.decision_id = decision_id
        self.action_run_id = action_run_id
        self.parent_action_run_id = parent_action_run_id
        self.meme_id = meme_id
        self.source_event_id = source_event_id
        self.step_index = step_index
        self.queue_wait_ms = queue_wait_ms
        self.metadata = metadata or {}
        self.measure_memory = measure_memory

        self._start_wall: float = 0.0
        self._start_process_cpu: float = 0.0
        self._start_thread_cpu: float = 0.0
        self._start_rss: int = 0
        self._process = psutil.Process(os.getpid())

        # Initialize metrics
        self.wall_time_ms: float = 0.0
        self.process_cpu_ms: float = 0.0
        self.thread_cpu_ms: float = 0.0
        self.py_allocated_bytes: int = 0
        self.rss_delta_bytes: int = 0
        self.peak_rss_bytes: int = 0
        self.succeeded: bool = False
        self.cancelled: bool = False
        self.timed_out: bool = False
        self.error_class: str | None = None

    async def __aenter__(self) -> Profiler:
        self._start_wall = time.perf_counter()
        self._start_process_cpu = time.process_time()
        self._start_thread_cpu = time.thread_time()

        if self.measure_memory:
            self._start_rss = self._process.memory_info().rss
            if not tracemalloc.is_tracing():
                tracemalloc.start()
            tracemalloc.clear_traces()

        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        end_wall = time.perf_counter()
        end_process_cpu = time.process_time()
        end_thread_cpu = time.thread_time()

        self.wall_time_ms = (end_wall - self._start_wall) * 1000
        self.process_cpu_ms = (end_process_cpu - self._start_process_cpu) * 1000
        self.thread_cpu_ms = (end_thread_cpu - self._start_thread_cpu) * 1000

        self.succeeded = exc_type is None
        self.error_class = exc_type.__name__ if exc_type else None
        self.cancelled = exc_type is asyncio.CancelledError
        self.timed_out = exc_type is asyncio.TimeoutError

        if self.measure_memory:
            current, peak = tracemalloc.get_traced_memory()
            self.py_allocated_bytes = current
            end_rss = self._process.memory_info().rss
            self.rss_delta_bytes = end_rss - self._start_rss
            self.peak_rss_bytes = self._process.memory_full_info().uss

    def get_measurement(self) -> ActionStepMeasurement:
        """Construct the ActionStepMeasurement from collected telemetry."""

        parsed_name = None
        try:
            parsed_name = CanonicalName.parse(self.step_name)
        except Exception:
            pass

        return ActionStepMeasurement(
            decision_id=self.decision_id,
            action_run_id=self.action_run_id,
            parent_action_run_id=self.parent_action_run_id,
            step_name=self.step_name,
            step_index=self.step_index,
            meme_id=self.meme_id,
            source_event_id=self.source_event_id,
            wall_time_ms=self.wall_time_ms,
            queue_wait_ms=self.queue_wait_ms,
            process_cpu_ms=self.process_cpu_ms,
            thread_cpu_ms=self.thread_cpu_ms,
            py_allocated_bytes=self.py_allocated_bytes,
            rss_delta_bytes=self.rss_delta_bytes,
            peak_rss_bytes=self.peak_rss_bytes,
            succeeded=self.succeeded,
            cancelled=self.cancelled,
            timed_out=self.timed_out,
            error_class=self.error_class,
            metadata=self.metadata,
            step_kind=parsed_name.kind if parsed_name else None,
            step_actor=parsed_name.actor if parsed_name else None,
            step_verb=parsed_name.verb if parsed_name else None,
            step_variant=parsed_name.variant if parsed_name else None,
        )


@asynccontextmanager
async def measure_step(
    step_name: str,
    decision_id: UUID,
    action_run_id: UUID,
    **kwargs: Any,
) -> AsyncIterator[Profiler]:
    """Convenience context manager for profiling a step."""
    profiler = Profiler(
        step_name=step_name, decision_id=decision_id, action_run_id=action_run_id, **kwargs
    )
    async with profiler:
        yield profiler
