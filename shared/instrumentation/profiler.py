from __future__ import annotations

import asyncio
import os
import time
import tracemalloc
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, TypeVar
from uuid import UUID

import psutil

from shared.domain.identifiers import utc_now

from .costing import CostEstimator, NoOpCostEstimator
from .models import ActionStepMeasurement

T = TypeVar("T")


class Profiler:
    """Async-friendly step profiler that builds a measurement after exit."""

    def __init__(
        self,
        step_name: str,
        decision_id: UUID,
        action_run_id: UUID,
        *,
        meme_id: UUID | None = None,
        parent_step_cost_id: UUID | None = None,
        source_event_id: UUID | None = None,
        step_index: int = 0,
        queue_wait_ms: float = 0.0,
        actor_system: str | None = None,
        actor_id: str | None = None,
        provider: str | None = None,
        model_name: str | None = None,
        model_version: str | None = None,
        prompt_version: str | None = None,
        retry_count: int = 0,
        metadata: dict[str, Any] | None = None,
        measure_memory: bool = False,
        cost_estimator: CostEstimator | None = None,
    ):
        self.step_name = step_name
        self.decision_id = decision_id
        self.action_run_id = action_run_id
        self.meme_id = meme_id
        self.parent_step_cost_id = parent_step_cost_id
        self.source_event_id = source_event_id
        self.step_index = step_index
        self.queue_wait_ms = queue_wait_ms
        self.actor_system = actor_system
        self.actor_id = actor_id
        self.provider = provider
        self.model_name = model_name
        self.model_version = model_version
        self.prompt_version = prompt_version
        self.retry_count = retry_count
        self.metadata = metadata or {}
        self.measure_memory = measure_memory
        self._cost_estimator = cost_estimator or NoOpCostEstimator()

        self._start_wall: float = 0.0
        self._start_process_cpu: float = 0.0
        self._start_thread_cpu: float = 0.0
        self._start_rss: int = 0
        self._start_allocated_bytes: int = 0
        self._process = psutil.Process(os.getpid())

        self.started_at: datetime | None = None
        self.finished_at: datetime | None = None
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

        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.cached_input_tokens: int = 0
        self.total_tokens: int = 0
        self.http_requests: int = 0
        self.db_queries: int = 0
        self.bytes_sent: int = 0
        self.bytes_received: int = 0
        self.estimated_model_usd: float = 0.0
        self.estimated_network_usd: float = 0.0
        self.estimated_db_usd: float = 0.0
        self.estimated_compute_cost: float = 0.0
        self.estimated_contention_cost: float = 0.0
        self.estimated_total_cost: float = 0.0
        self.induced_action_count: int = 0
        self.induced_estimated_cost: float = 0.0

        self._measurement: ActionStepMeasurement | None = None

    async def __aenter__(self) -> Profiler:
        self.started_at = utc_now()
        self._start_wall = time.perf_counter()
        self._start_process_cpu = time.process_time()
        self._start_thread_cpu = time.thread_time()

        if self.measure_memory:
            self._start_rss = self._process.memory_info().rss
            if not tracemalloc.is_tracing():
                tracemalloc.start()
            self._start_allocated_bytes = tracemalloc.get_traced_memory()[0]

        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.finished_at = utc_now()

        end_wall = time.perf_counter()
        end_process_cpu = time.process_time()
        end_thread_cpu = time.thread_time()

        self.wall_time_ms = (end_wall - self._start_wall) * 1000
        self.process_cpu_ms = (end_process_cpu - self._start_process_cpu) * 1000
        self.thread_cpu_ms = (end_thread_cpu - self._start_thread_cpu) * 1000

        self.succeeded = exc_type is None
        self.cancelled = exc_type is not None and issubclass(exc_type, asyncio.CancelledError)
        self.timed_out = exc_type is not None and issubclass(exc_type, TimeoutError)
        self.error_class = exc_type.__name__ if exc_type else None

        if self.measure_memory:
            current, _peak = tracemalloc.get_traced_memory()
            self.py_allocated_bytes = max(0, current - self._start_allocated_bytes)
            end_rss = self._process.memory_info().rss
            self.rss_delta_bytes = end_rss - self._start_rss
            self.peak_rss_bytes = max(self._start_rss, end_rss)

        measurement = ActionStepMeasurement(
            decision_id=self.decision_id,
            action_run_id=self.action_run_id,
            meme_id=self.meme_id,
            parent_step_cost_id=self.parent_step_cost_id,
            source_event_id=self.source_event_id,
            step_name=self.step_name,
            step_index=self.step_index,
            measured_at=self.finished_at or utc_now(),
            started_at=self.started_at,
            finished_at=self.finished_at,
            actor_system=self.actor_system,
            actor_id=self.actor_id,
            provider=self.provider,
            model_name=self.model_name,
            model_version=self.model_version,
            prompt_version=self.prompt_version,
            succeeded=self.succeeded,
            cancelled=self.cancelled,
            timed_out=self.timed_out,
            error_class=self.error_class,
            retry_count=self.retry_count,
            wall_time_ms=self.wall_time_ms,
            queue_wait_ms=self.queue_wait_ms,
            process_cpu_ms=self.process_cpu_ms,
            thread_cpu_ms=self.thread_cpu_ms,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cached_input_tokens=self.cached_input_tokens,
            total_tokens=self.total_tokens,
            http_requests=self.http_requests,
            db_queries=self.db_queries,
            bytes_sent=self.bytes_sent,
            bytes_received=self.bytes_received,
            py_allocated_bytes=self.py_allocated_bytes,
            rss_delta_bytes=self.rss_delta_bytes,
            peak_rss_bytes=self.peak_rss_bytes,
            estimated_model_usd=self.estimated_model_usd,
            estimated_network_usd=self.estimated_network_usd,
            estimated_db_usd=self.estimated_db_usd,
            estimated_compute_cost=self.estimated_compute_cost,
            estimated_contention_cost=self.estimated_contention_cost,
            estimated_total_cost=self.estimated_total_cost,
            induced_action_count=self.induced_action_count,
            induced_estimated_cost=self.induced_estimated_cost,
            metadata=dict(self.metadata),
        )
        self._measurement = self._cost_estimator.estimate(measurement)

    @property
    def measurement(self) -> ActionStepMeasurement:
        if self._measurement is None:
            raise RuntimeError(
                "Profiler measurements are only available after the profiling context exits."
            )
        return self._measurement

    def get_measurement(self) -> ActionStepMeasurement:
        return self.measurement


@asynccontextmanager
async def measure_step(
    step_name: str,
    decision_id: UUID,
    action_run_id: UUID,
    **kwargs: Any,
) -> AsyncIterator[Profiler]:
    profiler = Profiler(
        step_name=step_name,
        decision_id=decision_id,
        action_run_id=action_run_id,
        **kwargs,
    )
    async with profiler:
        yield profiler
