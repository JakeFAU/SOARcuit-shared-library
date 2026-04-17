from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from shared.domain.identifiers import generate_id, utc_now

from .naming import CanonicalName


class ActionStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class ActionRun:
    """Runtime and persistence-ready representation of a high-level action run."""

    decision_id: UUID
    action_name: str
    id: UUID = field(default_factory=generate_id)
    meme_id: UUID | None = None
    parent_action_run_id: UUID | None = None
    triggered_by_action_run_id: UUID | None = None
    source_event_id: UUID | None = None
    actor_system: str | None = None
    actor_id: str | None = None
    provider: str | None = None
    model_name: str | None = None
    model_version: str | None = None
    prompt_version: str | None = None
    status: ActionStatus = ActionStatus.PENDING
    started_at: datetime = field(default_factory=utc_now)
    finished_at: datetime | None = None
    confidence: float | None = None
    reason: str | None = None
    evidence: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    _canonical_name: CanonicalName = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._canonical_name = CanonicalName.parse(self.action_name)

    @property
    def action_run_id(self) -> UUID:
        return self.id

    @property
    def action_kind(self) -> str:
        return self._canonical_name.kind

    @property
    def action_actor(self) -> str:
        return self._canonical_name.actor

    @property
    def action_operation(self) -> str:
        return self._canonical_name.operation

    @property
    def action_variant(self) -> str | None:
        return self._canonical_name.variant

    def to_record(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "decision_id": self.decision_id,
            "parent_action_run_id": self.parent_action_run_id,
            "triggered_by_action_run_id": self.triggered_by_action_run_id,
            "meme_id": self.meme_id,
            "source_event_id": self.source_event_id,
            "action_name": self.action_name,
            "action_kind": self.action_kind,
            "action_actor": self.action_actor,
            "action_operation": self.action_operation,
            "action_variant": self.action_variant,
            "actor_system": self.actor_system,
            "actor_id": self.actor_id,
            "provider": self.provider,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "prompt_version": self.prompt_version,
            "status": self.status.value,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "confidence": self.confidence,
            "reason": self.reason,
            "evidence": self.evidence,
            "metadata": dict(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        record = self.to_record()
        record["action_run_id"] = self.action_run_id
        return record


@dataclass(slots=True)
class ActionStepMeasurement:
    """Runtime and persistence-ready representation of a measured action step."""

    decision_id: UUID
    action_run_id: UUID
    step_name: str
    step_index: int
    id: UUID = field(default_factory=generate_id)
    meme_id: UUID | None = None
    parent_step_cost_id: UUID | None = None
    source_event_id: UUID | None = None
    measured_at: datetime = field(default_factory=utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    actor_system: str | None = None
    actor_id: str | None = None
    provider: str | None = None
    model_name: str | None = None
    model_version: str | None = None
    prompt_version: str | None = None
    succeeded: bool = True
    cancelled: bool = False
    timed_out: bool = False
    error_class: str | None = None
    retry_count: int = 0
    wall_time_ms: float = 0.0
    queue_wait_ms: float = 0.0
    process_cpu_ms: float = 0.0
    thread_cpu_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    total_tokens: int = 0
    http_requests: int = 0
    db_queries: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    py_allocated_bytes: int = 0
    rss_delta_bytes: int = 0
    peak_rss_bytes: int = 0
    estimated_model_usd: float = 0.0
    estimated_network_usd: float = 0.0
    estimated_db_usd: float = 0.0
    estimated_compute_cost: float = 0.0
    estimated_contention_cost: float = 0.0
    estimated_total_cost: float = 0.0
    induced_action_count: int = 0
    induced_estimated_cost: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    _canonical_name: CanonicalName = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._canonical_name = CanonicalName.parse(self.step_name)

    @property
    def step_kind(self) -> str:
        return self._canonical_name.kind

    @property
    def step_actor(self) -> str:
        return self._canonical_name.actor

    @property
    def step_operation(self) -> str:
        return self._canonical_name.operation

    @property
    def step_variant(self) -> str | None:
        return self._canonical_name.variant

    def to_record(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "decision_id": self.decision_id,
            "action_run_id": self.action_run_id,
            "parent_step_cost_id": self.parent_step_cost_id,
            "meme_id": self.meme_id,
            "source_event_id": self.source_event_id,
            "step_index": self.step_index,
            "step_name": self.step_name,
            "step_kind": self.step_kind,
            "step_actor": self.step_actor,
            "step_operation": self.step_operation,
            "step_variant": self.step_variant,
            "measured_at": self.measured_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "actor_system": self.actor_system,
            "actor_id": self.actor_id,
            "provider": self.provider,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "prompt_version": self.prompt_version,
            "succeeded": self.succeeded,
            "cancelled": self.cancelled,
            "timed_out": self.timed_out,
            "error_class": self.error_class,
            "retry_count": self.retry_count,
            "wall_time_ms": self.wall_time_ms,
            "queue_wait_ms": self.queue_wait_ms,
            "process_cpu_ms": self.process_cpu_ms,
            "thread_cpu_ms": self.thread_cpu_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "total_tokens": self.total_tokens,
            "http_requests": self.http_requests,
            "db_queries": self.db_queries,
            "bytes_sent": self.bytes_sent,
            "bytes_received": self.bytes_received,
            "py_allocated_bytes": self.py_allocated_bytes,
            "rss_delta_bytes": self.rss_delta_bytes,
            "peak_rss_bytes": self.peak_rss_bytes,
            "estimated_model_usd": self.estimated_model_usd,
            "estimated_network_usd": self.estimated_network_usd,
            "estimated_db_usd": self.estimated_db_usd,
            "estimated_compute_cost": self.estimated_compute_cost,
            "estimated_contention_cost": self.estimated_contention_cost,
            "estimated_total_cost": self.estimated_total_cost,
            "induced_action_count": self.induced_action_count,
            "induced_estimated_cost": self.induced_estimated_cost,
            "metadata": dict(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        return self.to_record()
