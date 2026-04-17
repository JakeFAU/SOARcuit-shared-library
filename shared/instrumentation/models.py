from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from shared.domain.identifiers import generate_id, utc_now


class ActionStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class ActionRun:
    """Represents one high-level execution episode."""

    decision_id: UUID
    action_name: str
    id: UUID = field(default_factory=generate_id)
    action_run_id: UUID = field(default_factory=generate_id)
    parent_action_run_id: UUID | None = None
    meme_id: UUID | None = None
    source_event_id: str | None = None
    status: ActionStatus = ActionStatus.PENDING
    started_at: datetime = field(default_factory=utc_now)
    finished_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Ensure action_run_id defaults to id if not explicitly provided
        # (Though field default_factory already generates one)
        pass

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ActionStepMeasurement:
    """Represents one measured sub-step within an action run."""

    decision_id: UUID
    action_run_id: UUID
    step_name: str
    step_index: int
    id: UUID = field(default_factory=generate_id)
    parent_action_run_id: UUID | None = None
    meme_id: UUID | None = None
    source_event_id: str | None = None
    measured_at: datetime = field(default_factory=utc_now)

    # Telemetry fields
    wall_time_ms: float = 0.0
    queue_wait_ms: float = 0.0
    process_cpu_ms: float = 0.0
    thread_cpu_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    http_requests: int = 0
    db_queries: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    py_allocated_bytes: int = 0
    rss_delta_bytes: int = 0
    peak_rss_bytes: int = 0
    retries: int = 0
    succeeded: bool = True
    cancelled: bool = False
    timed_out: bool = False
    error_class: str | None = None
    estimated_usd_cost: float = 0.0
    estimated_compute_cost: float = 0.0
    estimated_contention_cost: float = 0.0
    estimated_total_cost: float = 0.0
    induced_action_count: int = 0
    induced_estimated_cost: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    # Parsed naming fields (populated if available)
    step_kind: str | None = None
    step_actor: str | None = None
    step_verb: str | None = None
    step_variant: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
