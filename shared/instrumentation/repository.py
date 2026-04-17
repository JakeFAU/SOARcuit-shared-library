from __future__ import annotations

from typing import Protocol, runtime_checkable

from .models import ActionRun, ActionStepMeasurement


@runtime_checkable
class InstrumentationRepository(Protocol):
    """
    Persistence Boundary for Telemetry.

    This protocol defines the contract for persisting high-level ActionRuns
    and low-level ActionStepMeasurements. It acts as the abstraction layer
    between the instrumentation logic and the database implementation.

    Implementations of this protocol (e.g., PostgresInstrumentationRepository)
    are responsible for handling the actual storage (e.g., using asyncpg or
    SQLAlchemy) without leaking DB specifics into the shared package.
    """

    async def record_action_run(self, run: ActionRun) -> None:
        """
        Persists a high-level action run episode.
        Should handle inserts or upserts if the run is updated (e.g., status changes).
        """
        ...

    async def record_step_measurement(self, measurement: ActionStepMeasurement) -> None:
        """
        Persists a single measured unit of work (ActionStep).
        Usually called after a Profiler context manager exits.
        """
        ...


class NoOpInstrumentationRepository:
    """Default implementation that performs no persistence (useful for local dev/tests)."""

    async def record_action_run(self, run: ActionRun) -> None:
        pass

    async def record_step_measurement(self, measurement: ActionStepMeasurement) -> None:
        pass
