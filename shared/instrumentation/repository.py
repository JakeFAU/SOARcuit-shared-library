from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from shared.infrastructure.logging import get_logger

from .models import ActionRun, ActionStepMeasurement

logger = get_logger(__name__)


@runtime_checkable
class InstrumentationRepository(Protocol):
    async def record_action_run(self, run: ActionRun) -> None:
        ...

    async def record_step_measurement(self, measurement: ActionStepMeasurement) -> None:
        ...


class NoOpInstrumentationRepository:
    async def record_action_run(self, run: ActionRun) -> None:
        del run

    async def record_step_measurement(self, measurement: ActionStepMeasurement) -> None:
        del measurement


@dataclass(slots=True)
class ListInstrumentationRepository:
    """In-memory sink for focused tests."""

    action_runs: list[ActionRun] = field(default_factory=list)
    step_measurements: list[ActionStepMeasurement] = field(default_factory=list)

    async def record_action_run(self, run: ActionRun) -> None:
        self.action_runs.append(run)

    async def record_step_measurement(self, measurement: ActionStepMeasurement) -> None:
        self.step_measurements.append(measurement)


class LoggingInstrumentationRepository:
    """Structured-log sink for services without persistence wiring."""

    async def record_action_run(self, run: ActionRun) -> None:
        logger.info("instrumentation_action_run", action_run=run.to_record())

    async def record_step_measurement(self, measurement: ActionStepMeasurement) -> None:
        logger.info(
            "instrumentation_step_measurement",
            step_measurement=measurement.to_record(),
        )


class MemeAwareInstrumentationRepository:
    """Forwards only records that are persistence-safe for the current schema."""

    def __init__(self, delegate: InstrumentationRepository):
        self._delegate = delegate

    async def record_action_run(self, run: ActionRun) -> None:
        if run.meme_id is None:
            logger.debug(
                "instrumentation_action_run_skipped_missing_meme_id",
                action_name=run.action_name,
                decision_id=str(run.decision_id),
                action_run_id=str(run.action_run_id),
            )
            return
        await self._delegate.record_action_run(run)

    async def record_step_measurement(self, measurement: ActionStepMeasurement) -> None:
        if measurement.meme_id is None:
            logger.debug(
                "instrumentation_step_skipped_missing_meme_id",
                step_name=measurement.step_name,
                decision_id=str(measurement.decision_id),
                action_run_id=str(measurement.action_run_id),
            )
            return
        await self._delegate.record_step_measurement(measurement)
