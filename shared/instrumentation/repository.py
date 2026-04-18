from __future__ import annotations

import json
import typing
from dataclasses import dataclass, field
from typing import Final, Protocol, runtime_checkable

from shared.infrastructure.logging import get_logger

if typing.TYPE_CHECKING:
    from asyncpg import Connection
    from asyncpg.pool import Pool

    from shared.config.config import DatabaseSettings

from .models import ActionRun, ActionStepMeasurement

logger = get_logger(__name__)

ACTION_RUN_COLUMNS: Final[tuple[str, ...]] = (
    "id",
    "decision_id",
    "parent_action_run_id",
    "triggered_by_action_run_id",
    "meme_id",
    "source_event_id",
    "action_name",
    "action_kind",
    "action_actor",
    "action_operation",
    "action_variant",
    "actor_system",
    "actor_id",
    "provider",
    "model_name",
    "model_version",
    "prompt_version",
    "status",
    "started_at",
    "finished_at",
    "confidence",
    "reason",
    "evidence",
    "metadata",
)
ACTION_RUN_COLUMN_LIST: Final[str] = ", ".join(ACTION_RUN_COLUMNS)
ACTION_RUN_PLACEHOLDERS: Final[str] = ", ".join(
    f"${index}" for index in range(1, len(ACTION_RUN_COLUMNS) + 1)
)
ACTION_RUN_SET_CLAUSE: Final[str] = ", ".join(
    f"{column} = EXCLUDED.{column}" for column in ACTION_RUN_COLUMNS if column != "id"
)
UPSERT_ACTION_RUN_SQL: Final[str] = f"""
INSERT INTO meme_action_runs ({ACTION_RUN_COLUMN_LIST})
VALUES ({ACTION_RUN_PLACEHOLDERS})
ON CONFLICT (id) DO UPDATE
SET
    {ACTION_RUN_SET_CLAUSE},
    updated_at = NOW()
""".strip()

STEP_MEASUREMENT_COLUMNS: Final[tuple[str, ...]] = (
    "id",
    "decision_id",
    "action_run_id",
    "parent_step_cost_id",
    "meme_id",
    "source_event_id",
    "step_index",
    "step_name",
    "step_kind",
    "step_actor",
    "step_operation",
    "step_variant",
    "measured_at",
    "started_at",
    "finished_at",
    "actor_system",
    "actor_id",
    "provider",
    "model_name",
    "model_version",
    "prompt_version",
    "succeeded",
    "cancelled",
    "timed_out",
    "error_class",
    "retry_count",
    "wall_time_ms",
    "queue_wait_ms",
    "process_cpu_ms",
    "thread_cpu_ms",
    "input_tokens",
    "output_tokens",
    "cached_input_tokens",
    "total_tokens",
    "http_requests",
    "db_queries",
    "bytes_sent",
    "bytes_received",
    "py_allocated_bytes",
    "rss_delta_bytes",
    "peak_rss_bytes",
    "estimated_model_usd",
    "estimated_network_usd",
    "estimated_db_usd",
    "estimated_compute_cost",
    "estimated_contention_cost",
    "estimated_total_cost",
    "induced_action_count",
    "induced_estimated_cost",
    "metadata",
)
STEP_MEASUREMENT_COLUMN_LIST: Final[str] = ", ".join(STEP_MEASUREMENT_COLUMNS)
STEP_MEASUREMENT_PLACEHOLDERS: Final[str] = ", ".join(
    f"${index}" for index in range(1, len(STEP_MEASUREMENT_COLUMNS) + 1)
)
STEP_MEASUREMENT_SET_CLAUSE: Final[str] = ", ".join(
    f"{column} = EXCLUDED.{column}" for column in STEP_MEASUREMENT_COLUMNS if column != "id"
)
UPSERT_STEP_MEASUREMENT_SQL: Final[str] = f"""
INSERT INTO meme_action_step_costs ({STEP_MEASUREMENT_COLUMN_LIST})
VALUES ({STEP_MEASUREMENT_PLACEHOLDERS})
ON CONFLICT (id) DO UPDATE
SET
    {STEP_MEASUREMENT_SET_CLAUSE}
""".strip()


@runtime_checkable
class InstrumentationRepository(Protocol):
    async def record_action_run(self, run: ActionRun) -> None: ...

    async def record_step_measurement(self, measurement: ActionStepMeasurement) -> None: ...


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


class AsyncpgInstrumentationRepository:
    """Repository that uses a shared asyncpg pool to persist instrumentation."""

    def __init__(
        self,
        config: DatabaseSettings,
    ) -> None:
        from shared.infrastructure.persistence import database_pool_manager

        self._config = config
        self._pool_manager = database_pool_manager

    async def record_action_run(
        self,
        run: ActionRun,
        *,
        db: Pool | Connection | None = None,
    ) -> None:
        from shared.infrastructure.persistence import acquire_connection

        record = run.to_record()
        values = tuple(
            json.dumps(record[column]) if column == "metadata" else record[column]
            for column in ACTION_RUN_COLUMNS
        )

        async with acquire_connection(db, config=self._config) as conn:
            await conn.execute(UPSERT_ACTION_RUN_SQL, *values)

    async def record_step_measurement(
        self,
        measurement: ActionStepMeasurement,
        *,
        db: Pool | Connection | None = None,
    ) -> None:
        from shared.infrastructure.persistence import acquire_connection

        record = measurement.to_record()
        values = tuple(
            json.dumps(record[column]) if column == "metadata" else record[column]
            for column in STEP_MEASUREMENT_COLUMNS
        )

        async with acquire_connection(db, config=self._config) as conn:
            await conn.execute(UPSERT_STEP_MEASUREMENT_SQL, *values)


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
