import asyncio
from uuid import uuid4

import pytest

from shared.instrumentation.naming import CanonicalName, InvalidCanonicalNameError
from shared.instrumentation.models import ActionStatus, ActionRun, ActionStepMeasurement
from shared.instrumentation.profiler import Profiler, measure_step
from shared.instrumentation.lifecycle import ActionRunManager
from shared.instrumentation.repository import (
    ListInstrumentationRepository,
    MemeAwareInstrumentationRepository,
)

def test_canonical_name_parsing():
    valid = "tool:wikipedia:lookup:summary"
    cn = CanonicalName.parse(valid)
    assert cn.kind == "tool"
    assert cn.actor == "wikipedia"
    assert cn.verb == "lookup"
    assert cn.variant == "summary"

    with pytest.raises(InvalidCanonicalNameError):
        CanonicalName.parse("invalid_name")

@pytest.mark.anyio
async def test_profiler_produces_measurement():
    decision_id = uuid4()
    run_id = uuid4()
    
    async with measure_step("tool:test:exec", decision_id, run_id) as p:
        await asyncio.sleep(0.01)
        p.metadata["test"] = True

    m = p.get_measurement()
    assert isinstance(m, ActionStepMeasurement)
    assert m.step_name == "tool:test:exec"
    assert m.step_kind == "tool"
    assert m.wall_time_ms >= 10
    assert m.succeeded is True
    assert m.metadata["test"] is True

@pytest.mark.anyio
async def test_profiler_exception_handling():
    decision_id = uuid4()
    run_id = uuid4()
    
    try:
        async with measure_step("infra:api:call", decision_id, run_id) as p:
            raise ValueError("API Error")
    except ValueError:
        pass

    m = p.get_measurement()
    assert m.succeeded is False
    assert m.error_class == "ValueError"

def test_action_run_lifecycle():
    decision_id = uuid4()
    manager = ActionRunManager.start("agent:research:process", decision_id)
    assert manager.action_run.status == ActionStatus.RUNNING
    
    manager.update_metadata(key="val")
    run = manager.finish()
    
    assert run.status == ActionStatus.SUCCEEDED
    assert run.finished_at is not None
    assert run.metadata["key"] == "val"

def test_action_run_failure():
    decision_id = uuid4()
    manager = ActionRunManager.start("infra:pubsub:publish", decision_id)
    run = manager.fail(metadata={"error": "timeout"})
    
    assert run.status == ActionStatus.FAILED
    assert run.metadata["error"] == "timeout"


def test_action_run_to_record_uses_schema_field_names():
    decision_id = uuid4()
    meme_id = uuid4()

    run = ActionRun(
        decision_id=decision_id,
        meme_id=meme_id,
        action_name="agent:research:process",
        metadata={"source": "test"},
    )

    record = run.to_record()

    assert record["id"] == run.id
    assert record["decision_id"] == decision_id
    assert record["meme_id"] == meme_id
    assert record["action_name"] == "agent:research:process"
    assert record["action_kind"] == "agent"
    assert record["action_actor"] == "research"
    assert record["action_operation"] == "process"
    assert record["action_variant"] is None


def test_step_measurement_to_record_uses_schema_field_names():
    decision_id = uuid4()
    run_id = uuid4()
    meme_id = uuid4()

    measurement = ActionStepMeasurement(
        decision_id=decision_id,
        action_run_id=run_id,
        meme_id=meme_id,
        step_name="tool:wikipedia:lookup:summary",
        step_index=3,
    )

    record = measurement.to_record()

    assert record["decision_id"] == decision_id
    assert record["action_run_id"] == run_id
    assert record["meme_id"] == meme_id
    assert record["step_name"] == "tool:wikipedia:lookup:summary"
    assert record["step_kind"] == "tool"
    assert record["step_actor"] == "wikipedia"
    assert record["step_operation"] == "lookup"
    assert record["step_variant"] == "summary"


@pytest.mark.anyio
async def test_profiler_requires_context_exit_before_measurement():
    decision_id = uuid4()
    run_id = uuid4()

    async with measure_step("tool:test:exec", decision_id, run_id) as profiler:
        with pytest.raises(RuntimeError, match="after the profiling context exits"):
            profiler.get_measurement()


@pytest.mark.anyio
async def test_meme_aware_repository_skips_records_without_meme_id():
    sink = ListInstrumentationRepository()
    repository = MemeAwareInstrumentationRepository(sink)
    decision_id = uuid4()
    run_id = uuid4()

    await repository.record_action_run(
        ActionRun(
            decision_id=decision_id,
            meme_id=None,
            action_name="agent:hippocampus:process",
        )
    )
    await repository.record_step_measurement(
        ActionStepMeasurement(
            decision_id=decision_id,
            action_run_id=run_id,
            meme_id=None,
            step_name="agent:spock:process",
            step_index=0,
        )
    )
    persisted_run = ActionRun(
        decision_id=decision_id,
        meme_id=uuid4(),
        action_name="agent:research:process",
    )
    persisted_measurement = ActionStepMeasurement(
        decision_id=decision_id,
        action_run_id=run_id,
        meme_id=uuid4(),
        step_name="tool:wikipedia:lookup",
        step_index=1,
    )

    await repository.record_action_run(persisted_run)
    await repository.record_step_measurement(persisted_measurement)

    assert sink.action_runs == [persisted_run]
    assert sink.step_measurements == [persisted_measurement]
