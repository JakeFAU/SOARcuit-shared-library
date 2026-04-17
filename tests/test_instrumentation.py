import pytest
import asyncio
from uuid import uuid4
from shared.instrumentation.naming import CanonicalName, InvalidCanonicalNameError
from shared.instrumentation.models import ActionStatus, ActionRun, ActionStepMeasurement
from shared.instrumentation.profiler import Profiler, measure_step
from shared.instrumentation.lifecycle import ActionRunManager

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
