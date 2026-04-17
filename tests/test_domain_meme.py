from uuid import UUID, uuid4

import pytest
from shared.domain.enums import MemeStatus, ObservationKind
from shared.domain.meme import EMBEDDING_DIMS, Meme, RawObservation
from shared.errors import ValidationError


@pytest.fixture
def valid_raw_observation_data():
    return {
        "fact": "The sky is blue.",
        "probability": 0.95,
        "kind": ObservationKind.EXPLICIT_FACT,
        "dimension": "natural_world",
        "evidence": "Visual observation",
        "analyst": "human_observer",
        "metadata": {"source": "direct"}
    }

def test_raw_observation_init(valid_raw_observation_data):
    obs = RawObservation(**valid_raw_observation_data)
    assert obs.fact == "The sky is blue."
    assert obs.probability == 0.95

def test_raw_observation_invalid_probability(valid_raw_observation_data):
    data = valid_raw_observation_data.copy()
    data["probability"] = 1.5
    with pytest.raises(ValidationError):
        RawObservation(**data)

def test_raw_observation_from_mapping(valid_raw_observation_data):
    obs = RawObservation.from_mapping(valid_raw_observation_data)
    assert obs.fact == "The sky is blue."

@pytest.fixture
def valid_meme_data():
    return {
        "content": "Intelligence is emergent.",
        "probability": 0.8,
        "kind": "philosophy",
        "dimension": "logic",
        "embedding": [0.1] * EMBEDDING_DIMS
    }

def test_meme_init(valid_meme_data):
    meme = Meme(**valid_meme_data)
    assert meme.content == "Intelligence is emergent."
    assert meme.status == MemeStatus.ACTIVE
    assert len(meme.embedding) == EMBEDDING_DIMS

def test_meme_invalid_embedding(valid_meme_data):
    data = valid_meme_data.copy()
    data["embedding"] = [0.1, 0.2] # Wrong dimensions
    with pytest.raises(ValidationError):
        Meme(**data)

def test_meme_from_mapping(valid_meme_data):
    data = valid_meme_data.copy()
    data["id"] = str(uuid4())
    meme = Meme.from_mapping(data)
    assert meme.content == "Intelligence is emergent."
    assert isinstance(meme.id, UUID)

def test_meme_from_observation(valid_raw_observation_data):
    obs = RawObservation(**valid_raw_observation_data)
    embedding = [0.0] * EMBEDDING_DIMS
    meme = Meme.from_observation(
        obs,
        embedding=embedding,
        expiration_days=30,
        default_importance=0.5,
        default_novelty=0.5,
        default_decay_rate=0.01
    )
    assert meme.content == obs.fact
    assert meme.source_type == obs.analyst
    assert meme.metadata["evidence"] == obs.evidence

def test_meme_to_record(valid_meme_data):
    meme = Meme(**valid_meme_data)
    record = meme.to_record()
    assert record["content"] == meme.content
    assert isinstance(record["id"], UUID)

def test_meme_to_message(valid_meme_data):
    meme = Meme(**valid_meme_data)
    message = meme.to_message()
    assert message["content"] == meme.content
    assert isinstance(message["id"], str)
    assert message["message_type"] == "meme"

def test_meme_to_audit_snapshot(valid_meme_data):
    meme = Meme(**valid_meme_data)
    snapshot = meme.to_audit_snapshot()
    assert snapshot["content"] == meme.content
    assert "embedding" not in snapshot

def test_normalize_analysis_payload():
    from shared.domain.meme import _normalize_analysis_payload
    
    # Mapping
    assert _normalize_analysis_payload({"a": 1}) == {"a": 1}
    
    # to_dict
    class MockAnalysis:
        def to_dict(self): return {"b": 2}
    assert _normalize_analysis_payload(MockAnalysis()) == {"b": 2}
    
    with pytest.raises(ValidationError):
        _normalize_analysis_payload("not valid")
