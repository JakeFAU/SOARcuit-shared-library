"""Tests for the shared canonical domain models."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from shared.domain import (
    EMBEDDING_DIMS,
    Meme,
    MemeStatus,
    MemeTruthState,
    ObservationKind,
    RawObservation,
    ValidationError,
)


def make_embedding() -> list[float]:
    return [0.1] * EMBEDDING_DIMS


class FakeNdArray:
    def __init__(self, values: object) -> None:
        self._values = values

    def tolist(self) -> object:
        return self._values


def test_raw_observation_from_mapping_validates_and_normalizes() -> None:
    observation = RawObservation.from_mapping(
        {
            "fact": "  System load spiked  ",
            "probability": "0.75",
            "kind": "EXPLICIT_FACT",
            "dimension": " ops ",
            "evidence": " metrics ",
            "analyst": " control ",
            "metadata": {"source": "test"},
        }
    )

    assert observation.fact == "System load spiked"
    assert observation.probability == pytest.approx(0.75)
    assert observation.kind is ObservationKind.EXPLICIT_FACT
    assert observation.dimension == "ops"
    assert observation.evidence == "metrics"
    assert observation.analyst == "control"
    assert observation.metadata == {"source": "test"}


def test_meme_from_observation_populates_defaults() -> None:
    observation = RawObservation.from_mapping(
        {
            "fact": "Important operational fact",
            "probability": 0.6,
            "kind": "logical_inference",
            "dimension": "ops",
            "evidence": "playbook",
            "analyst": "agent-1",
        }
    )
    now = datetime(2026, 4, 11, 12, 0, tzinfo=UTC)

    meme = Meme.from_observation(
        observation,
        embedding=make_embedding(),
        expiration_days=30,
        default_importance=0.5,
        default_novelty=0.6,
        default_decay_rate=0.1,
        now=now,
    )

    assert meme.content == observation.fact
    assert meme.kind == ObservationKind.LOGICAL_INFERENCE.value
    assert meme.status is MemeStatus.ACTIVE
    assert meme.truth_state is MemeTruthState.UNVERIFIED
    assert meme.expires_at == datetime(2026, 5, 11, 12, 0, tzinfo=UTC)
    assert meme.metadata["evidence"] == "playbook"
    assert len(meme.embedding or []) == EMBEDDING_DIMS


def test_meme_from_mapping_converts_values() -> None:
    parent_id = uuid4()
    conversation_id = uuid4()
    message_id = uuid4()
    meme = Meme.from_mapping(
        {
            "id": str(uuid4()),
            "created_at": "2026-04-11T15:00:00+00:00",
            "updated_at": "1775916000",
            "content": "Canonical memory",
            "embedding": make_embedding(),
            "tags": ["ops", "ops", " signal "],
            "probability": "0.4",
            "kind": "explicit_fact",
            "dimension": "ops",
            "status": "active",
            "truth_state": "plausible",
            "importance": "0.2",
            "novelty": "0.8",
            "decay_rate": "0.05",
            "expires_at": "1775916200",
            "parent_meme_id": str(parent_id),
            "source_type": " chat ",
            "source_conversation_id": str(conversation_id),
            "source_message_id": str(message_id),
            "last_accessed_at": "1775916300",
            "access_count": "4",
            "metadata": {"source": "test"},
            "last_hygiene_at": "1775916400",
        }
    )

    assert isinstance(meme.id, UUID)
    assert meme.updated_at.tzinfo is not None
    assert meme.tags == ["ops", "signal"]
    assert meme.status is MemeStatus.ACTIVE
    assert meme.truth_state is MemeTruthState.PLAUSIBLE
    assert meme.parent_meme_id == parent_id
    assert meme.source_type == "chat"
    assert meme.access_count == 4
    assert meme.metadata == {"source": "test"}


def test_meme_from_mapping_accepts_numpy_like_embedding() -> None:
    meme = Meme.from_mapping(
        {
            "content": "Canonical memory",
            "embedding": FakeNdArray([0.1] * EMBEDDING_DIMS),
            "probability": 0.4,
            "kind": "explicit_fact",
            "dimension": "ops",
        }
    )

    assert meme.embedding == pytest.approx([0.1] * EMBEDDING_DIMS)


def test_meme_to_message_accepts_analysis_mapping() -> None:
    meme = Meme(
        content="Canonical memory",
        probability=0.7,
        kind="explicit_fact",
        dimension="ops",
        embedding=make_embedding(),
    )

    payload = meme.to_message(
        analysis={
            "recommended_action": "active-refine",
            "marginal_voi": 6.5,
        }
    )

    assert payload["message_type"] == "meme"
    assert payload["analysis"]["recommended_action"] == "active-refine"
    assert payload["analysis"]["marginal_voi"] == pytest.approx(6.5)
    assert payload["id"] == str(meme.id)


def test_meme_rejects_invalid_embedding_length() -> None:
    with pytest.raises(ValidationError, match="exactly 3072 values"):
        Meme.from_mapping(
            {
                "content": "bad embedding",
                "embedding": [0.1],
                "probability": 0.5,
                "kind": "explicit_fact",
                "dimension": "ops",
            }
        )
