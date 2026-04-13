"""Tests for consolidated messaging logic."""

from typing import Any

import pytest
from pydantic import ValidationError
from shared.messaging.pubsub import build_outbound_observation, get_pubsub_client
from shared.messaging.schemas import RawObservation


def test_raw_observation_validation() -> None:
    """Verify that RawObservation correctly validates fields."""
    # Valid observation
    valid_data: dict[str, Any] = {
        "fact": "Leadership is decentralized.",
        "probability": 0.8,
        "kind": "logical_inference",
        "dimension": "delegation",
        "evidence": "Team members make autonomous decisions.",
        "analyst": "james",
        "metadata": {"source": "interview_4"},
    }
    obs = RawObservation(**valid_data)
    assert obs.fact == valid_data["fact"]
    assert obs.probability == 0.8
    assert obs.kind == "logical_inference"

    # Invalid probability (too high)
    invalid_data = valid_data.copy()
    invalid_data["probability"] = 1.5
    with pytest.raises(ValidationError):
        RawObservation(**invalid_data)

    # Invalid probability (too low)
    invalid_data = valid_data.copy()
    invalid_data["probability"] = -0.1
    with pytest.raises(ValidationError):
        RawObservation(**invalid_data)

    # Invalid kind
    invalid_data = valid_data.copy()
    invalid_data["kind"] = "not_a_kind"
    with pytest.raises(ValidationError):
        RawObservation(**invalid_data)

    # Missing field
    invalid_data = valid_data.copy()
    del invalid_data["fact"]
    with pytest.raises(ValidationError):
        RawObservation(**invalid_data)


def test_build_outbound_observation() -> None:
    """Verify that build_outbound_observation correctly maps fields and validates."""
    raw_obs: dict[str, Any] = {
        "fact": "Consistent reasoning observed.",
        "probability": 0.9,
        "kind": "explicit_fact",
        "dimension": "logical_consistency",
        "evidence": "Statement A followed directly from B.",
    }

    analyst = "spock"
    batch_timestamp = "2023-10-27T12:00:00Z"
    observation_count = 1
    observation_index = 0

    payload = build_outbound_observation(
        raw_obs,
        analyst=analyst,
        batch_timestamp=batch_timestamp,
        observation_count=observation_count,
        observation_index=observation_index,
    )

    assert payload["message_type"] == "raw_observation"
    assert payload["fact"] == raw_obs["fact"]
    assert payload["probability"] == 0.9
    assert payload["kind"] == "explicit_fact"
    assert payload["analyst"] == analyst
    assert payload["metadata"]["batch_timestamp"] == batch_timestamp
    assert payload["metadata"]["batch_observation_count"] == observation_count
    assert payload["metadata"]["batch_observation_index"] == observation_index
    assert payload["metadata"]["source_kind"] == "explicit_fact"


def test_build_outbound_observation_normalization() -> None:
    """Verify that build_outbound_observation handles dirty input kindly."""
    dirty_obs: dict[str, Any] = {
        "fact": "  Messy fact  ",
        "kind": "UNKNOWN_KIND",
        "probability": "0.7",  # String instead of float
    }

    payload = build_outbound_observation(
        dirty_obs,
        analyst="test-analyst",
        batch_timestamp="now",
        observation_count=1,
        observation_index=0,
    )

    assert payload["fact"] == "Messy fact"
    assert payload["kind"] == "logical_inference"  # Normalized
    assert payload["probability"] == 0.7  # Cast to float
    assert payload["analyst"] == "test-analyst"


def test_get_pubsub_client_singleton() -> None:
    """Verify that get_pubsub_client returns the same object instance."""
    client1 = get_pubsub_client()
    client2 = get_pubsub_client()
    assert client1 is client2
    assert id(client1) == id(client2)
