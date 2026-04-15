import base64
import os
from unittest.mock import MagicMock, patch

import pytest
from cloudevents.http import CloudEvent
from pydantic import ValidationError

from shared.domain.inference import clean_observations, configure_lm
from shared.domain.tracing import decode_pubsub_message


def test_configure_lm_success() -> None:
    """Verify configure_lm initializes DSPy correctly."""
    with patch("dspy.LM") as mock_lm, patch("dspy.configure") as mock_configure:
        with patch.dict(
            os.environ,
            {"GOOGLE_GENAI_KEY": "test-key", "MODEL_NAME": "test-model"},
            clear=True,
        ):
            import shared.domain.inference

            shared.domain.inference._lm_initialized = False

            configure_lm()

            mock_lm.assert_called_once_with(model="test-model", api_key="test-key")
            mock_configure.assert_called_once()


def test_configure_lm_missing_key() -> None:
    """Verify configure_lm raises ValidationError if API key is missing."""
    with patch.dict(os.environ, {}, clear=True):
        import shared.domain.inference

        shared.domain.inference._lm_initialized = False

        with pytest.raises(ValidationError):
            configure_lm()


def test_clean_observations() -> None:
    """Verify clean_observations correctly clamps, normalizes and deduplicates."""
    raw_observations = [
        {
            "fact": "Fact 1",
            "probability": 1.5,
            "kind": "explicit_fact",
            "dimension": "leadership",
            "evidence": "Evidence 1",
        },
        {
            "fact": "Fact 1",  # Duplicate
            "probability": 0.8,
            "kind": "logical_inference",
            "dimension": "leadership",
            "evidence": "Evidence 2",
        },
        {
            "fact": "Fact 2",
            "probability": -0.5,
            "kind": "unknown_kind",
            "dimension": "unknown_dim",
            "evidence": "",
        },
        "not-a-dict",
    ]

    allowed_dimensions = {"leadership", "logic"}

    cleaned = clean_observations(
        raw_observations,
        analyst_name="test-analyst",
        allowed_dimensions=allowed_dimensions,
        default_dimension="logic",
    )

    assert len(cleaned) == 2

    # Check first observation
    assert cleaned[0]["fact"] == "Fact 1"
    assert cleaned[0]["probability"] == 1.0  # Clamped from 1.5
    assert cleaned[0]["kind"] == "explicit_fact"
    assert cleaned[0]["dimension"] == "leadership"
    assert cleaned[0]["evidence"] == "Evidence 1"
    assert cleaned[0]["analyst"] == "test-analyst"

    # Check second observation
    assert cleaned[1]["fact"] == "Fact 2"
    assert cleaned[1]["probability"] == 0.0  # Clamped from -0.5
    assert cleaned[1]["kind"] == "logical_inference"  # Defaulted from unknown_kind
    assert cleaned[1]["dimension"] == "logic"  # Defaulted from unknown_dim
    assert cleaned[1]["evidence"] == "Fact 2"  # Defaulted from empty evidence


def test_decode_pubsub_message_success() -> None:
    """Verify decode_pubsub_message correctly decodes base64 data."""
    test_data = "Hello, World!"
    encoded_data = base64.b64encode(test_data.encode("utf-8")).decode("utf-8")

    cloud_event = MagicMock(spec=CloudEvent)
    cloud_event.data = {"message": {"data": encoded_data}}

    decoded = decode_pubsub_message(cloud_event)
    assert decoded == test_data


def test_decode_pubsub_message_failure() -> None:
    """Verify decode_pubsub_message raises ValueError on invalid data."""
    cloud_event = MagicMock(spec=CloudEvent)
    cloud_event.data = {"not-a-message": {}}

    with pytest.raises(ValueError, match="No data found in Cloud Event message"):
        decode_pubsub_message(cloud_event)
