import base64
import os
from unittest.mock import MagicMock, patch

import pytest
from cloudevents.http import CloudEvent
from pydantic import SecretStr, ValidationError
from shared.config.config import get_settings
from shared.config.config import LLMProvider as ProviderType
from shared.config.config import LLMSettings
from shared.domain.inference import clean_observations, configure_lm
from shared.domain.tracing import decode_pubsub_message


def test_configure_lm_initializes_gemini_chat_service_once() -> None:
    """Verify configure_lm caches a ChatService and warms the Gemini provider."""
    llm_settings = LLMSettings.model_construct(
        gemini_api_key=SecretStr("test-key"),
        default_provider=ProviderType.OPENAI,
        default_gemini_model="gemini-test-model",
    )
    app_settings = MagicMock(llm_settings=llm_settings)

    with patch("shared.domain.inference.get_settings", return_value=app_settings):
        with patch("shared.domain.inference.ChatService") as mock_chat_service:
            import shared.domain.inference

            shared.domain.inference._lm_initialized = False

            configure_lm()
            configure_lm()

            mock_chat_service.assert_called_once_with(llm_settings)
            mock_chat_service.return_value._get_provider.assert_called_once_with(
                ProviderType.GEMINI
            )


def test_configure_lm_missing_key() -> None:
    """Verify configure_lm raises ValidationError if API key is missing."""
    with patch.dict(os.environ, {}, clear=True):
        import shared.domain.inference

        get_settings.cache_clear()
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
