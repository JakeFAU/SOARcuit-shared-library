"""Tests for shared payload decoding, normalization, and classification."""

from __future__ import annotations

import base64
import json

import pytest
from shared.domain import MessageKind
from shared.messaging import (
    PayloadDecodeError,
    UnsupportedPayloadShapeError,
    detect_message_kind,
    expand_inbound_payloads,
    normalize_inbound_payload,
    parse_pubsub_payload,
)


def test_parse_pubsub_payload_accepts_direct_json() -> None:
    payload = parse_pubsub_payload(
        b'{"message_type":"meme","content":"x","probability":0.4,"kind":"explicit_fact","dimension":"ops"}'
    )

    assert payload["message_type"] == "meme"
    assert payload["content"] == "x"


def test_parse_pubsub_payload_unwraps_push_envelope() -> None:
    inner = base64.b64encode(
        b'{"fact":"x","probability":0.5,"kind":"explicit_fact","dimension":"ops","evidence":"metric","analyst":"agent"}'
    ).decode("utf-8")
    envelope = json.dumps(
        {"message": {"data": inner, "attributes": {"source": "push"}}}
    ).encode("utf-8")

    payload = parse_pubsub_payload(envelope)

    assert payload["fact"] == "x"
    assert payload["attributes"] == {"source": "push"}


def test_parse_pubsub_payload_normalizes_transcript_payload_with_defaults() -> None:
    payload = parse_pubsub_payload(b'{"conversation":"We missed sprint goals again."}')

    assert payload["message_type"] == "raw_observation"
    assert payload["fact"] == "We missed sprint goals again."
    assert payload["analyst"] == "unknown"
    assert payload["probability"] == pytest.approx(0.5)


def test_expand_inbound_payloads_preserves_batch_metadata() -> None:
    payloads = expand_inbound_payloads(
        {
            "analyst": "spock",
            "timestamp": "2026-04-12T12:00:00Z",
            "observations": [
                {
                    "fact": "First observation",
                    "probability": 0.6,
                    "kind": "logical_inference",
                    "dimension": "ops",
                    "evidence": "First quote.",
                },
                {
                    "fact": "Second observation",
                    "probability": 0.7,
                    "kind": "explicit_fact",
                    "dimension": "ops",
                    "evidence": "Second quote.",
                },
            ],
        }
    )

    assert len(payloads) == 2
    assert payloads[0]["metadata"]["batch_observation_index"] == 0
    assert payloads[1]["metadata"]["batch_observation_count"] == 2


def test_normalize_inbound_payload_preserves_unknown_fields_for_schema_drift() -> None:
    normalized = normalize_inbound_payload(
        {
            "message_type": "raw_observation",
            "fact": "Observation",
            "probability": 0.4,
            "kind": "explicit_fact",
            "dimension": "ops",
            "evidence": "metric",
            "analyst": "agent",
            "new_field": {"schema_version": 2},
        }
    )

    assert normalized["new_field"] == {"schema_version": 2}


def test_parse_pubsub_payload_rejects_non_utf8() -> None:
    with pytest.raises(PayloadDecodeError, match="valid UTF-8"):
        parse_pubsub_payload(b"\xff")


def test_detect_message_kind_rejects_unknown_shape() -> None:
    with pytest.raises(UnsupportedPayloadShapeError, match="Unable to infer"):
        detect_message_kind({"hello": "world"})


def test_detect_message_kind_normalizes_transcript_like_shape() -> None:
    assert (
        detect_message_kind({"conversation": "Jacob: We missed sprint goals again."})
        is MessageKind.RAW_OBSERVATION
    )
