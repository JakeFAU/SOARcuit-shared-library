"""Pub/Sub envelope decoding helpers."""

from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Mapping

from shared.messaging.errors import PayloadDecodeError


def decode_pubsub_message_data(message_data: bytes) -> dict[str, object]:
    """Decode Pub/Sub bytes into a normalized mapping before shape normalization."""

    payload = _parse_jsonish_bytes(message_data)
    return unwrap_pubsub_envelope(payload)


def unwrap_pubsub_envelope(payload: object) -> dict[str, object]:
    """Unwrap supported Pub/Sub or payload wrapper envelopes."""

    if not isinstance(payload, dict):
        raise PayloadDecodeError("Pub/Sub payload must decode to a JSON object.")

    if "message" in payload:
        envelope = payload["message"]
        if not isinstance(envelope, Mapping):
            raise PayloadDecodeError("Push envelope field 'message' must be an object.")
        encoded_data = envelope.get("data")
        if not isinstance(encoded_data, str):
            raise PayloadDecodeError("Push envelope is missing string field 'message.data'.")
        normalized = decode_pubsub_message_data(encoded_data.encode("utf-8"))
        attributes = envelope.get("attributes")
        if isinstance(attributes, Mapping):
            normalized["attributes"] = {str(key): value for key, value in attributes.items()}
        return normalized

    if (
        "payload" in payload
        and isinstance(payload["payload"], Mapping)
        and "message_type" in payload
    ):
        unwrapped = {str(key): value for key, value in payload["payload"].items()}
        unwrapped.setdefault("message_type", payload["message_type"])
        return unwrapped

    return {str(key): value for key, value in payload.items()}


def _parse_jsonish_bytes(message_data: bytes) -> object:
    try:
        decoded = message_data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PayloadDecodeError("Pub/Sub payload must be valid UTF-8.") from exc

    try:
        return json.loads(decoded)
    except json.JSONDecodeError:
        pass

    try:
        raw_json = base64.b64decode(decoded, validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError) as exc:
        raise PayloadDecodeError("Pub/Sub payload must be JSON or base64-encoded JSON.") from exc

    try:
        return json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise PayloadDecodeError("Base64 Pub/Sub payload did not contain valid JSON.") from exc
