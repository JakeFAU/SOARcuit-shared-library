"""Shared payload classification helpers."""

from __future__ import annotations

from collections.abc import Mapping

from shared.domain.enums import MessageKind
from shared.messaging.errors import (
    UnsupportedMessageTypeError,
    UnsupportedPayloadShapeError,
)
from shared.messaging.normalization import (
    MEME_MESSAGE_TYPES,
    RAW_MESSAGE_TYPES,
    extract_explicit_kind,
    has_meme_shape,
    has_raw_shape,
    normalize_inbound_payload,
    unsupported_shape_message,
)


def detect_message_kind(payload: Mapping[str, object]) -> MessageKind:
    """Classify an inbound payload as a raw observation or meme."""

    normalized_payload = normalize_inbound_payload(payload)
    explicit_kind = extract_explicit_kind(normalized_payload)
    if isinstance(explicit_kind, str):
        normalized = explicit_kind.strip().lower()
        if normalized in RAW_MESSAGE_TYPES:
            return MessageKind.RAW_OBSERVATION
        if normalized in MEME_MESSAGE_TYPES:
            return MessageKind.MEME
        raise UnsupportedMessageTypeError(
            f"Unsupported message_type {explicit_kind!r}."
        )

    if has_raw_shape(normalized_payload):
        return MessageKind.RAW_OBSERVATION
    if has_meme_shape(normalized_payload):
        return MessageKind.MEME
    raise UnsupportedPayloadShapeError(unsupported_shape_message(normalized_payload))
