"""Shared message parsing and normalization helpers."""

from shared.messaging.errors import (
    PayloadDecodeError,
    UnsupportedMessageTypeError,
    UnsupportedPayloadShapeError,
)
from shared.messaging.normalization import expand_inbound_payloads, normalize_inbound_payload
from shared.messaging.parser import parse_pubsub_payload
from shared.messaging.pubsub import (
    build_outbound_observation,
    get_pubsub_client,
    publish_memes,
    publish_observations,
)
from shared.messaging.schemas import RawObservation
from shared.messaging.validation import detect_message_kind

__all__ = [
    "PayloadDecodeError",
    "RawObservation",
    "UnsupportedMessageTypeError",
    "UnsupportedPayloadShapeError",
    "build_outbound_observation",
    "detect_message_kind",
    "expand_inbound_payloads",
    "get_pubsub_client",
    "normalize_inbound_payload",
    "parse_pubsub_payload",
    "publish_memes",
    "publish_observations",
]
