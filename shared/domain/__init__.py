"""Canonical shared domain types."""

from typing import Any

from shared.domain.enums import (
    ActionTopic,
    MemeStatus,
    MemeTruthState,
    MessageKind,
    ObservationKind,
    VOI_COMPONENT_NAMES,
)
from shared.domain.identifiers import utc_now
from shared.domain.meme import EMBEDDING_DIMS, Meme, RawObservation
from shared.domain.tracing import decode_pubsub_message, initialize_logger, initialize_tracing
from shared.errors import ValidationError


def configure_lm(*args: Any, **kwargs: Any) -> None:
    from shared.domain.inference import configure_lm as _configure_lm

    _configure_lm(*args, **kwargs)


def clean_observations(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
    from shared.domain.inference import clean_observations as _clean_observations

    return _clean_observations(*args, **kwargs)


__all__ = [
    "ActionTopic",
    "EMBEDDING_DIMS",
    "Meme",
    "MemeStatus",
    "MemeTruthState",
    "MessageKind",
    "ObservationKind",
    "RawObservation",
    "ValidationError",
    "VOI_COMPONENT_NAMES",
    "clean_observations",
    "configure_lm",
    "decode_pubsub_message",
    "initialize_logger",
    "initialize_tracing",
    "utc_now",
]
