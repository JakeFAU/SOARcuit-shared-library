"""Stable enums shared across SOARcuit services."""

from enum import StrEnum


class MessageKind(StrEnum):
    """Supported inbound and outbound message kinds."""

    RAW_OBSERVATION = "raw_observation"
    MEME = "meme"


class ObservationKind(StrEnum):
    """Observation categories accepted from upstream producers."""

    EXPLICIT_FACT = "explicit_fact"
    LOGICAL_INFERENCE = "logical_inference"


class MemeStatus(StrEnum):
    """Lifecycle states for a stored meme."""

    ACTIVE = "active"
    STALE = "stale"
    DISPUTED = "disputed"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"
    DELETED = "deleted"


class MemeTruthState(StrEnum):
    """Truth labels for a meme."""

    UNVERIFIED = "unverified"
    PLAUSIBLE = "plausible"
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    FALSE = "false"
