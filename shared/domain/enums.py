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
    COMPLETED = "completed"
    EJECTED = "ejected"


class MemeTruthState(StrEnum):
    """Truth labels for a meme."""

    UNVERIFIED = "unverified"
    PLAUSIBLE = "plausible"
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    FALSE = "false"


class ActionTopic(StrEnum):
    """Downstream Pub/Sub topics representing actions selected by Thalamus.

    Each topic corresponds to an 'active' component that performs specific
    refinement or research tasks on memes.
    """

    ACTIVE_REFINE = "active-refine"
    ACTIVE_RESEARCH = "active-research"
    ACTIVE_RETAG = "active-retag"
    ACTIVE_TRUTH_DETECTION = "active-truth-detection"
    ACTIVE_NOOP = "active-noop"


VOI_COMPONENT_NAMES = (
    "content",
    "depth",
    "truth",
    "connection",
    "momentum",
    "longevity",
)
