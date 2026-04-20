"""Canonical shared models for raw observations and memes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from hashlib import sha256
from typing import Any
from uuid import UUID

from shared.domain.enums import MemeStatus, MemeTruthState, MessageKind, ObservationKind
from shared.domain.identifiers import generate_id, utc_now
from shared.domain.validation import (
    optional_datetime,
    optional_float,
    optional_text,
    optional_uuid,
    require_embedding,
    require_enum,
    require_float,
    require_int,
    require_mapping,
    require_tags,
    require_text,
    require_uuid,
)
from shared.errors import ValidationError

EMBEDDING_DIMS = 768


@dataclass(slots=True)
class RawObservation:
    """A raw analyst observation waiting to be normalized into a meme."""

    fact: str
    probability: float
    kind: ObservationKind
    dimension: str
    evidence: str
    analyst: str
    parent_meme_id: UUID | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.fact = require_text(self.fact, "fact")
        self.probability = require_float(self.probability, "probability", minimum=0.0, maximum=1.0)
        self.kind = require_enum(ObservationKind, self.kind, "kind")
        self.dimension = require_text(self.dimension, "dimension")
        self.evidence = require_text(self.evidence, "evidence")
        self.analyst = require_text(self.analyst, "analyst")
        self.parent_meme_id = optional_uuid(self.parent_meme_id, "parent_meme_id")
        self.metadata = require_mapping(self.metadata, "metadata")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> RawObservation:
        """Build a validated raw observation from an untrusted mapping."""

        return cls(
            fact=require_text(payload.get("fact"), "fact"),
            probability=require_float(
                payload.get("probability"), "probability", minimum=0.0, maximum=1.0
            ),
            kind=require_enum(ObservationKind, payload.get("kind"), "kind"),
            dimension=require_text(payload.get("dimension"), "dimension"),
            evidence=require_text(payload.get("evidence"), "evidence"),
            analyst=require_text(payload.get("analyst"), "analyst"),
            parent_meme_id=optional_uuid(payload.get("parent_meme_id"), "parent_meme_id"),
            metadata=require_mapping(payload.get("metadata"), "metadata"),
        )


@dataclass(slots=True)
class Meme:
    """Canonical meme representation shared across services."""

    content: str
    probability: float
    kind: str
    dimension: str
    id: UUID = field(default_factory=generate_id)
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    content_hash: str | None = None
    embedding: list[float] | None = None
    tags: list[str] = field(default_factory=list)
    status: MemeStatus = MemeStatus.ACTIVE
    truth_state: MemeTruthState = MemeTruthState.UNVERIFIED
    importance: float | None = None
    novelty: float | None = None
    decay_rate: float | None = None
    expires_at: datetime | None = None
    parent_meme_id: UUID | None = None
    source_type: str | None = None
    source_conversation_id: UUID | None = None
    source_message_id: UUID | None = None
    last_accessed_at: datetime | None = None
    access_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    last_hygiene_at: datetime | None = None

    def __post_init__(self) -> None:
        self.content = require_text(self.content, "content")
        self.probability = require_float(self.probability, "probability", minimum=0.0, maximum=1.0)
        self.kind = require_text(self.kind, "kind")
        self.dimension = require_text(self.dimension, "dimension")
        self.id = require_uuid(self.id, "id")
        self.created_at = optional_datetime(self.created_at, "created_at") or utc_now()
        self.updated_at = optional_datetime(self.updated_at, "updated_at") or self.created_at
        self.status = require_enum(MemeStatus, self.status, "status")
        self.truth_state = require_enum(MemeTruthState, self.truth_state, "truth_state")
        self.content_hash = self.content_hash or sha256(self.content.encode("utf-8")).hexdigest()
        self.embedding = require_embedding(self.embedding, dimensions=EMBEDDING_DIMS)
        self.tags = require_tags(self.tags)
        self.importance = optional_float(self.importance, "importance", minimum=0.0, maximum=1.0)
        self.novelty = optional_float(self.novelty, "novelty", minimum=0.0, maximum=1.0)
        self.decay_rate = optional_float(self.decay_rate, "decay_rate", minimum=0.0)
        self.expires_at = optional_datetime(self.expires_at, "expires_at")
        self.parent_meme_id = optional_uuid(self.parent_meme_id, "parent_meme_id")
        self.source_type = optional_text(self.source_type, "source_type")
        self.source_conversation_id = optional_uuid(
            self.source_conversation_id, "source_conversation_id"
        )
        self.source_message_id = optional_uuid(self.source_message_id, "source_message_id")
        self.last_accessed_at = optional_datetime(self.last_accessed_at, "last_accessed_at")
        self.access_count = require_int(self.access_count, "access_count", minimum=0)
        self.metadata = require_mapping(self.metadata, "metadata")
        self.last_hygiene_at = optional_datetime(self.last_hygiene_at, "last_hygiene_at")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> Meme:
        """Build a meme from an untrusted payload."""

        created_at = optional_datetime(payload.get("created_at"), "created_at") or utc_now()
        updated_at = optional_datetime(payload.get("updated_at"), "updated_at") or created_at
        return cls(
            id=require_uuid(payload["id"], "id") if "id" in payload else generate_id(),
            created_at=created_at,
            updated_at=updated_at,
            content=require_text(payload.get("content"), "content"),
            content_hash=optional_text(payload.get("content_hash"), "content_hash"),
            embedding=require_embedding(payload.get("embedding"), dimensions=EMBEDDING_DIMS),
            tags=require_tags(payload.get("tags")),
            probability=require_float(
                payload.get("probability"), "probability", minimum=0.0, maximum=1.0
            ),
            kind=require_text(payload.get("kind"), "kind"),
            dimension=require_text(payload.get("dimension"), "dimension"),
            status=require_enum(MemeStatus, payload.get("status", MemeStatus.ACTIVE), "status"),
            truth_state=require_enum(
                MemeTruthState,
                payload.get("truth_state", MemeTruthState.UNVERIFIED),
                "truth_state",
            ),
            importance=optional_float(
                payload.get("importance"), "importance", minimum=0.0, maximum=1.0
            ),
            novelty=optional_float(payload.get("novelty"), "novelty", minimum=0.0, maximum=1.0),
            decay_rate=optional_float(payload.get("decay_rate"), "decay_rate", minimum=0.0),
            expires_at=optional_datetime(payload.get("expires_at"), "expires_at"),
            parent_meme_id=optional_uuid(payload.get("parent_meme_id"), "parent_meme_id"),
            source_type=optional_text(payload.get("source_type"), "source_type"),
            source_conversation_id=optional_uuid(
                payload.get("source_conversation_id"), "source_conversation_id"
            ),
            source_message_id=optional_uuid(payload.get("source_message_id"), "source_message_id"),
            last_accessed_at=optional_datetime(payload.get("last_accessed_at"), "last_accessed_at"),
            access_count=require_int(payload.get("access_count", 0), "access_count", minimum=0),
            metadata=require_mapping(payload.get("metadata"), "metadata"),
            last_hygiene_at=optional_datetime(payload.get("last_hygiene_at"), "last_hygiene_at"),
        )

    @classmethod
    def from_observation(
        cls,
        observation: RawObservation,
        *,
        embedding: list[float],
        expiration_days: int,
        default_importance: float,
        default_novelty: float,
        default_decay_rate: float,
        now: datetime | None = None,
    ) -> Meme:
        """Convert an observation into a canonical meme."""

        created_at = now or utc_now()
        return cls(
            content=observation.fact,
            probability=observation.probability,
            kind=observation.kind.value,
            dimension=observation.dimension,
            id=generate_id(),
            created_at=created_at,
            updated_at=created_at,
            embedding=embedding,
            tags=[],
            status=MemeStatus.ACTIVE,
            truth_state=MemeTruthState.UNVERIFIED,
            importance=default_importance,
            novelty=default_novelty,
            decay_rate=default_decay_rate,
            expires_at=created_at + timedelta(days=expiration_days),
            parent_meme_id=observation.parent_meme_id,
            source_type=observation.analyst,
            metadata={
                "evidence": observation.evidence,
                "observation_metadata": observation.metadata,
            },
        )

    def update_content(self, new_content: str) -> None:
        """Update the meme content and its corresponding hash."""
        self.content = require_text(new_content, "content")
        self.content_hash = sha256(self.content.encode("utf-8")).hexdigest()
        self.updated_at = utc_now()

    def to_record(self) -> dict[str, Any]:
        """Return database-friendly values for the meme."""

        return {
            "id": self.id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "content": self.content,
            "content_hash": self.content_hash,
            "embedding": self.embedding,
            "tags": self.tags,
            "probability": self.probability,
            "kind": self.kind,
            "dimension": self.dimension,
            "status": self.status.value,
            "truth_state": self.truth_state.value,
            "importance": self.importance,
            "novelty": self.novelty,
            "decay_rate": self.decay_rate,
            "expires_at": self.expires_at,
            "parent_meme_id": self.parent_meme_id,
            "source_type": self.source_type,
            "source_conversation_id": self.source_conversation_id,
            "source_message_id": self.source_message_id,
            "last_accessed_at": self.last_accessed_at,
            "access_count": self.access_count,
            "metadata": self.metadata,
            "last_hygiene_at": self.last_hygiene_at,
        }

    def to_message(self, *, analysis: object | None = None) -> dict[str, Any]:
        """Return a JSON-safe message payload."""

        payload = {
            "message_type": MessageKind.MEME.value,
            "schema_version": 1,
            "id": str(self.id),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "content": self.content,
            "content_hash": self.content_hash,
            "embedding": self.embedding,
            "tags": self.tags,
            "probability": self.probability,
            "kind": self.kind,
            "dimension": self.dimension,
            "status": self.status.value,
            "truth_state": self.truth_state.value,
            "importance": self.importance,
            "novelty": self.novelty,
            "decay_rate": self.decay_rate,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "parent_meme_id": str(self.parent_meme_id) if self.parent_meme_id else None,
            "source_type": self.source_type,
            "source_conversation_id": (
                str(self.source_conversation_id) if self.source_conversation_id else None
            ),
            "source_message_id": (str(self.source_message_id) if self.source_message_id else None),
            "last_accessed_at": (
                self.last_accessed_at.isoformat() if self.last_accessed_at else None
            ),
            "access_count": self.access_count,
            "metadata": self.metadata,
            "last_hygiene_at": (self.last_hygiene_at.isoformat() if self.last_hygiene_at else None),
        }
        if analysis is not None:
            payload["analysis"] = _normalize_analysis_payload(analysis)
        return payload

    def to_audit_snapshot(self) -> dict[str, Any]:
        """Return a compact audit snapshot."""

        return {
            "id": str(self.id),
            "content": self.content,
            "probability": self.probability,
            "kind": self.kind,
            "dimension": self.dimension,
            "status": self.status.value,
            "truth_state": self.truth_state.value,
        }


def _normalize_analysis_payload(analysis: object) -> dict[str, Any]:
    if isinstance(analysis, Mapping):
        return {str(key): value for key, value in analysis.items()}

    to_dict = getattr(analysis, "to_dict", None)
    if callable(to_dict):
        mapping = to_dict()
        if isinstance(mapping, Mapping):
            return {str(key): value for key, value in mapping.items()}

    raise ValidationError("analysis must be a mapping or support to_dict().")
