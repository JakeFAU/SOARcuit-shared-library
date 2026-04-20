"""Pydantic schemas for messaging."""

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class RawObservation(BaseModel):
    """A raw analyst observation waiting to be normalized into a meme."""

    model_config = ConfigDict(frozen=True)

    fact: str = Field(..., description="The core fact observed.")
    probability: float = Field(..., ge=0.0, le=1.0, description="Confidence score.")
    kind: Literal["explicit_fact", "logical_inference"] = Field(
        ..., description="The nature of the observation."
    )
    dimension: str = Field(..., description="The analytical dimension.")
    evidence: str = Field(..., description="Supporting evidence from the source.")
    analyst: str = Field(..., description="The analyst that produced the observation.")
    parent_meme_id: UUID | None = Field(default=None, description="The ID of the parent meme.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata.")
