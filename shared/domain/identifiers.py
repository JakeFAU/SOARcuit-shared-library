"""Shared identifier and timestamp helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid7  # type: ignore


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(UTC)


def generate_id() -> UUID:
    """Return a time-sortable UUID for new canonical entities."""

    return uuid7()
