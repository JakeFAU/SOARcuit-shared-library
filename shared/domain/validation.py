"""Shared validation and coercion helpers for canonical models."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, cast
from uuid import UUID

from shared.errors import ValidationError


def require_text(value: object, field_name: str) -> str:
    """Return a normalized non-empty string."""

    if not isinstance(value, str):
        raise ValidationError(f"{field_name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ValidationError(f"{field_name} must not be blank.")
    return normalized


def optional_text(value: object | None, field_name: str) -> str | None:
    """Return an optional normalized string."""

    if value is None:
        return None
    return require_text(value, field_name)


def require_float(
    value: object,
    field_name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Return a float with optional bounds."""

    try:
        converted = float(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{field_name} must be a float.") from exc
    if minimum is not None and converted < minimum:
        raise ValidationError(f"{field_name} must be >= {minimum}.")
    if maximum is not None and converted > maximum:
        raise ValidationError(f"{field_name} must be <= {maximum}.")
    return converted


def optional_float(
    value: object | None,
    field_name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    """Return an optional float with optional bounds."""

    if value is None:
        return None
    return require_float(value, field_name, minimum=minimum, maximum=maximum)


def require_int(value: object, field_name: str, *, minimum: int | None = None) -> int:
    """Return an int with an optional lower bound."""

    try:
        converted = int(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{field_name} must be an integer.") from exc
    if minimum is not None and converted < minimum:
        raise ValidationError(f"{field_name} must be >= {minimum}.")
    return converted


def require_uuid(value: object, field_name: str) -> UUID:
    """Return a UUID."""

    if isinstance(value, UUID):
        return value
    try:
        return UUID(str(value))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValidationError(f"{field_name} must be a UUID.") from exc


def optional_uuid(value: object | None, field_name: str) -> UUID | None:
    """Return an optional UUID."""

    if value is None:
        return None
    return require_uuid(value, field_name)


def require_datetime(value: object, field_name: str) -> datetime:
    """Return a timezone-aware datetime."""

    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=UTC)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValidationError(f"{field_name} must not be blank.")
        try:
            if stripped.replace(".", "", 1).isdigit():
                return datetime.fromtimestamp(float(stripped), tz=UTC)
            parsed = datetime.fromisoformat(stripped)
            return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
        except ValueError as exc:
            raise ValidationError(f"{field_name} must be a valid datetime.") from exc
    raise ValidationError(f"{field_name} must be a datetime-compatible value.")


def optional_datetime(value: object | None, field_name: str) -> datetime | None:
    """Return an optional datetime."""

    if value is None:
        return None
    return require_datetime(value, field_name)


def require_enum[EnumT: StrEnum](
    enum_cls: type[EnumT], value: object, field_name: str
) -> EnumT:
    """Return a validated string enum."""

    if isinstance(value, enum_cls):
        return value
    if not isinstance(value, str):
        raise ValidationError(
            f"{field_name} must be one of {[item.value for item in enum_cls]}."
        )
    try:
        return enum_cls(value.strip().lower())
    except ValueError as exc:
        raise ValidationError(
            f"{field_name} must be one of {[item.value for item in enum_cls]}."
        ) from exc


def require_mapping(value: object | None, field_name: str) -> dict[str, Any]:
    """Return a plain dict from any mapping-like object."""

    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    raise ValidationError(f"{field_name} must be an object.")


def require_tags(value: object | None) -> list[str]:
    """Return normalized unique tags."""

    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise ValidationError("tags must be a list of strings.")
    normalized: list[str] = []
    for raw_tag in value:
        tag = require_text(raw_tag, "tag")
        if tag not in normalized:
            normalized.append(tag)
    return normalized


def require_embedding(
    value: object | None,
    *,
    dimensions: int,
) -> list[float] | None:
    """Return an optional embedding vector with the expected dimensions."""

    if value is None:
        return None
    raw_embedding = _normalize_embedding_items(value)
    try:
        embedding = [float(cast(Any, item)) for item in raw_embedding]
    except (TypeError, ValueError) as exc:
        raise ValidationError(_embedding_type_error(value)) from exc
    if len(embedding) != dimensions:
        raise ValidationError(f"embedding must contain exactly {dimensions} values.")
    return embedding


def _normalize_embedding_items(value: object) -> list[Any]:
    items = _coerce_embedding_items(value)

    while len(items) == 1 and _is_embedding_array_like(items[0]):
        items = _coerce_embedding_items(items[0])

    if items and all(_is_embedding_array_like(item) for item in items):
        flattened_column: list[Any] = []
        for item in items:
            nested_items = _coerce_embedding_items(item)
            if len(nested_items) != 1:
                return items
            flattened_column.append(nested_items[0])
        return flattened_column

    return items


def _coerce_embedding_items(value: object) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        converted = tolist()
        if isinstance(converted, list):
            return converted
        if isinstance(converted, tuple):
            return list(converted)

    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise ValidationError(_embedding_type_error(value))

    try:
        return list(cast(Iterable[object], value))
    except TypeError as exc:
        raise ValidationError(_embedding_type_error(value)) from exc


def _is_embedding_array_like(value: object) -> bool:
    if isinstance(value, (list, tuple)):
        return True
    return callable(getattr(value, "tolist", None))


def _embedding_type_error(value: object) -> str:
    return f"embedding must be a 1D sequence of floats.\nnot {type(value).__name__}"
