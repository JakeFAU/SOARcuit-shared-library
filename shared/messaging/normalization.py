"""Inbound payload normalization into the canonical shared contract."""

from __future__ import annotations

import ast
import json
from collections.abc import Mapping

RAW_MESSAGE_TYPES = {"raw", "raw_observation", "raw-observation"}
MEME_MESSAGE_TYPES = {"meme"}
RAW_REQUIRED_KEYS = {"fact", "probability", "kind", "dimension", "evidence", "analyst"}
MEME_REQUIRED_KEYS = {"content", "probability", "kind", "dimension"}
TRANSCRIPT_LIKE_KEYS = {
    "conversation",
    "dialog",
    "dialogue",
    "messages",
    "transcript",
    "utterances",
}
TEXT_ALIAS_KEYS = ("fact", "content", "conversation", "transcript", "text", "body")
MESSAGE_SEQUENCE_KEYS = ("messages", "utterances")
ANALYST_ALIAS_KEYS = ("analyst", "agent", "author", "producer", "source", "speaker")
DIMENSION_ALIAS_KEYS = ("dimension", "category", "domain", "topic")
KIND_ALIAS_KEYS = ("kind", "type", "source_kind", "observation_kind")
PROBABILITY_ALIAS_KEYS = ("probability", "confidence", "score")
EVIDENCE_ALIAS_KEYS = ("evidence", "quote", "support", "justification", "excerpt")
EMBEDDING_CONTAINER_KEYS = ("embedding", "values", "vector", "data")
OBSERVATION_KIND_ALIASES = {
    "explicit_fact": "explicit_fact",
    "logical_inference": "logical_inference",
    "leadership_inference": "logical_inference",
}


def normalize_inbound_payload(payload: Mapping[str, object]) -> dict[str, object]:
    """Convert compatible inbound payload variants into the canonical contract."""

    normalized = {str(key): value for key, value in payload.items()}
    explicit_kind = extract_explicit_kind(normalized)
    if isinstance(explicit_kind, str) and "message_type" not in normalized:
        normalized["message_type"] = explicit_kind.strip().lower()

    if has_raw_shape(normalized):
        return _normalize_kind_aliases(normalized, preserve_source_kind=True)
    if has_meme_shape(normalized):
        return _normalize_meme_payload(normalized)

    single_observation = _normalize_single_observation_envelope(normalized)
    if single_observation is not None:
        return single_observation

    transcript_compat = _normalize_transcript_like_payload(normalized)
    if transcript_compat is not None:
        return transcript_compat

    return normalized


def expand_inbound_payloads(payload: Mapping[str, object]) -> list[dict[str, object]]:
    """Expand a compatible envelope into one or more canonical payloads."""

    normalized = {str(key): value for key, value in payload.items()}
    observations = normalized.get("observations")
    if not isinstance(observations, list):
        return [normalize_inbound_payload(normalized)]

    expanded: list[dict[str, object]] = []
    observation_count = len(observations)
    for index, raw_observation in enumerate(observations):
        if not isinstance(raw_observation, Mapping):
            continue
        expanded.append(
            normalize_inbound_payload(
                _build_observation_from_envelope(
                    normalized,
                    raw_observation,
                    observation_count=observation_count,
                    observation_index=index,
                )
            )
        )

    if expanded:
        return expanded
    return [normalize_inbound_payload(normalized)]


def extract_explicit_kind(payload: Mapping[str, object]) -> object | None:
    """Return the message type declared in the payload or attributes."""

    explicit_kind = payload.get("message_type")
    if explicit_kind is not None:
        return explicit_kind

    attributes = payload.get("attributes")
    if isinstance(attributes, Mapping):
        return attributes.get("message_type")
    return None


def has_raw_shape(payload: Mapping[str, object]) -> bool:
    """Return whether the payload already matches the raw observation contract."""

    return RAW_REQUIRED_KEYS.issubset(payload.keys())


def has_meme_shape(payload: Mapping[str, object]) -> bool:
    """Return whether the payload already matches the meme contract."""

    return MEME_REQUIRED_KEYS.issubset(payload.keys())


def unsupported_shape_message(payload: Mapping[str, object]) -> str:
    """Return a stable error message for unsupported payload shapes."""

    actual_keys = sorted(str(key) for key in payload.keys())
    return (
        "Unable to infer message type from payload keys. "
        f"Expected raw_observation keys {sorted(RAW_REQUIRED_KEYS)} or meme keys "
        f"{sorted(MEME_REQUIRED_KEYS)}. Received keys: {actual_keys}."
    )


def _looks_like_transcript_payload(payload: Mapping[str, object]) -> bool:
    return any(str(key).lower() in TRANSCRIPT_LIKE_KEYS for key in payload)


def _normalize_single_observation_envelope(
    payload: Mapping[str, object],
) -> dict[str, object] | None:
    observations = payload.get("observations")
    if not isinstance(observations, list) or len(observations) != 1:
        return None

    first = observations[0]
    if not isinstance(first, Mapping):
        return None
    return normalize_inbound_payload(
        _build_observation_from_envelope(
            payload,
            first,
            observation_count=1,
            observation_index=0,
        )
    )


def _build_observation_from_envelope(
    payload: Mapping[str, object],
    raw_observation: Mapping[str, object],
    *,
    observation_count: int,
    observation_index: int,
) -> dict[str, object]:
    observation = {str(key): value for key, value in raw_observation.items()}
    if "analyst" not in observation:
        analyst = _extract_text(payload, ANALYST_ALIAS_KEYS)
        if analyst is not None:
            observation["analyst"] = analyst

    metadata = _extract_metadata(observation)
    top_level_timestamp = payload.get("timestamp")
    if isinstance(top_level_timestamp, str) and top_level_timestamp.strip():
        metadata.setdefault("batch_timestamp", top_level_timestamp.strip())
    metadata.setdefault("batch_observation_count", observation_count)
    metadata.setdefault("batch_observation_index", observation_index)
    metadata["source_payload_keys"] = sorted(str(key) for key in payload.keys())
    observation["metadata"] = metadata
    return observation


def _normalize_kind_aliases(
    payload: Mapping[str, object], *, preserve_source_kind: bool = False
) -> dict[str, object]:
    normalized = {str(key): value for key, value in payload.items()}
    raw_kind = normalized.get("kind")
    canonical_kind = _canonicalize_observation_kind(raw_kind)
    if not isinstance(canonical_kind, str):
        return normalized

    normalized["kind"] = canonical_kind
    if preserve_source_kind and isinstance(raw_kind, str):
        source_kind = raw_kind.strip()
        if source_kind and source_kind != canonical_kind:
            metadata = _extract_metadata(normalized)
            metadata.setdefault("source_kind", source_kind)
            normalized["metadata"] = metadata
    return normalized


def _normalize_meme_payload(payload: Mapping[str, object]) -> dict[str, object]:
    normalized = _normalize_kind_aliases(payload)
    normalized["embedding"] = _normalize_embedding_value(normalized.get("embedding"))
    return normalized


def _canonicalize_observation_kind(value: object) -> object:
    if not isinstance(value, str):
        return value

    normalized = value.strip()
    if not normalized:
        return normalized
    return OBSERVATION_KIND_ALIASES.get(normalized.lower(), normalized)


def _normalize_embedding_value(value: object) -> object:
    if isinstance(value, Mapping):
        normalized_mapping = {str(key): item for key, item in value.items()}
        for key in EMBEDDING_CONTAINER_KEYS:
            if key not in normalized_mapping:
                continue
            return _normalize_embedding_value(normalized_mapping[key])
        if normalized_mapping and all(_is_int_like(key) for key in normalized_mapping):
            ordered_values = [
                item
                for _, item in sorted(
                    normalized_mapping.items(), key=lambda entry: int(entry[0])
                )
            ]
            return _normalize_embedding_value(ordered_values)
        return value

    if isinstance(value, (list, tuple)):
        items = list(value)
        if len(items) == 1 and isinstance(items[0], (Mapping, list, tuple, str)):
            return _normalize_embedding_value(items[0])
        return [
            _normalize_embedding_value(item)
            if isinstance(item, (Mapping, list, tuple))
            else item
            for item in items
        ]

    if not isinstance(value, str):
        return value

    stripped = value.strip()
    if not stripped:
        return value

    parsed = _parse_embedding_serialization(stripped)
    if parsed is not None:
        return _normalize_embedding_value(parsed)

    tokenized = _tokenize_embedding_string(stripped)
    if tokenized is not None:
        return tokenized

    return value


def _parse_embedding_serialization(value: str) -> object | None:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, (list, tuple, Mapping)):
        return parsed

    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return None
    if isinstance(parsed, (list, tuple, Mapping)):
        return parsed
    return None


def _tokenize_embedding_string(value: str) -> list[str] | None:
    candidate = value.strip()
    if candidate.lower().startswith("array(") and candidate.endswith(")"):
        candidate = candidate[6:-1].strip()

    if len(candidate) >= 2 and candidate[0] in "[{(" and candidate[-1] in "]})":
        candidate = candidate[1:-1].strip()

    if not candidate:
        return []

    flattened = candidate.replace("\n", " ").replace("\t", " ").replace(";", ",")
    raw_tokens: list[str] = []
    if "," in flattened:
        for chunk in flattened.split(","):
            raw_tokens.extend(chunk.split())
    else:
        raw_tokens = flattened.split()

    tokens = [_strip_wrapping_quotes(token) for token in raw_tokens if token.strip()]
    if not tokens or not all(_is_float_like(token) for token in tokens):
        return None
    return tokens


def _strip_wrapping_quotes(value: str) -> str:
    normalized = value.strip()
    if (
        len(normalized) >= 2
        and normalized[0] == normalized[-1]
        and normalized[0] in {"'", '"'}
    ):
        return normalized[1:-1].strip()
    return normalized


def _is_float_like(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def _is_int_like(value: str) -> bool:
    try:
        int(value)
    except ValueError:
        return False
    return True


def _normalize_transcript_like_payload(
    payload: Mapping[str, object],
) -> dict[str, object] | None:
    if (
        not _looks_like_transcript_payload(payload)
        and _extract_text(payload, TEXT_ALIAS_KEYS) is None
    ):
        return None

    fact = _extract_text(payload, TEXT_ALIAS_KEYS)
    if fact is None:
        fact = _render_message_sequence(payload)
    if fact is None:
        return None

    source_kind = _extract_text(payload, KIND_ALIAS_KEYS) or "logical_inference"
    normalized_kind = (
        "explicit_fact"
        if source_kind.strip().lower() == "explicit_fact"
        else "logical_inference"
    )
    evidence = _extract_text(payload, EVIDENCE_ALIAS_KEYS) or fact
    analyst = _extract_text(payload, ANALYST_ALIAS_KEYS) or "unknown"
    dimension = _extract_text(payload, DIMENSION_ALIAS_KEYS) or "conversation"
    probability = _extract_float(payload, PROBABILITY_ALIAS_KEYS, default=0.5)
    explicit_kind = extract_explicit_kind(payload)
    explicit_kind_normalized = (
        explicit_kind.strip().lower() if isinstance(explicit_kind, str) else None
    )

    metadata = _extract_metadata(payload)
    metadata.update(
        {
            "source_format": "transcript_compat",
            "source_kind": source_kind,
            "source_payload_keys": sorted(str(key) for key in payload.keys()),
        }
    )
    if explicit_kind_normalized in MEME_MESSAGE_TYPES:
        metadata.setdefault("analyst", analyst)
        metadata.setdefault("evidence", evidence)
        return {
            "message_type": "meme",
            "content": fact,
            "probability": probability,
            "kind": source_kind,
            "dimension": dimension,
            "metadata": metadata,
        }

    return {
        "message_type": "raw_observation",
        "fact": fact,
        "probability": probability,
        "kind": normalized_kind,
        "dimension": dimension,
        "evidence": evidence,
        "analyst": analyst,
        "metadata": metadata,
    }


def _render_message_sequence(payload: Mapping[str, object]) -> str | None:
    for key in MESSAGE_SEQUENCE_KEYS:
        messages = payload.get(key)
        if not isinstance(messages, list):
            continue
        rendered: list[str] = []
        for item in messages:
            if isinstance(item, str):
                line = item.strip()
                if line:
                    rendered.append(line)
                continue
            if not isinstance(item, Mapping):
                continue
            speaker = _extract_text(item, ("speaker", "role", "author", "name"))
            content = _extract_text(item, ("content", "text", "message", "body"))
            if content is None:
                continue
            rendered.append(f"{speaker}: {content}" if speaker else content)
        if rendered:
            return "\n".join(rendered)
    return None


def _extract_metadata(payload: Mapping[str, object]) -> dict[str, object]:
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        return {str(key): value for key, value in metadata.items()}
    return {}


def _extract_text(payload: Mapping[str, object], keys: tuple[str, ...]) -> str | None:
    contexts: list[Mapping[str, object]] = [payload]
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        contexts.append(metadata)
    attributes = payload.get("attributes")
    if isinstance(attributes, Mapping):
        contexts.append(attributes)

    for context in contexts:
        for key in keys:
            value = context.get(key)
            if isinstance(value, str):
                normalized = value.strip()
                if normalized:
                    return normalized
    return None


def _extract_float(
    payload: Mapping[str, object], keys: tuple[str, ...], *, default: float
) -> float:
    contexts: list[Mapping[str, object]] = [payload]
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        contexts.append(metadata)
    attributes = payload.get("attributes")
    if isinstance(attributes, Mapping):
        contexts.append(attributes)

    for context in contexts:
        for key in keys:
            value = context.get(key)
            try:
                if isinstance(value, (int, float, str)):
                    return max(0.0, min(1.0, float(value)))
            except (TypeError, ValueError):
                continue
    return default
