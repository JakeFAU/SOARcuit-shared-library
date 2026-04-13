"""Pub/Sub messaging helpers for SOARcuit."""

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from google.cloud import pubsub_v1
from opentelemetry import trace
from structlog import get_logger

from shared.messaging.schemas import RawObservation

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)

# Global cache for cold-start optimization
_pubsub_client: pubsub_v1.PublisherClient | None = None


def get_pubsub_client() -> pubsub_v1.PublisherClient:
    """Return a cached Pub/Sub publisher client with instant batch settings."""
    global _pubsub_client
    if _pubsub_client is None:
        _pubsub_client = pubsub_v1.PublisherClient(
            batch_settings=pubsub_v1.types.BatchSettings(max_messages=1)
        )
    return _pubsub_client


def build_outbound_observation(
    observation: Mapping[str, Any],
    *,
    analyst: str,
    batch_timestamp: str,
    observation_count: int,
    observation_index: int,
) -> dict[str, Any]:
    """Convert an analyst observation into the Thalamus raw_observation contract."""
    source_kind = str(observation.get("kind", "")).strip()
    normalized_kind = "explicit_fact" if source_kind == "explicit_fact" else "logical_inference"
    raw_metadata = observation.get("metadata")
    metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
    metadata.update(
        {
            "source_kind": source_kind or normalized_kind,
            "batch_timestamp": batch_timestamp,
            "batch_observation_count": observation_count,
            "batch_observation_index": observation_index,
        }
    )

    # Use Pydantic to validate and prepare the base observation
    # We allow extra fields in the input but only keep what's in the schema + message_type
    obs_data = {
        "fact": str(observation.get("fact", "")).strip(),
        "probability": observation.get("probability", 0.5),
        "kind": normalized_kind,
        "dimension": str(observation.get("dimension", "")).strip(),
        "evidence": str(observation.get("evidence", "")).strip(),
        "analyst": analyst,
        "metadata": metadata,
    }

    # Validate using Pydantic
    raw_obs = RawObservation(**obs_data)

    payload = raw_obs.model_dump()
    payload["message_type"] = "raw_observation"
    return payload


def publish_observations(
    observations: list[Mapping[str, Any]],
    *,
    topic_path: str,
    analyst: str,
    timestamp: str | None = None,
) -> list[str]:
    """Publish results to the downstream topic with tracing."""
    if not observations:
        logger.info("No observations to publish")
        return []

    batch_timestamp = timestamp or datetime.now(UTC).isoformat()
    outbound_messages = [
        build_outbound_observation(
            observation,
            analyst=analyst,
            batch_timestamp=batch_timestamp,
            observation_count=len(observations),
            observation_index=index,
        )
        for index, observation in enumerate(observations)
        if isinstance(observation, Mapping)
    ]

    if not outbound_messages:
        logger.info("No valid observations to publish")
        return []

    with tracer.start_as_current_span("pubsub_publish"):
        client = get_pubsub_client()
        futures = []
        for index, payload in enumerate(outbound_messages):
            futures.append(
                client.publish(
                    topic_path,
                    json.dumps(payload).encode("utf-8"),
                    analyst=analyst,
                    observation_count=str(len(outbound_messages)),
                    observation_index=str(index),
                    message_type="raw_observation",
                    source_kind=str(payload["metadata"]["source_kind"]),
                )
            )
        message_ids = [future.result(timeout=10) for future in futures]

    logger.info(
        "Published observations",
        topic=topic_path,
        count=len(message_ids),
        message_ids=message_ids,
    )
    return message_ids
