"""Consolidated tracing and logging helpers."""

import base64
from typing import Any

from cloudevents.http import CloudEvent
from opentelemetry import trace
from structlog import get_logger

# Pre-configured global tracer and logger instances
# Note: In Cloud Functions, you might want to call get_logger(analyst_name)
# to differentiate logs. These provide consistent formatting and instrumentation.

def initialize_tracing(service_name: str) -> trace.Tracer:
    """Get a tracer instance for the specified service name."""
    return trace.get_tracer(service_name)


def initialize_logger(service_name: str) -> Any:
    """Get a structlog logger for the specified service name."""
    return get_logger(service_name)


def decode_pubsub_message(cloud_event: CloudEvent) -> str:
    """Decode incoming Pub/Sub payload from a CloudEvent.

    Args:
        cloud_event: The CloudEvent from the Cloud Functions framework.

    Returns:
        The UTF-8 decoded message payload.

    Raises:
        ValueError: If no data or invalid data is found.
    """
    try:
        data = cloud_event.data.get("message", {}).get("data", "")
        if not data:
            raise ValueError("No data found in Cloud Event message")
        return base64.b64decode(data).decode("utf-8")
    except Exception as exc:
        raise ValueError(f"Failed to decode Pub/Sub message: {str(exc)}") from exc
