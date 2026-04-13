"""High-level message parsing entry points."""

from shared.messaging.envelope import decode_pubsub_message_data
from shared.messaging.normalization import normalize_inbound_payload


def parse_pubsub_payload(message_data: bytes) -> dict[str, object]:
    """Decode Pub/Sub bytes into a canonical payload dict."""

    return normalize_inbound_payload(decode_pubsub_message_data(message_data))
