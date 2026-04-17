import pytest
import json
import base64
from shared.messaging.envelope import (
    decode_pubsub_message_data,
    unwrap_pubsub_envelope,
)
from shared.messaging.errors import PayloadDecodeError

def test_decode_pubsub_message_data_json():
    data = json.dumps({"key": "value"}).encode("utf-8")
    decoded = decode_pubsub_message_data(data)
    assert decoded == {"key": "value"}

def test_decode_pubsub_message_data_base64():
    raw_json = json.dumps({"key": "value"})
    b64_data = base64.b64encode(raw_json.encode("utf-8")).decode("utf-8")
    decoded = decode_pubsub_message_data(b64_data.encode("utf-8"))
    assert decoded == {"key": "value"}

def test_decode_pubsub_message_data_invalid():
    with pytest.raises(PayloadDecodeError):
        decode_pubsub_message_data(b"\xff\xfe\xfd")

def test_unwrap_pubsub_envelope_push():
    envelope = {
        "message": {
            "data": base64.b64encode(json.dumps({"key": "value"}).encode("utf-8")).decode("utf-8"),
            "attributes": {"attr1": "val1"}
        }
    }
    unwrapped = unwrap_pubsub_envelope(envelope)
    assert unwrapped["key"] == "value"
    assert unwrapped["attributes"] == {"attr1": "val1"}

def test_unwrap_pubsub_envelope_wrapped():
    payload = {
        "payload": {"content": "test"},
        "message_type": "meme"
    }
    unwrapped = unwrap_pubsub_envelope(payload)
    assert unwrapped["content"] == "test"
    assert unwrapped["message_type"] == "meme"

def test_unwrap_pubsub_envelope_invalid_type():
    with pytest.raises(PayloadDecodeError):
        unwrap_pubsub_envelope(["not a dict"])

def test_unwrap_pubsub_envelope_missing_data():
    with pytest.raises(PayloadDecodeError):
        unwrap_pubsub_envelope({"message": {}})
