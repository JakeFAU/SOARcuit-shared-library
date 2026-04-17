import pytest
from shared.messaging.validation import detect_message_kind
from shared.domain.enums import MessageKind
from shared.messaging.errors import (
    UnsupportedMessageTypeError,
    UnsupportedPayloadShapeError,
)

def test_detect_message_kind_raw():
    payload = {
        "fact": "test",
        "probability": 0.5,
        "kind": "explicit_fact",
        "dimension": "d",
        "evidence": "e",
        "analyst": "a"
    }
    assert detect_message_kind(payload) == MessageKind.RAW_OBSERVATION

def test_detect_message_kind_meme():
    payload = {
        "content": "test",
        "probability": 0.5,
        "kind": "k",
        "dimension": "d"
    }
    assert detect_message_kind(payload) == MessageKind.MEME

def test_detect_message_kind_explicit():
    assert detect_message_kind({"message_type": "raw"}) == MessageKind.RAW_OBSERVATION
    assert detect_message_kind({"message_type": "meme"}) == MessageKind.MEME

def test_detect_message_kind_unsupported_type():
    with pytest.raises(UnsupportedMessageTypeError):
        detect_message_kind({"message_type": "unsupported"})

def test_detect_message_kind_unsupported_shape():
    with pytest.raises(UnsupportedPayloadShapeError):
        detect_message_kind({"unknown": "shape"})
