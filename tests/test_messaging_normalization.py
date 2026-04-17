from shared.messaging.normalization import (
    expand_inbound_payloads,
    extract_explicit_kind,
    has_meme_shape,
    has_raw_shape,
    normalize_inbound_payload,
    unsupported_shape_message,
)


def test_has_raw_shape():
    valid_raw = {
        "fact": "test",
        "probability": 0.9,
        "kind": "explicit_fact",
        "dimension": "test",
        "evidence": "test",
        "analyst": "test"
    }
    assert has_raw_shape(valid_raw) is True
    assert has_raw_shape({"fact": "test"}) is False

def test_has_meme_shape():
    valid_meme = {
        "content": "test",
        "probability": 0.9,
        "kind": "explicit_fact",
        "dimension": "test"
    }
    assert has_meme_shape(valid_meme) is True
    assert has_meme_shape({"content": "test"}) is False

def test_extract_explicit_kind():
    assert extract_explicit_kind({"message_type": "test"}) == "test"
    assert extract_explicit_kind({"attributes": {"message_type": "test"}}) == "test"
    assert extract_explicit_kind({}) is None

def test_normalize_inbound_payload_raw():
    payload = {
        "fact": "test",
        "probability": 0.9,
        "kind": "explicit_fact",
        "dimension": "test",
        "evidence": "test",
        "analyst": "test"
    }
    normalized = normalize_inbound_payload(payload)
    assert normalized["fact"] == "test"
    assert normalized["message_type"] == "raw_observation"

def test_normalize_inbound_payload_meme():
    payload = {
        "content": "test",
        "probability": 0.8,
        "kind": "logical_inference",
        "dimension": "test",
        "embedding": "[0.1, 0.2]"
    }
    normalized = normalize_inbound_payload(payload)
    assert normalized["content"] == "test"
    assert normalized["message_type"] == "meme"
    assert normalized["embedding"] == [0.1, 0.2]

def test_normalize_inbound_payload_transcript():
    payload = {
        "messages": [
            {"speaker": "user", "content": "hello"},
            {"speaker": "analyst", "content": "hi"}
        ],
        "message_type": "raw"
    }
    normalized = normalize_inbound_payload(payload)
    assert normalized["message_type"] == "raw_observation"
    assert "user: hello\nanalyst: hi" in normalized["fact"]

def test_expand_inbound_payloads():
    payload = {
        "observations": [
            {"fact": "obs1", "probability": 0.5, "kind": "kind1", "dimension": "dim1", "evidence": "ev1", "analyst": "an1"},
            {"fact": "obs2", "probability": 0.6, "kind": "kind2", "dimension": "dim2", "evidence": "ev2", "analyst": "an2"}
        ]
    }
    expanded = expand_inbound_payloads(payload)
    assert len(expanded) == 2
    assert expanded[0]["fact"] == "obs1"
    assert expanded[1]["fact"] == "obs2"

def test_unsupported_shape_message():
    msg = unsupported_shape_message({"a": 1})
    assert "Unable to infer message type" in msg
    assert "Received keys: ['a']" in msg

def test_normalize_embedding_various_formats():
    from shared.messaging.normalization import _normalize_embedding_value
    
    # List
    assert _normalize_embedding_value([0.1, 0.2]) == [0.1, 0.2]
    
    # JSON String
    assert _normalize_embedding_value("[0.1, 0.2]") == [0.1, 0.2]
    
    # dict with 'embedding' key
    assert _normalize_embedding_value({"embedding": [0.1, 0.2]}) == [0.1, 0.2]
    
    # Nested array
    assert _normalize_embedding_value([[0.1, 0.2]]) == [0.1, 0.2]
    
    # String with 'array()'
    assert _normalize_embedding_value("array(0.1, 0.2)") == ["0.1", "0.2"]

def test_normalize_transcript_like_meme():
    payload = {
        "text": "some meme content",
        "message_type": "meme",
        "confidence": 0.95
    }
    normalized = normalize_inbound_payload(payload)
    assert normalized["message_type"] == "meme"
    assert normalized["content"] == "some meme content"
    assert normalized["probability"] == 0.95

def test_normalize_embedding_malformed():
    from shared.messaging.normalization import _normalize_embedding_value
    # Non-numeric tokens
    assert _normalize_embedding_value("0.1, abc") == "0.1, abc"
    # Mapping with numeric keys but non-numeric values
    assert _normalize_embedding_value({"0": "a"}) == "a"

def test_normalize_transcript_like_minimal():
    # Test path where _render_message_sequence returns None but text is present
    payload = {
        "body": "just some text",
        "message_type": "raw"
    }
    normalized = normalize_inbound_payload(payload)
    assert normalized["fact"] == "just some text"
    assert normalized["message_type"] == "raw_observation"

def test_normalize_transcript_like_with_observations():
    # Test path where observations list is present but has more than 1 item
    payload = {
        "observations": [
            {"fact": "1", "kind": "explicit_fact", "dimension": "d", "probability": 0.5, "evidence": "e", "analyst": "a"},
            {"fact": "2", "kind": "explicit_fact", "dimension": "d", "probability": 0.5, "evidence": "e", "analyst": "a"}
        ]
    }
    expanded = expand_inbound_payloads(payload)
    assert len(expanded) == 2
    assert expanded[0]["fact"] == "1"
    assert expanded[0]["message_type"] == "raw_observation"

def test_normalize_kind_aliases():
    payload = {
        "fact": "test",
        "probability": 0.9,
        "kind": "leadership_inference",
        "dimension": "test",
        "evidence": "test",
        "analyst": "test"
    }
    normalized = normalize_inbound_payload(payload)
    assert normalized["kind"] == "logical_inference"
    assert normalized["metadata"]["source_kind"] == "leadership_inference"
