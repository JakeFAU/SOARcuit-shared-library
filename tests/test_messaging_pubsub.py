import pytest
import json
from unittest.mock import MagicMock, patch, Mock
from shared.messaging.pubsub import (
    get_pubsub_client,
    build_outbound_observation,
    publish_memes,
    publish_observations
)
from shared.domain.meme import Meme, EMBEDDING_DIMS

@pytest.fixture
def mock_pubsub_client():
    with patch("shared.messaging.pubsub.pubsub_v1.PublisherClient") as mock:
        # Reset global cache for each test
        import shared.messaging.pubsub
        shared.messaging.pubsub._pubsub_client = None
        yield mock

def test_get_pubsub_client(mock_pubsub_client):
    client = get_pubsub_client()
    assert client is not None
    mock_pubsub_client.assert_called_once()

def test_build_outbound_observation():
    obs = {
        "fact": "test",
        "kind": "explicit_fact",
        "dimension": "dim",
        "evidence": "ev",
        "metadata": {"custom": 1}
    }
    result = build_outbound_observation(
        obs, 
        analyst="test-analyst",
        batch_timestamp="2023-01-01T00:00:00Z",
        observation_count=1,
        observation_index=0
    )
    assert result["fact"] == "test"
    assert result["analyst"] == "test-analyst"
    assert result["message_type"] == "raw_observation"
    assert result["metadata"]["batch_observation_count"] == 1

def test_publish_memes(mock_pubsub_client):
    mock_instance = mock_pubsub_client.return_value
    mock_future = Mock()
    mock_future.result.return_value = "msg-1"
    mock_instance.publish.return_value = mock_future
    
    meme = Meme(
        content="test", 
        probability=0.5, 
        kind="k", 
        dimension="d", 
        embedding=[0.0]*EMBEDDING_DIMS
    )
    
    ids = publish_memes([meme], topic_path="topic", source="test-source")
    assert ids == ["msg-1"]
    assert mock_instance.publish.call_count == 1

def test_publish_observations(mock_pubsub_client):
    mock_instance = mock_pubsub_client.return_value
    mock_future = Mock()
    mock_future.result.return_value = "msg-2"
    mock_instance.publish.return_value = mock_future
    
    obs = [{"fact": "test", "kind": "explicit_fact", "dimension": "dim", "evidence": "ev"}]
    ids = publish_observations(obs, topic_path="topic", analyst="test-analyst")
    assert ids == ["msg-2"]
    assert mock_instance.publish.call_count == 1
