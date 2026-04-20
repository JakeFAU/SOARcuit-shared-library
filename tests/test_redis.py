"""Unit tests for the Redis hop tracker."""

from __future__ import annotations

import pytest
from fakeredis.aioredis import FakeRedis
from shared.config.config import RedisSettings
from shared.infrastructure.redis import RedisHopTracker
from unittest.mock import patch

@pytest.fixture
async def redis_settings():
    return RedisSettings(host="localhost", port=6379, hop_threshold=5)

@pytest.fixture
async def hop_tracker(redis_settings):
    tracker = RedisHopTracker(redis_settings)
    # Patch the _get_client to return a FakeRedis instance
    fake_redis = FakeRedis(decode_responses=True)
    
    @asynccontextmanager
    async def get_fake_client():
        yield fake_redis
        
    with patch.object(tracker, "_get_client", side_effect=get_fake_client):
        yield tracker
    
    await tracker.close()

from contextlib import asynccontextmanager

@pytest.mark.asyncio
async def test_increment_hop_basic():
    settings = RedisSettings(host="localhost", port=6379, hop_threshold=5)
    tracker = RedisHopTracker(settings)
    fake_redis = FakeRedis(decode_responses=True)
    
    @asynccontextmanager
    async def get_fake_client():
        yield fake_redis
        
    with patch.object(tracker, "_get_client", side_effect=get_fake_client):
        meme_id = "test-meme-123"
        
        # First increment
        count1 = await tracker.increment_hop(meme_id)
        assert count1 == 1
        
        # Second increment
        count2 = await tracker.increment_hop(meme_id)
        assert count2 == 2
        
        # Check TTL (fakeredis supports ttl)
        ttl = await fake_redis.ttl(f"meme:hop:{meme_id}")
        assert ttl > 0

@pytest.mark.asyncio
async def test_clear_hop():
    settings = RedisSettings(host="localhost", port=6379, hop_threshold=5)
    tracker = RedisHopTracker(settings)
    fake_redis = FakeRedis(decode_responses=True)
    
    @asynccontextmanager
    async def get_fake_client():
        yield fake_redis
        
    with patch.object(tracker, "_get_client", side_effect=get_fake_client):
        meme_id = "test-meme-456"
        await tracker.increment_hop(meme_id)
        assert await fake_redis.exists(f"meme:hop:{meme_id}")
        
        await tracker.clear_hop(meme_id)
        assert not await fake_redis.exists(f"meme:hop:{meme_id}")

@pytest.mark.asyncio
async def test_increment_hop_fail_open():
    settings = RedisSettings(host="localhost", port=6379, hop_threshold=5)
    tracker = RedisHopTracker(settings)
    
    # Simulate Redis failure
    with patch.object(tracker, "_get_client", side_effect=Exception("Redis down")):
        meme_id = "test-meme-789"
        count = await tracker.increment_hop(meme_id)
        # Should return 0 on error (fail-open)
        assert count == 0
