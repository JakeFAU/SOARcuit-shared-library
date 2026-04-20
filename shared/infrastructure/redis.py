"""Redis infrastructure for hop tracking and circuit breaking."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import redis.asyncio as redis
import structlog
from opentelemetry import trace
from opentelemetry.trace import SpanKind
from shared.config.config import RedisSettings
from shared.observability.tracer import set_span_attributes

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)

# Key prefix for hop counters
HOP_KEY_PREFIX = "meme:hop:"
DEFAULT_TTL = 86400  # 24 hours


class RedisHopTracker:
    """Manages meme hop counts using Redis."""

    def __init__(self, settings: RedisSettings):
        self._settings = settings
        self._pool = redis.ConnectionPool(
            host=settings.host,
            port=settings.port,
            decode_responses=True,
        )

    @asynccontextmanager
    async def _get_client(self) -> AsyncIterator[redis.Redis]:
        client = redis.Redis(connection_pool=self._pool)
        try:
            yield client
        finally:
            await client.close()

    async def increment_hop(self, meme_id: str) -> int:
        """
        Increment the hop count for a meme.

        Uses an atomic INCR. If the result is 1, sets a 24-hour TTL.
        """
        key = f"{HOP_KEY_PREFIX}{meme_id}"

        with tracer.start_as_current_span(
            "redis.increment_hop",
            kind=SpanKind.CLIENT,
            attributes={
                "db.system": "redis",
                "db.operation": "incr",
                "db.redis.key": key,
                "meme.id": meme_id,
            },
        ) as span:
            try:
                async with self._get_client() as client:
                    # Atomic increment
                    count = int(await client.incr(key))

                    # If this is the first hop, set expiration
                    if count == 1:
                        await client.expire(key, DEFAULT_TTL)

                    set_span_attributes(span, {"meme.hop_count": count})
                    return count
            except Exception as e:
                logger.error("redis_hop_increment_failed", error=str(e), meme_id=meme_id)
                # Fail-open: return 0 to allow message to proceed but signal error
                # Thalamus will check this and increment an error metric
                return 0

    async def clear_hop(self, meme_id: str) -> None:
        """Purge the hop counter for a meme."""
        key = f"{HOP_KEY_PREFIX}{meme_id}"

        with tracer.start_as_current_span(
            "redis.clear_hop",
            kind=SpanKind.CLIENT,
            attributes={
                "db.system": "redis",
                "db.operation": "del",
                "db.redis.key": key,
                "meme.id": meme_id,
            },
        ):
            try:
                async with self._get_client() as client:
                    await client.delete(key)
            except Exception as e:
                logger.error("redis_hop_clear_failed", error=str(e), meme_id=meme_id)

    async def close(self) -> None:
        """Close the connection pool."""
        await self._pool.disconnect()
