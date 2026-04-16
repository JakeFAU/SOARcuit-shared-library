"""Shared PostgreSQL pool management for SOARcuit services."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Final, cast

import asyncpg
import structlog
from asyncpg import Connection
from asyncpg.pool import Pool
from google.auth import default as google_auth_default
from google.auth.transport.requests import Request
from pgvector.asyncpg import register_vector
from shared.config import config as shared_config

logger = structlog.get_logger(__name__)

IAM_SCOPE: Final[str] = "https://www.googleapis.com/auth/cloud-platform"
IAM_REFRESH_SKEW: Final[timedelta] = timedelta(minutes=5)

DEFAULT_CLOUD_SQL_INSTANCE = shared_config.DEFAULT_CLOUD_SQL_INSTANCE
DEFAULT_DB_CONNECT_TIMEOUT_SECONDS = shared_config.DEFAULT_DB_CONNECT_TIMEOUT_SECONDS
DEFAULT_DB_MIN_POOL_SIZE = shared_config.DEFAULT_DB_MIN_POOL_SIZE
DatabaseAuthMode = shared_config.DatabaseAuthMode
DatabaseConnectionMode = shared_config.DatabaseConnectionMode
DatabaseSettings = shared_config.DatabaseSettings
SOARcuitDatabaseConfig = DatabaseSettings


class IAMTokenCache:
    """Thread-safe cache for IAM database tokens."""

    def __init__(self) -> None:
        self._token: str | None = None
        self._expiry: datetime | None = None
        self._lock = asyncio.Lock()

    async def get_token(self) -> str:
        async with self._lock:
            now = datetime.now(UTC)
            expiry = self._expiry
            if expiry is not None and expiry.tzinfo is None:
                expiry = expiry.replace(tzinfo=UTC)

            if self._token and expiry and now < (expiry - IAM_REFRESH_SKEW):
                return self._token

            credentials, _ = google_auth_default(scopes=[IAM_SCOPE])
            credentials.refresh(Request())  # type: ignore[no-untyped-call]
            token = cast(str | None, credentials.token)
            if not token:
                raise RuntimeError("Google credentials refresh did not return an access token.")

            self._token = token
            self._expiry = credentials.expiry
            if self._expiry is not None and self._expiry.tzinfo is None:
                self._expiry = self._expiry.replace(tzinfo=UTC)
            return self._token


_TOKEN_CACHE = IAMTokenCache()


def reset_token_cache() -> None:
    """Reset the global IAM token cache (primarily for testing)."""
    global _TOKEN_CACHE
    _TOKEN_CACHE = IAMTokenCache()


async def get_database_password(config: DatabaseSettings) -> str:
    """Returns the password or a fresh IAM token."""
    if config.auth_mode == DatabaseAuthMode.PASSWORD:
        if not config.password:
            raise ValueError("Password is required for PASSWORD auth mode")
        return config.password.get_secret_value()
    return await _TOKEN_CACHE.get_token()


async def init_connection(conn: Connection, enable_pgvector: bool = True) -> None:
    """Initialize a database connection with required extensions."""
    if enable_pgvector:
        await register_vector(conn)

    # Set search path or other session variables if needed
    await conn.execute("SET TIME ZONE 'UTC';")


class DatabasePoolManager:
    """Coordinates lazy creation and shutdown of a shared asyncpg pool."""

    def __init__(self) -> None:
        self._pool: Pool | None = None
        self._lock = asyncio.Lock()

    async def get_pool(self, config: DatabaseSettings) -> Pool:
        if self._pool is not None:
            return self._pool
        async with self._lock:
            if self._pool is None:
                self._pool = await create_db_pool(config)
        return self._pool

    async def close(self) -> None:
        async with self._lock:
            if self._pool is not None:
                logger.info("postgres_pool_closing")
                await self._pool.close()
                self._pool = None
                logger.info("postgres_pool_closed")


database_pool_manager = DatabasePoolManager()


@asynccontextmanager
async def acquire_connection(
    db: Pool | Connection | None = None,
    *,
    config: DatabaseSettings | None = None,
) -> AsyncIterator[Connection]:
    """Acquire a connection from the pool or use the provided one."""
    if db is None:
        if config is None:
            raise ValueError("Either db or config must be provided")
        db = await database_pool_manager.get_pool(config)

    if isinstance(db, Pool):
        async with db.acquire() as conn:
            yield cast(Connection, conn)
        return
    yield db


@asynccontextmanager
async def transaction(
    db: Pool | Connection | None = None,
    *,
    config: DatabaseSettings | None = None,
) -> AsyncIterator[Connection]:
    """Execute a block within a database transaction."""
    if db is None:
        if config is None:
            raise ValueError("Either db or config must be provided")
        db = await database_pool_manager.get_pool(config)

    if isinstance(db, Pool):
        async with db.acquire() as conn:
            async with conn.transaction():
                yield cast(Connection, conn)
        return
    async with db.transaction():
        yield db


async def create_db_pool(config: DatabaseSettings) -> Pool:
    """Create and initialize an asyncpg pool."""

    async def get_password() -> str:
        return await get_database_password(config)

    pool = await asyncpg.create_pool(
        user=config.user,
        password=get_password,
        database=config.database,
        host=config.host,
        port=config.port,
        min_size=config.min_pool_size,
        max_size=config.max_pool_size,
        timeout=config.connect_timeout,
        command_timeout=config.command_timeout,
        max_queries=config.max_queries,
        max_inactive_connection_lifetime=config.max_inactive_connection_lifetime,
        init=lambda conn: init_connection(conn, config.enable_pgvector),
    )

    if pool is None:
        raise RuntimeError("Failed to create database pool")

    return pool


async def ping_pool(pool: Pool) -> None:
    """Verify the pool is functional by executing a simple query."""
    async with pool.acquire() as conn:
        await conn.execute("SELECT 1")


@asynccontextmanager
async def database_pool(config: DatabaseSettings) -> AsyncIterator[Pool]:
    """Context manager for a managed database pool."""
    pool = await create_db_pool(config)
    try:
        yield pool
    finally:
        await pool.close()
