"""Shared PostgreSQL pool management for SOARcuit services."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any, Final, cast

import asyncpg
import structlog
from asyncpg import Connection
from asyncpg.pool import Pool
from google.auth import default as google_auth_default
from google.auth.transport.requests import Request
from pgvector.asyncpg import register_vector
from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = structlog.get_logger(__name__)

DEFAULT_CLOUD_SQL_INSTANCE: Final[str] = "soarcuit:us-east4:soarcuit-dev-us-east4-postgres"
IAM_SCOPE: Final[str] = "https://www.googleapis.com/auth/cloud-platform"
IAM_REFRESH_SKEW: Final[timedelta] = timedelta(minutes=5)
DEFAULT_DB_CONNECT_TIMEOUT_SECONDS: Final[int] = 30
DEFAULT_DB_MIN_POOL_SIZE: Final[int] = 1


class DatabaseAuthMode(StrEnum):
    """Authentication methods supported by the database pool."""

    PASSWORD = "password"
    IAM = "iam"


class DatabaseConnectionMode(StrEnum):
    """Transport modes supported by the database pool."""

    UNIX_SOCKET = "unix_socket"
    TCP = "tcp"


def _validate_positive(v: int) -> int:
    if v <= 0:
        raise ValueError("Value must be > 0")
    return v


class SOARcuitDatabaseConfig(BaseSettings):
    """Shared configuration for building an asyncpg pool."""

    database: str = Field(
        default="postgres",
        validation_alias=AliasChoices(
            "database", "DB_NAME", "THALAMUS_DB_NAME", "CORTEX_DB_NAME", "HIPPOCAMPUS_DB_NAME"
        ),
    )
    user: str = Field(
        ...,
        validation_alias=AliasChoices(
            "user", "DB_USER", "THALAMUS_DB_USER", "CORTEX_DB_USER", "HIPPOCAMPUS_DB_USER"
        ),
    )
    password: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "password",
            "DB_PASSWORD",
            "THALAMUS_DB_PASSWORD",
            "CORTEX_DB_PASSWORD",
            "HIPPOCAMPUS_DB_PASSWORD",
        ),
    )
    auth_mode: DatabaseAuthMode | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "auth_mode",
            "DB_AUTH_MODE",
            "THALAMUS_DB_AUTH_MODE",
            "CORTEX_DB_AUTH_MODE",
            "HIPPOCAMPUS_DB_AUTH_MODE",
        ),
    )
    connection_mode: DatabaseConnectionMode | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "connection_mode",
            "DB_CONNECTION_MODE",
            "THALAMUS_DB_CONNECTION_MODE",
            "CORTEX_DB_CONNECTION_MODE",
            "HIPPOCAMPUS_DB_CONNECTION_MODE",
        ),
    )
    host: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "host", "DB_HOST", "THALAMUS_DB_HOST", "CORTEX_DB_HOST", "HIPPOCAMPUS_DB_HOST"
        ),
    )
    port: int = Field(
        default=5432,
        validation_alias=AliasChoices(
            "port", "DB_PORT", "THALAMUS_DB_PORT", "CORTEX_DB_PORT", "HIPPOCAMPUS_DB_PORT"
        ),
    )
    cloud_sql_instance: str = Field(
        default=DEFAULT_CLOUD_SQL_INSTANCE,
        validation_alias=AliasChoices(
            "cloud_sql_instance",
            "DB_CLOUD_SQL_INSTANCE",
            "THALAMUS_DB_CLOUD_SQL_INSTANCE",
            "CORTEX_DB_CLOUD_SQL_INSTANCE",
            "HIPPOCAMPUS_DB_CLOUD_SQL_INSTANCE",
        ),
    )
    unix_socket_dir: str = Field(
        default="/cloudsql",
        validation_alias=AliasChoices(
            "unix_socket_dir",
            "DB_UNIX_SOCKET_DIR",
            "THALAMUS_DB_UNIX_SOCKET_DIR",
            "CORTEX_DB_UNIX_SOCKET_DIR",
            "HIPPOCAMPUS_DB_UNIX_SOCKET_DIR",
        ),
    )
    min_pool_size: int = Field(
        default=DEFAULT_DB_MIN_POOL_SIZE,
        validation_alias=AliasChoices(
            "min_pool_size",
            "DB_POOL_MIN_SIZE",
            "THALAMUS_DB_POOL_MIN_SIZE",
            "CORTEX_DB_POOL_MIN_SIZE",
            "HIPPOCAMPUS_DB_POOL_MIN_SIZE",
        ),
    )
    max_pool_size: int = Field(
        default=10,
        validation_alias=AliasChoices(
            "max_pool_size",
            "DB_POOL_MAX_SIZE",
            "THALAMUS_DB_POOL_MAX_SIZE",
            "CORTEX_DB_POOL_MAX_SIZE",
            "HIPPOCAMPUS_DB_POOL_MAX_SIZE",
        ),
    )
    connect_timeout: int = Field(
        default=DEFAULT_DB_CONNECT_TIMEOUT_SECONDS,
        validation_alias=AliasChoices(
            "connect_timeout",
            "DB_CONNECT_TIMEOUT_SECONDS",
            "THALAMUS_DB_CONNECT_TIMEOUT_SECONDS",
            "CORTEX_DB_CONNECT_TIMEOUT_SECONDS",
            "HIPPOCAMPUS_DB_CONNECT_TIMEOUT_SECONDS",
        ),
    )
    enable_pgvector: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "enable_pgvector",
            "DB_ENABLE_PGVECTOR",
            "THALAMUS_DB_ENABLE_PGVECTOR",
            "CORTEX_DB_ENABLE_PGVECTOR",
            "HIPPOCAMPUS_DB_ENABLE_PGVECTOR",
        ),
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
    )

    @model_validator(mode="before")
    @classmethod
    def resolve_defaults(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Resolve auth_mode
            if data.get("auth_mode") is None:
                password = data.get("password")
                data["auth_mode"] = DatabaseAuthMode.PASSWORD if password else DatabaseAuthMode.IAM

            # Resolve connection_mode
            if data.get("connection_mode") is None:
                host = data.get("host")
                data["connection_mode"] = (
                    DatabaseConnectionMode.TCP if host else DatabaseConnectionMode.UNIX_SOCKET
                )

            # Resolve host for UNIX socket if not provided
            if data.get("host") is None:
                if data.get("connection_mode") == DatabaseConnectionMode.TCP:
                    data["host"] = "127.0.0.1"
                else:
                    unix_socket_dir = data.get("unix_socket_dir", "/cloudsql")
                    cloud_sql_instance = data.get("cloud_sql_instance", DEFAULT_CLOUD_SQL_INSTANCE)
                    data["host"] = f"{unix_socket_dir.rstrip('/')}/{cloud_sql_instance}"
        return data


class IAMTokenCache:
    """Thread-safe cache for IAM database tokens."""

    def __init__(self) -> None:
        self._token: str | None = None
        self._expiry: datetime | None = None
        self._lock = asyncio.Lock()

    async def get_token(self) -> str:
        async with self._lock:
            now = datetime.now(UTC)
            if self._token and self._expiry and now < (self._expiry - IAM_REFRESH_SKEW):
                return self._token

            credentials, _ = google_auth_default(scopes=[IAM_SCOPE])
            credentials.refresh(Request())  # type: ignore[no-untyped-call]
            self._token = cast(str, credentials.token)
            self._expiry = credentials.expiry
            return self._token


_TOKEN_CACHE = IAMTokenCache()


async def get_database_password(config: SOARcuitDatabaseConfig) -> str:
    """Returns the password or a fresh IAM token."""
    if config.auth_mode == DatabaseAuthMode.PASSWORD:
        if not config.password:
            raise ValueError("Password is required for PASSWORD auth mode")
        return config.password
    return await _TOKEN_CACHE.get_token()


async def init_connection(conn: Connection, enable_pgvector: bool = True) -> None:
    """Initialize a database connection with required extensions."""
    if enable_pgvector:
        await register_vector(conn)

    # Set search path or other session variables if needed
    await conn.execute("SET TIME ZONE 'UTC';")


async def create_db_pool(config: SOARcuitDatabaseConfig) -> Pool:
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
        init=lambda conn: init_connection(conn, config.enable_pgvector),
    )

    if pool is None:
        raise RuntimeError("Failed to create database pool")

    return pool


@asynccontextmanager
async def database_pool(config: SOARcuitDatabaseConfig) -> AsyncIterator[Pool]:
    """Context manager for a managed database pool."""
    pool = await create_db_pool(config)
    try:
        yield pool
    finally:
        await pool.close()
