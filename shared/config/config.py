from datetime import timedelta
from enum import StrEnum
from typing import Final

import structlog
from pydantic import (
    AliasChoices,
    Field,
    SecretStr,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = structlog.get_logger(__name__)

DEFAULT_CLOUD_SQL_INSTANCE: Final[str] = "soarcuit:us-east4:soarcuit-dev-us-east4-postgres"
IAM_SCOPE: Final[str] = "https://www.googleapis.com/auth/cloud-platform"
IAM_REFRESH_SKEW: Final[timedelta] = timedelta(minutes=5)
DEFAULT_DB_CONNECT_TIMEOUT_SECONDS: Final[int] = 30
DEFAULT_DB_MIN_POOL_SIZE: Final[int] = 1


class ConfigurationError(Exception):
    """Raised when the configuration / settings are invalid."""

    def __init__(self, message: str, section: str | None = None) -> None:
        self.message = message
        self.section = section
        super().__init__(f"{section + ': ' if section else ''}{message}")


class DatabaseAuthMode(StrEnum):
    """Authentication methods supported by the database pool."""

    PASSWORD = "password"
    IAM = "iam"


class DatabaseConnectionMode(StrEnum):
    """Transport modes supported by the database pool."""

    UNIX_SOCKET = "unix_socket"
    TCP = "tcp"


class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
        case_sensitive=True,
    )

    database: str = Field(
        default="postgres",
        validation_alias=AliasChoices("DB_NAME", "SOAR_DB_NAME", "SOARCUIT_DB_NAME"),
    )

    user: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "DB_USER",
            "DATABASE_USER",
            "SOAR_DB_USER",
            "SOARCUIT_DB_USER",
        ),
    )

    password: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "DB_PASSWORD",
            "DATABASE_PASSWORD",
            "SOAR_DB_PASSWORD",
            "SOARCUIT_DB_PASSWORD",
        ),
    )
    auth_mode: DatabaseAuthMode | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "DB_AUTH_MODE",
            "SOAR_DB_AUTH_MODE",
            "SOARCUIT_DB_AUTH_MODE",
        ),
    )
    connection_mode: DatabaseConnectionMode | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "DB_CONNECTION_MODE",
            "SOAR_DB_CONNECTION_MODE",
            "SOARCUIT_DB_CONNECTION_MODE",
        ),
    )
    host: str | None = Field(
        default=None,
        validation_alias=AliasChoices("DB_HOST", "SOAR_DB_HOST", "SOARCUIT_DB_HOST"),
    )
    port: int = Field(
        default=5432,
        validation_alias=AliasChoices(
            "DB_PORT",
            "SOAR_DB_PORT",
            "SOARCUIT_DB_PORT",
        ),
    )
    cloud_sql_instance: str = Field(
        default=DEFAULT_CLOUD_SQL_INSTANCE,
        validation_alias=AliasChoices(
            "DB_CLOUD_SQL_INSTANCE",
            "SOAR_DB_CLOUD_SQL_INSTANCE",
            "SOARCUIT_DB_CLOUD_SQL_INSTANCE",
        ),
    )
    unix_socket_dir: str = Field(
        default="/cloudsql",
        validation_alias=AliasChoices(
            "DB_UNIX_SOCKET_DIR",
            "SOAR_DB_UNIX_SOCKET_DIR",
            "SOARCUIT_DB_UNIX_SOCKET_DIR",
        ),
    )
    min_pool_size: int = Field(
        default=DEFAULT_DB_MIN_POOL_SIZE,
        validation_alias=AliasChoices(
            "DB_POOL_MIN_SIZE",
            "SOAR_DB_POOL_MIN_SIZE",
            "SOARCUIT_DB_POOL_MIN_SIZE",
        ),
    )
    max_pool_size: int = Field(
        default=10,
        validation_alias=AliasChoices(
            "DB_POOL_MAX_SIZE",
            "SOAR_DB_POOL_MAX_SIZE",
            "SOARCUIT_DB_POOL_MAX_SIZE",
        ),
    )
    connect_timeout: int = Field(
        default=DEFAULT_DB_CONNECT_TIMEOUT_SECONDS,
        validation_alias=AliasChoices(
            "DB_CONNECT_TIMEOUT_SECONDS",
            "SOAR_DB_CONNECT_TIMEOUT_SECONDS",
            "SOARCUIT_DB_CONNECT_TIMEOUT_SECONDS",
        ),
    )
    command_timeout: float = Field(
        default=30.0,
        validation_alias=AliasChoices(
            "DB_COMMAND_TIMEOUT_SECONDS",
            "SOAR_DB_COMMAND_TIMEOUT_SECONDS",
            "SOARCUIT_DB_COMMAND_TIMEOUT_SECONDS",
        ),
    )
    statement_cache_size: int = Field(
        default=100,
        validation_alias=AliasChoices(
            "DB_STATEMENT_CACHE_SIZE",
            "SOAR_DB_STATEMENT_CACHE_SIZE",
            "SOARCUIT_DB_STATEMENT_CACHE_SIZE",
        ),
    )
    max_queries: int = Field(
        default=50_000,
        validation_alias=AliasChoices(
            "DB_MAX_QUERIES",
            "SOAR_DB_POOL_MAX_QUERIES",
            "SOARCUIT_DB_POOL_MAX_QUERIES",
        ),
    )
    max_inactive_connection_lifetime: float = Field(
        default=300.0,
        validation_alias=AliasChoices(
            "DB_MAX_INACTIVE_LIFETIME",
            "SOAR_DB_POOL_MAX_INACTIVE_LIFETIME",
            "SOARCUIT_DB_POOL_MAX_INACTIVE_LIFETIME",
        ),
    )
    enable_pgvector: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "DB_ENABLE_PGVECTOR",
            "SOAR_DB_ENABLE_PGVECTOR",
            "SOARCUIT_DB_ENABLE_PGVECTOR",
        ),
    )

    @model_validator(mode="after")
    def validate_settings(self) -> DatabaseSettings:
        """Makes Sure the DatabaseSettings are valid after the model has been initialized."""

        # make sure the user is set
        if not self.user:
            raise ConfigurationError("Database user must be set", section="DatabaseSettings")

        # Resolve auth_mode
        if self.auth_mode is None:
            self.auth_mode = DatabaseAuthMode.PASSWORD if self.password else DatabaseAuthMode.IAM

        # Resolve connection_mode
        if self.connection_mode is None:
            self.connection_mode = (
                DatabaseConnectionMode.TCP if self.host else DatabaseConnectionMode.UNIX_SOCKET
            )

        # Resolve host for UNIX socket if not provided
        if self.host is None:
            if self.connection_mode == DatabaseConnectionMode.TCP:
                self.host = "127.0.0.1"
            else:
                unix_socket_dir = self.unix_socket_dir
                cloud_sql_instance = self.cloud_sql_instance
                self.host = f"{unix_socket_dir.rstrip('/')}/{cloud_sql_instance}"
        return self


class LLMSettings(BaseSettings):
    """Settings for the LLM configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
        case_sensitive=True,
    )

    gemini_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "GEMINI_API_KEY",
            "SOAR_GEMINI_API_KEY",
            "SOARCUIT_GEMINI_API_KEY",
        ),
    )

    open_ai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "OPEN_AI_API_KEY",
            "SOAR_OPEN_AI_API_KEY",
            "SOARCUIT_OPEN_AI_API_KEY",
        ),
    )

    anthropic_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "ANTHROPIC_API_KEY",
            "SOAR_ANTHROPIC_API_KEY",
            "SOARCUIT_ANTHROPIC_API_KEY",
        ),
    )

    @model_validator(mode="after")
    def validate_settings(self) -> LLMSettings:
        """Validates the LLM settings after the model has been initialized."""
        if not self.gemini_api_key and not self.open_ai_api_key and not self.anthropic_api_key:
            raise ConfigurationError(
                "At least one LLM API key must be set (Gemini / OpenAI / Anthropic).",
                section="LLMSettings",
            )
        return self

    @property
    def gemini_enabled(self) -> bool:
        """Returns True if the Gemini API key is configured / enabled."""
        return self.gemini_api_key is not None

    @property
    def open_ai_enabled(self) -> bool:
        """Returns True if the OpenAI API key is configured / enabled."""
        return self.open_ai_api_key is not None

    @property
    def anthropic_enabled(self) -> bool:
        """Returns True if the Anthropic API key is configured / enabled."""
        return self.anthropic_api_key is not None


class AppSettings(BaseSettings):
    """Settings for the Application"""

    database_settings: DatabaseSettings = Field(
        default_factory=DatabaseSettings,
        validation_alias=AliasChoices(
            "DATABASE_SETTINGS",
            "SOAR_DATABASE_SETTINGS",
            "SOARCUIT_DATABASE_SETTINGS",
        ),
    )

    llm_settings: LLMSettings = Field(
        default_factory=LLMSettings,
        validation_alias=AliasChoices(
            "LLM_SETTINGS",
            "SOAR_LLM_SETTINGS",
            "SOARCUIT_LLM_SETTINGS",
        ),
    )

    @model_validator(mode="after")
    def validate_settings(self) -> AppSettings:
        """Validates the AppSettings after the model has been initialized."""
        # errors should have already been raised, but just in case
        if not self.database_settings or not self.llm_settings:
            raise ConfigurationError(
                "AppSettings must have both database_settings and llm_settings configured / initialized",
                section="AppSettings",
            )

        return self
