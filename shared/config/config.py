from __future__ import annotations

from abc import ABC
from enum import StrEnum
from functools import lru_cache
from typing import Final, Literal

from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_CLOUD_SQL_INSTANCE: Final[str] = "soarcuit:us-east4:soarcuit-dev-us-east4-postgres"
DEFAULT_DB_CONNECT_TIMEOUT_SECONDS: Final[int] = 30
DEFAULT_DB_MIN_POOL_SIZE: Final[int] = 1
DEFAULT_GCP_REGION: Final[str] = "us-east4"


class ConfigurationError(ValueError):
    """Raised when configuration is invalid or incomplete."""


class DatabaseAuthMode(StrEnum):
    PASSWORD = "password"
    IAM = "iam"


class DatabaseConnectionMode(StrEnum):
    UNIX_SOCKET = "unix_socket"
    TCP = "tcp"


class LLMProvider(StrEnum):
    GEMINI = "gemini"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class DatabaseSettings(BaseModel):
    """Database configuration for Cloud SQL / Postgres."""

    database: str = Field(description="Name of the database to connect to.")
    user: str = Field(description="Username to connect to the database.")

    password: SecretStr | None = Field(
        default=None,
        description="Password for database auth when using PASSWORD mode.",
    )

    auth_mode: DatabaseAuthMode | None = Field(
        default=None,
        description="Authentication mode. If omitted, inferred from password presence.",
    )
    connection_mode: DatabaseConnectionMode | None = Field(
        default=None,
        description="Connection mode. If omitted, inferred from host presence.",
    )

    host: str | None = Field(
        default=None,
        description="Database host when using TCP mode.",
    )
    port: int = Field(
        default=5432,
        gt=0,
        lt=65536,
        description="Database port when using TCP mode.",
    )

    cloud_sql_instance: str = Field(
        default=DEFAULT_CLOUD_SQL_INSTANCE,
        min_length=1,
        description="Cloud SQL instance connection name.",
    )
    unix_socket_dir: str = Field(
        default="/cloudsql",
        min_length=1,
        description="Directory under which the Cloud SQL Unix socket is mounted.",
    )

    min_pool_size: int = Field(
        default=DEFAULT_DB_MIN_POOL_SIZE,
        ge=1,
        description="Minimum number of connections to keep in the pool.",
    )
    max_pool_size: int = Field(
        default=10,
        ge=1,
        description="Maximum number of connections to keep in the pool.",
    )

    connect_timeout: int = Field(
        default=DEFAULT_DB_CONNECT_TIMEOUT_SECONDS,
        gt=0,
        description="Seconds to wait when establishing a database connection.",
    )
    command_timeout: float = Field(
        default=30.0,
        gt=0,
        description="Seconds to wait for a query / command to complete.",
    )

    statement_cache_size: int = Field(
        default=100,
        ge=0,
        description="Number of prepared statements to cache.",
    )
    max_queries: int = Field(
        default=50_000,
        ge=0,
        description="Maximum queries before recycling a pooled connection.",
    )
    max_inactive_connection_lifetime: int = Field(
        default=300,
        ge=0,
        description="Maximum idle lifetime, in seconds, for pooled connections.",
    )

    enable_pgvector: bool = Field(
        default=True,
        description="Whether pgvector-dependent features are enabled.",
    )

    @model_validator(mode="after")
    def validate_settings(self) -> DatabaseSettings:
        if self.min_pool_size > self.max_pool_size:
            raise ValueError("min_pool_size must be <= max_pool_size")

        if self.resolved_auth_mode == DatabaseAuthMode.PASSWORD and self.password is None:
            raise ValueError("password must be set when auth_mode is PASSWORD")

        if self.resolved_connection_mode == DatabaseConnectionMode.TCP and not self.host:
            raise ValueError("host must be set when connection_mode is TCP")

        return self

    @property
    def resolved_auth_mode(self) -> DatabaseAuthMode:
        return self.auth_mode or (
            DatabaseAuthMode.PASSWORD if self.password is not None else DatabaseAuthMode.IAM
        )

    @property
    def resolved_connection_mode(self) -> DatabaseConnectionMode:
        return self.connection_mode or (
            DatabaseConnectionMode.TCP if self.host else DatabaseConnectionMode.UNIX_SOCKET
        )

    @property
    def resolved_host(self) -> str:
        if self.resolved_connection_mode == DatabaseConnectionMode.TCP:
            if not self.host:
                raise ConfigurationError("host is required to resolve a TCP database connection")
            return self.host

        return f"{self.unix_socket_dir.rstrip('/')}/{self.cloud_sql_instance}"


class LLMSettings(BaseModel):
    """Configuration for model providers and embeddings."""

    gemini_api_key: SecretStr = Field(
        description="Required Gemini API key. Gemini is used for embeddings.",
    )
    openai_api_key: SecretStr | None = Field(
        default=None,
        description="Optional OpenAI API key.",
    )
    anthropic_api_key: SecretStr | None = Field(
        default=None,
        description="Optional Anthropic API key.",
    )

    default_provider: LLMProvider = Field(
        default=LLMProvider.GEMINI,
        description="Default provider to use for general LLM calls.",
    )

    default_gemini_model: str = Field(
        default="gemini-3-flash-preview",
        min_length=1,
        description="Default Gemini model for chat / generation.",
    )
    default_openai_model: str = Field(
        default="gpt-5-mini",
        min_length=1,
        description="Default OpenAI model for chat / generation.",
    )
    default_anthropic_model: str = Field(
        default="claude-3-5-haiku-latest",
        min_length=1,
        description="Default Anthropic model for chat / generation.",
    )

    embedding_provider: Literal["gemini"] = Field(
        default="gemini",
        description="Embedding provider. Fixed to Gemini in this application.",
    )
    embedding_model: str = Field(
        default="gemini-embedding-2-preview",
        min_length=1,
        description="Embedding model name.",
    )
    embedding_dimension: int = Field(
        default=768,
        gt=0,
        description="Expected embedding vector dimension.",
    )

    @model_validator(mode="after")
    def validate_settings(self) -> LLMSettings:
        if self.default_provider == LLMProvider.OPENAI and self.openai_api_key is None:
            raise ValueError("default_provider is OPENAI but openai_api_key is not configured")

        if self.default_provider == LLMProvider.ANTHROPIC and self.anthropic_api_key is None:
            raise ValueError(
                "default_provider is ANTHROPIC but anthropic_api_key is not configured"
            )

        return self

    @property
    def gemini_enabled(self) -> bool:
        return self.gemini_api_key is not None

    @property
    def openai_enabled(self) -> bool:
        return self.openai_api_key is not None

    @property
    def anthropic_enabled(self) -> bool:
        return self.anthropic_api_key is not None

    @property
    def default_model(self) -> str:
        match self.default_provider:
            case LLMProvider.GEMINI:
                return self.default_gemini_model
            case LLMProvider.OPENAI:
                return self.default_openai_model
            case LLMProvider.ANTHROPIC:
                return self.default_anthropic_model

        raise ConfigurationError(f"Unsupported default provider: {self.default_provider!r}")


class GCPSettings(BaseModel):
    """Google Cloud Platform configuration."""

    project_id: str = Field(
        min_length=1,
        description="Google Cloud project ID.",
    )
    project_name: str = Field(
        min_length=1,
        description="Human-readable project name.",
    )
    default_region: str = Field(
        default=DEFAULT_GCP_REGION,
        min_length=1,
        description="Default GCP region.",
    )


class ProviderModels(BaseModel, ABC):
    thinking_model: str
    default_model: str
    quick_model: str


class OpenAIModels(ProviderModels):
    thinking_model: str = Field(default="gpt-5.4", description="Thinking model for OpenAI.")
    default_model: str = Field(default="gpt-5", description="Default model for OpenAI.")
    quick_model: str = Field(default="gpt-5-mini", description="Quick model for OpenAI.")


class GeminiModels(ProviderModels):
    thinking_model: str = Field(
        default="gemini-3.1-pro-preview", description="Thinking model for Gemini."
    )
    default_model: str = Field(
        default="gemini-3-flash-preview", description="Default model for Gemini."
    )
    quick_model: str = Field(
        default="gemini-3.1-flash-lite-preview", description="Quick model for Gemini."
    )


class AnthropicModels(ProviderModels):
    thinking_model: str = Field(
        default="claude-opus-4-7", description="Thinking model for Anthropic."
    )
    default_model: str = Field(
        default="claude-sonnet-4-6", description="Default model for Anthropic."
    )
    quick_model: str = Field(default="claude-haiku-4-5", description="Quick model for Anthropic.")


class ModelNames(BaseModel):
    model_config = SettingsConfigDict(
        env_prefix="SOAR_MODELS_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
        case_sensitive=False,
    )

    openai: OpenAIModels = Field(default_factory=OpenAIModels)
    gemini: GeminiModels = Field(default_factory=GeminiModels)
    anthropic: AnthropicModels = Field(default_factory=AnthropicModels)


class AppSettings(BaseSettings):
    """Top-level application settings."""

    model_config = SettingsConfigDict(
        env_prefix="SOAR_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
        case_sensitive=False,
    )

    database_settings: DatabaseSettings
    llm_settings: LLMSettings
    gcp_settings: GCPSettings
    model_names: ModelNames


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()  # type: ignore[call-arg]
