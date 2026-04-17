"""
SOARcuit Configuration Management.

This module defines the central configuration schema for the SOARcuit ecosystem
using Pydantic Settings. It manages environment-specific variables, model hierarchies,
and external tool credentials with strict validation and type safety.

The configuration is hierarchical:
- AppSettings: The top-level entry point.
- LLMSettings: Provider keys and default model selections.
- GCPSettings: Project-level identity for Google Cloud.
- ModelNames: Tiered model aliases (quick, default, thinking) for dynamic dispatch.
- ExternalToolSettings: API keys and constraints for non-LLM services.
"""

from __future__ import annotations

from abc import ABC
from enum import StrEnum
from functools import lru_cache
import os
from typing import Final, Literal

from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_CLOUD_SQL_INSTANCE: Final[str] = "soarcuit:us-east4:soarcuit-dev-us-east4-postgres"
DEFAULT_DB_CONNECT_TIMEOUT_SECONDS: Final[int] = 30
DEFAULT_DB_MIN_POOL_SIZE: Final[int] = 1
DEFAULT_GCP_REGION: Final[str] = "us-east4"


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value is not None and value != "":
            return value
    return None


class ConfigurationError(ValueError):
    """Raised when configuration is invalid or incomplete."""


class DatabaseAuthMode(StrEnum):
    """Supported authentication methods for Postgres."""

    PASSWORD = "password"
    IAM = "iam"


class DatabaseConnectionMode(StrEnum):
    """Supported network topologies for Postgres."""

    UNIX_SOCKET = "unix_socket"
    TCP = "tcp"


class LLMProvider(StrEnum):
    """Supported LLM vendors."""

    GEMINI = "gemini"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class DatabaseSettings(BaseModel):
    """
    Postgres / Cloud SQL Connection Configuration.

    Handles both local TCP connections and production-grade Cloud SQL Proxy
    Unix sockets. Includes logic for inferring auth and connection modes
    based on provided fields.
    """

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
        """Cross-field validation for consistent connection modes."""
        if self.min_pool_size > self.max_pool_size:
            raise ValueError("min_pool_size must be <= max_pool_size")

        if self.resolved_auth_mode == DatabaseAuthMode.PASSWORD and self.password is None:
            raise ValueError("password must be set when auth_mode is PASSWORD")

        if self.resolved_connection_mode == DatabaseConnectionMode.TCP and not self.host:
            raise ValueError("host must be set when connection_mode is TCP")

        return self

    @property
    def resolved_auth_mode(self) -> DatabaseAuthMode:
        """Infers the auth mode from password presence if not explicit."""
        return self.auth_mode or (
            DatabaseAuthMode.PASSWORD if self.password is not None else DatabaseAuthMode.IAM
        )

    @property
    def resolved_connection_mode(self) -> DatabaseConnectionMode:
        """Infers connection mode from host presence if not explicit."""
        return self.connection_mode or (
            DatabaseConnectionMode.TCP if self.host else DatabaseConnectionMode.UNIX_SOCKET
        )

    @property
    def resolved_host(self) -> str:
        """Resolves the connection string target based on mode."""
        if self.resolved_connection_mode == DatabaseConnectionMode.TCP:
            if not self.host:
                raise ConfigurationError("host is required to resolve a TCP database connection")
            return self.host

        return f"{self.unix_socket_dir.rstrip('/')}/{self.cloud_sql_instance}"


class LLMSettings(BaseModel):
    """
    Global LLM Provider and Embedding Configuration.

    Defines the source of truth for vendor API keys and fallback model
    selections. Also configures the canonical embedding model used for
    vector operations.
    """

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
        """Ensures API keys are present for the selected default provider."""
        if self.default_provider == LLMProvider.OPENAI and self.openai_api_key is None:
            raise ValueError("default_provider is OPENAI but openai_api_key is not configured")

        if self.default_provider == LLMProvider.ANTHROPIC and self.anthropic_api_key is None:
            raise ValueError(
                "default_provider is ANTHROPIC but anthropic_api_key is not configured"
            )

        return self

    @property
    def gemini_enabled(self) -> bool:
        """Check if Gemini credentials are available."""
        return self.gemini_api_key is not None

    @property
    def openai_enabled(self) -> bool:
        """Check if OpenAI credentials are available."""
        return self.openai_api_key is not None

    @property
    def anthropic_enabled(self) -> bool:
        """Check if Anthropic credentials are available."""
        return self.anthropic_api_key is not None

    @property
    def default_model(self) -> str:
        """Resolves the default model name for the selected provider."""
        match self.default_provider:
            case LLMProvider.GEMINI:
                return self.default_gemini_model
            case LLMProvider.OPENAI:
                return self.default_openai_model
            case LLMProvider.ANTHROPIC:
                return self.default_anthropic_model

        raise ConfigurationError(f"Unsupported default provider: {self.default_provider!r}")


class GCPSettings(BaseModel):
    """Google Cloud Platform Project Identity."""

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


class ExternalToolSettings(BaseModel):
    """Credentials and Constraints for Third-Party Services."""

    tavily_api_key: SecretStr | None = Field(
        default=None,
        description="API key for Tavily search.",
    )
    github_token: SecretStr | None = Field(
        default=None,
        description="Personal access token for GitHub API.",
    )
    slack_webhook_url: SecretStr | None = Field(
        default=None,
        description="Webhook URL for Slack notifications.",
    )
    discord_webhook_url: SecretStr | None = Field(
        default=None,
        description="Webhook URL for Discord notifications.",
    )
    arxiv_max_results: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum results for arXiv searches.",
    )


class ProviderModels(BaseModel, ABC):
    """Abstract interface for tiered model assignments per provider."""

    thinking_model: str
    default_model: str
    quick_model: str


class OpenAIModels(ProviderModels):
    """Tiered assignments for OpenAI models."""

    thinking_model: str = Field(default="gpt-5.4", description="Thinking model for OpenAI.")
    default_model: str = Field(default="gpt-5", description="Default model for OpenAI.")
    quick_model: str = Field(default="gpt-5-mini", description="Quick model for OpenAI.")


class GeminiModels(ProviderModels):
    """Tiered assignments for Gemini models (Primary Stack)."""

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
    """Tiered assignments for Anthropic models."""

    thinking_model: str = Field(
        default="claude-opus-4-7", description="Thinking model for Anthropic."
    )
    default_model: str = Field(
        default="claude-sonnet-4-6", description="Default model for Anthropic."
    )
    quick_model: str = Field(default="claude-haiku-4-5", description="Quick model for Anthropic.")


class ModelNames(BaseModel):
    """
    Central Registry of Model Hierarchies.

    Allows agents to request a tier (e.g., 'quick') rather than a specific
    version, enabling easy system-wide model upgrades.
    """

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


def _database_settings_from_env() -> dict[str, object] | None:
    data: dict[str, object] = {}
    env_map = {
        "database": ("SOAR_DATABASE_SETTINGS__DATABASE", "THALAMUS_DB_NAME"),
        "user": ("SOAR_DATABASE_SETTINGS__USER", "THALAMUS_DB_USER"),
        "password": ("SOAR_DATABASE_SETTINGS__PASSWORD", "THALAMUS_DB_PASSWORD"),
        "auth_mode": ("SOAR_DATABASE_SETTINGS__AUTH_MODE", "THALAMUS_DB_AUTH_MODE"),
        "connection_mode": (
            "SOAR_DATABASE_SETTINGS__CONNECTION_MODE",
            "THALAMUS_DB_CONNECTION_MODE",
        ),
        "host": ("SOAR_DATABASE_SETTINGS__HOST", "THALAMUS_DB_HOST"),
        "port": ("SOAR_DATABASE_SETTINGS__PORT", "THALAMUS_DB_PORT"),
        "cloud_sql_instance": (
            "SOAR_DATABASE_SETTINGS__CLOUD_SQL_INSTANCE",
            "THALAMUS_DB_CLOUD_SQL_INSTANCE",
        ),
        "unix_socket_dir": (
            "SOAR_DATABASE_SETTINGS__UNIX_SOCKET_DIR",
            "THALAMUS_DB_UNIX_SOCKET_DIR",
        ),
        "min_pool_size": ("SOAR_DATABASE_SETTINGS__MIN_POOL_SIZE", "THALAMUS_DB_POOL_MIN_SIZE"),
        "max_pool_size": ("SOAR_DATABASE_SETTINGS__MAX_POOL_SIZE", "THALAMUS_DB_POOL_MAX_SIZE"),
        "connect_timeout": (
            "SOAR_DATABASE_SETTINGS__CONNECT_TIMEOUT",
            "THALAMUS_DB_CONNECT_TIMEOUT_SECONDS",
        ),
        "command_timeout": (
            "SOAR_DATABASE_SETTINGS__COMMAND_TIMEOUT",
            "THALAMUS_DB_COMMAND_TIMEOUT_SECONDS",
        ),
        "statement_cache_size": (
            "SOAR_DATABASE_SETTINGS__STATEMENT_CACHE_SIZE",
            "THALAMUS_DB_STATEMENT_CACHE_SIZE",
        ),
        "max_queries": ("SOAR_DATABASE_SETTINGS__MAX_QUERIES", "THALAMUS_DB_MAX_QUERIES"),
        "max_inactive_connection_lifetime": (
            "SOAR_DATABASE_SETTINGS__MAX_INACTIVE_CONNECTION_LIFETIME",
            "THALAMUS_DB_MAX_INACTIVE_CONNECTION_LIFETIME",
        ),
        "enable_pgvector": (
            "SOAR_DATABASE_SETTINGS__ENABLE_PGVECTOR",
            "THALAMUS_DB_ENABLE_PGVECTOR",
        ),
    }
    for field_name, env_names in env_map.items():
        value = _first_env(*env_names)
        if value is not None:
            data[field_name] = value
    return data or None


def _llm_settings_from_env() -> dict[str, object] | None:
    data: dict[str, object] = {}
    env_map = {
        "gemini_api_key": (
            "SOAR_LLM_SETTINGS__GEMINI_API_KEY",
            "GOOGLE_GENAI_KEY",
            "GEMINI_API_KEY",
            "THALAMUS_GOOGLE_GENAI_KEY",
        ),
        "openai_api_key": ("SOAR_LLM_SETTINGS__OPENAI_API_KEY", "OPENAI_API_KEY"),
        "anthropic_api_key": ("SOAR_LLM_SETTINGS__ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
        "default_provider": ("SOAR_LLM_SETTINGS__DEFAULT_PROVIDER",),
        "default_gemini_model": ("SOAR_LLM_SETTINGS__DEFAULT_GEMINI_MODEL",),
        "default_openai_model": ("SOAR_LLM_SETTINGS__DEFAULT_OPENAI_MODEL",),
        "default_anthropic_model": ("SOAR_LLM_SETTINGS__DEFAULT_ANTHROPIC_MODEL",),
        "embedding_provider": ("SOAR_LLM_SETTINGS__EMBEDDING_PROVIDER",),
        "embedding_model": (
            "SOAR_LLM_SETTINGS__EMBEDDING_MODEL",
            "THALAMUS_EMBEDDING_MODEL",
        ),
        "embedding_dimension": (
            "SOAR_LLM_SETTINGS__EMBEDDING_DIMENSION",
            "THALAMUS_EMBEDDING_DIMS",
        ),
    }
    for field_name, env_names in env_map.items():
        value = _first_env(*env_names)
        if value is not None:
            data[field_name] = value
    return data or None


def _gcp_settings_from_env() -> dict[str, object] | None:
    project_id = _first_env(
        "SOAR_GCP_SETTINGS__PROJECT_ID",
        "THALAMUS_GCP_PROJECT",
        "GOOGLE_CLOUD_PROJECT",
        "GCP_PROJECT",
        "GCLOUD_PROJECT",
    )
    project_name = _first_env(
        "SOAR_GCP_SETTINGS__PROJECT_NAME",
        "THALAMUS_GCP_PROJECT_NAME",
    )
    default_region = _first_env(
        "SOAR_GCP_SETTINGS__DEFAULT_REGION",
        "THALAMUS_GCP_REGION",
        "GOOGLE_CLOUD_REGION",
    )

    data: dict[str, object] = {}
    if project_id is not None:
        data["project_id"] = project_id
        data["project_name"] = project_name or project_id
    elif project_name is not None:
        data["project_name"] = project_name
    if default_region is not None:
        data["default_region"] = default_region
    return data or None


class AppSettings(BaseSettings):
    """
    SOARcuit Unified Application Configuration.

    The root configuration object, typically loaded from environment variables
    prefixed with 'SOAR_'.
    """

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
    model_names: ModelNames = Field(default_factory=ModelNames)
    external_tools: ExternalToolSettings = Field(default_factory=ExternalToolSettings)

    @model_validator(mode="before")
    @classmethod
    def populate_shared_settings_from_environment(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data

        values = dict(data)
        if "database_settings" not in values:
            database_settings = _database_settings_from_env()
            if database_settings is not None:
                values["database_settings"] = database_settings
        if "llm_settings" not in values:
            llm_settings = _llm_settings_from_env()
            if llm_settings is not None:
                values["llm_settings"] = llm_settings
        if "gcp_settings" not in values:
            gcp_settings = _gcp_settings_from_env()
            if gcp_settings is not None:
                values["gcp_settings"] = gcp_settings
        return values


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """
    Returns a cached instance of the application settings.

    This is the primary way for components to access configuration,
    ensuring consistent state across the library.
    """
    return AppSettings()  # type: ignore[call-arg]
