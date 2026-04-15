from __future__ import annotations

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


class SOARcuitBaseSettings(BaseSettings):
    """Common settings for all SOARcuit services."""
    gcp_project: str = Field(
        default="soarcuit", 
        validation_alias=AliasChoices(
            "gcp_project", "GCP_PROJECT", 
            "THALAMUS_GCP_PROJECT", "HIPPOCAMPUS_GCP_PROJECT", "CORTEX_GCP_PROJECT"
        )
    )
    environment: str = Field(
        default="dev", 
        validation_alias=AliasChoices(
            "environment", "ENVIRONMENT",
            "THALAMUS_ENVIRONMENT", "HIPPOCAMPUS_ENVIRONMENT", "CORTEX_ENVIRONMENT"
        )
    )
    google_genai_key: SecretStr = Field(
        ..., 
        validation_alias=AliasChoices(
            "google_genai_key", "GOOGLE_GENAI_KEY",
            "THALAMUS_GOOGLE_GENAI_KEY", "HIPPOCAMPUS_GOOGLE_GENAI_KEY", "CORTEX_GOOGLE_GENAI_KEY",
            "GOOGLE_API_KEY", "THALAMUS_GOOGLE_API_KEY", "HIPPOCAMPUS_GOOGLE_API_KEY", "CORTEX_GOOGLE_API_KEY"
        )
    )
    model_name: str = Field(
        default="gemini/gemini-3-flash-preview", 
        validation_alias=AliasChoices(
            "model_name", "MODEL_NAME",
            "THALAMUS_MODEL_NAME", "HIPPOCAMPUS_MODEL_NAME", "CORTEX_MODEL_NAME"
        )
    )
    embedding_model: str = Field(
        default="gemini-embedding-001", 
        validation_alias=AliasChoices(
            "embedding_model", "EMBEDDING_MODEL",
            "THALAMUS_EMBEDDING_MODEL", "HIPPOCAMPUS_EMBEDDING_MODEL", "CORTEX_EMBEDDING_MODEL"
        )
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls, yaml_file="settings.yaml"),
            file_secret_settings,
        )
