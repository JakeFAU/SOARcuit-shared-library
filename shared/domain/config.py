from __future__ import annotations
from typing import Any, Type
from pydantic import SecretStr, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

class SOARcuitBaseSettings(BaseSettings):
    """Common settings for all SOARcuit services."""
    gcp_project: str = Field(default="soarcuit", validation_alias="GCP_PROJECT")
    environment: str = Field(default="dev", validation_alias="ENVIRONMENT")
    google_genai_key: SecretStr = Field(..., validation_alias="GOOGLE_GENAI_KEY")
    model_name: str = Field(default="gemini/gemini-3-flash-preview", validation_alias="MODEL_NAME")
    embedding_model: str = Field(default="gemini-embedding-001", validation_alias="EMBEDDING_MODEL")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
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
