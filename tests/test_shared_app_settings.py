from __future__ import annotations

import pytest
from pydantic import ValidationError

from shared.config.config import (
    AppSettings,
    DatabaseAuthMode,
    DatabaseConnectionMode,
)


def test_app_settings_builds_nested_models_from_legacy_thalamus_env(
    monkeypatch,
) -> None:
    monkeypatch.setenv("THALAMUS_GCP_PROJECT", "soarcuit")
    monkeypatch.setenv("THALAMUS_DB_NAME", "memes")
    monkeypatch.setenv("THALAMUS_DB_USER", "postgres")
    monkeypatch.setenv("THALAMUS_DB_AUTH_MODE", "password")
    monkeypatch.setenv("THALAMUS_DB_PASSWORD", "secret")
    monkeypatch.setenv("THALAMUS_DB_CONNECTION_MODE", "unix_socket")
    monkeypatch.setenv("THALAMUS_DB_CLOUD_SQL_INSTANCE", "project:region:instance")
    monkeypatch.setenv("GOOGLE_GENAI_KEY", "test-key")
    monkeypatch.setenv("THALAMUS_EMBEDDING_MODEL", "gemini-embedding-001")
    monkeypatch.setenv("THALAMUS_EMBEDDING_DIMS", "3072")

    settings = AppSettings()

    assert settings.gcp_settings.project_id == "soarcuit"
    assert settings.gcp_settings.project_name == "soarcuit"
    assert settings.database_settings.database == "memes"
    assert settings.database_settings.user == "postgres"
    assert settings.database_settings.resolved_auth_mode == DatabaseAuthMode.PASSWORD
    assert settings.database_settings.resolved_connection_mode == DatabaseConnectionMode.UNIX_SOCKET
    assert settings.database_settings.cloud_sql_instance == "project:region:instance"
    assert settings.llm_settings.gemini_api_key.get_secret_value() == "test-key"
    assert settings.llm_settings.embedding_model == "gemini-embedding-001"
    assert settings.llm_settings.embedding_dimension == 3072
    assert settings.model_names.gemini.default_model == "gemini-3-flash-preview"


def test_app_settings_builds_nested_models_from_shared_soar_env(monkeypatch) -> None:
    monkeypatch.setenv("SOAR_GCP_SETTINGS__PROJECT_ID", "soarcuit")
    monkeypatch.setenv("SOAR_GCP_SETTINGS__PROJECT_NAME", "SOARcuit")
    monkeypatch.setenv("SOAR_DATABASE_SETTINGS__DATABASE", "memes")
    monkeypatch.setenv("SOAR_DATABASE_SETTINGS__USER", "postgres")
    monkeypatch.setenv("SOAR_LLM_SETTINGS__GEMINI_API_KEY", "test-key")

    settings = AppSettings()

    assert settings.gcp_settings.project_id == "soarcuit"
    assert settings.gcp_settings.project_name == "SOARcuit"
    assert settings.database_settings.database == "memes"
    assert settings.database_settings.user == "postgres"
    assert settings.llm_settings.gemini_api_key.get_secret_value() == "test-key"


def test_app_settings_still_requires_missing_shared_fields(monkeypatch) -> None:
    monkeypatch.delenv("THALAMUS_GCP_PROJECT", raising=False)
    monkeypatch.delenv("THALAMUS_DB_NAME", raising=False)
    monkeypatch.delenv("THALAMUS_DB_USER", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_KEY", raising=False)
    monkeypatch.delenv("SOAR_GCP_SETTINGS__PROJECT_ID", raising=False)
    monkeypatch.delenv("SOAR_DATABASE_SETTINGS__DATABASE", raising=False)
    monkeypatch.delenv("SOAR_DATABASE_SETTINGS__USER", raising=False)
    monkeypatch.delenv("SOAR_LLM_SETTINGS__GEMINI_API_KEY", raising=False)

    with pytest.raises(ValidationError):
        AppSettings()
