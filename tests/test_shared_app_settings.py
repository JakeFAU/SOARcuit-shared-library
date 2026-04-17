from __future__ import annotations

import pytest
from pydantic import ValidationError

from shared.config.config import AppSettings


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
    assert settings.model_names.gemini.default_model == "gemini-3-flash-preview"


def test_app_settings_still_requires_missing_shared_fields(monkeypatch) -> None:
    monkeypatch.delenv("SOAR_GCP_SETTINGS__PROJECT_ID", raising=False)
    monkeypatch.delenv("SOAR_DATABASE_SETTINGS__DATABASE", raising=False)
    monkeypatch.delenv("SOAR_DATABASE_SETTINGS__USER", raising=False)
    monkeypatch.delenv("SOAR_LLM_SETTINGS__GEMINI_API_KEY", raising=False)

    with pytest.raises(ValidationError):
        AppSettings()
