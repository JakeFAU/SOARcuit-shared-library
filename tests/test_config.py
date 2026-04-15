import pytest
import os
from pydantic import ValidationError
from shared.domain.config import SOARcuitBaseSettings

def test_missing_required_key():
    # Ensure GOOGLE_GENAI_KEY is NOT in environment
    if "GOOGLE_GENAI_KEY" in os.environ:
        os.environ.pop("GOOGLE_GENAI_KEY")
    with pytest.raises(ValidationError):
        SOARcuitBaseSettings()

def test_default_values(monkeypatch):
    monkeypatch.setenv("GOOGLE_GENAI_KEY", "test-key")
    settings = SOARcuitBaseSettings()
    assert settings.gcp_project == "soarcuit"
    assert settings.environment == "dev"
    assert settings.google_genai_key.get_secret_value() == "test-key"
    assert settings.model_name == "gemini/gemini-3-flash-preview"
    assert settings.embedding_model == "gemini-embedding-001"

def test_env_override(monkeypatch):
    monkeypatch.setenv("GOOGLE_GENAI_KEY", "test-key")
    monkeypatch.setenv("GCP_PROJECT", "custom-project")
    monkeypatch.setenv("ENVIRONMENT", "prod")
    monkeypatch.setenv("MODEL_NAME", "custom-model")
    monkeypatch.setenv("EMBEDDING_MODEL", "custom-embedding")
    
    settings = SOARcuitBaseSettings()
    assert settings.gcp_project == "custom-project"
    assert settings.environment == "prod"
    assert settings.model_name == "custom-model"
    assert settings.embedding_model == "custom-embedding"

def test_yaml_override(monkeypatch, tmp_path):
    # Change CWD to tmp_path
    monkeypatch.chdir(tmp_path)
    
    # Ensure no environment variables interfere
    monkeypatch.delenv("GCP_PROJECT", raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    
    # Create settings.yaml
    yaml_content = """
GCP_PROJECT: yaml-project
ENVIRONMENT: staging
"""
    (tmp_path / "settings.yaml").write_text(yaml_content)
    
    monkeypatch.setenv("GOOGLE_GENAI_KEY", "test-key")
    
    settings = SOARcuitBaseSettings()
    assert settings.gcp_project == "yaml-project"
    assert settings.environment == "staging"
