import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from shared.agents.basic import QuickClassifier, ResearchAnalyst, StrategicPlanner, ResearchOpenAIAgent
from shared.config.config import AppSettings, LLMSettings, LLMProvider as ProviderType, ModelNames
from pydantic import SecretStr

@pytest.fixture
def app_settings():
    return AppSettings.model_construct(
        llm_settings=LLMSettings.model_construct(
            openai_api_key=SecretStr("sk-test"),
            gemini_api_key=SecretStr("goog-test"),
            default_provider=ProviderType.GEMINI,
        ),
        model_names=ModelNames()
    )

def test_quick_classifier_init(app_settings):
    agent = QuickClassifier(settings=app_settings)
    assert agent.name == "QuickClassifier"
    assert agent.model == app_settings.model_names.gemini.quick_model
    assert "duckduckgo_search" in agent.tools

def test_research_analyst_init(app_settings):
    agent = ResearchAnalyst(settings=app_settings)
    assert agent.name == "ResearchAnalyst"
    assert agent.model == app_settings.model_names.gemini.default_model
    assert "tavily_search" in agent.tools
    assert "arxiv_search" in agent.tools

def test_strategic_planner_init(app_settings):
    agent = StrategicPlanner(settings=app_settings)
    assert agent.name == "StrategicPlanner"
    assert agent.model == app_settings.model_names.gemini.thinking_model
    assert "github_repo_info" in agent.tools

def test_research_openai_agent_init(app_settings):
    agent = ResearchOpenAIAgent(settings=app_settings)
    assert agent.name == "ResearchOpenAIAgent"
    assert agent.model == app_settings.model_names.openai.thinking_model
    assert "wikipedia_search" in agent.tools
