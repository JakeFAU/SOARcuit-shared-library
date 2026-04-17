import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from shared.agents.basic import QuickClassifier, ResearchAnalyst
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
        database_settings={
            "database": "test",
            "user": "test"
        },
        gcp_settings={
            "project_id": "test",
            "project_name": "test"
        },
        model_names=ModelNames()
    )

def test_quick_classifier_agent_init(app_settings):
    with patch("shared.agents.basic.ChatService") as mock_chat_service:
        agent = QuickClassifier(settings=app_settings)
        assert agent.name == "QuickClassifier"
        assert agent.model == app_settings.model_names.gemini.quick_model
        assert "duckduckgo_search" in agent.tools

def test_research_analyst_agent_init(app_settings):
    with patch("shared.agents.basic.ChatService") as mock_chat_service:
        agent = ResearchAnalyst(settings=app_settings)
        assert agent.name == "ResearchAnalyst"
        assert agent.model == app_settings.model_names.gemini.default_model
        assert "wikipedia_search" in agent.tools
        assert "tavily_search" in agent.tools

@pytest.mark.anyio
async def test_agent_decide_integration(app_settings):
    from shared.llm.types import ChatMessage, Role, AgentIntent
    
    with patch("shared.agents.basic.ChatService") as mock_chat_service_cls:
        mock_service = MagicMock()
        mock_chat_service_cls.return_value = mock_service
        
        expected_intent = AgentIntent(
            thought="I should search for Python.",
            plan="Search wikipedia",
            final_answer="Python is a language."
        )
        mock_service.chat_structured = AsyncMock(return_value=expected_intent)
        
        agent = QuickClassifier(settings=app_settings)
        intent = await agent.decide([ChatMessage(role=Role.USER, content="What is Python?")])
        
        assert intent.final_answer == "Python is a language."
        mock_service.chat_structured.assert_called_once()
