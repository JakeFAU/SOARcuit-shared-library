import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from shared.agents.basic import FastOpenAIAgent, ResearchOpenAIAgent
from shared.config.config import AppSettings, LLMSettings, LLMProvider as ProviderType, ModelNames
from pydantic import SecretStr

@pytest.fixture
def app_settings():
    return AppSettings(
        llm_settings=LLMSettings(
            openai_api_key=SecretStr("sk-test"),
            gemini_api_key=SecretStr("goog-test"),
            default_provider=ProviderType.OPENAI,
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

def test_fast_openai_agent_init(app_settings):
    with patch("shared.agents.basic.ChatService") as mock_chat_service:
        agent = FastOpenAIAgent(settings=app_settings)
        assert agent.name == "FastOpenAIAgent"
        assert agent.model == app_settings.model_names.openai.quick_model
        assert "duckduckgo_search" in agent.tools

def test_research_openai_agent_init(app_settings):
    with patch("shared.agents.basic.ChatService") as mock_chat_service:
        agent = ResearchOpenAIAgent(settings=app_settings)
        assert agent.name == "ResearchOpenAIAgent"
        assert agent.model == app_settings.model_names.openai.thinking_model
        assert "wikipedia_search" in agent.tools
        assert "duckduckgo_search" in agent.tools

@pytest.mark.anyio
async def test_agent_decide_integration(app_settings):
    # This tests the base Agent.decide using one of the basic agents
    from shared.llm.types import ChatMessage, Role, AgentIntent, ToolRequest
    
    with patch("shared.agents.basic.ChatService") as mock_chat_service_cls:
        mock_service = MagicMock()
        mock_chat_service_cls.return_value = mock_service
        
        expected_intent = AgentIntent(
            thought="I should search for Python.",
            plan="Search wikipedia",
            final_answer="Python is a language."
        )
        mock_service.chat_structured = AsyncMock(return_value=expected_intent)
        
        agent = FastOpenAIAgent(settings=app_settings)
        intent = await agent.decide([ChatMessage(role=Role.USER, content="What is Python?")])
        
        assert intent.final_answer == "Python is a language."
        mock_service.chat_structured.assert_called_once()
