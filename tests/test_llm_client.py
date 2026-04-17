from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel, SecretStr
from shared.config.config import LLMProvider as ProviderType
from shared.config.config import LLMSettings
from shared.llm.client import ChatService
from shared.llm.types import ChatMessage, ChatResponse, Role, TokenUsage


class MockResponseModel(BaseModel):
    answer: str

@pytest.fixture
def llm_settings():
    return LLMSettings(
        openai_api_key=SecretStr("sk-test"),
        anthropic_api_key=SecretStr("ant-test"),
        gemini_api_key=SecretStr("goog-test"),
        default_provider=ProviderType.OPENAI,
        default_openai_model="gpt-4o"
    )

@pytest.mark.anyio
async def test_chat_service_resolve_provider(llm_settings):
    service = ChatService(llm_settings)
    
    # OpenAI
    p, m = service._resolve_provider("gpt-4")
    assert "OpenAIProvider" in str(type(p))
    assert m == "gpt-4"
    
    # Anthropic
    p, m = service._resolve_provider("claude-3")
    assert "AnthropicProvider" in str(type(p))
    assert m == "claude-3"
    
    # Gemini
    p, m = service._resolve_provider("gemini-pro")
    assert "GeminiProvider" in str(type(p))
    assert m == "gemini-pro"
    
    # Default
    p, m = service._resolve_provider(None)
    assert m == llm_settings.default_openai_model

@pytest.mark.anyio
async def test_chat_service_chat(llm_settings):
    service = ChatService(llm_settings)
    mock_provider = AsyncMock()
    mock_provider.chat.return_value = ChatResponse(
        content="hello",
        usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        model="gpt-4"
    )
    
    with patch.object(service, "_resolve_provider", return_value=(mock_provider, "gpt-4")):
        resp = await service.chat([ChatMessage(role=Role.USER, content="hi")])
        assert resp.content == "hello"
        mock_provider.chat.assert_called_once()

@pytest.mark.anyio
async def test_chat_service_chat_structured(llm_settings):
    service = ChatService(llm_settings)
    mock_provider = AsyncMock()
    mock_provider.chat_structured.return_value = MockResponseModel(answer="42")
    
    with patch.object(service, "_resolve_provider", return_value=(mock_provider, "gpt-4")):
        resp = await service.chat_structured(
            messages=[ChatMessage(role=Role.USER, content="meaning?")],
            response_model=MockResponseModel
        )
        assert resp.answer == "42"
        mock_provider.chat_structured.assert_called_once()
