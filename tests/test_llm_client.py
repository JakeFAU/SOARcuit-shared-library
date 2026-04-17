from uuid import uuid4
from unittest.mock import patch

import pytest
from pydantic import BaseModel, SecretStr

from shared.config.config import LLMSettings, LLMProvider as ProviderType
from shared.llm.client import ChatService
from shared.llm.types import ChatMessage, ChatResponse, Role, TokenUsage

class MockResponseModel(BaseModel):
    answer: str


class OpenAIProvider:
    async def chat(self, *args, **kwargs):
        return ChatResponse(
            content="hello",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            model="gpt-4",
        )

    async def chat_structured(self, *args, **kwargs):
        return MockResponseModel(answer="42")

@pytest.fixture
def llm_settings():
    return LLMSettings.model_construct(
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
    assert m == "gpt-4o"

@pytest.mark.anyio
async def test_chat_service_chat(llm_settings):
    service = ChatService(llm_settings)
    mock_provider = OpenAIProvider()

    decision_id = uuid4()
    run_id = uuid4()

    with patch.object(service, "_resolve_provider", return_value=(mock_provider, "gpt-4")):
        resp = await service.chat(
            [ChatMessage(role=Role.USER, content="hi")],
            decision_id=decision_id,
            action_run_id=run_id,
        )
        assert resp.content == "hello"
        assert resp.measurement is not None
        assert resp.measurement.total_tokens == 15
        assert resp.measurement.decision_id == decision_id
        assert resp.measurement.step_name == "model:openai:chat"
        assert resp.measurement.finished_at is not None

@pytest.mark.anyio
async def test_chat_service_chat_structured(llm_settings):
    service = ChatService(llm_settings)
    mock_provider = OpenAIProvider()

    decision_id = uuid4()
    run_id = uuid4()

    with patch.object(service, "_resolve_provider", return_value=(mock_provider, "gpt-4")):
        resp = await service.chat_structured(
            messages=[ChatMessage(role=Role.USER, content="meaning?")],
            response_model=MockResponseModel,
            decision_id=decision_id,
            action_run_id=run_id,
        )
        assert resp.answer == "42"
        measurement = getattr(resp, "_measurement")
        assert measurement.step_name == "model:openai:structured_output"
        assert measurement.finished_at is not None
