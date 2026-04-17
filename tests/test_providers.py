from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel, SecretStr
from shared.config.config import LLMSettings
from shared.llm.providers.anthropic import AnthropicProvider
from shared.llm.providers.gemini import GeminiProvider
from shared.llm.providers.openai import OpenAIProvider
from shared.llm.types import ChatMessage, Role


class MockResponseModel(BaseModel):
    answer: str

@pytest.fixture
def llm_settings():
    return LLMSettings(
        openai_api_key=SecretStr("sk-test"),
        anthropic_api_key=SecretStr("ant-test"),
        gemini_api_key=SecretStr("goog-test"),
        default_openai_model="gpt-4o",
        default_anthropic_model="claude-3-5-sonnet",
        default_gemini_model="gemini-1.5-pro"
    )

@pytest.mark.anyio
async def test_openai_provider_chat(llm_settings):
    provider = OpenAIProvider(llm_settings)
    mock_client = MagicMock()
    provider._client = mock_client
    
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock(message=MagicMock(content="hello"))]
    mock_completion.usage = MagicMock(prompt_tokens=1, completion_tokens=1, total_tokens=2)
    mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)
    
    resp = await provider.chat([ChatMessage(role=Role.USER, content="hi")])
    assert resp.content == "hello"
    mock_client.chat.completions.create.assert_called_once()

@pytest.mark.anyio
async def test_openai_provider_chat_structured(llm_settings):
    provider = OpenAIProvider(llm_settings)
    mock_client = MagicMock()
    provider._client = mock_client
    
    mock_parse = MagicMock()
    mock_parse.choices = [MagicMock(message=MagicMock(parsed=MockResponseModel(answer="42")))]
    mock_client.beta.chat.completions.parse = AsyncMock(return_value=mock_parse)
    
    resp = await provider.chat_structured(
        [ChatMessage(role=Role.USER, content="hi")],
        response_model=MockResponseModel
    )
    assert resp.answer == "42"

@pytest.mark.anyio
async def test_anthropic_provider_chat(llm_settings):
    provider = AnthropicProvider(llm_settings)
    mock_client = MagicMock()
    provider._client = mock_client
    
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(type="text", text="hello")]
    mock_msg.usage = MagicMock(input_tokens=1, output_tokens=1)
    mock_client.messages.create = AsyncMock(return_value=mock_msg)
    
    resp = await provider.chat([ChatMessage(role=Role.USER, content="hi")])
    assert resp.content == "hello"

@pytest.mark.anyio
async def test_anthropic_provider_chat_structured(llm_settings):
    provider = AnthropicProvider(llm_settings)
    mock_client = MagicMock()
    provider._client = mock_client
    
    mock_msg = MagicMock()
    # Tool use block
    mock_block = MagicMock(type="tool_use", input={"answer": "42"})
    mock_block.name = "MockResponseModel" # Correctly set the name attribute
    mock_msg.content = [mock_block]
    mock_msg.usage = MagicMock(input_tokens=1, output_tokens=1)
    mock_client.messages.create = AsyncMock(return_value=mock_msg)
    
    resp = await provider.chat_structured(
        [ChatMessage(role=Role.USER, content="hi")],
        response_model=MockResponseModel
    )
    assert resp.answer == "42"

@pytest.mark.anyio
async def test_gemini_provider_chat(llm_settings):
    provider = GeminiProvider(llm_settings)
    mock_client = MagicMock()
    provider._client = mock_client
    
    mock_resp = MagicMock()
    mock_resp.text = "hello"
    mock_resp.usage_metadata = MagicMock(prompt_token_count=1, candidates_token_count=1, total_token_count=2)
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_resp)
    
    resp = await provider.chat([ChatMessage(role=Role.USER, content="hi")])
    assert resp.content == "hello"

@pytest.mark.anyio
async def test_gemini_provider_chat_structured(llm_settings):
    provider = GeminiProvider(llm_settings)
    mock_client = MagicMock()
    provider._client = mock_client
    
    mock_resp = MagicMock()
    mock_resp.text = '{"answer": "42"}'
    mock_resp.usage_metadata = MagicMock(prompt_token_count=1, candidates_token_count=1, total_token_count=2)
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_resp)
    
    resp = await provider.chat_structured(
        [ChatMessage(role=Role.USER, content="hi")],
        response_model=MockResponseModel
    )
    assert resp.answer == "42"

def test_openai_provider_client_missing_key():
    # Use model_construct to bypass validation if needed, or just provide other required keys
    settings = LLMSettings.model_construct(
        openai_api_key=None,
        gemini_api_key=SecretStr("goog-test")
    )
    provider = OpenAIProvider(settings)
    with pytest.raises(ValueError, match="OpenAI API key is missing"):
        _ = provider.client

def test_anthropic_provider_client_missing_key():
    settings = LLMSettings.model_construct(
        anthropic_api_key=None,
        gemini_api_key=SecretStr("goog-test")
    )
    provider = AnthropicProvider(settings)
    with pytest.raises(ValueError, match="Anthropic API key is missing"):
        _ = provider.client

def test_gemini_provider_client_missing_key():
    settings = LLMSettings.model_construct(gemini_api_key=None)
    provider = GeminiProvider(settings)
    with pytest.raises(ValueError, match="Gemini API key is missing"):
        _ = provider.client
