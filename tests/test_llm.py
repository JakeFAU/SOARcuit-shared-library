from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from shared.llm.client import ChatService
from shared.llm.providers import (
    AnthropicChatProvider,
    GeminiChatProvider,
    OpenAIChatProvider,
)
from shared.llm.types import ChatMessage, ChatResponse, StructuredResponse, Usage
from shared.registry.model_registry.registry import LLMModel, LLMProvider, ModelRegistry


class Recipe(BaseModel):
    name: str
    ingredients: list[str]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def llm_settings() -> MagicMock:
    settings = MagicMock()
    settings.gemini_enabled = True
    settings.open_ai_enabled = True
    settings.anthropic_enabled = True
    settings.gemini_api_key = MagicMock(get_secret_value=MagicMock(return_value="gk"))
    settings.open_ai_api_key = MagicMock(get_secret_value=MagicMock(return_value="ok"))
    settings.anthropic_api_key = MagicMock(
        get_secret_value=MagicMock(return_value="ak")
    )
    return settings


@pytest.fixture()
def model_registry(llm_settings: MagicMock) -> ModelRegistry:
    registry = ModelRegistry(llm_settings=llm_settings)
    registry.register(
        LLMModel(provider=LLMProvider.OPENAI, model_id="gpt-test"),
    )
    registry.register(
        LLMModel(provider=LLMProvider.GOOGLE, model_id="gemini-test"),
    )
    registry.register(
        LLMModel(provider=LLMProvider.ANTHROPIC, model_id="claude-test"),
    )
    return registry


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class TestChatMessage:
    def test_frozen(self) -> None:
        msg = ChatMessage(role="user", content="hi")
        with pytest.raises(AttributeError):
            msg.content = "bye"  # type: ignore[misc]

    def test_roles(self) -> None:
        assert ChatMessage(role="user", content="a").role == "user"
        assert ChatMessage(role="assistant", content="b").role == "assistant"


class TestChatResponse:
    def test_with_usage(self) -> None:
        r = ChatResponse(
            text="hi", model="m", usage=Usage(input_tokens=1, output_tokens=2)
        )
        assert r.usage is not None
        assert r.usage.input_tokens == 1

    def test_without_usage(self) -> None:
        r = ChatResponse(text="hi", model="m")
        assert r.usage is None


class TestStructuredResponse:
    def test_generic_data(self) -> None:
        recipe = Recipe(name="pasta", ingredients=["noodles"])
        r = StructuredResponse(data=recipe, model="m")
        assert r.data.name == "pasta"


# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------


class TestOpenAIChatProvider:
    @pytest.mark.anyio()
    async def test_chat(self) -> None:
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.model = "gpt-test"
        mock_response.choices = [MagicMock(message=MagicMock(content="hello back"))]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
        mock_client.chat.completions.create.return_value = mock_response

        provider = OpenAIChatProvider.__new__(OpenAIChatProvider)
        provider._client = mock_client

        result = await provider.chat(
            "gpt-test",
            [ChatMessage(role="user", content="hi")],
            system="be nice",
        )
        assert result.text == "hello back"
        assert result.model == "gpt-test"
        assert result.usage == Usage(input_tokens=10, output_tokens=5)

        call_kwargs = mock_client.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        assert messages[0] == {"role": "system", "content": "be nice"}
        assert messages[1] == {"role": "user", "content": "hi"}

    @pytest.mark.anyio()
    async def test_chat_structured(self) -> None:
        mock_client = AsyncMock()
        parsed_recipe = Recipe(name="pasta", ingredients=["flour", "water"])
        mock_response = MagicMock()
        mock_response.model = "gpt-test"
        mock_response.choices = [MagicMock(message=MagicMock(parsed=parsed_recipe))]
        mock_response.usage = MagicMock(prompt_tokens=20, completion_tokens=15)
        mock_client.beta.chat.completions.parse.return_value = mock_response

        provider = OpenAIChatProvider.__new__(OpenAIChatProvider)
        provider._client = mock_client

        result = await provider.chat_structured(
            "gpt-test",
            [ChatMessage(role="user", content="give me a recipe")],
            output_type=Recipe,
        )
        assert result.data.name == "pasta"
        assert result.data.ingredients == ["flour", "water"]


# ---------------------------------------------------------------------------
# Gemini provider
# ---------------------------------------------------------------------------


class TestGeminiChatProvider:
    @pytest.mark.anyio()
    async def test_chat(self) -> None:
        mock_response = MagicMock()
        mock_response.text = "gemini says hi"
        mock_response.usage_metadata = MagicMock(
            prompt_token_count=8, candidates_token_count=4
        )

        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        provider = GeminiChatProvider.__new__(GeminiChatProvider)
        provider._client = mock_client

        result = await provider.chat(
            "gemini-test",
            [ChatMessage(role="user", content="hi")],
        )
        assert result.text == "gemini says hi"
        assert result.usage == Usage(input_tokens=8, output_tokens=4)

    @pytest.mark.anyio()
    async def test_chat_structured(self) -> None:
        mock_response = MagicMock()
        mock_response.text = '{"name": "salad", "ingredients": ["lettuce"]}'
        mock_response.usage_metadata = MagicMock(
            prompt_token_count=10, candidates_token_count=8
        )

        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        provider = GeminiChatProvider.__new__(GeminiChatProvider)
        provider._client = mock_client

        result = await provider.chat_structured(
            "gemini-test",
            [ChatMessage(role="user", content="recipe")],
            output_type=Recipe,
        )
        assert result.data.name == "salad"


# ---------------------------------------------------------------------------
# Anthropic provider
# ---------------------------------------------------------------------------


class TestAnthropicChatProvider:
    @pytest.mark.anyio()
    async def test_chat(self) -> None:
        text_block = MagicMock(type="text", text="claude says hi")
        mock_response = MagicMock()
        mock_response.model = "claude-test"
        mock_response.content = [text_block]
        mock_response.usage = MagicMock(input_tokens=12, output_tokens=6)

        mock_client = AsyncMock()
        mock_client.messages.create.return_value = mock_response

        provider = AnthropicChatProvider.__new__(AnthropicChatProvider)
        provider._client = mock_client

        result = await provider.chat(
            "claude-test",
            [ChatMessage(role="user", content="hi")],
            system="be helpful",
        )
        assert result.text == "claude says hi"
        assert result.usage == Usage(input_tokens=12, output_tokens=6)

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["system"] == "be helpful"

    @pytest.mark.anyio()
    async def test_chat_structured(self) -> None:
        tool_block = MagicMock(
            type="tool_use",
            input={"name": "soup", "ingredients": ["water", "salt"]},
        )
        mock_response = MagicMock()
        mock_response.model = "claude-test"
        mock_response.content = [tool_block]
        mock_response.usage = MagicMock(input_tokens=15, output_tokens=10)

        mock_client = AsyncMock()
        mock_client.messages.create.return_value = mock_response

        provider = AnthropicChatProvider.__new__(AnthropicChatProvider)
        provider._client = mock_client

        result = await provider.chat_structured(
            "claude-test",
            [ChatMessage(role="user", content="recipe")],
            output_type=Recipe,
        )
        assert result.data.name == "soup"
        assert result.data.ingredients == ["water", "salt"]


# ---------------------------------------------------------------------------
# ChatService
# ---------------------------------------------------------------------------


class TestChatService:
    @pytest.mark.anyio()
    async def test_routes_to_correct_provider(
        self, model_registry: ModelRegistry
    ) -> None:
        service = ChatService(model_registry)

        mock_provider = MagicMock()
        expected = ChatResponse(text="ok", model="gpt-test")
        mock_provider.chat = AsyncMock(return_value=expected)
        service._providers[LLMProvider.OPENAI] = mock_provider

        result = await service.chat(
            "gpt-test",
            [ChatMessage(role="user", content="hi")],
        )
        assert result.text == "ok"
        mock_provider.chat.assert_awaited_once()

    @pytest.mark.anyio()
    async def test_routes_structured_to_correct_provider(
        self, model_registry: ModelRegistry
    ) -> None:
        service = ChatService(model_registry)

        recipe = Recipe(name="test", ingredients=[])
        mock_provider = MagicMock()
        expected = StructuredResponse(data=recipe, model="gemini-test")
        mock_provider.chat_structured = AsyncMock(return_value=expected)
        service._providers[LLMProvider.GOOGLE] = mock_provider

        result = await service.chat_structured(
            "gemini-test",
            [ChatMessage(role="user", content="recipe")],
            output_type=Recipe,
        )
        assert result.data.name == "test"

    def test_caches_providers(self, model_registry: ModelRegistry) -> None:
        service = ChatService(model_registry)
        with patch("openai.AsyncOpenAI", return_value=MagicMock()):
            p1 = service._get_provider(LLMProvider.OPENAI)
            p2 = service._get_provider(LLMProvider.OPENAI)
        assert p1 is p2
