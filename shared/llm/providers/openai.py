from typing import Any

import openai
from openai.types.chat import ChatCompletionMessageParam

from ...config.config import LLMSettings
from ..base import LLMProvider
from ..types import ChatMessage, ChatResponse, Role, T, TokenUsage


class OpenAIProvider(LLMProvider):
    """
    OpenAI provider implementation.
    Handles SDK client as a singleton per provider instance.
    """

    def __init__(self, settings: LLMSettings):
        self._settings = settings
        self._client: openai.AsyncOpenAI | None = None
        self._default_model = settings.default_openai_model

    @property
    def client(self) -> openai.AsyncOpenAI:
        if self._client is None:
            if not self._settings.openai_api_key:
                raise ValueError("OpenAI API key is missing in configuration.")
            self._client = openai.AsyncOpenAI(
                api_key=self._settings.openai_api_key.get_secret_value()
            )
        return self._client

    def _convert_messages(self, messages: list[ChatMessage]) -> list[ChatCompletionMessageParam]:
        formatted: list[ChatCompletionMessageParam] = []
        for m in messages:
            if m.role == Role.SYSTEM:
                formatted.append({"role": "system", "content": m.content})
            elif m.role == Role.USER:
                formatted.append({"role": "user", "content": m.content})
            elif m.role == Role.ASSISTANT:
                formatted.append({"role": "assistant", "content": m.content})
        return formatted

    async def chat(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        model = model or self._default_model

        response = await self.client.chat.completions.create(
            model=model,
            messages=self._convert_messages(messages),
            **kwargs,
        )

        usage = TokenUsage(
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
            total_tokens=response.usage.total_tokens if response.usage else 0,
        )

        return ChatResponse(
            content=response.choices[0].message.content or "",
            usage=usage,
            model=model,
        )

    async def chat_structured(
        self,
        messages: list[ChatMessage],
        response_model: type[T],
        model: str | None = None,
        **kwargs: Any,
    ) -> T:
        model = model or self._default_model

        response = await self.client.beta.chat.completions.parse(
            model=model,
            messages=self._convert_messages(messages),
            response_format=response_model,
            **kwargs,
        )

        parsed = response.choices[0].message.parsed
        if parsed is None:
            # Fallback or error handling for parsing failure
            raise ValueError(f"OpenAI failed to parse structured output for model {model}")

        return parsed
