from typing import Any

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam

from ...config.config import LLMSettings
from ..base import LLMProvider
from ..types import ChatMessage, ChatResponse, Role, T, TokenUsage


class AnthropicProvider(LLMProvider):
    """
    Anthropic provider implementation.
    Handles SDK client as a singleton per provider instance.
    """

    def __init__(self, settings: LLMSettings):
        self._settings = settings
        self._client: AsyncAnthropic | None = None
        self._default_model = settings.default_anthropic_model

    @property
    def client(self) -> AsyncAnthropic:
        if self._client is None:
            if not self._settings.anthropic_api_key:
                raise ValueError("Anthropic API key is missing in configuration.")
            self._client = AsyncAnthropic(
                api_key=self._settings.anthropic_api_key.get_secret_value()
            )
        return self._client

    def _prepare_messages(
        self, messages: list[ChatMessage]
    ) -> tuple[str | None, list[MessageParam]]:
        system_prompt = None
        anthropic_msgs: list[MessageParam] = []
        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_prompt = msg.content
            elif msg.role == Role.USER:
                anthropic_msgs.append({"role": "user", "content": msg.content})
            elif msg.role == Role.ASSISTANT:
                anthropic_msgs.append({"role": "assistant", "content": msg.content})
        return system_prompt, anthropic_msgs

    async def chat(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        model = model or self._default_model
        system, formatted_msgs = self._prepare_messages(messages)

        # We ensure system is passed only if it's not None
        create_kwargs: dict[str, Any] = {
            "model": model,
            "messages": formatted_msgs,
            "max_tokens": kwargs.pop("max_tokens", 4096),
            **kwargs,
        }
        if system:
            create_kwargs["system"] = system

        response = await self.client.messages.create(**create_kwargs)

        usage = TokenUsage(
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
        )

        content = "".join(block.text for block in response.content if block.type == "text")

        return ChatResponse(
            content=content,
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
        system, formatted_msgs = self._prepare_messages(messages)

        tool_name = response_model.__name__
        tool_schema = {
            "name": tool_name,
            "description": f"Returns a {tool_name} object.",
            "input_schema": response_model.model_json_schema(),
        }

        create_kwargs: dict[str, Any] = {
            "model": model,
            "messages": formatted_msgs,
            "tools": [tool_schema],
            "tool_choice": {"type": "tool", "name": tool_name},
            "max_tokens": kwargs.pop("max_tokens", 4096),
            **kwargs,
        }
        if system:
            create_kwargs["system"] = system

        response = await self.client.messages.create(**create_kwargs)

        for block in response.content:
            if block.type == "tool_use" and block.name == tool_name:
                return response_model.model_validate(block.input)

        raise ValueError(f"Anthropic failed to return tool_use for {tool_name} with model {model}")
