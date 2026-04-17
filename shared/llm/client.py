from typing import Any

from shared.infrastructure.logging import get_logger
from shared.observability.tracer import get_tracer

from ..config.config import LLMProvider as ProviderType
from ..config.config import LLMSettings
from .base import LLMProvider
from .providers.anthropic import AnthropicProvider
from .providers.gemini import GeminiProvider
from .providers.openai import OpenAIProvider
from .types import ChatMessage, ChatResponse, T

logger = get_logger("llm_client")
tracer = get_tracer("llm_client")


class ChatService:
    """
    High-level service for interacting with LLMs.
    Singletons for each provider are lazily instantiated and cached.
    """

    def __init__(self, settings: LLMSettings):
        self._settings = settings
        # Cache for provider instances (Singletons per ChatService)
        self._providers: dict[ProviderType, LLMProvider] = {}

    def _get_provider(self, provider_type: ProviderType) -> LLMProvider:
        """Lazily instantiates and returns a cached provider."""
        if provider_type not in self._providers:
            match provider_type:
                case ProviderType.OPENAI:
                    self._providers[provider_type] = OpenAIProvider(self._settings)
                case ProviderType.GEMINI:
                    self._providers[provider_type] = GeminiProvider(self._settings)
                case ProviderType.ANTHROPIC:
                    self._providers[provider_type] = AnthropicProvider(self._settings)
                case _:
                    raise ValueError(f"Unsupported provider type: {provider_type}")
        return self._providers[provider_type]

    def _resolve_provider(self, model: str | None) -> tuple[LLMProvider, str | None]:
        """
        Resolves the appropriate provider based on the model name.
        If no model is provided, defaults to the configured default provider.
        """
        if model is None:
            provider_type = self._settings.default_provider
            return self._get_provider(provider_type), self._settings.default_model

        # Simple heuristic to resolve provider from model name
        if any(model.startswith(p) for p in ("gpt", "o1", "o3")):
            return self._get_provider(ProviderType.OPENAI), model
        if model.startswith("claude"):
            return self._get_provider(ProviderType.ANTHROPIC), model
        if model.startswith("gemini"):
            return self._get_provider(ProviderType.GEMINI), model

        # Fallback to default provider with the specific model requested
        return self._get_provider(self._settings.default_provider), model

    async def chat(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """
        Unified chat interface. Respects configuration defaults.
        """
        provider, resolved_model = self._resolve_provider(model)

        with tracer.start_as_current_span("llm.chat") as span:
            span.set_attribute("llm.model", resolved_model or "default")
            span.set_attribute("llm.message_count", len(messages))

            logger.info("llm_chat_started", model=resolved_model)
            response = await provider.chat(messages, model=resolved_model, **kwargs)

            span.set_attribute("llm.usage.prompt_tokens", response.usage.prompt_tokens)
            span.set_attribute("llm.usage.completion_tokens", response.usage.completion_tokens)
            logger.info(
                "llm_chat_completed", model=response.model, tokens=response.usage.total_tokens
            )

            return response

    async def chat_structured(
        self,
        messages: list[ChatMessage],
        response_model: type[T],
        model: str | None = None,
        **kwargs: Any,
    ) -> T:
        """
        Unified structured output interface. Respects configuration defaults.
        """
        provider, resolved_model = self._resolve_provider(model)

        with tracer.start_as_current_span("llm.chat_structured") as span:
            span.set_attribute("llm.model", resolved_model or "default")
            span.set_attribute("llm.response_model", response_model.__name__)

            logger.info(
                "llm_structured_chat_started",
                model=resolved_model,
                target_model=response_model.__name__,
            )
            response = await provider.chat_structured(
                messages, response_model=response_model, model=resolved_model, **kwargs
            )
            logger.info("llm_structured_chat_completed")

            return response
