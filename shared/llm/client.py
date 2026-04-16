from typing import Any

from ..config.config import LLMProvider as ProviderType
from ..config.config import LLMSettings
from .base import LLMProvider
from .providers.anthropic import AnthropicProvider
from .providers.gemini import GeminiProvider
from .providers.openai import OpenAIProvider
from .types import ChatMessage, ChatResponse, T


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
        return await provider.chat(messages, model=resolved_model, **kwargs)

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
        return await provider.chat_structured(
            messages, response_model=response_model, model=resolved_model, **kwargs
        )
