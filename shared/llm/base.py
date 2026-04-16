from abc import ABC, abstractmethod
from typing import Any

from .types import ChatMessage, ChatResponse, T


class LLMProvider(ABC):
    """Base interface for all LLM providers."""

    @abstractmethod
    async def chat(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """
        Sends a list of messages to the LLM and returns a unified response.
        """
        pass

    @abstractmethod
    async def chat_structured(
        self,
        messages: list[ChatMessage],
        response_model: type[T],
        model: str | None = None,
        **kwargs: Any,
    ) -> T:
        """
        Sends a list of messages to the LLM and returns a parsed Pydantic model.
        """
        pass
