"""
LLM Provider Implementations.

This package contains concrete implementations of the `LLMProvider` interface
for different vendors (Gemini, OpenAI, Anthropic). Each provider is responsible
for:
1. Mapping the unified `ChatMessage` format to the vendor-specific SDK format.
2. Executing the model call asynchronously.
3. Normalizing the response, including token usage and content.
4. Handling structured output generation (via native parsing or tool-calling).
"""

from .anthropic import AnthropicProvider
from .gemini import GeminiProvider
from .openai import OpenAIProvider

__all__ = ["AnthropicProvider", "GeminiProvider", "OpenAIProvider"]
