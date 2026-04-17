"""
SOARcuit Shared Library.

This library provides the foundational architecture for the SOARcuit event-driven
LLM memory system. It enforces a strict "Thinking vs. Acting" separation for agents,
a unified interface for multi-turn chat with structured outputs, and a comprehensive
instrumentation layer for economic optimization (mVOI vs. MC).

Key Packages:
- shared.llm: Unified LLM client and Agent framework.
- shared.instrumentation: High-fidelity telemetry and cost measurement.
- shared.config: Config-driven environment management.
- shared.domain: Canonical data models for memes and observations.
- shared.messaging: Pub/Sub normalization and envelope handling.
"""

from .infrastructure.logging import configure_logging
from .llm import (
    ARXIV_TOOL,
    DUCKDUCKGO_TOOL,
    GITHUB_REPO_TOOL,
    TAVILY_TOOL,
    WIKIPEDIA_TOOL,
    Agent,
    AgentIntent,
    BaseTool,
    ChatMessage,
    ChatResponse,
    ChatService,
    Role,
    SessionOrchestrator,
    TokenUsage,
    ToolDispatcher,
    ToolRequest,
    ToolResult,
)

__all__ = [
    "Agent",
    "AgentIntent",
    "ChatService",
    "SessionOrchestrator",
    "ToolDispatcher",
    "BaseTool",
    "WIKIPEDIA_TOOL",
    "DUCKDUCKGO_TOOL",
    "TAVILY_TOOL",
    "ARXIV_TOOL",
    "GITHUB_REPO_TOOL",
    "ChatMessage",
    "ChatResponse",
    "Role",
    "TokenUsage",
    "ToolRequest",
    "ToolResult",
    "configure_logging",
]
