"""
Unified Agent Framework.

This package implements the 'Thinking vs. Acting' architectural pattern.

- 'Thinking' is handled by the stateless Agent class, which uses Abstract Planning
  to decompose goals into structured intents.
- 'Acting' is handled by the ToolDispatcher, which manages concurrent execution,
  telemetry, and error normalization.
- The lifecycle is orchestrated by the SessionOrchestrator.

This separation ensures that reasoning logic is pure and highly testable, while
execution mechanics are robust and observable.
"""

from .agent import Agent
from .client import ChatService
from .dispatcher import ToolDispatcher
from .orchestrator import SessionOrchestrator
from .tools.base import BaseTool
from .tools.standard import (
    ARXIV_TOOL,
    DUCKDUCKGO_TOOL,
    GITHUB_REPO_TOOL,
    TAVILY_TOOL,
    WIKIPEDIA_TOOL,
)
from .types import (
    AgentIntent,
    ChatMessage,
    ChatResponse,
    Role,
    TokenUsage,
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
]
