from __future__ import annotations

from enum import StrEnum
from typing import Any, TypeVar

from pydantic import BaseModel, Field

from shared.instrumentation.models import ActionStepMeasurement

T = TypeVar("T", bound=BaseModel)


class Role(StrEnum):
    """Supported roles in a chat conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    """A single message in a chat conversation."""

    role: Role
    content: str


class TokenUsage(BaseModel):
    """Token consumption details."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatResponse(BaseModel):
    """Unified response from any LLM provider."""

    model_config = {"arbitrary_types_allowed": True}

    content: str
    usage: TokenUsage = Field(default_factory=TokenUsage)
    model: str
    measurement: ActionStepMeasurement | None = None


class ToolRequest(BaseModel):
    """Instruction from the LLM to call a specific tool."""

    tool_name: str
    arguments: dict[str, Any]


class AgentIntent(BaseModel):
    """The structured decision point for the agent at each step."""

    thought: str = Field(
        ..., description="Internal reasoning: What is the current state? What is missing?"
    )
    plan: str = Field(..., description="Abstract plan for the next steps to reach the goal.")
    actions: list[ToolRequest] = Field(
        default_factory=list, description="List of tools requested to gather information."
    )
    final_answer: str | None = Field(
        None, description="The final comprehensive response if the goal is achieved."
    )


class ToolResult(BaseModel):
    """Normalized result of a tool execution."""

    tool_name: str
    success: bool
    output: Any | None = None
    error: str | None = None
    latency_ms: float = 0.0
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    measurement: ActionStepMeasurement | None = None


ChatResponse.model_rebuild()
ToolResult.model_rebuild()
