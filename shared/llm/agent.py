from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from shared.infrastructure.logging import get_logger
from shared.observability.tracer import get_tracer

from .client import ChatService
from .tools.base import BaseTool
from .types import AgentIntent, ChatMessage, Role

logger = get_logger("agent")
tracer = get_tracer("agent")

T = TypeVar("T", bound=BaseModel)


class Agent:
    """
    Stateless Reasoning Engine (The 'Thinker').

    The Agent is responsible for decomposing a high-level user goal into an
    abstract plan and a sequence of tool requests. It operates as a pure function
    of (History, ToolContracts) -> Intent.

    Design Principles:
    1. Statelessness: The Agent does not maintain conversation history internally;
       it is passed the full history at every decision point.
    2. Abstract Planning: Every decision step forces the LLM to provide a 'thought'
       and a 'plan' before requesting actions, ensuring strategic alignment.
    3. Dynamic Discovery: Tools are described to the LLM using their
       `to_instruction()` method, ensuring zero-config tool integration.

    Args:
        name: Unique identifier for the agent (used in telemetry).
        base_instructions: The persona and primary directive of the agent.
        llm_service: The ChatService used for model interactions.
        tools: List of BaseTool instances the agent is authorized to use.
        max_iterations: Safety guard to prevent infinite reasoning loops.
        model: Specific model override (e.g., 'gemini-3.1-pro-preview').
    """

    def __init__(
        self,
        name: str,
        base_instructions: str,
        llm_service: ChatService,
        tools: list[BaseTool] | None = None,
        max_iterations: int = 5,
        model: str | None = None,
    ):
        self.name = name
        self.base_instructions = base_instructions
        self.llm_service = llm_service
        self.tools = {t.name: t for t in (tools or [])}
        self.max_iterations = max_iterations
        self.model = model

    def _build_system_prompt(self) -> str:
        """
        Dynamically assembles the system prompt.

        Includes the base persona, the operational protocol (Analyze -> Plan -> Request -> Converge),
        and the auto-generated tool contracts from the provided toolset.
        """
        tool_instructions = "\n".join(t.to_instruction() for t in self.tools.values())

        return (
            f"{self.base_instructions}\n\n"
            "## OPERATIONAL PROTOCOL\n"
            "1. ANALYZE: Review the user request and available information.\n"
            "2. PLAN: Update your abstract plan. Identify which tools can provide missing data.\n"
            "3. REQUEST: List any tool calls required to gather information for the current plan step.\n"
            "4. CONVERGE: Synthesize observations into a final answer once the plan is complete.\n\n"
            "## AVAILABLE TOOLS\n"
            f"{tool_instructions}\n"
            "## CONSTRAINTS\n"
            "- Be concise and technically precise.\n"
            "- If you need data from multiple sources, request them in the same turn.\n"
            "- Never hallucinate. Use 'final_answer' only when confident."
        )

    async def decide(self, history: list[ChatMessage]) -> AgentIntent:
        """
        Invokes the LLM to determine the next reasoning step.

        Takes the current conversation history and returns an AgentIntent containing
        the internal thought process, the revised plan, and any requested actions.
        """
        system_msg = ChatMessage(role=Role.SYSTEM, content=self._build_system_prompt())
        messages = [system_msg] + history
        return await self.llm_service.chat_structured(
            messages=messages,
            response_model=AgentIntent,
            model=self.model,
        )
