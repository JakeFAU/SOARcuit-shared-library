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
    Production-grade Agent implementation with Abstract Planning.
    Dynamically builds instructions from Tool definitions.
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
        """Constructs the system prompt using abstract planning and tool instructions."""
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
        """Asks the LLM for the next step (Thought -> Plan -> Actions/FinalAnswer)."""
        system_msg = ChatMessage(role=Role.SYSTEM, content=self._build_system_prompt())
        messages = [system_msg] + history
        return await self.llm_service.chat_structured(
            messages=messages,
            response_model=AgentIntent,
            model=self.model,
        )
