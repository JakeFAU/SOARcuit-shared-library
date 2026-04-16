from __future__ import annotations

from shared.infrastructure.logging import get_logger
from shared.observability.tracer import get_tracer

from .agent import Agent
from .dispatcher import ToolDispatcher
from .types import ChatMessage, Role

logger = get_logger("orchestrator")
tracer = get_tracer("orchestrator")


class SessionOrchestrator:
    """
    Stateful manager for the reasoning loop.
    Coordinates between the Agent (Thinker) and ToolDispatcher (Actor).
    """

    def __init__(self, agent: Agent, dispatcher: ToolDispatcher, max_iterations: int = 5):
        self.agent = agent
        self.dispatcher = dispatcher
        self.max_iterations = max_iterations

    async def run(self, user_input: str) -> str:
        """
        Drives the reasoning loop until a final answer is found or max turns reached.
        Uses OpenTelemetry to trace the entire session.
        """
        with tracer.start_as_current_span(f"orchestrator.{self.agent.name}.session") as span:
            span.set_attribute("agent.name", self.agent.name)

            history: list[ChatMessage] = [ChatMessage(role=Role.USER, content=user_input)]

            for i in range(self.max_iterations):
                with tracer.start_as_current_span(f"cycle.{i}"):
                    # 1. Ask the Thinker
                    intent = await self.agent.decide(history)

                    # Log the thought and plan
                    logger.info(
                        "Agent decided", iteration=i, thought=intent.thought, plan=intent.plan
                    )

                    # Record reasoning in history
                    history.append(
                        ChatMessage(
                            role=Role.ASSISTANT,
                            content=f"Thought: {intent.thought}\nPlan: {intent.plan}",
                        )
                    )

                    if intent.final_answer:
                        span.set_attribute("session.status", "completed")
                        return intent.final_answer

                    if not intent.actions:
                        logger.warning("Agent stopped without actions or final answer.")
                        break

                    # 2. Execute via Actor
                    results = await self.dispatcher.execute(intent.actions)

                    # 3. Format observations
                    obs_lines = []
                    for res in results:
                        status = "SUCCESS" if res.success else "ERROR"
                        obs_lines.append(
                            f"Observation ({res.tool_name}) [{status}]: {res.output or res.error}"
                        )

                    history.append(ChatMessage(role=Role.USER, content="\n".join(obs_lines)))

            span.set_attribute("session.status", "failed_max_iterations")
            return "I'm sorry, I reached the limit of my reasoning capacity for this request."
