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
    Stateful Lifecycle Manager (The 'Heart').

    The Orchestrator drives the iterative reasoning loop by coordinating between
    the Agent (Thinker) and the ToolDispatcher (Actor). It maintains the
    conversation history and enforces execution constraints.

    Workflow:
    1. Initialize history with the user's input.
    2. Loop (up to max_iterations):
       a. Ask the Agent for an Intent (Thought + Plan + Actions).
       b. Append the Assistant's reasoning to the history.
       c. If Intent contains a Final Answer, return it and exit.
       d. Pass the requested Actions to the ToolDispatcher.
       e. Capture the Results and append them to the history as Observations.
    3. Return a failure message if the iteration limit is reached.

    Args:
        agent: The stateless Agent instance providing reasoning logic.
        dispatcher: The ToolDispatcher instance providing execution mechanics.
        max_iterations: Maximum number of Thought-Action-Observation cycles permitted.
    """

    def __init__(self, agent: Agent, dispatcher: ToolDispatcher, max_iterations: int = 5):
        self.agent = agent
        self.dispatcher = dispatcher
        self.max_iterations = max_iterations

    async def run(self, user_input: str) -> str:
        """
        Executes the full reasoning loop for a given user query.

        This method is the primary entry point for agentic execution. It is fully
        instrumented with OpenTelemetry spans, capturing every cycle and tool
        execution for downstream performance and cost analysis.
        """
        with tracer.start_as_current_span(f"orchestrator.{self.agent.name}.run") as span:
            span.set_attribute("agent.name", self.agent.name)
            span.set_attribute("user.input_length", len(user_input))

            logger.info("session_started", agent_name=self.agent.name)

            history: list[ChatMessage] = [ChatMessage(role=Role.USER, content=user_input)]

            for i in range(self.max_iterations):
                with tracer.start_as_current_span(f"reasoning_cycle.{i}") as cycle_span:
                    cycle_span.set_attribute("cycle.index", i)

                    # 1. Ask the Thinker
                    intent = await self.agent.decide(history)

                    cycle_span.set_attribute("reasoning.thought", intent.thought)
                    cycle_span.set_attribute("reasoning.plan", intent.plan)

                    logger.info(
                        "agent_decided",
                        iteration=i,
                        thought=intent.thought,
                        plan=intent.plan,
                        action_count=len(intent.actions),
                    )

                    # Record reasoning in history
                    history.append(
                        ChatMessage(
                            role=Role.ASSISTANT,
                            content=f"Thought: {intent.thought}\nPlan: {intent.plan}",
                        )
                    )

                    if intent.final_answer:
                        cycle_span.add_event("final_answer_reached")
                        span.set_attribute("session.status", "completed")
                        logger.info("session_completed", iteration=i)
                        return intent.final_answer

                    if not intent.actions:
                        cycle_span.add_event("early_termination_no_actions")
                        logger.warning("agent_stopped_no_intent", iteration=i)
                        break

                    # 2. Execute via Actor (ToolDispatcher)
                    cycle_span.add_event(
                        "tool_execution_started", {"tool_count": len(intent.actions)}
                    )
                    results = await self.dispatcher.execute(intent.actions)

                    # 3. Format observations and inject back into history
                    obs_lines = []
                    success_count = 0
                    for res in results:
                        status = "SUCCESS" if res.success else "ERROR"
                        if res.success:
                            success_count += 1
                        obs_lines.append(
                            f"Observation ({res.tool_name}) [{status}]: {res.output or res.error}"
                        )

                    cycle_span.set_attribute("cycle.success_rate", success_count / len(results))
                    history.append(ChatMessage(role=Role.USER, content="\n".join(obs_lines)))

            span.set_attribute("session.status", "max_iterations_reached")
            logger.error("session_failed_max_iterations", max_iterations=self.max_iterations)
            return "I'm sorry, I reached the limit of my reasoning capacity for this request."
