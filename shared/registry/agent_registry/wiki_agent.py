from shared.llm.agent import Agent
from shared.llm.client import ChatService
from shared.llm.tools.standard import WIKIPEDIA_TOOL


def create_wikipedia_agent(llm_service: ChatService) -> Agent:
    """
    Creates a reasoning-driven Research Agent.

    The agent uses Gemini 3.0 Flash Preview and Abstract Planning to
    decompose complex queries into Wikipedia search strategies.
    """
    base_instructions = (
        "You are the SOARcuit Research Lead. Your objective is to provide "
        "comprehensive, verified answers to user queries using the available "
        "knowledge tools. You operate with extreme technical precision."
    )

    return Agent(
        name="wikipedia_researcher",
        base_instructions=base_instructions,
        llm_service=llm_service,
        tools=[WIKIPEDIA_TOOL],
        model="gemini-3.0-flash-preview",
        max_iterations=5,
    )
