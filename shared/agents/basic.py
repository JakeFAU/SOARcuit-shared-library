from shared.config.config import AppSettings, get_settings
from shared.infrastructure.logging import get_logger
from shared.llm.agent import Agent
from shared.llm.client import ChatService
from shared.llm.tools.standard import DUCKDUCKGO_TOOL, WIKIPEDIA_TOOL
from shared.observability.tracer import get_tracer

tracer = get_tracer("agent")
logger = get_logger("agent")


class FastOpenAIAgent(Agent):
    """A high-speed OpenAI agent for quick, tool-augmented responses."""

    def __init__(self, settings: AppSettings | None = None):
        cfg = settings or get_settings()
        service = ChatService(settings=cfg.llm_settings)
        super().__init__(
            name="FastOpenAIAgent",
            llm_service=service,
            base_instructions="Answer the question using your knowledge and the tools available.",
            tools=[DUCKDUCKGO_TOOL],
            model=cfg.model_names.openai.quick_model,
        )


class ResearchOpenAIAgent(Agent):
    """A deep-reasoning OpenAI agent for comprehensive, verified research."""

    def __init__(self, settings: AppSettings | None = None):
        cfg = settings or get_settings()
        service = ChatService(settings=cfg.llm_settings)
        super().__init__(
            name="ResearchOpenAIAgent",
            llm_service=service,
            base_instructions="Answer the question using your knowledge and the tools available.",
            tools=[WIKIPEDIA_TOOL, DUCKDUCKGO_TOOL],
            model=cfg.model_names.openai.thinking_model,
        )
