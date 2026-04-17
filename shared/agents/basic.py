from __future__ import annotations

from shared.config.config import AppSettings, get_settings
from shared.infrastructure.logging import get_logger
from shared.llm import (
    ARXIV_TOOL,
    DUCKDUCKGO_TOOL,
    GITHUB_REPO_TOOL,
    TAVILY_TOOL,
    WIKIPEDIA_TOOL,
    Agent,
    ChatService,
)
from shared.observability.tracer import get_tracer

tracer = get_tracer("agent")
logger = get_logger("agent")


class QuickClassifier(Agent):
    """
    Tier 1: High-Speed / Low-Cost.
    Uses the configured Gemini quick_model.
    Best for: Tagging, routing, and basic fact-checking.
    """

    def __init__(self, settings: AppSettings | None = None):
        cfg = settings or get_settings()
        service = ChatService(settings=cfg.llm_settings)
        super().__init__(
            name="QuickClassifier",
            llm_service=service,
            base_instructions=(
                "You are a high-speed classification agent. Your goal is to analyze inbound "
                "data and apply precise metadata tags or route it to the correct downstream "
                "process. You prioritize speed and categorical accuracy."
            ),
            tools=[DUCKDUCKGO_TOOL],
            model=cfg.model_names.gemini.quick_model,
            max_iterations=2,
        )


class ResearchAnalyst(Agent):
    """
    Tier 2: Balanced / Multi-Step Reasoning.
    Uses the configured Gemini default_model.
    Best for: Cross-referencing sources, identifying patterns, and grounded summarization.
    """

    def __init__(self, settings: AppSettings | None = None):
        cfg = settings or get_settings()
        service = ChatService(settings=cfg.llm_settings)
        super().__init__(
            name="ResearchAnalyst",
            llm_service=service,
            base_instructions=(
                "You are a research analyst. Your goal is to synthesize information from "
                "multiple trusted sources (Wikipedia, ArXiv, Web) to provide grounded, "
                "high-confidence answers. You balance depth with execution efficiency."
            ),
            tools=[WIKIPEDIA_TOOL, TAVILY_TOOL, ARXIV_TOOL],
            model=cfg.model_names.gemini.default_model,
            max_iterations=5,
        )


class StrategicPlanner(Agent):
    """
    Tier 3: Deep-Reasoning / Planning.
    Uses the configured Gemini thinking_model.
    Best for: Complex multi-step planning, technical gap analysis, and logical strategy.
    """

    def __init__(self, settings: AppSettings | None = None):
        cfg = settings or get_settings()
        service = ChatService(settings=cfg.llm_settings)
        super().__init__(
            name="StrategicPlanner",
            llm_service=service,
            base_instructions=(
                "You are a senior strategic reasoning engine. Your objective is to resolve "
                "complex, ambiguous technical queries by decomposing them into structured "
                "research plans. You prioritize logical consistency and depth above all else."
            ),
            tools=[TAVILY_TOOL, ARXIV_TOOL, GITHUB_REPO_TOOL, WIKIPEDIA_TOOL],
            model=cfg.model_names.gemini.thinking_model,
            max_iterations=10,
        )


class ResearchOpenAIAgent(Agent):
    """Fallback / Specialty: Deep-reasoning OpenAI agent using configured thinking_model."""

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
