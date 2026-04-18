"""Consolidated inference helpers for domain inference."""

from typing import Any

from structlog import get_logger

from shared.config.config import AppSettings, LLMProvider, LLMSettings, get_settings
from shared.domain.config import SOARcuitBaseSettings
from shared.llm.client import ChatService

logger = get_logger(__name__)

# Global cache for cold-start optimization
_lm_initialized = False
_chat_service: ChatService | None = None


def _resolve_llm_settings(
    settings: SOARcuitBaseSettings | LLMSettings | AppSettings | None,
) -> LLMSettings:
    if settings is None:
        return get_settings().llm_settings

    if isinstance(settings, AppSettings):
        return settings.llm_settings

    if isinstance(settings, LLMSettings):
        return settings

    return LLMSettings(gemini_api_key=settings.google_genai_key)


def configure_lm(
    settings: SOARcuitBaseSettings | LLMSettings | AppSettings | None = None,
) -> None:
    """Initialize and cache the shared Gemini chat client.

    Args:
        settings: Optional settings instance. If not provided, shared app
            settings are loaded and the configured Gemini default model is used.
    """
    global _lm_initialized
    global _chat_service

    if _lm_initialized:
        return

    llm_settings = _resolve_llm_settings(settings)
    service = ChatService(llm_settings)
    service._get_provider(LLMProvider.GEMINI)
    _chat_service = service
    _lm_initialized = True

    logger.info("LLM client configured", model_name=llm_settings.default_gemini_model)


def clean_observations(
    observations: list[Any],
    *,
    analyst_name: str,
    allowed_dimensions: set[str],
    default_dimension: str,
    max_count: int = 5,
) -> list[dict[str, Any]]:
    """Validate, normalize, and deduplicate model observations.

    Args:
        observations: Raw list of observations from the model.
        analyst_name: Name of the analyst (e.g., 'james', 'spock').
        allowed_dimensions: Set of valid dimension names for this analyst.
        default_dimension: Dimension to use if the model output is invalid.
        max_count: Maximum number of observations to keep.

    Returns:
        A list of cleaned and normalized observation dictionaries.
    """
    allowed_kinds = {"explicit_fact", "logical_inference"}

    cleaned: list[dict[str, Any]] = []
    seen_facts: set[str] = set()

    for raw in observations[:max_count]:
        if not isinstance(raw, dict):
            continue

        fact = str(raw.get("fact", "")).strip()
        kind = str(raw.get("kind", "logical_inference")).strip().lower()
        dimension = str(raw.get("dimension", default_dimension)).strip()
        evidence = str(raw.get("evidence", "")).strip() or fact

        if not fact:
            continue

        fact_key = fact.lower()
        if fact_key in seen_facts:
            continue

        try:
            probability = float(raw.get("probability", 0.5))
        except TypeError, ValueError:
            probability = 0.5

        probability = max(0.0, min(1.0, probability))

        if kind not in allowed_kinds:
            kind = "logical_inference"

        if dimension not in allowed_dimensions:
            dimension = default_dimension

        cleaned.append(
            {
                "fact": fact,
                "probability": probability,
                "kind": kind,
                "dimension": dimension,
                "evidence": evidence,
                "analyst": analyst_name,
            }
        )
        seen_facts.add(fact_key)

    return cleaned
