"""Consolidated inference helpers for DSPy-based analysts."""

import os
from typing import Any

import dspy
from structlog import get_logger

logger = get_logger(__name__)

# Global cache for cold-start optimization
_lm_initialized = False


def configure_lm() -> None:
    """Initialize DSPy using the environment variables.

    Expects:
    - MODEL_NAME: The model identifier (e.g., 'gemini/gemini-3-flash-preview')
    - GOOGLE_GENAI_KEY: The API key for Google GenAI
    """
    global _lm_initialized
    if _lm_initialized:
        return

    model_name = os.getenv("MODEL_NAME", "gemini/gemini-3-flash-preview")
    api_key = os.getenv("GOOGLE_GENAI_KEY")

    if not api_key:
        logger.error("Missing GOOGLE_GENAI_KEY environment variable")
        raise ValueError("GOOGLE_GENAI_KEY not set")

    lm = dspy.LM(model=model_name, api_key=api_key)
    dspy.configure(lm=lm)
    _lm_initialized = True

    logger.info("DSPy LM configured", model_name=model_name)


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
        except (TypeError, ValueError):
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
