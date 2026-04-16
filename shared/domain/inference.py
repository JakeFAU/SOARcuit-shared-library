"""Consolidated inference helpers for DSPy-based analysts."""

import sys
import types
from typing import Any

from structlog import get_logger

from shared.domain.config import SOARcuitBaseSettings

logger = get_logger(__name__)

try:
    import dspy
except ModuleNotFoundError:
    dspy = types.ModuleType("dspy")

    def _missing_dspy(*args: Any, **kwargs: Any) -> Any:
        raise ModuleNotFoundError("dspy is not installed.")

    dspy.LM = _missing_dspy  # type: ignore[attr-defined]
    dspy.configure = _missing_dspy  # type: ignore[attr-defined]
    sys.modules.setdefault("dspy", dspy)

# Global cache for cold-start optimization
_lm_initialized = False


def configure_lm(settings: SOARcuitBaseSettings | None = None) -> None:
    """Initialize DSPy using the environment variables or provided settings.

    Args:
        settings: Optional SOARcuitBaseSettings instance. If not provided,
                 it will attempt to load from environment/YAML.
    """
    global _lm_initialized
    if _lm_initialized:
        return

    if settings is None:
        try:
            settings = SOARcuitBaseSettings()  # type: ignore[call-arg]
        except Exception as e:
            logger.error("Failed to load settings for LM configuration", error=str(e))
            raise

    model_name = settings.model_name
    api_key = settings.google_genai_key.get_secret_value()

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
