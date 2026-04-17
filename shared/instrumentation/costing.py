from __future__ import annotations

from typing import Protocol, runtime_checkable

from .models import ActionStepMeasurement


@runtime_checkable
class CostEstimator(Protocol):
    """
    Economic Modeling Interface.

    The CostEstimator is responsible for enriching raw telemetry (wall time, tokens, etc.)
    with calculated internal and external costs.

    Separation of Concerns:
    - The Profiler captures what happened (observed data).
    - The CostEstimator decides what it cost (modeled data).

    This allows us to update pricing models or add complex internal compute
    cost heuristics without touching the instrumentation logic.
    """

    def estimate(self, measurement: ActionStepMeasurement) -> ActionStepMeasurement:
        """
        Enriches the measurement with cost estimates.

        Fields modified:
        - estimated_usd_cost (Direct external API costs)
        - estimated_compute_cost (Internal hardware/runtime costs)
        - estimated_total_cost (Combined internal + external)
        """
        ...


class NoOpCostEstimator:
    """Default implementation that performs no cost estimation (returns as-is)."""

    def estimate(self, measurement: ActionStepMeasurement) -> ActionStepMeasurement:
        return measurement
