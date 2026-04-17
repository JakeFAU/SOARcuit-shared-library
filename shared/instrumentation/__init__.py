"""
SOARcuit Instrumentation & Cost Measurement.

This package provides high-fidelity telemetry for async actions. It is designed
to track the realized marginal cost of execution episodes, allowing for
direct comparison against mVOI (marginal Value of Information) deltas.

Core Concepts:
1. Canonical Naming: Strict <kind>:<actor>:<verb>[:<variant>] format.
2. ActionRun: Represents a high-level execution episode (e.g., a research agent run).
3. ActionStepMeasurement: Represents a single measured unit of work (e.g., a tool call).
4. Profiler: An async-friendly context manager for capturing wall time, CPU, and memory.

This data flows from individual services into a central measurement store for
economic optimization.
"""

from .lifecycle import ActionRunManager
from .models import ActionRun, ActionStatus, ActionStepMeasurement
from .naming import CanonicalName, InvalidCanonicalNameError
from .profiler import Profiler, measure_step

__all__ = [
    "ActionRunManager",
    "ActionRun",
    "ActionStatus",
    "ActionStepMeasurement",
    "CanonicalName",
    "InvalidCanonicalNameError",
    "Profiler",
    "measure_step",
]
