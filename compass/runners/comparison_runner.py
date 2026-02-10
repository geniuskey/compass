"""Multi-solver comparison runner."""
from __future__ import annotations
import logging
from typing import Dict, List
from compass.core.types import SimulationResult
from compass.runners.single_run import SingleRunner
from compass.analysis.solver_comparison import SolverComparison

logger = logging.getLogger(__name__)

class ComparisonRunner:
    """Run same simulation with multiple solvers and compare."""

    @staticmethod
    def run(config: dict, solver_configs: List[dict]) -> dict:
        """Run comparison across solvers."""
        results = []
        labels = []
        for sc in solver_configs:
            cfg = {**config, "solver": sc}
            name = sc.get("name", sc.get("solver", "unknown"))
            logger.info(f"Running solver: {name}")
            result = SingleRunner.run(cfg)
            results.append(result)
            labels.append(name)
        comparison = SolverComparison(results, labels)
        return {
            "results": results,
            "labels": labels,
            "summary": comparison.summary(),
        }
