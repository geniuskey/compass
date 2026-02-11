"""Multi-solver comparison runner."""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from compass.analysis.solver_comparison import SolverComparison
from compass.runners.single_run import SingleRunner

logger = logging.getLogger(__name__)

class ComparisonRunner:
    """Run same simulation with multiple solvers and compare."""

    @staticmethod
    def _run_single_solver(config: dict, solver_config: dict) -> tuple[str, object]:
        """Run a single solver and return (label, result)."""
        cfg = {**config, "solver": solver_config}
        name = solver_config.get("name", solver_config.get("solver", "unknown"))
        logger.info(f"Running solver: {name}")
        result = SingleRunner.run(cfg)
        return name, result

    @staticmethod
    def run(config: dict, solver_configs: list[dict], max_workers: int | None = None) -> dict:
        """Run comparison across solvers.

        Solvers are executed in parallel using ThreadPoolExecutor for I/O-bound
        workloads (e.g., solvers that release the GIL during C/CUDA computation).

        Args:
            config: Base simulation configuration.
            solver_configs: List of solver configuration dicts.
            max_workers: Maximum parallel workers (default: number of solvers).

        Returns:
            Dict with 'results', 'labels', and 'summary' keys.
        """
        if max_workers is None:
            max_workers = len(solver_configs)

        # Submit all solver runs in parallel
        labels_results: list[tuple[int, str, object]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {}
            for idx, sc in enumerate(solver_configs):
                future = executor.submit(
                    ComparisonRunner._run_single_solver, config, sc
                )
                future_to_idx[future] = idx

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                name, result = future.result()
                labels_results.append((idx, name, result))

        # Sort by original order to preserve deterministic output
        labels_results.sort(key=lambda x: x[0])
        results = [lr[2] for lr in labels_results]
        labels = [lr[1] for lr in labels_results]

        comparison = SolverComparison(results, labels)
        return {
            "results": results,
            "labels": labels,
            "summary": comparison.summary(),
        }
