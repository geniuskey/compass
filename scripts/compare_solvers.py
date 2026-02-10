#!/usr/bin/env python3
"""COMPASS solver comparison script.

Runs the same simulation with multiple solvers and generates a comparison report.

Usage:
    python scripts/compare_solvers.py
    python scripts/compare_solvers.py experiment=solver_comparison
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run solver comparison."""
    logger.info("COMPASS Solver Comparison")

    config = OmegaConf.to_container(cfg, resolve=True)

    experiment = config.get("experiment", {})
    solver_configs = experiment.get("solvers", [config.get("solver", {})])

    from compass.runners.comparison_runner import ComparisonRunner

    result = ComparisonRunner.run(config, solver_configs)

    logger.info("Comparison Summary:")
    summary = result["summary"]
    for key, value in summary.items():
        logger.info(f"  {key}: {value}")


if __name__ == "__main__":
    main()
