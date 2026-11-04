#!/usr/bin/env python3
"""COMPASS simulation entry point using Hydra configuration.

Usage:
    python scripts/run_simulation.py
    python scripts/run_simulation.py solver=torcwa pixel=default_bsi_1um
    python scripts/run_simulation.py source=wavelength_sweep solver.params.fourier_order=[11,11]
    python scripts/run_simulation.py +experiment=solver_comparison   # also roi_sweep,
                                                                     # optimize_microlens
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run COMPASS simulation."""
    logger.info("COMPASS Simulation")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    config = OmegaConf.to_container(cfg, resolve=True)

    # Validate the composed config before running: typos and invalid values
    # fail fast here instead of being silently ignored downstream.
    from compass.core import CompassConfig

    CompassConfig.model_validate(config)

    from compass.io.export import ResultExporter
    from compass.runners.experiment_runner import run_experiment

    out = run_experiment(config)
    logger.info(f"Experiment kind: {out['kind']}")

    output_dir = Path(cfg.get("output_dir", "./outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    single = len(out["results"]) == 1 and out["kind"] == "single"
    for label, result in out["results"].items():
        logger.info(f"[{label}] wavelengths: {result.wavelengths}")
        for pixel_name, qe in result.qe_per_pixel.items():
            logger.info(f"  {pixel_name}: QE mean={qe.mean():.4f}")
        for name, arr in (("R", result.reflection), ("T", result.transmission),
                          ("A", result.absorption)):
            if arr is not None:
                logger.info(f"  {name} mean={arr.mean():.4f}")
        stem = output_dir if single else output_dir / label
        ResultExporter.to_csv(result, f"{stem}/qe_spectrum.csv")
        ResultExporter.to_json(result, f"{stem}/result_summary.json")

    extra = {k: v for k, v in out.items() if k in ("summary", "optimization")}
    if extra:
        import json

        with open(output_dir / f"{out['kind']}.json", "w") as f:
            json.dump(extra, f, indent=2, default=str)
        logger.info(f"{out['kind']}: {json.dumps(extra, default=str)[:2000]}")

    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
