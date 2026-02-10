#!/usr/bin/env python3
"""COMPASS simulation entry point using Hydra configuration.

Usage:
    python scripts/run_simulation.py
    python scripts/run_simulation.py solver=torcwa pixel=default_bsi_1um
    python scripts/run_simulation.py source=wavelength_sweep solver.params.fourier_order=[11,11]
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

    from compass.runners.single_run import SingleRunner

    result = SingleRunner.run(config)

    # Log results summary
    logger.info(f"Wavelengths: {result.wavelengths}")
    for pixel_name, qe in result.qe_per_pixel.items():
        logger.info(f"  {pixel_name}: QE mean={qe.mean():.4f}")

    if result.reflection is not None:
        logger.info(f"  R mean={result.reflection.mean():.4f}")
    if result.transmission is not None:
        logger.info(f"  T mean={result.transmission.mean():.4f}")
    if result.absorption is not None:
        logger.info(f"  A mean={result.absorption.mean():.4f}")

    # Save results
    output_dir = cfg.get("output_dir", "./outputs")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    from compass.io.export import ResultExporter

    ResultExporter.to_csv(result, f"{output_dir}/qe_spectrum.csv")
    ResultExporter.to_json(result, f"{output_dir}/result_summary.json")

    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
