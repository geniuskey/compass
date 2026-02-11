"""ROI sweep runner — sweep across sensor image plane."""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from compass.core.types import SimulationResult
from compass.runners.single_run import SingleRunner

logger = logging.getLogger(__name__)

class ROISweepRunner:
    """Run simulations at different sensor ROI positions."""

    @staticmethod
    def _build_roi_config(config: dict, ih: float, cra: float) -> dict:
        """Build a modified config dict for a specific ROI position.

        Args:
            config: Base simulation configuration.
            ih: Image height.
            cra: Chief ray angle in degrees.

        Returns:
            Modified config dict with CRA and microlens shift applied.
        """
        cfg = dict(config)
        if "pixel" in cfg:
            pixel = dict(cfg["pixel"])
            layers = dict(pixel.get("layers", {}))
            ml = dict(layers.get("microlens", {}))
            shift = dict(ml.get("shift", {}))
            shift["mode"] = "auto_cra"
            shift["cra_deg"] = cra
            ml["shift"] = shift
            layers["microlens"] = ml
            pixel["layers"] = layers
            cfg["pixel"] = pixel
        if "source" in cfg:
            source = dict(cfg["source"])
            angle = dict(source.get("angle", {}))
            angle["theta_deg"] = cra
            source["angle"] = angle
            cfg["source"] = source
        return cfg

    @staticmethod
    def _run_single_roi(config: dict, ih: float, cra: float) -> tuple[str, SimulationResult]:
        """Run simulation for a single ROI position."""
        cfg = ROISweepRunner._build_roi_config(config, ih, cra)
        result = SingleRunner.run(cfg)
        return f"ih_{ih:.2f}", result

    @staticmethod
    def run(
        config: dict,
        roi_config: dict,
        max_workers: int | None = None,
    ) -> dict[str, SimulationResult]:
        """Execute ROI sweep. Each ROI has different CRA and microlens shift.

        Image heights are processed in parallel using ThreadPoolExecutor.

        Args:
            config: Base simulation configuration.
            roi_config: ROI sweep configuration with image_heights and cra_table.
            max_workers: Maximum parallel workers (default: number of image heights).

        Returns:
            Dict mapping ROI keys to SimulationResult instances.
        """
        image_heights = roi_config.get("image_heights", [0.0])
        cra_table = roi_config.get("cra_table", [])

        if max_workers is None:
            max_workers = len(image_heights)

        results: dict[str, SimulationResult] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ih = {}
            for ih in image_heights:
                cra = ROISweepRunner._interpolate_cra(ih, cra_table)
                future = executor.submit(
                    ROISweepRunner._run_single_roi, config, ih, cra
                )
                future_to_ih[future] = ih

            for future in as_completed(future_to_ih):
                key, result = future.result()
                results[key] = result

        return results

    @staticmethod
    def _interpolate_cra(image_height: float, cra_table: list) -> float:
        if not cra_table:
            return 0.0
        ihs = [entry.get("image_height", 0) for entry in cra_table]
        cras = [entry.get("cra_deg", 0) for entry in cra_table]
        return float(np.interp(image_height, ihs, cras))
