"""ROI sweep runner — sweep across sensor image plane."""
from __future__ import annotations

import logging

import numpy as np

from compass.core.types import SimulationResult
from compass.runners.single_run import SingleRunner

logger = logging.getLogger(__name__)

class ROISweepRunner:
    """Run simulations at different sensor ROI positions."""

    @staticmethod
    def run(config: dict, roi_config: dict) -> dict[str, SimulationResult]:
        """Execute ROI sweep. Each ROI has different CRA and microlens shift."""
        image_heights = roi_config.get("image_heights", [0.0])
        cra_table = roi_config.get("cra_table", [])
        results = {}
        for ih in image_heights:
            # Interpolate CRA for this image height
            cra = ROISweepRunner._interpolate_cra(ih, cra_table)
            # Modify config with CRA
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
            result = SingleRunner.run(cfg)
            results[f"ih_{ih:.2f}"] = result
        return results

    @staticmethod
    def _interpolate_cra(image_height: float, cra_table: list) -> float:
        if not cra_table:
            return 0.0
        ihs = [entry.get("image_height", 0) for entry in cra_table]
        cras = [entry.get("cra_deg", 0) for entry in cra_table]
        return float(np.interp(image_height, ihs, cras))
