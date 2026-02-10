"""Export simulation results to CSV, JSON."""
from __future__ import annotations
import json
import logging
from pathlib import Path
import numpy as np
from compass.core.types import SimulationResult

logger = logging.getLogger(__name__)

class ResultExporter:
    """Export results to various formats."""

    @staticmethod
    def to_csv(result: SimulationResult, filepath: str) -> None:
        """Export QE spectrum to CSV."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        wl = result.wavelengths
        pixels = sorted(result.qe_per_pixel.keys())
        header = "wavelength_um," + ",".join(f"QE_{p}" for p in pixels)
        data = np.column_stack([wl] + [result.qe_per_pixel[p] for p in pixels])
        np.savetxt(str(path), data, delimiter=",", header=header, comments="")
        logger.info(f"Exported QE to {filepath}")

    @staticmethod
    def to_json(result: SimulationResult, filepath: str) -> None:
        """Export result summary to JSON."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "wavelengths_um": result.wavelengths.tolist(),
            "qe_per_pixel": {k: v.tolist() for k, v in result.qe_per_pixel.items()},
            "metadata": {k: str(v) for k, v in result.metadata.items()},
        }
        if result.reflection is not None:
            summary["reflection"] = result.reflection.tolist()
        if result.transmission is not None:
            summary["transmission"] = result.transmission.tolist()
        with open(str(path), "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Exported summary to {filepath}")
