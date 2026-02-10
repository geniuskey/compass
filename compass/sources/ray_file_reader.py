"""Zemax ray file reader."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

class RayFileReader:
    """Read CRA/MRA data from Zemax JSON or CSV files."""

    @staticmethod
    def read_json(filepath: str) -> dict:
        """Read Zemax JSON ray file."""
        with open(filepath) as f:
            data = json.load(f)
        return dict(data)

    @staticmethod
    def read_csv(filepath: str) -> dict:
        """Read CSV with columns: image_height, cra_deg, mra_deg, f_number."""
        data = np.loadtxt(filepath, delimiter=",", skiprows=1)
        return {
            "image_height": data[:, 0].tolist(),
            "cra_deg": data[:, 1].tolist(),
            "mra_deg": data[:, 2].tolist() if data.shape[1] > 2 else [],
            "f_number": data[:, 3].tolist() if data.shape[1] > 3 else [],
        }

    @staticmethod
    def read(filepath: str, format: str = "auto") -> dict:
        """Read ray file with auto-detected format."""
        path = Path(filepath)
        if format == "auto":
            format = "zemax_json" if path.suffix == ".json" else "csv"
        if format == "zemax_json":
            return RayFileReader.read_json(filepath)
        return RayFileReader.read_csv(filepath)
