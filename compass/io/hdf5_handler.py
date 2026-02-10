"""HDF5 result storage and retrieval."""
from __future__ import annotations

import logging
from pathlib import Path

import yaml  # type: ignore[import-untyped]

from compass.core.types import SimulationResult

logger = logging.getLogger(__name__)

class HDF5Handler:
    """Save and load simulation results in HDF5 format."""

    @staticmethod
    def save_result(result: SimulationResult, config: dict, filepath: str, save_fields: bool = False, compression: str = "gzip", compression_level: int = 4) -> None:
        """Save simulation result to HDF5."""
        import h5py
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(str(path), "w") as f:
            meta = f.create_group("metadata")
            meta.attrs["config_yaml"] = yaml.dump(config)
            for k, v in result.metadata.items():
                try:
                    meta.attrs[k] = v
                except TypeError:
                    meta.attrs[k] = str(v)
            qe_grp = f.create_group("qe")
            qe_grp.create_dataset("wavelengths", data=result.wavelengths, compression=compression, compression_opts=compression_level)
            for name, qe in result.qe_per_pixel.items():
                qe_grp.create_dataset(f"pixel_{name}", data=qe, compression=compression, compression_opts=compression_level)
            eb_grp = f.create_group("energy_balance")
            if result.reflection is not None:
                eb_grp.create_dataset("reflection", data=result.reflection, compression=compression, compression_opts=compression_level)
            if result.transmission is not None:
                eb_grp.create_dataset("transmission", data=result.transmission, compression=compression, compression_opts=compression_level)
            if result.absorption is not None:
                eb_grp.create_dataset("absorption", data=result.absorption, compression=compression, compression_opts=compression_level)
            if save_fields and result.fields:
                fields_grp = f.create_group("fields")
                for wl_key, field_data in result.fields.items():
                    wl_grp = fields_grp.create_group(wl_key)
                    if field_data.Ex is not None:
                        wl_grp.create_dataset("Ex", data=field_data.Ex, compression=compression, compression_opts=compression_level)
                    if field_data.Ey is not None:
                        wl_grp.create_dataset("Ey", data=field_data.Ey, compression=compression, compression_opts=compression_level)
                    if field_data.Ez is not None:
                        wl_grp.create_dataset("Ez", data=field_data.Ez, compression=compression, compression_opts=compression_level)
        logger.info(f"Saved result to {filepath}")

    @staticmethod
    def load_result(filepath: str) -> tuple:
        """Load simulation result from HDF5. Returns (SimulationResult, config_dict)."""
        import h5py
        with h5py.File(filepath, "r") as f:
            config = yaml.safe_load(f["metadata"].attrs.get("config_yaml", "{}"))
            metadata = dict(f["metadata"].attrs)
            wavelengths = f["qe"]["wavelengths"][:]
            qe_per_pixel = {}
            for key in f["qe"]:
                if key.startswith("pixel_"):
                    pixel_name = key[6:]
                    qe_per_pixel[pixel_name] = f["qe"][key][:]
            reflection = f["energy_balance"]["reflection"][:] if "reflection" in f["energy_balance"] else None
            transmission = f["energy_balance"]["transmission"][:] if "transmission" in f["energy_balance"] else None
            absorption = f["energy_balance"]["absorption"][:] if "absorption" in f["energy_balance"] else None
            result = SimulationResult(
                qe_per_pixel=qe_per_pixel,
                wavelengths=wavelengths,
                reflection=reflection,
                transmission=transmission,
                absorption=absorption,
                metadata=metadata,
            )
        return result, config
