"""HDF5 result storage and retrieval.

Layout (format version 2)::

    /                         attrs: format_version, config_json
    /metadata                 attrs: scalar metadata (str / int / float / bool)
    /metadata_json/<key>      JSON string dataset for non-scalar metadata
    /qe/wavelengths
    /qe/pixel_<name>
    /energy_balance/{reflection,transmission,absorption}
    /fields/<wl_key>/{Ex,Ey,Ez,x,y,z}      (save_fields=True)
    /poynting/<key>                         (save_fields=True)

HDF5 object names cannot contain "/", so names are percent-escaped.
Version-1 files (config as YAML in /metadata attrs) are still readable.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import yaml  # type: ignore[import-untyped]

from compass.core.types import FieldData, SimulationResult

logger = logging.getLogger(__name__)

FORMAT_VERSION = 2
_FIELD_KEYS = ("Ex", "Ey", "Ez", "x", "y", "z")


def _escape(name: str) -> str:
    return name.replace("%", "%25").replace("/", "%2F")


def _unescape(name: str) -> str:
    return name.replace("%2F", "/").replace("%25", "%")


def _jsonable(obj: Any) -> Any:
    """json.dumps default: numpy scalars/arrays and other objects."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        if np.iscomplexobj(obj):
            return {"__complex__": True, "real": obj.real.tolist(), "imag": obj.imag.tolist()}
        return obj.tolist()
    if isinstance(obj, (set, tuple)):
        return list(obj)
    if isinstance(obj, Path):
        return str(obj)
    return repr(obj)


def _is_scalar_attr(v: Any) -> bool:
    return isinstance(v, (str, bool, int, float, np.bool_, np.integer, np.floating))


class HDF5Handler:
    """Save and load simulation results in HDF5 format."""

    @staticmethod
    def save_result(
        result: SimulationResult,
        config: dict,
        filepath: str,
        save_fields: bool = False,
        compression: str = "gzip",
        compression_level: int = 4,
    ) -> None:
        """Save a simulation result (and the config that produced it) to HDF5."""
        import h5py

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        opts = {"compression": compression, "compression_opts": compression_level}

        def dataset(group, name: str, data) -> None:
            arr = np.asarray(data)
            group.create_dataset(_escape(name), data=arr, **(opts if arr.ndim else {}))

        with h5py.File(str(path), "w") as f:
            f.attrs["format_version"] = FORMAT_VERSION
            f.attrs["config_json"] = json.dumps(config, default=_jsonable)

            meta = f.create_group("metadata")
            meta_json = f.create_group("metadata_json")
            for k, v in result.metadata.items():
                if _is_scalar_attr(v):
                    meta.attrs[k] = v
                else:
                    # Datasets avoid the 64 KiB attribute limit for large values.
                    meta_json.create_dataset(_escape(k), data=json.dumps(v, default=_jsonable))

            qe_grp = f.create_group("qe")
            dataset(qe_grp, "wavelengths", result.wavelengths)
            for name, qe in result.qe_per_pixel.items():
                dataset(qe_grp, f"pixel_{name}", qe)

            eb_grp = f.create_group("energy_balance")
            for key in ("reflection", "transmission", "absorption"):
                arr = getattr(result, key)
                if arr is not None:
                    dataset(eb_grp, key, arr)

            if save_fields and result.fields:
                fields_grp = f.create_group("fields")
                for wl_key, field_data in result.fields.items():
                    wl_grp = fields_grp.create_group(_escape(wl_key))
                    for comp in _FIELD_KEYS:
                        arr = getattr(field_data, comp)
                        if arr is not None:
                            dataset(wl_grp, comp, arr)
            if save_fields and result.poynting:
                p_grp = f.create_group("poynting")
                for key, arr in result.poynting.items():
                    dataset(p_grp, key, arr)
        logger.info(f"Saved result to {filepath}")

    @staticmethod
    def load_result(filepath: str) -> tuple[SimulationResult, dict]:
        """Load a simulation result from HDF5. Returns (SimulationResult, config)."""
        import h5py

        with h5py.File(filepath, "r") as f:
            meta_attrs = f["metadata"].attrs
            if "config_json" in f.attrs:
                config = json.loads(f.attrs["config_json"])
                metadata = {k: _attr_value(meta_attrs[k]) for k in meta_attrs}
            else:  # format version 1
                config = _load_legacy_yaml(meta_attrs.get("config_yaml", "{}"))
                metadata = {
                    k: _attr_value(meta_attrs[k]) for k in meta_attrs if k != "config_yaml"
                }
            if "metadata_json" in f:
                for k, ds in f["metadata_json"].items():
                    raw = ds[()]
                    metadata[_unescape(k)] = json.loads(
                        raw.decode() if isinstance(raw, bytes) else raw
                    )

            wavelengths = f["qe"]["wavelengths"][:]
            qe_per_pixel = {
                _unescape(key)[6:]: f["qe"][key][:]
                for key in f["qe"]
                if key.startswith("pixel_")
            }
            eb = f["energy_balance"]

            def opt(name: str):
                return eb[name][:] if name in eb else None

            fields = None
            if "fields" in f:
                fields = {
                    _unescape(wl_key): FieldData(
                        **{c: grp[c][()] for c in _FIELD_KEYS if c in grp}
                    )
                    for wl_key, grp in f["fields"].items()
                }
            poynting = None
            if "poynting" in f:
                poynting = {_unescape(k): ds[()] for k, ds in f["poynting"].items()}

            result = SimulationResult(
                qe_per_pixel=qe_per_pixel,
                wavelengths=wavelengths,
                fields=fields,
                poynting=poynting,
                reflection=opt("reflection"),
                transmission=opt("transmission"),
                absorption=opt("absorption"),
                metadata=metadata,
            )
        return result, config


def _attr_value(v: Any) -> Any:
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, np.generic):
        return v.item()
    return v


def _load_legacy_yaml(text: Any) -> dict:
    """Read a version-1 config attribute (yaml.dump may have emitted numpy tags)."""
    if isinstance(text, bytes):
        text = text.decode()
    try:
        return yaml.safe_load(text) or {}
    except yaml.YAMLError as e:
        logger.warning(f"Could not parse legacy config_yaml ({e}); returning empty config")
        return {}
