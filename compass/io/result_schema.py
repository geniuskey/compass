"""Result data schema definitions.

Defines the canonical HDF5 layout that :class:`compass.io.hdf5_handler.HDF5Handler`
reads and writes, and provides validation helpers to verify existing files
against the schema.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# Re-export core types for convenience
from compass.core.types import SimulationResult, FieldData, LayerSlice

logger = logging.getLogger(__name__)

# Bump this when the HDF5 layout changes in a backwards-incompatible way.
SCHEMA_VERSION: str = "1.0"


@dataclass
class _DatasetDef:
    """Specification for a single HDF5 dataset within a group."""
    name: str
    dtype: str = "float64"
    required: bool = True
    description: str = ""


@dataclass
class _GroupDef:
    """Specification for a single HDF5 group."""
    name: str
    required: bool = True
    attrs: Dict[str, str] = field(default_factory=dict)
    datasets: List[_DatasetDef] = field(default_factory=list)
    description: str = ""


class ResultSchema:
    """Canonical HDF5 schema for COMPASS simulation results.

    The schema mirrors the layout used by
    :class:`compass.io.hdf5_handler.HDF5Handler`:

    .. code-block:: text

        /
        +-- metadata/           (group)
        |   attrs: config_yaml, solver, runtime_seconds, ...
        +-- qe/                 (group)
        |   +-- wavelengths     (dataset, float64, 1-D)
        |   +-- pixel_*         (datasets, float64, 1-D per pixel)
        +-- energy_balance/     (group)
        |   +-- reflection      (dataset, float64, 1-D, optional)
        |   +-- transmission    (dataset, float64, 1-D, optional)
        |   +-- absorption      (dataset, float64, 1-D, optional)
        +-- fields/             (group, optional)
            +-- <wavelength>/   (sub-groups)
                +-- Ex          (dataset, complex128, 3-D, optional)
                +-- Ey          (dataset, complex128, 3-D, optional)
                +-- Ez          (dataset, complex128, 3-D, optional)
    """

    # Class-level group definitions ----------------------------------------

    GROUPS: List[_GroupDef] = [
        _GroupDef(
            name="metadata",
            required=True,
            attrs={"config_yaml": "YAML string of run configuration"},
            description="Simulation metadata and configuration.",
        ),
        _GroupDef(
            name="qe",
            required=True,
            datasets=[
                _DatasetDef(
                    name="wavelengths",
                    dtype="float64",
                    required=True,
                    description="Wavelength array in um.",
                ),
            ],
            description="Quantum-efficiency data per pixel.",
        ),
        _GroupDef(
            name="energy_balance",
            required=True,
            datasets=[
                _DatasetDef(name="reflection", dtype="float64", required=False,
                            description="Reflection spectrum R(lambda)."),
                _DatasetDef(name="transmission", dtype="float64", required=False,
                            description="Transmission spectrum T(lambda)."),
                _DatasetDef(name="absorption", dtype="float64", required=False,
                            description="Absorption spectrum A(lambda)."),
            ],
            description="Energy balance arrays.",
        ),
        _GroupDef(
            name="fields",
            required=False,
            description="Optional electromagnetic field snapshots.",
        ),
    ]

    # Public API -----------------------------------------------------------

    @classmethod
    def validate(cls, filepath: str) -> List[str]:
        """Validate an HDF5 file against the COMPASS result schema.

        Args:
            filepath: Path to the HDF5 file.

        Returns:
            List of validation error strings. An empty list means the file
            is fully compliant.
        """
        import h5py

        errors: List[str] = []

        try:
            f = h5py.File(filepath, "r")
        except Exception as exc:
            return [f"Cannot open file: {exc}"]

        try:
            for group_def in cls.GROUPS:
                if group_def.name not in f:
                    if group_def.required:
                        errors.append(
                            f"Missing required group '/{group_def.name}'."
                        )
                    continue

                grp = f[group_def.name]

                # Check required attributes
                for attr_name in group_def.attrs:
                    if attr_name not in grp.attrs:
                        errors.append(
                            f"Missing attribute '{attr_name}' in "
                            f"group '/{group_def.name}'."
                        )

                # Check datasets
                for ds_def in group_def.datasets:
                    if ds_def.name not in grp:
                        if ds_def.required:
                            errors.append(
                                f"Missing required dataset "
                                f"'/{group_def.name}/{ds_def.name}'."
                            )
                        continue

                    ds = grp[ds_def.name]
                    actual_dtype = np.dtype(ds.dtype).name
                    expected_base = np.dtype(ds_def.dtype).name
                    if actual_dtype != expected_base:
                        errors.append(
                            f"Dataset '/{group_def.name}/{ds_def.name}' has "
                            f"dtype '{actual_dtype}', expected '{expected_base}'."
                        )

            # QE pixel datasets: at least one pixel_* dataset expected
            if "qe" in f:
                pixel_keys = [k for k in f["qe"] if k.startswith("pixel_")]
                if not pixel_keys:
                    errors.append(
                        "Group '/qe' contains no pixel datasets "
                        "(expected 'pixel_*')."
                    )
                # All pixel datasets should match wavelength length
                if "wavelengths" in f["qe"]:
                    n_wl = f["qe"]["wavelengths"].shape[0]
                    for pk in pixel_keys:
                        if f["qe"][pk].shape[0] != n_wl:
                            errors.append(
                                f"Dataset '/qe/{pk}' length "
                                f"({f['qe'][pk].shape[0]}) does not match "
                                f"wavelengths length ({n_wl})."
                            )
        finally:
            f.close()

        if errors:
            logger.warning(
                "Schema validation found %d issue(s) in '%s'.",
                len(errors), filepath,
            )
        else:
            logger.info("Schema validation passed for '%s'.", filepath)

        return errors

    @classmethod
    def schema_version(cls) -> str:
        """Return the current schema version string."""
        return SCHEMA_VERSION

    @classmethod
    def describe(cls) -> str:
        """Return a human-readable description of the schema layout."""
        lines: List[str] = [
            f"COMPASS HDF5 Result Schema v{SCHEMA_VERSION}",
            "=" * 50,
        ]
        for gdef in cls.GROUPS:
            req = "required" if gdef.required else "optional"
            lines.append(f"\n/{gdef.name}/  ({req})")
            if gdef.description:
                lines.append(f"  {gdef.description}")
            for attr_name, attr_desc in gdef.attrs.items():
                lines.append(f"  attr  {attr_name}: {attr_desc}")
            for ds_def in gdef.datasets:
                ds_req = "required" if ds_def.required else "optional"
                lines.append(
                    f"  dataset  {ds_def.name}  "
                    f"[{ds_def.dtype}, {ds_req}]"
                )
                if ds_def.description:
                    lines.append(f"           {ds_def.description}")
        return "\n".join(lines)
