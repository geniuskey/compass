#!/usr/bin/env python3
"""Smoke-check optional RCWA/FDTD backends installed in the uv environment."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from typing import Any


def check_jax() -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    value = float(jnp.sum(jnp.array([1.0, 2.0, 3.0])))
    return {
        "ok": value == 6.0,
        "version": getattr(jax, "__version__", None),
        "devices": [str(device) for device in jax.devices()],
        "calculation": value,
    }


def check_fdtd() -> dict[str, Any]:
    import fdtd

    fdtd.set_backend("numpy")
    grid = fdtd.Grid(shape=(8, 8, 8), grid_spacing=1e-6)
    return {
        "ok": True,
        "version": getattr(fdtd, "__version__", None),
        "grid_shape": str(grid.shape),
        "backend_type": type(fdtd.backend).__name__,
    }


def check_fmmax() -> dict[str, Any]:
    import fmmax
    import jax.numpy as jnp
    from fmmax import basis

    expansion = basis.generate_expansion(
        primitive_lattice_vectors=basis.LatticeVectors(
            u=jnp.array([1.0, 0.0]),
            v=jnp.array([0.0, 1.0]),
        ),
        approximate_num_terms=9,
    )
    return {
        "ok": True,
        "version": getattr(fmmax, "__version__", None),
        "num_terms": int(expansion.num_terms),
    }


def check_torcwa() -> dict[str, Any]:
    import torch
    import torcwa

    sim = torcwa.rcwa(freq=1.0 / 0.55, order=[0, 0], L=[1.0, 1.0])
    sim.add_input_layer(eps=1.0)
    sim.add_output_layer(eps=1.0)
    sim.set_incident_angle(0.0, 0.0)
    sim.add_layer(thickness=0.1, eps=2.25)
    sim.solve_global_smatrix()
    sim.source_planewave(amplitude=[1.0, 0.0], direction="forward", notation="xy")
    transmission = sim.S_parameters(
        orders=[0, 0],
        direction="forward",
        port="transmission",
        polarization="xx",
    )
    return {
        "ok": torch.isfinite(transmission).all().item(),
        "version": getattr(torcwa, "__version__", None),
        "t00_real": float(transmission.real),
        "t00_imag": float(transmission.imag),
    }


def check_grcwa() -> dict[str, Any]:
    import grcwa

    grcwa.set_backend("numpy")
    obj = grcwa.obj(
        3,
        L1=[1.0, 0.0],
        L2=[0.0, 1.0],
        freq=1.0,
        theta=0.0,
        phi=0.0,
        verbose=0,
    )
    obj.Add_LayerUniform(0, 1.0)
    obj.Add_LayerUniform(0.1, 2.25)
    obj.Add_LayerUniform(0, 1.0)
    obj.Init_Setup(Gmethod=0)
    obj.MakeExcitationPlanewave(0, 1, 0, 0, order=0)
    reflection, transmission = obj.RT_Solve(normalize=1)
    return {
        "ok": True,
        "version": getattr(grcwa, "__version__", None),
        "reflection": float(reflection),
        "transmission": float(transmission),
    }


def check_meent() -> dict[str, Any]:
    import meent
    import numpy as np

    mee = meent.call_mee(
        backend=0,
        pol=0,
        n_top=1.0,
        n_bot=1.0,
        theta=0.0,
        phi=0.0,
        fto=[0, 0],
        period=[1000.0, 1000.0],
        wavelength=550.0,
        thickness=[100.0],
        type_complex=np.complex128,
    )
    mee.ucell = np.ones((1, 1, 1), dtype=np.complex128) * 1.5
    result = mee.conv_solve()
    return {
        "ok": True,
        "version": getattr(meent, "__version__", None),
        "reflection_sum": float(np.sum(result.de_ri)),
        "transmission_sum": float(np.sum(result.de_ti)),
    }


def check_meep() -> dict[str, Any]:
    try:
        import meep as mp
    except ImportError as exc:
        return {
            "ok": False,
            "available": False,
            "error": repr(exc),
            "note": "MIT photonics Meep is normally installed as pymeep from conda-forge, not PyPI.",
        }

    required = ("Simulation", "Medium", "Vector3", "Block", "PML")
    has_api = all(hasattr(mp, attr) for attr in required)
    return {
        "ok": has_api,
        "available": has_api,
        "version": getattr(mp, "__version__", None),
        "file": getattr(mp, "__file__", None),
        "required_api": {attr: hasattr(mp, attr) for attr in required},
        "note": None if has_api else "Imported module is not the MIT photonics Meep API.",
    }


def run_check(name: str, func) -> dict[str, Any]:
    try:
        return func()
    except Exception as exc:
        return {
            "ok": False,
            "error": repr(exc),
            "traceback": traceback.format_exc(limit=5),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-check installed solver backends.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any backend check fails, including Meep.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checks = {
        "torcwa": check_torcwa,
        "grcwa": check_grcwa,
        "meent": check_meent,
        "fmmax": check_fmmax,
        "fdtd": check_fdtd,
        "jax": check_jax,
        "meep": check_meep,
    }
    results = {name: run_check(name, func) for name, func in checks.items()}
    print(json.dumps(results, indent=2, default=str))

    if args.strict and not all(result.get("ok", False) for result in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
