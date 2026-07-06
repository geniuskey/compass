"""Regression tests for per-pixel QE grid indexing.

eps grids from PixelStack are indexed [row=y, col=x] (shape (ny, nx)).
These tests build a 1x2 unit cell where only the LEFT pixel's photodiode
column absorbs, on a deliberately non-square grid (ny != nx). A solver
that swaps the x/y axes when slicing the PD footprint attributes
absorption to the wrong pixel (or reads out of the absorbing region),
which these tests catch.
"""

from __future__ import annotations

import numpy as np
import pytest

from compass.core.types import LayerSlice
from compass.geometry.pixel_stack import PixelStack
from compass.solvers.fdtd.flaport_solver import FlaportFdtdSolver
from compass.solvers.rcwa.fmmax_solver import FmmaxSolver
from compass.solvers.rcwa.grcwa_solver import GrcwaSolver
from compass.solvers.rcwa.meent_solver import MeentSolver
from compass.solvers.rcwa.torcwa_solver import TorcwaSolver

NX, NY = 128, 64  # non-square grid so an axis swap cannot go unnoticed


@pytest.fixture
def pixel_stack_1x2() -> PixelStack:
    """1x2 unit cell (R | G), pitch 1um -> domain (lx=2, ly=1)."""
    config = {
        "pixel": {
            "pitch": 1.0,
            "unit_cell": [1, 2],
            "bayer_map": [["R", "G"]],
        }
    }
    return PixelStack(config)


def _absorbing_left_pixel_slice(stack: PixelStack) -> LayerSlice:
    """Silicon-depth slice absorbing ONLY under the left (R) pixel's PD."""
    lx, _ly = stack.domain_size
    eps = np.full((NY, NX), 1.5 + 0j, dtype=complex)
    # Left PD footprint: x in [0.15, 0.85] um of lx=2 um
    ix0 = int(0.15 / lx * NX)
    ix1 = int(np.ceil(0.85 / lx * NX))
    eps[:, ix0:ix1] += 0.5j
    # PD z-window is [1.5, 3.5] um for the default 3um silicon
    return LayerSlice(
        z_start=2.0,
        z_end=2.5,
        thickness=0.5,
        eps_grid=eps,
        name="silicon_slice_test",
        material="silicon",
    )


@pytest.mark.parametrize(
    "solver_cls",
    [GrcwaSolver, MeentSolver, FmmaxSolver],
    ids=["grcwa", "meent", "fmmax"],
)
def test_eps_weight_qe_maps_absorption_to_correct_pixel(solver_cls, pixel_stack_1x2):
    solver = solver_cls({"name": "test"})
    solver.setup_geometry(pixel_stack_1x2)
    layer_slices = [_absorbing_left_pixel_slice(pixel_stack_1x2)]

    qe = solver._compute_per_pixel_qe(layer_slices, wavelength=0.55, total_absorption=0.4)

    assert qe["R_0_0"] > 0.5, "absorbing left pixel must carry the QE"
    assert qe["G_0_1"] == pytest.approx(0.0, abs=1e-12), "non-absorbing right pixel must be 0"


def test_torcwa_fallback_qe_maps_absorption_to_correct_pixel(pixel_stack_1x2):
    solver = TorcwaSolver({"name": "torcwa"})
    solver.setup_geometry(pixel_stack_1x2)
    layer_slices = [_absorbing_left_pixel_slice(pixel_stack_1x2)]

    qe = solver._compute_per_pixel_qe(
        None, None, layer_slices, wavelength=0.55, total_absorption=0.4
    )

    assert qe["R_0_0"] > 0.5
    assert qe["G_0_1"] == pytest.approx(0.0, abs=1e-12)


def test_flaport_volume_qe_maps_absorption_to_correct_pixel(pixel_stack_1x2):
    solver = FlaportFdtdSolver({"name": "fdtd_flaport"})
    solver.setup_geometry(pixel_stack_1x2)
    stack = pixel_stack_1x2

    lx, _ly = stack.domain_size
    nz = 32
    eps_3d = np.full((NY, NX, nz), 1.5 + 0j, dtype=complex)
    ix0 = int(0.15 / lx * NX)
    ix1 = int(np.ceil(0.85 / lx * NX))
    eps_3d[:, ix0:ix1, :] += 0.5j  # absorb over the full z-column of the left PD

    qe = solver._compute_per_pixel_qe(eps_3d, wavelength=0.55, total_absorption=0.4)

    assert qe["R_0_0"] > 0.5
    assert qe["G_0_1"] == pytest.approx(0.0, abs=1e-12)
