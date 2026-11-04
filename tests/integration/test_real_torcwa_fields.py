"""Validate torcwa solved-field extraction against energy conservation.

Executes the real torcwa backend (skipped when not installed). Checks that
get_field_distribution returns reconstructed fields, not the permittivity
approximation, by testing physical invariants:

* cell-averaged Sz above/below the stack equals -(1-R)cos(theta) / -T cos(theta)
* Poynting theorem in an absorbing slab: dSz/dz = -k0 eps'' |E|^2
"""

import numpy as np
import pytest
import yaml

from compass.core.types import Layer
from compass.geometry.pixel_stack import PixelStack
from compass.solvers.base import SolverFactory

pytest.importorskip("torcwa")
import compass.solvers.rcwa.torcwa_solver  # noqa: F401

N, K, D, WL = 1.5, 0.05, 0.3, 0.55


def _slab_solver(theta_deg=0.0, pol="TE"):
    stack = PixelStack({"pixel": {"pitch": 1.0, "unit_cell": [1, 1]}})
    stack.material_db.register_constant("absorbing_glass", n=N, k=K)
    stack.layers = [Layer("slab", 0.0, D, D, "absorbing_glass")]
    solver = SolverFactory.create(
        "torcwa",
        {"name": "torcwa", "params": {"fourier_order": [1, 1], "dtype": "complex128"}},
    )
    solver.setup_geometry(stack)
    solver.setup_source({
        "wavelength": {"mode": "single", "value": WL},
        "angle": {"theta_deg": theta_deg, "phi_deg": 0.0},
        "polarization": pol,
    })
    return solver, solver.run()


@pytest.mark.parametrize("theta_deg", [0.0, 30.0])
@pytest.mark.parametrize("pol", ["TE", "TM"])
def test_slab_flux_matches_rt(theta_deg, pol):
    solver, result = _slab_solver(theta_deg, pol)
    sz = solver.get_field_distribution("Sz", "xz", 0.5)
    assert sz.shape == (solver.FIELD_N_XY, solver.FIELD_N_Z)
    cos_t = np.cos(np.deg2rad(theta_deg))
    np.testing.assert_allclose(sz[:, -1], -(1 - result.reflection[0]) * cos_t, rtol=1e-6)
    np.testing.assert_allclose(sz[:, 0], -result.transmission[0] * cos_t, rtol=1e-6)


@pytest.mark.parametrize("theta_deg", [0.0, 30.0])
def test_slab_poynting_theorem(theta_deg):
    solver, _ = _slab_solver(theta_deg)
    sz = solver.get_field_distribution("Sz", "xz", 0.5).mean(axis=0)
    e2 = solver.get_field_distribution("|E|2", "xz", 0.5).mean(axis=0)
    z = np.linspace(0.0, D, sz.size)
    dsz = np.gradient(sz, z)
    k0 = 2 * np.pi / WL
    absorbed = k0 * (2 * N * K) * e2
    inner = slice(5, -5)
    np.testing.assert_allclose(dsz[inner], -absorbed[inner], rtol=2e-3)


def test_slab_xy_matches_xz_at_same_height():
    solver, _ = _slab_solver()
    z_idx = 64
    z = np.linspace(0.0, D, solver.FIELD_N_Z)[z_idx]
    xz = solver.get_field_distribution("|E|2", "xz", 0.5)
    xy = solver.get_field_distribution("|E|2", "xy", float(z))
    assert xy.shape == (solver.FIELD_N_XY, solver.FIELD_N_XY)
    np.testing.assert_allclose(xy.mean(), xz[:, z_idx].mean(), rtol=1e-6)


def test_component_sum_equals_total_intensity():
    solver, _ = _slab_solver(30.0, "TM")
    parts = [solver.get_field_distribution(c, "yz", 0.25) for c in ("Ex", "Ey", "Ez")]
    total = solver.get_field_distribution("|E|2", "yz", 0.25)
    assert all((p >= 0).all() for p in parts)
    np.testing.assert_allclose(sum(parts), total, rtol=1e-10)
    assert parts[2].max() > 0  # TM at oblique incidence has Ez


def test_bsi_pixel_flux_balance():
    """Patterned Bayer pixel: cell-averaged flux matches 1-R and T."""
    with open("configs/pixel/default_bsi_1um.yaml") as f:
        cfg = yaml.safe_load(f)
    stack = PixelStack(cfg if "pixel" in cfg else {"pixel": cfg})
    solver = SolverFactory.create(
        "torcwa",
        {"name": "torcwa", "params": {"fourier_order": [3, 3], "n_lens_slices": 5}},
    )
    solver.setup_geometry(stack)
    solver.setup_source({"wavelength": {"mode": "single", "value": 0.55}, "polarization": "TE"})
    result = solver.run()
    assert result.metadata["qe_method"] == "field_integration"

    layers = solver._last_layer_info
    z_top = max(s.z_end for s in layers)
    z_bot = min(s.z_start for s in layers)
    top = solver.get_field_distribution("Sz", "xy", z_top).mean()
    bot = solver.get_field_distribution("Sz", "xy", z_bot).mean()
    assert top == pytest.approx(-(1 - result.reflection[0]), abs=1e-5)
    assert bot == pytest.approx(-result.transmission[0], abs=1e-5)

    # Patterned structure: the solved field varies laterally.
    e2 = solver.get_field_distribution("|E|2", "xz", 0.5)
    assert np.ptp(e2[:, e2.shape[1] // 2]) > 1e-3
