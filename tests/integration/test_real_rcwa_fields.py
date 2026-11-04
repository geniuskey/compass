"""Validate RCWA solved-field extraction and field-based QE on real backends.

Executes torcwa, grcwa, meent and fmmax (each skipped when not installed). Checks that
get_field_distribution returns reconstructed fields, not the permittivity
approximation, by testing physical invariants:

* cell-averaged Sz above/below the stack equals -(1-R)cos(theta) / -T cos(theta)
* Poynting theorem in an absorbing slab: dSz/dz = -k0 eps'' |E|^2
"""

import importlib

import numpy as np
import pytest
import yaml

from compass.core.types import Layer
from compass.geometry.pixel_stack import PixelStack
from compass.solvers.base import SolverFactory
from compass.solvers.rcwa.fields import FIELD_N_XY, FIELD_N_Z

N, K, D, WL = 1.5, 0.05, 0.3, 0.55

SLAB_PARAMS = {
    "torcwa": {"fourier_order": [1, 1]},
    "grcwa": {"nG": 9},
    "meent": {"fourier_order": [1, 1]},
    "fmmax": {"fourier_order": [1, 1]},
}
PIXEL_PARAMS = {
    "torcwa": {"fourier_order": [3, 3]},
    "grcwa": {"nG": 49},
    "meent": {"fourier_order": [3, 3]},
    "fmmax": {"fourier_order": [3, 3]},
}
BACKENDS = pytest.mark.parametrize("backend", ["torcwa", "grcwa", "meent", "fmmax"])


def _require(backend):
    pytest.importorskip(backend)
    if backend == "fmmax":
        pytest.importorskip("jax")
    importlib.import_module(f"compass.solvers.rcwa.{backend}_solver")


def _slab_solver(backend, theta_deg=0.0, pol="TE"):
    _require(backend)
    stack = PixelStack({"pixel": {"pitch": 1.0, "unit_cell": [1, 1]}})
    stack.material_db.register_constant("absorbing_glass", n=N, k=K)
    stack.layers = [Layer("slab", 0.0, D, D, "absorbing_glass")]
    solver = SolverFactory.create(
        backend,
        {"name": backend, "params": {"dtype": "complex128", **SLAB_PARAMS[backend]}},
    )
    solver.setup_geometry(stack)
    solver.setup_source({
        "wavelength": {"mode": "single", "value": WL},
        "angle": {"theta_deg": theta_deg, "phi_deg": 0.0},
        "polarization": pol,
    })
    return solver, solver.run()


def _bsi_solver(backend, theta_deg=0.0, wavelengths=(0.55,)):
    _require(backend)
    with open("configs/pixel/default_bsi_1um.yaml") as f:
        cfg = yaml.safe_load(f)
    solver = SolverFactory.create(
        backend,
        {"name": backend,
         "params": {"n_lens_slices": 5, "dtype": "complex128", **PIXEL_PARAMS[backend]}},
    )
    solver.setup_geometry(PixelStack(cfg if "pixel" in cfg else {"pixel": cfg}))
    solver.setup_source({
        "wavelength": {"mode": "list", "values": list(wavelengths)},
        "angle": {"theta_deg": theta_deg, "phi_deg": 0.0},
        "polarization": "TE",
    })
    return solver, solver.run()


@BACKENDS
@pytest.mark.parametrize("theta_deg", [0.0, 30.0])
@pytest.mark.parametrize("pol", ["TE", "TM"])
def test_slab_flux_matches_rt(backend, theta_deg, pol):
    solver, result = _slab_solver(backend, theta_deg, pol)
    sz = solver.get_field_distribution("Sz", "xz", 0.5)
    assert sz.shape == (FIELD_N_XY, FIELD_N_Z)
    cos_t = np.cos(np.deg2rad(theta_deg))
    np.testing.assert_allclose(sz[:, -1], -(1 - result.reflection[0]) * cos_t, rtol=1e-6)
    np.testing.assert_allclose(sz[:, 0], -result.transmission[0] * cos_t, rtol=1e-6)


@BACKENDS
@pytest.mark.parametrize("theta_deg", [0.0, 30.0])
def test_slab_poynting_theorem(backend, theta_deg):
    solver, _ = _slab_solver(backend, theta_deg)
    sz = solver.get_field_distribution("Sz", "xz", 0.5).mean(axis=0)
    e2 = solver.get_field_distribution("|E|2", "xz", 0.5).mean(axis=0)
    z = np.linspace(0.0, D, sz.size)
    dsz = np.gradient(sz, z)
    absorbed = (2 * np.pi / WL) * (2 * N * K) * e2
    inner = slice(5, -5)
    np.testing.assert_allclose(dsz[inner], -absorbed[inner], rtol=2e-3)


@BACKENDS
def test_slab_xy_matches_xz_at_same_height(backend):
    solver, _ = _slab_solver(backend)
    z_idx = 64
    z = np.linspace(0.0, D, FIELD_N_Z)[z_idx]
    xz = solver.get_field_distribution("|E|2", "xz", 0.5)
    xy = solver.get_field_distribution("|E|2", "xy", float(z))
    assert xy.shape == (FIELD_N_XY, FIELD_N_XY)
    np.testing.assert_allclose(xy.mean(), xz[:, z_idx].mean(), rtol=1e-6)


@BACKENDS
def test_component_sum_equals_total_intensity(backend):
    solver, _ = _slab_solver(backend, 30.0, "TM")
    parts = [solver.get_field_distribution(c, "yz", 0.25) for c in ("Ex", "Ey", "Ez")]
    total = solver.get_field_distribution("|E|2", "yz", 0.25)
    assert all((p >= 0).all() for p in parts)
    np.testing.assert_allclose(sum(parts), total, rtol=1e-10)
    assert parts[2].max() > 0  # TM at oblique incidence has Ez


@BACKENDS
def test_bsi_pixel_flux_balance(backend):
    """Patterned Bayer pixel: cell-averaged flux matches 1-R and T."""
    solver, result = _bsi_solver(backend)
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


@BACKENDS
def test_bsi_pixel_qe_is_physical(backend):
    """Field-integrated QE: bounded by A, and each color peaks in its own band."""
    _, result = _bsi_solver(backend, wavelengths=(0.45, 0.55, 0.65))
    qe = result.qe_per_pixel
    # Sum of per-pixel QE (each normalized to one pixel area) / n_pixels <= A.
    mean_qe = np.mean([v for v in qe.values()], axis=0)
    assert np.all(mean_qe <= result.absorption + 1e-6)
    assert np.argmax(qe["B_1_1"]) == 0
    assert np.argmax(qe["G_0_1"]) == 1
    assert np.argmax(qe["R_0_0"]) == 2
    # Color filtering: off-band QE well below in-band QE.
    assert qe["B_1_1"][1] < 0.3 * qe["B_1_1"][0]


@pytest.mark.parametrize("backend", ["meent", "fmmax"])
def test_li_rule_backends_pin_slice_absorption_to_flux(backend, caplog):
    """meent/fmmax (Li-rule factorization): absorbed power per slice is the
    exact flux difference, so the field-based total equals 1 - R - T."""
    import logging

    with caplog.at_level(logging.DEBUG, logger="compass.solvers.rcwa.fields"):
        _, result = _bsi_solver(backend)
    lines = [r.getMessage() for r in caplog.records if "field-integrated absorption" in r.message]
    assert lines, "expected the integrator's debug summary"
    total = float(lines[-1].split("absorption ")[1].split(",")[0])
    assert total == pytest.approx(result.absorption[0], abs=0.01)


def test_rcwa_backends_agree_on_bsi_qe():
    """Field-integrated per-pixel QE agrees across the RCWA backends."""
    available = []
    for backend in ("torcwa", "meent", "fmmax"):
        try:
            _require(backend)
        except pytest.skip.Exception:
            continue
        available.append(backend)
    if len(available) < 2:
        pytest.skip("need at least two RCWA backends")
    qes = {b: _bsi_solver(b)[1].qe_per_pixel for b in available}
    ref = qes[available[0]]
    for b in available[1:]:
        for key, val in ref.items():
            assert qes[b][key][0] == pytest.approx(val[0], abs=0.05), (b, key)
