"""Compare installed RCWA adapters against analytical thin-film optics.

These tests execute the real optional backends. They skip when a backend is
not installed; mock-based orchestration tests cannot establish optical accuracy.
"""

import numpy as np
import pytest

from compass.core.types import Layer, LayerSlice
from compass.geometry.pixel_stack import PixelStack
from compass.runners.single_run import SingleRunner
from compass.solvers.base import SolverFactory
from compass.solvers.tmm.tmm_core import transfer_matrix_1d
from compass.sources.cone_illumination import ConeIllumination


@pytest.mark.parametrize("backend", ["grcwa", "meent", "torcwa", "fmmax"])
@pytest.mark.parametrize("theta_deg,phi_deg", [(0.0, 0.0), (30.0, 45.0)])
@pytest.mark.parametrize("extinction", [0.0, 0.05])
def test_slab_matches_tmm(backend, theta_deg, phi_deg, extinction):
    pytest.importorskip(backend)
    stack = PixelStack({"pixel": {"pitch": 1.0, "unit_cell": [1, 1]}})
    stack.material_db.register_constant("test_glass", n=1.5, k=extinction)
    stack.layers = [Layer("slab", 0.0, 0.1, 0.1, "test_glass")]
    solver = SolverFactory.create(
        backend,
        {"name": backend, "params": {"nG": 3, "fourier_order": [1, 1]}},
    )
    solver.setup_geometry(stack)
    solver.setup_source({
        "wavelength": {"mode": "list", "values": [0.45, 0.55, 0.65]},
        "angle": {"theta_deg": theta_deg, "phi_deg": phi_deg},
        "polarization": "unpolarized",
    })
    result = solver.run()
    expected = np.array([
        np.mean([
            transfer_matrix_1d(
                np.array([1.0, 1.5 + 1j * extinction, 1.0], dtype=complex),
                np.array([np.inf, 0.1, np.inf]),
                wl, np.deg2rad(theta_deg), pol,
            )
            for pol in ("TE", "TM")
        ], axis=0)
        for wl in result.wavelengths
    ])
    np.testing.assert_allclose(result.reflection, expected[:, 0], atol=5e-5)
    np.testing.assert_allclose(result.transmission, expected[:, 1], atol=5e-5)
    np.testing.assert_allclose(result.absorption, expected[:, 2], atol=5e-5)
    assert not result.metadata.get("failed_runs")
    if backend == "fmmax":
        assert result.metadata["num_terms"] == 9


def test_meent_torch_absorbing_slab_matches_tmm():
    pytest.importorskip("meent")
    pytest.importorskip("torch")
    stack = PixelStack({"pixel": {"pitch": 1.0, "unit_cell": [1, 1]}})
    stack.material_db.register_constant("test_glass", n=1.5, k=0.05)
    stack.layers = [Layer("slab", 0.0, 0.1, 0.1, "test_glass")]
    solver = SolverFactory.create(
        "meent",
        {"name": "meent", "params": {"backend": "torch", "fourier_order": [1, 1]}},
        device="cpu",
    )
    solver.setup_geometry(stack)
    solver.setup_source({"wavelength": {"mode": "single", "value": 0.55},
                         "polarization": "TE"})
    result = solver.run()
    expected = transfer_matrix_1d(
        np.array([1.0, 1.5 + 0.05j, 1.0]),
        np.array([np.inf, 0.1, np.inf]), 0.55, 0.0, "TE",
    )
    assert not result.metadata.get("failed_runs")
    np.testing.assert_allclose(
        [result.reflection[0], result.transmission[0], result.absorption[0]],
        expected, atol=5e-5,
    )


@pytest.mark.parametrize("backend", ["torcwa", "grcwa"])
def test_patterned_grid_preserves_physical_xy_coordinates(backend, monkeypatch):
    library = pytest.importorskip(backend)
    # LayerSlice uses rows=y, columns=x; the backend APIs use x first.
    epsilon = np.full((64, 64), 2.25 + 0.01j)
    epsilon[4, 9] = 4.0 + 0.02j
    stack = PixelStack({"pixel": {"pitch": 1.0, "unit_cell": [1, 1]}})
    stack.layers = [Layer("slab", 0., .1, .1, "air")]
    stack.photodiodes = []
    monkeypatch.setattr(stack, "get_layer_slices", lambda *a, **kw: [
        LayerSlice(0., .1, .1, epsilon, "slab", "air"),
    ])
    captured = []
    if backend == "torcwa":
        original = library.rcwa.add_layer

        def capture(self, *args, **kwargs):
            captured.append(kwargs["eps"].detach().cpu().numpy())
            return original(self, *args, **kwargs)

        monkeypatch.setattr(library.rcwa, "add_layer", capture)
    else:
        original = library.obj.GridLayer_geteps

        def capture(self, values):
            captured.append(np.asarray(values).reshape(64, 64))
            return original(self, values)

        monkeypatch.setattr(library.obj, "GridLayer_geteps", capture)
    solver = SolverFactory.create(backend, {"name": backend, "params": {
        "fourier_order": [1, 1], "nG": 10, "grid_multiplier": 0,
        "dtype": "complex128",
    }})
    solver.setup_geometry(stack)
    solver.setup_source({"wavelength": {"mode": "single", "value": .55},
                         "polarization": "TE"})
    result = solver.run()
    assert not result.metadata.get("failed_runs")
    assert len(captured) == 1
    assert captured[0][9, 4] == epsilon[4, 9]
    assert captured[0][4, 9] == epsilon[9, 4]


def test_single_runner_integrates_cone_angles():
    config = {
        "pixel": {"pitch": 1.0, "unit_cell": [1, 1]},
        "solver": {"name": "tmm"},
        "source": {
            "type": "cone_illumination",
            "wavelength": {"mode": "single", "value": 0.55},
            "cone": {
                "cra_deg": 25.0,
                "f_number": 1.4,
                "sampling": {"type": "fibonacci", "n_points": 5},
            },
        },
    }
    result = SingleRunner.run(config)
    expected = 0.0
    cone = ConeIllumination(cra_deg=25.0, f_number=1.4, n_points=5)
    for theta, phi, weight in cone.get_sampling_points():
        angle_config = {
            **config,
            "source": {
                "type": "planewave",
                "wavelength": config["source"]["wavelength"],
                "angle": {"theta_deg": theta, "phi_deg": phi},
            },
        }
        expected += weight * SingleRunner.run(angle_config).reflection[0]
    assert result.metadata["cone_n_points"] == 5
    assert result.reflection[0] == pytest.approx(expected)
    normal_config = {**config, "source": {"type": "planewave"}}
    assert abs(result.reflection[0] - SingleRunner.run(normal_config).reflection[0]) > 1e-5
