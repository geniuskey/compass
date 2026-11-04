"""Ensure independent epsilon sampling survives the adapter execution path."""

from types import SimpleNamespace

import numpy as np
import pytest

from compass.core.types import Layer
from compass.geometry.pixel_stack import PixelStack
from compass.solvers.base import SolverFactory


@pytest.mark.parametrize("backend", ["meent", "fmmax"])
@pytest.mark.parametrize("multiplier,expected_yx", [(0, [16, 16]), (None, [33, 21])])
def test_explicit_grid_override_and_default_oversampling(
    backend, multiplier, expected_yx, monkeypatch
):
    """Exercise sampling and metadata without expensive eigendecompositions."""
    library = pytest.importorskip(backend)
    stack = PixelStack({"pixel": {"pitch": 0.8, "unit_cell": [1, 1]}})
    stack.material_db.register_constant("glass", n=1.5)
    stack.layers = [Layer("slab", 0.0, 0.1, 0.1, "glass")]
    stack.photodiodes = []
    sampled_shapes = []
    original = stack.get_layer_slices

    def capture(wavelength, nx, ny, **kwargs):
        layers = original(wavelength, nx, ny, **kwargs)
        sampled_shapes.extend(list(layer.eps_grid.shape) for layer in layers)
        return layers

    monkeypatch.setattr(stack, "get_layer_slices", capture)
    params = {"fourier_order": [3, 5], "epsilon_grid": 16}
    if multiplier is not None:
        params["grid_multiplier"] = multiplier
    solver = SolverFactory.create(backend, {"name": backend, "params": params})
    if backend == "meent":

        def call_mee(**_kwargs):
            assert _kwargs["fourier_type"] == 0
            assert _kwargs["enhanced_dfs"] is True
            mee = SimpleNamespace()

            def solve():
                assert list(mee.ucell.shape[1:]) == expected_yx
                return SimpleNamespace(de_ri=np.array([0.1]), de_ti=np.array([0.8]))

            mee.conv_solve = solve
            return mee

        monkeypatch.setattr(library, "call_mee", call_mee)
    else:
        monkeypatch.setattr(solver, "_solve_single", lambda **_kwargs: (0.1, 0.8, 0.1, None))
    solver.setup_geometry(stack)
    solver.setup_source({"wavelength": {"mode": "single", "value": 0.55}, "polarization": "TE"})
    result = solver.run()
    assert not result.metadata.get("failed_runs")
    assert sampled_shapes and all(shape == expected_yx for shape in sampled_shapes)
    assert result.metadata["epsilon_grid_requested"] == 16
    assert result.metadata["epsilon_grid_shape_yx"] == expected_yx
    assert result.metadata["grid_multiplier"] == (3 if multiplier is None else multiplier)


@pytest.mark.parametrize("fourier_type,enhanced_dfs", [(0, False), (1, True)])
def test_meent_propagates_fourier_algorithm(fourier_type, enhanced_dfs, monkeypatch):
    library = pytest.importorskip("meent")
    stack = PixelStack({"pixel": {"pitch": 0.8, "unit_cell": [1, 1]}})
    stack.layers = [Layer("slab", 0.0, 0.1, 0.1, "air")]
    stack.photodiodes = []

    def call_mee(**kwargs):
        assert kwargs["fourier_type"] == fourier_type
        assert kwargs["enhanced_dfs"] == enhanced_dfs
        return SimpleNamespace(
            conv_solve=lambda: SimpleNamespace(de_ri=np.array([0.0]), de_ti=np.array([1.0]))
        )

    monkeypatch.setattr(library, "call_mee", call_mee)
    solver = SolverFactory.create(
        "meent",
        {
            "name": "meent",
            "params": {
                "fourier_order": [1, 2],
                "epsilon_grid": 8,
                "grid_multiplier": 0,
                "fourier_type": fourier_type,
                "enhanced_dfs": enhanced_dfs,
            },
        },
    )
    solver.setup_geometry(stack)
    solver.setup_source({"wavelength": {"mode": "single", "value": 0.55}, "polarization": "TE"})
    result = solver.run()
    assert not result.metadata.get("failed_runs")
    assert result.metadata["fourier_type"] == fourier_type
    assert result.metadata["enhanced_dfs"] == enhanced_dfs
    assert result.metadata["fourier_work_grid_shape_yx"] == ([16, 8] if fourier_type == 0 else None)


@pytest.mark.parametrize("enhanced_dfs", [False, True])
def test_meent_work_grid_metadata_matches_upstream_repeat(enhanced_dfs, monkeypatch):
    pytest.importorskip("meent")
    from meent.on_numpy.emsolver import convolution_matrix

    from compass.solvers.rcwa.meent_solver import _dfs_work_grid_shape

    observed = []

    def capture(cell, *_args, **_kwargs):
        observed.append(list(cell.shape))
        return np.eye(15, dtype=complex)

    monkeypatch.setattr(convolution_matrix, "dfs2d", capture)
    convolution_matrix.to_conv_mat_raster_discrete(
        np.ones((1, 4, 8)), 1, 2, enhanced_dfs=enhanced_dfs
    )
    expected = _dfs_work_grid_shape([4, 8], [1, 2], enhanced_dfs)
    assert len(observed) == 3
    assert all(shape == expected for shape in observed)
