"""Regressions for cell-centred voxelization and field/epsilon axis agreement."""

from types import SimpleNamespace

import numpy as np
import pytest

from compass.core.types import Layer, LayerSlice, PhotodiodeSpec
from compass.geometry.pixel_stack import PixelStack


def test_voxels_sample_cell_centres_without_dropping_top_cell():
    stack = PixelStack({"pixel": {"pitch": 0.2, "unit_cell": [1, 1]}})
    stack.material_db.register_constant("glass", n=1.5, k=0.05)
    stack.layers = [Layer("slab", 0.0, 0.1, 0.1, "glass")]
    grid = stack.get_permittivity_grid(0.55, nx=3, ny=5, nz=8)
    np.testing.assert_allclose(grid, (1.5 + 0.05j) ** 2)


def test_torcwa_qe_transposes_xy_fields_back_to_yx_epsilon():
    torch = pytest.importorskip("torch")
    from compass.solvers.rcwa.torcwa_solver import TorcwaSolver

    stack = PixelStack({"pixel": {"pitch": 1.0, "unit_cell": [1, 1]}})
    stack.layers = [Layer("silicon", 0.0, 0.1, 0.1, "silicon")]
    stack.photodiodes = [PhotodiodeSpec((0.4, 0, 0.05), (0.18, 0.2, 0.1), (0, 0), "R")]
    epsilon = np.ones((3, 5), complex)
    epsilon[1, 4] += 1j
    layer = LayerSlice(0.0, 0.1, 0.1, epsilon, "silicon", "silicon")

    def field_xy(_li, x, y, **_kwargs):
        assert (len(x), len(y)) == (5, 3)
        field = torch.zeros((5, 3), dtype=torch.complex128)
        field[4, 1] = 1.0
        zero = torch.zeros_like(field)
        return [field, zero, zero], [zero, zero, zero]

    solver = TorcwaSolver({"name": "torcwa"})
    solver.setup_geometry(stack)
    solver.setup_source({"wavelength": {"mode": "single", "value": 0.55}, "polarization": "TE"})
    expected = (2 * np.pi / 0.55) * 0.1 / 15
    result = solver._compute_per_pixel_qe_fields(
        SimpleNamespace(field_xy=field_xy), torch, [layer], 0.55, expected
    )
    assert result["R_0_0"] == pytest.approx(expected, rel=1e-6)


def test_fdtdz_refuses_to_discard_material_absorption():
    from compass.solvers.fdtd.fdtdz_solver import FdtdzSolver

    with pytest.raises(NotImplementedError, match="refusing to drop"):
        FdtdzSolver._run_single_wavelength(
            None,
            None,
            None,
            0.55,
            np.ones((2, 3, 4), complex) + 0.001j,
            3,
            2,
            4,
            0.01,
            1,
            0.5,
            1,
            1,
            1,
            2,
            "TE",
        )


def test_meep_does_not_replace_unknown_material_with_air():
    from compass.solvers.fdtd.meep_solver import MeepSolver

    stack = PixelStack({"pixel": {"pitch": 0.1, "unit_cell": [1, 1]}})
    stack.layers = [Layer("slab", 0.0, 0.1, 0.1, "air")]
    solver = MeepSolver({"name": "meep"})
    solver.setup_geometry(stack)
    with pytest.raises(ValueError, match="Unknown Meep material"):
        solver._get_meep_material(None, "__missing_benchmark_material__", 0.55, True)
