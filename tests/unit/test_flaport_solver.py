"""Unit tests for the flaport FDTD solver adapter."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from compass.core.types import Layer, PhotodiodeSpec
from compass.geometry.pixel_stack import PixelStack
from compass.solvers.fdtd.flaport_solver import FlaportFdtdSolver


def _solver(params=None):
    return FlaportFdtdSolver({"name": "fdtd_flaport", "params": params or {}}, device="cpu")


class TestSetup:
    def test_setup_geometry_rejects_none(self):
        with pytest.raises(ValueError, match="must not be None"):
            _solver().setup_geometry(None)

    def test_setup_geometry_rejects_empty_layers(self):
        ps = MagicMock()
        ps.layers = []
        with pytest.raises(ValueError, match="at least one layer"):
            _solver().setup_geometry(ps)

    def test_run_without_setup_raises(self):
        with pytest.raises(RuntimeError, match="setup_geometry"):
            _solver().run()

    def test_oblique_incidence_rejected(self):
        with pytest.raises(NotImplementedError):
            _solver().setup_source({
                "wavelength": {"mode": "single", "value": 0.55},
                "angle": {"theta_deg": 10.0},
            })


class TestGridDims:
    def test_lateral_period_is_exact(self):
        """The grid period equals the domain (old code: (int(l/gs)+1) cells)."""
        solver = _solver({"grid_spacing": 0.015})
        solver.setup_geometry(PixelStack({"pixel": {"pitch": 1.0, "unit_cell": [2, 2]}}))
        gs, n_x, n_y, _ = solver._grid_dims()
        assert n_x * gs == pytest.approx(2.0) and n_y * gs == pytest.approx(2.0)
        assert gs == pytest.approx(0.015, rel=0.01)

    def test_incommensurate_domain_raises(self):
        solver = _solver({"grid_spacing": 0.3})
        stack = MagicMock()
        stack.layers = [MagicMock()]
        stack.domain_size = (1.0, 1.1)
        stack.z_range = (0.0, 1.0)
        solver.setup_geometry(stack)
        with pytest.raises(ValueError, match="common cell size"):
            solver._grid_dims()


class TestFieldQE:
    def test_absorption_maps_to_absorbing_pixel(self):
        """QE uses the field in silicon under each PD footprint, [x, y] order."""
        stack = PixelStack({"pixel": {"pitch": 1.0, "unit_cell": [1, 2],
                                      "bayer_map": [["R", "G"]]}})
        stack.layers = [Layer("silicon", 0.0, 0.5, 0.5, "silicon")]
        stack.photodiodes = [
            PhotodiodeSpec((0.0, 0.0, 0.25), (1.0, 1.0, 0.5), (0, c), "RG"[c]) for c in range(2)
        ]
        solver = _solver()
        solver.setup_geometry(stack)
        gs, nx, ny, nz = 0.1, 20, 10, 5
        eps = np.full((ny, nx, nz), 12.0 + 0.0j)
        eps[:, :10, :] += 0.5j  # only the left (R) pixel absorbs
        E = np.ones((nx, ny, nz, 3), complex) / np.sqrt(3)  # |E|^2 = 1
        qe = solver._field_qe(E, eps, gs, 0.55, p_inc=1.0, total_absorption=0.5)
        k0 = 2 * np.pi / 0.55
        assert qe["R_0_0"] == pytest.approx(min(1.0, k0 * 0.5 * 0.5))
        assert qe["G_0_1"] == 0.0


class TestRealBackend:
    """Run the real fdtd library on tiny cells (skipped if not installed)."""

    @pytest.fixture(autouse=True)
    def _need_fdtd(self):
        pytest.importorskip("fdtd")
        pytest.importorskip("torch")

    @staticmethod
    def _slab_stack(material=("glass", 1.5, 0.05)):
        stack = PixelStack({"pixel": {"pitch": 0.04, "unit_cell": [1, 1]}})
        name, n, k = material
        stack.material_db.register_constant(name, n=n, k=k)
        stack.layers = [
            Layer("air_b", 0.0, 0.1, 0.1, "air"),
            Layer("slab", 0.1, 0.4, 0.3, name),
            Layer("air", 0.4, 0.6, 0.2, "air"),
        ]
        return stack

    def test_slab_matches_tmm(self):
        """Soft source + field subtraction: R is no longer pinned to 0."""
        from compass.solvers.tmm.tmm_core import transfer_matrix_1d

        solver = _solver({"grid_spacing": 0.01, "runtime": 100, "pml_layers": 10})
        solver.setup_geometry(self._slab_stack())
        solver.setup_source({"wavelength": {"mode": "single", "value": 0.55},
                             "polarization": "TE"})
        result = solver.run()
        n = np.array([1.0, 1.5 + 0.05j, 1.0])
        d = np.array([np.inf, 0.3, np.inf])
        R, T, _ = transfer_matrix_1d(n, d, 0.55)
        assert result.reflection[0] == pytest.approx(R, abs=0.01)
        assert result.transmission[0] == pytest.approx(T, abs=0.01)
        assert result.metadata["qe_method"] == "field_integration"
        xz = solver.get_field_distribution("|E|2", "xz", 0.02)
        assert xz.shape[1] == round(0.6 / 0.01) and np.all(np.isfinite(xz))

    def test_metal_permittivity_is_rejected(self):
        solver = _solver({"grid_spacing": 0.01, "runtime": 100, "pml_layers": 10})
        solver.setup_geometry(self._slab_stack(("metal", 0.5, 3.0)))
        solver.setup_source({"wavelength": {"mode": "single", "value": 0.55},
                             "polarization": "TE"})
        result = solver.run()
        assert np.isnan(result.reflection[0])
        assert "Re(eps) < 1" in result.metadata["failed_runs"][0]["error"]
