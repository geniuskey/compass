"""Unit tests for fmmax RCWA solver adapter."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from compass.core.types import SimulationResult
from compass.solvers.base import SolverBase, SolverFactory


# ---------------------------------------------------------------------------
# Helpers: build a fake fmmax + jax ecosystem for import mocking
# ---------------------------------------------------------------------------


def _make_fake_jax():
    """Create a mock jax module with jax.numpy."""
    jax = types.ModuleType("jax")
    jnp = types.ModuleType("jax.numpy")

    # jax.numpy dtypes and functions
    jnp.complex64 = np.complex64
    jnp.complex128 = np.complex128

    def _jnp_array(data, dtype=None):
        return np.asarray(data, dtype=dtype)

    def _jnp_zeros(shape, dtype=None):
        return np.zeros(shape, dtype=dtype)

    def _jnp_ones(shape, dtype=None):
        return np.ones(shape, dtype=dtype)

    def _jnp_sum(x, **kw):
        return np.sum(x, **kw)

    def _jnp_abs(x):
        return np.abs(x)

    jnp.array = _jnp_array
    jnp.zeros = _jnp_zeros
    jnp.ones = _jnp_ones
    jnp.sum = _jnp_sum
    jnp.abs = _jnp_abs

    jax.numpy = jnp
    jax.config = MagicMock()
    # scipy's array_api_compat checks for jax.Array; provide a dummy
    jax.Array = type("Array", (), {})
    return jax, jnp


def _make_fake_fmmax():
    """Create a mock fmmax module."""
    fmmax = types.ModuleType("fmmax")
    fmmax_basis = types.ModuleType("fmmax.basis")
    fmmax_fmm = types.ModuleType("fmmax.fmm")

    # basis types
    class FakeLatticeVectors:
        def __init__(self, u, v):
            self.u = u
            self.v = v

    class FakeTruncation:
        CIRCULAR = "CIRCULAR"

    fmmax_basis.LatticeVectors = FakeLatticeVectors
    fmmax_basis.Truncation = FakeTruncation()

    class FakeExpansion:
        primitive_lattice_vectors = FakeLatticeVectors(
            u=np.array([1.0, 0.0]), v=np.array([0.0, 1.0])
        )

    fmmax_basis.generate_expansion = MagicMock(return_value=FakeExpansion())

    # fmm types
    class FakeFormulation:
        POL = "POL"
        NORMAL = "NORMAL"
        JONES = "JONES"
        JONES_DIRECT = "JONES_DIRECT"

    fmmax_fmm.Formulation = FakeFormulation()

    # S-matrix mock
    class FakeSMatrix:
        s11 = np.array([[0.1]])
        s12 = np.array([[0.0]])
        s21 = np.array([[0.8]])
        s22 = np.array([[0.0]])

    class FakeSolveResult:
        s_matrix = FakeSMatrix()

    fmmax_fmm.eigensolve_isotropic_media = MagicMock(
        return_value=FakeSolveResult()
    )
    fmmax_fmm.append_layer = MagicMock(return_value=FakeSMatrix())

    fmmax.basis = fmmax_basis
    fmmax.fmm = fmmax_fmm

    return fmmax, fmmax_basis, fmmax_fmm


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fmmax_config():
    """Default fmmax solver configuration."""
    return {
        "name": "fmmax",
        "type": "rcwa",
        "params": {
            "fourier_order": [5, 5],
            "fmm_formulation": "jones",
            "dtype": "complex64",
        },
    }


@pytest.fixture
def simple_pixel_config():
    """Minimal pixel configuration for PixelStack."""
    return {
        "pixel": {
            "pitch": 1.0,
            "unit_cell": [2, 2],
            "layers": {
                "air": {"thickness": 0.5, "material": "air"},
                "microlens": {
                    "enabled": True,
                    "height": 0.4,
                    "radius_x": 0.4,
                    "radius_y": 0.4,
                    "material": "polymer_n1p56",
                    "profile": {
                        "type": "superellipse",
                        "n": 2.5,
                        "alpha": 1.0,
                    },
                    "shift": {"mode": "none"},
                },
                "planarization": {"thickness": 0.2, "material": "sio2"},
                "color_filter": {
                    "thickness": 0.5,
                    "pattern": "bayer_rggb",
                    "materials": {
                        "R": "cf_red",
                        "G": "cf_green",
                        "B": "cf_blue",
                    },
                    "grid": {"enabled": False},
                },
                "barl": {"layers": []},
                "silicon": {
                    "thickness": 2.0,
                    "material": "silicon",
                    "photodiode": {
                        "position": [0.0, 0.0, 0.3],
                        "size": [0.6, 0.6, 1.5],
                    },
                    "dti": {"enabled": False},
                },
            },
            "bayer_map": [["R", "G"], ["G", "B"]],
        }
    }


@pytest.fixture
def source_config():
    """Minimal source configuration."""
    return {
        "wavelength": {
            "mode": "list",
            "values": [0.45, 0.55, 0.65],
        },
        "angle": {"theta_deg": 0.0, "phi_deg": 0.0},
        "polarization": "unpolarized",
    }


@pytest.fixture
def _patch_fmmax_imports():
    """Patch fmmax and jax into sys.modules for the duration of a test."""
    jax, jnp = _make_fake_jax()
    fmmax, fmmax_basis, fmmax_fmm = _make_fake_fmmax()

    modules = {
        "jax": jax,
        "jax.numpy": jnp,
        "fmmax": fmmax,
        "fmmax.basis": fmmax_basis,
        "fmmax.fmm": fmmax_fmm,
    }
    with patch.dict(sys.modules, modules):
        yield


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFmmaxSolverRegistration:
    """Tests for SolverFactory registration."""

    def setup_method(self):
        self._saved_registry = SolverFactory._registry.copy()

    def teardown_method(self):
        SolverFactory._registry = self._saved_registry

    def test_fmmax_registered_in_factory(self):
        """fmmax should be registered in SolverFactory after import."""
        # Force import to trigger registration
        import compass.solvers.rcwa.fmmax_solver  # noqa: F401

        assert "fmmax" in SolverFactory._registry

    def test_factory_create_returns_fmmax_solver(self):
        """SolverFactory.create('fmmax', ...) returns an FmmaxSolver."""
        from compass.solvers.rcwa.fmmax_solver import FmmaxSolver

        # Manually register since setup_method saved a snapshot before import
        SolverFactory.register("fmmax", FmmaxSolver)

        config = {
            "name": "fmmax",
            "type": "rcwa",
            "params": {"fmm_formulation": "jones"},
        }
        solver = SolverFactory.create("fmmax", config)
        assert isinstance(solver, SolverBase)
        assert solver.name == "fmmax"
        assert solver.solver_type == "rcwa"


class TestFmmaxSolverInit:
    """Tests for FmmaxSolver initialization."""

    def test_default_formulation(self):
        """Default FMM formulation should be jones."""
        from compass.solvers.rcwa.fmmax_solver import FmmaxSolver

        solver = FmmaxSolver({"name": "fmmax", "type": "rcwa", "params": {}})
        assert solver._fmm_formulation == "jones"

    def test_custom_formulation(self):
        """Custom FMM formulation is stored."""
        from compass.solvers.rcwa.fmmax_solver import FmmaxSolver

        solver = FmmaxSolver(
            {
                "name": "fmmax",
                "type": "rcwa",
                "params": {"fmm_formulation": "pol"},
            }
        )
        assert solver._fmm_formulation == "pol"

    def test_invalid_formulation_raises(self):
        """Invalid FMM formulation raises ValueError."""
        from compass.solvers.rcwa.fmmax_solver import FmmaxSolver

        with pytest.raises(ValueError, match="Unknown fmm_formulation"):
            FmmaxSolver(
                {
                    "name": "fmmax",
                    "type": "rcwa",
                    "params": {"fmm_formulation": "invalid_method"},
                }
            )

    def test_default_dtype(self):
        """Default dtype should be complex64."""
        from compass.solvers.rcwa.fmmax_solver import FmmaxSolver

        solver = FmmaxSolver({"name": "fmmax", "type": "rcwa", "params": {}})
        assert solver._dtype_str == "complex64"


class TestFmmaxSetupGeometry:
    """Tests for setup_geometry validation."""

    def test_none_pixel_stack_raises(self):
        """Passing None as pixel_stack raises ValueError."""
        from compass.solvers.rcwa.fmmax_solver import FmmaxSolver

        solver = FmmaxSolver({"name": "fmmax", "params": {}})
        with pytest.raises(ValueError, match="pixel_stack must not be None"):
            solver.setup_geometry(None)

    def test_empty_layers_raises(self):
        """PixelStack with no layers raises ValueError."""
        from compass.solvers.rcwa.fmmax_solver import FmmaxSolver

        mock_stack = MagicMock()
        mock_stack.layers = []
        solver = FmmaxSolver({"name": "fmmax", "params": {}})
        with pytest.raises(
            ValueError, match="pixel_stack must have at least one layer"
        ):
            solver.setup_geometry(mock_stack)

    def test_valid_pixel_stack_stored(self, simple_pixel_config):
        """Valid pixel_stack is stored after setup_geometry."""
        from compass.solvers.rcwa.fmmax_solver import FmmaxSolver
        from compass.geometry.pixel_stack import PixelStack

        solver = FmmaxSolver({"name": "fmmax", "params": {}})
        ps = PixelStack(simple_pixel_config)
        solver.setup_geometry(ps)
        assert solver._pixel_stack is ps


class TestFmmaxSetupSource:
    """Tests for setup_source validation."""

    def test_valid_source_config(self, source_config):
        """Valid source config creates PlanewaveSource."""
        from compass.solvers.rcwa.fmmax_solver import FmmaxSolver

        solver = FmmaxSolver({"name": "fmmax", "params": {}})
        solver.setup_source(source_config)
        assert solver._source is not None
        assert solver._source.n_wavelengths == 3

    def test_empty_wavelengths_raises(self):
        """Empty wavelengths array raises ValueError."""
        from compass.solvers.rcwa.fmmax_solver import FmmaxSolver

        solver = FmmaxSolver({"name": "fmmax", "params": {}})
        with pytest.raises(ValueError, match="wavelengths array must not be empty"):
            solver.setup_source(
                {
                    "wavelength": {"mode": "list", "values": []},
                    "angle": {"theta_deg": 0.0, "phi_deg": 0.0},
                    "polarization": "TE",
                }
            )

    def test_negative_wavelength_raises(self):
        """Negative wavelength raises ValueError."""
        from compass.solvers.rcwa.fmmax_solver import FmmaxSolver

        solver = FmmaxSolver({"name": "fmmax", "params": {}})
        with pytest.raises(ValueError, match="all wavelengths must be positive"):
            solver.setup_source(
                {
                    "wavelength": {"mode": "list", "values": [-0.5]},
                    "angle": {"theta_deg": 0.0, "phi_deg": 0.0},
                    "polarization": "TE",
                }
            )


class TestFmmaxRunPrerequisites:
    """Tests for run() prerequisite checks."""

    def test_run_without_geometry_raises(self, source_config):
        """run() without setup_geometry raises RuntimeError."""
        from compass.solvers.rcwa.fmmax_solver import FmmaxSolver

        solver = FmmaxSolver({"name": "fmmax", "params": {}})
        solver.setup_source(source_config)
        with pytest.raises(RuntimeError, match="setup_geometry"):
            solver.run()

    def test_run_without_source_raises(self, simple_pixel_config):
        """run() without setup_source raises RuntimeError."""
        from compass.solvers.rcwa.fmmax_solver import FmmaxSolver
        from compass.geometry.pixel_stack import PixelStack

        solver = FmmaxSolver({"name": "fmmax", "params": {}})
        ps = PixelStack(simple_pixel_config)
        solver.setup_geometry(ps)
        with pytest.raises(RuntimeError, match="setup_source"):
            solver.run()


class TestFmmaxRunWithMocks:
    """Tests for run() with mocked fmmax library."""

    @pytest.mark.usefixtures("_patch_fmmax_imports")
    def test_run_returns_simulation_result(
        self, fmmax_config, simple_pixel_config, source_config
    ):
        """run() returns a valid SimulationResult."""
        from compass.solvers.rcwa.fmmax_solver import FmmaxSolver
        from compass.geometry.pixel_stack import PixelStack

        solver = FmmaxSolver(fmmax_config)
        ps = PixelStack(simple_pixel_config)
        solver.setup_geometry(ps)
        solver.setup_source(source_config)

        result = solver.run()

        assert isinstance(result, SimulationResult)
        assert len(result.wavelengths) == 3
        assert result.reflection is not None
        assert result.transmission is not None
        assert result.absorption is not None
        assert len(result.reflection) == 3
        assert len(result.transmission) == 3
        assert len(result.absorption) == 3

    @pytest.mark.usefixtures("_patch_fmmax_imports")
    def test_run_metadata_contains_solver_info(
        self, fmmax_config, simple_pixel_config, source_config
    ):
        """Metadata contains solver name and formulation."""
        from compass.solvers.rcwa.fmmax_solver import FmmaxSolver
        from compass.geometry.pixel_stack import PixelStack

        solver = FmmaxSolver(fmmax_config)
        ps = PixelStack(simple_pixel_config)
        solver.setup_geometry(ps)
        solver.setup_source(source_config)

        result = solver.run()

        assert result.metadata["solver_name"] == "fmmax"
        assert result.metadata["fmm_formulation"] == "jones"
        assert result.metadata["fourier_order"] == [5, 5]
        assert result.metadata["dtype"] == "complex64"

    @pytest.mark.usefixtures("_patch_fmmax_imports")
    def test_run_qe_per_pixel_keys(
        self, fmmax_config, simple_pixel_config, source_config
    ):
        """QE per pixel should have entries for all pixels in the Bayer map."""
        from compass.solvers.rcwa.fmmax_solver import FmmaxSolver
        from compass.geometry.pixel_stack import PixelStack

        solver = FmmaxSolver(fmmax_config)
        ps = PixelStack(simple_pixel_config)
        solver.setup_geometry(ps)
        solver.setup_source(source_config)

        result = solver.run()

        # 2x2 Bayer: R_0_0, G_0_1, G_1_0, B_1_1
        assert len(result.qe_per_pixel) == 4
        for key, spectrum in result.qe_per_pixel.items():
            assert len(spectrum) == 3  # 3 wavelengths


class TestFmmaxGetFieldDistribution:
    """Tests for get_field_distribution fallback."""

    def test_no_simulation_returns_zeros(self):
        """Without prior run, returns zeros."""
        from compass.solvers.rcwa.fmmax_solver import FmmaxSolver

        solver = FmmaxSolver({"name": "fmmax", "params": {}})
        field = solver.get_field_distribution("|E|2", "xz", 0.0)
        assert isinstance(field, np.ndarray)
        assert field.shape == (64, 64)
        assert np.allclose(field, 0.0)

    @pytest.mark.usefixtures("_patch_fmmax_imports")
    def test_xz_field_after_run(
        self, fmmax_config, simple_pixel_config, source_config
    ):
        """After run, xz field extraction returns non-trivial array."""
        from compass.solvers.rcwa.fmmax_solver import FmmaxSolver
        from compass.geometry.pixel_stack import PixelStack

        solver = FmmaxSolver(fmmax_config)
        ps = PixelStack(simple_pixel_config)
        solver.setup_geometry(ps)
        solver.setup_source(source_config)
        solver.run()

        field = solver.get_field_distribution("|E|2", "xz", 0.0)
        assert isinstance(field, np.ndarray)
        assert field.ndim == 2
        assert field.shape[0] == 64

    @pytest.mark.usefixtures("_patch_fmmax_imports")
    def test_xy_field_after_run(
        self, fmmax_config, simple_pixel_config, source_config
    ):
        """After run, xy field extraction returns non-trivial array."""
        from compass.solvers.rcwa.fmmax_solver import FmmaxSolver
        from compass.geometry.pixel_stack import PixelStack

        solver = FmmaxSolver(fmmax_config)
        ps = PixelStack(simple_pixel_config)
        solver.setup_geometry(ps)
        solver.setup_source(source_config)
        solver.run()

        field = solver.get_field_distribution("|E|2", "xy", 0.5)
        assert isinstance(field, np.ndarray)
        assert field.ndim == 2


class TestFmmaxConfigParams:
    """Tests for configuration parameter handling."""

    def test_all_valid_formulations_accepted(self):
        """All documented FMM formulations should be accepted."""
        from compass.solvers.rcwa.fmmax_solver import FmmaxSolver

        for formulation in ["pol", "normal", "jones", "jonesdirect"]:
            solver = FmmaxSolver(
                {
                    "name": "fmmax",
                    "params": {"fmm_formulation": formulation},
                }
            )
            assert solver._fmm_formulation == formulation

    def test_dtype_complex128(self):
        """complex128 dtype is stored correctly."""
        from compass.solvers.rcwa.fmmax_solver import FmmaxSolver

        solver = FmmaxSolver(
            {
                "name": "fmmax",
                "params": {"dtype": "complex128"},
            }
        )
        assert solver._dtype_str == "complex128"

    def test_missing_params_uses_defaults(self):
        """Missing params section uses default values."""
        from compass.solvers.rcwa.fmmax_solver import FmmaxSolver

        solver = FmmaxSolver({"name": "fmmax"})
        assert solver._fmm_formulation == "jones"
        assert solver._dtype_str == "complex64"
