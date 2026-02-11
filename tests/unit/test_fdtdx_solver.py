"""Unit tests for FDTDX JAX-based 3D FDTD solver adapter.

All tests work WITHOUT fdtdx or jax installed by using mocks.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from compass.core.types import FieldData, SimulationResult
from compass.solvers.base import SolverBase, SolverFactory

# ---------------------------------------------------------------------------
# Helpers: build fake fdtdx + jax ecosystem for import mocking
# ---------------------------------------------------------------------------


def _make_fake_jax():
    """Create a mock jax module with jax.numpy."""
    jax = types.ModuleType("jax")
    jnp = types.ModuleType("jax.numpy")

    jnp.float32 = np.float32
    jnp.float64 = np.float64

    def _jnp_array(data, dtype=None):
        return np.asarray(data, dtype=dtype)

    def _jnp_zeros(shape, dtype=None):
        return np.zeros(shape, dtype=dtype)

    def _jnp_ones(shape, dtype=None):
        return np.ones(shape, dtype=dtype)

    jnp.array = _jnp_array
    jnp.zeros = _jnp_zeros
    jnp.ones = _jnp_ones

    jax.numpy = jnp
    jax.config = MagicMock()
    jax.Array = type("Array", (), {})
    return jax, jnp


def _make_fake_fdtdx(nx=51, ny=51, nz=176):
    """Create a mock fdtdx module that returns plausible field arrays.

    Args:
        nx, ny, nz: Expected grid dimensions for field arrays.
    """
    fdtdx_mod = types.ModuleType("fdtdx")

    def _simulate(**kwargs):
        """Mock simulate function returning field dict."""
        eps = kwargs.get("epsilon")
        if eps is not None:
            shape = eps.shape  # (nz, ny, nx)
        else:
            shape = (nz, ny, nx)

        # Create plausible field data with some non-zero values
        rng = np.random.RandomState(42)
        fields = {
            "ex": rng.randn(*shape).astype(np.float32) * 0.1 + 0.5,
            "ey": rng.randn(*shape).astype(np.float32) * 0.1,
            "ez": rng.randn(*shape).astype(np.float32) * 0.01,
            "hx": rng.randn(*shape).astype(np.float32) * 0.1,
            "hy": rng.randn(*shape).astype(np.float32) * 0.1 + 0.5,
            "hz": rng.randn(*shape).astype(np.float32) * 0.01,
        }
        return fields

    fdtdx_mod.simulate = _simulate
    return fdtdx_mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fdtdx_config():
    """Default FDTDX solver configuration.

    Uses a large grid_spacing (0.5 um) to keep grids small in tests.
    """
    return {
        "name": "fdtdx",
        "type": "fdtd",
        "params": {
            "grid_spacing": 0.5,
            "pml_layers": 3,
            "time_steps": 10,
            "dtype": "float32",
            "courant_factor": 0.5,
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
def source_config_te():
    """Source configuration with TE polarization (single pol run)."""
    return {
        "wavelength": {
            "mode": "list",
            "values": [0.55],
        },
        "angle": {"theta_deg": 0.0, "phi_deg": 0.0},
        "polarization": "TE",
    }


def _mock_get_photodiode_mask(nx, ny, nz):
    """Create a mock return value for PixelStack.get_photodiode_mask.

    Returns a tuple of (full_mask, per_pixel_masks) with small arrays
    that avoid the bitwise_or dtype bug in GeometryBuilder.
    """
    full_mask = np.zeros((ny, nx, nz), dtype=np.float64)
    per_pixel = {
        "R_0_0": np.zeros((ny, nx, nz), dtype=np.float64),
        "G_0_1": np.zeros((ny, nx, nz), dtype=np.float64),
        "G_1_0": np.zeros((ny, nx, nz), dtype=np.float64),
        "B_1_1": np.zeros((ny, nx, nz), dtype=np.float64),
    }
    # Put some absorption in the masks
    for mask in per_pixel.values():
        mask[ny // 2, nx // 2, nz // 2] = 1.0
    return full_mask, per_pixel


@pytest.fixture
def _patch_fdtdx_imports():
    """Patch fdtdx and jax into sys.modules for the duration of a test."""
    jax, jnp = _make_fake_jax()
    fdtdx_mod = _make_fake_fdtdx()

    modules = {
        "jax": jax,
        "jax.numpy": jnp,
        "fdtdx": fdtdx_mod,
    }
    # Patch the module-level flags and mock get_photodiode_mask
    # to avoid GeometryBuilder's bitwise_or dtype bug
    with patch.dict(sys.modules, modules), \
         patch("compass.solvers.fdtd.fdtdx_solver._JAX_AVAILABLE", True), \
         patch("compass.solvers.fdtd.fdtdx_solver._FDTDX_AVAILABLE", True), \
         patch(
             "compass.geometry.pixel_stack.PixelStack.get_photodiode_mask",
             side_effect=lambda nx, ny, nz: _mock_get_photodiode_mask(nx, ny, nz),
         ):
        yield


# ---------------------------------------------------------------------------
# Tests: Registration
# ---------------------------------------------------------------------------


class TestFdtdxSolverRegistration:
    """Tests for SolverFactory registration."""

    def setup_method(self):
        self._saved_registry = SolverFactory._registry.copy()

    def teardown_method(self):
        SolverFactory._registry = self._saved_registry

    def test_fdtdx_registered_in_factory(self):
        """fdtdx should be registered in SolverFactory after import."""
        import compass.solvers.fdtd.fdtdx_solver  # noqa: F401

        assert "fdtdx" in SolverFactory._registry

    def test_factory_create_returns_fdtdx_solver(self):
        """SolverFactory.create('fdtdx', ...) returns an FdtdxSolver."""
        from compass.solvers.fdtd.fdtdx_solver import FdtdxSolver

        SolverFactory.register("fdtdx", FdtdxSolver)

        config = {
            "name": "fdtdx",
            "type": "fdtd",
            "params": {"grid_spacing": 0.02},
        }
        solver = SolverFactory.create("fdtdx", config)
        assert isinstance(solver, SolverBase)
        assert solver.name == "fdtdx"
        assert solver.solver_type == "fdtd"

    def test_fdtdx_in_try_import_map(self):
        """fdtdx should be in the SolverFactory _try_import map."""
        # Access the import map indirectly -- try_import should not raise
        # for a known solver name
        from compass.solvers.base import SolverFactory

        # The real check is that _try_import doesn't crash
        SolverFactory._try_import("fdtdx")


# ---------------------------------------------------------------------------
# Tests: Initialization
# ---------------------------------------------------------------------------


class TestFdtdxSolverInit:
    """Tests for FdtdxSolver initialization."""

    def test_default_params(self):
        """Default parameters are set when config has empty params."""
        from compass.solvers.fdtd.fdtdx_solver import FdtdxSolver

        solver = FdtdxSolver({"name": "fdtdx", "type": "fdtd", "params": {}})
        assert solver._grid_spacing == 0.02
        assert solver._pml_layers == 10
        assert solver._time_steps == 2000
        assert solver._dtype_str == "float32"
        assert solver._courant_factor == 0.5

    def test_custom_params(self):
        """Custom parameters override defaults."""
        from compass.solvers.fdtd.fdtdx_solver import FdtdxSolver

        solver = FdtdxSolver({
            "name": "fdtdx",
            "type": "fdtd",
            "params": {
                "grid_spacing": 0.01,
                "pml_layers": 20,
                "time_steps": 5000,
                "dtype": "float64",
                "courant_factor": 0.3,
            },
        })
        assert solver._grid_spacing == 0.01
        assert solver._pml_layers == 20
        assert solver._time_steps == 5000
        assert solver._dtype_str == "float64"
        assert solver._courant_factor == 0.3

    def test_missing_params_section(self):
        """Missing params section entirely uses defaults."""
        from compass.solvers.fdtd.fdtdx_solver import FdtdxSolver

        solver = FdtdxSolver({"name": "fdtdx"})
        assert solver._grid_spacing == 0.02
        assert solver._pml_layers == 10
        assert solver._time_steps == 2000


# ---------------------------------------------------------------------------
# Tests: setup_geometry validation
# ---------------------------------------------------------------------------


class TestFdtdxSetupGeometry:
    """Tests for setup_geometry validation."""

    def test_none_pixel_stack_raises(self):
        """Passing None as pixel_stack raises ValueError."""
        from compass.solvers.fdtd.fdtdx_solver import FdtdxSolver

        solver = FdtdxSolver({"name": "fdtdx", "params": {}})
        with pytest.raises(ValueError, match="pixel_stack must not be None"):
            solver.setup_geometry(None)

    def test_empty_layers_raises(self):
        """PixelStack with no layers raises ValueError."""
        from compass.solvers.fdtd.fdtdx_solver import FdtdxSolver

        mock_stack = MagicMock()
        mock_stack.layers = []
        solver = FdtdxSolver({"name": "fdtdx", "params": {}})
        with pytest.raises(
            ValueError, match="pixel_stack must have at least one layer"
        ):
            solver.setup_geometry(mock_stack)

    def test_valid_pixel_stack_stored(self, simple_pixel_config):
        """Valid pixel_stack is stored after setup_geometry."""
        from compass.geometry.pixel_stack import PixelStack
        from compass.solvers.fdtd.fdtdx_solver import FdtdxSolver

        solver = FdtdxSolver({"name": "fdtdx", "params": {}})
        ps = PixelStack(simple_pixel_config)
        solver.setup_geometry(ps)
        assert solver._pixel_stack is ps


# ---------------------------------------------------------------------------
# Tests: setup_source validation
# ---------------------------------------------------------------------------


class TestFdtdxSetupSource:
    """Tests for setup_source validation."""

    def test_valid_source_config(self, source_config):
        """Valid source config creates PlanewaveSource."""
        from compass.solvers.fdtd.fdtdx_solver import FdtdxSolver

        solver = FdtdxSolver({"name": "fdtdx", "params": {}})
        solver.setup_source(source_config)
        assert solver._source is not None
        assert solver._source.n_wavelengths == 3

    def test_empty_wavelengths_raises(self):
        """Empty wavelengths array raises ValueError."""
        from compass.solvers.fdtd.fdtdx_solver import FdtdxSolver

        solver = FdtdxSolver({"name": "fdtdx", "params": {}})
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
        from compass.solvers.fdtd.fdtdx_solver import FdtdxSolver

        solver = FdtdxSolver({"name": "fdtdx", "params": {}})
        with pytest.raises(ValueError, match="all wavelengths must be positive"):
            solver.setup_source(
                {
                    "wavelength": {"mode": "list", "values": [-0.5]},
                    "angle": {"theta_deg": 0.0, "phi_deg": 0.0},
                    "polarization": "TE",
                }
            )


# ---------------------------------------------------------------------------
# Tests: run() prerequisites
# ---------------------------------------------------------------------------


class TestFdtdxRunPrerequisites:
    """Tests for run() prerequisite checks."""

    def test_run_without_geometry_raises(self, source_config):
        """run() without setup_geometry raises RuntimeError."""
        from compass.solvers.fdtd.fdtdx_solver import FdtdxSolver

        solver = FdtdxSolver({"name": "fdtdx", "params": {}})
        solver.setup_source(source_config)
        with pytest.raises(RuntimeError, match="setup_geometry"):
            solver.run()

    def test_run_without_source_raises(self, simple_pixel_config):
        """run() without setup_source raises RuntimeError."""
        from compass.geometry.pixel_stack import PixelStack
        from compass.solvers.fdtd.fdtdx_solver import FdtdxSolver

        solver = FdtdxSolver({"name": "fdtdx", "params": {}})
        ps = PixelStack(simple_pixel_config)
        solver.setup_geometry(ps)
        with pytest.raises(RuntimeError, match="setup_source"):
            solver.run()


# ---------------------------------------------------------------------------
# Tests: run() with mocked fdtdx
# ---------------------------------------------------------------------------


class TestFdtdxRunWithMocks:
    """Tests for run() with mocked fdtdx library."""

    @pytest.mark.usefixtures("_patch_fdtdx_imports")
    def test_run_returns_simulation_result(
        self, fdtdx_config, simple_pixel_config, source_config_te
    ):
        """run() returns a valid SimulationResult."""
        from compass.geometry.pixel_stack import PixelStack
        from compass.solvers.fdtd.fdtdx_solver import FdtdxSolver

        solver = FdtdxSolver(fdtdx_config)
        ps = PixelStack(simple_pixel_config)
        solver.setup_geometry(ps)
        solver.setup_source(source_config_te)

        result = solver.run()

        assert isinstance(result, SimulationResult)
        assert len(result.wavelengths) == 1
        assert result.reflection is not None
        assert result.transmission is not None
        assert result.absorption is not None
        assert len(result.reflection) == 1
        assert len(result.transmission) == 1
        assert len(result.absorption) == 1

    @pytest.mark.usefixtures("_patch_fdtdx_imports")
    def test_run_metadata_contains_solver_info(
        self, fdtdx_config, simple_pixel_config, source_config_te
    ):
        """Metadata contains solver name and configuration details."""
        from compass.geometry.pixel_stack import PixelStack
        from compass.solvers.fdtd.fdtdx_solver import FdtdxSolver

        solver = FdtdxSolver(fdtdx_config)
        ps = PixelStack(simple_pixel_config)
        solver.setup_geometry(ps)
        solver.setup_source(source_config_te)

        result = solver.run()

        assert result.metadata["solver_name"] == "fdtdx"
        assert result.metadata["grid_spacing"] == 0.5
        assert result.metadata["pml_layers"] == 3
        assert result.metadata["time_steps"] == 10
        assert result.metadata["courant_factor"] == 0.5
        assert result.metadata["device"] == "cpu"
        assert "grid_size" in result.metadata
        assert len(result.metadata["grid_size"]) == 3

    @pytest.mark.usefixtures("_patch_fdtdx_imports")
    def test_run_qe_per_pixel_keys(
        self, fdtdx_config, simple_pixel_config, source_config_te
    ):
        """QE per pixel should have entries for all pixels in the Bayer map."""
        from compass.geometry.pixel_stack import PixelStack
        from compass.solvers.fdtd.fdtdx_solver import FdtdxSolver

        solver = FdtdxSolver(fdtdx_config)
        ps = PixelStack(simple_pixel_config)
        solver.setup_geometry(ps)
        solver.setup_source(source_config_te)

        result = solver.run()

        # 2x2 Bayer: R_0_0, G_0_1, G_1_0, B_1_1
        assert len(result.qe_per_pixel) == 4
        for _key, spectrum in result.qe_per_pixel.items():
            assert len(spectrum) == 1  # 1 wavelength

    @pytest.mark.usefixtures("_patch_fdtdx_imports")
    def test_run_rta_values_physical(
        self, fdtdx_config, simple_pixel_config, source_config_te
    ):
        """R, T, A values should be in [0, 1] range."""
        from compass.geometry.pixel_stack import PixelStack
        from compass.solvers.fdtd.fdtdx_solver import FdtdxSolver

        solver = FdtdxSolver(fdtdx_config)
        ps = PixelStack(simple_pixel_config)
        solver.setup_geometry(ps)
        solver.setup_source(source_config_te)

        result = solver.run()

        assert np.all(result.reflection >= 0.0)
        assert np.all(result.reflection <= 1.0)
        assert np.all(result.transmission >= 0.0)
        assert np.all(result.transmission <= 1.0)
        assert np.all(result.absorption >= 0.0)
        assert np.all(result.absorption <= 1.0)

    @pytest.mark.usefixtures("_patch_fdtdx_imports")
    def test_run_unpolarized_averages_te_tm(
        self, fdtdx_config, simple_pixel_config, source_config
    ):
        """Unpolarized run should average TE and TM results."""
        from compass.geometry.pixel_stack import PixelStack
        from compass.solvers.fdtd.fdtdx_solver import FdtdxSolver

        solver = FdtdxSolver(fdtdx_config)
        ps = PixelStack(simple_pixel_config)
        solver.setup_geometry(ps)
        solver.setup_source(source_config)

        result = solver.run()

        # Unpolarized with 3 wavelengths should produce 3-element arrays
        assert len(result.reflection) == 3
        assert len(result.transmission) == 3
        assert len(result.absorption) == 3
        # The results should still be physical (averaged)
        assert np.all(result.reflection >= 0.0)
        assert np.all(result.reflection <= 1.0)


# ---------------------------------------------------------------------------
# Tests: NaN/Inf guard
# ---------------------------------------------------------------------------


class TestFdtdxNanInfGuard:
    """Tests for NaN/Inf detection in output."""

    def test_nan_in_fields_triggers_warning(
        self, fdtdx_config, simple_pixel_config, source_config_te
    ):
        """NaN in simulation output should trigger a warning.

        We need NaN to propagate into the final R/T/A arrays. This happens
        when the source region has a valid non-zero power but the monitor
        regions produce NaN from field computations.
        """
        jax, jnp = _make_fake_jax()

        fdtdx_mod = types.ModuleType("fdtdx")

        def _simulate_nan(**kwargs):
            eps = kwargs.get("epsilon")
            shape = eps.shape
            # Create fields where the source region is valid (non-zero)
            # but monitor regions have NaN, so NaN propagates to R/T
            ex = np.ones(shape, dtype=np.float32) * 0.5
            hy = np.ones(shape, dtype=np.float32) * 0.5
            # Inject NaN at reflection and transmission monitor planes
            # The reflection monitor is at pml_layers + source_offset + 2
            # = 3 + 5 + 2 = 10; transmission is at nz - pml - offset - 2
            ex[10, :, :] = np.nan
            hy[10, :, :] = np.nan
            fields = {
                "ex": ex,
                "ey": np.zeros(shape, dtype=np.float32),
                "ez": np.zeros(shape, dtype=np.float32),
                "hx": np.zeros(shape, dtype=np.float32),
                "hy": hy,
                "hz": np.zeros(shape, dtype=np.float32),
            }
            return fields

        fdtdx_mod.simulate = _simulate_nan

        modules = {
            "jax": jax,
            "jax.numpy": jnp,
            "fdtdx": fdtdx_mod,
        }
        with patch.dict(sys.modules, modules), \
             patch("compass.solvers.fdtd.fdtdx_solver._JAX_AVAILABLE", True), \
             patch("compass.solvers.fdtd.fdtdx_solver._FDTDX_AVAILABLE", True), \
             patch(
                 "compass.geometry.pixel_stack.PixelStack.get_photodiode_mask",
                 side_effect=lambda nx, ny, nz: _mock_get_photodiode_mask(nx, ny, nz),
             ):
            from compass.geometry.pixel_stack import PixelStack
            from compass.solvers.fdtd.fdtdx_solver import FdtdxSolver

            solver = FdtdxSolver(fdtdx_config)
            ps = PixelStack(simple_pixel_config)
            solver.setup_geometry(ps)
            solver.setup_source(source_config_te)

            with pytest.warns(UserWarning, match="NaN/Inf detected"):
                solver.run()


# ---------------------------------------------------------------------------
# Tests: get_field_distribution
# ---------------------------------------------------------------------------


class TestFdtdxGetFieldDistribution:
    """Tests for get_field_distribution fallback."""

    def test_no_simulation_returns_zeros(self):
        """Without prior run, returns zeros."""
        from compass.solvers.fdtd.fdtdx_solver import FdtdxSolver

        solver = FdtdxSolver({"name": "fdtdx", "params": {}})
        field = solver.get_field_distribution("|E|2", "xz", 0.0)
        assert isinstance(field, np.ndarray)
        assert field.shape == (64, 64)
        assert np.allclose(field, 0.0)

    def test_no_fields_returns_zeros_xy(self):
        """Without field data, xy plane returns zeros."""
        from compass.solvers.fdtd.fdtdx_solver import FdtdxSolver

        solver = FdtdxSolver({"name": "fdtdx", "params": {}})
        field = solver.get_field_distribution("|E|2", "xy", 0.5)
        assert isinstance(field, np.ndarray)
        assert field.shape == (64, 64)
        assert np.allclose(field, 0.0)

    def test_no_fields_returns_zeros_yz(self):
        """Without field data, yz plane returns zeros."""
        from compass.solvers.fdtd.fdtdx_solver import FdtdxSolver

        solver = FdtdxSolver({"name": "fdtdx", "params": {}})
        field = solver.get_field_distribution("Ex", "yz", 0.0)
        assert isinstance(field, np.ndarray)
        assert field.shape == (64, 64)
        assert np.allclose(field, 0.0)

    def test_field_with_stored_data(self):
        """With stored field data, returns non-trivial slice."""
        from compass.solvers.fdtd.fdtdx_solver import FdtdxSolver

        solver = FdtdxSolver({"name": "fdtdx", "params": {}})

        # Manually inject field data
        rng = np.random.RandomState(99)
        fd = FieldData(
            Ex=rng.randn(10, 12, 20).astype(np.complex64),
            Ey=rng.randn(10, 12, 20).astype(np.complex64),
            Ez=rng.randn(10, 12, 20).astype(np.complex64),
            x=np.linspace(0, 2, 12),
            y=np.linspace(0, 2, 10),
            z=np.linspace(0, 4, 20),
        )
        solver._last_fields = {"wl_0.55": fd}

        field = solver.get_field_distribution("|E|2", "xz", 0.0)
        assert isinstance(field, np.ndarray)
        assert field.ndim == 2
        # xz slice at fixed y: shape should be (nx, nz) = (12, 20)
        assert field.shape == (12, 20)


# ---------------------------------------------------------------------------
# Tests: Config parameter handling
# ---------------------------------------------------------------------------


class TestFdtdxConfigParams:
    """Tests for configuration parameter handling."""

    def test_grid_spacing_stored(self):
        """grid_spacing from config is stored."""
        from compass.solvers.fdtd.fdtdx_solver import FdtdxSolver

        solver = FdtdxSolver({
            "name": "fdtdx",
            "params": {"grid_spacing": 0.05},
        })
        assert solver._grid_spacing == 0.05

    def test_pml_layers_stored(self):
        """pml_layers from config is stored."""
        from compass.solvers.fdtd.fdtdx_solver import FdtdxSolver

        solver = FdtdxSolver({
            "name": "fdtdx",
            "params": {"pml_layers": 15},
        })
        assert solver._pml_layers == 15

    def test_time_steps_stored(self):
        """time_steps from config is stored."""
        from compass.solvers.fdtd.fdtdx_solver import FdtdxSolver

        solver = FdtdxSolver({
            "name": "fdtdx",
            "params": {"time_steps": 3000},
        })
        assert solver._time_steps == 3000

    def test_courant_factor_stored(self):
        """courant_factor from config is stored."""
        from compass.solvers.fdtd.fdtdx_solver import FdtdxSolver

        solver = FdtdxSolver({
            "name": "fdtdx",
            "params": {"courant_factor": 0.4},
        })
        assert solver._courant_factor == 0.4

    def test_dtype_stored(self):
        """dtype from config is stored."""
        from compass.solvers.fdtd.fdtdx_solver import FdtdxSolver

        solver = FdtdxSolver({
            "name": "fdtdx",
            "params": {"dtype": "float64"},
        })
        assert solver._dtype_str == "float64"

    def test_solver_name_property(self):
        """Solver name property returns config name."""
        from compass.solvers.fdtd.fdtdx_solver import FdtdxSolver

        solver = FdtdxSolver({"name": "fdtdx", "type": "fdtd", "params": {}})
        assert solver.name == "fdtdx"
        assert solver.solver_type == "fdtd"
