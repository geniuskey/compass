"""Unit tests for SolverBase ABC and SolverFactory."""

import numpy as np
import pytest

from compass.core.types import SimulationResult
from compass.solvers.base import SolverBase, SolverFactory

# ---------------------------------------------------------------------------
# Concrete mock subclass for testing
# ---------------------------------------------------------------------------

class MockSolver(SolverBase):
    """Concrete subclass of SolverBase for testing."""

    def __init__(self, config: dict, device: str = "cpu"):
        super().__init__(config, device)
        self._geometry_set = False
        self._source_set = False
        self._run_called = False

    def setup_geometry(self, pixel_stack) -> None:
        self._pixel_stack = pixel_stack
        self._geometry_set = True

    def setup_source(self, source_config: dict) -> None:
        self._source_config = source_config
        self._source_set = True

    def run(self) -> SimulationResult:
        self._run_called = True
        return SimulationResult(
            qe_per_pixel={"R_0_0": np.array([0.5, 0.6, 0.7])},
            wavelengths=np.array([0.45, 0.55, 0.65]),
            reflection=np.array([0.1, 0.15, 0.2]),
            transmission=np.array([0.3, 0.25, 0.2]),
            absorption=np.array([0.6, 0.6, 0.6]),
            metadata={},
        )

    def get_field_distribution(
        self, component: str, plane: str, position: float
    ) -> np.ndarray:
        return np.ones((10, 10))


class FailingSolver(SolverBase):
    """Solver that raises on run."""

    def setup_geometry(self, pixel_stack) -> None:
        pass

    def setup_source(self, source_config: dict) -> None:
        pass

    def run(self) -> SimulationResult:
        raise RuntimeError("Intentional failure")

    def get_field_distribution(
        self, component: str, plane: str, position: float
    ) -> np.ndarray:
        return np.zeros((1, 1))


# ---------------------------------------------------------------------------
# SolverBase Tests
# ---------------------------------------------------------------------------


class TestSolverBaseInit:
    """Tests for SolverBase initialization and properties."""

    def test_init_stores_config_and_device(self):
        """Config and device are stored on the instance."""
        cfg = {"name": "test_solver", "type": "rcwa"}
        solver = MockSolver(cfg, device="cuda")
        assert solver.config is cfg
        assert solver.device == "cuda"

    def test_default_device_is_cpu(self):
        """Default device should be 'cpu'."""
        solver = MockSolver({})
        assert solver.device == "cpu"

    def test_pixel_stack_initially_none(self):
        """_pixel_stack should be None before setup_geometry."""
        solver = MockSolver({})
        assert solver._pixel_stack is None

    def test_source_config_initially_none(self):
        """_source_config should be None before setup_source."""
        solver = MockSolver({})
        assert solver._source_config is None


class TestSolverBaseProperties:
    """Tests for name and solver_type properties."""

    def test_name_from_config(self):
        """Name is read from config 'name' key."""
        solver = MockSolver({"name": "my_solver"})
        assert solver.name == "my_solver"

    def test_name_fallback_to_class_name(self):
        """Name falls back to class name when config has no 'name' key."""
        solver = MockSolver({})
        assert solver.name == "MockSolver"

    def test_solver_type_from_config(self):
        """Solver type is read from config 'type' key."""
        solver = MockSolver({"type": "fdtd"})
        assert solver.solver_type == "fdtd"

    def test_solver_type_default_unknown(self):
        """Solver type defaults to 'unknown' when not in config."""
        solver = MockSolver({})
        assert solver.solver_type == "unknown"


class TestSolverBaseAbstractMethods:
    """Tests for abstract method interface."""

    def test_setup_geometry_stores_pixel_stack(self):
        """setup_geometry stores the pixel stack."""
        solver = MockSolver({})
        mock_stack = object()
        solver.setup_geometry(mock_stack)
        assert solver._pixel_stack is mock_stack
        assert solver._geometry_set is True

    def test_setup_source_stores_config(self):
        """setup_source stores source configuration."""
        solver = MockSolver({})
        src = {"wavelength": 0.55}
        solver.setup_source(src)
        assert solver._source_config is src
        assert solver._source_set is True

    def test_run_returns_simulation_result(self):
        """run() returns a SimulationResult."""
        solver = MockSolver({})
        result = solver.run()
        assert isinstance(result, SimulationResult)
        assert solver._run_called is True

    def test_get_field_distribution_returns_array(self):
        """get_field_distribution returns a numpy array."""
        solver = MockSolver({})
        field = solver.get_field_distribution("Ex", "xy", 0.0)
        assert isinstance(field, np.ndarray)
        assert field.shape == (10, 10)


class TestValidateEnergyBalance:
    """Tests for SolverBase.validate_energy_balance."""

    def test_perfect_conservation(self):
        """R + T + A = 1 exactly passes."""
        solver = MockSolver({})
        result = SimulationResult(
            qe_per_pixel={},
            wavelengths=np.array([0.55]),
            reflection=np.array([0.2]),
            transmission=np.array([0.3]),
            absorption=np.array([0.5]),
        )
        assert solver.validate_energy_balance(result) is True

    def test_small_violation_within_tolerance(self):
        """Small violation within default 1% tolerance passes."""
        solver = MockSolver({})
        result = SimulationResult(
            qe_per_pixel={},
            wavelengths=np.array([0.55]),
            reflection=np.array([0.2]),
            transmission=np.array([0.3]),
            absorption=np.array([0.505]),  # total = 1.005
        )
        assert solver.validate_energy_balance(result) is True

    def test_large_violation_fails(self):
        """Large violation exceeding tolerance fails."""
        solver = MockSolver({})
        result = SimulationResult(
            qe_per_pixel={},
            wavelengths=np.array([0.55]),
            reflection=np.array([0.5]),
            transmission=np.array([0.5]),
            absorption=np.array([0.5]),  # total = 1.5
        )
        assert solver.validate_energy_balance(result) is False

    def test_custom_tolerance(self):
        """Custom tolerance is respected."""
        solver = MockSolver({})
        result = SimulationResult(
            qe_per_pixel={},
            wavelengths=np.array([0.55]),
            reflection=np.array([0.4]),
            transmission=np.array([0.4]),
            absorption=np.array([0.22]),  # total = 1.02
        )
        assert solver.validate_energy_balance(result, tolerance=0.05) is True
        assert solver.validate_energy_balance(result, tolerance=0.01) is False

    def test_missing_reflection_returns_true(self):
        """Missing reflection data returns True (skip check)."""
        solver = MockSolver({})
        result = SimulationResult(
            qe_per_pixel={},
            wavelengths=np.array([0.55]),
            reflection=None,
            transmission=np.array([0.5]),
        )
        assert solver.validate_energy_balance(result) is True

    def test_missing_transmission_returns_true(self):
        """Missing transmission data returns True (skip check)."""
        solver = MockSolver({})
        result = SimulationResult(
            qe_per_pixel={},
            wavelengths=np.array([0.55]),
            reflection=np.array([0.5]),
            transmission=None,
        )
        assert solver.validate_energy_balance(result) is True

    def test_no_absorption_only_rt(self):
        """When absorption is None, only R + T is checked."""
        solver = MockSolver({})
        result = SimulationResult(
            qe_per_pixel={},
            wavelengths=np.array([0.55]),
            reflection=np.array([0.3]),
            transmission=np.array([0.3]),
            absorption=None,
        )
        # R + T = 0.6, total = 0.6, |0.6 - 1| = 0.4 > 0.01
        assert solver.validate_energy_balance(result) is False


class TestRunTimed:
    """Tests for SolverBase.run_timed."""

    def test_run_timed_adds_metadata(self):
        """run_timed adds runtime and solver metadata."""
        solver = MockSolver({"name": "my_test", "type": "rcwa"}, device="cpu")
        result = solver.run_timed()
        assert "runtime_seconds" in result.metadata
        assert result.metadata["runtime_seconds"] > 0
        assert result.metadata["solver_name"] == "my_test"
        assert result.metadata["solver_type"] == "rcwa"
        assert result.metadata["device"] == "cpu"

    def test_run_timed_returns_simulation_result(self):
        """run_timed returns a SimulationResult."""
        solver = MockSolver({})
        result = solver.run_timed()
        assert isinstance(result, SimulationResult)
        assert len(result.wavelengths) == 3


# ---------------------------------------------------------------------------
# SolverFactory Tests
# ---------------------------------------------------------------------------


class TestSolverFactory:
    """Tests for SolverFactory register/create pattern."""

    def setup_method(self):
        """Clean registry before each test to avoid cross-contamination."""
        self._saved_registry = SolverFactory._registry.copy()

    def teardown_method(self):
        """Restore original registry after each test."""
        SolverFactory._registry = self._saved_registry

    def test_register_and_create(self):
        """Register a solver and create it by name."""
        SolverFactory.register("mock_test", MockSolver)
        solver = SolverFactory.create("mock_test", {"name": "mock_test"})
        assert isinstance(solver, MockSolver)
        assert solver.name == "mock_test"

    def test_create_passes_config_and_device(self):
        """Config and device are passed through to the solver constructor."""
        SolverFactory.register("mock_test2", MockSolver)
        cfg = {"name": "test", "foo": "bar"}
        solver = SolverFactory.create("mock_test2", cfg, device="cuda")
        assert solver.config is cfg
        assert solver.device == "cuda"

    def test_create_unknown_raises_valueerror(self):
        """Creating an unregistered solver raises ValueError."""
        with pytest.raises(ValueError, match="Unknown solver"):
            SolverFactory.create("nonexistent_solver_xyz", {})

    def test_list_solvers(self):
        """list_solvers returns registered solver names."""
        SolverFactory.register("mock_list_test", MockSolver)
        solvers = SolverFactory.list_solvers()
        assert "mock_list_test" in solvers

    def test_register_overwrites_existing(self):
        """Registering with same name overwrites previous solver class."""
        SolverFactory.register("overwrite_test", MockSolver)
        SolverFactory.register("overwrite_test", FailingSolver)
        solver = SolverFactory.create("overwrite_test", {})
        assert isinstance(solver, FailingSolver)

    def test_create_error_message_lists_available(self):
        """ValueError message includes available solver names."""
        SolverFactory.register("available_one", MockSolver)
        with pytest.raises(ValueError) as exc_info:
            SolverFactory.create("bad_name", {})
        assert "available_one" in str(exc_info.value)
