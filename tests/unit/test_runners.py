"""Unit tests for runners: SingleRunner, SweepRunner, ComparisonRunner, ROISweepRunner."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from compass.core.types import SimulationResult
from compass.runners.comparison_runner import ComparisonRunner
from compass.runners.roi_sweep_runner import ROISweepRunner
from compass.runners.single_run import SingleRunner
from compass.runners.sweep_runner import SweepRunner

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(n_wl=5, runtime=1.0):
    """Create a SimulationResult for mocking."""
    return SimulationResult(
        qe_per_pixel={"R_0_0": np.linspace(0.3, 0.7, n_wl)},
        wavelengths=np.linspace(0.4, 0.7, n_wl),
        reflection=np.full(n_wl, 0.1),
        transmission=np.full(n_wl, 0.3),
        absorption=np.full(n_wl, 0.6),
        metadata={"runtime_seconds": runtime},
    )


def _make_mock_solver(result=None):
    """Create a mock solver with all needed methods."""
    solver = MagicMock()
    if result is None:
        result = _make_result()
    solver.run_timed.return_value = result
    solver.validate_energy_balance.return_value = True
    solver.name = "mock"
    solver.solver_type = "rcwa"
    return solver


def _base_config():
    """Minimal config for SingleRunner."""
    return {
        "pixel": {
            "pitch": 1.0,
            "unit_cell": [2, 2],
            "layers": {},
        },
        "solver": {"name": "test_mock_solver"},
        "source": {"wavelength_range": [0.4, 0.7]},
        "compute": {"backend": "cpu"},
    }


# ---------------------------------------------------------------------------
# SingleRunner Tests
# ---------------------------------------------------------------------------


class TestSingleRunner:
    """Tests for SingleRunner.run."""

    @patch("compass.runners.single_run.SolverFactory")
    @patch("compass.runners.single_run.PixelStack")
    @patch("compass.runners.single_run.MaterialDB")
    def test_run_returns_simulation_result(self, mock_mat_db, mock_ps, mock_factory):
        """SingleRunner.run returns a SimulationResult."""
        expected = _make_result()
        mock_solver = _make_mock_solver(expected)
        mock_factory.create.return_value = mock_solver

        result = SingleRunner.run(_base_config())

        assert isinstance(result, SimulationResult)
        np.testing.assert_array_equal(result.wavelengths, expected.wavelengths)

    @patch("compass.runners.single_run.SolverFactory")
    @patch("compass.runners.single_run.PixelStack")
    @patch("compass.runners.single_run.MaterialDB")
    def test_run_calls_solver_pipeline(self, mock_mat_db, mock_ps, mock_factory):
        """SingleRunner calls setup_geometry, setup_source, run_timed in order."""
        mock_solver = _make_mock_solver()
        mock_factory.create.return_value = mock_solver

        SingleRunner.run(_base_config())

        mock_solver.setup_geometry.assert_called_once()
        mock_solver.setup_source.assert_called_once()
        mock_solver.run_timed.assert_called_once()
        mock_solver.validate_energy_balance.assert_called_once()

    @patch("compass.runners.single_run.SolverFactory")
    @patch("compass.runners.single_run.PixelStack")
    @patch("compass.runners.single_run.MaterialDB")
    def test_run_uses_solver_name_from_config(self, mock_mat_db, mock_ps, mock_factory):
        """Solver name is taken from config.solver.name."""
        mock_factory.create.return_value = _make_mock_solver()
        cfg = _base_config()
        cfg["solver"]["name"] = "grcwa"

        SingleRunner.run(cfg)

        mock_factory.create.assert_called_once_with("grcwa", cfg["solver"], "cpu")

    @patch("compass.runners.single_run.SolverFactory")
    @patch("compass.runners.single_run.PixelStack")
    @patch("compass.runners.single_run.MaterialDB")
    def test_run_default_solver_is_torcwa(self, mock_mat_db, mock_ps, mock_factory):
        """Default solver name is 'torcwa' when not specified."""
        mock_factory.create.return_value = _make_mock_solver()
        cfg = _base_config()
        del cfg["solver"]["name"]

        SingleRunner.run(cfg)

        args = mock_factory.create.call_args
        assert args[0][0] == "torcwa"

    @patch("compass.runners.single_run.SolverFactory")
    @patch("compass.runners.single_run.PixelStack")
    @patch("compass.runners.single_run.MaterialDB")
    def test_run_uses_device_from_compute_config(self, mock_mat_db, mock_ps, mock_factory):
        """Device is read from compute.backend config key."""
        mock_factory.create.return_value = _make_mock_solver()
        cfg = _base_config()
        cfg["compute"]["backend"] = "cuda"

        SingleRunner.run(cfg)

        args = mock_factory.create.call_args
        assert args[0][2] == "cuda"

    @patch("compass.runners.single_run.SolverFactory")
    @patch("compass.runners.single_run.PixelStack")
    @patch("compass.runners.single_run.MaterialDB")
    def test_run_missing_solver_config_uses_defaults(self, mock_mat_db, mock_ps, mock_factory):
        """Missing solver/source/compute keys use empty dicts."""
        mock_factory.create.return_value = _make_mock_solver()
        cfg = {"pixel": {"pitch": 1.0, "unit_cell": [2, 2], "layers": {}}}

        SingleRunner.run(cfg)

        mock_factory.create.assert_called_once()


# ---------------------------------------------------------------------------
# SweepRunner Tests
# ---------------------------------------------------------------------------


class TestSweepRunner:
    """Tests for SweepRunner.run."""

    @patch("compass.runners.sweep_runner.SingleRunner")
    def test_sweep_delegates_to_single_runner(self, mock_single):
        """SweepRunner delegates to SingleRunner.run."""
        expected = _make_result()
        mock_single.run.return_value = expected

        result = SweepRunner.run(_base_config())

        mock_single.run.assert_called_once()
        assert result is expected

    @patch("compass.runners.sweep_runner.SingleRunner")
    def test_sweep_passes_config(self, mock_single):
        """SweepRunner passes the full config to SingleRunner."""
        mock_single.run.return_value = _make_result()
        cfg = _base_config()

        SweepRunner.run(cfg)

        mock_single.run.assert_called_once_with(cfg)


# ---------------------------------------------------------------------------
# ComparisonRunner Tests
# ---------------------------------------------------------------------------


class TestComparisonRunner:
    """Tests for ComparisonRunner.run."""

    @patch("compass.runners.comparison_runner.SolverComparison")
    @patch("compass.runners.comparison_runner.SingleRunner")
    def test_runs_each_solver_config(self, mock_single, mock_comparison):
        """ComparisonRunner runs SingleRunner once per solver config."""
        mock_single.run.return_value = _make_result()
        mock_comparison_instance = MagicMock()
        mock_comparison_instance.summary.return_value = {}
        mock_comparison.return_value = mock_comparison_instance

        solver_configs = [
            {"name": "torcwa"},
            {"name": "grcwa"},
            {"name": "meent"},
        ]
        ComparisonRunner.run(_base_config(), solver_configs)

        assert mock_single.run.call_count == 3

    @patch("compass.runners.comparison_runner.SolverComparison")
    @patch("compass.runners.comparison_runner.SingleRunner")
    def test_returns_results_labels_summary(self, mock_single, mock_comparison):
        """ComparisonRunner returns dict with results, labels, summary."""
        mock_single.run.return_value = _make_result()
        mock_comparison_instance = MagicMock()
        mock_comparison_instance.summary.return_value = {"test": "data"}
        mock_comparison.return_value = mock_comparison_instance

        solver_configs = [{"name": "torcwa"}, {"name": "grcwa"}]
        output = ComparisonRunner.run(_base_config(), solver_configs)

        assert "results" in output
        assert "labels" in output
        assert "summary" in output
        assert len(output["results"]) == 2
        assert output["labels"] == ["torcwa", "grcwa"]

    @patch("compass.runners.comparison_runner.SolverComparison")
    @patch("compass.runners.comparison_runner.SingleRunner")
    def test_labels_from_solver_name(self, mock_single, mock_comparison):
        """Labels come from solver config 'name' key."""
        mock_single.run.return_value = _make_result()
        mock_comparison_instance = MagicMock()
        mock_comparison_instance.summary.return_value = {}
        mock_comparison.return_value = mock_comparison_instance

        solver_configs = [{"name": "solver_a"}, {"name": "solver_b"}]
        output = ComparisonRunner.run(_base_config(), solver_configs)

        assert output["labels"] == ["solver_a", "solver_b"]

    @patch("compass.runners.comparison_runner.SolverComparison")
    @patch("compass.runners.comparison_runner.SingleRunner")
    def test_merges_solver_config_into_base(self, mock_single, mock_comparison):
        """Each solver config is merged into the base config's 'solver' key."""
        mock_single.run.return_value = _make_result()
        mock_comparison_instance = MagicMock()
        mock_comparison_instance.summary.return_value = {}
        mock_comparison.return_value = mock_comparison_instance

        base = _base_config()
        solver_configs = [{"name": "torcwa", "params": {"order": 9}}]
        ComparisonRunner.run(base, solver_configs)

        call_cfg = mock_single.run.call_args[0][0]
        assert call_cfg["solver"]["name"] == "torcwa"
        assert call_cfg["solver"]["params"] == {"order": 9}

    @patch("compass.runners.comparison_runner.SolverComparison")
    @patch("compass.runners.comparison_runner.SingleRunner")
    def test_creates_solver_comparison(self, mock_single, mock_comparison):
        """SolverComparison is instantiated with results and labels."""
        results = [_make_result(), _make_result()]
        mock_single.run.side_effect = results
        mock_comparison_instance = MagicMock()
        mock_comparison_instance.summary.return_value = {}
        mock_comparison.return_value = mock_comparison_instance

        solver_configs = [{"name": "a"}, {"name": "b"}]
        ComparisonRunner.run(_base_config(), solver_configs)

        mock_comparison.assert_called_once()
        call_args = mock_comparison.call_args[0]
        assert len(call_args[0]) == 2  # results list
        assert call_args[1] == ["a", "b"]  # labels


# ---------------------------------------------------------------------------
# ROISweepRunner Tests
# ---------------------------------------------------------------------------


class TestROISweepRunner:
    """Tests for ROISweepRunner."""

    @patch("compass.runners.roi_sweep_runner.SingleRunner")
    def test_runs_for_each_image_height(self, mock_single):
        """ROISweepRunner runs SingleRunner once per image height."""
        mock_single.run.return_value = _make_result()

        roi_config = {"image_heights": [0.0, 0.3, 0.6]}
        results = ROISweepRunner.run(_base_config(), roi_config)

        assert mock_single.run.call_count == 3
        assert "ih_0.00" in results
        assert "ih_0.30" in results
        assert "ih_0.60" in results

    @patch("compass.runners.roi_sweep_runner.SingleRunner")
    def test_default_single_image_height(self, mock_single):
        """Default is single image height at 0.0."""
        mock_single.run.return_value = _make_result()

        results = ROISweepRunner.run(_base_config(), {})

        assert mock_single.run.call_count == 1
        assert "ih_0.00" in results

    @patch("compass.runners.roi_sweep_runner.SingleRunner")
    def test_cra_interpolation(self, mock_single):
        """CRA is interpolated from cra_table for each image height."""
        mock_single.run.return_value = _make_result()

        roi_config = {
            "image_heights": [0.0, 0.5, 1.0],
            "cra_table": [
                {"image_height": 0.0, "cra_deg": 0.0},
                {"image_height": 1.0, "cra_deg": 30.0},
            ],
        }
        ROISweepRunner.run(_base_config(), roi_config)

        # Check that the CRA for ih=0.5 is interpolated to ~15 degrees
        calls = mock_single.run.call_args_list
        second_cfg = calls[1][0][0]
        assert second_cfg["source"]["angle"]["theta_deg"] == pytest.approx(15.0)

    def test_interpolate_cra_empty_table(self):
        """Empty CRA table returns 0.0."""
        cra = ROISweepRunner._interpolate_cra(0.5, [])
        assert cra == 0.0

    def test_interpolate_cra_exact_match(self):
        """Exact image height match returns exact CRA."""
        table = [
            {"image_height": 0.0, "cra_deg": 0.0},
            {"image_height": 0.5, "cra_deg": 15.0},
            {"image_height": 1.0, "cra_deg": 30.0},
        ]
        cra = ROISweepRunner._interpolate_cra(0.5, table)
        assert cra == pytest.approx(15.0)

    def test_interpolate_cra_between_points(self):
        """CRA is linearly interpolated between table entries."""
        table = [
            {"image_height": 0.0, "cra_deg": 0.0},
            {"image_height": 1.0, "cra_deg": 20.0},
        ]
        cra = ROISweepRunner._interpolate_cra(0.25, table)
        assert cra == pytest.approx(5.0)

    @patch("compass.runners.roi_sweep_runner.SingleRunner")
    def test_modifies_microlens_shift_config(self, mock_single):
        """ROISweepRunner sets microlens shift mode to auto_cra."""
        mock_single.run.return_value = _make_result()

        cfg = _base_config()
        cfg["pixel"]["layers"] = {"microlens": {"shift": {}}}

        roi_config = {
            "image_heights": [0.5],
            "cra_table": [
                {"image_height": 0.0, "cra_deg": 0.0},
                {"image_height": 1.0, "cra_deg": 30.0},
            ],
        }
        ROISweepRunner.run(cfg, roi_config)

        call_cfg = mock_single.run.call_args[0][0]
        ml_shift = call_cfg["pixel"]["layers"]["microlens"]["shift"]
        assert ml_shift["mode"] == "auto_cra"
        assert ml_shift["cra_deg"] == pytest.approx(15.0)

    @patch("compass.runners.roi_sweep_runner.SingleRunner")
    def test_results_are_simulation_results(self, mock_single):
        """All returned values are SimulationResult instances."""
        expected = _make_result()
        mock_single.run.return_value = expected

        results = ROISweepRunner.run(_base_config(), {"image_heights": [0.0, 0.5]})

        for _key, val in results.items():
            assert isinstance(val, SimulationResult)
