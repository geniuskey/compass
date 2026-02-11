"""Unit tests for SolverComparison."""

import numpy as np
import pytest

from compass.analysis.solver_comparison import SolverComparison
from compass.core.types import SimulationResult


def _make_result(qe_per_pixel, runtime=0.0):
    """Helper to create a SimulationResult with given QE data and runtime."""
    n_wl = len(next(iter(qe_per_pixel.values())))
    return SimulationResult(
        qe_per_pixel=qe_per_pixel,
        wavelengths=np.linspace(0.4, 0.7, n_wl),
        metadata={"runtime_seconds": runtime},
    )


class TestQEDifference:
    """Tests for SolverComparison.qe_difference."""

    def test_identical_results_zero_diff(self):
        """Identical results give zero difference."""
        qe = {"R_0_0": np.array([0.5, 0.6])}
        r1 = _make_result(qe)
        r2 = _make_result(qe)
        comp = SolverComparison([r1, r2], ["ref", "other"])
        diffs = comp.qe_difference()
        key = "other_vs_ref_R_0_0"
        assert key in diffs
        np.testing.assert_allclose(diffs[key], 0.0, atol=1e-15)

    def test_known_difference(self):
        """Known QE difference is computed correctly."""
        r1 = _make_result({"px": np.array([0.50, 0.40])})
        r2 = _make_result({"px": np.array([0.55, 0.35])})
        comp = SolverComparison([r1, r2], ["A", "B"])
        diffs = comp.qe_difference()
        key = "B_vs_A_px"
        np.testing.assert_allclose(diffs[key], [0.05, 0.05])

    def test_reference_not_in_output(self):
        """Reference solver is not compared against itself."""
        r1 = _make_result({"px": np.array([0.5])})
        r2 = _make_result({"px": np.array([0.6])})
        comp = SolverComparison([r1, r2], ["ref", "other"])
        diffs = comp.qe_difference()
        assert not any("ref_vs_ref" in k for k in diffs)

    def test_multiple_pixels(self):
        """Differences computed for all shared pixels."""
        r1 = _make_result({"A": np.array([0.3]), "B": np.array([0.5])})
        r2 = _make_result({"A": np.array([0.4]), "B": np.array([0.4])})
        comp = SolverComparison([r1, r2], ["ref", "test"])
        diffs = comp.qe_difference()
        assert "test_vs_ref_A" in diffs
        assert "test_vs_ref_B" in diffs

    def test_missing_pixel_in_other_skipped(self):
        """Pixel present in reference but missing in other is skipped."""
        r1 = _make_result({"A": np.array([0.3]), "B": np.array([0.5])})
        r2 = _make_result({"A": np.array([0.4])})  # B missing
        comp = SolverComparison([r1, r2], ["ref", "test"])
        diffs = comp.qe_difference()
        assert "test_vs_ref_A" in diffs
        assert "test_vs_ref_B" not in diffs

    def test_three_solvers(self):
        """Works with three solvers compared against reference."""
        r1 = _make_result({"px": np.array([0.5])})
        r2 = _make_result({"px": np.array([0.6])})
        r3 = _make_result({"px": np.array([0.7])})
        comp = SolverComparison([r1, r2, r3], ["ref", "s2", "s3"])
        diffs = comp.qe_difference()
        assert "s2_vs_ref_px" in diffs
        assert "s3_vs_ref_px" in diffs
        assert diffs["s2_vs_ref_px"][0] == pytest.approx(0.1)
        assert diffs["s3_vs_ref_px"][0] == pytest.approx(0.2)


class TestQERelativeError:
    """Tests for SolverComparison.qe_relative_error."""

    def test_known_relative_error(self):
        """Relative error is correctly computed in percent."""
        r1 = _make_result({"px": np.array([0.50])})
        r2 = _make_result({"px": np.array([0.55])})
        comp = SolverComparison([r1, r2], ["ref", "test"])
        errors = comp.qe_relative_error()
        key = "test_vs_ref_px"
        # 100 * |0.55 - 0.50| / 0.50 = 10%
        assert errors[key][0] == pytest.approx(10.0)

    def test_zero_reference_no_inf(self):
        """Zero reference QE does not produce inf (clamped denominator)."""
        r1 = _make_result({"px": np.array([0.0])})
        r2 = _make_result({"px": np.array([0.1])})
        comp = SolverComparison([r1, r2], ["ref", "test"])
        errors = comp.qe_relative_error()
        assert np.all(np.isfinite(errors["test_vs_ref_px"]))

    def test_identical_zero_error(self):
        """Identical results yield 0% relative error."""
        qe = {"px": np.array([0.4, 0.5])}
        r1 = _make_result(qe)
        r2 = _make_result(qe)
        comp = SolverComparison([r1, r2], ["ref", "test"])
        errors = comp.qe_relative_error()
        np.testing.assert_allclose(errors["test_vs_ref_px"], 0.0, atol=1e-10)


class TestRuntimeComparison:
    """Tests for SolverComparison.runtime_comparison."""

    def test_basic_runtimes(self):
        """Runtime values extracted from metadata."""
        r1 = _make_result({"px": np.array([0.5])}, runtime=1.5)
        r2 = _make_result({"px": np.array([0.5])}, runtime=3.2)
        comp = SolverComparison([r1, r2], ["fast", "slow"])
        runtimes = comp.runtime_comparison()
        assert runtimes["fast"] == 1.5
        assert runtimes["slow"] == 3.2

    def test_missing_runtime_defaults_zero(self):
        """Missing runtime_seconds metadata defaults to 0.0."""
        r1 = SimulationResult(
            qe_per_pixel={"px": np.array([0.5])},
            wavelengths=np.array([0.55]),
            metadata={},
        )
        comp = SolverComparison([r1], ["solver"])
        runtimes = comp.runtime_comparison()
        assert runtimes["solver"] == 0.0


class TestSummary:
    """Tests for SolverComparison.summary."""

    def test_summary_keys(self):
        """Summary dict has expected top-level keys."""
        r1 = _make_result({"px": np.array([0.5, 0.6])}, runtime=1.0)
        r2 = _make_result({"px": np.array([0.55, 0.62])}, runtime=2.0)
        comp = SolverComparison([r1, r2], ["ref", "test"])
        s = comp.summary()
        assert "max_qe_diff" in s
        assert "mean_qe_diff" in s
        assert "max_qe_relative_error_pct" in s
        assert "runtimes_seconds" in s

    def test_summary_values_correct(self):
        """Summary values match manual computation."""
        r1 = _make_result({"px": np.array([0.50, 0.40])}, runtime=1.0)
        r2 = _make_result({"px": np.array([0.55, 0.45])}, runtime=2.5)
        comp = SolverComparison([r1, r2], ["ref", "test"])
        s = comp.summary()
        key = "test_vs_ref_px"
        assert s["max_qe_diff"][key] == pytest.approx(0.05)
        assert s["mean_qe_diff"][key] == pytest.approx(0.05)
        assert s["runtimes_seconds"]["ref"] == 1.0
        assert s["runtimes_seconds"]["test"] == 2.5

    def test_custom_reference_idx(self):
        """Custom reference_idx changes the comparison baseline."""
        r1 = _make_result({"px": np.array([0.5])}, runtime=1.0)
        r2 = _make_result({"px": np.array([0.6])}, runtime=2.0)
        comp = SolverComparison([r1, r2], ["A", "B"], reference_idx=1)
        diffs = comp.qe_difference()
        # A is compared against B (reference)
        assert "A_vs_B_px" in diffs
        assert diffs["A_vs_B_px"][0] == pytest.approx(0.1)
