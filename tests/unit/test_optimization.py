"""Unit tests for the COMPASS optimization / inverse design module."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from compass.core.types import SimulationResult
from compass.optimization.history import OptimizationHistory
from compass.optimization.objectives import (
    CompositeObjective,
    EnergyBalanceRegularizer,
    MaximizePeakQE,
    MaximizeQE,
    MinimizeCrosstalk,
)
from compass.optimization.optimizer import OptimizationResult, PixelOptimizer
from compass.optimization.parameters import (
    BARLThicknesses,
    ColorFilterThickness,
    MicrolensHeight,
    MicrolensRadii,
    MicrolensSquareness,
    ParameterSpace,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(
    n_wl: int = 5,
    qe_value: float = 0.5,
    n_pixels: int = 4,
    reflection: float = 0.1,
    transmission: float = 0.3,
) -> SimulationResult:
    """Create a SimulationResult with controllable QE and energy balance."""
    wl = np.linspace(0.4, 0.7, n_wl)
    colors = ["R", "G", "G", "B"]
    qe_per_pixel = {}
    for i in range(n_pixels):
        c = colors[i % len(colors)]
        r = i // 2
        col = i % 2
        name = f"{c}_{r}_{col}"
        qe_per_pixel[name] = np.full(n_wl, qe_value)
    absorption = 1.0 - reflection - transmission
    return SimulationResult(
        qe_per_pixel=qe_per_pixel,
        wavelengths=wl,
        reflection=np.full(n_wl, reflection),
        transmission=np.full(n_wl, transmission),
        absorption=np.full(n_wl, absorption),
        metadata={},
    )


def _base_config() -> dict:
    """Minimal config dict for parameter tests."""
    return {
        "pixel": {
            "pitch": 1.0,
            "unit_cell": [2, 2],
            "layers": {
                "microlens": {
                    "height": 0.6,
                    "radius_x": 0.48,
                    "radius_y": 0.48,
                    "profile": {"n": 2.5},
                },
                "color_filter": {
                    "thickness": 0.6,
                },
                "barl": {
                    "layers": [
                        {"thickness": 0.02, "material": "sin"},
                        {"thickness": 0.03, "material": "sio2"},
                    ],
                },
            },
        },
        "solver": {"name": "meent"},
        "source": {},
        "compute": {"backend": "cpu"},
    }


# ===========================================================================
# Objective Function Tests
# ===========================================================================


class TestMaximizeQE:
    """Tests for MaximizeQE objective."""

    def test_returns_negative_mean_qe(self):
        """MaximizeQE returns negative of mean QE for minimization."""
        result = _make_result(qe_value=0.5)
        obj = MaximizeQE()
        val = obj.evaluate(result)
        assert val == pytest.approx(-0.5)

    def test_wavelength_range_filter(self):
        """Only wavelengths inside the range are included."""
        result = _make_result(n_wl=10)
        # Wavelengths span 0.4 to 0.7
        obj_full = MaximizeQE()
        obj_half = MaximizeQE(wavelength_range=(0.5, 0.6))
        val_full = obj_full.evaluate(result)
        val_half = obj_half.evaluate(result)
        # Both should be -0.5 since QE is uniform
        assert val_full == pytest.approx(-0.5)
        assert val_half == pytest.approx(-0.5)

    def test_target_pixels_filter(self):
        """Only specified pixels contribute to the average."""
        result = _make_result(qe_value=0.5, n_pixels=4)
        # Override one pixel to have different QE
        first_key = next(iter(result.qe_per_pixel.keys()))
        result.qe_per_pixel[first_key] = np.full(5, 0.9)
        obj = MaximizeQE(target_pixels=[first_key])
        val = obj.evaluate(result)
        assert val == pytest.approx(-0.9)

    def test_name(self):
        obj = MaximizeQE()
        assert obj.name() == "MaximizeQE"

    def test_empty_pixels_returns_zero(self):
        """No matching pixels returns 0."""
        result = _make_result()
        obj = MaximizeQE(target_pixels=["nonexistent_0_0"])
        assert obj.evaluate(result) == 0.0


class TestMinimizeCrosstalk:
    """Tests for MinimizeCrosstalk objective."""

    def test_uniform_qe_crosstalk(self):
        """With uniform QE, each pixel gets 1/n fraction so crosstalk = 1 - 1/n."""
        result = _make_result(qe_value=0.5, n_pixels=4)
        obj = MinimizeCrosstalk()
        val = obj.evaluate(result)
        # Each pixel fraction = 0.25, crosstalk = 1 - 0.25 = 0.75
        assert val == pytest.approx(0.75)

    def test_single_pixel_no_crosstalk(self):
        """Single pixel has no crosstalk (returns 0)."""
        result = _make_result(n_pixels=1)
        obj = MinimizeCrosstalk()
        assert obj.evaluate(result) == 0.0

    def test_name(self):
        obj = MinimizeCrosstalk()
        assert obj.name() == "MinimizeCrosstalk"


class TestMaximizePeakQE:
    """Tests for MaximizePeakQE objective."""

    def test_returns_negative_peak(self):
        """Returns negative of the peak QE for the channel."""
        result = _make_result(qe_value=0.5)
        # Set green pixels to have a peak
        for key in result.qe_per_pixel:
            if key.startswith("G_"):
                qe = np.full(5, 0.5)
                qe[2] = 0.8  # peak at middle wavelength
                result.qe_per_pixel[key] = qe
        obj = MaximizePeakQE(channel="G")
        val = obj.evaluate(result)
        assert val == pytest.approx(-0.8)

    def test_channel_not_found(self):
        """Channel with no matching pixels returns 0."""
        result = _make_result()
        obj = MaximizePeakQE(channel="W")
        assert obj.evaluate(result) == 0.0

    def test_name_includes_channel(self):
        obj = MaximizePeakQE(channel="R")
        assert "R" in obj.name()


class TestEnergyBalanceRegularizer:
    """Tests for EnergyBalanceRegularizer."""

    def test_good_balance_no_penalty(self):
        """Zero penalty when R+T+A=1 within tolerance."""
        result = _make_result(reflection=0.1, transmission=0.3)
        obj = EnergyBalanceRegularizer(tolerance=0.01)
        assert obj.evaluate(result) == 0.0

    def test_bad_balance_nonzero_penalty(self):
        """Nonzero penalty when R+T+A deviates from 1."""
        result = _make_result()
        # Force an energy imbalance
        result.reflection = np.full(5, 0.5)
        result.transmission = np.full(5, 0.5)
        result.absorption = np.full(5, 0.5)
        # total = 1.5, error = 0.5
        obj = EnergyBalanceRegularizer(tolerance=0.01, penalty_weight=10.0)
        val = obj.evaluate(result)
        expected = 10.0 * (0.5 - 0.01) ** 2
        assert val == pytest.approx(expected)

    def test_missing_reflection_returns_zero(self):
        """No penalty when reflection data is missing."""
        result = _make_result()
        result.reflection = None
        obj = EnergyBalanceRegularizer()
        assert obj.evaluate(result) == 0.0

    def test_name(self):
        obj = EnergyBalanceRegularizer()
        assert obj.name() == "EnergyBalanceRegularizer"


class TestCompositeObjective:
    """Tests for CompositeObjective."""

    def test_weighted_sum(self):
        """Composite returns weighted sum of components."""
        result = _make_result(qe_value=0.5, n_pixels=4)
        obj1 = MaximizeQE()  # returns -0.5
        obj2 = MinimizeCrosstalk()  # returns 0.75
        composite = CompositeObjective([(1.0, obj1), (0.5, obj2)])
        val = composite.evaluate(result)
        expected = 1.0 * (-0.5) + 0.5 * 0.75
        assert val == pytest.approx(expected)

    def test_empty_raises(self):
        """Empty objective list raises ValueError."""
        with pytest.raises(ValueError):
            CompositeObjective([])

    def test_name_shows_components(self):
        obj1 = MaximizeQE()
        obj2 = MinimizeCrosstalk()
        composite = CompositeObjective([(1.0, obj1), (0.5, obj2)])
        name = composite.name()
        assert "MaximizeQE" in name
        assert "MinimizeCrosstalk" in name
        assert "Composite" in name


# ===========================================================================
# Parameter Tests
# ===========================================================================


class TestMicrolensHeight:
    """Tests for MicrolensHeight parameter."""

    def test_get_value(self):
        cfg = _base_config()
        p = MicrolensHeight(cfg)
        val = p.get_value()
        assert val == pytest.approx([0.6])

    def test_set_value(self):
        cfg = _base_config()
        p = MicrolensHeight(cfg)
        p.set_value(np.array([0.8]))
        assert cfg["pixel"]["layers"]["microlens"]["height"] == pytest.approx(0.8)

    def test_bounds_clipping(self):
        """Values outside bounds are clipped."""
        cfg = _base_config()
        p = MicrolensHeight(cfg, min_val=0.2, max_val=1.0)
        p.set_value(np.array([2.0]))
        assert cfg["pixel"]["layers"]["microlens"]["height"] == pytest.approx(1.0)
        p.set_value(np.array([0.01]))
        assert cfg["pixel"]["layers"]["microlens"]["height"] == pytest.approx(0.2)

    def test_size_is_one(self):
        p = MicrolensHeight(_base_config())
        assert p.size == 1

    def test_name(self):
        p = MicrolensHeight(_base_config())
        assert p.name == "microlens_height"


class TestMicrolensSquareness:
    """Tests for MicrolensSquareness parameter."""

    def test_get_set_roundtrip(self):
        cfg = _base_config()
        p = MicrolensSquareness(cfg)
        p.set_value(np.array([3.5]))
        assert p.get_value() == pytest.approx([3.5])


class TestBARLThicknesses:
    """Tests for BARLThicknesses parameter."""

    def test_get_returns_all_layer_thicknesses(self):
        cfg = _base_config()
        p = BARLThicknesses(cfg)
        val = p.get_value()
        np.testing.assert_allclose(val, [0.02, 0.03])

    def test_set_updates_each_layer(self):
        cfg = _base_config()
        p = BARLThicknesses(cfg)
        p.set_value(np.array([0.04, 0.05]))
        layers = cfg["pixel"]["layers"]["barl"]["layers"]
        assert layers[0]["thickness"] == pytest.approx(0.04)
        assert layers[1]["thickness"] == pytest.approx(0.05)

    def test_size_matches_layers(self):
        cfg = _base_config()
        p = BARLThicknesses(cfg)
        assert p.size == 2

    def test_empty_barl(self):
        cfg = _base_config()
        cfg["pixel"]["layers"]["barl"]["layers"] = []
        p = BARLThicknesses(cfg)
        assert p.size == 0
        assert len(p.get_value()) == 0


class TestColorFilterThickness:
    """Tests for ColorFilterThickness parameter."""

    def test_get_set(self):
        cfg = _base_config()
        p = ColorFilterThickness(cfg)
        assert p.get_value() == pytest.approx([0.6])
        p.set_value(np.array([0.8]))
        assert cfg["pixel"]["layers"]["color_filter"]["thickness"] == pytest.approx(0.8)


class TestMicrolensRadii:
    """Tests for MicrolensRadii parameter."""

    def test_size_is_two(self):
        p = MicrolensRadii(_base_config())
        assert p.size == 2

    def test_get_returns_rx_ry(self):
        cfg = _base_config()
        p = MicrolensRadii(cfg)
        np.testing.assert_allclose(p.get_value(), [0.48, 0.48])

    def test_set_updates_both(self):
        cfg = _base_config()
        p = MicrolensRadii(cfg)
        p.set_value(np.array([0.35, 0.40]))
        assert cfg["pixel"]["layers"]["microlens"]["radius_x"] == pytest.approx(0.35)
        assert cfg["pixel"]["layers"]["microlens"]["radius_y"] == pytest.approx(0.40)


# ===========================================================================
# ParameterSpace Tests
# ===========================================================================


class TestParameterSpace:
    """Tests for ParameterSpace vector conversion and bounds."""

    def test_to_vector_from_vector_roundtrip(self):
        """Converting to vector and back preserves values."""
        cfg = _base_config()
        ps = ParameterSpace([
            MicrolensHeight(cfg),
            MicrolensSquareness(cfg),
        ])
        v = ps.to_vector()
        np.testing.assert_allclose(v, [0.6, 2.5])

        # Modify and convert back
        new_v = np.array([0.8, 3.0])
        ps.from_vector(new_v)
        np.testing.assert_allclose(ps.to_vector(), [0.8, 3.0])

    def test_total_size(self):
        cfg = _base_config()
        ps = ParameterSpace([
            MicrolensHeight(cfg),
            BARLThicknesses(cfg),  # 2 layers
            MicrolensRadii(cfg),   # 2 radii
        ])
        assert ps.total_size == 5  # 1 + 2 + 2

    def test_bounds_concatenation(self):
        cfg = _base_config()
        ps = ParameterSpace([
            MicrolensHeight(cfg, min_val=0.1, max_val=1.5),
            ColorFilterThickness(cfg, min_val=0.3, max_val=1.2),
        ])
        lo, hi = ps.get_bounds()
        np.testing.assert_allclose(lo, [0.1, 0.3])
        np.testing.assert_allclose(hi, [1.5, 1.2])

    def test_from_vector_wrong_length_raises(self):
        cfg = _base_config()
        ps = ParameterSpace([MicrolensHeight(cfg)])
        with pytest.raises(ValueError, match="Vector length"):
            ps.from_vector(np.array([1.0, 2.0]))

    def test_empty_params_raises(self):
        with pytest.raises(ValueError):
            ParameterSpace([])

    def test_names(self):
        cfg = _base_config()
        ps = ParameterSpace([
            MicrolensHeight(cfg),
            MicrolensSquareness(cfg),
        ])
        assert ps.names == ["microlens_height", "microlens_squareness"]

    def test_describe(self):
        cfg = _base_config()
        ps = ParameterSpace([MicrolensHeight(cfg, min_val=0.1, max_val=1.5)])
        desc = ps.describe()
        assert len(desc) == 1
        assert desc[0]["name"] == "microlens_height"
        assert desc[0]["size"] == 1
        assert desc[0]["current_value"] == [pytest.approx(0.6)]


# ===========================================================================
# OptimizationHistory Tests
# ===========================================================================


class TestOptimizationHistory:
    """Tests for OptimizationHistory."""

    def test_record_and_retrieve(self):
        h = OptimizationHistory()
        h.record(0, np.array([1.0, 2.0]), 0.5)
        h.record(1, np.array([1.1, 2.1]), 0.4)
        assert h.n_records == 2
        rec = h.get(0)
        assert rec["iteration"] == 0
        assert rec["objective"] == pytest.approx(0.5)

    def test_best(self):
        h = OptimizationHistory()
        h.record(0, np.array([1.0]), 0.5)
        h.record(1, np.array([1.1]), 0.3)
        h.record(2, np.array([1.2]), 0.4)
        best = h.best()
        assert best["iteration"] == 1
        assert best["objective"] == pytest.approx(0.3)

    def test_best_empty(self):
        h = OptimizationHistory()
        assert h.best() == {}

    def test_objectives_list(self):
        h = OptimizationHistory()
        h.record(0, np.array([1.0]), 0.5)
        h.record(1, np.array([1.1]), 0.3)
        assert h.objectives() == [pytest.approx(0.5), pytest.approx(0.3)]

    def test_save_and_load(self, tmp_path):
        """Round-trip save/load preserves records."""
        h = OptimizationHistory()
        h.record(0, np.array([1.0, 2.0]), 0.5, metadata={"elapsed": 1.2})
        h.record(1, np.array([1.1, 2.1]), 0.4, metadata={"elapsed": 2.5})

        fpath = tmp_path / "history.json"
        h.save(fpath)

        loaded = OptimizationHistory.load(fpath)
        assert loaded.n_records == 2
        assert loaded.get(0)["objective"] == pytest.approx(0.5)
        np.testing.assert_allclose(loaded.get(1)["params"], [1.1, 2.1])

    def test_to_dict(self):
        h = OptimizationHistory()
        h.record(0, np.array([1.0]), 0.5)
        d = h.to_dict()
        assert d["n_records"] == 1
        assert len(d["records"]) == 1

    def test_params_history(self):
        h = OptimizationHistory()
        h.record(0, np.array([1.0, 2.0]), 0.5)
        h.record(1, np.array([3.0, 4.0]), 0.3)
        ph = h.params_history()
        assert len(ph) == 2
        assert ph[0] == [pytest.approx(1.0), pytest.approx(2.0)]


# ===========================================================================
# PixelOptimizer Tests
# ===========================================================================


class TestPixelOptimizer:
    """Tests for PixelOptimizer with mocked solver."""

    @patch("compass.optimization.optimizer.SingleRunner")
    def test_optimizer_iterates_and_improves(self, mock_runner):
        """Optimizer calls the solver multiple times and tracks history."""
        call_count = {"n": 0}

        def fake_run(config):
            """Return better QE as height increases toward 0.8."""
            call_count["n"] += 1
            height = config.get("pixel", {}).get("layers", {}).get("microlens", {}).get("height", 0.6)
            qe = 0.3 + 0.5 * (1.0 - abs(height - 0.8))
            return _make_result(qe_value=min(qe, 0.95))

        mock_runner.run.side_effect = fake_run

        cfg = _base_config()
        ps = ParameterSpace([MicrolensHeight(cfg, min_val=0.2, max_val=1.2)])
        obj = MaximizeQE()

        optimizer = PixelOptimizer(
            base_config=cfg,
            parameter_space=ps,
            objective=obj,
            solver_name="meent",
            method="nelder-mead",
            max_iterations=20,
            tolerance=1e-3,
        )
        result = optimizer.optimize()

        assert isinstance(result, OptimizationResult)
        assert result.n_evaluations > 1
        assert result.history.n_records > 1
        assert result.best_objective < 0  # negative QE

    @patch("compass.optimization.optimizer.SingleRunner")
    def test_optimizer_result_structure(self, mock_runner):
        """OptimizationResult has all expected fields."""
        mock_runner.run.return_value = _make_result(qe_value=0.5)

        cfg = _base_config()
        ps = ParameterSpace([MicrolensHeight(cfg)])
        obj = MaximizeQE()

        optimizer = PixelOptimizer(
            base_config=cfg,
            parameter_space=ps,
            objective=obj,
            max_iterations=5,
        )
        result = optimizer.optimize()

        assert result.best_params is not None
        assert isinstance(result.best_objective, float)
        assert isinstance(result.history, OptimizationHistory)
        assert isinstance(result.n_evaluations, int)
        assert isinstance(result.converged, bool)
        assert isinstance(result.final_config, dict)

    @patch("compass.optimization.optimizer.SingleRunner")
    def test_invalid_method_raises(self, mock_runner):
        """Unknown method raises ValueError."""
        cfg = _base_config()
        ps = ParameterSpace([MicrolensHeight(cfg)])
        obj = MaximizeQE()

        optimizer = PixelOptimizer(
            base_config=cfg,
            parameter_space=ps,
            objective=obj,
            method="invalid_method",
        )
        with pytest.raises(ValueError, match="Unknown optimization method"):
            optimizer.optimize()

    @patch("compass.optimization.optimizer.SingleRunner")
    def test_simulation_failure_returns_penalty(self, mock_runner):
        """Failed simulation returns large penalty, not crash."""
        mock_runner.run.side_effect = RuntimeError("Solver crashed")

        cfg = _base_config()
        ps = ParameterSpace([MicrolensHeight(cfg)])
        obj = MaximizeQE()

        optimizer = PixelOptimizer(
            base_config=cfg,
            parameter_space=ps,
            objective=obj,
            max_iterations=3,
        )
        # Should not raise; penalty is returned
        result = optimizer.optimize()
        assert result.best_objective >= 1e6 or result.n_evaluations >= 1

    @patch("compass.optimization.optimizer.SingleRunner")
    def test_differential_evolution_method(self, mock_runner):
        """Differential evolution method runs without error."""
        mock_runner.run.return_value = _make_result(qe_value=0.5)

        cfg = _base_config()
        ps = ParameterSpace([MicrolensHeight(cfg, min_val=0.2, max_val=1.0)])
        obj = MaximizeQE()

        optimizer = PixelOptimizer(
            base_config=cfg,
            parameter_space=ps,
            objective=obj,
            method="differential-evolution",
            max_iterations=3,
        )
        result = optimizer.optimize()
        assert result.n_evaluations > 0


# ===========================================================================
# OptimizationResult Tests
# ===========================================================================


class TestOptimizationResult:
    """Tests for OptimizationResult dataclass."""

    def test_construction(self):
        result = OptimizationResult(
            best_params=np.array([0.8]),
            best_objective=-0.75,
            history=OptimizationHistory(),
            n_evaluations=50,
            converged=True,
            final_config={"pixel": {}},
            final_result=None,
        )
        assert result.converged is True
        assert result.best_objective == pytest.approx(-0.75)
        assert result.n_evaluations == 50
