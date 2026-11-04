"""Unit tests for ConeIlluminationRunner angular averaging."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from compass.core.types import SimulationResult
from compass.runners.cone_runner import ConeIlluminationRunner

MODULE = "compass.runners.cone_runner"
WL = np.array([0.45, 0.55, 0.65])


def _config(n_points=5, energy_check=None, source_extra=None):
    solver = {"name": "mock_solver"}
    if energy_check is not None:
        solver["stability"] = {"energy_check": energy_check}
    source = {
        "type": "cone_illumination",
        "wavelength_range": [0.45, 0.65],
        "cone": {
            "cra_deg": 5.0,
            "f_number": 2.8,
            "sampling": {"type": "fibonacci", "n_points": n_points},
            "weighting": "cosine",
        },
    }
    source.update(source_extra or {})
    return {
        "pixel": {"pitch": 1.0, "unit_cell": [2, 2], "layers": {}},
        "solver": solver,
        "source": source,
        "compute": {"backend": "cpu"},
    }


def _result_for_theta(theta_deg, failed=None):
    """Angle-dependent result so weighted averaging is observable."""
    r = np.full(len(WL), 0.01 * theta_deg)
    t = np.full(len(WL), 0.2)
    return SimulationResult(
        qe_per_pixel={"R_0_0": np.full(len(WL), theta_deg), "G_0_1": np.ones(len(WL))},
        wavelengths=WL,
        reflection=r,
        transmission=t,
        absorption=1.0 - r - t,
        metadata={"failed_runs": failed} if failed else {},
    )


class _Harness:
    """Patches the runner's collaborators and records angles seen by the solver."""

    def __init__(self, failed_at_first=None):
        self.angles = []
        self.solver = MagicMock()
        self._failed_at_first = failed_at_first

        def setup_source(cfg):
            self.angles.append((cfg["angle"]["theta_deg"], cfg["angle"]["phi_deg"]))

        def run():
            theta = self.angles[-1][0]
            failed = self._failed_at_first if len(self.angles) == 1 else None
            return _result_for_theta(theta, failed)

        self.solver.setup_source.side_effect = setup_source
        self.solver.run.side_effect = run
        self.patches = [
            patch(f"{MODULE}.SolverFactory.create", return_value=self.solver),
            patch(f"{MODULE}.PixelStack"),
            patch(f"{MODULE}.MaterialDB"),
            patch(f"{MODULE}.resolve_compute_device", return_value="cpu"),
        ]

    def __enter__(self):
        mocks = [p.start() for p in self.patches]
        self.factory_create = mocks[0]
        return self

    def __exit__(self, *exc):
        for p in self.patches:
            p.stop()


def _expected_points(cfg):
    from compass.sources.cone_illumination import ConeIllumination

    cone = cfg["source"]["cone"]
    return ConeIllumination(
        cra_deg=cone["cra_deg"],
        f_number=cone["f_number"],
        n_points=cone["sampling"]["n_points"],
        sampling=cone["sampling"]["type"],
        weighting=cone["weighting"],
    ).get_sampling_points()


class TestConeRunner:
    def test_runs_solver_once_per_angle(self):
        cfg = _config(n_points=7)
        points = _expected_points(cfg)
        with _Harness() as h:
            ConeIlluminationRunner.run(cfg)
        assert h.solver.run.call_count == len(points)
        np.testing.assert_allclose(h.angles, [(t, p) for t, p, _ in points])
        h.solver.setup_geometry.assert_called_once()

    def test_weighted_average(self):
        cfg = _config(n_points=7)
        points = _expected_points(cfg)
        thetas = np.array([t for t, _, _ in points])
        weights = np.array([w for _, _, w in points])
        assert weights.sum() == pytest.approx(1.0)
        with _Harness() as h:
            result = ConeIlluminationRunner.run(cfg)
        del h
        mean_theta = float(np.sum(weights * thetas))
        np.testing.assert_allclose(result.qe_per_pixel["R_0_0"], mean_theta)
        np.testing.assert_allclose(result.qe_per_pixel["G_0_1"], 1.0)
        np.testing.assert_allclose(result.reflection, 0.01 * mean_theta)
        np.testing.assert_allclose(result.transmission, 0.2)
        np.testing.assert_allclose(
            result.reflection + result.transmission + result.absorption, 1.0
        )
        np.testing.assert_array_equal(result.wavelengths, WL)

    def test_source_config_not_mutated(self):
        cfg = _config(source_extra={"angle": {"theta_deg": 0.0, "phi_deg": 0.0}})
        with _Harness():
            ConeIlluminationRunner.run(cfg)
        assert cfg["source"]["angle"] == {"theta_deg": 0.0, "phi_deg": 0.0}

    def test_metadata(self):
        cfg = _config(n_points=5)
        with _Harness():
            result = ConeIlluminationRunner.run(cfg)
        md = result.metadata
        assert md["runner"] == "ConeIlluminationRunner"
        assert md["solver_name"] == "mock_solver"
        assert md["device"] == "cpu"
        assert md["cone_f_number"] == 2.8
        assert md["cone_cra_deg"] == 5.0
        assert md["cone_n_points"] == len(_expected_points(cfg))
        assert md["cone_sampling"] == "fibonacci"
        assert md["cone_weighting"] == "cosine"
        assert md["runtime_seconds"] >= 0
        assert "failed_runs" not in md

    def test_failed_runs_tagged_with_angle(self):
        cfg = _config()
        first_theta = _expected_points(cfg)[0][0]
        with _Harness(failed_at_first=[{"wavelength": 0.55, "error": "nan"}]):
            result = ConeIlluminationRunner.run(cfg)
        failed = result.metadata["failed_runs"]
        assert len(failed) == 1
        assert failed[0]["wavelength"] == 0.55
        assert failed[0]["theta_deg"] == pytest.approx(first_theta)
        assert "phi_deg" in failed[0]

    def test_energy_check_default_tolerance(self):
        cfg = _config(n_points=3)
        with _Harness() as h:
            ConeIlluminationRunner.run(cfg)
        assert h.solver.validate_energy_balance.call_count == h.solver.run.call_count
        assert h.solver.validate_energy_balance.call_args.kwargs["tolerance"] == 0.01

    def test_energy_check_custom_tolerance(self):
        cfg = _config(n_points=3, energy_check={"enabled": True, "tolerance": 0.05})
        with _Harness() as h:
            ConeIlluminationRunner.run(cfg)
        assert h.solver.validate_energy_balance.call_args.kwargs["tolerance"] == 0.05

    def test_energy_check_disabled(self):
        cfg = _config(n_points=3, energy_check={"enabled": False})
        with _Harness() as h:
            ConeIlluminationRunner.run(cfg)
        h.solver.validate_energy_balance.assert_not_called()

    def test_missing_rta_arrays_leave_zero(self):
        cfg = _config(n_points=3)
        with _Harness() as h:
            h.solver.run.side_effect = lambda: SimulationResult(
                qe_per_pixel={"R_0_0": np.ones(3)}, wavelengths=WL
            )
            result = ConeIlluminationRunner.run(cfg)
        np.testing.assert_allclose(result.reflection, 0.0)
        np.testing.assert_allclose(result.qe_per_pixel["R_0_0"], 1.0)

    def test_no_points_raises(self):
        cfg = _config()
        with _Harness(), patch(
            f"{MODULE}.ConeIllumination.get_sampling_points", return_value=[]
        ), pytest.raises(RuntimeError, match="No angular sampling points"):
            ConeIlluminationRunner.run(cfg)

    def test_solver_factory_receives_config_and_device(self):
        cfg = _config()
        with _Harness() as h:
            ConeIlluminationRunner.run(cfg)
        h.factory_create.assert_called_once_with("mock_solver", cfg["solver"], "cpu")

    def test_single_runner_dispatches_to_cone_runner(self):
        from compass.runners.single_run import SingleRunner

        cfg = _config()
        sentinel = SimulationResult(qe_per_pixel={}, wavelengths=WL)
        with patch.object(ConeIlluminationRunner, "run", return_value=sentinel) as run:
            out = SingleRunner.run(cfg)
        assert out is sentinel
        run.assert_called_once()
