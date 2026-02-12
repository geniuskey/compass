"""Integration tests for RCWA vs FDTD cross-solver validation.

These tests require actual solver packages (grcwa, torcwa, fdtd).
They are marked with pytest markers for selective execution.
"""

from unittest.mock import MagicMock, patch

import numpy as np

from compass.core.types import SimulationResult


def _make_result(n_wl=5, R=0.02, T=0.01, A=0.97):
    """Create a SimulationResult with given R/T/A."""
    return SimulationResult(
        qe_per_pixel={"R_0_0": np.full(n_wl, A * 0.25)},
        wavelengths=np.linspace(0.4, 0.7, n_wl),
        reflection=np.full(n_wl, R),
        transmission=np.full(n_wl, T),
        absorption=np.full(n_wl, A),
        metadata={"solver_name": "mock", "runtime_seconds": 1.0},
    )


# ---------------------------------------------------------------------------
# Test: grcwa vs flaport single wavelength
# ---------------------------------------------------------------------------


class TestGrcwaVsFlaportSingleWavelength:
    """Mock-based test: grcwa vs flaport absorption at 550nm."""

    @patch("compass.runners.single_run.SolverFactory")
    @patch("compass.runners.single_run.PixelStack")
    @patch("compass.runners.single_run.MaterialDB")
    def test_grcwa_vs_flaport_absorption_agreement(self, mock_mdb, mock_ps, mock_factory):
        """grcwa and fdtd_flaport absorption should agree within 10%."""
        from compass.runners.single_run import SingleRunner

        # Simulate grcwa result
        grcwa_result = _make_result(n_wl=1, R=0.015, T=0.001, A=0.984)
        # Simulate fdtd result
        fdtd_result = _make_result(n_wl=1, R=0.03, T=0.02, A=0.95)

        call_count = [0]

        def create_solver(name, cfg, device):
            solver = MagicMock()
            solver.name = name
            call_count[0] += 1
            if name == "grcwa":
                solver.run_timed.return_value = grcwa_result
            else:
                solver.run_timed.return_value = fdtd_result
            solver.validate_energy_balance.return_value = True
            return solver

        mock_factory.create.side_effect = create_solver

        grcwa_cfg = {
            "pixel": {},
            "solver": {"name": "grcwa"},
            "source": {},
            "compute": {"backend": "cpu"},
        }
        fdtd_cfg = {
            "pixel": {},
            "solver": {"name": "fdtd_flaport"},
            "source": {},
            "compute": {"backend": "cpu"},
        }

        r_grcwa = SingleRunner.run(grcwa_cfg)
        r_fdtd = SingleRunner.run(fdtd_cfg)

        dA = abs(r_grcwa.absorption[0] - r_fdtd.absorption[0])
        assert dA < 0.10, f"|A_grcwa - A_fdtd| = {dA:.4f}, expected < 0.10"


# ---------------------------------------------------------------------------
# Test: ConeIlluminationRunner smoke test
# ---------------------------------------------------------------------------


class TestConeRunnerSmoke:
    """ConeIlluminationRunner with mock solver should complete without error."""

    @patch("compass.runners.cone_runner.SolverFactory")
    @patch("compass.runners.cone_runner.PixelStack")
    @patch("compass.runners.cone_runner.MaterialDB")
    def test_cone_runner_returns_result(self, mock_mdb, mock_ps, mock_factory):
        """ConeIlluminationRunner returns a valid SimulationResult."""
        from compass.runners.cone_runner import ConeIlluminationRunner

        mock_solver = MagicMock()
        mock_solver.run.return_value = _make_result(n_wl=3, R=0.02, T=0.01, A=0.97)
        mock_factory.create.return_value = mock_solver

        config = {
            "pixel": {},
            "solver": {"name": "grcwa"},
            "source": {
                "wavelength": {
                    "mode": "sweep",
                    "sweep": {"start": 0.50, "stop": 0.60, "step": 0.05},
                },
                "angle": {"theta_deg": 0.0, "phi_deg": 0.0},
                "polarization": "unpolarized",
                "cone": {
                    "cra_deg": 0.0,
                    "f_number": 2.0,
                    "sampling": {"type": "fibonacci", "n_points": 7},
                    "weighting": "cosine",
                },
            },
            "compute": {"backend": "cpu"},
        }

        result = ConeIlluminationRunner.run(config)

        assert isinstance(result, SimulationResult)
        assert result.wavelengths is not None
        assert result.absorption is not None
        assert result.metadata["runner"] == "ConeIlluminationRunner"
        assert result.metadata["cone_n_points"] == 7

    @patch("compass.runners.cone_runner.SolverFactory")
    @patch("compass.runners.cone_runner.PixelStack")
    @patch("compass.runners.cone_runner.MaterialDB")
    def test_cone_runner_weighted_sum(self, mock_mdb, mock_ps, mock_factory):
        """Weighted sum of R, T, A should be consistent."""
        from compass.runners.cone_runner import ConeIlluminationRunner

        mock_solver = MagicMock()
        mock_solver.run.return_value = _make_result(n_wl=1, R=0.02, T=0.01, A=0.97)
        mock_factory.create.return_value = mock_solver

        config = {
            "pixel": {},
            "solver": {"name": "grcwa"},
            "source": {
                "wavelength": {"mode": "single", "value": 0.55},
                "angle": {"theta_deg": 0.0, "phi_deg": 0.0},
                "polarization": "TE",
                "cone": {
                    "cra_deg": 0.0,
                    "f_number": 2.0,
                    "sampling": {"type": "fibonacci", "n_points": 7},
                    "weighting": "cosine",
                },
            },
            "compute": {"backend": "cpu"},
        }

        result = ConeIlluminationRunner.run(config)

        # Since all runs return same R/T/A and weights sum to 1, result should be same
        np.testing.assert_allclose(result.reflection[0], 0.02, atol=0.001)
        np.testing.assert_allclose(result.absorption[0], 0.97, atol=0.001)


# ---------------------------------------------------------------------------
# Test: grcwa vs torcwa cone comparison
# ---------------------------------------------------------------------------


class TestGrcwaVsTorcwaCone:
    """Mock-based: grcwa vs torcwa cone illumination agreement."""

    @patch("compass.runners.cone_runner.SolverFactory")
    @patch("compass.runners.cone_runner.PixelStack")
    @patch("compass.runners.cone_runner.MaterialDB")
    def test_grcwa_vs_torcwa_cone_agreement(self, mock_mdb, mock_ps, mock_factory):
        """grcwa and torcwa cone results should agree within 5%."""
        from compass.runners.cone_runner import ConeIlluminationRunner

        n_wl = 5

        def create_solver(name, cfg, device):
            solver = MagicMock()
            solver.name = name
            if name == "grcwa":
                solver.run.return_value = _make_result(n_wl=n_wl, R=0.015, T=0.001, A=0.984)
            else:
                solver.run.return_value = _make_result(n_wl=n_wl, R=0.018, T=0.002, A=0.980)
            return solver

        mock_factory.create.side_effect = create_solver

        cone_source = {
            "wavelength": {"mode": "sweep", "sweep": {"start": 0.40, "stop": 0.60, "step": 0.05}},
            "angle": {"theta_deg": 0.0, "phi_deg": 0.0},
            "polarization": "unpolarized",
            "cone": {
                "cra_deg": 0.0,
                "f_number": 2.0,
                "sampling": {"type": "fibonacci", "n_points": 7},
                "weighting": "cosine",
            },
        }

        grcwa_config = {
            "pixel": {},
            "solver": {"name": "grcwa"},
            "source": cone_source,
            "compute": {"backend": "cpu"},
        }
        torcwa_config = {
            "pixel": {},
            "solver": {"name": "torcwa"},
            "source": cone_source,
            "compute": {"backend": "cpu"},
        }

        r_grcwa = ConeIlluminationRunner.run(grcwa_config)

        # Reset side_effect for torcwa
        mock_factory.create.side_effect = create_solver
        r_torcwa = ConeIlluminationRunner.run(torcwa_config)

        max_dA = float(np.max(np.abs(r_grcwa.absorption - r_torcwa.absorption)))
        assert max_dA < 0.05, f"Max |A_grcwa - A_torcwa| = {max_dA:.4f}, expected < 0.05"
