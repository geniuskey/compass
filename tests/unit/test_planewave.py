"""Unit tests for PlanewaveSource."""

import numpy as np
import pytest

from compass.sources.planewave import PlanewaveSource


class TestPlanewaveSource:
    """Tests for planewave source model."""

    def test_single_wavelength(self):
        config = {
            "wavelength": {"mode": "single", "value": 0.55},
            "angle": {"theta_deg": 0.0, "phi_deg": 0.0},
            "polarization": "TE",
        }
        src = PlanewaveSource.from_config(config)
        assert len(src.wavelengths) == 1
        assert src.wavelengths[0] == 0.55

    def test_sweep_mode(self):
        config = {
            "wavelength": {
                "mode": "sweep",
                "sweep": {"start": 0.40, "stop": 0.70, "step": 0.10},
            },
            "angle": {"theta_deg": 0.0, "phi_deg": 0.0},
            "polarization": "TE",
        }
        src = PlanewaveSource.from_config(config)
        assert len(src.wavelengths) == 4  # 0.40, 0.50, 0.60, 0.70

    def test_list_mode(self):
        config = {
            "wavelength": {
                "mode": "list",
                "values": [0.45, 0.53, 0.63],
            },
            "angle": {"theta_deg": 10.0, "phi_deg": 45.0},
            "polarization": "TM",
        }
        src = PlanewaveSource.from_config(config)
        assert len(src.wavelengths) == 3
        assert src.theta_deg == 10.0
        assert src.phi_deg == 45.0

    def test_unpolarized_runs(self):
        src = PlanewaveSource(
            wavelengths=np.array([0.55]),
            polarization="unpolarized",
        )
        runs = src.get_polarization_runs()
        assert runs == ["TE", "TM"]

    def test_te_single_run(self):
        src = PlanewaveSource(
            wavelengths=np.array([0.55]),
            polarization="TE",
        )
        runs = src.get_polarization_runs()
        assert runs == ["TE"]

    def test_angle_conversion(self):
        src = PlanewaveSource(
            wavelengths=np.array([0.55]),
            theta_deg=30.0,
            phi_deg=45.0,
        )
        assert src.theta_rad == pytest.approx(np.deg2rad(30.0))
        assert src.phi_rad == pytest.approx(np.deg2rad(45.0))

    def test_to_solver_params(self):
        src = PlanewaveSource(
            wavelengths=np.array([0.45, 0.55, 0.65]),
            theta_deg=15.0,
            polarization="unpolarized",
        )
        params = src.to_solver_params()
        assert len(params["wavelengths"]) == 3
        assert params["polarization"] == "unpolarized"
        assert len(params["polarization_runs"]) == 2
