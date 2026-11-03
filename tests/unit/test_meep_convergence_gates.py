"""Do not confuse temporal drift or different losses with spatial convergence."""

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def gate(tmp_path):
    path = Path(__file__).resolve().parents[2] / "scripts/check_meep_pml_convergence.py"
    spec = importlib.util.spec_from_file_location("meep_width_gate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUT = tmp_path
    return module


def save_run(directory, resolution, *, extinction=0.02, drift=0.0001, offset=0.0):
    spectra = []
    for wl in [0.45, 0.55, 0.65]:
        series = [
            {"duration": t, "rta": [0.2 + offset + i * drift, 0.7 - offset - i * drift, 0.1]}
            for i, t in enumerate([20, 40, 80])
        ]
        spectra.append({"wavelength_um": wl, "rta": series[-1]["rta"], "time_series": series})
    result = {
        "status": "ok",
        "resolution": resolution,
        "padding": 0.4,
        "pml": 0.35,
        "extinction": extinction,
        "fwidth_factor": 0.15,
        "method": "time_domain",
        "adaptive_time": True,
        "spectra": spectra,
    }
    (directory / f"meep_reference_r{resolution}.json").write_text(json.dumps(result))


def test_width_check_can_start_before_unrelated_solver_jobs_finish(gate, tmp_path):
    save_run(tmp_path, 120)
    save_run(tmp_path, 160, offset=0.002)
    assert gate.primary_reference_ready()
    assert not (tmp_path / "studies_finished.json").exists()


def test_coarse_time_drift_cannot_be_called_grid_convergence(gate, tmp_path):
    save_run(tmp_path, 120, drift=0.002)
    save_run(tmp_path, 160, offset=0.002)
    assert not gate.primary_reference_ready()


def test_different_loss_cannot_validate_same_grid_problem(gate, tmp_path):
    save_run(tmp_path, 120, extinction=0.0)
    save_run(tmp_path, 160)
    assert not gate.primary_reference_ready()


def test_large_spatial_difference_requires_another_resolution(gate, tmp_path):
    save_run(tmp_path, 120)
    save_run(tmp_path, 160, offset=0.01)
    assert not gate.primary_reference_ready()
