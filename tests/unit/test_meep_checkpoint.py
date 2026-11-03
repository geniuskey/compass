"""Checkpoint safety without importing the native Meep runtime."""

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def worker():
    path = Path(__file__).resolve().parents[2] / "scripts/benchmark_meep_reference.py"
    spec = importlib.util.spec_from_file_location("meep_checkpoint_worker", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checkpoint_rejects_different_protocol_without_overwriting(worker, tmp_path):
    path = tmp_path / "reference.json"
    worker.atomic_result_write(path, {"resolution": 120, "status": "running"})
    original = path.read_bytes()
    with pytest.raises(ValueError, match="protocol differs"):
        worker.resumable_result(path, {"resolution": 160})
    assert path.read_bytes() == original


def test_checkpoint_atomic_failure_preserves_previous_result(worker, tmp_path):
    path = tmp_path / "reference.json"
    worker.atomic_result_write(path, {"status": "running", "spectra": []})
    original = path.read_bytes()
    with pytest.raises(ValueError):
        worker.atomic_result_write(path, {"rta": [float("nan")]})
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_empty_protocol_can_resume_before_first_wavelength(worker, tmp_path):
    path = tmp_path / "reference.json"
    protocol = {"checkpoint_schema": 1, "wavelengths": [0.45, 0.55]}
    initial = worker.resumable_result(path, protocol)
    worker.atomic_result_write(path, initial)
    assert worker.resumable_result(path, protocol) == initial
