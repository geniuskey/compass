"""Durable wavelength progress never masquerades as a completed spectrum."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_wrapper():
    source = Path(__file__).resolve().parents[2] / "scripts/benchmark_wsl_solvers.py"
    spec = importlib.util.spec_from_file_location("wsl_progress_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.fdtdx_progress_wrapper


def test_fdtdx_progress_preserves_result_and_actual_device(tmp_path):
    result = (0.2, 0.7, 0.1, "cuda:0")
    path = tmp_path / "progress" / "sample.jsonl"
    context = {"repeat": 2}
    wrapped = load_wrapper()(lambda *_a, **_kw: result, path, context)
    assert wrapped(SimpleNamespace(shape=(2, 3, 4)), 0.55, 0.01, 80, 35, "cuda", "TE") is result
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert [e["event"] for e in events] == ["wavelength_started", "wavelength_completed"]
    assert events[-1]["actual_device"] == "cuda:0"
    assert events[-1]["total_grid_size_xyz"] == [3, 2, 234]
    assert events[-1]["repeat"] == 2
    assert "status" not in events[-1]
    assert events[-1]["elapsed_s"] >= 0


def test_fdtdx_progress_rejects_nonfinite_result(tmp_path):
    path = tmp_path / "sample.jsonl"
    wrapped = load_wrapper()(lambda *_a, **_kw: (float("nan"), 0, 0, "cuda:0"), path, {})
    with pytest.raises(RuntimeError, match="Invalid native"):
        wrapped(SimpleNamespace(shape=(2, 3, 4)), 0.55, 0.01, 80, 35, "cuda", "TE")
    event = json.loads(path.read_text().splitlines()[-1])
    assert event["event"] == "wavelength_failed"
    assert "rta" not in event
