"""Tests for compass.compute.resolve_compute_device and SolverFactory device handling."""

from __future__ import annotations

import pytest

from compass import compute
from compass.compute import resolve_compute_device
from compass.solvers.base import SolverFactory


@pytest.fixture
def cuda_available(monkeypatch):
    monkeypatch.setattr(compute, "_torch_cuda_available", lambda: True)


@pytest.fixture
def cuda_unavailable(monkeypatch):
    monkeypatch.setattr(compute, "_torch_cuda_available", lambda: False)


class TestResolveComputeDevice:
    def test_cpu_is_always_cpu(self, cuda_available):
        assert resolve_compute_device("cpu") == "cpu"

    def test_auto_selects_cuda_when_available(self, cuda_available):
        assert resolve_compute_device("auto") == "cuda"

    def test_auto_falls_back_to_cpu(self, cuda_unavailable):
        assert resolve_compute_device("auto") == "cpu"

    def test_default_backend_is_auto(self, cuda_unavailable):
        assert resolve_compute_device() == "cpu"

    def test_explicit_cuda_preserved(self, cuda_available):
        assert resolve_compute_device("cuda") == "cuda"

    def test_explicit_cuda_without_cuda_raises(self, cuda_unavailable):
        with pytest.raises(RuntimeError, match="compute=auto"):
            resolve_compute_device("cuda")

    @pytest.mark.parametrize("mps_available", [True, False])
    def test_mps_falls_back_to_cpu(self, monkeypatch, caplog, mps_available):
        monkeypatch.setattr(compute, "_torch_mps_available", lambda: mps_available)
        with caplog.at_level("WARNING", logger="compass.compute"):
            assert resolve_compute_device("mps", solver_name="torcwa") == "cpu"
        assert "torcwa" in caplog.text

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unsupported compute backend"):
            resolve_compute_device("tpu")


class _RecordingSolver:
    def __init__(self, config, device):
        self.config = config
        self.device = device


class TestSolverFactoryDevice:
    @pytest.fixture(autouse=True)
    def _register(self, monkeypatch):
        monkeypatch.setitem(SolverFactory._registry, "_recording", _RecordingSolver)

    def test_auto_is_resolved(self, cuda_unavailable):
        solver = SolverFactory.create("_recording", {}, "auto")
        assert solver.device == "cpu"

    def test_mps_is_resolved_to_cpu(self, cuda_unavailable):
        solver = SolverFactory.create("_recording", {}, "mps")
        assert solver.device == "cpu"

    def test_cpu_passes_through(self, cuda_available):
        solver = SolverFactory.create("_recording", {}, "cpu")
        assert solver.device == "cpu"


class TestGpuId:
    def test_gpu_zero_is_plain_cuda(self, cuda_available):
        assert resolve_compute_device("cuda", gpu_id=0) == "cuda"

    def test_gpu_id_selects_indexed_device(self, monkeypatch, cuda_available):
        monkeypatch.setattr(compute, "_torch_cuda_device_count", lambda: 2)
        assert resolve_compute_device("cuda", gpu_id=1) == "cuda:1"
        assert resolve_compute_device("auto", gpu_id=1) == "cuda:1"

    def test_gpu_id_out_of_range_raises(self, monkeypatch, cuda_available):
        monkeypatch.setattr(compute, "_torch_cuda_device_count", lambda: 1)
        with pytest.raises(RuntimeError, match="gpu_id=1"):
            resolve_compute_device("cuda", gpu_id=1)

    def test_negative_gpu_id_raises(self, cuda_available):
        with pytest.raises(ValueError, match="gpu_id"):
            resolve_compute_device("cuda", gpu_id=-1)

    def test_gpu_id_ignored_on_cpu(self, cuda_unavailable):
        assert resolve_compute_device("auto", gpu_id=3) == "cpu"

    @pytest.mark.parametrize(("device", "index"), [("cuda", 0), ("cuda:2", 2), ("cpu", 0)])
    def test_cuda_device_index(self, device, index):
        assert compute.cuda_device_index(device) == index
