"""Unit tests for ResultExporter (CSV, JSON)."""

import json

import numpy as np
import pytest

from compass.core.types import SimulationResult
from compass.io.export import ResultExporter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(
    n_wl=5,
    pixels=None,
    include_rta=False,
    metadata=None,
):
    """Create a SimulationResult for testing."""
    wavelengths = np.linspace(0.4, 0.7, n_wl)
    if pixels is None:
        pixels = {"R_0_0": np.linspace(0.2, 0.8, n_wl)}
    return SimulationResult(
        qe_per_pixel=pixels,
        wavelengths=wavelengths,
        reflection=np.array([0.1] * n_wl) if include_rta else None,
        transmission=np.array([0.3] * n_wl) if include_rta else None,
        absorption=np.array([0.6] * n_wl) if include_rta else None,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# CSV Export Tests
# ---------------------------------------------------------------------------


class TestCSVExport:
    """Tests for ResultExporter.to_csv."""

    def test_csv_file_created(self, tmp_path):
        """CSV file is created at the specified path."""
        result = _make_result()
        filepath = str(tmp_path / "qe.csv")
        ResultExporter.to_csv(result, filepath)
        assert (tmp_path / "qe.csv").exists()

    def test_csv_header_contains_pixel_names(self, tmp_path):
        """CSV header lists wavelength and pixel QE columns."""
        pixels = {"G_0_1": np.array([0.5, 0.6]), "R_0_0": np.array([0.3, 0.4])}
        result = _make_result(n_wl=2, pixels=pixels)
        filepath = str(tmp_path / "header.csv")
        ResultExporter.to_csv(result, filepath)

        with open(filepath) as f:
            header = f.readline().strip()
        assert "wavelength_um" in header
        assert "QE_G_0_1" in header
        assert "QE_R_0_0" in header

    def test_csv_data_correct_shape(self, tmp_path):
        """CSV data has correct number of rows and columns."""
        n_wl = 4
        pixels = {
            "A": np.random.rand(n_wl),
            "B": np.random.rand(n_wl),
            "C": np.random.rand(n_wl),
        }
        result = _make_result(n_wl=n_wl, pixels=pixels)
        filepath = str(tmp_path / "shape.csv")
        ResultExporter.to_csv(result, filepath)

        data = np.loadtxt(filepath, delimiter=",", skiprows=1)
        assert data.shape == (n_wl, 4)  # wavelength + 3 pixels

    def test_csv_wavelength_values(self, tmp_path):
        """Wavelength column matches the original data."""
        result = _make_result(n_wl=3)
        filepath = str(tmp_path / "wl.csv")
        ResultExporter.to_csv(result, filepath)

        data = np.loadtxt(filepath, delimiter=",", skiprows=1)
        np.testing.assert_allclose(data[:, 0], result.wavelengths, atol=1e-10)

    def test_csv_qe_values(self, tmp_path):
        """QE values in CSV match the original data."""
        qe = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        result = _make_result(n_wl=5, pixels={"px": qe})
        filepath = str(tmp_path / "qe_vals.csv")
        ResultExporter.to_csv(result, filepath)

        data = np.loadtxt(filepath, delimiter=",", skiprows=1)
        np.testing.assert_allclose(data[:, 1], qe, atol=1e-10)

    def test_csv_creates_parent_dirs(self, tmp_path):
        """Parent directories are created if they do not exist."""
        result = _make_result()
        filepath = str(tmp_path / "sub" / "dir" / "output.csv")
        ResultExporter.to_csv(result, filepath)
        assert (tmp_path / "sub" / "dir" / "output.csv").exists()

    def test_csv_single_wavelength(self, tmp_path):
        """Single wavelength exports without error."""
        result = _make_result(n_wl=1, pixels={"px": np.array([0.5])})
        filepath = str(tmp_path / "single.csv")
        ResultExporter.to_csv(result, filepath)

        data = np.loadtxt(filepath, delimiter=",", skiprows=1, ndmin=2)
        assert data.shape == (1, 2)


# ---------------------------------------------------------------------------
# JSON Export Tests
# ---------------------------------------------------------------------------


class TestJSONExport:
    """Tests for ResultExporter.to_json."""

    def test_json_file_created(self, tmp_path):
        """JSON file is created at the specified path."""
        result = _make_result()
        filepath = str(tmp_path / "result.json")
        ResultExporter.to_json(result, filepath)
        assert (tmp_path / "result.json").exists()

    def test_json_contains_wavelengths(self, tmp_path):
        """JSON output contains wavelengths_um array."""
        result = _make_result(n_wl=3)
        filepath = str(tmp_path / "wl.json")
        ResultExporter.to_json(result, filepath)

        with open(filepath) as f:
            data = json.load(f)
        assert "wavelengths_um" in data
        assert len(data["wavelengths_um"]) == 3

    def test_json_contains_qe_per_pixel(self, tmp_path):
        """JSON output contains qe_per_pixel dict."""
        pixels = {"R_0_0": np.array([0.3, 0.4]), "G_0_1": np.array([0.5, 0.6])}
        result = _make_result(n_wl=2, pixels=pixels)
        filepath = str(tmp_path / "qe.json")
        ResultExporter.to_json(result, filepath)

        with open(filepath) as f:
            data = json.load(f)
        assert "R_0_0" in data["qe_per_pixel"]
        assert "G_0_1" in data["qe_per_pixel"]
        assert data["qe_per_pixel"]["R_0_0"] == pytest.approx([0.3, 0.4])

    def test_json_includes_rta_when_present(self, tmp_path):
        """Reflection/transmission are included when present."""
        result = _make_result(include_rta=True)
        filepath = str(tmp_path / "rta.json")
        ResultExporter.to_json(result, filepath)

        with open(filepath) as f:
            data = json.load(f)
        assert "reflection" in data
        assert "transmission" in data

    def test_json_excludes_rta_when_none(self, tmp_path):
        """Reflection/transmission are excluded when None."""
        result = _make_result(include_rta=False)
        filepath = str(tmp_path / "no_rta.json")
        ResultExporter.to_json(result, filepath)

        with open(filepath) as f:
            data = json.load(f)
        assert "reflection" not in data
        assert "transmission" not in data

    def test_json_metadata_as_strings(self, tmp_path):
        """Metadata values are converted to strings in JSON."""
        result = _make_result(metadata={"runtime": 1.5, "solver": "test"})
        filepath = str(tmp_path / "meta.json")
        ResultExporter.to_json(result, filepath)

        with open(filepath) as f:
            data = json.load(f)
        assert "metadata" in data
        # All values should be strings per the implementation
        for v in data["metadata"].values():
            assert isinstance(v, str)

    def test_json_creates_parent_dirs(self, tmp_path):
        """Parent directories are created if they do not exist."""
        result = _make_result()
        filepath = str(tmp_path / "nested" / "output.json")
        ResultExporter.to_json(result, filepath)
        assert (tmp_path / "nested" / "output.json").exists()

    def test_json_valid_format(self, tmp_path):
        """Output is valid JSON that can be parsed."""
        result = _make_result(n_wl=3, include_rta=True)
        filepath = str(tmp_path / "valid.json")
        ResultExporter.to_json(result, filepath)

        with open(filepath) as f:
            data = json.load(f)  # Should not raise
        assert isinstance(data, dict)
