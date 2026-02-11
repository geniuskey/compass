"""Unit tests for HDF5Handler save/load round-trip."""

import numpy as np
import pytest

from compass.core.types import FieldData, SimulationResult
from compass.io.hdf5_handler import HDF5Handler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(
    n_wl=5,
    n_pixels=2,
    include_rta=True,
    include_fields=False,
    metadata=None,
):
    """Create a SimulationResult for testing."""
    wavelengths = np.linspace(0.4, 0.7, n_wl)
    qe_per_pixel = {}
    for i in range(n_pixels):
        qe_per_pixel[f"pixel_{i}"] = np.random.rand(n_wl)

    reflection = np.random.rand(n_wl) * 0.3 if include_rta else None
    transmission = np.random.rand(n_wl) * 0.3 if include_rta else None
    absorption = np.random.rand(n_wl) * 0.4 if include_rta else None

    fields = None
    if include_fields:
        fields = {
            f"wl_{wavelengths[0]:.3f}": FieldData(
                Ex=np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4),
                Ey=np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4),
                Ez=np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4),
            ),
        }

    return SimulationResult(
        qe_per_pixel=qe_per_pixel,
        wavelengths=wavelengths,
        reflection=reflection,
        transmission=transmission,
        absorption=absorption,
        fields=fields,
        metadata=metadata or {"solver_name": "test", "runtime_seconds": 1.23},
    )


def _make_config():
    """Create a sample config dict."""
    return {
        "pixel": {"pitch": 1.0, "unit_cell": [2, 2]},
        "solver": {"name": "torcwa", "type": "rcwa"},
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHDF5RoundTrip:
    """Tests for save_result / load_result round-trip."""

    def test_basic_round_trip(self, tmp_path):
        """Save and load produces equivalent data."""
        result = _make_result()
        config = _make_config()
        filepath = str(tmp_path / "test_result.h5")

        HDF5Handler.save_result(result, config, filepath)
        loaded_result, loaded_config = HDF5Handler.load_result(filepath)

        np.testing.assert_array_almost_equal(
            loaded_result.wavelengths, result.wavelengths
        )
        for pixel_name in result.qe_per_pixel:
            np.testing.assert_array_almost_equal(
                loaded_result.qe_per_pixel[pixel_name],
                result.qe_per_pixel[pixel_name],
            )

    def test_rta_round_trip(self, tmp_path):
        """Reflection, transmission, absorption survive round-trip."""
        result = _make_result(include_rta=True)
        config = _make_config()
        filepath = str(tmp_path / "rta.h5")

        HDF5Handler.save_result(result, config, filepath)
        loaded, _ = HDF5Handler.load_result(filepath)

        np.testing.assert_array_almost_equal(loaded.reflection, result.reflection)
        np.testing.assert_array_almost_equal(loaded.transmission, result.transmission)
        np.testing.assert_array_almost_equal(loaded.absorption, result.absorption)

    def test_no_rta_round_trip(self, tmp_path):
        """Missing R/T/A data loads as None."""
        result = _make_result(include_rta=False)
        config = _make_config()
        filepath = str(tmp_path / "no_rta.h5")

        HDF5Handler.save_result(result, config, filepath)
        loaded, _ = HDF5Handler.load_result(filepath)

        assert loaded.reflection is None
        assert loaded.transmission is None
        assert loaded.absorption is None

    def test_config_round_trip(self, tmp_path):
        """Config dict is preserved through save/load."""
        result = _make_result()
        config = _make_config()
        filepath = str(tmp_path / "config.h5")

        HDF5Handler.save_result(result, config, filepath)
        _, loaded_config = HDF5Handler.load_result(filepath)

        assert loaded_config["pixel"]["pitch"] == 1.0
        assert loaded_config["solver"]["name"] == "torcwa"

    def test_metadata_round_trip(self, tmp_path):
        """Metadata attributes survive round-trip."""
        result = _make_result(metadata={"solver_name": "meent", "runtime_seconds": 2.5})
        config = _make_config()
        filepath = str(tmp_path / "meta.h5")

        HDF5Handler.save_result(result, config, filepath)
        loaded, _ = HDF5Handler.load_result(filepath)

        assert loaded.metadata["solver_name"] == "meent"
        assert float(loaded.metadata["runtime_seconds"]) == pytest.approx(2.5)

    def test_fields_saved_when_requested(self, tmp_path):
        """Field data is saved when save_fields=True."""
        result = _make_result(include_fields=True)
        config = _make_config()
        filepath = str(tmp_path / "fields.h5")

        HDF5Handler.save_result(result, config, filepath, save_fields=True)

        import h5py
        with h5py.File(filepath, "r") as f:
            assert "fields" in f
            wl_keys = list(f["fields"].keys())
            assert len(wl_keys) == 1
            assert "Ex" in f["fields"][wl_keys[0]]
            assert "Ey" in f["fields"][wl_keys[0]]
            assert "Ez" in f["fields"][wl_keys[0]]

    def test_fields_not_saved_by_default(self, tmp_path):
        """Field data is NOT saved when save_fields=False (default)."""
        result = _make_result(include_fields=True)
        config = _make_config()
        filepath = str(tmp_path / "no_fields.h5")

        HDF5Handler.save_result(result, config, filepath)

        import h5py
        with h5py.File(filepath, "r") as f:
            assert "fields" not in f


class TestHDF5EdgeCases:
    """Edge case tests for HDF5Handler."""

    def test_single_wavelength(self, tmp_path):
        """Single wavelength data saves and loads correctly."""
        result = _make_result(n_wl=1, n_pixels=1)
        config = _make_config()
        filepath = str(tmp_path / "single_wl.h5")

        HDF5Handler.save_result(result, config, filepath)
        loaded, _ = HDF5Handler.load_result(filepath)

        assert len(loaded.wavelengths) == 1
        assert len(loaded.qe_per_pixel) == 1

    def test_many_pixels(self, tmp_path):
        """Multiple pixels all survive round-trip."""
        result = _make_result(n_pixels=8)
        config = _make_config()
        filepath = str(tmp_path / "many_pixels.h5")

        HDF5Handler.save_result(result, config, filepath)
        loaded, _ = HDF5Handler.load_result(filepath)

        assert len(loaded.qe_per_pixel) == 8
        for key in result.qe_per_pixel:
            assert key in loaded.qe_per_pixel

    def test_creates_parent_directories(self, tmp_path):
        """Parent directories are created automatically."""
        result = _make_result()
        config = _make_config()
        filepath = str(tmp_path / "sub" / "dir" / "result.h5")

        HDF5Handler.save_result(result, config, filepath)
        loaded, _ = HDF5Handler.load_result(filepath)
        assert len(loaded.wavelengths) > 0

    def test_metadata_with_non_serializable_value(self, tmp_path):
        """Non-serializable metadata values are converted to strings."""
        result = _make_result(metadata={"list_val": [1, 2, 3], "solver_name": "test"})
        config = _make_config()
        filepath = str(tmp_path / "nonserializable.h5")

        # Should not raise
        HDF5Handler.save_result(result, config, filepath)
        loaded, _ = HDF5Handler.load_result(filepath)
        assert "solver_name" in loaded.metadata

    def test_empty_qe_dict(self, tmp_path):
        """Empty qe_per_pixel dict saves and loads."""
        result = SimulationResult(
            qe_per_pixel={},
            wavelengths=np.array([0.55]),
            metadata={},
        )
        config = _make_config()
        filepath = str(tmp_path / "empty_qe.h5")

        HDF5Handler.save_result(result, config, filepath)
        loaded, _ = HDF5Handler.load_result(filepath)
        assert len(loaded.qe_per_pixel) == 0

    def test_load_nonexistent_file_raises(self):
        """Loading a nonexistent file raises an error."""
        with pytest.raises(Exception):
            HDF5Handler.load_result("/nonexistent/path/data.h5")

    def test_compression_options(self, tmp_path):
        """Custom compression options are accepted."""
        result = _make_result()
        config = _make_config()
        filepath = str(tmp_path / "compressed.h5")

        HDF5Handler.save_result(
            result, config, filepath,
            compression="gzip", compression_level=9,
        )
        loaded, _ = HDF5Handler.load_result(filepath)
        np.testing.assert_array_almost_equal(
            loaded.wavelengths, result.wavelengths
        )
