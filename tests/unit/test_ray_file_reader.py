"""Unit tests for RayFileReader."""

import json

import pytest

from compass.sources.ray_file_reader import RayFileReader


@pytest.fixture
def json_ray_file(tmp_path):
    """Create a temporary JSON ray file."""
    data = {
        "image_height": [0.0, 0.5, 1.0],
        "cra_deg": [0.0, 10.0, 20.0],
        "mra_deg": [0.0, 5.0, 10.0],
        "f_number": [2.0, 2.0, 2.0],
    }
    filepath = tmp_path / "rays.json"
    filepath.write_text(json.dumps(data))
    return str(filepath)


@pytest.fixture
def csv_ray_file_4col(tmp_path):
    """Create a temporary CSV ray file with 4 columns."""
    content = "image_height,cra_deg,mra_deg,f_number\n"
    content += "0.0,0.0,0.0,2.0\n"
    content += "0.5,10.0,5.0,2.0\n"
    content += "1.0,20.0,10.0,2.8\n"
    filepath = tmp_path / "rays.csv"
    filepath.write_text(content)
    return str(filepath)


@pytest.fixture
def csv_ray_file_2col(tmp_path):
    """Create a CSV with only 2 columns (image_height, cra_deg)."""
    content = "image_height,cra_deg\n"
    content += "0.0,0.0\n"
    content += "1.0,15.0\n"
    filepath = tmp_path / "rays_2col.csv"
    filepath.write_text(content)
    return str(filepath)


class TestReadJSON:
    """Tests for RayFileReader.read_json."""

    def test_reads_all_fields(self, json_ray_file):
        """All fields from JSON are read correctly."""
        data = RayFileReader.read_json(json_ray_file)
        assert "image_height" in data
        assert "cra_deg" in data
        assert "mra_deg" in data
        assert "f_number" in data

    def test_correct_values(self, json_ray_file):
        """Values match what was written."""
        data = RayFileReader.read_json(json_ray_file)
        assert data["image_height"] == [0.0, 0.5, 1.0]
        assert data["cra_deg"] == [0.0, 10.0, 20.0]

    def test_returns_dict(self, json_ray_file):
        """Return type is a dict."""
        data = RayFileReader.read_json(json_ray_file)
        assert isinstance(data, dict)


class TestReadCSV:
    """Tests for RayFileReader.read_csv."""

    def test_four_column_csv(self, csv_ray_file_4col):
        """4-column CSV reads all fields correctly."""
        data = RayFileReader.read_csv(csv_ray_file_4col)
        assert len(data["image_height"]) == 3
        assert len(data["cra_deg"]) == 3
        assert len(data["mra_deg"]) == 3
        assert len(data["f_number"]) == 3

    def test_correct_values(self, csv_ray_file_4col):
        """CSV values match expected data."""
        data = RayFileReader.read_csv(csv_ray_file_4col)
        assert data["image_height"][0] == pytest.approx(0.0)
        assert data["cra_deg"][1] == pytest.approx(10.0)
        assert data["f_number"][2] == pytest.approx(2.8)

    def test_two_column_csv_missing_fields(self, csv_ray_file_2col):
        """2-column CSV returns empty lists for missing columns."""
        data = RayFileReader.read_csv(csv_ray_file_2col)
        assert len(data["image_height"]) == 2
        assert len(data["cra_deg"]) == 2
        assert data["mra_deg"] == []
        assert data["f_number"] == []

    def test_returns_lists(self, csv_ray_file_4col):
        """All returned values are Python lists."""
        data = RayFileReader.read_csv(csv_ray_file_4col)
        for key in ["image_height", "cra_deg", "mra_deg", "f_number"]:
            assert isinstance(data[key], list)


class TestReadAutoDetect:
    """Tests for RayFileReader.read with auto-detection."""

    def test_auto_detect_json(self, json_ray_file):
        """Auto-detect identifies .json extension."""
        data = RayFileReader.read(json_ray_file)
        assert "image_height" in data
        assert data["cra_deg"] == [0.0, 10.0, 20.0]

    def test_auto_detect_csv(self, csv_ray_file_4col):
        """Auto-detect identifies .csv extension."""
        data = RayFileReader.read(csv_ray_file_4col)
        assert len(data["image_height"]) == 3

    def test_explicit_format_json(self, json_ray_file):
        """Explicit format='zemax_json' works."""
        data = RayFileReader.read(json_ray_file, format="zemax_json")
        assert "cra_deg" in data

    def test_explicit_format_csv(self, csv_ray_file_4col):
        """Explicit format='csv' works."""
        data = RayFileReader.read(csv_ray_file_4col, format="csv")
        assert len(data["cra_deg"]) == 3

    def test_nonexistent_file_raises(self, tmp_path):
        """Reading a nonexistent file raises an exception."""
        with pytest.raises((FileNotFoundError, OSError)):
            RayFileReader.read(str(tmp_path / "nonexistent.json"))

    def test_unknown_extension_defaults_to_csv(self, tmp_path):
        """Non-.json extension defaults to CSV format in auto mode."""
        content = "image_height,cra_deg,mra_deg,f_number\n0.0,0.0,0.0,2.0\n1.0,10.0,5.0,2.8\n"
        filepath = tmp_path / "rays.txt"
        filepath.write_text(content)
        data = RayFileReader.read(str(filepath))
        assert len(data["image_height"]) == 2


class TestEdgeCases:
    """Edge case tests for RayFileReader."""

    def test_empty_json_object(self, tmp_path):
        """Empty JSON object returns empty dict."""
        filepath = tmp_path / "empty.json"
        filepath.write_text("{}")
        data = RayFileReader.read_json(str(filepath))
        assert data == {}

    def test_single_row_csv_raises(self, tmp_path):
        """CSV with a single data row triggers IndexError (numpy 1-D array edge case)."""
        content = "image_height,cra_deg,mra_deg,f_number\n0.5,12.0,6.0,2.0\n"
        filepath = tmp_path / "single.csv"
        filepath.write_text(content)
        # Known limitation: np.loadtxt returns 1-D array for single row,
        # causing data[:, 0] to fail with IndexError.
        with pytest.raises(IndexError):
            RayFileReader.read_csv(str(filepath))

    def test_json_with_extra_fields(self, tmp_path):
        """JSON with extra fields preserves them."""
        data_in = {
            "image_height": [0.0],
            "cra_deg": [0.0],
            "extra_field": "metadata",
        }
        filepath = tmp_path / "extra.json"
        filepath.write_text(json.dumps(data_in))
        data = RayFileReader.read_json(str(filepath))
        assert data["extra_field"] == "metadata"
