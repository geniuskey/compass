"""Unit tests for configuration schema."""


from compass.core.config_schema import (
    CompassConfig,
    PixelConfig,
    SolverConfig,
    SourceConfig,
)


class TestConfigSchema:
    """Tests for Pydantic config schema."""

    def test_default_config(self):
        """Default config should be valid."""
        cfg = CompassConfig()
        assert cfg.pixel.pitch == 1.0
        assert cfg.solver.name == "torcwa"
        assert cfg.source.type == "planewave"

    def test_pixel_config(self):
        cfg = PixelConfig(pitch=0.8, unit_cell=(4, 4))
        assert cfg.pitch == 0.8
        assert cfg.unit_cell == (4, 4)

    def test_solver_config_rcwa(self):
        cfg = SolverConfig(name="torcwa", type="rcwa")
        assert cfg.name == "torcwa"
        assert cfg.type == "rcwa"

    def test_solver_config_fdtd(self):
        cfg = SolverConfig(name="fdtd_flaport", type="fdtd")
        assert cfg.type == "fdtd"

    def test_source_config_planewave(self):
        cfg = SourceConfig(type="planewave", polarization="TE")
        assert cfg.type == "planewave"
        assert cfg.polarization == "TE"

    def test_stability_defaults(self):
        cfg = SolverConfig()
        assert cfg.stability.precision_strategy == "mixed"
        assert cfg.stability.allow_tf32 is False
        assert cfg.stability.eigendecomp_device == "cpu"
        assert cfg.stability.fourier_factorization == "li_inverse"

    def test_bayer_map_default(self):
        cfg = PixelConfig()
        assert cfg.bayer_map == [["R", "G"], ["G", "B"]]

    def test_full_config_serialization(self):
        """Config should be serializable to dict."""
        cfg = CompassConfig()
        d = cfg.model_dump()
        assert isinstance(d, dict)
        assert "pixel" in d
        assert "solver" in d
        assert "source" in d
