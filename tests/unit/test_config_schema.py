"""Unit tests for configuration schema."""


from compass.core.config_schema import (
    ColorFilterConfig,
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
        assert cfg.stability.fourier_factorization == "li_inverse"

    def test_bayer_map_default(self):
        cfg = PixelConfig()
        assert cfg.bayer_map == [["R", "G"], ["G", "B"]]

    def test_color_filter_channel_schema(self):
        cfg = ColorFilterConfig(
            red={"material": "cf_red", "thickness": 0.68, "contact_angle": 64.0},
            green={"material": "cf_green", "thickness": 0.62, "contact_angle": 70.0},
            blue={"material": "cf_blue", "thickness": 0.74, "contact_angle": 58.0},
            grid={
                "enabled": True,
                "width": 0.05,
                "thickness": 0.45,
                "material": "tungsten",
                "corner_radius": 0.02,
            },
        )

        assert cfg.red is not None
        assert cfg.red.thickness == 0.68
        assert cfg.green is not None
        assert cfg.green.contact_angle == 70.0
        assert cfg.grid.thickness == 0.45
        assert cfg.grid.height == 0.6
        assert cfg.grid.corner_radius == 0.02
        assert cfg.n_slices == 8

    def test_full_config_serialization(self):
        """Config should be serializable to dict."""
        cfg = CompassConfig()
        d = cfg.model_dump()
        assert isinstance(d, dict)
        assert "pixel" in d
        assert "solver" in d
        assert "source" in d
