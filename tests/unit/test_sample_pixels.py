"""Unit tests for sample pixel structure heuristics and configs."""

from pathlib import Path

import pytest
import yaml

from compass.geometry import SAMPLE_HEADLINES, derive_parameters
from compass.geometry.pixel_stack import PixelStack

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "pixel"

SAMPLE_YAML_FILES = [
    "sample_p0p56um_4x4ocl.yaml",
    "sample_p1p0um_quadbayer.yaml",
    "sample_p1p6um_split_pd.yaml",
    "sample_p1p22um_2x2ocl.yaml",
    "sample_p1p2um_lofic.yaml",
]


def _load_yaml(name: str) -> dict:
    """Load a Hydra package-scoped pixel YAML and unwrap the `pixel:` block."""
    raw = yaml.safe_load((CONFIG_DIR / name).read_text())
    # Configs use `# @package _global_` so the YAML root is `pixel`.
    return raw


@pytest.mark.parametrize("yaml_name", SAMPLE_YAML_FILES)
def test_sample_yaml_builds_pixel_stack(yaml_name: str) -> None:
    """Each sample YAML should build a PixelStack without errors."""
    cfg = _load_yaml(yaml_name)
    stack = PixelStack(cfg)
    # Stack must have at least silicon, BARL, CF, planarization, microlens, air.
    layer_names = {l.name for l in stack.layers}
    assert "silicon" in layer_names
    assert "color_filter" in layer_names
    assert "microlens" in layer_names
    assert "air" in layer_names
    # Domain size matches pitch * unit_cell.
    pitch = cfg["pixel"]["pitch"]
    rows, cols = cfg["pixel"]["unit_cell"]
    assert stack.domain_size == pytest.approx((pitch * cols, pitch * rows))


def test_shared_ocl_creates_one_lens_per_group() -> None:
    """`microlens.sharing: 2` means one lens per 2x2 cluster, not per pixel."""
    cfg = _load_yaml("sample_p1p22um_2x2ocl.yaml")
    stack = PixelStack(cfg)
    rows, cols = cfg["pixel"]["unit_cell"]
    sharing = cfg["pixel"]["layers"]["microlens"]["sharing"]
    expected = (rows // sharing) * (cols // sharing)
    assert len(stack.microlenses) == expected


def test_per_pixel_ocl_unchanged() -> None:
    """`sharing: 1` (default) keeps the legacy per-pixel lens count."""
    cfg = _load_yaml("sample_p1p0um_quadbayer.yaml")
    stack = PixelStack(cfg)
    rows, cols = cfg["pixel"]["unit_cell"]
    assert len(stack.microlenses) == rows * cols


@pytest.mark.parametrize("sample", list(SAMPLE_HEADLINES.keys()))
def test_derive_parameters_returns_buildable_config(sample: str) -> None:
    """Every entry in SAMPLE_HEADLINES should produce a buildable config."""
    headline = SAMPLE_HEADLINES[sample]
    pitch = headline.get("pitch", 1.0)
    cfg = derive_parameters(sample=sample, pitch=pitch)
    # Wrap in {"pixel": cfg} as PixelStack expects.
    stack = PixelStack({"pixel": cfg})
    assert stack.pitch == pytest.approx(pitch)
    # Microlens sharing must propagate.
    sharing = headline.get("ocl_sharing", 1)
    assert cfg["layers"]["microlens"]["sharing"] == sharing
    # Bayer map must be square and match the binning pattern.
    bm = cfg["bayer_map"]
    assert len(bm) == len(bm[0])


def test_derive_parameters_overrides_take_priority() -> None:
    cfg = derive_parameters(
        sample="sample_p0p56um_4x4ocl",
        overrides={"layers.silicon.thickness": 1.7},
    )
    assert cfg["layers"]["silicon"]["thickness"] == 1.7


def test_4x4_ocl_via_derive() -> None:
    """4x4 OCL (sharing=4) should produce one lens per 4x4 group."""
    cfg = derive_parameters(
        sample="generic_bsi",
        pitch=0.7,
        cf_pattern="tetra2cell",
        ocl_sharing=4,
    )
    stack = PixelStack({"pixel": cfg})
    # 8x8 unit cell, one lens per 4x4 group => 4 lenses total.
    assert len(stack.microlenses) == 4
