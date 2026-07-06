"""Pydantic configuration schema for COMPASS simulations."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    """Base model that rejects unknown keys.

    A typo in a YAML config (e.g. `thicknes:`) must fail validation
    instead of being silently ignored.
    """

    model_config = ConfigDict(extra="forbid")


class MicrolensProfileConfig(StrictModel):
    type: str = "superellipse"
    n: float = 2.5
    alpha: float = 1.0


class MicrolensShiftConfig(StrictModel):
    mode: Literal["none", "manual", "auto_cra"] = "auto_cra"
    cra_deg: float = 0.0
    shift_x: float = 0.0
    shift_y: float = 0.0
    ref_wavelength: float = 0.55  # Wavelength for refractive index lookup (um)


class MicrolensConfig(StrictModel):
    enabled: bool = True
    height: float = 0.6
    radius_x: float = 0.48
    radius_y: float = 0.48
    material: str = "polymer_n1p56"
    profile: MicrolensProfileConfig = Field(default_factory=MicrolensProfileConfig)
    shift: MicrolensShiftConfig = Field(default_factory=MicrolensShiftConfig)
    gap: float = 0.0
    # Flat residual layer of lens material left under the curved cap by the
    # reflow / etch-back process. Real microlenses are never zero-thickness at
    # their edges; a thin planar slab of the same polymer remains. Modelled as a
    # uniform slab of `base_thickness` below the superellipse cap.
    base_thickness: float = 0.0
    # Multi-pixel lens sharing (e.g. Sony 2x2 OCL Quad Bayer, Samsung Hexadeca 4x4 OCL).
    # `sharing` = N means one microlens covers an N x N group of pixels (which
    # typically share the same color in Quad/Nona/Tetra2 Bayer arrangements).
    # When sharing > 1, the lens radius is auto-scaled to ~ sharing * pitch / 2
    # unless radius_x / radius_y are explicitly set in the config.
    sharing: int = 1


class GridConfig(StrictModel):
    enabled: bool = True
    width: float = 0.05
    thickness: float | None = None
    height: float = 0.6
    material: str = "tungsten"
    corner_radius: float = 0.0


class ColorFilterChannelConfig(StrictModel):
    material: str | None = None
    thickness: float | None = None
    contact_angle: float = 90.0


class ColorFilterConfig(StrictModel):
    thickness: float = 0.6
    pattern: str = "bayer_rggb"
    materials: dict[str, str] = Field(
        default_factory=lambda: {"R": "cf_red", "G": "cf_green", "B": "cf_blue"}
    )
    red: ColorFilterChannelConfig | None = None
    green: ColorFilterChannelConfig | None = None
    blue: ColorFilterChannelConfig | None = None
    grid: GridConfig = Field(default_factory=GridConfig)
    n_slices: int = 8


class BarlLayerConfig(StrictModel):
    thickness: float
    material: str


class BarlConfig(StrictModel):
    layers: list[BarlLayerConfig] = Field(default_factory=list)


class PhotodiodeConfig(StrictModel):
    position: tuple[float, float, float] = (0.0, 0.0, 0.5)
    size: tuple[float, float, float] = (0.7, 0.7, 2.0)


class DtiLinerConfig(StrictModel):
    """Conformal high-k passivation liner on the DTI trench sidewalls.

    Real BSI DTI trenches are lined with a thin high-k film (Al2O3, HfO2,
    Ta2O5, ...) that both passivates the etched silicon surface and carries a
    negative fixed charge. Optically it is a thin high-index ring between the
    silicon and the lower-index trench fill. The liner is modelled as a
    conformal ring of `thickness` (um) inside the trench outline, surrounding
    the core fill material.
    """

    enabled: bool = False
    material: str = "al2o3"
    thickness: float = 0.0


class DtiConfig(StrictModel):
    enabled: bool = True
    mode: Literal["fdti", "bdti"] = "fdti"
    width: float = 0.1
    depth: float = 3.0
    material: str = "sio2"
    # Conformal high-k passivation liner on the trench sidewalls.
    liner: DtiLinerConfig = Field(default_factory=DtiLinerConfig)
    # Sidewall angle in degrees measured from the substrate plane (90 = vertical
    # walls, no taper). DTI is etched from the backside, so the trench is widest
    # at the opening (`width`) and narrows with depth when taper_angle < 90.
    taper_angle: float = 90.0
    # Number of staircase z-slices used when the trench is tapered. Ignored for
    # purely vertical trenches.
    n_slices: int = 6


class SurfaceTextureConfig(StrictModel):
    """Backside silicon nanostructure for light trapping / NIR enhancement.

    Modern NIR-enhanced BSI sensors etch an inverted-pyramid array (IPA) into
    the silicon backside facing the incoming light. The graded silicon fill
    fraction acts as a moth-eye anti-reflection / light-trapping layer that
    significantly boosts long-wavelength (700-1000 nm) quantum efficiency.
    Modelled as a staircase of pyramidal pits carved from the top of the
    silicon layer and back-filled with `fill_material`.
    """

    enabled: bool = False
    type: Literal["inverted_pyramid"] = "inverted_pyramid"
    height: float = 0.3
    # Pyramid array period in um. Defaults to the pixel pitch when None.
    period: float | None = None
    fill_material: str = "sio2"
    n_slices: int = 8


class SiliconConfig(StrictModel):
    thickness: float = 3.0
    material: str = "silicon"
    photodiode: PhotodiodeConfig = Field(default_factory=PhotodiodeConfig)
    dti: DtiConfig = Field(default_factory=DtiConfig)
    surface_texture: SurfaceTextureConfig = Field(default_factory=SurfaceTextureConfig)


class SimpleLayerConfig(StrictModel):
    thickness: float
    material: str


class LayersConfig(StrictModel):
    air: SimpleLayerConfig = Field(
        default_factory=lambda: SimpleLayerConfig(thickness=1.0, material="air")
    )
    microlens: MicrolensConfig = Field(default_factory=MicrolensConfig)
    planarization: SimpleLayerConfig = Field(
        default_factory=lambda: SimpleLayerConfig(thickness=0.3, material="sio2")
    )
    color_filter: ColorFilterConfig = Field(default_factory=ColorFilterConfig)
    barl: BarlConfig = Field(default_factory=BarlConfig)
    silicon: SiliconConfig = Field(default_factory=SiliconConfig)


class PixelConfig(StrictModel):
    pitch: float = 1.0
    unit_cell: tuple[int, int] = (2, 2)
    layers: LayersConfig = Field(default_factory=LayersConfig)
    bayer_map: list[list[str]] = Field(default_factory=lambda: [["R", "G"], ["G", "B"]])


class EnergyCheckConfig(StrictModel):
    enabled: bool = True
    tolerance: float = 0.02
    auto_retry_float64: bool = True


class StabilityConfig(StrictModel):
    # precision_strategy and fourier_factorization are consumed by the
    # diagnostics pre-simulation checks (compass.solvers.rcwa.stability);
    # allow_tf32 is applied by PrecisionManager and the torcwa adapter.
    precision_strategy: Literal["float32", "float64", "mixed", "adaptive"] = "mixed"
    allow_tf32: bool = False
    fourier_factorization: Literal["naive", "li_inverse", "normal_vector"] = "li_inverse"
    energy_check: EnergyCheckConfig = Field(default_factory=EnergyCheckConfig)


class SolverConfig(StrictModel):
    name: str = "torcwa"
    type: Literal["rcwa", "fdtd", "tmm"] = "rcwa"
    params: dict = Field(default_factory=lambda: {"fourier_order": [9, 9], "dtype": "complex64"})
    stability: StabilityConfig = Field(default_factory=StabilityConfig)


class WavelengthSweepConfig(StrictModel):
    start: float = 0.38
    stop: float = 0.78
    step: float = 0.01


class WavelengthConfig(StrictModel):
    mode: Literal["single", "sweep", "list"] = "single"
    value: float | None = 0.55
    sweep: WavelengthSweepConfig | None = None
    values: list[float] | None = None


class AngleConfig(StrictModel):
    theta_deg: float = 0.0
    phi_deg: float = 0.0


class ConeSamplingConfig(StrictModel):
    type: Literal[
        "fibonacci",
        "sunflower",
        "rings",
        "halton",
        "hammersley",
        "gauss",
        "gaussian_quadrature",
        "grid",
    ] = "fibonacci"
    n_points: int = 37


class ConeConfig(StrictModel):
    cra_deg: float = 0.0
    f_number: float = 2.0
    pupil_shape: Literal["circular", "elliptical"] = "circular"
    sampling: ConeSamplingConfig = Field(default_factory=ConeSamplingConfig)
    weighting: str = "cosine"


class RayFileConfig(StrictModel):
    enabled: bool = False
    path: str = ""
    format: Literal["zemax_json", "csv"] = "zemax_json"


class SourceConfig(StrictModel):
    type: Literal["planewave", "cone_illumination"] = "planewave"
    wavelength: WavelengthConfig = Field(default_factory=WavelengthConfig)
    angle: AngleConfig = Field(default_factory=AngleConfig)
    polarization: Literal["TE", "TM", "unpolarized"] = "unpolarized"
    cone: ConeConfig | None = None
    ray_file: RayFileConfig | None = None


class ComputeConfig(StrictModel):
    backend: Literal["auto", "cuda", "cpu", "mps"] = "auto"
    gpu_id: int = 0
    num_workers: int = 4


class CompassConfig(BaseModel):
    """Top-level COMPASS configuration.

    Unlike the nested models, the top level allows extra keys: experiment
    overlays (configs/experiment/*.yaml) add sections such as `experiment`,
    `optimization`, or per-experiment sweep tables that are consumed by
    dedicated runners rather than this schema.
    """

    model_config = ConfigDict(extra="allow")

    pixel: PixelConfig = Field(default_factory=PixelConfig)
    solver: SolverConfig = Field(default_factory=SolverConfig)
    source: SourceConfig = Field(default_factory=SourceConfig)
    compute: ComputeConfig = Field(default_factory=ComputeConfig)
    experiment_name: str = "default"
    output_dir: str = "./outputs"
    seed: int = 42
