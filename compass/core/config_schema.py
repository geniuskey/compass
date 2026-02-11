"""Pydantic configuration schema for COMPASS simulations."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class MicrolensProfileConfig(BaseModel):
    type: str = "superellipse"
    n: float = 2.5
    alpha: float = 1.0


class MicrolensShiftConfig(BaseModel):
    mode: Literal["none", "manual", "auto_cra"] = "auto_cra"
    cra_deg: float = 0.0
    shift_x: float = 0.0
    shift_y: float = 0.0
    ref_wavelength: float = 0.55  # Wavelength for refractive index lookup (um)


class MicrolensConfig(BaseModel):
    enabled: bool = True
    height: float = 0.6
    radius_x: float = 0.48
    radius_y: float = 0.48
    material: str = "polymer_n1p56"
    profile: MicrolensProfileConfig = Field(default_factory=MicrolensProfileConfig)
    shift: MicrolensShiftConfig = Field(default_factory=MicrolensShiftConfig)
    gap: float = 0.0


class GridConfig(BaseModel):
    enabled: bool = True
    width: float = 0.05
    height: float = 0.6
    material: str = "tungsten"


class ColorFilterConfig(BaseModel):
    thickness: float = 0.6
    pattern: str = "bayer_rggb"
    materials: dict[str, str] = Field(default_factory=lambda: {"R": "cf_red", "G": "cf_green", "B": "cf_blue"})
    grid: GridConfig = Field(default_factory=GridConfig)


class BarlLayerConfig(BaseModel):
    thickness: float
    material: str


class BarlConfig(BaseModel):
    layers: list[BarlLayerConfig] = Field(default_factory=list)


class PhotodiodeConfig(BaseModel):
    position: tuple[float, float, float] = (0.0, 0.0, 0.5)
    size: tuple[float, float, float] = (0.7, 0.7, 2.0)


class DtiConfig(BaseModel):
    enabled: bool = True
    width: float = 0.1
    depth: float = 3.0
    material: str = "sio2"


class SiliconConfig(BaseModel):
    thickness: float = 3.0
    material: str = "silicon"
    photodiode: PhotodiodeConfig = Field(default_factory=PhotodiodeConfig)
    dti: DtiConfig = Field(default_factory=DtiConfig)


class SimpleLayerConfig(BaseModel):
    thickness: float
    material: str


class LayersConfig(BaseModel):
    air: SimpleLayerConfig = Field(default_factory=lambda: SimpleLayerConfig(thickness=1.0, material="air"))
    microlens: MicrolensConfig = Field(default_factory=MicrolensConfig)
    planarization: SimpleLayerConfig = Field(default_factory=lambda: SimpleLayerConfig(thickness=0.3, material="sio2"))
    color_filter: ColorFilterConfig = Field(default_factory=ColorFilterConfig)
    barl: BarlConfig = Field(default_factory=BarlConfig)
    silicon: SiliconConfig = Field(default_factory=SiliconConfig)


class PixelConfig(BaseModel):
    pitch: float = 1.0
    unit_cell: tuple[int, int] = (2, 2)
    layers: LayersConfig = Field(default_factory=LayersConfig)
    bayer_map: list[list[str]] = Field(default_factory=lambda: [["R", "G"], ["G", "B"]])


class EnergyCheckConfig(BaseModel):
    enabled: bool = True
    tolerance: float = 0.02
    auto_retry_float64: bool = True


class StabilityConfig(BaseModel):
    precision_strategy: Literal["float32", "float64", "mixed", "adaptive"] = "mixed"
    allow_tf32: bool = False
    eigendecomp_device: Literal["cpu", "gpu", "cusolver"] = "cpu"
    fourier_factorization: Literal["naive", "li_inverse", "normal_vector"] = "li_inverse"
    energy_check: EnergyCheckConfig = Field(default_factory=EnergyCheckConfig)
    eigenvalue_broadening: float = 1e-10
    condition_number_warning: float = 1e12


class ConvergenceConfig(BaseModel):
    auto_converge: bool = False
    order_range: tuple[int, int] = (5, 25)
    qe_tolerance: float = 0.01
    spacing_range: tuple[float, float] | None = None


class SolverConfig(BaseModel):
    name: str = "torcwa"
    type: Literal["rcwa", "fdtd"] = "rcwa"
    params: dict = Field(default_factory=lambda: {"fourier_order": [9, 9], "dtype": "complex64"})
    convergence: ConvergenceConfig = Field(default_factory=ConvergenceConfig)
    stability: StabilityConfig = Field(default_factory=StabilityConfig)


class WavelengthSweepConfig(BaseModel):
    start: float = 0.38
    stop: float = 0.78
    step: float = 0.01


class WavelengthConfig(BaseModel):
    mode: Literal["single", "sweep", "list"] = "single"
    value: float | None = 0.55
    sweep: WavelengthSweepConfig | None = None
    values: list[float] | None = None


class AngleConfig(BaseModel):
    theta_deg: float = 0.0
    phi_deg: float = 0.0


class ConeSamplingConfig(BaseModel):
    type: Literal["grid", "fibonacci", "gaussian_quadrature"] = "fibonacci"
    n_points: int = 37


class ConeConfig(BaseModel):
    cra_deg: float = 0.0
    f_number: float = 2.0
    pupil_shape: Literal["circular", "elliptical"] = "circular"
    sampling: ConeSamplingConfig = Field(default_factory=ConeSamplingConfig)
    weighting: str = "cosine"


class RayFileConfig(BaseModel):
    enabled: bool = False
    path: str = ""
    format: Literal["zemax_json", "csv"] = "zemax_json"


class SourceConfig(BaseModel):
    type: Literal["planewave", "cone_illumination"] = "planewave"
    wavelength: WavelengthConfig = Field(default_factory=WavelengthConfig)
    angle: AngleConfig = Field(default_factory=AngleConfig)
    polarization: Literal["TE", "TM", "unpolarized"] = "unpolarized"
    cone: ConeConfig | None = None
    ray_file: RayFileConfig | None = None


class ComputeConfig(BaseModel):
    backend: Literal["auto", "cuda", "cpu", "mps"] = "auto"
    gpu_id: int = 0
    num_workers: int = 4


class CompassConfig(BaseModel):
    """Top-level COMPASS configuration."""

    pixel: PixelConfig = Field(default_factory=PixelConfig)
    solver: SolverConfig = Field(default_factory=SolverConfig)
    source: SourceConfig = Field(default_factory=SourceConfig)
    compute: ComputeConfig = Field(default_factory=ComputeConfig)
    experiment_name: str = "default"
    output_dir: str = "./outputs"
    seed: int = 42
