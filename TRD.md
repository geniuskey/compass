# TRD: COMPASS — Technical Requirements Document

## 1. 아키텍처 개요

```
compass/
├── configs/                        # Hydra configuration hierarchy
│   ├── config.yaml                 # Root config (defaults 조합)
│   ├── pixel/                      # Pixel stack 구조 정의
│   │   ├── default_bsi_1um.yaml
│   │   ├── default_bsi_0p8um.yaml
│   │   └── custom/                 # 사용자 정의 (.gitignore)
│   ├── solver/                     # 솔버별 설정
│   │   ├── grcwa.yaml
│   │   ├── torcwa.yaml
│   │   ├── meent.yaml
│   │   ├── fdtd_flaport.yaml
│   │   ├── fdtdz.yaml
│   │   └── meep.yaml
│   ├── source/                     # 광원 설정
│   │   ├── planewave.yaml
│   │   ├── wavelength_sweep.yaml
│   │   └── cone_illumination.yaml
│   ├── experiment/                 # 실험 조합
│   │   ├── qe_benchmark.yaml
│   │   ├── solver_comparison.yaml
│   │   └── roi_sweep.yaml
│   ├── compute/                    # 하드웨어 설정
│   │   ├── cuda.yaml
│   │   ├── cpu.yaml
│   │   └── mps.yaml
│   └── local/                      # 로컬 전용 (.gitignore)
│       └── my_overrides.yaml
│
├── compass/                    # 메인 패키지
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config_schema.py        # Pydantic/dataclass 스키마
│   │   ├── pixel_stack.py          # 통합 구조 모델 (solver-agnostic)
│   │   ├── material_db.py          # 재료 물성 DB
│   │   ├── geometry_builder.py     # 파라메트릭 지오메트리 생성
│   │   └── units.py                # 단위 변환 유틸리티
│   │
│   ├── solvers/
│   │   ├── __init__.py
│   │   ├── base.py                 # SolverBase ABC
│   │   ├── rcwa/
│   │   │   ├── __init__.py
│   │   │   ├── grcwa_solver.py
│   │   │   ├── torcwa_solver.py
│   │   │   └── meent_solver.py
│   │   └── fdtd/
│   │       ├── __init__.py
│   │       ├── flaport_solver.py
│   │       ├── fdtdz_solver.py
│   │       └── meep_solver.py
│   │
│   ├── sources/
│   │   ├── __init__.py
│   │   ├── planewave.py
│   │   ├── cone_illumination.py    # Exit pupil cone 조명
│   │   └── ray_file_reader.py      # Zemax JSON ray file reader
│   │
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── qe_calculator.py        # QE 계산 (absorbed power in PD)
│   │   ├── crosstalk.py            # 크로스토크 분석
│   │   ├── energy_balance.py       # R+T+A=1 검증
│   │   └── solver_comparison.py    # 솔버 간 비교 분석
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── field_plot_2d.py        # 2D 필드 맵 (matplotlib)
│   │   ├── structure_plot_2d.py    # 2D 구조 단면도
│   │   ├── viewer_3d.py            # 3D interactive viewer (pyvista/plotly)
│   │   ├── qe_plot.py              # QE 스펙트럼/비교 플롯
│   │   └── report_generator.py     # 자동 비교 리포트 생성
│   │
│   ├── io/
│   │   ├── __init__.py
│   │   ├── hdf5_handler.py         # HDF5 결과 저장/로드
│   │   ├── result_schema.py        # 결과 데이터 스키마
│   │   └── export.py               # CSV, JSON 내보내기
│   │
│   └── runners/
│       ├── __init__.py
│       ├── single_run.py           # 단일 시뮬레이션
│       ├── sweep_runner.py         # 파라미터 sweep
│       ├── roi_sweep_runner.py     # ROI별 sweep
│       └── comparison_runner.py    # 솔버 비교 실행
│
├── materials/                      # 재료 DB 파일들
│   ├── silicon_palik.csv
│   ├── silicon_green2008.csv
│   ├── sio2.csv
│   ├── si3n4.csv
│   ├── color_filter_red.csv
│   ├── color_filter_green.csv
│   ├── color_filter_blue.csv
│   └── tungsten.csv
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── benchmarks/                 # 알려진 해석해와 비교
│
├── notebooks/                      # Jupyter 예제
│   ├── 01_quick_start.ipynb
│   ├── 02_structure_visualization.ipynb
│   ├── 03_solver_comparison.ipynb
│   └── 04_roi_sweep.ipynb
│
├── scripts/
│   ├── run_simulation.py           # Hydra entry point
│   └── compare_solvers.py
│
├── pyproject.toml
└── README.md
```

---

## 2. Configuration 설계 (Hydra)

### 2.1 Root Config

```yaml
# configs/config.yaml
defaults:
  - pixel: default_bsi_1um
  - solver: torcwa
  - source: planewave
  - compute: cuda
  - _self_

experiment_name: "default"
output_dir: "./outputs/${experiment_name}/${now:%Y-%m-%d_%H-%M-%S}"
seed: 42
```

### 2.2 Pixel Stack Config

```yaml
# configs/pixel/default_bsi_1um.yaml
# @package _global_
pixel:
  pitch: 1.0                    # um
  unit_cell: [2, 2]             # NxM pixels in unit cell

  layers:
    air:
      thickness: 1.0            # um (simulation region above lens)
      material: "air"

    microlens:
      enabled: true
      height: 0.6               # um
      radius_x: 0.48            # um (semi-axis)
      radius_y: 0.48
      material: "polymer_n1p56"
      profile:
        type: "superellipse"
        n: 2.5                  # squareness parameter
        alpha: 1.0              # curvature parameter
      shift:                    # CRA-dependent shift (per pixel 또는 global)
        mode: "auto_cra"        # "none", "manual", "auto_cra"
        cra_deg: 0.0
      gap: 0.0                  # inter-lens gap (um)

    planarization:
      thickness: 0.3
      material: "sio2"

    color_filter:
      thickness: 0.6
      pattern: "bayer_rggb"     # "bayer_rggb", "bayer_grbg", "quad_bayer", "custom"
      materials:
        R: "cf_red"
        G: "cf_green"
        B: "cf_blue"
      grid:
        enabled: true
        width: 0.05             # um
        height: 0.6             # um (same as CF or different)
        material: "tungsten"

    barl:                         # Bottom Anti-Reflective Layers (다층 반사방지막)
      layers:                     # 순서: Si에 가까운 층부터 위로
        - thickness: 0.010        # um
          material: "sio2"
        - thickness: 0.025
          material: "hfo2"
        - thickness: 0.015
          material: "sio2"
        - thickness: 0.030
          material: "si3n4"

    silicon:
      thickness: 3.0            # um
      material: "silicon"
      photodiode:
        position: [0.0, 0.0, 0.5]   # relative to pixel center, z from Si top
        size: [0.7, 0.7, 2.0]       # um
        # PD는 실제로는 doping profile이지만, 광학 시뮬레이션에서는
        # Si 영역 내 특정 volume에서의 흡수 에너지를 적분하는 방식
      dti:
        enabled: true
        width: 0.1              # um
        depth: 3.0              # um (full depth 또는 partial)
        material: "sio2"

  # Bayer pattern definition (2x2 unit)
  bayer_map:                    # [row][col] -> color
    - ["R", "G"]
    - ["G", "B"]
```

### 2.3 Solver Config

```yaml
# configs/solver/torcwa.yaml
# @package _global_
solver:
  name: "torcwa"
  type: "rcwa"

  params:
    fourier_order: [9, 9]       # [Nx, Ny] Fourier truncation orders
    dtype: "complex64"          # "complex64" or "complex128"
    allow_tf32: false           # TF32 tensor core (faster but less precise)

  convergence:
    auto_converge: false        # 자동 수렴 테스트
    order_range: [5, 25]        # min, max Fourier order
    qe_tolerance: 0.01          # QE 변화 < 1% 시 수렴 판정
```

```yaml
# configs/solver/fdtd_flaport.yaml
# @package _global_
solver:
  name: "fdtd_flaport"
  type: "fdtd"

  params:
    grid_spacing: 0.02          # um (20nm discretization)
    runtime: 200                # femtoseconds
    pml_layers: 15              # PML thickness in grid cells
    courant_number: null        # auto (default: 1/sqrt(3) for 3D)
    dtype: "float64"

  convergence:
    auto_converge: false
    spacing_range: [0.01, 0.05]
    qe_tolerance: 0.01
```

### 2.4 Source Config

```yaml
# configs/source/planewave.yaml
# @package _global_
source:
  type: "planewave"
  wavelength:
    mode: "single"              # "single", "sweep", "list"
    value: 0.55                 # um (single)
    # sweep: {start: 0.38, stop: 0.78, step: 0.01}  # (sweep mode)
    # values: [0.45, 0.53, 0.63]                     # (list mode)
  angle:
    theta_deg: 0.0              # polar angle
    phi_deg: 0.0                # azimuthal angle
  polarization: "unpolarized"   # "TE", "TM", "unpolarized"
```

```yaml
# configs/source/cone_illumination.yaml
# @package _global_
source:
  type: "cone_illumination"
  wavelength:
    mode: "sweep"
    sweep: {start: 0.38, stop: 0.78, step: 0.02}

  cone:
    cra_deg: 15.0               # Chief Ray Angle
    f_number: 2.0               # → half-cone angle = arcsin(1/(2*f_number))
    pupil_shape: "circular"     # "circular", "elliptical"
    sampling:
      type: "fibonacci"         # "grid", "fibonacci", "gaussian_quadrature"
      n_points: 37              # angular sampling points within cone
    weighting: "cosine"         # "uniform", "cosine", "cos4", "gaussian", "custom"

  # 또는 Zemax ray file에서 읽기
  ray_file:
    enabled: false
    path: "path/to/rays.json"
    format: "zemax_json"        # "zemax_json", "csv"
```

### 2.5 Experiment Config

```yaml
# configs/experiment/solver_comparison.yaml
# @package _global_
defaults:
  - /pixel: default_bsi_1um
  - /source: wavelength_sweep

experiment:
  name: "solver_comparison_2x2"
  description: "Compare RCWA and FDTD solvers on 2x2 Bayer unit cell"

  solvers:                      # 복수 솔버 동시 실행
    - solver: "torcwa"
      params:
        fourier_order: [11, 11]
    - solver: "meent"
      params:
        fourier_order: [11, 11]
        backend: "torch"
    - solver: "fdtd_flaport"
      params:
        grid_spacing: 0.015

  comparison:
    reference_solver: "torcwa"  # 기준 솔버
    metrics: ["qe_diff", "qe_relative_error", "energy_balance", "runtime"]
    generate_report: true
```

---

## 3. 핵심 클래스 설계

### 3.1 SolverBase (Abstract Base Class)

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

@dataclass
class SimulationResult:
    """Solver-agnostic result container"""
    qe_per_pixel: Dict[str, np.ndarray]     # {"R_0_0": array([...]), "G_0_1": ...}
    wavelengths: np.ndarray                  # [um]
    fields: Optional[Dict] = None           # {"Ex": 3D array, "Ey": ..., "Ez": ..., ...}
    poynting: Optional[Dict] = None         # {"Sx": ..., "Sy": ..., "Sz": ...}
    reflection: Optional[np.ndarray] = None # R(λ)
    transmission: Optional[np.ndarray] = None  # T(λ)
    absorption: Optional[np.ndarray] = None    # A(λ)
    metadata: Dict = None                   # solver info, timing, convergence

class SolverBase(ABC):
    """Abstract base class for all EM solvers"""

    def __init__(self, config: dict, device: str = "cuda"):
        self.config = config
        self.device = device

    @abstractmethod
    def setup_geometry(self, pixel_stack: 'PixelStack') -> None:
        """Convert PixelStack to solver-specific geometry representation"""
        pass

    @abstractmethod
    def setup_source(self, source_config: dict) -> None:
        """Configure excitation source"""
        pass

    @abstractmethod
    def run(self) -> SimulationResult:
        """Execute simulation and return standardized results"""
        pass

    @abstractmethod
    def get_field_distribution(self, component: str, plane: str,
                                position: float) -> np.ndarray:
        """Extract 2D field slice: component in {Ex,Ey,Ez,|E|2},
           plane in {xy,xz,yz}, position along normal axis"""
        pass

    def validate_energy_balance(self, result: SimulationResult,
                                 tolerance: float = 0.01) -> bool:
        """Check R + T + A ≈ 1"""
        if result.reflection is not None and result.absorption is not None:
            total = result.reflection + result.transmission + result.absorption
            return np.allclose(total, 1.0, atol=tolerance)
        return True  # skip if not available
```

### 3.2 PixelStack (통합 구조 모델)

```python
@dataclass
class Layer:
    """Generic layer in the pixel stack"""
    name: str
    z_start: float          # um, bottom of layer
    z_end: float            # um, top of layer
    thickness: float        # um
    base_material: str      # material name
    geometry: Optional['GeometrySpec'] = None  # patterned layer info

@dataclass
class MicrolensSpec:
    """Microlens geometry specification"""
    height: float
    radius_x: float
    radius_y: float
    material: str
    profile_type: str       # "superellipse", "spherical", "aspherical"
    n_param: float          # superellipse squareness
    alpha_param: float      # curvature
    shift_x: float = 0.0   # CRA offset
    shift_y: float = 0.0

@dataclass
class PhotodiodeSpec:
    """Photodiode region definition"""
    position: tuple         # (x, y, z) relative to pixel center
    size: tuple             # (dx, dy, dz) in um
    pixel_index: tuple      # (row, col) in unit cell

class PixelStack:
    """Solver-agnostic pixel stack representation"""

    def __init__(self, config: dict):
        self.pitch = config['pixel']['pitch']
        self.unit_cell = tuple(config['pixel']['unit_cell'])
        self.layers: List[Layer] = []
        self.microlenses: List[MicrolensSpec] = []
        self.photodiodes: List[PhotodiodeSpec] = []
        self.bayer_map: List[List[str]] = []
        self._build_from_config(config)

    def _build_from_config(self, config: dict):
        """Construct full 3D stack from YAML parameters"""
        # ... layer-by-layer construction ...
        pass

    def get_permittivity_grid(self, wavelength: float,
                               nx: int, ny: int, nz: int) -> np.ndarray:
        """Generate 3D permittivity distribution for a given wavelength.
           Returns complex epsilon array of shape (nx, ny, nz)"""
        pass

    def get_layer_slices(self) -> List[Dict]:
        """Get z-wise layer decomposition for RCWA solvers.
           Each slice: {z_start, z_end, eps_2d(nx, ny)}"""
        pass

    def get_photodiode_mask(self, nx: int, ny: int, nz: int) -> np.ndarray:
        """Binary mask for photodiode volumes"""
        pass
```

### 3.3 GeometryBuilder

```python
class GeometryBuilder:
    """Parametric geometry generation for all solver types"""

    @staticmethod
    def superellipse_lens(x: np.ndarray, y: np.ndarray,
                          center_x: float, center_y: float,
                          rx: float, ry: float,
                          height: float, n: float, alpha: float,
                          shift_x: float = 0.0, shift_y: float = 0.0) -> np.ndarray:
        """Generate superellipse microlens height map.
        z(x,y) = h * (1 - r^2)^(1/(2α))
        where r = (|x/rx|^n + |y/ry|^n)^(1/n)
        """
        dx = x - (center_x + shift_x)
        dy = y - (center_y + shift_y)
        r = (np.abs(dx / rx)**n + np.abs(dy / ry)**n)**(1.0 / n)
        r = np.clip(r, 0, 1)
        z = height * (1.0 - r**2)**(1.0 / (2.0 * alpha))
        z[r > 1.0] = 0.0
        return z

    @staticmethod
    def bayer_pattern(unit_cell: tuple, pattern: str = "rggb") -> List[List[str]]:
        """Generate Bayer color filter map for NxM unit cell"""
        base = {
            "rggb": [["R", "G"], ["G", "B"]],
            "grbg": [["G", "R"], ["B", "G"]],
            "gbrg": [["G", "B"], ["R", "G"]],
            "bggr": [["B", "G"], ["G", "R"]],
        }
        tile = base[pattern.lower()]
        rows, cols = unit_cell
        full = []
        for r in range(rows):
            row = []
            for c in range(cols):
                row.append(tile[r % 2][c % 2])
            full.append(row)
        return full

    @staticmethod
    def dti_grid(pitch: float, unit_cell: tuple,
                 dti_width: float, dti_depth: float) -> np.ndarray:
        """Generate DTI grid pattern as 2D binary mask"""
        pass
```

---

## 4. 솔버 어댑터 구현 가이드

### 4.1 RCWA 솔버 공통 패턴

RCWA 솔버는 z축 방향으로 layer를 분해하여 각 layer의 2D 유전율(permittivity) 분포를 Fourier 전개한다.

**핵심 변환 흐름:**
1. PixelStack → layer_slices (z별 2D epsilon 맵)
2. Microlens → multiple thin slices (staircase approximation)
3. Color filter → patterned layer with Bayer map
4. Silicon → absorbing medium (complex epsilon)
5. Photodiode → absorption monitor 위치

**각 RCWA 솔버의 차이점:**

| 항목 | grcwa | torcwa | meent |
|---|---|---|---|
| 단위계 | normalized | nm (SI-like) | nm 또는 normalized |
| 입력 형식 | epsilon grid (numpy) | epsilon grid (torch tensor) | vector/raster modeling |
| 출력 | diffraction efficiency | S-parameters | diffraction efficiency |
| 필드 추출 | layer별 field reconstruction | layer별 field + z-sweep | built-in field visualization |
| GPU | autograd backend | torch.cuda | JAX/PyTorch GPU |

### 4.2 FDTD 솔버 공통 패턴

FDTD는 3D 유전율 격자를 직접 구성하고 시간 영역에서 Maxwell 방정식을 풀어낸다.

**핵심 변환 흐름:**
1. PixelStack → 3D permittivity grid (nx, ny, nz)
2. Microlens → continuous surface → voxelized
3. Source → PlaneSource at top boundary
4. PML or PBC boundary conditions
5. Detector → DFT monitor planes for field/power
6. Photodiode → power absorption monitor

**각 FDTD 솔버 차이점:**

| 항목 | flaport/fdtd | fdtdz | meep |
|---|---|---|---|
| 언어 | Python (PyTorch) | JAX (CUDA kernel) | C++ (Python binding) |
| 입력 | Object placement | Permittivity array | Geometry primitives |
| 경계 | PML, Periodic | PML (z only), Periodic (xy) | PML, Periodic, Bloch |
| Source | LineSource, PlaneSource | Custom | PlaneWave, Gaussian |
| 분산 매질 | 제한적 | 미지원 (workaround) | Lorentz-Drude 지원 |
| 병렬화 | single GPU | single GPU (극한 최적화) | MPI + GPU |

### 4.3 RCWA-FDTD 변환 시 주의사항

1. **Microlens staircase**: RCWA는 렌즈를 z방향 thin slice로 근사. slice 수가 결과에 큰 영향 → convergence test 필수 (최소 20~50 slices)
2. **PBC vs PML**: RCWA는 inherently periodic. FDTD에서도 PBC 사용 시 비교 공정. PML 사용 시 unit cell 경계 효과 주의
3. **Source normalization**: RCWA는 입사 power normalized. FDTD는 source amplitude 정의 후 후처리. 정규화 방식 통일 필요
4. **Dispersive materials**: Silicon의 n,k는 파장 의존. RCWA는 파장별 계산이므로 문제 없음. FDTD는 Lorentz-Drude fitting 또는 파장별 개별 시뮬레이션 필요
5. **QE 계산 방식**: RCWA는 각 layer의 absorption을 direct 계산. FDTD는 Poynting vector flux 차이 또는 volume absorption integral 사용

---

## 4.4 RCWA 수치 안정성 (Numerical Stability) — 발산 대책

### 4.4.1 문제 진단: 왜 발산하는가

RCWA의 핵심 연산은 각 layer에서의 **eigenvalue decomposition**이다. 구조의 Fourier 전개된 permittivity 행렬로부터 eigenvalue problem `Ω² · W = W · Λ²`를 풀어 각 layer의 전자기 모드를 구한다. 이 과정에서 발생하는 수치 불안정성은 크게 4가지 원인으로 분류된다.

**원인 1: Eigendecomposition의 고유한 불안정성**
- 행렬의 condition number가 클 때 (고굴절률 대비가 큰 구조: metal grid + air 등) eigenvalue 계산이 불안정
- 중복 또는 근접한 eigenvalue 존재 시 eigenvector가 수치적으로 결정 불가
- float32에서는 유효숫자 ~7자리 → 행렬 크기가 커질수록 (높은 Fourier order) 정밀도 부족

**원인 2: T-matrix (Transfer Matrix)의 exponential 발산**
- 전통적 T-matrix 방식에서 `exp(+λ·d)` 항이 evanescent mode에 대해 지수적으로 증가
- 두꺼운 layer (Silicon 3~4um) + 높은 Fourier order → overflow 즉시 발생
- 이는 S-matrix 방식으로 전환하면 원천적으로 해결 가능

**원인 3: Fourier 전개의 Gibbs 현상**
- 불연속 permittivity 경계 (metal grid, DTI 경계)에서 Fourier 급수의 overshooting
- Li의 Fourier factorization rule을 적용하지 않으면 TM 편광에서 수렴이 극도로 느림
- 특히 crossed grating (2D 패턴)에서 Normal Vector Method 미적용 시 발산에 가까운 진동

**원인 4: GPU float32 연산의 정밀도 한계**
- `torch.complex64` (= float32 real + float32 imag): 유효숫자 ~7자리
- 대형 행렬의 eigendecomp에서 누적 오차가 결과를 지배
- TF32 Tensor Core (NVIDIA Ampere+) 활성화 시 더 악화 (유효숫자 ~3.5자리로 감소)
- PyTorch의 `torch.linalg.eig()` 자체가 GPU에서 CPU보다 불안정한 경우 존재

### 4.4.2 대책 체계 (Defense-in-Depth)

#### Layer 1: 산술 정밀도 관리

```python
class PrecisionManager:
    """RCWA 수치 정밀도 관리"""

    @staticmethod
    def configure(config: dict):
        # 1. TF32 비활성화 (필수)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        # 2. dtype 선택 전략
        #    - complex128 (float64): 안정적이지만 GPU에서 2~4x 느림
        #    - complex64 (float32): 빠르지만 발산 위험
        #    - 전략: float64로 eigendecomp, 나머지는 float32 (mixed precision)
        pass

    @staticmethod
    def mixed_precision_eigen(matrix: torch.Tensor) -> tuple:
        """Eigendecomp만 float64로 수행, 결과를 float32로 반환"""
        orig_dtype = matrix.dtype
        matrix_f64 = matrix.to(torch.complex128)

        # CPU fallback for eigendecomp (GPU eig보다 안정적)
        if matrix_f64.is_cuda:
            eigenvalues, eigenvectors = torch.linalg.eig(matrix_f64.cpu())
            eigenvalues = eigenvalues.to(matrix.device)
            eigenvectors = eigenvectors.to(matrix.device)
        else:
            eigenvalues, eigenvectors = torch.linalg.eig(matrix_f64)

        return eigenvalues.to(orig_dtype), eigenvectors.to(orig_dtype)
```

**구체적 정밀도 전략:**

| 연산 단계 | 권장 precision | 근거 |
|---|---|---|
| Permittivity convolution matrix | float32 OK | 단순 행렬곱, 오차 누적 적음 |
| Eigendecomposition | **float64 필수** | condition number 민감, 핵심 불안정 원인 |
| S-matrix Redheffer star product | float32 가능 (주의) | exponential은 S-matrix에서 bounded |
| Field reconstruction | float32 OK | 최종 출력, 상대적으로 안정 |

#### Layer 2: S-matrix 알고리즘 강제 (T-matrix 금지)

```python
class StableSMatrixAlgorithm:
    """
    Redheffer Star Product 기반 S-matrix recursion.
    T-matrix의 exp(+λd) 발산을 원천 차단.

    핵심: 모든 layer 전파에서 exp(-|λ|d) 만 사용 (감쇠항만 허용)
    참고: Moharam et al., JOSA A 12(5), 1995
          Rumpf, PIERB 35, 2011
    """

    @staticmethod
    def redheffer_star_product(SA: dict, SB: dict) -> dict:
        """
        Two S-matrices의 Redheffer star product.
        SA, SB: {'S11': ..., 'S12': ..., 'S21': ..., 'S22': ...}
        모든 중간 행렬이 bounded → overflow 불가
        """
        I = torch.eye(SA['S11'].shape[0], dtype=SA['S11'].dtype,
                      device=SA['S11'].device)

        # D = (I - SB['S11'] @ SA['S22'])^{-1}
        D = torch.linalg.solve(I - SB['S11'] @ SA['S22'], I)
        # F = (I - SA['S22'] @ SB['S11'])^{-1}
        F = torch.linalg.solve(I - SA['S22'] @ SB['S11'], I)

        S = {}
        S['S11'] = SA['S11'] + SA['S12'] @ D @ SB['S11'] @ SA['S21']
        S['S12'] = SA['S12'] @ D @ SB['S12']
        S['S21'] = SB['S21'] @ F @ SA['S21']
        S['S22'] = SB['S22'] + SB['S21'] @ F @ SA['S22'] @ SB['S12']
        return S

    @staticmethod
    def layer_smatrix(W, V, eigenvalues, thickness, k0):
        """
        단일 layer의 S-matrix 계산.
        exp(-j·λ·k0·d) 만 사용 — evanescent mode는 자연 감쇠
        """
        X = torch.diag(torch.exp(-1j * eigenvalues * k0 * thickness))
        # X의 모든 원소는 |X_ii| <= 1 (evanescent) 또는 |X_ii| = 1 (propagating)
        # → overflow 불가
        # ... S-matrix 조립 ...
        pass
```

#### Layer 3: Fourier Factorization Rule (Li's Rule) 적용

```python
class LiFactorization:
    """
    Li의 Fourier factorization rule 적용.
    불연속 permittivity의 Fourier 급수 수렴성을 획기적으로 개선.

    핵심 규칙:
    - Type 1 (Laurent rule): 연속 함수의 곱 → 일반 convolution
    - Type 2 (Inverse rule): f·g가 연속이고 f가 불연속 →
      [[f·g]] = [[f]]^{-1} · [[g]] 가 아닌 [[1/f]]^{-1} · [[f·g]] 사용
    - Type 3: f와 g 모두 불연속이고 곱도 불연속 → 사용 불가, NV method 필요

    참고: Li, JOSA A 13(9), 1870-1876, 1996
    """

    @staticmethod
    def convolution_matrix_with_inverse_rule(
        eps_grid: torch.Tensor,
        n_harmonics: int
    ) -> torch.Tensor:
        """
        역규칙(inverse rule) 적용한 convolution matrix 생성.
        eps의 Fourier 변환 대신 1/eps의 Fourier 변환의 역행렬 사용.
        Metal grid, DTI 경계 등 고대비 불연속에서 필수.
        """
        eps_inv = 1.0 / eps_grid
        convmat_inv = toeplitz_from_fft(eps_inv, n_harmonics)
        return torch.linalg.inv(convmat_inv)

    @staticmethod
    def normal_vector_method(
        eps_grid: torch.Tensor,
        nx: int, ny: int,
        n_harmonics: int
    ) -> tuple:
        """
        Normal Vector Method for 2D crossed gratings.
        구조 경계에서의 법선 벡터 필드를 생성하여
        E-field 분해 (tangential/normal) 후 적절한 factorization 적용.

        참고: Schuster et al., JOSA A 24(9), 2880-2890, 2007
              Götz et al., Opt. Express 16(22), 17295-17301, 2008
        """
        # 1. 구조 경계 검출 (gradient of eps)
        # 2. 법선 벡터 필드 n(x,y) 계산
        # 3. Tangential/Normal 분해 행렬 구성
        # 4. 각 성분에 올바른 factorization rule 적용
        pass
```

#### Layer 4: Eigenvalue 후처리 및 검증

```python
class EigenvalueStabilizer:
    """Eigenvalue 결과의 안정화 및 검증"""

    @staticmethod
    def fix_degenerate_eigenvalues(
        eigenvalues: torch.Tensor,
        eigenvectors: torch.Tensor,
        broadening: float = 1e-10
    ) -> tuple:
        """
        근접/중복 eigenvalue 처리.
        torcwa 방식: broadening parameter로 gradient 안정화.
        """
        # 중복 eigenvalue 검출
        diffs = torch.abs(eigenvalues.unsqueeze(-1) - eigenvalues.unsqueeze(-2))
        degenerate_mask = (diffs < broadening) & (diffs > 0)

        if degenerate_mask.any():
            # Gram-Schmidt orthogonalization of degenerate eigenvectors
            eigenvectors = orthogonalize_degenerate_subspace(
                eigenvectors, eigenvalues, broadening)

        return eigenvalues, eigenvectors

    @staticmethod
    def select_propagation_direction(eigenvalues: torch.Tensor) -> torch.Tensor:
        """
        Eigenvalue의 부호 선택 (forward/backward propagating mode 결정).
        sqrt(λ²)에서 올바른 branch 선택 실패 시 에너지 보존 위반.

        규칙:
        - Re(λ) > 0 이면 forward propagating
        - Re(λ) ≈ 0 이면 Im(λ) > 0 선택 (감쇠 방향)
        - 잘못된 부호 → 에너지가 증가하는 비물리적 결과
        """
        sqrt_eigenvalues = torch.sqrt(eigenvalues)

        # 올바른 branch: Im(sqrt_λ) < 0 (시간 convention e^{-iωt})
        # 또는 Re(sqrt_λ) > 0 (propagating mode)
        wrong_sign = torch.real(sqrt_eigenvalues) < 0
        wrong_sign_2 = (torch.abs(torch.real(sqrt_eigenvalues)) < 1e-10) & \
                       (torch.imag(sqrt_eigenvalues) < 0)
        flip_mask = wrong_sign | wrong_sign_2
        sqrt_eigenvalues[flip_mask] *= -1

        return sqrt_eigenvalues

    @staticmethod
    def validate_energy_conservation(
        S_matrix: dict,
        tolerance: float = 0.05
    ) -> dict:
        """
        S-matrix에서 에너지 보존 검증.
        |S11|² + |S21|² ≤ 1 (lossless), = 1 (lossless + non-absorbing)
        위반 시 경고 또는 해당 파장점 재계산 (float64 fallback).
        """
        R = torch.sum(torch.abs(S_matrix['S11'])**2, dim=-1)
        T = torch.sum(torch.abs(S_matrix['S21'])**2, dim=-1)
        total = R + T  # for non-absorbing: should be ≈ 1

        violation_mask = total > 1.0 + tolerance
        result = {
            'valid': not violation_mask.any(),
            'max_violation': (total - 1.0).max().item(),
            'violation_indices': torch.where(violation_mask)[0].tolist()
        }
        return result
```

#### Layer 5: Adaptive Precision Fallback

```python
class AdaptivePrecisionRunner:
    """
    발산 감지 시 자동으로 정밀도를 올리는 adaptive runner.
    기본 전략: float32 시도 → 검증 실패 → float64 재시도 → CPU fallback
    """

    def run_with_fallback(self, solver, wavelength, config):
        """
        3단계 fallback 전략:
        1. GPU float32 (fast) → energy check
        2. GPU float64 (stable) → energy check
        3. CPU float64 (most stable) → final result
        """
        strategies = [
            {'dtype': 'complex64',  'device': 'cuda', 'label': 'GPU-f32'},
            {'dtype': 'complex128', 'device': 'cuda', 'label': 'GPU-f64'},
            {'dtype': 'complex128', 'device': 'cpu',  'label': 'CPU-f64'},
        ]

        for strategy in strategies:
            try:
                result = solver.run(
                    wavelength=wavelength,
                    dtype=strategy['dtype'],
                    device=strategy['device']
                )

                # 에너지 보존 검증
                energy_check = EigenvalueStabilizer.validate_energy_conservation(
                    result.s_matrix)

                if energy_check['valid']:
                    if strategy['label'] != 'GPU-f32':
                        logger.warning(
                            f"λ={wavelength:.3f}um: fallback to {strategy['label']}"
                        )
                    return result
                else:
                    logger.warning(
                        f"λ={wavelength:.3f}um: energy violation "
                        f"{energy_check['max_violation']:.4f} "
                        f"with {strategy['label']}, trying next strategy"
                    )

            except (RuntimeError, torch.linalg.LinAlgError) as e:
                logger.warning(
                    f"λ={wavelength:.3f}um: {strategy['label']} failed: {e}"
                )
                continue

        raise RuntimeError(
            f"All precision strategies failed for λ={wavelength:.3f}um. "
            f"Consider reducing Fourier order or checking structure definition."
        )
```

### 4.4.3 Config에서의 안정성 설정

```yaml
# configs/solver/torcwa.yaml 에 추가
solver:
  name: "torcwa"
  type: "rcwa"

  stability:
    # Precision 전략
    precision_strategy: "mixed"     # "float32", "float64", "mixed", "adaptive"
    # mixed: eigendecomp만 float64, 나머지 float32
    # adaptive: float32 시도 후 실패 시 자동 fallback

    # TF32 제어
    allow_tf32: false               # 절대 true로 설정하지 말 것

    # Eigendecomp 위치
    eigendecomp_device: "cpu"       # "cpu" (안정적) 또는 "gpu" (빠르지만 위험)

    # Li factorization
    fourier_factorization: "li_inverse"  # "naive", "li_inverse", "normal_vector"
    # naive: 기본 Laurent rule만 (불안정, 테스트용만)
    # li_inverse: Li inverse rule (1D/2D 모두 적용, 권장)
    # normal_vector: NV method (2D crossed grating 최적, 가장 안정적)

    # 에너지 보존 검증
    energy_check:
      enabled: true
      tolerance: 0.02              # R+T+A > 1+tolerance 시 경고
      auto_retry_float64: true     # 위반 시 float64로 자동 재시도

    # Eigenvalue broadening (중복 eigenvalue 처리)
    eigenvalue_broadening: 1.0e-10

    # Condition number 모니터링
    condition_number_warning: 1.0e+12  # 이 값 초과 시 경고 로그
```

### 4.4.4 발산 발생 시나리오별 대응 요약

| 시나리오 | 증상 | 원인 | 대응 |
|---|---|---|---|
| 높은 Fourier order + float32 | QE > 1 또는 NaN | Eigendecomp 정밀도 부족 | `precision_strategy: "mixed"` 또는 `"adaptive"` |
| Metal grid (W) 포함 구조 | TM편광에서 느린 수렴/진동 | Gibbs 현상 + 잘못된 factorization | `fourier_factorization: "li_inverse"` 이상 |
| 두꺼운 Si layer (>3um) | Overflow, Inf 값 | T-matrix exponential 발산 | S-matrix 알고리즘 강제 (torcwa는 기본 S-matrix) |
| 근접 eigenvalue 발생 | Eigenvector 방향 불안정 | 대칭 구조에서의 degenerate mode | `eigenvalue_broadening` 조정, Gram-Schmidt |
| TF32 활성 상태 | 전 파장에서 큰 오차 | Tensor core의 낮은 정밀도 | `allow_tf32: false` (필수) |
| 높은 k 재료 (짧은 파장 Si) | 흡수층에서 큰 Im(ε) | 큰 imaginary eigenvalue → 불안정 | Float64 + Fourier order 낮춤 |
| Microlens staircase 다수 layer | 누적 오차 | 50+ layer의 S-matrix star product | Layer 병합 전략 또는 float64 |

### 4.4.5 자동 진단 도구

```python
class StabilityDiagnostics:
    """시뮬레이션 전/중/후 안정성 진단"""

    @staticmethod
    def pre_simulation_check(pixel_stack, solver_config) -> List[str]:
        """시뮬레이션 전 위험 요소 사전 진단"""
        warnings = []

        # 1. 고대비 재료 검사
        max_eps_ratio = compute_max_epsilon_contrast(pixel_stack)
        if max_eps_ratio > 50:
            warnings.append(
                f"High epsilon contrast ratio ({max_eps_ratio:.0f}). "
                f"Recommend fourier_factorization: 'li_inverse' or 'normal_vector'"
            )

        # 2. Fourier order vs precision 검사
        order = solver_config['params']['fourier_order']
        matrix_size = (2 * order[0] + 1) * (2 * order[1] + 1)
        if matrix_size > 200 and solver_config.get('stability', {}).get(
            'precision_strategy') == 'float32':
            warnings.append(
                f"Large matrix ({matrix_size}x{matrix_size}) with float32. "
                f"High risk of eigendecomp instability. "
                f"Recommend precision_strategy: 'mixed'"
            )

        # 3. Layer 두께 검사
        for layer in pixel_stack.layers:
            if layer.thickness > 2.0:  # um
                warnings.append(
                    f"Thick layer '{layer.name}' ({layer.thickness}um). "
                    f"Ensure S-matrix algorithm is used (not T-matrix)."
                )

        # 4. TF32 상태 검사
        if torch.backends.cuda.matmul.allow_tf32:
            warnings.append(
                "TF32 is ENABLED. This will cause numerical instability in RCWA. "
                "Set torch.backends.cuda.matmul.allow_tf32 = False"
            )

        return warnings

    @staticmethod
    def post_simulation_check(result: SimulationResult) -> dict:
        """시뮬레이션 후 결과 검증"""
        report = {}

        # QE 범위 검사
        for pixel_name, qe in result.qe_per_pixel.items():
            if np.any(qe < -0.01) or np.any(qe > 1.01):
                report[pixel_name] = {
                    'status': 'FAILED',
                    'issue': f'QE out of range [{qe.min():.4f}, {qe.max():.4f}]'
                }

        # NaN/Inf 검사
        if result.fields is not None:
            for name, field in result.fields.items():
                if np.any(np.isnan(field)) or np.any(np.isinf(field)):
                    report[f'field_{name}'] = {
                        'status': 'FAILED',
                        'issue': 'NaN or Inf detected'
                    }

        return report
```

---

## 5. Material Database 설계

### 5.1 파일 포맷

```csv
# materials/silicon_palik.csv
# wavelength(um), n, k
0.350, 5.565, 3.004
0.360, 5.827, 2.989
...
0.780, 3.696, 0.006
```

### 5.2 Material Registry

```python
class MaterialDB:
    """Central material property database"""

    def __init__(self, db_path: str = "materials/"):
        self._materials: Dict[str, MaterialData] = {}
        self._load_builtin(db_path)

    def get_epsilon(self, name: str, wavelength: float) -> complex:
        """Get complex permittivity at given wavelength.
           ε = (n + ik)²
        """
        mat = self._materials[name]
        n, k = mat.interpolate(wavelength)
        return (n + 1j * k) ** 2

    def get_epsilon_spectrum(self, name: str,
                             wavelengths: np.ndarray) -> np.ndarray:
        """Vectorized epsilon over wavelength array"""
        pass
```

### 5.3 분석 모델 지원

```yaml
# 재료 정의 방식 3가지
material_definitions:
  # 1. 상수
  air:
    type: "constant"
    n: 1.0
    k: 0.0

  # 2. 테이블 (CSV 참조)
  silicon:
    type: "tabulated"
    file: "silicon_green2008.csv"
    interpolation: "cubic_spline"

  # 3. 분석 모델
  polymer_n1p56:
    type: "cauchy"
    params:
      A: 1.56
      B: 0.004   # um²
      C: 0.0     # um⁴
```

---

## 6. Compute Backend 설계

### 6.1 Device Abstraction

```python
class ComputeBackend:
    """Hardware abstraction layer"""

    @staticmethod
    def get_device(config: dict) -> str:
        backend = config.get('compute', {}).get('backend', 'auto')
        if backend == 'auto':
            if torch.cuda.is_available():
                return 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'
            else:
                return 'cpu'
        return backend

    @staticmethod
    def to_tensor(array: np.ndarray, device: str,
                   dtype: str = 'float32') -> 'torch.Tensor':
        """Convert numpy array to device tensor"""
        import torch
        torch_dtype = getattr(torch, dtype)
        return torch.tensor(array, dtype=torch_dtype, device=device)
```

### 6.2 플랫폼별 주의사항

| 플랫폼 | GPU 가속 | 주의사항 |
|---|---|---|
| Linux + NVIDIA | CUDA (primary) | torch.cuda, JAX CUDA |
| macOS Apple Silicon | MPS (실험적) | complex tensor 연산 제한, eigendecomp CPU fallback |
| macOS Intel | CPU only | float64 권장 |
| Windows + NVIDIA | CUDA | WSL2 권장 |
| CPU any | numpy/torch CPU | float64 사용, 병렬화 제한 |

**MPS 제한사항:**
- Complex tensor 연산 미지원 → RCWA eigendecomp CPU fallback 필수
- torcwa, meent의 PyTorch backend에서 mps device 사용 시 성능 이점 제한적
- FDTD는 real tensor만 사용하므로 MPS 활용 가능성 있음

---

## 7. 시각화 모듈 설계

### 7.1 2D Structure Cross-section

```python
def plot_pixel_cross_section(pixel_stack: PixelStack,
                              plane: str = 'xz',
                              position: float = 0.0,
                              wavelength: float = 0.55):
    """
    구조 단면도 with 재료별 색상 매핑
    - plane: 'xz' (수직 단면), 'xy' (수평 단면)
    - 각 재료에 고유 색상 할당
    - 경계면 윤곽선 표시
    - 레이어 이름 annotation
    """
```

### 7.2 2D Field Plot

```python
def plot_field_2d(result: SimulationResult,
                   component: str = '|E|2',
                   plane: str = 'xz',
                   position: float = 0.0,
                   overlay_structure: bool = True):
    """
    EM 필드 2D 맵
    - 구조 윤곽선 오버레이 가능
    - colorbar with physical units
    - 다중 파장 subplot 가능
    """
```

### 7.3 3D Interactive Viewer

```python
def view_pixel_3d(pixel_stack: PixelStack,
                   backend: str = 'pyvista'):
    """
    Interactive 3D visualization
    - backend: 'pyvista' (desktop), 'plotly' (notebook/web)
    - 레이어별 on/off toggle
    - 재료별 색상/투명도
    - Photodiode 영역 하이라이트
    - DTI 구조 표시
    """
```

**3D Viewer 기술 선택:**
- **pyvista**: VTK 기반, 고품질 3D 렌더링, Jupyter 통합
- **plotly**: Web 기반, 가벼움, 인터랙티브
- 두 backend 모두 지원하되 pyvista를 primary로

---

## 8. 결과 저장 스키마 (HDF5)

```
result.h5
├── metadata/
│   ├── config.yaml           # 전체 설정 파일 (재현성)
│   ├── solver_info            # 솔버 버전, 실행 시간
│   └── timestamp
├── qe/
│   ├── wavelengths            # (N_wavelength,)
│   ├── pixel_R_0_0            # (N_wavelength,) - QE spectrum per pixel
│   ├── pixel_G_0_1
│   ├── pixel_G_1_0
│   ├── pixel_B_1_1
│   └── crosstalk_matrix       # (N_pixel, N_pixel, N_wavelength)
├── fields/                    # Optional (large data)
│   ├── Ex/                    # per wavelength
│   │   ├── lambda_450nm       # 3D complex array
│   │   └── ...
│   ├── Ey/
│   └── Ez/
├── energy_balance/
│   ├── reflection             # (N_wavelength,)
│   ├── transmission
│   └── absorption
└── convergence/               # Optional
    ├── qe_vs_order            # RCWA convergence data
    └── qe_vs_resolution       # FDTD convergence data
```

---

## 9. 의존성

### 9.1 Core Dependencies
```
python >= 3.10
numpy >= 1.24
torch >= 2.0
hydra-core >= 1.3
omegaconf >= 2.3
pydantic >= 2.0
h5py >= 3.8
matplotlib >= 3.7
scipy >= 1.10
pyyaml >= 6.0
```

### 9.2 Solver Dependencies (Optional)
```
# RCWA
grcwa >= 0.1.0        # pip install grcwa
torcwa >= 0.1.4       # pip install torcwa
meent >= 0.12.0       # pip install meent

# FDTD
fdtd >= 0.3.0         # pip install fdtd (flaport)
fdtdz                  # pip install fdtdz (JAX required)
meep                   # conda install -c conda-forge pymeep

# JAX (for meent/fdtdz)
jax[cuda] >= 0.4.0    # or jax[cpu]
```

### 9.3 Visualization Dependencies
```
pyvista >= 0.39
plotly >= 5.14
jupyter >= 1.0
```

---

## 10. 테스트 전략

### 10.1 Unit Tests
- MaterialDB: 보간 정확도 검증 (문헌값 대비)
- GeometryBuilder: 렌즈 형상 정확도, Bayer 패턴 생성
- PixelStack: config → 구조 변환 정확성
- 각 Solver adapter: 단일 slab 구조에 대한 해석해 비교

### 10.2 Integration Tests
- Config → PixelStack → Solver → Result 전체 파이프라인
- 동일 구조의 솔버 간 QE 비교 (±5% 이내)
- HDF5 저장 → 로드 왕복 검증

### 10.3 Benchmark Tests
- **Slab verification**: 단일 균일 매질 → 해석적 Fresnel 반사 대비 (0.1% 이내)
- **Grating benchmark**: 문헌 기반 회절 격자 → grating efficiency 비교
- **Energy conservation**: 모든 구조에서 R + T + A ≈ 1 (1% 이내)
- **Convergence**: Fourier order / grid resolution vs QE 수렴 곡선

---

## 11. Agent Teams 개발 전략

### 11.1 개요: 왜 Agent Teams인가

COMPASS는 광학 시뮬레이션, 수치해석, GPU 최적화, 시각화, 테스트 등 이질적인 전문 영역이 동시에 필요한 프로젝트다. 단일 세션으로 개발하면 context window 제약, 도메인 전환 비용, 순차 실행의 병목이 심각해진다.

**Claude Code Agent Teams**를 활용하면:
- **병렬 개발**: 독립적인 모듈을 동시에 구현 (RCWA adapter / FDTD adapter / 시각화 / 테스트)
- **전문화**: 각 에이전트가 특정 도메인에 집중 (수치해석 전문 vs. UI/시각화 전문)
- **교차 검증**: 에이전트 간 코드 리뷰 및 결과 비교로 품질 향상
- **Context 격리**: 각 에이전트가 자체 context window를 가지므로 대형 코드베이스에서도 집중도 유지

### 11.2 팀 구조 설계

```
┌─────────────────────────────────────────────────────┐
│                   COMPASS Team Lead                  │
│  역할: 아키텍처 결정, 태스크 분배, 통합 검증, 충돌 해결  │
│  CLAUDE.md: 프로젝트 컨벤션, 코드 표준, 의존성 그래프    │
└──────────┬──────────┬──────────┬──────────┬─────────┘
           │          │          │          │
    ┌──────▼──┐ ┌─────▼────┐ ┌──▼───────┐ ┌▼─────────┐
    │ Agent 1 │ │ Agent 2  │ │ Agent 3  │ │ Agent 4  │
    │ Core    │ │ Solvers  │ │ Viz & IO │ │ QA &     │
    │ Engine  │ │ & Physics│ │          │ │ Benchmark│
    └─────────┘ └──────────┘ └──────────┘ └──────────┘
```

### 11.3 에이전트 역할 정의

#### Agent 1: Core Engine (기반 구조 전담)

```yaml
name: "core-engine"
scope:
  owns:
    - compass/config/          # Hydra config system
    - compass/geometry/        # PixelStack, GeometryBuilder
    - compass/materials/       # MaterialDB
    - compass/core/            # Units, constants, base classes
  reads:
    - compass/solvers/base.py  # SolverBase ABC 참조
  context_prompt: |
    You are the Core Engine agent for COMPASS.
    Your responsibility: Hydra config system, PixelStack geometry builder,
    MaterialDB, and all foundation classes. Focus on clean APIs that
    solver agents will consume. All geometry generation must be
    solver-agnostic. Use float64 for geometry calculations.
    Key constraint: PixelStack must produce both layer-slice output
    (for RCWA) and voxel-grid output (for FDTD) from the same config.
tasks:
  phase1:
    - "1.1: Hydra config + pyproject.toml (priority: P0)"
    - "1.2: MaterialDB with CSV, interpolation, Cauchy (P0)"
    - "1.3: PixelStack + GeometryBuilder — superellipse, Bayer, DTI (P0)"
    - "1.4: Unit system, coordinate conventions (P0)"
  phase2:
    - "2.1: SolverBase ABC + SimulationResult dataclass (P0)"
    - "2.5: PlanewaveSource module (P0)"
```

#### Agent 2: Solvers & Physics (솔버 구현 전담)

```yaml
name: "solvers-physics"
scope:
  owns:
    - compass/solvers/rcwa/    # torcwa, grcwa, meent adapters
    - compass/solvers/fdtd/    # flaport, fdtdz, meep adapters
    - compass/solvers/rcwa/stability.py  # 수치 안정성 모듈
    - compass/sources/         # Planewave, Cone illumination
    - compass/analysis/        # QE calculator, solver comparison
  reads:
    - compass/core/            # Base classes
    - compass/geometry/        # PixelStack API
    - compass/materials/       # MaterialDB API
  context_prompt: |
    You are the Solvers & Physics agent for COMPASS.
    Your responsibility: all EM solver adapters (RCWA: torcwa, grcwa, meent;
    FDTD: flaport, fdtdz, meep), numerical stability module, source models,
    and QE analysis.
    CRITICAL: RCWA numerical stability is the #1 priority. Implement
    5-layer defense: mixed precision, S-matrix only, Li factorization,
    eigenvalue stabilization, adaptive fallback. See TRD Section 4.4.
    All solvers must satisfy energy conservation R+T+A≈1 within 1%.
    Cross-validation target: ΔQE < 5% between RCWA solvers.
tasks:
  phase2:
    - "2.2: torcwa adapter — primary RCWA solver (P0)"
    - "2.3: grcwa adapter — cross-validation target (P1)"
    - "2.4: meent adapter — multi-backend (P1)"
    - "2.6: RCWA numerical stability module — 5-layer defense (P0)"
  phase4:
    - "4.1: flaport/fdtd adapter (P1)"
    - "4.2: fdtdz adapter (P2)"
    - "4.3: meep adapter (P2)"
  phase5:
    - "5.1: Cone illumination model (P1)"
    - "5.3: ROI sweep runner (P1)"
  phase6:
    - "6.1: QE calculator — RCWA vs FDTD 통합 (P0)"
    - "6.2: Solver comparison module (P1)"
```

#### Agent 3: Visualization & I/O (시각화, 문서화, 입출력)

```yaml
name: "viz-io"
scope:
  owns:
    - compass/viz/             # 2D/3D plotting modules
    - compass/io/              # HDF5, CSV, JSON export
    - docs/                    # VitePress 문서 사이트 전체
    - notebooks/               # Jupyter examples
  reads:
    - compass/core/            # SimulationResult 구조
    - compass/geometry/        # PixelStack (구조 시각화)
  context_prompt: |
    You are the Visualization & I/O agent for COMPASS.
    Your responsibility: all plotting (2D cross-sections, field plots,
    QE spectral plots, 3D interactive viewer), HDF5/CSV data storage,
    VitePress documentation site, and Jupyter notebooks.
    DOCUMENTATION IS A FIRST-CLASS DELIVERABLE. The VitePress site
    must serve as both educational resource (Theory section for
    non-experts) and technical reference (API docs, config reference).
    Theory pages must use everyday analogies, interactive Vue
    components, and minimal jargon. All math equations must be
    followed by plain-language explanations.
    See TRD Section 12 for complete VitePress structure.
tasks:
  phase3:
    - "3.1: 2D structure cross-section plots (P0)"
    - "3.2: 2D field distribution plots (P1)"
    - "3.3: 3D interactive viewer (P1)"
    - "3.4: QE spectral comparison plots (P0)"
  phase7:
    - "7.1: HDF5 storage module (P1)"
    - "7.2: CSV/JSON export (P1)"
  phase8:
    - "8.1: VitePress project setup (P0)"
    - "8.2: Theory — light & EM basics (P1)"
    - "8.3: Theory — RCWA & FDTD explanation (P1, with Agent 2 review)"
    - "8.4: Guide — installation & quickstart (P0)"
    - "8.5: Guide — advanced topics (P1)"
    - "8.6: Reference — API & config docs (P1)"
    - "8.7: Cookbook — practical recipes (P1)"
    - "8.8: Jupyter notebooks (P1)"
    - "8.9: References & project info (P2)"
```

#### Agent 4: QA & Benchmark (품질 보증 전담)

```yaml
name: "qa-benchmark"
scope:
  owns:
    - tests/                   # unit, integration, benchmark tests
    - benchmarks/              # Performance benchmark suite
    - compass/diagnostics/     # StabilityDiagnostics, validation
  reads:
    - compass/                 # 전체 코드베이스 (리뷰 목적)
  context_prompt: |
    You are the QA & Benchmark agent for COMPASS.
    Your responsibility: comprehensive testing, code review of other
    agents' output, benchmark suite (Fresnel slab, grating, energy
    conservation), and stability diagnostics validation.
    CRITICAL ROLE: You are the gatekeeper. No code merges without:
    1. Unit tests passing (pytest)
    2. Energy conservation check (R+T+A < 1.01)
    3. Fresnel slab benchmark < 0.5% error
    4. No NaN/Inf in field outputs
    5. Type checking (mypy) clean
    Challenge other agents' implementations. If solver results
    look suspicious, raise it immediately via team messaging.
tasks:
  phase9_continuous:
    - "9.1: Unit tests per module — run continuously (P0)"
    - "9.2: Benchmark suite — Fresnel, grating, energy conservation (P1)"
    - "9.3: Cross-agent code review (P1)"
    - "9.4: CI configuration — pytest, mypy, ruff (P1)"
```

### 11.4 Phase별 팀 운영 전략

#### Phase 1-2: Foundation + First Solver (Week 1-3)

```
패턴: Pipeline (순차 의존) + Fan-Out (병렬 작업)

┌──────────┐    ┌──────────┐    ┌──────────┐
│ Agent 1  │───▶│ Agent 2  │───▶│ Agent 4  │
│ Core     │    │ Solvers  │    │ QA       │
│ (P1 W1)  │    │ (P2 W2)  │    │ (검증)   │
└──────────┘    └──────────┘    └──────────┘
                     │
                ┌────▼─────┐
                │ Agent 3  │  ← Agent 1 완료 후 병렬 시작
                │ Viz 기반  │
                └──────────┘
```

**Task 흐름:**
1. **Agent 1** (Week 1): config, MaterialDB, PixelStack, units 완성 → task list에 "foundation-complete" 등록
2. **Agent 2** (Week 2, blocked_by: foundation-complete): SolverBase ABC 정의, torcwa adapter 구현, stability module
3. **Agent 3** (Week 2, blocked_by: foundation-complete): 구조 시각화 모듈 시작 (PixelStack만 있으면 가능)
4. **Agent 4** (Week 1부터): unit test 작성, 다른 에이전트 산출물 즉시 검증

**에이전트 간 메시징 예시:**
```
Agent 1 → team-lead: "PixelStack.get_layer_slices() API 확정. 
  반환: List[LayerSlice(z_start, z_end, eps_grid)]. Agent 2 참조."

Agent 2 → Agent 1: "MaterialDB에서 wavelength 범위 밖 extrapolation 요청 시 
  동작이 undefined. clamp vs exception 중 결정 필요."

Agent 1 → Agent 2: "clamp + warning log로 결정. extrapolation_mode 
  config 추가함."

Agent 4 → team-lead: "torcwa adapter의 energy conservation test에서 
  metal grid 구조 Fourier order 10에서 R+T+A=1.08. 
  Agent 2의 stability module 적용 전인지 확인 필요."
```

#### Phase 3-4: Visualization + FDTD (Week 3-5)

```
패턴: Fan-Out (전면 병렬)

┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│ Agent 1  │  │ Agent 2  │  │ Agent 3  │  │ Agent 4  │
│ Cone src │  │ FDTD     │  │ 2D/3D    │  │ 벤치마크  │
│ ROI sweep│  │ adapters │  │ viz + IO │  │ 스위트    │
└──────────┘  └──────────┘  └──────────┘  └──────────┘
      ↕              ↕             ↕             ↕
  ════════════ shared task list ════════════════════
```

**이 단계의 핵심:** 4개 에이전트가 완전 독립적으로 작업. Agent 4가 지속적으로 다른 에이전트 산출물을 pull하여 테스트.

#### Phase 5-6: Advanced Sources + Analysis (Week 5-7)

```
패턴: Map-Reduce (분산 계산 → 결과 통합)

Agent 2: Cone illumination 구현
Agent 1: ROI sweep runner 구현  
                    ↘
                  Agent 4: 솔버 비교 자동화
                    ↗          ↓
Agent 3: QE 비교 리포트 시각화 ← 비교 결과 수신
```

#### Phase 7-9: I/O + Documentation + Testing (Week 7-8)

```
패턴: Fan-Out + Competitive Review

Agent 3: HDF5/CSV + Jupyter notebooks
Agent 4: 전체 integration test + 최종 벤치마크
Agent 2: API documentation 리뷰 (물리적 정확성)
Agent 1: README, INSTALL guide
```

### 11.5 CLAUDE.md 설계

프로젝트 루트에 두는 CLAUDE.md는 모든 에이전트가 공유하는 컨텍스트:

```markdown
# COMPASS — CLAUDE.md

## Project Overview
COMPASS (Cross-solver Optical Modeling Platform for Advanced Sensor Simulation)
CMOS image sensor pixel optics의 multi-solver EM 시뮬레이션 교차검증 플랫폼.

## Architecture Quick Reference
compass/
├── config/         # Hydra configs (Agent 1 owns)
├── core/           # Base classes, units (Agent 1 owns)
├── geometry/       # PixelStack, GeometryBuilder (Agent 1 owns)
├── materials/      # MaterialDB (Agent 1 owns)
├── solvers/
│   ├── base.py     # SolverBase ABC (Agent 1 owns)
│   ├── rcwa/       # torcwa, grcwa, meent, stability (Agent 2 owns)
│   └── fdtd/       # flaport, fdtdz, meep (Agent 2 owns)
├── sources/        # Planewave, Cone illumination (Agent 2 owns)
├── analysis/       # QE calc, solver comparison (Agent 2 owns)
├── viz/            # Plotting modules (Agent 3 owns)
├── io/             # HDF5, CSV export (Agent 3 owns)
└── diagnostics/    # Stability checks (Agent 4 owns)
docs/                # VitePress documentation site (Agent 3 owns)
├── theory/          # 이론 기초 (비전공자 대상)
├── guide/           # 사용 가이드
├── reference/       # API & config 레퍼런스
├── cookbook/         # 실전 레시피
└── contributing/    # 기여 가이드

## Code Conventions
- Python 3.10+, type hints mandatory (mypy strict)
- Docstring: Google style, 물리 수식은 LaTeX notation
- Units: um (length), degrees (angle input) → radians (internal)
- Coordinate: z-axis = stack direction (air → Si, top → bottom)
- Naming: snake_case (functions/variables), PascalCase (classes)
- Linter: ruff, formatter: black
- Test: pytest, minimum 80% coverage per module

## Critical Physics Constraints
- Energy conservation: R + T + A ≈ 1 (tolerance < 0.02)
- QE range: 0 ≤ QE ≤ 1 (violation = bug)
- RCWA: S-matrix ONLY (T-matrix 절대 금지)
- RCWA: TF32 비활성화 필수 (torch.backends.cuda.matmul.allow_tf32 = False)
- RCWA: eigendecomp은 float64 또는 CPU fallback 권장
- Li factorization: 고대비 구조에서 inverse rule 필수

## Cross-Agent Communication Protocol
- API 변경 시 반드시 team-lead에게 메시지
- 새 public method/class 추가 시 docstring + type hints 필수
- Solver 결과 비정상 감지 시 Agent 4에게 즉시 메시지
- 공유 dataclass 수정 시 affected agents에 notification
```

### 11.6 태스크 의존성 그래프 (Task DAG)

```
Tasks with blockedBy relationships:

1.1 config          ──┐
1.2 materialdb      ──┤
1.3 pixelstack      ──┼──▶ [foundation-complete]
1.4 units           ──┘         │
                                │
         ┌──────────────────────┼───────────────────┐
         ▼                      ▼                   ▼
    2.1 solver_base        3.1 struct_viz      9.1 unit_tests
    2.5 planewave              │                    │
         │                 3.4 qe_plots         (continuous)
         ▼                     │
    2.2 torcwa ◄───────────────┤
    2.6 stability              │
         │                     ▼
         ├──▶ 2.3 grcwa   3.2 field_plots
         ├──▶ 2.4 meent   3.3 3d_viewer
         │
         ▼
    6.1 qe_calculator
    6.2 solver_comparison
         │
         ├──▶ 4.1 flaport_fdtd
         ├──▶ 4.2 fdtdz
         ├──▶ 4.3 meep
         │
         ▼
    5.1 cone_illumination
    5.3 roi_sweep
         │
         ▼
    7.1 hdf5_storage
    7.2 csv_export
         │
         ▼
    8.1 notebooks
    8.2 documentation
```

### 11.7 실행 명령 예시

```bash
# 1. Agent Teams 활성화
export CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1

# 2. Team Lead에서 팀 생성 및 에이전트 스폰
# (Team Lead 프롬프트)
Create an agent team called "compass-dev".
Read CLAUDE.md for project context.

Spawn 4 teammates:

1. "core-engine" — owns compass/config, compass/geometry,
   compass/materials, compass/core. Start with Task 1.1-1.4.

2. "solvers-physics" — owns compass/solvers, compass/sources,
   compass/analysis. Blocked by foundation-complete.
   When unblocked, start with Task 2.1, 2.2, 2.6.

3. "viz-io" — owns compass/viz, compass/io, notebooks, docs.
   Blocked by foundation-complete.
   When unblocked, start with Task 3.1, 3.4.

4. "qa-benchmark" — owns tests, benchmarks. Start immediately.
   Write unit test stubs for all modules. Review code from
   other agents as it completes. Run energy conservation
   checks on every solver output.

Create tasks from TASKS.md with proper blockedBy relationships.
Enable self-claim so agents pick up next tasks automatically.

# 3. 개별 에이전트 모니터링
# tmux pane에서 각 에이전트의 작업 상황 확인 가능

# 4. 에이전트 간 디버깅 세션 (경쟁적 가설 검증)
# 예: RCWA 발산 원인 조사
Team lead: "torcwa QE > 1 at λ=450nm for metal grid structure.
  Spawn a debug sub-team:
  - solvers-physics: check eigendecomp precision
  - qa-benchmark: reproduce with minimal config, test float64
  Have them share findings and converge on root cause."
```

### 11.8 에이전트 간 충돌 방지 규칙

| 규칙 | 설명 |
|---|---|
| **파일 소유권** | 각 에이전트는 자신의 scope 내 파일만 수정. 공유 파일(base.py 등)은 team-lead 승인 필요 |
| **인터페이스 동결** | Phase 1 완료 후 public API(SolverBase, PixelStack, SimulationResult)는 동결. 변경 시 전체 팀 공지 |
| **브랜치 전략** | 각 에이전트는 `agent/{name}/{task-id}` 브랜치에서 작업. merge는 Agent 4 리뷰 후 team-lead가 수행 |
| **Task claiming** | 자동 self-claim 활성화. 동시 claim 방지를 위해 파일 잠금 사용 |
| **공유 dataclass** | `SimulationResult`, `LayerSlice` 등 공유 구조체 변경 시 affected agents에 반드시 메시지 |

### 11.9 비용/효율 고려사항

```
예상 토큰 사용량 (Phase별):
┌─────────────┬───────────┬──────────────┬─────────────┐
│ Phase       │ 에이전트 수│ 예상 기간     │ 비고         │
├─────────────┼───────────┼──────────────┼─────────────┤
│ Phase 1     │ 2 (1+4)   │ 1 week       │ 순차 의존 多  │
│ Phase 2     │ 4 (전체)  │ 1.5 weeks    │ 최대 병렬화   │
│ Phase 3-4   │ 4 (전체)  │ 2 weeks      │ 전면 병렬     │
│ Phase 5-6   │ 3 (1,2,4) │ 1.5 weeks    │ Map-Reduce   │
│ Phase 7-9   │ 4 (전체)  │ 1 week       │ 문서화 병렬   │
└─────────────┴───────────┴──────────────┴─────────────┘

병렬화 이점:
- 순차 실행 시 예상: 8-10 weeks
- Agent Teams 병렬화: 5-6 weeks (약 40% 단축)
- 가장 큰 이점: Phase 3-4에서 솔버/시각화/테스트 동시 진행
```

### 11.10 Fallback 전략

Agent Teams가 기대만큼 효과적이지 않은 경우의 대안:

1. **Subagent 방식**: Team Lead가 단일 세션에서 subagent를 순차 호출. 병렬성은 떨어지지만 context 공유가 쉬움
2. **Task tool 기반 builder-validator**: builder 에이전트가 구현 → validator 에이전트가 검증하는 2-agent 패턴으로 단순화
3. **단일 세션 + TASKS.md 체크리스트**: 가장 단순. 한 에이전트가 TASKS.md를 순서대로 처리. context 초과 시 새 세션에서 이어서 진행

---

## 12. VitePress 문서화 설계

### 12.1 문서화 철학

COMPASS의 문서화는 **3단계 학습 경로**를 제공한다:

1. **이론 기초 (Theory)**: 광학, 전자기학, 이미지 센서에 대한 배경 지식. 광학 비전공자도 RCWA/FDTD가 무엇을 하는지 이해할 수 있도록 직관적 비유와 시각 자료 활용
2. **프로젝트 가이드 (Guide)**: COMPASS 설치, 첫 시뮬레이션, 결과 해석까지의 hands-on 경로
3. **레퍼런스 (Reference)**: API 문서, config 옵션, 재료 데이터베이스, 솔버 파라미터 상세

### 12.2 사이트 구조

```
docs/
├── .vitepress/
│   ├── config.mts              # VitePress 설정
│   └── theme/
│       ├── index.ts            # 커스텀 테마 확장
│       └── custom.css          # 스타일 커스터마이징
├── public/
│   ├── logo.svg                # COMPASS 로고
│   ├── images/
│   │   ├── theory/             # 이론 설명 다이어그램
│   │   ├── guide/              # 튜토리얼 스크린샷
│   │   └── architecture/       # 시스템 구조도
│   └── favicon.ico
│
├── index.md                    # 랜딩 페이지 (Hero)
│
├── theory/                     # ═══ Part 1: 이론 기초 ═══
│   ├── index.md                # 이론 섹션 개요
│   ├── light-basics.md         # 1.1 빛의 기초
│   ├── electromagnetic-waves.md # 1.2 전자기파와 맥스웰 방정식
│   ├── thin-film-optics.md     # 1.3 박막 광학 (반사, 투과, 간섭)
│   ├── diffraction.md          # 1.4 회절과 주기 구조
│   ├── rcwa-explained.md       # 1.5 RCWA 완전 해설
│   ├── fdtd-explained.md       # 1.6 FDTD 완전 해설
│   ├── rcwa-vs-fdtd.md         # 1.7 RCWA vs FDTD 비교
│   ├── image-sensor-optics.md  # 1.8 이미지 센서 광학 구조
│   ├── quantum-efficiency.md   # 1.9 양자 효율(QE)의 물리
│   └── numerical-stability.md  # 1.10 수치 안정성과 발산 문제
│
├── guide/                      # ═══ Part 2: 사용 가이드 ═══
│   ├── index.md                # 가이드 개요
│   ├── installation.md         # 2.1 설치 가이드
│   ├── quickstart.md           # 2.2 5분 퀵스타트
│   ├── first-simulation.md     # 2.3 첫 번째 시뮬레이션
│   ├── pixel-stack-config.md   # 2.4 픽셀 스택 구성하기
│   ├── material-database.md    # 2.5 재료 데이터베이스 활용
│   ├── choosing-solver.md      # 2.6 솔버 선택 가이드
│   ├── running-rcwa.md         # 2.7 RCWA 시뮬레이션 실행
│   ├── running-fdtd.md         # 2.8 FDTD 시뮬레이션 실행
│   ├── cross-validation.md     # 2.9 솔버 간 교차검증
│   ├── cone-illumination.md    # 2.10 Cone 조명 모델
│   ├── roi-sweep.md            # 2.11 ROI 스윕 분석
│   ├── visualization.md        # 2.12 결과 시각화
│   ├── troubleshooting.md      # 2.13 트러블슈팅
│   └── advanced/
│       ├── stability-tuning.md  # 수치 안정성 튜닝
│       ├── gpu-optimization.md  # GPU 최적화
│       ├── custom-materials.md  # 커스텀 재료 추가
│       └── batch-simulation.md  # 대량 시뮬레이션 자동화
│
├── reference/                  # ═══ Part 3: API 레퍼런스 ═══
│   ├── index.md                # 레퍼런스 개요
│   ├── api/
│   │   ├── pixel-stack.md      # PixelStack API
│   │   ├── geometry-builder.md # GeometryBuilder API
│   │   ├── material-db.md      # MaterialDB API
│   │   ├── solver-base.md      # SolverBase ABC
│   │   ├── solver-torcwa.md    # torcwa adapter API
│   │   ├── solver-grcwa.md     # grcwa adapter API
│   │   ├── solver-meent.md     # meent adapter API
│   │   ├── solver-fdtd.md      # FDTD solver APIs
│   │   ├── sources.md          # Source 모듈 API
│   │   ├── qe-calculator.md    # QECalculator API
│   │   └── visualization.md    # Viz 모듈 API
│   ├── config/
│   │   ├── pixel-config.md     # pixel/ YAML 상세
│   │   ├── solver-config.md    # solver/ YAML 상세
│   │   ├── source-config.md    # source/ YAML 상세
│   │   ├── stability-config.md # stability 설정 상세
│   │   └── compute-config.md   # compute 환경 설정
│   ├── materials/
│   │   ├── built-in.md         # 내장 재료 목록 및 출처
│   │   ├── optical-constants.md # n,k 데이터 형식
│   │   └── dispersion-models.md # Cauchy, Sellmeier, Lorentz-Drude
│   └── cli.md                  # CLI 명령어 레퍼런스
│
├── cookbook/                    # ═══ Part 4: 실전 레시피 ═══
│   ├── index.md                # 레시피 개요
│   ├── bsi-2x2-basic.md        # BSI 2×2 Bayer 기본 시뮬레이션
│   ├── metal-grid-effect.md    # Metal grid 유무에 따른 QE 비교
│   ├── microlens-optimization.md # 마이크로렌즈 형상 최적화
│   ├── cra-shift-analysis.md   # CRA 기반 렌즈 쉬프트 분석
│   ├── barl-design.md          # BARL 반사방지막 설계
│   ├── dti-crosstalk.md        # DTI 깊이별 크로스토크 분석
│   ├── wavelength-sweep.md     # 가시광 전 영역 QE 스펙트럼
│   └── solver-benchmark.md     # 솔버 비교 벤치마크 재현
│
├── contributing/               # ═══ Part 5: 기여 가이드 ═══
│   ├── index.md                # 기여 방법 개요
│   ├── development-setup.md    # 개발 환경 구축
│   ├── adding-solver.md        # 새 솔버 추가 방법
│   ├── adding-material.md      # 새 재료 추가 방법
│   └── code-conventions.md     # 코드 컨벤션
│
└── about/                      # ═══ Part 6: 프로젝트 정보 ═══
    ├── changelog.md            # 변경 이력
    ├── roadmap.md              # 로드맵
    ├── references.md           # 참고 문헌
    └── license.md              # 라이선스
```

### 12.3 Theory 섹션 상세 설계 — 비전공자를 위한 친절한 설명

#### 1.1 빛의 기초 (`theory/light-basics.md`)

**목표**: 물리학 배경 없이도 '빛이 물질과 만나면 무슨 일이 벌어지는지' 이해

**구성:**
- 빛은 무엇인가? — 파동과 입자의 이중성 (일상적 비유: 호수에 던진 돌의 파문)
- 파장과 색의 관계 — 380nm(보라)~780nm(빨강) 가시광선 스펙트럼
- 빛이 물질을 만나면 — 반사(거울), 투과(유리), 흡수(검은 천)
- 굴절과 굴절률(n) — 수영장 바닥이 얕아 보이는 이유
- 흡수와 소광계수(k) — 선글라스가 빛을 줄이는 원리
- 복소 굴절률 ñ = n + ik — 이것 하나로 재료의 광학 성질을 기술
- **인터랙티브 요소**: 파장 슬라이더로 색상 변화 시각화 (Vue 컴포넌트)

#### 1.2 전자기파와 맥스웰 방정식 (`theory/electromagnetic-waves.md`)

**목표**: 수식 없이 '왜 컴퓨터 시뮬레이션이 필요한지' 이해

**구성:**
- 전기장과 자기장 — 자석 주위의 철가루 패턴 비유
- 맥스웰 방정식의 의미 — "빛은 전기장과 자기장이 서로를 만들며 퍼져나가는 것"
- 왜 정확한 풀이가 어려운가 — 단순한 구조(평면)는 손으로 풀 수 있지만, 복잡한 3D 구조는 컴퓨터 필요
- EM 시뮬레이션의 두 가지 접근 — 주파수 영역(RCWA) vs 시간 영역(FDTD)
- **비유**: RCWA는 "정지 사진 촬영" (주파수별 스냅샷), FDTD는 "비디오 촬영" (시간에 따른 파동 전파)

#### 1.3 박막 광학 (`theory/thin-film-optics.md`)

**목표**: 이미지 센서의 박막 스택이 왜 중요한지 이해

**구성:**
- 박막 간섭 — 비눗방울 무지개색의 원리
- 반사 방지 코팅(ARC) — 안경 렌즈의 반사방지막
- BARL(Bottom Anti-Reflective Layers) — 왜 여러 겹을 쌓는가
- Fresnel 방정식 — 경계면에서의 반사/투과 비율 (기본 수식 제공, 직관적 설명 병행)
- Transfer Matrix Method — 다층 박막 계산의 기본 도구
- **시각화**: 파장별 반사율 변화 인터랙티브 플롯

#### 1.4 회절과 주기 구조 (`theory/diffraction.md`)

**목표**: 이미지 센서의 컬러 필터 격자, metal grid가 왜 '회절 격자'인지 이해

**구성:**
- 회절이란 — CD/DVD 표면에서 무지개가 보이는 이유
- 주기 구조와 회절 차수 — 격자가 빛을 여러 방향으로 보내는 것
- Bragg 조건과 Floquet-Bloch 정리 — 주기 구조에서 빛의 행동 규칙
- 왜 이미지 센서에서 중요한가 — pixel pitch가 파장과 비슷한 크기일 때 회절 효과 무시 불가
- **비유**: "창문 블라인드 사이로 빛이 퍼지는 것 = 회절"

#### 1.5 RCWA 완전 해설 (`theory/rcwa-explained.md`)

**목표**: RCWA가 어떤 문제를 어떻게 푸는지, 비유와 단계별 설명

**구성:**
- RCWA의 핵심 아이디어 — "구조를 Fourier 급수로 전개하고, 각 layer에서 고유 모드를 찾는다"
- Step-by-step 설명:
  1. 구조를 z방향으로 slice — 계단 근사 (staircase approximation)
  2. 각 slice의 permittivity를 Fourier 급수로 전개
  3. 각 slice에서 Maxwell 방정식 → eigenvalue problem → 고유 모드 계산
  4. S-matrix로 모든 slice 연결 → 전체 구조의 반사/투과 계산
- 왜 Fourier order가 중요한가 — 해상도 비유 (JPEG 압축률과 유사)
- 수렴성 문제 — order를 높이면 정확하지만 느려지고, 불안정해질 수 있음
- Li의 Fourier factorization rule — 불연속 경계에서의 Gibbs 현상 해결
- **수치 안정성 이슈 (float32 발산)** — 실전 경험에서 나온 교훈 (TRD 4.4 연계)
- **다이어그램**: 전체 RCWA 플로우차트 (Mermaid)

```mermaid
flowchart TD
    A[Pixel Stack 정의] --> B[z방향 staircase slicing]
    B --> C[각 slice의 ε(x,y) Fourier 전개]
    C --> D[Eigenvalue Problem 풀기]
    D --> E{수치 안정성 검사}
    E -->|안정| F[S-matrix 조립 - Redheffer Star Product]
    E -->|불안정| G[Float64 fallback / CPU 전환]
    G --> D
    F --> H[R, T, A 계산]
    H --> I[QE 도출]
    I --> J{에너지 보존 검증}
    J -->|R+T+A≈1| K[결과 출력]
    J -->|위반| L[경고 + 재시뮬레이션]
```

#### 1.6 FDTD 완전 해설 (`theory/fdtd-explained.md`)

**목표**: FDTD의 시간 영역 접근을 직관적으로 이해

**구성:**
- FDTD의 핵심 아이디어 — "시간과 공간을 격자로 나누고, 매 시간 단계마다 전기장/자기장을 업데이트"
- Yee Grid — 전기장과 자기장을 반 격자만큼 어긋나게 배치하는 이유
- CFL 조건 — 시간 간격이 너무 크면 발산 (빛보다 빠르게 정보가 전파되면 안 됨)
- 경계 조건 — PBC (주기), PML (흡수 경계)
- Broadband vs Narrowband — FDTD는 한 번에 넓은 파장 가능, 단 dispersive material은 예외
- **비유**: "수영장에 돌을 던지고 파문이 벽에 부딪히는 과정을 고속 카메라로 찍는 것"
- **애니메이션**: FDTD 시뮬레이션 진행 과정 GIF

#### 1.7 RCWA vs FDTD 비교 (`theory/rcwa-vs-fdtd.md`)

**목표**: 언제 어떤 솔버를 쓰면 좋은지 실용적 판단 기준 제공

**구성:**

| 특성 | RCWA | FDTD |
|---|---|---|
| 계산 영역 | 주파수 영역 | 시간 영역 |
| 적합한 구조 | 주기 구조, 평탄 layer 다수 | 임의 3D, 비주기 가능 |
| 파장 스윕 | 파장별 개별 계산 (빠름) | 광대역 한 번에 가능 |
| 메모리 | Fourier order에 의존 | voxel 수에 의존 |
| 금속 구조 | 가능, Li rule 필요 | 자연스러움, Lorentz-Drude fitting |
| 비주기 구조 | PML 필요 (비효율적) | 자연스러움 |
| 결과 신뢰도 | 주기 구조에서 최고 | 범용적 |

- 교차검증의 의미 — 둘 다 같은 답을 내면 '아마 맞을 것'
- COMPASS가 둘 다 지원하는 이유 — 단일 솔버 결과를 믿기 어려움

#### 1.8 이미지 센서 광학 구조 (`theory/image-sensor-optics.md`)

**목표**: CMOS 이미지 센서의 물리적 구조를 층별로 이해

**구성:**
- BSI (Backside Illumination) 구조 개요
- 층별 역할:
  - **마이크로렌즈** — 빛을 photodiode로 집광 (돋보기 비유)
  - **평탄화층** — 렌즈 아래를 매끈하게
  - **컬러 필터** — RGB 색 분리 (색안경 비유)
  - **Metal Grid** — 인접 픽셀로의 빛 누출 차단 (칸막이 비유)
  - **BARL** — Silicon 표면에서의 반사 최소화 (여러 겹 코팅)
  - **Silicon + Photodiode** — 빛→전자 변환 (태양전지 비유)
  - **DTI** — 픽셀 간 전기적/광학적 격리 (벽 비유)
- CRA (Chief Ray Angle) — 센서 가장자리로 갈수록 빛이 비스듬히 입사
- **인터랙티브 단면도**: 각 층을 클릭하면 상세 설명 팝업

#### 1.9 양자 효율의 물리 (`theory/quantum-efficiency.md`)

**목표**: QE가 정확히 무엇이고, 왜 100%가 될 수 없는지 이해

**구성:**
- QE 정의: 입사 광자 중 몇 %가 전자로 변환되는가
- QE를 낮추는 요인: 반사 손실, 흡수 안 되고 투과, 인접 픽셀 누출 (crosstalk)
- 파장별 QE가 다른 이유: Silicon의 흡수 깊이 (파장이 길수록 깊이 침투)
- RCWA/FDTD에서 QE 계산 방법
- **실제 센서 QE 곡선 예시**: 전형적인 BSI 센서의 R/G/B 채널 QE 스펙트럼

#### 1.10 수치 안정성과 발산 문제 (`theory/numerical-stability.md`)

**목표**: 시뮬레이션이 '틀린 답'을 줄 수 있는 이유와 COMPASS의 대응책

**구성:**
- 부동소수점의 한계 — float32 vs float64 (우유를 mL vs μL로 재는 것)
- RCWA에서의 발산 4대 원인 (TRD 4.4.1 기반, 비전공자 언어로 재설명)
- COMPASS의 5-layer 방어 전략 (TRD 4.4.2 기반, 비전공자 언어로 재설명)
- 에너지 보존 검증 — R + T + A = 1이 안 되면 뭔가 잘못된 것
- "시뮬레이션 결과를 어떻게 믿을 수 있는가" — 교차검증의 실제 workflow

### 12.4 VitePress 설정

```typescript
// docs/.vitepress/config.mts
import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'COMPASS',
  description: 'Cross-solver Optical Modeling Platform for Advanced Sensor Simulation',
  lang: 'ko-KR',

  head: [
    ['link', { rel: 'icon', href: '/favicon.ico' }],
    // KaTeX for math rendering
    ['link', { rel: 'stylesheet', href: 'https://cdn.jsdelivr.net/npm/katex@0.16/dist/katex.min.css' }],
  ],

  // Markdown 확장: 수식, Mermaid 다이어그램
  markdown: {
    math: true,  // KaTeX 수식 지원 ($inline$, $$block$$)
    // mermaid plugin은 별도 설치 필요: vitepress-plugin-mermaid
  },

  themeConfig: {
    logo: '/logo.svg',
    siteTitle: 'COMPASS',

    nav: [
      { text: '이론', link: '/theory/' },
      { text: '가이드', link: '/guide/' },
      { text: '레퍼런스', link: '/reference/' },
      { text: '레시피', link: '/cookbook/' },
      {
        text: '더보기',
        items: [
          { text: '기여 가이드', link: '/contributing/' },
          { text: '변경 이력', link: '/about/changelog' },
          { text: '로드맵', link: '/about/roadmap' },
        ]
      }
    ],

    sidebar: {
      '/theory/': [
        {
          text: '🔬 이론 기초',
          items: [
            { text: '개요', link: '/theory/' },
            { text: '빛의 기초', link: '/theory/light-basics' },
            { text: '전자기파와 맥스웰 방정식', link: '/theory/electromagnetic-waves' },
            { text: '박막 광학', link: '/theory/thin-film-optics' },
            { text: '회절과 주기 구조', link: '/theory/diffraction' },
          ]
        },
        {
          text: '🧮 시뮬레이션 방법론',
          items: [
            { text: 'RCWA 완전 해설', link: '/theory/rcwa-explained' },
            { text: 'FDTD 완전 해설', link: '/theory/fdtd-explained' },
            { text: 'RCWA vs FDTD 비교', link: '/theory/rcwa-vs-fdtd' },
          ]
        },
        {
          text: '📷 이미지 센서',
          items: [
            { text: '센서 광학 구조', link: '/theory/image-sensor-optics' },
            { text: '양자 효율(QE)의 물리', link: '/theory/quantum-efficiency' },
            { text: '수치 안정성과 발산 문제', link: '/theory/numerical-stability' },
          ]
        }
      ],
      '/guide/': [
        {
          text: '🚀 시작하기',
          items: [
            { text: '개요', link: '/guide/' },
            { text: '설치', link: '/guide/installation' },
            { text: '5분 퀵스타트', link: '/guide/quickstart' },
            { text: '첫 번째 시뮬레이션', link: '/guide/first-simulation' },
          ]
        },
        {
          text: '⚙️ 시뮬레이션 설정',
          items: [
            { text: '픽셀 스택 구성', link: '/guide/pixel-stack-config' },
            { text: '재료 데이터베이스', link: '/guide/material-database' },
            { text: '솔버 선택 가이드', link: '/guide/choosing-solver' },
          ]
        },
        {
          text: '▶️ 실행 & 분석',
          items: [
            { text: 'RCWA 시뮬레이션', link: '/guide/running-rcwa' },
            { text: 'FDTD 시뮬레이션', link: '/guide/running-fdtd' },
            { text: '솔버 교차검증', link: '/guide/cross-validation' },
            { text: 'Cone 조명 모델', link: '/guide/cone-illumination' },
            { text: 'ROI 스윕 분석', link: '/guide/roi-sweep' },
            { text: '결과 시각화', link: '/guide/visualization' },
          ]
        },
        {
          text: '🔧 고급',
          collapsed: true,
          items: [
            { text: '수치 안정성 튜닝', link: '/guide/advanced/stability-tuning' },
            { text: 'GPU 최적화', link: '/guide/advanced/gpu-optimization' },
            { text: '커스텀 재료', link: '/guide/advanced/custom-materials' },
            { text: '대량 시뮬레이션', link: '/guide/advanced/batch-simulation' },
          ]
        },
        {
          text: '❓ 문제 해결',
          items: [
            { text: '트러블슈팅', link: '/guide/troubleshooting' },
          ]
        }
      ],
      '/reference/': [
        {
          text: '📚 API',
          items: [
            { text: '개요', link: '/reference/' },
            { text: 'PixelStack', link: '/reference/api/pixel-stack' },
            { text: 'GeometryBuilder', link: '/reference/api/geometry-builder' },
            { text: 'MaterialDB', link: '/reference/api/material-db' },
            { text: 'SolverBase', link: '/reference/api/solver-base' },
            { text: 'torcwa Adapter', link: '/reference/api/solver-torcwa' },
            { text: 'grcwa Adapter', link: '/reference/api/solver-grcwa' },
            { text: 'meent Adapter', link: '/reference/api/solver-meent' },
            { text: 'FDTD Solvers', link: '/reference/api/solver-fdtd' },
            { text: 'Sources', link: '/reference/api/sources' },
            { text: 'QECalculator', link: '/reference/api/qe-calculator' },
            { text: 'Visualization', link: '/reference/api/visualization' },
          ]
        },
        {
          text: '📝 Config',
          items: [
            { text: 'Pixel Config', link: '/reference/config/pixel-config' },
            { text: 'Solver Config', link: '/reference/config/solver-config' },
            { text: 'Source Config', link: '/reference/config/source-config' },
            { text: 'Stability Config', link: '/reference/config/stability-config' },
            { text: 'Compute Config', link: '/reference/config/compute-config' },
          ]
        },
        {
          text: '🧪 Materials',
          items: [
            { text: '내장 재료 목록', link: '/reference/materials/built-in' },
            { text: '광학 상수 형식', link: '/reference/materials/optical-constants' },
            { text: '분산 모델', link: '/reference/materials/dispersion-models' },
          ]
        },
        {
          text: '💻 CLI',
          items: [
            { text: '명령어 레퍼런스', link: '/reference/cli' },
          ]
        }
      ],
      '/cookbook/': [
        {
          text: '🍳 실전 레시피',
          items: [
            { text: '개요', link: '/cookbook/' },
            { text: 'BSI 2×2 기본', link: '/cookbook/bsi-2x2-basic' },
            { text: 'Metal Grid 효과', link: '/cookbook/metal-grid-effect' },
            { text: '마이크로렌즈 최적화', link: '/cookbook/microlens-optimization' },
            { text: 'CRA 쉬프트 분석', link: '/cookbook/cra-shift-analysis' },
            { text: 'BARL 설계', link: '/cookbook/barl-design' },
            { text: 'DTI 크로스토크', link: '/cookbook/dti-crosstalk' },
            { text: '파장 스윕', link: '/cookbook/wavelength-sweep' },
            { text: '솔버 벤치마크', link: '/cookbook/solver-benchmark' },
          ]
        }
      ],
      '/contributing/': [
        {
          text: '🤝 기여하기',
          items: [
            { text: '개요', link: '/contributing/' },
            { text: '개발 환경 구축', link: '/contributing/development-setup' },
            { text: '새 솔버 추가', link: '/contributing/adding-solver' },
            { text: '새 재료 추가', link: '/contributing/adding-material' },
            { text: '코드 컨벤션', link: '/contributing/code-conventions' },
          ]
        }
      ],
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/user/compass' },
    ],

    search: {
      provider: 'local',  // 내장 검색 (flexsearch 기반)
    },

    editLink: {
      pattern: 'https://github.com/user/compass/edit/main/docs/:path',
      text: '이 페이지 수정 제안',
    },

    footer: {
      message: 'COMPASS — Cross-solver Optical Modeling Platform for Advanced Sensor Simulation',
      copyright: 'MIT License',
    },

    outline: {
      level: [2, 3],
      label: '목차',
    },
  },
})
```

### 12.5 커스텀 Vue 컴포넌트 (인터랙티브 교육 요소)

Theory 섹션의 이해도를 높이기 위한 인터랙티브 컴포넌트:

```
docs/.vitepress/theme/components/
├── WavelengthSlider.vue       # 파장↔색상 인터랙티브 변환
├── FresnelCalculator.vue      # n1, n2 입력 → R, T 실시간 계산
├── StackVisualizer.vue        # 픽셀 스택 단면도 인터랙티브 뷰어
├── FourierOrderDemo.vue       # Fourier order 변화 → 구조 근사 시각화
├── SMatrixFlowchart.vue       # RCWA 계산 과정 단계별 애니메이션
└── EnergyBalanceChecker.vue   # R+T+A=1 검증 시각화
```

**예시: WavelengthSlider.vue** — Theory 1.1에서 사용

```vue
<template>
  <div class="wavelength-demo">
    <input type="range" v-model="wavelength" min="380" max="780" step="1" />
    <div class="color-display" :style="{ backgroundColor: wavelengthToRGB(wavelength) }">
      <span>{{ wavelength }} nm</span>
    </div>
    <p>이 파장의 빛은 <strong>{{ colorName }}</strong>으로 보입니다.</p>
    <p>Silicon에서의 흡수 깊이: <strong>{{ absorptionDepth }} μm</strong></p>
  </div>
</template>
```

**예시: FresnelCalculator.vue** — Theory 1.3에서 사용

```vue
<!-- 사용자가 n1(공기), n2(유리/Si 등) 입력 → 실시간 R, T 계산 -->
<!-- 입사각 슬라이더 → Brewster 각, 전반사 각 시각화 -->
```

### 12.6 수식 렌더링 규칙

KaTeX를 활용한 수식 표현 가이드:

- **인라인 수식**: `$n = c/v$` → 본문 중 간단한 수식
- **블록 수식**: Theory 섹션에서 주요 방정식

```markdown
Maxwell 방정식:

$$
\nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}
$$

Fresnel 반사 계수 (수직 입사):

$$
r = \frac{n_1 - n_2}{n_1 + n_2}, \quad R = |r|^2
$$

RCWA eigenvalue problem:

$$
\Omega^2 \mathbf{W} = \mathbf{W} \boldsymbol{\Lambda}^2
$$
```

- **모든 수식 직후에 직관적 설명 추가**:
  > "위 식은 '두 재료의 굴절률 차이가 클수록 반사가 많다'는 것을 의미합니다. 유리(n≈1.5)에 수직으로 빛을 쏘면 약 4%가 반사됩니다."

### 12.7 문서 작성 원칙

1. **비유 우선**: 모든 물리 개념에 일상적 비유를 먼저 제시, 수식은 그 다음
2. **점진적 깊이**: 각 페이지 내에서 기초→중급→고급 순서로 구성. 고급 내용은 collapsible 섹션
3. **코드 예제 동반**: 이론 설명 직후에 COMPASS에서의 구현 예시 코드 제시
4. **시각 자료 풍부**: 모든 주요 개념에 다이어그램, 플롯, 인터랙티브 컴포넌트
5. **한국어 기본, 영문 용어 병기**: "양자 효율(Quantum Efficiency, QE)" 형태
6. **상호 참조**: Theory ↔ Guide ↔ Reference 간 자연스러운 링크
7. **"왜?"에 답하기**: 단순히 '무엇'만이 아니라, '왜 이렇게 하는지' 항상 설명

### 12.8 빌드 및 배포

```bash
# 개발 서버
cd docs && npx vitepress dev

# 프로덕션 빌드
npx vitepress build
# 출력: docs/.vitepress/dist/

# 배포 옵션
# 1. GitHub Pages (권장)
# 2. Netlify / Vercel
# 3. Self-hosted (Oracle Cloud Free Tier)
```

### 12.9 package.json (docs)

```json
{
  "name": "compass-docs",
  "private": true,
  "scripts": {
    "docs:dev": "vitepress dev",
    "docs:build": "vitepress build",
    "docs:preview": "vitepress preview"
  },
  "devDependencies": {
    "vitepress": "^1.6.0",
    "vue": "^3.5.0",
    "vitepress-plugin-mermaid": "^2.0.0",
    "markdown-it-katex": "^2.0.3"
  }
}
```
