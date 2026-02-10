# Tasks: COMPASS — Development Task Breakdown

## Phase 0: Agent Teams 환경 구축 (Day 1)

### Task 0.1: CLAUDE.md 및 프로젝트 컨벤션 작성
**Priority: P0 | Estimated: 2 hours**
**Dependencies: None**

**Description:**
모든 에이전트가 참조할 CLAUDE.md 작성. 프로젝트 개요, 아키텍처, 코드 컨벤션, 물리적 제약조건, 파일 소유권 규칙 포함.

**Acceptance Criteria:**
- [ ] CLAUDE.md 완성 (TRD Section 11.5 기반)
- [ ] 디렉토리 구조 생성 (`compass/config`, `compass/core`, `compass/geometry`, `compass/materials`, `compass/solvers/rcwa`, `compass/solvers/fdtd`, `compass/sources`, `compass/analysis`, `compass/viz`, `compass/io`, `compass/diagnostics`, `tests/`, `notebooks/`, `benchmarks/`)
- [ ] pyproject.toml 초기 설정 (optional dependency groups)
- [ ] `.gitignore`, `ruff.toml`, `mypy.ini` 기본 설정

---

### Task 0.2: Agent Team 스폰 및 태스크 등록
**Priority: P0 | Estimated: 1 hour**
**Dependencies: 0.1**

**Description:**
Agent Teams 활성화 후 4개 에이전트(core-engine, solvers-physics, viz-io, qa-benchmark) 스폰. TASKS.md의 모든 태스크를 shared task list에 등록하고 blockedBy 관계 설정.

**Acceptance Criteria:**
- [ ] `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` 환경 변수 설정
- [ ] 4개 에이전트 정상 스폰 확인 (tmux/iTerm2 pane 분할)
- [ ] 전체 태스크 DAG 등록 완료 (TRD Section 11.6 참조)
- [ ] 에이전트 간 메시징 테스트 (ping-pong)
- [ ] self-claim 활성화 확인

---

### Task 0.3: Shared Interface Contracts 정의
**Priority: P0 | Estimated: 2 hours**
**Dependencies: 0.1**

**Description:**
에이전트 간 공유되는 핵심 인터페이스를 먼저 정의하여 병렬 개발 시 충돌 방지. 구현체는 비워두고 type signature + docstring만 작성.

**Acceptance Criteria:**
- [ ] `compass/core/types.py`: `SimulationResult`, `LayerSlice`, `FieldData` dataclass stubs
- [ ] `compass/solvers/base.py`: `SolverBase` ABC (method signatures + docstrings)
- [ ] `compass/geometry/pixel_stack.py`: `PixelStack` public API stubs
- [ ] `compass/materials/database.py`: `MaterialDB` public API stubs
- [ ] mypy type check 통과 (stub level)

**Key Files:**
- `compass/core/types.py`
- `compass/solvers/base.py`
- `compass/geometry/pixel_stack.py`
- `compass/materials/database.py`

---

## Phase 1: Foundation (Week 1-2)

### Task 1.1: Project Scaffolding & Configuration System
**Priority: P0 | Estimated: 2 days**
**Dependencies: None**

**Description:**
프로젝트 디렉토리 구조 생성, pyproject.toml 설정, Hydra configuration 시스템 구축.

**Acceptance Criteria:**
- [ ] 디렉토리 구조가 TRD의 아키텍처와 일치
- [ ] `pyproject.toml` with optional dependency groups: `[rcwa]`, `[fdtd]`, `[viz]`, `[all]`
- [ ] Hydra root config 동작: `python scripts/run_simulation.py` 실행 시 default config 로드
- [ ] Config composition 동작: `python scripts/run_simulation.py solver=torcwa pixel=default_bsi_1um`
- [ ] Local override 지원: `configs/local/` 디렉토리 `.gitignore` 처리
- [ ] Config 스키마 validation: Pydantic 또는 OmegaConf structured config로 타입 체크

**Key Files:**
- `configs/config.yaml`, `configs/pixel/default_bsi_1um.yaml`, `configs/solver/*.yaml`
- `configs/source/planewave.yaml`, `configs/compute/cuda.yaml`
- `compass/core/config_schema.py`
- `pyproject.toml`, `README.md`

---

### Task 1.2: Material Database
**Priority: P0 | Estimated: 1.5 days**
**Dependencies: 1.1**

**Description:**
재료 물성 데이터베이스 구현. CSV 기반 tabulated data, 상수, 분석 모델(Cauchy, Sellmeier) 지원.

**Acceptance Criteria:**
- [ ] `MaterialDB` 클래스: 이름으로 재료 조회, 파장별 complex epsilon 반환
- [ ] Built-in materials: Silicon (Palik + Green 2008), SiO₂, Si₃N₄, TiO₂, Tungsten, Air
- [ ] Color filter materials: R, G, B (합리적인 n, k 스펙트럼 - 문헌 또는 추정치)
- [ ] Polymer microlens material (constant n ≈ 1.56)
- [ ] 보간 방식: cubic spline (default), linear
- [ ] 사용자 정의 CSV 로드 기능
- [ ] Cauchy 모델: `n(λ) = A + B/λ² + C/λ⁴` 계산
- [ ] Unit test: Si의 550nm에서 n ≈ 4.08, k ≈ 0.028 (Green 2008 기준) 검증

**Key Files:**
- `compass/core/material_db.py`
- `materials/*.csv`
- `tests/unit/test_material_db.py`

**Data Sources:**
- Silicon: https://refractiveindex.info (Green 2008 dataset)
- SiO₂, Si₃N₄: Palik handbook
- Color Filters: STMicroelectronics / 논문 기반 추정 또는 Generic Lorentz model

---

### Task 1.3: Geometry Builder & PixelStack
**Priority: P0 | Estimated: 3 days**
**Dependencies: 1.1, 1.2**

**Description:**
YAML 파라미터로부터 solver-agnostic한 픽셀 스택 구조를 생성하는 모듈. Microlens(superellipse), Color Filter(Bayer), DTI, Photodiode 영역을 모두 포함.

**Acceptance Criteria:**
- [ ] `PixelStack` 클래스: config dict → 전체 stack 구조 생성
- [ ] 각 layer의 z 위치 자동 계산 (아래에서 위로 누적)
- [ ] `GeometryBuilder.superellipse_lens()`: 파라미터 → 2D height map
- [ ] `GeometryBuilder.bayer_pattern()`: unit cell size + pattern type → color map
- [ ] `GeometryBuilder.dti_grid()`: pitch, width → binary mask
- [ ] `PixelStack.get_permittivity_grid(wavelength, nx, ny, nz)`: 3D epsilon array 생성
- [ ] `PixelStack.get_layer_slices(wavelength, nx, ny)`: RCWA용 layer 분해
- [ ] Microlens staircase approximation: n_slices configurable (default: 30)
- [ ] CRA-based microlens shift: `shift = tan(CRA) × height_above_PD`
- [ ] Unit test: 2×2 Bayer pattern 정확성, 렌즈 높이맵 대칭성

**Key Files:**
- `compass/core/pixel_stack.py`
- `compass/core/geometry_builder.py`
- `compass/core/units.py`
- `tests/unit/test_pixel_stack.py`, `tests/unit/test_geometry_builder.py`

**주의사항:**
- 렌즈 staircase 근사 시 lens/air 경계의 filling fraction이 정확해야 함
- DTI는 Si layer 내부에만 존재, CF grid는 CF layer에만 존재 → z 범위 체크 필수
- Photodiode 위치/크기는 Si 내 relative 좌표로 정의

---

### Task 1.4: Unit System & Coordinate Convention
**Priority: P0 | Estimated: 0.5 days**
**Dependencies: 1.1**

**Description:**
프로젝트 전체에서 사용하는 단위계 및 좌표 규약 정의.

**Acceptance Criteria:**
- [ ] 내부 단위: um (micrometers) for length, eV or um for wavelength
- [ ] 좌표 규약: x,y = lateral, z = stack direction (빛 진행 방향 = -z, air=top=z_max, Si=bottom=z_min)
- [ ] 단위 변환 유틸리티: `um_to_nm()`, `wavelength_to_frequency()`, `eV_to_um()` 등
- [ ] 각도 규약: theta = polar (z축으로부터), phi = azimuthal (x축으로부터), 단위 = degrees (내부 변환 radian)
- [ ] 문서화: `compass/core/units.py` 에 규약 docstring

---

## Phase 2: First Solver Integration (Week 2-3)

### Task 2.1: SolverBase Abstract Class
**Priority: P0 | Estimated: 1 day**
**Dependencies: 1.3**

**Description:**
모든 솔버가 구현해야 하는 ABC 정의. SimulationResult dataclass 포함.

**Acceptance Criteria:**
- [ ] `SolverBase` ABC with methods: `setup_geometry()`, `setup_source()`, `run()`, `get_field_distribution()`
- [ ] `SimulationResult` dataclass: qe_per_pixel, wavelengths, fields, reflection, transmission, absorption, metadata
- [ ] `validate_energy_balance()` method
- [ ] Solver registry: `SolverFactory.create(name, config, device)` → solver instance

**Key Files:**
- `compass/solvers/base.py`

---

### Task 2.2: torcwa Solver Adapter (Primary RCWA)
**Priority: P0 | Estimated: 3 days**
**Dependencies: 2.1, 1.3**

**Description:**
torcwa를 wrapping하여 SolverBase 인터페이스에 맞추는 adapter 구현. 이것이 첫 번째 동작하는 솔버.

**Acceptance Criteria:**
- [ ] `TorcwaSolver(SolverBase)` 구현
- [ ] `setup_geometry()`: PixelStack → torcwa layer 구조 변환
  - Microlens → staircase slices → `sim.add_layer(thickness, eps)` 반복
  - Color Filter → patterned eps layer
  - DTI → Si layer 내 patterned region
- [ ] `setup_source()`: wavelength, angle → `sim.set_incident_angle(inc_ang, azi_ang)`
- [ ] `run()`: wavelength sweep → per-pixel QE 계산
  - QE = Si 내 PD 영역의 absorbed power / incident power
  - Absorbed power: Poynting flux 차이 (z_top_PD - z_bottom_PD) 또는 volume absorption
- [ ] `get_field_distribution()`: torcwa의 field reconstruction 사용
- [ ] Energy balance 검증 통과 (< 1% 오차)
- [ ] Benchmark: 단일 Si slab → Fresnel 이론값 대비 < 0.5% 오차

**Key Files:**
- `compass/solvers/rcwa/torcwa_solver.py`
- `tests/integration/test_torcwa_solver.py`

**구현 가이드:**
```python
# torcwa 사용 패턴
import torcwa
sim = torcwa.rcwa(freq=1/wavelength, order=order, L=L, dtype=dtype, device=device)
sim.add_input_layer(eps=1.0)  # air
# ... add layers ...
sim.set_incident_angle(inc_ang=theta_rad, azi_ang=phi_rad)
sim.solve_global_smatrix()
# R, T 추출 → A = 1 - R - T
```

---

### Task 2.3: grcwa Solver Adapter
**Priority: P1 | Estimated: 2 days**
**Dependencies: 2.1, 1.3**

**Description:**
기존 grcwa를 wrapping하여 SolverBase에 맞추는 adapter. torcwa와의 cross-validation 대상.

**Acceptance Criteria:**
- [ ] `GrcwaSolver(SolverBase)` 구현
- [ ] grcwa의 단위계/규약을 내부 단위계로 변환
- [ ] torcwa와 동일 구조에서 QE 비교: ΔQE < 1% (동일 Fourier order)
- [ ] 에너지 보존 검증

**Key Files:**
- `compass/solvers/rcwa/grcwa_solver.py`

---

### Task 2.4: meent Solver Adapter
**Priority: P1 | Estimated: 2 days**
**Dependencies: 2.1, 1.3**

**Description:**
meent (multi-backend RCWA)를 SolverBase에 맞추는 adapter. NumPy/JAX/PyTorch backend 선택 가능.

**Acceptance Criteria:**
- [ ] `MeentSolver(SolverBase)` 구현
- [ ] Backend 선택: config에서 `meent_backend: "numpy" | "jax" | "torch"` 지정
- [ ] meent의 vector modeling 활용 (가능한 경우)
- [ ] torcwa 대비 QE 비교: ΔQE < 2%

**Key Files:**
- `compass/solvers/rcwa/meent_solver.py`

---

### Task 2.5: Planewave Source Module
**Priority: P0 | Estimated: 1 day**
**Dependencies: 2.1**

**Description:**
Planewave 광원 정의 모듈. Single wavelength, sweep, list mode 지원.

**Acceptance Criteria:**
- [ ] `PlanewaveSource` 클래스: wavelength(s), angle(theta, phi), polarization
- [ ] Sweep mode: `(start, stop, step)` → wavelength array 생성
- [ ] Unpolarized: TE + TM 평균 자동 처리
- [ ] 솔버에 전달할 source parameters dict 생성

**Key Files:**
- `compass/sources/planewave.py`

---

### Task 2.6: RCWA Numerical Stability Module
**Priority: P0 | Estimated: 3 days**
**Dependencies: 2.1, 2.2**

**Description:**
RCWA 발산 방지를 위한 수치 안정성 모듈. TRD Section 4.4의 5-layer defense 구현. grcwa를 PyTorch로 포팅할 때 경험한 발산 문제의 근본 원인(eigendecomp 정밀도, T-matrix overflow, Gibbs 현상, float32 한계)에 대한 체계적 대책.

**Acceptance Criteria:**
- [ ] `PrecisionManager`: TF32 비활성화 강제, mixed precision eigendecomp (eigendecomp만 float64)
- [ ] `StableSMatrixAlgorithm`: Redheffer star product 기반 S-matrix recursion 구현 (T-matrix 사용 금지)
- [ ] `LiFactorization`: Li inverse rule 구현 (1D/2D), 고대비 permittivity 경계에서 수렴성 개선
- [ ] `EigenvalueStabilizer`: 중복 eigenvalue 감지/처리, propagation direction 올바른 branch 선택, 에너지 보존 검증
- [ ] `AdaptivePrecisionRunner`: float32 → float64 → CPU float64 3단계 자동 fallback
- [ ] `StabilityDiagnostics`: 시뮬레이션 전 위험 요소 사전 진단, 후 결과 검증 (QE 범위, NaN/Inf, 에너지 보존)
- [ ] Config 연동: `solver.stability` 섹션에서 모든 안정성 파라미터 설정 가능

**테스트 시나리오:**
- [ ] Metal grid (W) + Air 고대비 구조: Li rule 적용 전후 수렴성 비교
- [ ] Fourier order [5,5]→[15,15]→[25,25] sweep: float32 vs mixed vs float64 발산 여부
- [ ] 4um Si layer: S-matrix vs T-matrix 비교 (T-matrix가 발산하는 것 확인)
- [ ] 50-slice microlens staircase: 누적 오차 모니터링
- [ ] Adaptive fallback: 의도적으로 불안정 조건 생성 → 자동 복구 확인

**Key Files:**
- `compass/solvers/rcwa/stability.py`
- `compass/solvers/rcwa/li_factorization.py`
- `compass/solvers/rcwa/smatrix.py`
- `tests/unit/test_stability.py`

**참고문헌:**
- Moharam et al., "Stable implementation of RCWA: Enhanced transmittance matrix approach," JOSA A 12(5), 1995
- Li, "Use of Fourier series in the analysis of discontinuous periodic structures," JOSA A 13(9), 1996
- Schuster et al., "Normal vector method for convergence improvement using RCWA," JOSA A 24(9), 2007
- torcwa: eigenvalue broadening parameter 구현 (Kim & Lee, CPC 282, 2023)

---

## Phase 3: Visualization (Week 3-4)

### Task 3.1: 2D Structure Cross-section Plot
**Priority: P0 | Estimated: 2 days**
**Dependencies: 1.3**

**Description:**
PixelStack의 XZ, YZ, XY 단면도 시각화. 구조가 올바르게 생성되었는지 확인하는 핵심 도구.

**Acceptance Criteria:**
- [ ] `plot_pixel_cross_section(pixel_stack, plane='xz', y_position=0.0)`
- [ ] 재료별 색상 매핑 (Si=회색, SiO₂=연파랑, CF_R=빨강, CF_G=초록, CF_B=파랑, W=노랑, polymer=연보라, Air=흰색)
- [ ] 경계면 실선 표시
- [ ] 레이어 이름 annotation (좌측 or 우측)
- [ ] XY 단면: 특정 z 위치의 permittivity real part 표시
- [ ] XZ/YZ 단면: 전체 스택 구조 표시
- [ ] 그리드 간격 표시, 축 라벨 (um)
- [ ] Microlens 곡면이 staircase 없이 smooth하게 표시되어야 함 (시각화에서는 원래 수식 사용)

**Key Files:**
- `compass/visualization/structure_plot_2d.py`

---

### Task 3.2: 2D Field Distribution Plot
**Priority: P1 | Estimated: 1.5 days**
**Dependencies: 2.2**

**Description:**
시뮬레이션 결과의 EM 필드를 2D 맵으로 표시.

**Acceptance Criteria:**
- [ ] `plot_field_2d(result, component='|E|2', plane='xz', position=0.0)`
- [ ] Components: Ex, Ey, Ez, |E|², Sz (Poynting z-component)
- [ ] 구조 윤곽선 overlay 옵션
- [ ] Colorbar with 물리적 단위
- [ ] Multi-wavelength subplot 지원
- [ ] Log scale 옵션 (흡수 영역에서 유용)

**Key Files:**
- `compass/visualization/field_plot_2d.py`

---

### Task 3.3: 3D Interactive Viewer
**Priority: P1 | Estimated: 2.5 days**
**Dependencies: 1.3**

**Description:**
PixelStack의 3D interactive visualization. 구조 검증의 최종 확인용.

**Acceptance Criteria:**
- [ ] pyvista backend: desktop/Jupyter 3D viewer
- [ ] plotly backend: lightweight web-friendly 3D viewer
- [ ] 레이어별 on/off toggle
- [ ] 재료별 색상 + 투명도 설정
- [ ] Microlens 곡면 smooth 렌더링 (isosurface 또는 parametric surface)
- [ ] DTI 구조 표시 (반투명)
- [ ] Photodiode 영역 하이라이트 (다른 색상)
- [ ] 마우스 회전/줌/팬
- [ ] 카메라 프리셋: top-down, side-view, isometric

**Key Files:**
- `compass/visualization/viewer_3d.py`

---

### Task 3.4: QE Spectral Response Plot
**Priority: P0 | Estimated: 1 day**
**Dependencies: 2.2**

**Description:**
QE(λ) 스펙트럼 곡선 및 솔버 간 비교 플롯.

**Acceptance Criteria:**
- [ ] `plot_qe_spectrum(result)`: R, G, B 채널별 QE(λ) 곡선 (한 그래프에)
- [ ] `plot_qe_comparison(results: List[SimulationResult], labels: List[str])`: 복수 솔버 비교
- [ ] 크로스토크 히트맵: pixel × pixel × wavelength
- [ ] Angular response: QE(θ) at fixed λ
- [ ] 에러바 또는 차이 subplot 옵션

**Key Files:**
- `compass/visualization/qe_plot.py`

---

## Phase 4: FDTD Integration (Week 4-5)

### Task 4.1: flaport/fdtd Solver Adapter
**Priority: P1 | Estimated: 3 days**
**Dependencies: 2.1, 1.3**

**Description:**
flaport/fdtd (Python-native 3D FDTD with PyTorch backend)를 SolverBase에 통합.

**Acceptance Criteria:**
- [ ] `FlaportFdtdSolver(SolverBase)` 구현
- [ ] PixelStack → fdtd.Grid + Object 배치 변환
- [ ] PlaneSource 설정 (wavelength, angle)
- [ ] PBC (lateral) + PML (top/bottom) 경계
- [ ] DFT monitor로 스펙트럼 추출
- [ ] QE 계산: PD 영역의 absorbed power 적분
- [ ] 단일 Si slab Fresnel 벤치마크
- [ ] RCWA 대비 QE 비교: < 5%

**Key Files:**
- `compass/solvers/fdtd/flaport_solver.py`

**주의사항:**
- FDTD grid spacing: λ_min / 20 이상 (380nm → ~19nm → grid_spacing ≤ 0.019 um)
- Si의 높은 k 값에서 충분한 time step 필요 (steady state 도달 확인)
- dispersive Si 처리: 단일 파장별 run 또는 broadband source + Lorentz fitting

---

### Task 4.2: fdtdz Solver Adapter (Optional/Advanced)
**Priority: P2 | Estimated: 3 days**
**Dependencies: 2.1, 1.3**

**Description:**
fdtdz (JAX-based extreme GPU FDTD)를 SolverBase에 통합. 성능 비교용.

**Acceptance Criteria:**
- [ ] `FdtdzSolver(SolverBase)` 구현
- [ ] JAX GPU 환경 설정 (CUDA 필수)
- [ ] Permittivity array 변환
- [ ] 제한사항 문서화: PML은 z 방향만, dispersive material 미지원

---

### Task 4.3: meep Solver Adapter (Optional/Advanced)
**Priority: P2 | Estimated: 3 days**
**Dependencies: 2.1, 1.3**

**Description:**
Meep (가장 성숙한 오픈소스 FDTD)를 SolverBase에 통합.

**Acceptance Criteria:**
- [ ] `MeepSolver(SolverBase)` 구현
- [ ] Meep geometry primitives로 변환
- [ ] Lorentz-Drude material model을 활용한 dispersive Si 시뮬레이션
- [ ] Bloch periodic boundary condition 지원
- [ ] RCWA 대비 QE 비교

---

## Phase 5: Advanced Source & Sweep (Week 5-6)

### Task 5.1: Cone Illumination (Exit Pupil Model)
**Priority: P1 | Estimated: 3 days**
**Dependencies: 2.2**

**Description:**
렌즈 exit pupil에서의 cone illumination 모사. 실제 카메라 환경에서의 QE 예측에 필수.

**Acceptance Criteria:**
- [ ] `ConeIllumination` 클래스: CRA, F-number → angular sampling grid
- [ ] Sampling methods: uniform grid, Fibonacci spiral, Gaussian quadrature
- [ ] Weighting functions: uniform, cosine, cos⁴θ, Gaussian, custom callable
- [ ] 각 sampling point에 대한 planewave 시뮬레이션 실행 → weighted 합산
- [ ] F-number → half-cone angle 변환: `α = arcsin(1/(2*F#))`
- [ ] 수렴 테스트: sampling points 수 vs QE 변화
- [ ] 벤치마크: CRA=0, F/2.0 → on-axis QE vs single planewave QE 비교

**Key Files:**
- `compass/sources/cone_illumination.py`
- `configs/source/cone_illumination.yaml`

**참고 논문/리소스:**
- Vaillant et al., "Uniform illumination and rigorous EM simulations applied to CMOS image sensors," Opt. Express (2007)
- Ansys Lumerical CMOS Sensor Camera workflow: CRA/MRA extraction from Zemax
- STMicroelectronics IQE Tool methodology

---

### Task 5.2: Zemax Ray File Reader
**Priority: P2 | Estimated: 1.5 days**
**Dependencies: 5.1**

**Description:**
Zemax OpticStudio에서 export한 JSON ray file 읽기. CRA, MRA, intensity vs image height.

**Acceptance Criteria:**
- [ ] `RayFileReader`: JSON format parsing (Ansys/Zemax 호환)
- [ ] CSV format parsing (image_height, cra_deg, mra_deg, f_number)
- [ ] CRA → microlens shift 자동 계산
- [ ] Image height → ROI position 매핑

**Key Files:**
- `compass/sources/ray_file_reader.py`

---

### Task 5.3: ROI Sweep Runner
**Priority: P1 | Estimated: 2 days**
**Dependencies: 5.1, 5.2**

**Description:**
센서 전체 이미지 면에서의 QE 분포 예측. 각 ROI 위치별 unit cell 시뮬레이션 수행.

**Acceptance Criteria:**
- [ ] `ROISweepRunner`: CRA vs image_height curve + microlens shift map 입력
- [ ] 각 ROI에서:
  - CRA 결정 → cone illumination 설정
  - Microlens shift 적용 → PixelStack 수정
  - 시뮬레이션 실행
  - QE 수집
- [ ] 출력: QE(R,G,B) vs image_height 2D map
- [ ] Relative Illumination (RI) 계산
- [ ] 병렬 실행: ROI별 independent → multiprocessing / job queue

**Key Files:**
- `compass/runners/roi_sweep_runner.py`
- `configs/experiment/roi_sweep.yaml`

---

### Task 5.4: Wavelength Sweep Runner
**Priority: P0 | Estimated: 1 day**
**Dependencies: 2.2**

**Description:**
파장 sweep 자동화. RCWA(파장별 독립) 및 FDTD(broadband 또는 파장별) 모두 지원.

**Acceptance Criteria:**
- [ ] `SweepRunner`: wavelength array → 솔버별 최적 실행 전략 선택
- [ ] RCWA: 파장별 독립 실행, GPU batch 가능한 경우 batch
- [ ] FDTD: broadband source (지원 시) 또는 파장별 순차 실행
- [ ] 진행 표시: tqdm progress bar
- [ ] 결과 누적 → SimulationResult aggregation

**Key Files:**
- `compass/runners/sweep_runner.py`

---

## Phase 6: Analysis & Comparison (Week 6-7)

### Task 6.1: QE Calculator
**Priority: P0 | Estimated: 1.5 days**
**Dependencies: 2.2**

**Description:**
Solver 결과로부터 정확한 QE를 계산하는 공통 모듈.

**Acceptance Criteria:**
- [ ] RCWA 방식: layer absorption → PD 영역만 추출
- [ ] FDTD 방식: Poynting flux (top/bottom of PD) 또는 volume absorption integral
- [ ] Per-pixel QE: Bayer map에 따라 각 픽셀의 PD 위치/크기로 분리
- [ ] 크로스토크 정의: 인접 CF와 다른 색의 PD에 도달한 에너지 비율
- [ ] QE normalization: incident power 기준 (planewave or cone integrated)

**Key Files:**
- `compass/analysis/qe_calculator.py`

---

### Task 6.2: Solver Comparison Module
**Priority: P1 | Estimated: 2 days**
**Dependencies: 6.1**

**Description:**
복수 솔버의 결과를 체계적으로 비교하는 분석 모듈.

**Acceptance Criteria:**
- [ ] `SolverComparison` 클래스: List[SimulationResult] → 비교 메트릭
- [ ] 메트릭: ΔQE (absolute), relative error (%), field correlation, energy balance error
- [ ] Convergence 비교: RCWA Fourier order vs QE, FDTD resolution vs QE
- [ ] Performance 비교: 실행 시간, GPU 메모리 사용량
- [ ] 자동 리포트 생성: HTML 또는 Jupyter notebook template

**Key Files:**
- `compass/analysis/solver_comparison.py`
- `compass/visualization/report_generator.py`

---

### Task 6.3: Comparison Runner
**Priority: P1 | Estimated: 1 day**
**Dependencies: 6.2**

**Description:**
Hydra config에서 복수 솔버를 지정하여 동일 구조에 대해 순차 실행 + 비교.

**Acceptance Criteria:**
- [ ] `configs/experiment/solver_comparison.yaml` 의 solvers 리스트 처리
- [ ] 각 솔버 순차 실행 → 결과 수집
- [ ] SolverComparison 호출 → 비교 리포트 생성
- [ ] 결과 + 리포트를 output_dir에 저장

---

## Phase 7: I/O & Persistence (Week 7)

### Task 7.1: HDF5 Result Storage
**Priority: P1 | Estimated: 1.5 days**
**Dependencies: 2.1**

**Description:**
시뮬레이션 결과를 HDF5 포맷으로 저장/로드. 재현성을 위해 config도 함께 저장.

**Acceptance Criteria:**
- [ ] TRD의 HDF5 스키마 구현
- [ ] `save_result(result, config, filepath)`: full result + metadata 저장
- [ ] `load_result(filepath)` → SimulationResult + config
- [ ] 대용량 필드 데이터 선택적 저장 (config에서 `save_fields: true/false`)
- [ ] 압축: gzip level 4 (default)
- [ ] 결과 브라우징: `list_results(output_dir)` → 요약 테이블

**Key Files:**
- `compass/io/hdf5_handler.py`
- `compass/io/result_schema.py`

---

### Task 7.2: Export Module
**Priority: P2 | Estimated: 0.5 days**
**Dependencies: 7.1**

**Description:**
결과를 CSV, JSON 등 범용 포맷으로 내보내기.

**Acceptance Criteria:**
- [ ] QE 스펙트럼 → CSV (wavelength, QE_R, QE_G, QE_B, ...)
- [ ] 비교 결과 → JSON summary
- [ ] Matplotlib figure → PNG/SVG 자동 저장

---

## Phase 8: VitePress 문서화 & Examples (Week 7-9)

### Task 8.1: VitePress 프로젝트 초기 설정
**Priority: P0 | Estimated: 1 day**
**Dependencies: 0.1**
**Agent: viz-io**

**Description:**
VitePress 프로젝트 스캐폴딩, 테마 설정, 디렉토리 구조 생성. KaTeX 수식, Mermaid 다이어그램 지원 구성.

**Acceptance Criteria:**
- [ ] `docs/` 디렉토리 초기화 (`npx vitepress init`)
- [ ] `docs/.vitepress/config.mts` 완성 (TRD Section 12.4 기반: nav, sidebar, search, KaTeX, Mermaid)
- [ ] 커스텀 테마 확장 (`docs/.vitepress/theme/index.ts`, `custom.css`)
- [ ] 랜딩 페이지 (`docs/index.md`) — Hero, Features 섹션
- [ ] `docs/package.json` 의존성 설정
- [ ] `npm run docs:dev`로 로컬 빌드 정상 동작
- [ ] GitHub Pages 배포 CI 워크플로우 (`.github/workflows/docs.yml`)

---

### Task 8.2: Theory 섹션 — 빛과 전자기학 기초
**Priority: P1 | Estimated: 3 days**
**Dependencies: 8.1**
**Agent: viz-io (작성) + solvers-physics (물리 정확성 리뷰)**

**Description:**
광학 비전공자를 위한 이론 기초 문서. 일상적 비유와 인터랙티브 시각 자료로 접근성 확보. 모든 수식 직후에 직관적 해석 추가.

**Acceptance Criteria:**
- [ ] `theory/light-basics.md`: 빛의 본질, 파장↔색, 반사/투과/흡수, 굴절률 (n, k)
- [ ] `theory/electromagnetic-waves.md`: 맥스웰 방정식의 의미 (수식 최소화, 개념 중심)
- [ ] `theory/thin-film-optics.md`: 박막 간섭, ARC/BARL 원리, Fresnel 방정식
- [ ] `theory/diffraction.md`: 회절 격자, Floquet 정리, pixel pitch와 파장 관계
- [ ] 각 페이지에 최소 1개 다이어그램/그림
- [ ] 물리 전공자 리뷰 통과 (Agent 2)

**인터랙티브 컴포넌트:**
- [ ] `WavelengthSlider.vue`: 파장 슬라이더 → 색상 + Si 흡수깊이 표시
- [ ] `FresnelCalculator.vue`: n1, n2, 입사각 입력 → R/T 실시간 계산

---

### Task 8.3: Theory 섹션 — RCWA & FDTD 해설
**Priority: P1 | Estimated: 3 days**
**Dependencies: 8.2**
**Agent: solvers-physics (작성) + viz-io (시각화)**

**Description:**
RCWA와 FDTD 시뮬레이션 방법론의 완전 해설. 비전공자가 '무엇을 왜 계산하는지' 이해할 수 있는 수준. 수치 안정성 이슈도 실전 경험 기반으로 설명.

**Acceptance Criteria:**
- [ ] `theory/rcwa-explained.md`: Step-by-step (staircase → Fourier → eigen → S-matrix → QE), Mermaid 플로우차트
- [ ] `theory/fdtd-explained.md`: Yee grid, CFL 조건, PBC/PML, broadband 특성
- [ ] `theory/rcwa-vs-fdtd.md`: 비교 테이블, 사용 시나리오별 권장 솔버
- [ ] `theory/image-sensor-optics.md`: BSI 구조 층별 설명, CRA 개념
- [ ] `theory/quantum-efficiency.md`: QE 정의, 파장 의존성, 손실 요인
- [ ] `theory/numerical-stability.md`: 발산 4대 원인, 5-layer 방어 전략 (비전공자 언어)

**인터랙티브 컴포넌트:**
- [ ] `FourierOrderDemo.vue`: Fourier order 변화에 따른 구조 근사 품질 시각화
- [ ] `StackVisualizer.vue`: 픽셀 스택 인터랙티브 단면도 (클릭 → 상세 설명)

---

### Task 8.4: Guide 섹션 — 설치 & 퀵스타트
**Priority: P0 | Estimated: 2 days**
**Dependencies: 8.1, Phase 1-2 완료**
**Agent: viz-io**

**Description:**
설치부터 첫 시뮬레이션까지의 hands-on 가이드. 복사-붙여넣기로 바로 실행 가능.

**Acceptance Criteria:**
- [ ] `guide/installation.md`: pip install, CUDA/MPS/CPU별 설치, 환경 변수 설정
- [ ] `guide/quickstart.md`: 5분 내 첫 QE 계산 완료 (최소 config, 최소 코드)
- [ ] `guide/first-simulation.md`: BSI 2×2 unit cell → torcwa → QE 스펙트럼 → 시각화 전체 과정
- [ ] 각 단계에 예상 출력 스크린샷/플롯 포함
- [ ] 트러블슈팅 FAQ 연결

---

### Task 8.5: Guide 섹션 — 심화 가이드
**Priority: P1 | Estimated: 3 days**
**Dependencies: 8.4, Phase 3-6**
**Agent: viz-io + solvers-physics**

**Acceptance Criteria:**
- [ ] `guide/pixel-stack-config.md`: YAML config 작성법, 파라미터별 설명, 예제
- [ ] `guide/material-database.md`: 내장 재료, 커스텀 CSV 추가, 분산 모델 선택
- [ ] `guide/choosing-solver.md`: 솔버 비교 의사결정 트리
- [ ] `guide/running-rcwa.md`: torcwa/grcwa/meent 실행, stability 설정
- [ ] `guide/running-fdtd.md`: flaport/fdtdz/meep 실행
- [ ] `guide/cross-validation.md`: 솔버 간 비교 워크플로우, ΔQE 해석
- [ ] `guide/cone-illumination.md`: F-number, CRA, 샘플링 설정
- [ ] `guide/roi-sweep.md`: 센서 전체 ROI별 QE map 생성
- [ ] `guide/visualization.md`: 2D/3D 플롯 커스터마이징
- [ ] `guide/troubleshooting.md`: 에러 메시지별 원인 & 해결
- [ ] Advanced 가이드: stability tuning, GPU 최적화, custom materials, batch sim

---

### Task 8.6: Reference 섹션 — API & Config 문서
**Priority: P1 | Estimated: 3 days**
**Dependencies: Phase 1-6 완료**
**Agent: qa-benchmark (docstring 기반 추출) + viz-io (포맷팅)**

**Description:**
Python docstring에서 자동/수동 추출한 API 문서 + config YAML 상세 레퍼런스.

**Acceptance Criteria:**
- [ ] API 문서: PixelStack, GeometryBuilder, MaterialDB, SolverBase, 각 adapter, Sources, QECalculator, Viz
- [ ] 각 API 페이지: class/method signature, 파라미터 설명, 반환값, 예제 코드
- [ ] Config 문서: pixel, solver, source, stability, compute 각 YAML 옵션 상세 (기본값, 허용 범위, 설명)
- [ ] Materials 문서: 내장 재료 목록 (n,k 출처 포함), 광학 상수 형식, Cauchy/Sellmeier/Lorentz-Drude
- [ ] CLI 명령어 레퍼런스

---

### Task 8.7: Cookbook — 실전 레시피
**Priority: P1 | Estimated: 2 days**
**Dependencies: 8.5, Phase 6**
**Agent: solvers-physics (시뮬레이션 수행) + viz-io (문서화)**

**Description:**
실제 센서 설계 시나리오를 재현하는 step-by-step 레시피. 각 레시피는 YAML config + Python 코드 + 결과 해석을 포함.

**Acceptance Criteria:**
- [ ] `bsi-2x2-basic.md`: 가장 기본적인 BSI Bayer 시뮬레이션
- [ ] `metal-grid-effect.md`: Grid 유무에 따른 QE, crosstalk 비교
- [ ] `microlens-optimization.md`: 렌즈 파라미터 (n, α) sweep → 최적 QE
- [ ] `cra-shift-analysis.md`: CRA 0°~30° → 렌즈 shift → QE 변화
- [ ] `barl-design.md`: 다층 반사방지막 두께 최적화
- [ ] `dti-crosstalk.md`: DTI 깊이별 crosstalk 분석
- [ ] `wavelength-sweep.md`: 380-780nm 전 가시광 QE 스펙트럼
- [ ] `solver-benchmark.md`: torcwa vs grcwa vs FDTD 벤치마크 재현 가이드

---

### Task 8.8: Jupyter Notebook 예제
**Priority: P1 | Estimated: 2 days**
**Dependencies: Phase 1-6 완료**
**Agent: viz-io**

**Description:**
VitePress 문서와 연계되는 실행 가능한 Jupyter notebook 예제.

**Acceptance Criteria:**
- [ ] `01_quick_start.ipynb`: 2×2 unit cell, torcwa, single wavelength → QE
- [ ] `02_structure_visualization.ipynb`: 다양한 구조 정의 → 2D/3D 시각화
- [ ] `03_solver_comparison.ipynb`: torcwa vs grcwa vs meent → QE 비교
- [ ] `04_roi_sweep.ipynb`: 센서 ROI별 QE map 생성
- [ ] `05_stability_demo.ipynb`: float32 vs float64 발산 사례 + fallback 시연
- [ ] 각 notebook에서 VitePress 문서 해당 페이지로의 링크 제공

---

### Task 8.9: 참고문헌 & 프로젝트 정보
**Priority: P2 | Estimated: 1 day**
**Dependencies: 8.2, 8.3**
**Agent: viz-io**

**Acceptance Criteria:**
- [ ] `about/references.md`: 전체 참고 논문/책/코드 목록 (BibTeX 형식 + 인라인 설명)
- [ ] `about/changelog.md`: 버전별 변경 이력 템플릿
- [ ] `about/roadmap.md`: 향후 계획 (Web UI, 자동 최적화, TCAD 연계 등)
- [ ] `about/license.md`: 라이선스 정보
- [ ] `contributing/` 5개 페이지: 개발 환경, 솔버 추가, 재료 추가, 코드 컨벤션

---

## Phase 9: Testing & Benchmarking (Continuous)

### Task 9.1: Unit Test Suite
**Priority: P0 | Estimated: Continuous**
**Dependencies: Each module**

각 모듈과 동시에 작성. pytest 기반.

---

### Task 9.2: Benchmark Suite
**Priority: P1 | Estimated: 2 days**
**Dependencies: Phase 2-4**

**Acceptance Criteria:**
- [ ] Fresnel slab test: 단일 유전체 slab → 해석해 대비 < 0.5%
- [ ] Grating test: 1D/2D grating → 문헌값 대비
- [ ] Energy conservation: 다양한 구조에서 R+T+A = 1 (< 1%)
- [ ] Convergence test: Fourier order sweep → QE saturation curve
- [ ] Performance benchmark: 2×2, 4×4 unit cell × 31 wavelengths → 실행 시간

---

## 우선순위 요약

| Priority | Tasks | 완료 시 얻는 것 |
|---|---|---|
| **P0** | 0.1~0.3, 1.1~1.4, 2.1~2.2, 2.5~2.6, 3.1, 3.4, 5.4, 6.1, 9.1 | Agent Teams 환경 구축, 단일 솔버(torcwa) 동작, 수치 안정성 보장, QE 계산, 구조/결과 시각화 |
| **P1** | 2.3~2.4, 3.2~3.3, 4.1, 5.1, 5.3, 6.2~6.3, 7.1, 8.1~8.2, 9.2 | 솔버 비교, cone illumination, ROI sweep, 전체 기능 |
| **P2** | 4.2~4.3, 5.2, 7.2 | 고급 FDTD, Zemax 연동, export |

---

## 기술적 리스크 & 대응

| 리스크 | 영향 | 대응 |
|---|---|---|
| **RCWA eigendecomp 발산 (float32)** | **QE>1, NaN, 전체 결과 신뢰 불가** | **Mixed precision (eigen만 f64), adaptive fallback, TF32 금지** |
| **Fourier factorization 미적용** | **TM편광에서 수렴 실패, metal grid 구조 시뮬 불가** | **Li inverse rule 필수 적용, NV method for 2D** |
| Microlens staircase 수렴 느림 | RCWA QE 부정확 | Convergence test 자동화, 최소 30 slices |
| FDTD dispersive Si 처리 | 파장별 개별 run 필요 → 느림 | Lorentz fitting (meep) 또는 파장별 batch |
| MPS complex tensor 미지원 | macOS GPU 가속 불가 | CPU fallback, 성능 경고 표시 |
| 솔버간 QE 불일치 > 5% | 비교 신뢰도 저하 | 단순 구조부터 검증, 원인 분석 workflow |
| Color filter n,k 데이터 부족 | QE 정확도 저하 | Generic Lorentz model + 사용자 CSV 지원 |
| 대규모 unit cell GPU OOM | 8×8+ 시뮬레이션 불가 | 메모리 추정 함수, resolution 자동 조정 |
