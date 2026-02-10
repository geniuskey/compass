# PRD: COMPASS — Cross-solver Optical Modeling Platform for Advanced Sensor Simulation

## 1. 프로젝트 개요

### 1.1 배경 및 동기
CMOS 이미지 센서 픽셀의 광학 설계 최적화에서 시뮬레이션 결과의 **정합성(consistency)** 검증은 핵심적인 과제이다. grcwa(GPU-RCWA) 기반 시뮬레이터를 사용하여 pixel-level QE 시뮬레이션을 수행해왔으나, 단일 솔버의 결과만으로는 신뢰도를 보장하기 어렵다. 동일한 구조에 대해 RCWA와 FDTD 등 서로 다른 수치 해법의 결과를 직접 비교함으로써 시뮬레이션 신뢰도를 높이고, 각 솔버의 적용 범위와 한계를 명확히 파악할 수 있다.

### 1.2 프로젝트 목표
- **통합 구조 정의**: 하나의 YAML config로 픽셀 스택 구조를 정의하고, 복수의 EM 솔버에서 동일 구조를 시뮬레이션
- **솔버 교체 가능성**: RCWA(grcwa, torcwa, meent) 및 FDTD(flaport/fdtd, fdtdz, meep) 솔버를 플러그인 방식으로 교체
- **현실적 조명 모델**: planewave 뿐만 아니라 렌즈 exit pupil에서의 cone illumination, CRA 기반 위치별 조명 재현
- **스케일 가능한 시뮬레이션**: 단일 픽셀부터 10×10 unit cell까지, 센서 ROI 별 sweep 지원
- **3D CAD 없는 파라메트릭 모델링**: 모든 구조를 YAML 파라미터만으로 정의
- **멀티 플랫폼 GPU 가속**: CUDA (primary), CPU fallback, Apple MPS 실험적 지원

### 1.3 핵심 산출물
- QE (Quantum Efficiency) per pixel per wavelength
- 크로스토크 맵 (인접 픽셀로의 광 누설)
- EM 필드 분포 (Ex, Ey, Ez, |E|²) 의 2D/3D 시각화
- 솔버 간 비교 리포트 (QE 차이, 필드 차이, 수렴성 분석)

---

## 2. 사용자 요구사항

### 2.1 구조 정의 (Pixel Stack)

빛은 위(air)에서 아래(silicon)로 진행하며, 각 층의 파라미터를 독립적으로 정의:

| 층 (Layer) | 핵심 파라미터 | 비고 |
|---|---|---|
| **Air** | - | 입사 매질, n=1.0 |
| **Microlens** | height, radius_x, radius_y, refractive_index, profile(superellipse params: n, α), array_pitch, shift_x/y(CRA offset) | 슈퍼엘립스 모델, 픽셀별 독립 shift |
| **Planarization (OC)** | thickness, refractive_index | Over-coat layer |
| **Color Filter + Grid** | cf_thickness, cf_materials(R/G/B별 n,k 스펙트럼), grid_width, grid_height, grid_material(W 또는 SiO₂), Bayer pattern type | 2×2 Bayer(RGGB), Quad Bayer 등 |
| **BARL (Bottom Anti-Reflective Layers)** | n_layers, per-layer thickness/material | 다층 박막 스택 (예: SiO₂/HfO₂/SiO₂), 광대역 반사 방지 |
| **Silicon + Photodiode** | si_thickness, pd_position(x,y,z), pd_size(x,y,z), dti_width, dti_depth, dti_material | BSI 구조, DTI(Deep Trench Isolation) 포함 |

**추가 구조 요소:**
- **DTI (Deep Trench Isolation)**: 픽셀 간 광학적 격리, 위치/크기/재료 정의
- **Metal grid**: 컬러 필터 사이 금속 격자, 광 차단 역할
- **Low-n layer**: 마이크로렌즈 하부 저굴절 층 (선택적)

### 2.2 Unit Cell 구성

| 구성 | 설명 | 용도 |
|---|---|---|
| 1×1 | 단일 픽셀 | 빠른 테스트, 수렴성 확인 |
| 2×2 | 최소 Bayer unit | 기본 QE/크로스토크 분석 |
| 4×4 | 2×Bayer 반복 | 주변 효과 포함 분석 |
| 6×6, 8×8, 10×10 | 대규모 array | edge effect, 정밀 크로스토크 |

- 주기 경계 조건(PBC)을 사용하여 무한 반복 모사
- FDTD의 경우 PML 경계 조건으로 유한 영역 모사도 가능

### 2.3 광원 및 조명 조건

#### 2.3.1 기본 Planewave
- 파장: 가시광 (380~780nm), NIR 확장 가능 (780~1100nm)
- 입사각: θ (polar), φ (azimuthal) 
- 편광: TE, TM, unpolarized (TE+TM 평균)

#### 2.3.2 Wavelength Sweep
- 단일 파장 시뮬레이션의 sweep (예: 380~780nm, 5nm step)
- RCWA: 파장별 독립 계산 (병렬화 용이)
- FDTD: broadband source 사용 가능 (단일 런으로 스펙트럼 추출)

#### 2.3.3 렌즈 Exit Pupil 조명 모사 (Cone Illumination)
- **문제 정의**: 실제 카메라에서 각 픽셀에 도달하는 빛은 렌즈의 exit pupil 전체 면적에서 온다. 이때:
  - Chief Ray Angle (CRA): exit pupil 중심에서 pixel 중심으로의 각도
  - Marginal Ray Angle (MRA): exit pupil 가장자리에서의 최대 각도
  - F-number가 cone의 반각을 결정
  - 센서 위치(image height)에 따라 CRA가 변함
  - Pupil 위치별 intensity 분포가 다름 (cos⁴θ 법칙 등)

- **구현 방식**:
  1. CRA(θ_chief, φ_chief)와 F-number로 illumination cone 정의
  2. Cone 내 angular sampling (theta, phi grid)
  3. 각 sampling point에 대한 planewave 시뮬레이션 수행
  4. Intensity weighting 적용 후 결과 적분
  5. Weighting function options: uniform, cosine, Gaussian, user-defined

- **입력 형식**:
  - 직접 파라미터 입력: CRA, F-number, pupil shape
  - JSON ray file (Zemax OpticStudio 호환): CRA/MRA vs image height 테이블
  - CSV 테이블: image_height, cra, f_number

#### 2.3.4 센서 ROI별 Unit Cell Sweep
- 센서 전체 이미지 면에서의 QE 분포 예측
- 입력: 센서 크기, CRA vs image height curve, microlens shift map
- 각 ROI 위치마다:
  1. 해당 위치의 CRA, F-number 결정
  2. 해당 CRA에 맞는 microlens shift 적용
  3. Unit cell 시뮬레이션 수행
  4. QE 결과 수집
- 출력: QE(R,G,B) vs image_height 맵, relative illumination 맵

### 2.4 재료 물성 (Material Database)

#### 2.4.1 기본 제공 재료
- **Silicon**: Palik 또는 Green 2008 데이터 (n, k vs λ)
- **SiO₂, Si₃N₄, TiO₂**: 일반 유전체
- **Color Filter**: R/G/B/IR 흡수 스펙트럼 (n, k vs λ)
- **Tungsten (W)**: metal grid 재료
- **Polymer**: microlens 재료 (일반적으로 n ≈ 1.5~1.6)

#### 2.4.2 재료 정의 방식
- 내장 material DB (YAML 포맷)
- 사용자 정의 CSV (wavelength, n, k)
- 분석 모델: Sellmeier, Cauchy, Drude-Lorentz

### 2.5 출력 및 시각화

#### 2.5.1 핵심 결과값
- **QE per pixel**: 각 픽셀의 파장별 양자효율
- **Spectral Response**: QE(λ) 곡선 (R, G, B 채널)
- **Angular Response**: QE(θ) 곡선
- **Crosstalk Matrix**: 인접 픽셀로의 에너지 누설 비율

#### 2.5.2 시각화 요구사항
- **2D Field Plot**: XY, XZ, YZ 단면의 |E|², Poynting vector
- **2D Cross-section Plot**: 구조 단면도 with 재료 색상 매핑
- **3D Viewer**: 픽셀 스택의 interactive 3D 시각화 (구조 확인용)
  - 각 레이어 on/off toggle
  - 재료별 색상/투명도 설정
  - 마우스 회전/줌/팬
- **비교 Plot**: 솔버 간 QE 비교, 필드 차이 맵

### 2.6 솔버 간 비교

#### 2.6.1 비교 대상
| 솔버 | 타입 | GPU | 특징 |
|---|---|---|---|
| grcwa | RCWA | ✅ (autograd) | 기존 사용 중, numpy+autograd backend |
| torcwa | RCWA | ✅ (PyTorch) | PyTorch native, S-parameter 출력 |
| meent | RCWA | ✅ (NumPy/JAX/PyTorch) | 멀티 백엔드, vector modeling |
| flaport/fdtd | FDTD | ✅ (PyTorch) | Python-native 3D FDTD |
| fdtdz | FDTD | ✅ (JAX) | Systolic GPU FDTD, 극한 성능 |
| meep | FDTD | ✅ (선택적) | 가장 성숙한 오픈소스 FDTD |

#### 2.6.2 비교 메트릭
- QE 절대값 차이 (Δ QE)
- QE 상대 오차 (%)
- 필드 분포 상관 계수
- 에너지 보존 검증 (R + T + A = 1)
- 수렴성 (RCWA: Fourier order, FDTD: grid resolution)
- 실행 시간 / GPU 메모리 사용량

---

## 3. 비기능 요구사항

### 3.1 성능
- 단일 파장, 2×2 unit cell RCWA: < 10초 (RTX 3090 기준)
- Wavelength sweep 31점, 2×2: < 5분
- FDTD는 RCWA 대비 10~100× 느릴 수 있음 (허용)

### 3.2 확장성
- 새로운 솔버 추가: Solver ABC 구현만으로 플러그인 가능
- 새로운 레이어 타입 추가: LayerBuilder 등록만으로 확장
- 새로운 재료 추가: YAML/CSV 파일 추가만으로 가능

### 3.3 재현성
- 모든 시뮬레이션 설정을 YAML로 완전히 재현 가능
- 결과 저장 시 설정 파일 자동 포함 (HDF5 + YAML metadata)
- 시뮬레이션 버전 태깅

### 3.4 플랫폼 호환성
- **Primary**: Linux + NVIDIA CUDA
- **Secondary**: macOS (CPU + MPS 실험적)
- **Fallback**: 모든 플랫폼 CPU 동작 보장

---

## 4. 범위 제한 (Out of Scope)

- 3D CAD 기반 구조 입력 (GDSII, STL 등)
- 전기적 시뮬레이션 (charge transport, TCAD)
- ISP (Image Signal Processing) 파이프라인
- 렌즈 설계 자체 (Zemax 등과의 인터페이스만 제공)
- 실시간 GUI (CLI + Jupyter 우선, Web UI는 Phase 2)

---

## 5. 성공 기준

1. 동일 구조에 대해 2개 이상의 솔버가 QE ±5% 이내 일치
2. 에너지 보존 오차 < 1%
3. Ansys Lumerical 또는 문헌값 대비 합리적 일치 (단순 구조 벤치마크)
4. 2×2 unit cell, 31 wavelength sweep이 30분 이내 완료 (RCWA 솔버)
5. 구조 시각화에서 레이어 배치, 재료 할당이 올바른지 육안 확인 가능

---

## 7. 개발 방법론: Agent Teams

COMPASS의 개발은 **Claude Code Agent Teams**를 활용한 multi-agent 병렬 개발 방식을 채택한다.

**4-Agent 구조:**
- **Core Engine**: 기반 구조 (config, geometry, materials, base classes)
- **Solvers & Physics**: EM 솔버 구현, 수치 안정성, 소스 모델, QE 분석
- **Viz & I/O**: 시각화, 데이터 저장, 문서화
- **QA & Benchmark**: 테스트, 코드 리뷰, 벤치마크, 검증

**기대 효과:**
- 순차 대비 약 40% 개발 기간 단축 (8-10주 → 5-6주)
- 에이전트 간 교차 검증으로 코드 품질 향상
- Context 격리로 대규모 코드베이스에서도 집중도 유지

상세 설계는 TRD Section 11을 참조.

---

## 8. 용어 정의

| 용어 | 설명 |
|---|---|
| QE (Quantum Efficiency) | 입사 광자 대비 photodiode에서 흡수된 광자 비율 |
| CRA (Chief Ray Angle) | Exit pupil 중심과 pixel 중심을 잇는 광선의 광축 대비 각도 |
| MRA (Marginal Ray Angle) | Exit pupil 가장자리에서 pixel로 향하는 광선 각도 |
| RCWA | Rigorous Coupled-Wave Analysis, 주기 구조의 주파수 영역 EM 해석법 |
| FDTD | Finite-Difference Time-Domain, 시간 영역 EM 해석법 |
| PBC | Periodic Boundary Condition, 주기 경계 조건 |
| PML | Perfectly Matched Layer, 흡수 경계 조건 |
| DTI | Deep Trench Isolation, 픽셀 간 깊은 트렌치 격리 구조 |
| BSI | Backside Illumination, 후면 조사형 이미지 센서 |
| Bayer Pattern | RGGB 배열의 컬러 필터 패턴 |
| Unit Cell | 시뮬레이션 반복 단위 (예: 2×2 Bayer) |
| ROI | Region of Interest, 센서 내 관심 영역 |
