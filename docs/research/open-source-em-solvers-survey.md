# 오픈소스 RCWA / FDTD 광학 솔버 종합 조사

> 조사일: 2026-02-11 | COMPASS 프로젝트 참고용

---

## 1. RCWA 솔버

### 1.1 COMPASS에 이미 통합된 솔버

| 솔버 | 언어 | GPU | AD | 라이선스 | ⭐ | 상태 | 비고 |
|------|------|-----|-----|---------|-----|------|------|
| **torcwa** | Python/PyTorch | CUDA | ✅ PyTorch | LGPL | ~171 | ⚠️ 저활성 (2023~) | S-matrix, 메타서피스 역설계에 적합. 개발 정체 |
| **grcwa** | Python/autograd | ❌ | ✅ autograd | GPL | ~94 | ❌ 중단 (2020~) | CPU 전용. 토폴로지 최적화. 사실상 방치 |
| **meent** | Python (NumPy/JAX/PyTorch) | JAX/PyTorch | ✅ JAX/PyTorch | **MIT** | ~112 | ✅ 활발 | 3개 백엔드 지원, ML 통합 최적. 한국 기업(KC-ML2) 개발 |

**평가:** meent가 라이선스(MIT), 유지보수, 유연성 면에서 가장 우수. grcwa는 장기적으로 통합 유지 재고 필요.

### 1.2 통합 후보 (높은 우선순위)

| 솔버 | 언어 | GPU | AD | 라이선스 | ⭐ | 상태 | 핵심 강점 |
|------|------|-----|-----|---------|-----|------|----------|
| **fmmax** | Python/JAX | JAX | ✅ | **MIT** | ~137 | ✅ 활발 (Meta) | **최고 수준 벡터 FMM** (Li 역규칙 확장), Brillouin zone 적분, 배칭 지원. AR/VR 광학 |
| **torchrdit** | Python/PyTorch | CUDA | ✅ | GPL-3.0 | ~11 | ✅ 활발 | **고유값분해 불필요** (R-DIT), 기존 RCWA 대비 16.2× 속도향상, GDS import/export |
| **S4** (phoebe-p fork) | C++/Lua/Python | ❌ | ❌ | GPL-2.0 | ~166 | ⚠️ Fork 활발 | **RCWA 레퍼런스 구현**, Li 역규칙, Solcore/RayFlare 통합. 25년 역사 |

**fmmax 상세:**
- Meta Reality Labs에서 AR/VR 회절광학 설계용으로 개발
- 4가지 벡터 FMM 공식 (Pol, Normal, Jones, Jones-direct) → 수렴성 최고
- JAX JIT 컴파일 + 배칭으로 다중 구성 동시 계산
- 이방성(anisotropic) 및 자성 재료 지원

**torchrdit 상세:**
- 전통적 RCWA의 병목인 고유값분해를 제거한 R-DIT(Rigorous Diffraction Interface Theory)
- PyTorch 기반 → 기존 torcwa 코드와 유사한 인터페이스 가능
- 빠른 Fourier Factorization (POL, NORMAL, JONES, JONES_DIRECT)
- 분산 재료 자동 유전율 피팅

### 1.3 특수 용도 / 참고용

| 솔버 | 언어 | GPU | 라이선스 | ⭐ | 핵심 특징 |
|------|------|-----|---------|-----|----------|
| **inkstone** | Python/NumPy | ❌ | AGPL-3.0 | ~63 | **텐서 유전율/투자율** (이방성, 자기광학, 자이로자성). 부분 재계산 최적화 |
| **nannos** | Python | ✅ | GPL-3.0 | ~20 | 다중 FMM 공식, AD, GPU 가속. GitLab 호스팅 |
| **rcwa_tf** | Python/TensorFlow | CUDA | BSD-3 | ~51 | TF 기반, Lorentzian broadening (그래디언트 안정화), 배치 최적화 |
| **rcwa (edmundsj)** | Python | ❌ | **MIT** | ~134 | **refractiveindex.info 내장 DB**, TMM+RCWA, 타원편광측정 |
| **EMUstack** | Fortran/Python | ❌ | GPL-3.0 | ~28 | 하이브리드 2D-FEM + 산란행렬. 금속/플라즈모닉 구조에 강점 |
| **MESH** | C++ | ❌(MPI) | GPL-3.0 | ~33 | RCWA + 열복사 전달 (근장/원장). Stanford Fan Group |
| **RETICOLO** | MATLAB | ❌ | Freeware | N/A | **산업계 표준** (Zeiss, Intel, Apple, Samsung). V10(2025.01) 이방성 지원. 25년 역사 |
| **rcwa4d** | Python | ❌ | **MIT** | ~40 | 비정합 주기 (뒤틀린 이중층/moiré 구조). Stanford Fan Group |
| **RCWA.jl** | Julia | CUDA | GPL-3.0 | ~46 | S-matrix + ETM, 고유값-free 알고리즘, CUDA 5× 속도향상 |
| **EMpy** | Python | ❌ | **MIT** | ~219 | TMM + RCWA + 모드 솔버. 가장 많은 GitHub 스타 |

---

## 2. FDTD 솔버

### 2.1 COMPASS에 이미 통합된 솔버

| 솔버 | 언어 | GPU | AD | 라이선스 | ⭐ | 상태 | 비고 |
|------|------|-----|-----|---------|-----|------|------|
| **Meep** | C++/Python | ❌ | ✅ (adjoint) | GPL-2.0+ | ~1,500 | ✅ 활발 | 가장 성숙한 오픈소스 FDTD. MPI 병렬화. GPU 미지원이 단점 |
| **flaport/fdtd** | Python/PyTorch | CUDA | ✅ PyTorch | **MIT** | ~650 | ⚠️ 저활성 | 교육/프로토타이핑용. 단순한 API. 고급 재료모델 부재 |
| **fdtdz** | C++/CUDA/JAX | CUDA | ✅ JAX | **MIT** | ~146 | ✅ 활발 (Google) | Meep 대비 ~100× 속도. 2.5D 한정. 단순 유전체만 지원 |

**평가:** Meep = 범용 레퍼런스, fdtdz = 속도 특화(2.5D), flaport = 교육용. 범용 3D GPU FDTD가 부족.

### 2.2 통합 후보 (높은 우선순위)

| 솔버 | 언어 | GPU | AD | 라이선스 | ⭐ | 상태 | 핵심 강점 |
|------|------|-----|-----|---------|-----|------|----------|
| **FDTDX** | Python/JAX | JAX (멀티GPU) | ✅ | **MIT** | ~203 | ✅ 매우 활발 | **대규모 3D 역설계 최적**. 멀티GPU. Maxwell 시간역행 활용 메모리 효율적 그래디언트. JOSS 출판 |
| **Khronos.jl** | Julia | CUDA/ROCm/Metal/OneAPI | ✅ | **MIT** | ~66 | ✅ 활발 (Meta) | **멀티벤더 GPU** (NVIDIA+AMD+Apple+Intel). 순수 Julia. 미분가능 |
| **ceviche** | Python/autograd | ❌ | ✅ | **MIT** | ~390 | ⚠️ 저활성 | 미분가능 EM의 선구자. 2D FDFD+FDTD. Stanford Fan Group |

**FDTDX 상세:**
- JAX 기반 완전 미분가능 3D FDTD
- 멀티GPU 스케일링 → 수십억 그리드셀 시뮬레이션 가능
- Maxwell 방정식의 시간 가역성을 활용한 메모리 효율적 그래디언트
- JOSS(Journal of Open Source Software) 출판 (2025)
- MIT 라이선스 + 활발한 개발 → 장기 통합에 유리

**Khronos.jl 상세:**
- Meta Research 개발
- 유일한 멀티벤더 GPU 지원 (CUDA, ROCm, Metal, OneAPI)
- KernelAbstractions.jl 기반으로 벤더 독립적 GPU 코드
- DFT 모니터 + 수렴 기반 자동 중단
- Julia 생태계 의존성이 단점

### 2.3 특수 용도 / 참고용

| 솔버 | 언어 | GPU | 라이선스 | ⭐ | 핵심 특징 |
|------|------|-----|---------|-----|----------|
| **openEMS** | C++/MATLAB/Python | ❌(OpenMP/MPI) | GPL-3.0 | ~628 | EC-FDTD, 원통좌표, **RF/안테나/마이크로파 특화** |
| **gprMax** | Python/Cython | CUDA | GPL-3.0 | ~788 | **GPR(지중레이더) 특화**, CUDA 30× 속도향상. 토양 모델 |
| **EMOPT** | Python/C | ❌(MPI) | BSD-3 | ~110 | FDFD(2D/3D) + CW-FDTD, **형상 최적화** (경계 스무딩), 인접법 |
| **fdtd3d** | C++ | CUDA/MPI | GPL-3.0 | ~150 | MPI+OpenMP+CUDA, **크로스 아키텍처** (x64/ARM/RISC-V/Wasm) |
| **GSvit** | C/C++ | CUDA | GPL-2.0 | N/A | 나노스케일 광학 (SNOM, 거칠기), GPU 가속 |
| **Luminescent.jl** | Julia | CUDA | **MIT** | ~60 | 미분가능 FDTD (Zygote.jl), 반도체 포토닉스+음향+RF |
| **Angora** | C++ | ❌ | GPL | Small | 생의학 산란 특화 |

### 2.4 상용 / 비오픈소스 (참고)

| 솔버 | 비고 |
|------|------|
| **Tidy3D** (Flexcompute) | Python 클라이언트는 LGPL이나 **연산 엔진은 상용 클라우드**. 로컬 실행 불가. 매우 빠름 |
| **Lumerical** (Ansys) | **완전 상용**. 산업 표준 GUI. CUDA GPU 솔버. 학술 할인 있음 |

---

## 3. 기술 트렌드 분석

### 3.1 미분가능 EM 시뮬레이션 (Differentiable EM)
2024-2026 가장 큰 트렌드. 역설계(inverse design)와 토폴로지 최적화를 위해 AD(자동미분) 내장이 필수화되는 추세.

| 세대 | 대표 솔버 | AD 프레임워크 | 성능 |
|------|----------|--------------|------|
| 1세대 (2019) | ceviche, grcwa | autograd (CPU) | 느림, 2D |
| 2세대 (2021-23) | torcwa, flaport/fdtd | PyTorch | GPU 가속, 3D 제한적 |
| 3세대 (2024-26) | **fmmax, FDTDX, fdtdz, meent** | **JAX** | JIT+멀티GPU, 대규모 3D |

**결론:** JAX 생태계가 EM 솔버의 주류로 부상. PyTorch 기반도 건재하나, JAX의 JIT 컴파일 + vmap + pmap이 EM 시뮬레이션에 더 적합.

### 3.2 GPU 가속 현황
| 접근법 | 솔버 예시 | 속도향상 |
|--------|----------|---------|
| PyTorch CUDA | torcwa, flaport | 5-20× |
| JAX JIT+CUDA | fmmax, FDTDX, fdtdz, meent | 10-100× |
| 커스텀 CUDA 커널 | fdtdz | ~100× (Meep 대비) |
| Julia CUDA.jl | Khronos.jl, RCWA.jl | 5-10× |
| 멀티GPU | FDTDX, Khronos.jl | 선형 스케일링 |

### 3.3 라이선스 분포
| 라이선스 | 솔버 수 | 상용 활용 |
|----------|---------|----------|
| **MIT** | 10 (meent, fmmax, fdtdz, FDTDX, ceviche, rcwa4d, EMpy, flaport, Khronos.jl, Luminescent.jl) | ✅ 자유 |
| **BSD-3** | 2 (rcwa_tf, EMOPT) | ✅ 자유 |
| **GPL/LGPL** | 12+ (torcwa, grcwa, S4, nannos, RCWA.jl, Meep, openEMS, gprMax, torchrdit 등) | ⚠️ 제한적 |
| **AGPL** | 1 (inkstone) | ❌ 매우 제한적 |

---

## 4. COMPASS 통합 권장사항

### Tier 1: 강력 추천 (MIT 라이선스, 활발한 개발, 높은 성능)

| 솔버 | 유형 | 이유 |
|------|------|------|
| **fmmax** | RCWA | MIT, Meta 지원, 최고 수렴성 (벡터 FMM), JAX 배칭. meent의 JAX 백엔드와 시너지 가능 |
| **FDTDX** | FDTD | MIT, 멀티GPU 3D, 완전 미분가능, JOSS 출판. fdtdz의 2.5D 한계를 보완 |

### Tier 2: 검토 권장 (특수 강점)

| 솔버 | 유형 | 이유 |
|------|------|------|
| **torchrdit** | RCWA(R-DIT) | 고유값분해 제거로 대폭 속도향상. GPL이 걸림돌 |
| **S4** (phoebe-p fork) | RCWA | C++ 성능의 레퍼런스 구현. 검증용으로 유용. GPL |
| **ceviche** | FDTD/FDFD | 2D 미분가능 EM. 빠른 프로토타이핑용. MIT |

### Tier 3: 장기 관찰

| 솔버 | 유형 | 이유 |
|------|------|------|
| **Khronos.jl** | FDTD | 멀티벤더 GPU가 매력적이나 Julia 의존성 |
| **inkstone** | RCWA | 텐서 유전율 지원이 유일하나 AGPL |
| **Luminescent.jl** | FDTD | Julia 미분가능 FDTD. 아직 초기 단계 |

### 기존 통합 솔버 평가

| 솔버 | 유지 권장 | 비고 |
|------|----------|------|
| **meent** ✅ | 적극 유지 | MIT, 활발, 3개 백엔드, 최고의 유연성 |
| **torcwa** ⚠️ | 유지하되 주시 | LGPL, 개발 정체. meent로 대체 가능성 |
| **grcwa** ❌ | 폐기 검토 | GPL, 2020년 이후 중단. CPU 전용. 열위 |
| **Meep** ✅ | 적극 유지 | 범용 레퍼런스. GPU 미지원이 아쉬움 |
| **flaport** ⚠️ | 유지하되 주시 | MIT, 저활성. 교육/프로토타입용으로만 가치 |
| **fdtdz** ✅ | 유지 | MIT, Google 지원, 극한 속도. 2.5D 한계 인지 |

---

## 5. 전체 솔버 요약표

### RCWA (20개)

| # | 솔버 | 언어 | GPU | AD | 라이선스 | ⭐ | 상태 |
|---|------|------|-----|-----|---------|-----|------|
| 1 | **meent** | Py (NumPy/JAX/PyTorch) | ✅ | ✅ | MIT | 112 | ✅ 활발 |
| 2 | **fmmax** | Py/JAX | ✅ | ✅ | MIT | 137 | ✅ 활발 |
| 3 | **torcwa** | Py/PyTorch | ✅ | ✅ | LGPL | 171 | ⚠️ |
| 4 | **S4** | C++/Lua/Py | ❌ | ❌ | GPL-2.0 | 166 | ⚠️ Fork |
| 5 | **EMpy** | Py | ❌ | ❌ | MIT | 219 | ✅ |
| 6 | **rcwa (edmundsj)** | Py | ❌ | ❌ | MIT | 134 | ⚠️ |
| 7 | **grcwa** | Py/autograd | ❌ | ✅ | GPL | 94 | ❌ 중단 |
| 8 | **inkstone** | Py/NumPy | ❌ | ❌ | AGPL | 63 | ⚠️ |
| 9 | **rcwa_tf** | Py/TF | ✅ | ✅ | BSD-3 | 51 | ⚠️ |
| 10 | **RCWA.jl** | Julia | ✅ | ❌ | GPL-3.0 | 46 | ✅ |
| 11 | **rcwa4d** | Py | ❌ | ❌ | MIT | 40 | ⚠️ |
| 12 | **MESH** | C++ | ❌ | ❌ | GPL-3.0 | 33 | ⚠️ |
| 13 | **EMUstack** | Fortran/Py | ❌ | ❌ | GPL-3.0 | 28 | ⚠️ |
| 14 | **nannos** | Py | ✅ | ✅ | GPL-3.0 | 20 | ⚠️ |
| 15 | **torchrdit** | Py/PyTorch | ✅ | ✅ | GPL-3.0 | 11 | ✅ |
| 16 | **RETICOLO** | MATLAB | ❌ | ❌ | Freeware | N/A | ✅ |
| 17 | **PPML** | MATLAB | ❌ | ❌ | Free | N/A | ⚠️ |
| 18-20 | 교육용 3종 | Py | ❌ | ❌ | Various | <10 | ⚠️ |

### FDTD (17개)

| # | 솔버 | 언어 | GPU | AD | 라이선스 | ⭐ | 상태 |
|---|------|------|-----|-----|---------|-----|------|
| 1 | **Meep** | C++/Py | ❌ | ✅ adj | GPL-2.0+ | 1,500 | ✅ 활발 |
| 2 | **gprMax** | Py/Cython | CUDA | ❌ | GPL-3.0 | 788 | ✅ |
| 3 | **flaport/fdtd** | Py/PyTorch | ✅ | ✅ | MIT | 650 | ⚠️ |
| 4 | **openEMS** | C++ | ❌ | ❌ | GPL-3.0 | 628 | ✅ |
| 5 | **ceviche** | Py | ❌ | ✅ | MIT | 390 | ⚠️ |
| 6 | **FDTDX** | Py/JAX | ✅ 멀티 | ✅ | MIT | 203 | ✅ 매우 활발 |
| 7 | **Tidy3D** | Py | ☁️ 클라우드 | ✅ | LGPL/상용 | 164 | ✅ (비오픈소스) |
| 8 | **fdtd3d** | C++ | CUDA/MPI | ❌ | GPL-3.0 | 150 | ✅ |
| 9 | **fdtdz** | C++/CUDA/JAX | ✅ | ✅ | MIT | 146 | ✅ |
| 10 | **EMOPT** | Py/C | ❌ | ✅ adj | BSD-3 | 110 | ⚠️ |
| 11 | **PhotonTorch** | Py/PyTorch | ✅ | ✅ | MIT | 81 | ⚠️ (회로 시뮬) |
| 12 | **Khronos.jl** | Julia | ✅ 멀티벤더 | ✅ | MIT | 66 | ✅ |
| 13 | **Luminescent.jl** | Julia | ✅ | ✅ | MIT | 60 | ✅ |
| 14 | **GSvit** | C/C++ | CUDA | ❌ | GPL-2.0 | N/A | ✅ |
| 15 | **Angora** | C++ | ❌ | ❌ | GPL | Small | ⚠️ |
| 16 | **MaxwellFDTD.jl** | Julia | ❌ | ❌ | N/A | Small | ⚠️ |
| 17 | **REMS** | Rust | ❌ | ❌ | N/A | Tiny | ❌ (1D PoC) |

---

*참고: Rust 기반 RCWA 솔버는 현재(2026) 존재하지 않음. FDTD는 1D PoC(REMS)만 존재.*
