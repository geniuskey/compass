# Comprehensive Survey of Open-Source RCWA / FDTD Optical Solvers

> Survey date: 2026-02-11 | For COMPASS project reference

---

## 1. RCWA Solvers

### 1.1 Solvers Already Integrated in COMPASS

| Solver | Language | GPU | AD | License | ⭐ | Status | Notes |
|------|------|-----|-----|---------|-----|------|------|
| **torcwa** | Python/PyTorch | CUDA | ✅ PyTorch | LGPL | ~171 | ⚠️ Low activity (2023~) | S-matrix, suitable for metasurface inverse design. Development stalled |
| **grcwa** | Python/autograd | ❌ | ✅ autograd | GPL | ~94 | ❌ Discontinued (2020~) | CPU only. Topology optimization. Effectively abandoned |
| **meent** | Python (NumPy/JAX/PyTorch) | JAX/PyTorch | ✅ JAX/PyTorch | **MIT** | ~112 | ✅ Active | 3 backends supported, best for ML integration. Developed by Korean company (KC-ML2) |

**Assessment:** meent is the best in terms of licensing (MIT), maintenance, and flexibility. grcwa needs reassessment for long-term integration viability.

### 1.2 Integration Candidates (High Priority)

| Solver | Language | GPU | AD | License | ⭐ | Status | Key Strengths |
|------|------|-----|-----|---------|-----|------|----------|
| **fmmax** | Python/JAX | JAX | ✅ | **MIT** | ~137 | ✅ Active (Meta) | **State-of-the-art vector FMM** (Li inverse rule extension), Brillouin zone integration, batching support. AR/VR optics |
| **torchrdit** | Python/PyTorch | CUDA | ✅ | GPL-3.0 | ~11 | ✅ Active | **No eigenvalue decomposition required** (R-DIT), 16.2x speedup over conventional RCWA, GDS import/export |
| **S4** (phoebe-p fork) | C++/Lua/Python | ❌ | ❌ | GPL-2.0 | ~166 | ⚠️ Fork active | **RCWA reference implementation**, Li inverse rule, Solcore/RayFlare integration. 25-year history |

**fmmax details:**
- Developed by Meta Reality Labs for AR/VR diffractive optics design
- 4 vector FMM formulations (Pol, Normal, Jones, Jones-direct) → best convergence
- JAX JIT compilation + batching for simultaneous computation of multiple configurations
- Anisotropic and magnetic material support

**torchrdit details:**
- R-DIT (Rigorous Diffraction Interface Theory) eliminates eigenvalue decomposition, the traditional RCWA bottleneck
- PyTorch-based → similar interface to existing torcwa code possible
- Fast Fourier Factorization (POL, NORMAL, JONES, JONES_DIRECT)
- Automatic permittivity fitting for dispersive materials

### 1.3 Special Purpose / Reference

| Solver | Language | GPU | License | ⭐ | Key Features |
|------|------|-----|---------|-----|----------|
| **inkstone** | Python/NumPy | ❌ | AGPL-3.0 | ~63 | **Tensor permittivity/permeability** (anisotropic, magneto-optic, gyromagnetic). Partial recomputation optimization |
| **nannos** | Python | ✅ | GPL-3.0 | ~20 | Multiple FMM formulations, AD, GPU acceleration. GitLab hosted |
| **rcwa_tf** | Python/TensorFlow | CUDA | BSD-3 | ~51 | TF-based, Lorentzian broadening (gradient stabilization), batch optimization |
| **rcwa (edmundsj)** | Python | ❌ | **MIT** | ~134 | **Built-in refractiveindex.info DB**, TMM+RCWA, ellipsometry |
| **EMUstack** | Fortran/Python | ❌ | GPL-3.0 | ~28 | Hybrid 2D-FEM + scattering matrix. Strong for metallic/plasmonic structures |
| **MESH** | C++ | ❌(MPI) | GPL-3.0 | ~33 | RCWA + thermal radiation transfer (near-field/far-field). Stanford Fan Group |
| **RETICOLO** | MATLAB | ❌ | Freeware | N/A | **Industry standard** (Zeiss, Intel, Apple, Samsung). V10 (2025.01) anisotropy support. 25-year history |
| **rcwa4d** | Python | ❌ | **MIT** | ~40 | Incommensurate periodicity (twisted bilayer/moiré structures). Stanford Fan Group |
| **RCWA.jl** | Julia | CUDA | GPL-3.0 | ~46 | S-matrix + ETM, eigenvalue-free algorithm, CUDA 5x speedup |
| **EMpy** | Python | ❌ | **MIT** | ~219 | TMM + RCWA + mode solver. Most GitHub stars |

---

## 2. FDTD Solvers

### 2.1 Solvers Already Integrated in COMPASS

| Solver | Language | GPU | AD | License | ⭐ | Status | Notes |
|------|------|-----|-----|---------|-----|------|------|
| **Meep** | C++/Python | ❌ | ✅ (adjoint) | GPL-2.0+ | ~1,500 | ✅ Active | Most mature open-source FDTD. MPI parallelization. Lack of GPU support is a drawback |
| **flaport/fdtd** | Python/PyTorch | CUDA | ✅ PyTorch | **MIT** | ~650 | ⚠️ Low activity | For education/prototyping. Simple API. Lacks advanced material models |
| **fdtdz** | C++/CUDA/JAX | CUDA | ✅ JAX | **MIT** | ~146 | ✅ Active (Google) | ~100x speed vs Meep. Limited to 2.5D. Only simple dielectrics supported |

**Assessment:** Meep = general-purpose reference, fdtdz = speed-focused (2.5D), flaport = educational. Lacking a general-purpose 3D GPU FDTD.

### 2.2 Integration Candidates (High Priority)

| Solver | Language | GPU | AD | License | ⭐ | Status | Key Strengths |
|------|------|-----|-----|---------|-----|------|----------|
| **FDTDX** | Python/JAX | JAX (multi-GPU) | ✅ | **MIT** | ~203 | ✅ Very active | **Optimal for large-scale 3D inverse design**. Multi-GPU. Memory-efficient gradients using Maxwell time-reversal. JOSS published |
| **Khronos.jl** | Julia | CUDA/ROCm/Metal/OneAPI | ✅ | **MIT** | ~66 | ✅ Active (Meta) | **Multi-vendor GPU** (NVIDIA+AMD+Apple+Intel). Pure Julia. Differentiable |
| **ceviche** | Python/autograd | ❌ | ✅ | **MIT** | ~390 | ⚠️ Low activity | Pioneer of differentiable EM. 2D FDFD+FDTD. Stanford Fan Group |

**FDTDX details:**
- JAX-based fully differentiable 3D FDTD
- Multi-GPU scaling → billions of grid cells simulation possible
- Memory-efficient gradients leveraging time-reversibility of Maxwell's equations
- Published in JOSS (Journal of Open Source Software) (2025)
- MIT license + active development → favorable for long-term integration

**Khronos.jl details:**
- Developed by Meta Research
- Only solver with multi-vendor GPU support (CUDA, ROCm, Metal, OneAPI)
- Vendor-independent GPU code based on KernelAbstractions.jl
- DFT monitors + convergence-based automatic termination
- Julia ecosystem dependency is a drawback

### 2.3 Special Purpose / Reference

| Solver | Language | GPU | License | ⭐ | Key Features |
|------|------|-----|---------|-----|----------|
| **openEMS** | C++/MATLAB/Python | ❌(OpenMP/MPI) | GPL-3.0 | ~628 | EC-FDTD, cylindrical coordinates, **RF/antenna/microwave specialized** |
| **gprMax** | Python/Cython | CUDA | GPL-3.0 | ~788 | **GPR (Ground Penetrating Radar) specialized**, CUDA 30x speedup. Soil models |
| **EMOPT** | Python/C | ❌(MPI) | BSD-3 | ~110 | FDFD (2D/3D) + CW-FDTD, **shape optimization** (boundary smoothing), adjoint method |
| **fdtd3d** | C++ | CUDA/MPI | GPL-3.0 | ~150 | MPI+OpenMP+CUDA, **cross-architecture** (x64/ARM/RISC-V/Wasm) |
| **GSvit** | C/C++ | CUDA | GPL-2.0 | N/A | Nanoscale optics (SNOM, roughness), GPU accelerated |
| **Luminescent.jl** | Julia | CUDA | **MIT** | ~60 | Differentiable FDTD (Zygote.jl), semiconductor photonics+acoustics+RF |
| **Angora** | C++ | ❌ | GPL | Small | Biomedical scattering specialized |

### 2.4 Commercial / Non-Open-Source (Reference)

| Solver | Notes |
|------|------|
| **Tidy3D** (Flexcompute) | Python client is LGPL but **computation engine is commercial cloud**. Cannot run locally. Very fast |
| **Lumerical** (Ansys) | **Fully commercial**. Industry standard GUI. CUDA GPU solver. Academic discounts available |

---

## 3. Technology Trend Analysis

### 3.1 Differentiable EM Simulation (Differentiable EM)
The biggest trend of 2024-2026. Built-in AD (automatic differentiation) is becoming essential for inverse design and topology optimization.

| Generation | Representative Solvers | AD Framework | Performance |
|------|----------|--------------|------|
| 1st generation (2019) | ceviche, grcwa | autograd (CPU) | Slow, 2D |
| 2nd generation (2021-23) | torcwa, flaport/fdtd | PyTorch | GPU accelerated, limited 3D |
| 3rd generation (2024-26) | **fmmax, FDTDX, fdtdz, meent** | **JAX** | JIT+multi-GPU, large-scale 3D |

**Conclusion:** The JAX ecosystem is emerging as the mainstream for EM solvers. PyTorch-based solvers remain viable, but JAX's JIT compilation + vmap + pmap are better suited for EM simulation.

### 3.2 GPU Acceleration Status
| Approach | Solver Examples | Speedup |
|--------|----------|---------|
| PyTorch CUDA | torcwa, flaport | 5-20x |
| JAX JIT+CUDA | fmmax, FDTDX, fdtdz, meent | 10-100x |
| Custom CUDA kernels | fdtdz | ~100x (vs Meep) |
| Julia CUDA.jl | Khronos.jl, RCWA.jl | 5-10x |
| Multi-GPU | FDTDX, Khronos.jl | Linear scaling |

### 3.3 License Distribution
| License | Solver Count | Commercial Use |
|----------|---------|----------|
| **MIT** | 10 (meent, fmmax, fdtdz, FDTDX, ceviche, rcwa4d, EMpy, flaport, Khronos.jl, Luminescent.jl) | ✅ Free |
| **BSD-3** | 2 (rcwa_tf, EMOPT) | ✅ Free |
| **GPL/LGPL** | 12+ (torcwa, grcwa, S4, nannos, RCWA.jl, Meep, openEMS, gprMax, torchrdit, etc.) | ⚠️ Restricted |
| **AGPL** | 1 (inkstone) | ❌ Very restricted |

---

## 4. COMPASS Integration Recommendations

### Tier 1: Strongly Recommended (MIT License, Active Development, High Performance)

| Solver | Type | Rationale |
|------|------|------|
| **fmmax** | RCWA | MIT, Meta-backed, best convergence (vector FMM), JAX batching. Potential synergy with meent's JAX backend |
| **FDTDX** | FDTD | MIT, multi-GPU 3D, fully differentiable, JOSS published. Complements fdtdz's 2.5D limitation |

### Tier 2: Recommended for Review (Special Strengths)

| Solver | Type | Rationale |
|------|------|------|
| **torchrdit** | RCWA (R-DIT) | Major speedup by eliminating eigenvalue decomposition. GPL is an obstacle |
| **S4** (phoebe-p fork) | RCWA | C++ performance reference implementation. Useful for verification. GPL |
| **ceviche** | FDTD/FDFD | 2D differentiable EM. For rapid prototyping. MIT |

### Tier 3: Long-Term Watch

| Solver | Type | Rationale |
|------|------|------|
| **Khronos.jl** | FDTD | Multi-vendor GPU is attractive but Julia dependency |
| **inkstone** | RCWA | Unique tensor permittivity support but AGPL |
| **Luminescent.jl** | FDTD | Julia differentiable FDTD. Still early stage |

### Existing Integrated Solver Assessment

| Solver | Retention Recommendation | Notes |
|------|----------|------|
| **meent** ✅ | Actively retain | MIT, active, 3 backends, best flexibility |
| **torcwa** ⚠️ | Retain but monitor | LGPL, development stalled. Potential replacement by meent |
| **grcwa** ❌ | Consider deprecation | GPL, discontinued since 2020. CPU only. Inferior |
| **Meep** ✅ | Actively retain | General-purpose reference. Lack of GPU support is regrettable |
| **flaport** ⚠️ | Retain but monitor | MIT, low activity. Only valuable for education/prototyping |
| **fdtdz** ✅ | Retain | MIT, Google-backed, extreme speed. Acknowledge 2.5D limitation |

---

## 5. Complete Solver Summary Tables

### RCWA (20 solvers)

| # | Solver | Language | GPU | AD | License | ⭐ | Status |
|---|------|------|-----|-----|---------|-----|------|
| 1 | **meent** | Py (NumPy/JAX/PyTorch) | ✅ | ✅ | MIT | 112 | ✅ Active |
| 2 | **fmmax** | Py/JAX | ✅ | ✅ | MIT | 137 | ✅ Active |
| 3 | **torcwa** | Py/PyTorch | ✅ | ✅ | LGPL | 171 | ⚠️ |
| 4 | **S4** | C++/Lua/Py | ❌ | ❌ | GPL-2.0 | 166 | ⚠️ Fork |
| 5 | **EMpy** | Py | ❌ | ❌ | MIT | 219 | ✅ |
| 6 | **rcwa (edmundsj)** | Py | ❌ | ❌ | MIT | 134 | ⚠️ |
| 7 | **grcwa** | Py/autograd | ❌ | ✅ | GPL | 94 | ❌ Discontinued |
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
| 18-20 | 3 educational solvers | Py | ❌ | ❌ | Various | <10 | ⚠️ |

### FDTD (17 solvers)

| # | Solver | Language | GPU | AD | License | ⭐ | Status |
|---|------|------|-----|-----|---------|-----|------|
| 1 | **Meep** | C++/Py | ❌ | ✅ adj | GPL-2.0+ | 1,500 | ✅ Active |
| 2 | **gprMax** | Py/Cython | CUDA | ❌ | GPL-3.0 | 788 | ✅ |
| 3 | **flaport/fdtd** | Py/PyTorch | ✅ | ✅ | MIT | 650 | ⚠️ |
| 4 | **openEMS** | C++ | ❌ | ❌ | GPL-3.0 | 628 | ✅ |
| 5 | **ceviche** | Py | ❌ | ✅ | MIT | 390 | ⚠️ |
| 6 | **FDTDX** | Py/JAX | ✅ Multi | ✅ | MIT | 203 | ✅ Very active |
| 7 | **Tidy3D** | Py | ☁️ Cloud | ✅ | LGPL/Commercial | 164 | ✅ (Non-open-source) |
| 8 | **fdtd3d** | C++ | CUDA/MPI | ❌ | GPL-3.0 | 150 | ✅ |
| 9 | **fdtdz** | C++/CUDA/JAX | ✅ | ✅ | MIT | 146 | ✅ |
| 10 | **EMOPT** | Py/C | ❌ | ✅ adj | BSD-3 | 110 | ⚠️ |
| 11 | **PhotonTorch** | Py/PyTorch | ✅ | ✅ | MIT | 81 | ⚠️ (Circuit sim) |
| 12 | **Khronos.jl** | Julia | ✅ Multi-vendor | ✅ | MIT | 66 | ✅ |
| 13 | **Luminescent.jl** | Julia | ✅ | ✅ | MIT | 60 | ✅ |
| 14 | **GSvit** | C/C++ | CUDA | ❌ | GPL-2.0 | N/A | ✅ |
| 15 | **Angora** | C++ | ❌ | ❌ | GPL | Small | ⚠️ |
| 16 | **MaxwellFDTD.jl** | Julia | ❌ | ❌ | N/A | Small | ⚠️ |
| 17 | **REMS** | Rust | ❌ | ❌ | N/A | Tiny | ❌ (1D PoC) |

---

<SolverComparisonChart />

*Note: As of 2026, no Rust-based RCWA solver exists. For FDTD, only a 1D PoC (REMS) exists.*
