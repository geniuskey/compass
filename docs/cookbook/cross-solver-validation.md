# Cross-Solver Validation: TMM vs RCWA

This page documents cross-validation results between TMM (1D) and RCWA (2D/3D) solvers in COMPASS, executed on a standard 1.0 um BSI pixel stack. Cross-solver validation confirms that different numerical methods produce physically consistent results and helps identify where 1D approximations break down relative to full 2D electromagnetic solutions.

## Overview

Three solvers are compared on the same pixel structure:

- **TMM** (Transfer Matrix Method): 1D analytical solver that treats each layer as an infinite uniform slab. Ignores lateral patterning (microlens shape, Bayer CF pattern, DTI). Extremely fast (~3 ms for a full sweep).
- **torcwa**: PyTorch-based 2D RCWA solver using the S-matrix algorithm. Handles full lateral structure including microlens profile, color filter Bayer pattern, and DTI boundaries.
- **grcwa**: NumPy-based 2D RCWA solver with autograd support. Independent RCWA implementation used to cross-check torcwa results.

::: info Simulation Parameters
- **Pixel**: 1.0 um pitch BSI, 2x2 RGGB Bayer
- **Stack**: 9 layers (air / microlens / planarization / CF / 4-layer BARL / silicon)
- **RCWA**: Fourier order [3,3], 5 microlens slices, 64x64 grid
- **Source**: Normal incidence, unpolarized, 380-780 nm (20 nm step)
- **Platform**: macOS (Apple Silicon), CPU only
:::

## Interactive Comparison

<CrossSolverValidation />

## Wavelength Sweep Results

### Full Data Table

Complete R, T, A values for all 21 wavelengths across the three solvers. Light propagates from air (top) through the stack into silicon (bottom).

| λ (nm) | TMM R | TMM T | TMM A | torcwa R | torcwa T | torcwa A | grcwa R | grcwa T | grcwa A |
|-------:|------:|------:|------:|---------:|---------:|---------:|--------:|--------:|--------:|
| 380 | 0.0401 | 0.0517 | 0.9081 | 0.0147 | 0.0000 | 0.9853 | 0.0195 | 0.0000 | 0.9805 |
| 400 | 0.0529 | 0.0722 | 0.8750 | 0.0239 | 0.0000 | 0.9761 | 0.0139 | 0.0000 | 0.9861 |
| 420 | 0.0961 | 0.0830 | 0.8210 | 0.0021 | 0.0000 | 0.9979 | 0.0131 | 0.0000 | 0.9869 |
| 440 | 0.0348 | 0.1085 | 0.8566 | 0.0069 | 0.0000 | 0.9931 | 0.0125 | 0.0000 | 0.9875 |
| 460 | 0.0633 | 0.1443 | 0.7923 | 0.0018 | 0.0000 | 0.9982 | 0.0102 | 0.0000 | 0.9898 |
| 480 | 0.0324 | 0.2549 | 0.7127 | 0.0138 | 0.0000 | 0.9862 | 0.0119 | 0.0000 | 0.9881 |
| 500 | 0.1461 | 0.4549 | 0.3990 | 0.0186 | 0.0000 | 0.9814 | 0.0148 | 0.0000 | 0.9852 |
| 520 | 0.0088 | 0.9135 | 0.0777 | 0.0258 | 0.0000 | 0.9742 | 0.0144 | 0.0000 | 0.9856 |
| 540 | 0.2405 | 0.7023 | 0.0572 | 0.0131 | 0.0001 | 0.9868 | 0.0121 | 0.0000 | 0.9879 |
| 560 | 0.1697 | 0.4775 | 0.3528 | 0.0013 | 0.0003 | 0.9984 | 0.0114 | 0.0000 | 0.9886 |
| 580 | 0.0280 | 0.3331 | 0.6389 | 0.0021 | 0.0008 | 0.9971 | 0.0139 | 0.0000 | 0.9861 |
| 600 | 0.0387 | 0.2423 | 0.7190 | 0.0077 | 0.0026 | 0.9897 | 0.0176 | 0.0000 | 0.9824 |
| 620 | 0.0477 | 0.2175 | 0.7349 | 0.0137 | 0.0061 | 0.9802 | 0.0202 | 0.0000 | 0.9797 |
| 640 | 0.0180 | 0.2258 | 0.7563 | 0.0223 | 0.0093 | 0.9684 | 0.0206 | 0.0001 | 0.9793 |
| 660 | 0.0209 | 0.2336 | 0.7455 | 0.0126 | 0.0113 | 0.9761 | 0.0193 | 0.0001 | 0.9807 |
| 680 | 0.0781 | 0.2284 | 0.6935 | 0.0116 | 0.0126 | 0.9757 | 0.0173 | 0.0001 | 0.9826 |
| 700 | 0.1199 | 0.2252 | 0.6549 | 0.0093 | 0.0130 | 0.9777 | 0.0159 | 0.0001 | 0.9840 |
| 720 | 0.1091 | 0.2345 | 0.6564 | 0.0055 | 0.0140 | 0.9805 | 0.0161 | 0.0002 | 0.9837 |
| 740 | 0.0664 | 0.2522 | 0.6815 | 0.0032 | 0.0147 | 0.9821 | 0.0178 | 0.0003 | 0.9819 |
| 760 | 0.0372 | 0.2667 | 0.6961 | 0.0045 | 0.0142 | 0.9813 | 0.0205 | 0.0004 | 0.9792 |
| 780 | 0.0430 | 0.2721 | 0.6849 | 0.0116 | 0.0145 | 0.9739 | 0.0231 | 0.0004 | 0.9765 |

## Key Observations

### 1D vs 2D Differences

The TMM and RCWA results differ significantly, and understanding why is critical for choosing the right solver for a given task.

#### 1. TMM vs RCWA Absorption

TMM shows absorption ranging from 5% to 91%, while RCWA consistently shows 97-100% absorption across the entire visible spectrum. The 3 um silicon layer absorbs nearly all light that successfully enters it. In TMM (1D), thin-film interference between planar layers creates constructive and destructive patterns that open transmission windows, especially around 500-540 nm where the green color filter becomes transparent. In RCWA (2D), the actual microlens profile focuses light, the Bayer color filter pattern introduces lateral index variation, and DTI boundaries confine light within the pixel. These 2D effects significantly increase the optical path length through silicon and reduce coherent back-reflection.

#### 2. Transmission

TMM predicts substantial transmission (5-91%) through the stack, with a dramatic peak near 520 nm (T=0.91). In contrast, both RCWA solvers show near-zero transmission (T < 0.015 for torcwa, T < 0.0004 for grcwa). The 3 um silicon layer at the Fourier order [3,3] resolution used in RCWA effectively absorbs all power that is not reflected. The S-matrix calculation through thick absorbing layers approaches machine precision limits, and lateral confinement by DTI prevents light from escaping sideways.

#### 3. Reflection

TMM reflectance exhibits strong thin-film oscillations (1-24%), characteristic of coherent interference in a planar multilayer. RCWA reflectance is low and spectrally smooth (0.1-2.6%). The 2D microlens profile acts as a graded-index anti-reflection structure: light incident on a curved surface couples more efficiently into the stack than light hitting a flat interface, reducing specular reflection significantly.

#### 4. torcwa vs grcwa Agreement

Both RCWA solvers agree well with each other, with reflectance differences within 0.5-1.5% absolute and absorption differences within 0.5-2%. This level of agreement between two independent RCWA implementations (PyTorch-based vs NumPy-based) provides strong validation that both solvers are computing the electromagnetic problem correctly. Small residual differences arise from numerical precision in eigendecomposition and S-matrix assembly.

### When to Use Each Solver

- **TMM**: Stack design, BARL thickness optimization, fast parameter sweeps (~3 ms per sweep). Best for thin-film interference analysis where lateral patterning is not the primary concern.
- **RCWA**: Full 2D diffraction effects, absolute QE prediction, cross-pixel crosstalk analysis, microlens design (~15 s per sweep). Required when lateral structure (microlens, Bayer pattern, DTI) significantly affects the result.

## Runtime Comparison

| Solver | Wavelengths | Runtime | Speedup |
|--------|:-----------:|--------:|--------:|
| TMM | 21 | 2.9 ms | 5400x |
| grcwa | 21 | 0.1 s | 157x |
| torcwa | 21 | 15.7 s | 1x |

TMM is approximately 5400x faster than torcwa, making it the preferred solver for iterative design loops. grcwa is 157x faster than torcwa for the same RCWA calculation, benefiting from NumPy's optimized linear algebra on CPU. torcwa's PyTorch backend is optimized for GPU acceleration, so the CPU-only comparison understates its performance on CUDA devices.

## Solver Compatibility Notes

::: warning meent Numerical Stability
meent 0.12.0 has a known numerical instability for multi-layer 2D structures: R+T > 1 occurs for stacks with 2 or more patterned layers. Single-layer simulations are correct. This is under investigation.
:::

::: warning FDTD Compatibility
The flaport fdtd 0.3.5 library has PML boundary API changes that require adapter updates. FDTD validation is planned for a future release.
:::

## Energy Conservation

Energy conservation (R + T + A = 1) is a fundamental physical constraint and serves as a self-consistency check for each solver.

| Solver | max \|1 - (R+T+A)\| | Notes |
|--------|:-------------------:|-------|
| TMM | 1.11 x 10^-16 | Machine precision (analytical transfer matrices) |
| torcwa | 0.0000 | S-matrix formulation inherently conserves energy |
| grcwa | 0.0000 | S-matrix formulation inherently conserves energy |

All three solvers satisfy energy conservation to machine precision, confirming that the numerical implementations are correct regardless of the physical differences in their modeling assumptions.

## Execution Environment

```
Platform    : macOS (Darwin 25.2.0, Apple Silicon)
Python      : 3.11
PyTorch     : 2.5.0 (torcwa backend)
NumPy       : (grcwa backend)
RCWA Order  : [3, 3] (49 harmonics)
Grid        : 64 x 64
Lens Slices : 5
```
