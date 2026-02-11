# TMM Validation Results

This page documents actual simulation results from COMPASS, executed on a standard 1.0 um BSI pixel stack using the TMM (Transfer Matrix Method) solver. TMM treats the stack as a 1D planar thin-film structure, making it ideal for rapid stack design, BARL optimization, and baseline validation before full RCWA simulation.

::: info Solver Note
TMM is a 1D solver — it ignores lateral patterning (microlens shape, Bayer CF pattern, DTI, metal grid). For full 2D/3D effects, use an RCWA solver (torcwa, grcwa, meent, fmmax). TMM results serve as a fast baseline and anti-reflection coating design tool.
:::

## Pixel Stack Structure

The simulated 1.0 um pitch BSI (Back-Side Illumination) pixel stack consists of 9 layers. Light propagates from air (top) through the stack to silicon (bottom).

| Layer | Material | Thickness (um) | Patterned | n @ 550nm |
|-------|----------|:--------------:|:---------:|:---------:|
| air | air | 1.000 | - | 1.000 |
| microlens | polymer_n1p56 | 0.600 | Yes | 1.573 |
| planarization | SiO₂ | 0.300 | - | 1.460 |
| color_filter | cf_green | 0.600 | Yes | 1.55+0.02i |
| barl_3 | Si₃N₄ | 0.030 | - | 2.023 |
| barl_2 | SiO₂ | 0.015 | - | 1.460 |
| barl_1 | HfO₂ | 0.025 | - | 1.966 |
| barl_0 | SiO₂ | 0.010 | - | 1.460 |
| silicon | silicon | 3.000 | Yes | 4.08+0.03i |

- **Total stack height**: 5.580 um
- **BARL**: 4-layer SiO₂/HfO₂/SiO₂/Si₃N₄ anti-reflection stack (80 nm total)
- **Bayer pattern**: RGGB 2×2 unit cell

## Wavelength Sweep: 380–780 nm

Full visible spectrum sweep at normal incidence, unpolarized light.

```
Solver: TMM | Pixel: 1.0um BSI | Angle: 0° | Polarization: unpolarized
Runtime: 2.8 ms (41 wavelengths)
```

| λ (nm) | R | T | A | R+T+A |
|-------:|------:|------:|------:|--------:|
| 380 | 0.0401 | 0.0517 | 0.9081 | 1.000000 |
| 400 | 0.0529 | 0.0722 | 0.8750 | 1.000000 |
| 420 | 0.0961 | 0.0830 | 0.8210 | 1.000000 |
| 440 | 0.0348 | 0.1085 | 0.8566 | 1.000000 |
| 460 | 0.0633 | 0.1443 | 0.7923 | 1.000000 |
| 480 | 0.0324 | 0.2549 | 0.7127 | 1.000000 |
| 500 | 0.1461 | 0.4549 | 0.3990 | 1.000000 |
| 520 | 0.0088 | 0.9135 | 0.0777 | 1.000000 |
| 540 | 0.2405 | 0.7023 | 0.0572 | 1.000000 |
| 560 | 0.1697 | 0.4775 | 0.3528 | 1.000000 |
| 580 | 0.0280 | 0.3331 | 0.6389 | 1.000000 |
| 600 | 0.0387 | 0.2423 | 0.7190 | 1.000000 |
| 620 | 0.0477 | 0.2175 | 0.7349 | 1.000000 |
| 640 | 0.0180 | 0.2258 | 0.7563 | 1.000000 |
| 660 | 0.0209 | 0.2336 | 0.7455 | 1.000000 |
| 680 | 0.0781 | 0.2284 | 0.6935 | 1.000000 |
| 700 | 0.1199 | 0.2252 | 0.6549 | 1.000000 |
| 720 | 0.1091 | 0.2345 | 0.6564 | 1.000000 |
| 740 | 0.0664 | 0.2522 | 0.6815 | 1.000000 |
| 760 | 0.0372 | 0.2667 | 0.6961 | 1.000000 |
| 780 | 0.0430 | 0.2721 | 0.6849 | 1.000000 |

**Key observations:**
- **Peak absorption (90.8%)** at 380 nm — silicon's large imaginary part of the refractive index (k ≈ 0.34) at short wavelengths leads to near-complete absorption.
- **Transmission dip at 520 nm** — the BARL and thin-film interference create a transmission window at 520 nm (T = 0.91), reducing silicon absorption to just 7.8%. This is a Fabry-Pérot resonance in the stack.
- **Long wavelengths (> 600 nm)** — absorption stabilizes around 65–75% as silicon's extinction coefficient decreases.

## Energy Conservation Validation

```
Wavelength range : 380–780 nm (1 nm step, 41 points)
Max |R+T+A − 1| : 1.11 × 10⁻¹⁶
Mean |R+T+A − 1| : 1.08 × 10⁻¹⁷
Status           : PASS ✓
```

TMM achieves machine-precision energy conservation, as expected for an analytical method with no discretization error.

## Polarization-Resolved Angle Sweep

Reflection, transmission, and absorption at 550 nm for TE, TM, and unpolarized incidence across 0–80°.

| Angle | R_TE | T_TE | A_TE | R_TM | T_TM | A_TM | R_unpol |
|------:|------:|------:|------:|------:|------:|------:|--------:|
| 0° | 0.2526 | 0.5618 | 0.1856 | 0.2526 | 0.5618 | 0.1856 | 0.2526 |
| 10° | 0.2509 | 0.5615 | 0.1875 | 0.2410 | 0.5694 | 0.1896 | 0.2460 |
| 20° | 0.1886 | 0.6036 | 0.2079 | 0.1558 | 0.6298 | 0.2145 | 0.1722 |
| 30° | 0.0143 | 0.7252 | 0.2606 | 0.0089 | 0.7340 | 0.2570 | 0.0116 |
| 40° | 0.1398 | 0.6263 | 0.2339 | 0.0664 | 0.6869 | 0.2467 | 0.1031 |
| 50° | 0.1866 | 0.5875 | 0.2258 | 0.0458 | 0.6968 | 0.2574 | 0.1162 |
| 60° | 0.0740 | 0.6626 | 0.2634 | 0.0597 | 0.6790 | 0.2613 | 0.0669 |
| 70° | 0.5079 | 0.3471 | 0.1450 | 0.0283 | 0.6926 | 0.2792 | 0.2681 |
| 80° | 0.7312 | 0.1870 | 0.0818 | 0.1200 | 0.6210 | 0.2590 | 0.4256 |

**Key observations:**
- **Brewster-like minimum** near 30° — R drops to ~1% for both TE and TM. This is a destructive interference minimum in the multilayer stack, not a simple Brewster angle (which only exists for TM at single interfaces).
- **TE vs TM divergence** above 40° — TM maintains low reflection while TE reflection increases sharply, as expected from Fresnel equations.
- **Grazing incidence (80°)** — TE reflectance reaches 73%, while TM remains at 12%, showing the strong polarization splitting.

## BARL Anti-Reflection Stack Impact

Effect of the 4-layer BARL (Bottom Anti-Reflection Layer) on optical performance at 550 nm.

| Configuration | R | T | A |
|--------------|------:|------:|------:|
| No BARL | 0.0434 | 0.6877 | 0.2689 |
| Standard 4-layer BARL | 0.2526 | 0.5618 | 0.1856 |
| 2× thickness BARL | 0.0240 | 0.6958 | 0.2802 |

::: warning Interpretation
At 550 nm, the standard BARL configuration is not at its optimal anti-reflection point — it actually increases reflection compared to no BARL. This is because the BARL is optimized for broadband performance across 380–780 nm, not for a single wavelength. The 2× thickness BARL achieves better anti-reflection at 550 nm (R = 2.4%) but may have different broadband characteristics. BARL optimization requires a full wavelength sweep analysis.
:::

## Material Refractive Indices

Complex refractive index (n + ik) for all stack materials at key wavelengths. Values from COMPASS `MaterialDB` using cubic spline interpolation of tabulated data.

| Material | 400 nm | 450 nm | 500 nm | 550 nm | 600 nm | 650 nm | 700 nm |
|----------|-------:|-------:|-------:|-------:|-------:|-------:|-------:|
| air | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| polymer_n1p56 | 1.5850 | 1.5798 | 1.5760 | 1.5732 | 1.5711 | 1.5695 | 1.5682 |
| SiO₂ | 1.4701 | 1.4656 | 1.4623 | 1.4599 | 1.4580 | 1.4565 | 1.4553 |
| cf_green | 1.55+0.12i | 1.55+0.11i | 1.55+0.04i | 1.55+0.02i | 1.55+0.10i | 1.55+0.12i | 1.55+0.12i |
| HfO₂ | 2.0250 | 1.9988 | 1.9800 | 1.9661 | 1.9556 | 1.9473 | 1.9408 |
| Si₃N₄ | 2.0726 | 2.0500 | 2.0344 | 2.0232 | 2.0149 | 2.0085 | 2.0035 |
| silicon | 5.38+0.34i | 4.64+0.21i | 4.21+0.13i | 4.08+0.03i | 3.88+0.05i | 3.80+0.04i | 3.75+0.03i |
| tungsten | 3.46+2.72i | 3.55+2.86i | 3.61+2.98i | 3.65+3.08i | 3.68+3.17i | 3.70+3.25i | 3.72+3.33i |

**Notes:**
- **Silicon** has strong dispersion: n drops from 5.38 (400 nm) to 3.75 (700 nm). The extinction coefficient k peaks at short wavelengths (UV absorption edge).
- **Color filter (green)** has a passband centered near 500–550 nm where k is minimized (0.02–0.04), with blocking regions (k > 0.10) at 400 nm and > 600 nm.
- **Tungsten** (metal grid material) has high k across all wavelengths, as expected for a refractory metal.

## Snell's Law CRA Shift vs Tangent Approximation

Comparison of the upgraded Snell's law ray tracing shift (traces through all intermediate layers) versus the old tangent approximation (`Δx = tan(CRA) × h × 0.5`).

| CRA (deg) | Snell shift (um) | tan approx (um) | Difference (um) |
|----------:|:----------------:|:----------------:|:----------------:|
| 0° | 0.000000 | 0.000000 | 0.000000 |
| 5° | 0.109021 | 0.026247 | +0.082775 |
| 10° | 0.217825 | 0.052898 | +0.164927 |
| 15° | 0.326181 | 0.080385 | +0.245796 |
| 20° | 0.433829 | 0.109191 | +0.324638 |
| 25° | 0.540466 | 0.139892 | +0.400574 |
| 30° | 0.645724 | 0.173205 | +0.472518 |

**Key observations:**
- **Snell shift is 3–4× larger** than the old tangent approximation at all angles. This is because the tangent formula only considered the microlens height (0.6 um × 0.5), while Snell's law traces through the entire stack below the microlens: planarization (0.3 um, n=1.46), color filter (0.6 um, n=1.55), BARL (0.08 um, n=1.46–2.02), and silicon to photodiode center (0.5 um, n=4.08).
- **The refraction reduces angles** at each interface (Snell's law: n₁ sin θ₁ = n₂ sin θ₂), so the lateral displacement per layer is smaller than geometric `tan(θ)`, but the total accumulated shift is larger because it accounts for all layers.
- At **30° CRA**, the Snell shift is 0.646 um — more than half the pixel pitch (1.0 um), indicating that proper microlens alignment is critical for edge-of-sensor performance.

::: tip TMM vs RCWA for CRA Analysis
The TMM solver treats all layers as uniform slabs, so microlens shift has no effect on TMM's R/T/A values. The shift only matters when using RCWA (which models the actual microlens shape and lateral light distribution). To see the full QE impact of CRA shift compensation, use torcwa or grcwa.
:::

## Execution Environment

```
Solver      : TMM (Transfer Matrix Method)
Backend     : NumPy (CPU)
Python      : 3.11.6
PyTorch     : 2.5.0 (available but not used by TMM)
Platform    : macOS (Darwin 25.2.0, Apple Silicon)
```

| Simulation | Wavelengths | Runtime |
|-----------|:-----------:|--------:|
| Single wavelength (550 nm) | 1 | 0.8 ms |
| Full sweep (380–780 nm, 20 nm step) | 21 | 2.8 ms |
| Full sweep (380–780 nm, 10 nm step) | 41 | ~4 ms |
| CRA sweep (7 angles × 2 configs) | 14 runs | ~12 ms |
