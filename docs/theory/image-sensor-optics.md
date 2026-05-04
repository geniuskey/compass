# Image Sensor Optics

::: tip 선수 지식 | Prerequisites
[CMOS 이미지 센서란?](/introduction/what-is-cmos-sensor) → [픽셀 해부학](/introduction/pixel-anatomy) → 이 페이지
:::

This page describes the optical structure of a backside-illuminated (BSI) CMOS image sensor pixel, which is the primary simulation target of COMPASS.

## BSI pixel architecture

In a BSI pixel, light enters through the silicon backside (opposite to the wiring side). The pixel stack, from top (light-entry side) to bottom, consists of:

```
              Incident light
                   |
                   v
    +---------------------------------+
    |             Air                  |
    +---------------------------------+
    |          Microlens               |   Focuses light into pixel center
    +---------------------------------+
    |       Planarization (SiO2)       |   Uniform dielectric
    +---------------------------------+
    |  Color Filter (Bayer pattern)    |   Wavelength-selective absorption
    |  + Metal Grid (W)               |   Optical isolation between pixels
    +---------------------------------+
    |   BARL (anti-reflection stack)   |   Minimizes reflection at CF/Si
    +---------------------------------+
    |         Silicon                  |   Absorbs photons, generates e-h pairs
    |   [Photodiode regions]           |   Collects charge
    |   [DTI trenches]                 |   Prevents optical/electrical crosstalk
    +---------------------------------+
```

<StackVisualizer />

## Microlens

The microlens is a curved polymer structure that focuses incoming light toward the center of the pixel. In COMPASS, microlenses are modeled as superellipse profiles:

$$h(x,y) = H \left(1 - \left|\frac{x - c_x}{r_x}\right|^n - \left|\frac{y - c_y}{r_y}\right|^n \right)^{1/\alpha}$$

Parameters:
- $H$: lens height (typical: 0.4-0.8 um)
- $r_x, r_y$: semi-axes (typical: slightly less than half the pitch)
- $n$: squareness parameter (2.0 = ellipse, higher = more square)
- $\alpha$: curvature control

The microlens center can be shifted to account for the **Chief Ray Angle (CRA)** -- the angle at which light arrives from the camera lens. Pixels at the edge of the image sensor receive light at a larger CRA, so their microlenses must be shifted to maintain good light collection.

## Color filter array

The color filter array (CFA) selectively absorbs light to create color sensitivity. The most common pattern is the **Bayer RGGB** arrangement:

```
  +---+---+
  | R | G |
  +---+---+
  | G | B |
  +---+---+
```

Each color filter material has a wavelength-dependent complex refractive index with absorption ($k > 0$) outside its passband and low absorption within the passband. The filters absorb unwanted wavelengths while transmitting the target color.

<BayerPatternViewer />

The **metal grid** between color filter sub-pixels (typically tungsten, 40-80 nm wide) provides optical isolation, preventing light from leaking between adjacent color channels.

## BARL: Bottom Anti-Reflection Layer

The BARL is a thin multi-layer dielectric stack at the interface between the color filter and silicon. Its purpose is to minimize reflection at this high-contrast interface.

Without BARL, the reflection at a color-filter/silicon interface ($n \approx 1.55$ to $n \approx 4.0$) is:

$$R = \left(\frac{n_\text{Si} - n_\text{CF}}{n_\text{Si} + n_\text{CF}}\right)^2 \approx 20\%$$

A well-designed BARL stack can reduce this to under 5% across the visible spectrum. The specific materials, layer count, and stacking order vary by vendor — there is no single canonical recipe.

## Silicon and photodiode

Silicon is the absorbing medium where photon-to-electron conversion occurs. The absorption depth depends strongly on wavelength:

| Wavelength | Color | Absorption depth in Si |
|------------|-------|------------------------|
| 400 nm | Violet | ~0.1 um |
| 450 nm | Blue | ~0.4 um |
| 550 nm | Green | ~1.7 um |
| 650 nm | Red | ~3.3 um |
| 800 nm | NIR | ~10 um |

This means blue light is absorbed near the surface while red/NIR light requires several micrometers of silicon. Typical BSI pixel silicon thickness is 2-4 um.

The **photodiode** occupies a defined region within the silicon. Only photons absorbed within the photodiode volume contribute to the photocurrent. COMPASS models this by integrating the absorbed power within the photodiode bounding box.

## Deep Trench Isolation (DTI)

DTI is a vertical trench filled with a low-index material (typically SiO2, $n \approx 1.46$) that optically isolates adjacent pixels within the silicon. The large index contrast between silicon ($n \approx 3.5-4.0$) and SiO2 causes total internal reflection at the trench walls, preventing light from crossing into neighboring pixels.

DTI is critical for:
- Reducing optical crosstalk
- Improving color fidelity
- Maintaining MTF (modulation transfer function)

## Optical phenomena in BSI pixels

| Effect | Mechanism | Impact on QE |
|--------|-----------|--------------|
| Thin-film interference | Multi-beam interference in BARL/planarization | Spectral ripple |
| Diffraction | Sub-wavelength metal grid | Angle-dependent light redistribution |
| Waveguide modes | DTI creates a silicon waveguide | Traps light, can enhance or degrade QE |
| Microlens focusing | Refraction | Concentrates light in pixel center |
| Optical crosstalk | Light leaking to adjacent pixels | Reduces color accuracy |
| Total internal reflection | High-index Si / low-index surroundings | Traps light, increases effective path length |

All of these effects are captured automatically by the full-wave EM solvers (RCWA, FDTD) in COMPASS.


## Optical characterization theory for image sensors

To make useful design decisions, sensor optics must be understood through both **structure** and **measurement theory**. The frameworks below connect stack choices directly to measurable outcomes.

### 1) Spectral metrics

- **External Quantum Efficiency (EQE)**
  $$\mathrm{EQE}(\lambda,\theta,\phi)=\frac{N_e}{N_{ph}}$$
- **Internal Quantum Efficiency (IQE)**
- **Responsivity**
  $$R_i(\lambda)=\frac{q\,\mathrm{EQE}(\lambda)}{h c/\lambda}\quad [\mathrm{A/W}]$$

Interpretation:
- High IQE with low EQE usually indicates front-side optical losses (reflection, filter loss, parasitic stack absorption).
- Significant overlap between R/G/B response curves increases metameric error risk and degrades channel separation.

### 2) Angular metrics (CRA dependence)

- **Angular QE map**: $\mathrm{QE}(\lambda,\theta,\phi)$
- **CRA roll-off**: relative sensitivity drop from on-axis to off-axis incidence
- **Azimuthal asymmetry**: orientation-dependent response due to anisotropic microlens/grid geometry

A practical sweep window is typically 0°-30° (or equivalent to the lens F/# cone).

### 3) Spatial metrics (resolution and leakage)

- **Optical crosstalk matrix** (channel-to-channel leakage)
- **PSF / LSF / MTF**
- **Ensquared energy** in the target photodiode area

Crosstalk reduction and MTF improvement are often competing objectives; DTI and grid changes can improve one while degrading the other in specific wavelength/angle regimes.

### 4) Loss-budget decomposition

$$\eta_{\mathrm{sys}} \approx \eta_{\mathrm{surf}}\cdot\eta_{\mathrm{CF}}\cdot\eta_{\mathrm{stack}}\cdot\eta_{\mathrm{Si\,abs}}\cdot\eta_{\mathrm{PD\,collect}}$$

This decomposition helps identify which layer dominates loss at each wavelength.

## Suggested additions to reach textbook depth

1. **Wave-optics foundations**: complex index, polarization, coherence, phase delay in multilayers.
2. **Stack-level design rules**: microlens offset vs CRA, CF thickness tuning, BARL quarter-wave intuition vs multi-objective optimization.
3. **Pixel scaling laws**: sub-1 um diffraction/near-field behavior and QE-MTF-crosstalk tradeoffs.
4. **Standardized evaluation protocol**: wavelength/angle sampling, ROI definitions, normalization and reporting conventions.
5. **Simulation-to-measurement correlation**: fitting flow for EQE/angular data with material and geometry uncertainty handling.
6. **Failure-mode diagnostics**: energy non-conservation, boundary reflections, mesh dispersion, misleading but numerically stable outputs.
7. **Case studies**: parameter-sweep examples (DTI width, BARL thickness, microlens offset).

## Practical execution plan

- **Phase 1 (Definitions):** Lock metric definitions and notation on this page.
- **Phase 2 (Tool mapping):** Add explicit links from each metric to relevant `/simulator` pages.
- **Phase 3 (Quantitative examples):** Embed COMPASS-based plots/tables for spectral, angular, and crosstalk outputs.
- **Phase 4 (Reproducible protocol):** Publish fixed benchmark conditions (mesh, BCs, normalization, convergence thresholds).
- **Phase 5 (Advanced appendix):** Add polarization/coherence/process-variation sensitivity and uncertainty budgeting.


## High-impact additions to prioritize next

1. **Unified definitions table**
   - QE, EQE, IQE, responsivity, transmittance, and absorptance in one canonical reference.
2. **Measurement-to-simulation alignment example**
   - Overlay measured and simulated EQE under identical wavelength/angle/polarization normalization.
3. **Channel crosstalk matrix walkthrough**
   - Provide a 3x3 leakage example and how to interpret off-diagonal penalties.
4. **CRA design rule cards**
   - Practical sweep ranges for microlens offset from center to edge pixels plus common failure signatures.
5. **Numerical confidence checklist**
   - Mesh, boundary condition, Fourier order, and energy-conservation checks in one quick QA page.

## Authoring template for each subsection

- **Problem statement**: Why this metric matters.
- **Core definition**: At least one equation and units.
- **Interpretation**: What high/low values imply physically.
- **Common pitfalls**: 2-3 frequent misinterpretations.
- **COMPASS practice links**: 1-2 simulator/cookbook pages.
- **Validation criterion**: Reproducibility threshold (e.g., sweep delta < 1%).


## Unified notation and definitions (quick reference)

| Term | Symbol | Definition | Unit | Notes |
|---|---|---|---|---|
| Quantum Efficiency | QE | Collected electrons divided by incident photons (generic usage) | - | Often clarified into EQE/IQE |
| External Quantum Efficiency | EQE | $$N_e / N_{ph,inc}$$ | - | Includes reflection/transmission/parasitic optical losses |
| Internal Quantum Efficiency | IQE | $$N_e / N_{ph,abs}$$ | - | Conditions on absorbed photons only |
| Responsivity | $R_i(\lambda)$ | $$I_{ph}/P_{opt}=q\,EQE/(hc/\lambda)$$ | A/W | Convenient for electro-optical budgeting |
| Transmittance | $T$ | Transmitted optical power over incident power | - | Layer or stack dependent |
| Reflectance | $R$ | Reflected optical power over incident power | - | Angle and polarization dependent |
| Absorptance | $A$ | Absorbed optical power over incident power | - | For passive stacks: $A+R+T=1$ |

## 3x3 crosstalk matrix interpretation example

Define each element as leakage from source channel (column) to detected channel (row):

$$
\mathbf{C}=\begin{bmatrix}
0.91 & 0.05 & 0.02\\
0.06 & 0.90 & 0.07\\
0.03 & 0.05 & 0.91
\end{bmatrix}
$$

Interpretation checklist:
- Diagonal terms ($C_{RR},C_{GG},C_{BB}$) represent in-channel retention; higher is better.
- Off-diagonal terms quantify spectral-spatial leakage and directly correlate with color-mixing artifacts.
- A practical target is to maximize diagonal terms while controlling the largest off-diagonal entry under CRA sweep.
- Report this matrix together with wavelength and incident-angle conditions; otherwise values are not comparable.


## Measurement-to-simulation EQE alignment example

The purpose of alignment is to separate **model error** from **condition mismatch**.

### Fixed-condition setup

- Wavelength grid: 400-700 nm, 10 nm step
- Incident cone: CRA 0° (on-axis) first, then CRA sweep (0°-30°)
- Polarization: TE/TM separated, then unpolarized average
- Normalization: incident photon flux at the sensor entrance plane
- ROI: photodiode active area identical to mask definition

### Recommended alignment workflow

1. **Lock optical constants**: use the same material $n,k$ dataset for fitting and reporting.
2. **Match geometry reference**: align nominal thicknesses and include known process offsets.
3. **Fit low-sensitivity parameters first**: planarization and CF thickness before microlens/DTI fine tuning.
4. **Use two-stage objective**:
   - Stage A: minimize spectral RMSE on on-axis EQE.
   - Stage B: minimize angular RMSE while constraining on-axis degradation.
5. **Report residuals by region**: blue/green/red bands and off-axis bins separately.

### Minimal report table (example)

| Metric | Value (example) | Note |
|---|---:|---|
| On-axis EQE RMSE (400-700 nm) | 0.018 | After Stage A |
| Angular EQE RMSE (0°-30°) | 0.026 | After Stage B |
| Max channel bias | 2.1 %p | Worst among R/G/B |
| Largest off-axis error bin | 25°-30° | CRA edge behavior |

This format makes regressions traceable when materials, geometry, or solver settings change.
