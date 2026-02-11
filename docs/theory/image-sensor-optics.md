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

A well-designed BARL stack (e.g., SiO2/HfO2/Si3N4) can reduce this to under 5% across the visible spectrum.

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
