---
title: Vendor Pixel Structures
description: Recent (2022–2024) flagship CMOS image sensor pixel structures from Samsung, Sony, and OmniVision, with ready-to-run COMPASS configs and a methodology for determining the layer parameters that vendors do not publicly disclose.
---

# Vendor Pixel Structures

This guide documents publicly-known pixel architectures from the three leading CMOS image sensor vendors over the last three years and shows how to simulate them in COMPASS. Each architecture ships as a ready-to-run YAML config under `configs/pixel/`, and a Python helper (`compass.geometry.derive_parameters`) generates a complete config from a small set of vendor-disclosed inputs.

## Why a vendor config is non-trivial

Vendors publish *headline* parameters in datasheets and press releases — pixel pitch, color-filter binning, OCL sharing, and the marketing names of structural innovations (2-Layer Transistor, TheiaCel, Tetra²pixel). The internal numbers needed for an electromagnetic simulation are almost never disclosed:

- microlens sag / radius / material refractive index
- color-filter, planarization, BARL stack thicknesses
- DTI trench width, depth, and fill material
- photodiode lateral footprint and z-extent

COMPASS supplies physically reasonable defaults for these via empirical scaling rules (Section [Parameter determination](#parameter-determination)).

## Supported flagship sensors

| Vendor / model                          | Config file                                | Pitch    | CFA pattern        | OCL sharing | Notable feature                                  |
|-----------------------------------------|--------------------------------------------|----------|--------------------|-------------|--------------------------------------------------|
| Samsung **ISOCELL HP9** (2024)          | `samsung_hp9_0p56um.yaml`                  | 0.56 µm | Tetra²pixel (4×4)  | 1           | High-RI microlens, F-DTI SiO₂                    |
| Samsung **GNJ-class 50 MP** (2024)      | `samsung_gnj_50mp_1p0um.yaml`              | 1.0 µm  | Quad Bayer (2×2)   | 1           | F-DTI with SiO₂ fill                              |
| Sony **LYTIA LYT-900** (2024)           | `sony_lyt900_1p6um.yaml`                   | 1.6 µm  | Standard Bayer     | 1           | 2-Layer Transistor (PD volume ≈ 2×)              |
| Sony **2×2 OCL Quad** (IMX803-class)    | `sony_2x2ocl_quad_1p22um.yaml`             | 1.22 µm | Quad Bayer         | 2           | One large lens per 2×2 same-color group          |
| OmniVision **OV50K40** (2024)           | `omnivision_ov50k40_1p2um.yaml`            | 1.2 µm  | Quad Bayer         | 2           | TheiaCel (LOFIC) + Quad PD 2×2 OCL                |

Run any of them with Hydra:

```bash
python scripts/run_simulation.py pixel=samsung_hp9_0p56um source=wavelength_sweep
python scripts/run_simulation.py pixel=sony_lyt900_1p6um solver=torcwa
python scripts/run_simulation.py pixel=omnivision_ov50k40_1p2um source=cone_illumination
```

### Samsung ISOCELL HP9 — 200 MP, 0.56 µm, Tetra²pixel

The HP9 is the first 200 MP telephoto sensor for smartphones. Distinctive optical features modelled here:

- **Pitch = 0.56 µm**, 1/1.4″ format.
- **Tetra²pixel** color filter (announced as "Hexadeca" / 16-cell binning): 4×4 same-color group, 8×8 super-pixel.
- **High-refractive-index microlens** ("new material" announced for HP9). Modelled via the `polymer_hri_n1p70` (TiO₂-doped polymer, n ≈ 1.70) entry registered in `MaterialDB`.
- **F-DTI with SiO₂** fill (Samsung's GNJ generation upgraded the trench fill from polysilicon to oxide for lower crosstalk).

### Samsung GNJ-class 50 MP — 1.0 µm

Generic 50 MP main-camera pixel (GN3/GNJ generation). Quad Bayer + per-pixel OCL + F-DTI SiO₂.

### Sony LYTIA LYT-900 — 1.6 µm, 2-Layer Transistor

Sony's first 1-inch 50 MP smartphone sensor with the **2-Layer Transistor (2LT)** stacked CMOS technology presented at IEDM 2021. Photodiodes and pixel transistors are placed on different substrate layers, freeing up area for the photodiode and approximately doubling the saturation signal level.

Optical model: the photodiode now occupies most of the pixel footprint laterally and extends deeper, both modelled via larger `photodiode.size` in `silicon`.

### Sony 2×2 OCL Quad Bayer (IMX803-class)

Phone main-camera sensor with **2×2 on-chip lens** technology: a single microlens covers a 2×2 same-color group, enabling all-pixel phase-detection autofocus. Modelled with `microlens.sharing: 2` so each cluster of four pixels shares one large lens.

### OmniVision OV50K40 — TheiaCel HDR

50 MP, 1.2 µm. The **TheiaCel** brand name is OmniVision's implementation of a Lateral Overflow Integration Capacitor (LOFIC). Optically, the LOFIC capacitor consumes part of the in-pixel silicon real-estate, which we model as a slightly smaller photodiode footprint. Quad PD adds 2×2 OCL on top.

## Parameter determination

There are three complementary approaches to filling in the layer parameters that vendors do not disclose.

### 1. Empirical scaling rules (default)

`compass.geometry.derive_parameters` ships pitch-based scaling rules derived from public ISSCC, IEDM, SPIE pixel-architecture papers (2018–2024) and TechInsights cross-section reports for sub-µm to 2 µm BSI pixels:

```python
from compass.geometry import derive_parameters, PixelStack

cfg = derive_parameters(vendor="samsung_hp9", cra_deg=20.0)
stack = PixelStack({"pixel": cfg})
```

The rules currently used are:

| Quantity                     | Default rule                                        |
|------------------------------|-----------------------------------------------------|
| Microlens sag                | `min(0.95, 0.42·pitch + 0.20)` µm                   |
| Microlens semi-axis          | `(sharing·pitch − 2·gap) / 2` µm                    |
| Microlens gap                | `0.02 + 0.02·min(pitch, 2)` µm                      |
| Color-filter thickness       | piecewise: `0.35 + 0.25·(pitch/0.7)` for sub-µm; `min(0.90, 0.45·pitch + 0.15)` otherwise |
| Planarization thickness      | `0.20 + 0.10·min(pitch, 2)/2` µm                    |
| DTI trench width             | `0.05 + 0.025·min(pitch, 2)` µm (process-limited)   |
| Silicon epi thickness        | `min(4.5, 1.4 + 1.5·pitch)` µm                      |
| Photodiode footprint         | `0.70·pitch` (`0.88·pitch` for 2LT, `0.65·pitch` for LOFIC) |
| Photodiode z-extent          | `0.67·t_Si` (`0.85·t_Si` for 2LT)                   |

Architecture flags (`two_layer_transistor`, `lofic`) drive the photodiode geometry adjustments. These defaults reproduce the cross-sections of public TechInsights reports to within ~30%.

### 2. Calibration to measured QE / crosstalk

If you have measured QE(λ) per color or measured spatial crosstalk, fit the unknown parameters with the optimization framework:

```python
from compass.geometry import derive_parameters
from compass.optimization import (
    ParameterSpace, MicrolensHeight, BARLThicknesses, Optimizer
)

base = derive_parameters(vendor="samsung_gnj", cra_deg=0.0)

space = ParameterSpace([
    MicrolensHeight(min=0.4, max=0.8),
    BARLThicknesses(min=0.005, max=0.05, n_layers=4),
])

# Custom objective: L2 distance to measured QE curves.
optimizer = Optimizer(method="L-BFGS-B", parameter_space=space)
best = optimizer.minimize(
    objective=lambda params: l2_to_measured(params, measured_qe),
    base_config=base,
)
```

This is the recommended workflow when matching a real product.

### 3. Direct measurement (TechInsights / SEM cross-section)

When a die-shot SEM image is available, measure layer thicknesses directly and override the heuristic via the `overrides` argument:

```python
cfg = derive_parameters(
    vendor="samsung_hp9",
    overrides={
        "layers.silicon.thickness": 1.85,      # measured Si thickness
        "layers.color_filter.thickness": 0.42, # measured CF thickness
        "layers.silicon.dti.width": 0.075,
    },
)
```

`overrides` accepts dotted keys at any depth and wins over the heuristic.

## Vendor headlines reference

The vendor-disclosed headline values are stored in `compass.geometry.VENDOR_HEADLINES` and can be used as the entry point for `derive_parameters`:

```python
>>> from compass.geometry import VENDOR_HEADLINES
>>> VENDOR_HEADLINES["samsung_hp9"]
{'pitch': 0.56, 'format': '1/1.4"', 'megapixels': 200,
 'cf_pattern': 'tetra2cell', 'ocl_sharing': 1,
 'microlens_material': 'polymer_hri_n1p70', 'dti_fill': 'sio2',
 'year': 2024}
```

## Sources

- Samsung Newsroom (Jun 2024), *Samsung Unveils Versatile Image Sensors for Superior Smartphone Photography* — HP9 announcement.
- Samsung Semiconductor product pages — ISOCELL HP9, GN-series.
- Samsung Newsroom (Feb 2020), *Reveals the Tech Behind Its New 108MP Nonacell Image Sensor*.
- Sony Semiconductor Solutions, *2-Layer Transistor Pixel* technology page.
- Sony Semiconductor Solutions, *All-pixel AF / 2×2 OCL* technology page.
- OmniVision press release (Mar 2024), *OV50K40 with TheiaCel*.
- Public TechInsights summaries of recent flagship CIS cross-sections.
- Hwang & Kim, *Sensors* 23, 702 (2023) — Snell-law CRA shift used for `auto_cra`.
