---
title: Sample Pixel Structures
description: Representative recent-generation CMOS image sensor pixel structures with ready-to-run COMPASS configs and a methodology for determining the layer parameters that are not publicly disclosed.
---

# Sample Pixel Structures

This guide documents a set of representative CMOS image sensor pixel architectures spanning the 0.56–1.6 µm pitch range and shows how to simulate them in COMPASS. Each sample ships as a ready-to-run YAML config under `configs/pixel/`, and a Python helper (`compass.geometry.derive_parameters`) generates a complete config from a small set of headline inputs.

## Why filling in a pixel config is non-trivial

Public information about modern flagship pixels is typically limited to *headline* parameters — pixel pitch, color-filter binning, OCL sharing, and the marketing names of structural innovations (split-substrate transistors, LOFIC, 4×4 super-cell binning). The internal numbers needed for an electromagnetic simulation are almost never disclosed:

- microlens sag / radius / material refractive index
- color-filter, planarization, BARL stack thicknesses
- DTI trench width, depth, and fill material
- photodiode lateral footprint and z-extent

COMPASS supplies physically reasonable defaults for these via empirical scaling rules (Section [Parameter determination](#parameter-determination)).

## Sample pixel structures

| Key                         | Config file                          | Pitch    | CFA pattern        | OCL sharing | Notable feature                                                |
|-----------------------------|--------------------------------------|----------|--------------------|-------------|----------------------------------------------------------------|
| `sample_p0p56um_4x4ocl`    | `sample_p0p56um_4x4ocl.yaml`        | 0.56 µm | 4×4 super-cell     | 1           | Sub-µm pixel, high-RI microlens (n ≈ 1.70), F-DTI SiO₂           |
| `sample_p1p0um_quadbayer`  | `sample_p1p0um_quadbayer.yaml`      | 1.0 µm  | Quad Bayer (2×2)   | 1           | Standard 50-MP-class main-camera pixel, F-DTI SiO₂              |
| `sample_p1p6um_split_pd`   | `sample_p1p6um_split_pd.yaml`       | 1.6 µm  | Standard Bayer     | 1           | Split-substrate transistors → enlarged photodiode (~2× volume) |
| `sample_p1p22um_2x2ocl`    | `sample_p1p22um_2x2ocl.yaml`        | 1.22 µm | Quad Bayer         | 2           | One large microlens per 2×2 same-color group (all-pixel PDAF)   |
| `sample_p1p2um_lofic`      | `sample_p1p2um_lofic.yaml`          | 1.2 µm  | Quad Bayer         | 2           | LOFIC HDR — capacitor reduces PD footprint, 2×2 OCL              |

Run any of them with Hydra:

```bash
python scripts/run_simulation.py pixel=sample_p0p56um_4x4ocl source=wavelength_sweep
python scripts/run_simulation.py pixel=sample_p1p6um_split_pd solver=torcwa
python scripts/run_simulation.py pixel=sample_p1p2um_lofic source=cone_illumination
```

### `sample_p0p56um_4x4ocl` — sub-µm pixel with 4×4 binning

Smallest-pitch sample. Distinctive optical features:

- **Pitch = 0.56 µm**.
- **4×4 same-color super-cell** color filter (16-cell binning); 8×8 unit cell.
- **High-refractive-index microlens** modelled via the `polymer_hri_n1p70` (TiO₂-doped polymer, n ≈ 1.70) entry registered in `MaterialDB`.
- **F-DTI with SiO₂** fill (oxide-fill DTI for lower crosstalk).

### `sample_p1p0um_quadbayer` — 1.0 µm Quad Bayer

Generic 50-MP-class main-camera pixel. Quad Bayer + per-pixel OCL + F-DTI SiO₂.

### `sample_p1p6um_split_pd` — 1.6 µm split-substrate pixel

Large-pitch sample with photodiode and pixel transistors placed on different substrate layers, freeing up area for the photodiode and approximately doubling the saturation signal level.

Optical model: the photodiode now occupies most of the pixel footprint laterally and extends deeper, both modelled via a larger `photodiode.size` in `silicon`.

### `sample_p1p22um_2x2ocl` — 2×2 OCL Quad Bayer

Phone main-camera-class pixel with **2×2 on-chip lens** sharing: a single microlens covers a 2×2 same-color group, enabling all-pixel phase-detection autofocus. Modelled with `microlens.sharing: 2`.

### `sample_p1p2um_lofic` — LOFIC HDR

50-MP-class, 1.2 µm. **LOFIC** (Lateral Overflow Integration Capacitor) consumes part of the in-pixel silicon real-estate for HDR, which we model as a slightly smaller photodiode footprint. Quad PD adds 2×2 OCL on top.

## Parameter determination

There are three complementary approaches to filling in the layer parameters that are not publicly disclosed.

### 1. Empirical scaling rules (default)

`compass.geometry.derive_parameters` ships pitch-based scaling rules derived from public ISSCC, IEDM, SPIE pixel-architecture papers (2018–2024) and reverse-engineering cross-section reports for sub-µm to 2 µm BSI pixels:

```python
from compass.geometry import derive_parameters, PixelStack

cfg = derive_parameters(sample="sample_p0p56um_4x4ocl", cra_deg=20.0)
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
| Photodiode footprint         | `0.70·pitch` (`0.88·pitch` for split-PD, `0.65·pitch` for LOFIC) |
| Photodiode z-extent          | `0.67·t_Si` (`0.85·t_Si` for split-PD)              |

Architecture flags (`split_pd`, `lofic`) drive the photodiode geometry adjustments. These defaults reproduce typical published cross-sections to within ~30%.

### 2. Calibration to measured QE / crosstalk

If you have measured QE(λ) per color or measured spatial crosstalk, fit the unknown parameters with the optimization framework:

```python
from compass.geometry import derive_parameters
from compass.optimization import (
    ParameterSpace, MicrolensHeight, BARLThicknesses, Optimizer
)

base = derive_parameters(sample="sample_p1p0um_quadbayer", cra_deg=0.0)

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

### 3. Direct measurement (SEM / die-shot cross-section)

When a die-shot SEM image is available, measure layer thicknesses directly and override the heuristic via the `overrides` argument:

```python
cfg = derive_parameters(
    sample="sample_p0p56um_4x4ocl",
    overrides={
        "layers.silicon.thickness": 1.85,      # measured Si thickness
        "layers.color_filter.thickness": 0.42, # measured CF thickness
        "layers.silicon.dti.width": 0.075,
    },
)
```

`overrides` accepts dotted keys at any depth and wins over the heuristic.

## Sample headlines reference

The headline values are stored in `compass.geometry.SAMPLE_HEADLINES` and can be used as the entry point for `derive_parameters`:

```python
>>> from compass.geometry import SAMPLE_HEADLINES
>>> SAMPLE_HEADLINES["sample_p0p56um_4x4ocl"]
{'pitch': 0.56, 'megapixels': 200, 'cf_pattern': 'tetra2cell',
 'ocl_sharing': 1, 'microlens_material': 'polymer_hri_n1p70',
 'dti_fill': 'sio2', 'year': 2024}
```

## References

The sample structures above are inspired by publicly-available descriptions of recent commercial CIS products. The configs are intentionally generic — values are derived from the empirical scaling rules in this guide rather than from any single vendor process — but the following links provide background reading for the technologies they illustrate:

- 0.56 µm sub-µm pixel with 4×4 super-cell binning and high-RI microlens — see [Samsung ISOCELL HP9 announcement (Jun 2024)](https://news.samsung.com/global/samsung-unveils-versatile-image-sensors-for-superior-smartphone-photography) and the [Samsung Image Sensor product pages](https://semiconductor.samsung.com/image-sensor/).
- 1.0 µm Quad Bayer + F-DTI oxide fill — see the [Samsung ISOCELL GN-series product pages](https://semiconductor.samsung.com/image-sensor/mobile-image-sensor/isocell-gn/) and Samsung Newsroom (Feb 2020), *Reveals the Tech Behind Its New 108 MP Nonacell Image Sensor*.
- 1.6 µm split-substrate pixel — see [Sony Semiconductor Solutions — 2-Layer Transistor pixel technology](https://www.sony-semicon.com/en/technology/lsi/2layer-transistor-pixel.html) (IEDM 2021).
- 1.22 µm 2×2 OCL Quad Bayer — see [Sony Semiconductor Solutions — All-pixel autofocus / 2×2 OCL technology](https://www.sony-semicon.com/en/technology/mobile/index.html).
- 1.2 µm LOFIC HDR — see [OmniVision OV50K40 press release (Mar 2024)](https://www.ovt.com/news-events/product-releases/) and the OmniVision TheiaCel technology brief.
- Public TechInsights summaries of recent flagship CIS cross-sections (cross-section thickness / DTI width references).
- Hwang & Kim, *Sensors* 23, 702 (2023) — Snell-law CRA shift used for `auto_cra`.
