---
title: Thin Film Stack Designer
---

# Thin Film Stack Designer

Design arbitrary multilayer thin film coatings with full control over layer count, order, materials, and thickness. Visualize both reflectance and transmittance spectra in real time.

<BarlOptimizer />

## Supported Materials (20+)

| Category | Materials |
|----------|-----------|
| Low-n dielectrics | SiO₂, MgF₂, CaF₂, Al₂O₃ |
| High-n dielectrics | Si₃N₄, HfO₂, TiO₂, Ta₂O₅, ZrO₂, Nb₂O₅, AlN |
| Semiconductors | Silicon, Ge, SiC, ZnS, ZnSe |
| Conductors / TCO | Tungsten, ITO |
| Polymers / Glass | Polymer (n=1.56), BK7 Glass |

## Built-in Presets

- **BARL 4-layer**: Default BSI pixel anti-reflection stack (SiO₂/HfO₂/SiO₂/Si₃N₄)
- **2-layer AR**: Simple MgF₂/TiO₂ anti-reflection coating
- **Broadband AR**: 4-layer design for wide spectral coverage
- **HR Blue mirror**: 6-layer high-reflectance stack for blue wavelengths
- **NIR-cut filter**: 5-layer near-infrared rejection filter on glass

::: tip
Use the "Auto Optimize" button to find thickness combinations that minimize average reflectance in your target wavelength band. You can freely add, remove, and reorder layers before optimizing.
:::
