---
title: BARL Optimizer
---

# BARL Optimizer

The Bottom Anti-Reflection Layer (BARL) is a critical multilayer coating between the color filter and silicon. Its purpose is to minimize reflection at the silicon interface, maximizing the light that reaches the photodiode.

<BarlOptimizer />

## Design Principles

A well-designed BARL uses quarter-wave interference to cancel reflections. The default BSI pixel uses a 4-layer stack:

| Layer | Material | Default | Role |
|-------|----------|---------|------|
| 1 | SiO₂ | 10 nm | Low-n spacer |
| 2 | HfO₂ | 25 nm | High-n matching |
| 3 | SiO₂ | 15 nm | Phase tuning |
| 4 | Si₃N₄ | 30 nm | Index gradient to CF |

::: tip
Use the "Auto Optimize" button to find the thickness combination that minimizes average reflectance in your target wavelength band.
:::
