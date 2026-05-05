---
title: COMPASS Guide
description: Practical workflow guide for installing COMPASS, configuring CMOS image sensor pixels, choosing solvers, and running validated simulations.
---

# Guide

Use this section when you are ready to run COMPASS rather than study the physics. It connects installation, pixel-stack configuration, solver execution, validation, and cookbook-style recipes into one practical workflow.

## Pick a Starting Point

| Goal | Start here | Continue with |
|---|---|---|
| Install COMPASS | [Installation](./installation.md) | [Quick Start](./quickstart.md) |
| Run a first pixel simulation | [First Simulation](./first-simulation.md) | [Visualization](./visualization.md) |
| Configure a BSI pixel | [Pixel Stack Config](./pixel-stack-config.md) | [Sample Pixel Structures](./sample-pixels.md) |
| Choose a solver | [Choosing a Solver](./choosing-solver.md) | [Cross-Validation](./cross-validation.md) |
| Tune numerical settings | [Convergence Study](/cookbook/convergence-study) | [Solver Comparison](/cookbook/solver-benchmark) |
| Move toward system metrics | [Signal Simulation](./signal-simulation.md) | [Signal Chain Color Accuracy](/cookbook/signal-chain-color-accuracy) |

## How the Guide is Organized

1. **Getting Started** gets COMPASS installed and runs a minimal simulation.
2. **Configuration** explains YAML files, material data, pixel stacks, and solver selection.
3. **Running Solvers** covers RCWA, FDTD, and cross-validation workflows.
4. **Advanced** covers cone illumination, signal simulation, ROI sweeps, inverse design, visualization, and troubleshooting.
5. **Recipes** are task-focused cookbook pages for common design and validation jobs.

::: tip
If a term in this guide feels unfamiliar, use [Theory](/theory/) for the physics background and come back here for the runnable workflow.
:::
