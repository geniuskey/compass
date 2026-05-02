# Optical Simulation Chapter Overview

This chapter explains the numerical methods COMPASS uses to solve Maxwell's equations for an image sensor pixel. It covers *how* a solver actually computes reflection, transmission, and absorption — not the optics or sensor architecture, which are covered in the previous chapters.

## Why we need numerical methods

Maxwell's equations have closed-form solutions only for trivial geometries (a planar multilayer, a single sphere). A real BSI pixel has 3D periodic structures — color filter grids, sub-wavelength microlenses, DTI walls — for which no analytical solution exists. We discretize the problem and solve it on a computer.

There are three families of methods worth distinguishing:

| Family | Domain | What it discretizes | COMPASS solver(s) |
|---|---|---|---|
| **Analytical / TMM** | 1D planar | Closed-form transfer matrices per layer | `tmm` |
| **Modal / RCWA** | Frequency domain, periodic | Permittivity in Fourier basis, fields as Bloch modes | `torcwa`, `grcwa`, `meent`, `fmmax` |
| **Time domain / FDTD** | Time domain, arbitrary geometry | Yee-grid finite differences in space and time | `fdtd_flaport`, `fdtdz`, `fdtdx`, `meep` |

### A note on permittivity

All solvers internally represent materials by their complex permittivity $\varepsilon = \tilde{n}^2 = (n + ik)^2$. COMPASS stores material data as $(n, k)$ pairs in `MaterialDB` and converts to $\varepsilon$ when building the simulation geometry. RCWA solvers expand $\varepsilon(x, y)$ in a 2D Fourier series; FDTD solvers sample $\varepsilon$ on the Yee grid; TMM uses $\varepsilon$ per layer directly.

## Choosing a solver

A rough decision tree:

- **1D planar stack** (no lateral structure): use TMM. It is ~1000× faster than RCWA and exact for this geometry.
- **2D / 3D periodic, frequency-domain QE sweep**: use RCWA. Fast for moderate Fourier order, well-suited to sub-wavelength gratings.
- **Strongly resonant or large lateral extent, broadband transient effects**: use FDTD. Slower per wavelength but handles any geometry.
- **Cross-validation**: run two methods of different families and compare — see [RCWA vs FDTD](./rcwa-vs-fdtd.md).

## What this chapter covers

| Page | Topic | Key takeaway |
|---|---|---|
| [RCWA Explained](./rcwa-explained.md) | Fourier expansion, eigenmodes, S-matrix recursion | Why RCWA is fast for periodic stacks |
| [FDTD Explained](./fdtd-explained.md) | Yee lattice, update equations, Courant condition | Why FDTD is general but more expensive |
| [RCWA vs FDTD](./rcwa-vs-fdtd.md) | Side-by-side comparison and use cases | When to use which |
| [Numerical Stability](./numerical-stability.md) | Eigenvalue conditioning, S-matrix vs T-matrix, mixed precision | How COMPASS keeps RCWA stable at high Fourier order |

::: tip Prerequisites
This chapter assumes the [Optics](./optics-intro.md) chapter — especially [Electromagnetic Waves](./electromagnetic-waves.md) and [Diffraction](./diffraction.md). If terms like *Bloch mode* or *S-matrix* are new, read those first.
:::
