---
title: FDTD Explained
description: Finite-difference time-domain simulation for image-sensor optics, including Yee grids, leapfrog updates, CFL stability, PML boundaries, sources, monitors, convergence, and practical COMPASS settings.
---

# FDTD Explained

::: tip Prerequisites
[Electromagnetic Waves](/theory/optics/electromagnetic-waves) -> this page.
If FDTD is new to you, start with the [solver selection guide](/guide/choosing-solver).
:::

FDTD (Finite-Difference Time-Domain) solves Maxwell's equations directly in space and time. Instead of expanding a periodic pixel into Fourier harmonics like RCWA, it voxelizes the geometry, injects a time-dependent source, and advances the electric and magnetic fields step by step.

For image sensors, that makes FDTD valuable when you need a real-space view of the field: how light bends through a microlens, how it scatters from a metal grid, how it leaks across DTI, or how a finite/non-periodic layout behaves. The cost is that the grid must resolve both the smallest geometry and the shortest wavelength inside the highest-index material.

## What problem FDTD solves

In a non-magnetic optical stack, the time-domain curl equations are:

$$\frac{\partial \mathbf{H}}{\partial t} = -\frac{1}{\mu_0}\nabla \times \mathbf{E}$$

$$\frac{\partial \mathbf{E}}{\partial t} =
\frac{1}{\varepsilon_0 \varepsilon_r(\mathbf{r})}
\left(\nabla \times \mathbf{H} - \mathbf{J}\right)$$

FDTD replaces the continuous derivatives with finite differences on a rectangular grid:

$$x_i = i\Delta x,\quad y_j = j\Delta y,\quad z_k = k\Delta z,\quad t_n = n\Delta t$$

The output can be interpreted in two ways:

- **Time-domain fields**: snapshots of $\mathbf{E}(t)$ and $\mathbf{H}(t)$.
- **Frequency-domain observables**: Fourier-transformed fields accumulated during the run, such as reflectance, transmittance, absorption, and QE at selected wavelengths.

::: info Broadband does not mean free
A short pulse can cover many wavelengths in one run, but the simulation still needs enough physical time for all relevant frequencies and resonant tails to decay. A broadband FDTD result is only meaningful after both grid and time-window convergence are checked.
:::

## Mental model

Think of FDTD as a movie camera for Maxwell's equations:

1. The pixel stack is converted to a 3D array of material values.
2. A source injects a pulse or continuous wave.
3. Fields propagate, scatter, interfere, and absorb on the grid.
4. Monitors collect flux and field data.
5. The run stops when the source has passed and residual energy has decayed.

Unlike RCWA, FDTD does not require every layer to be laterally periodic. Periodic boundaries are optional. That flexibility is why FDTD is often used as a cross-check for RCWA or as a reference method for difficult geometry.

## From pixel stack to FDTD grid

COMPASS turns a `PixelStack` into a voxelized permittivity volume:

| Physical feature | FDTD representation | Main risk |
|---|---|---|
| Air, planarization, BARL | Uniform voxel regions | Thin films may be under-resolved |
| Microlens | Staircase or smoothed 3D shape | Curvature needs fine x-y-z resolution |
| Color filter | Absorbing/dispersive voxel region | Wrong material loss changes QE directly |
| Metal grid | High-loss, high-contrast voxels | Skin depth and sharp corners need fine grid |
| DTI/BDTI | Silicon/oxide/trench voxel boundaries | Crosstalk depends strongly on boundary placement |
| Photodiode | Absorption integration volume | Monitor region must match the electrical collection model |
| Top/bottom open space | PML absorbing layers | Too-close PML reflects near fields |

The solver sees only the discretized grid. Always verify the voxelized geometry before treating field maps as physical insight.

## The Yee lattice

FDTD normally uses the Yee lattice: electric and magnetic field components are staggered in both space and time. Each curl update uses nearby field samples in the natural orientation for Maxwell's equations.

<YeeCellViewer />

The staggering has two practical consequences:

- Field components are not stored at the same point, so energy density and flux monitors may interpolate fields.
- Dielectric interfaces can sit between field samples, so staircasing and subpixel averaging matter near sharp material boundaries.

## Leapfrog update

The electric and magnetic fields are updated alternately:

1. $\mathbf{H}^{n+1/2}$ is updated from $\mathbf{E}^{n}$.
2. $\mathbf{E}^{n+1}$ is updated from $\mathbf{H}^{n+1/2}$.
3. The process repeats until the desired physical time has elapsed.

For example, one electric-field component in a non-magnetic, nondispersive medium is:

$$E_x^{n+1}(i,j,k) = E_x^n(i,j,k) +
\frac{\Delta t}{\varepsilon_0 \varepsilon_r(i,j,k)}
\left[
\frac{H_z^{n+1/2}(i,j,k) - H_z^{n+1/2}(i,j-1,k)}{\Delta y}
-
\frac{H_y^{n+1/2}(i,j,k) - H_y^{n+1/2}(i,j,k-1)}{\Delta z}
\right]$$

The other components follow the same curl pattern. In lossy or dispersive media, additional material-update terms are included so that $\varepsilon(\omega)$ and absorption are represented correctly.

## Stability: the CFL limit

The time step must satisfy the Courant-Friedrichs-Lewy (CFL) condition. For a 3D Cartesian grid:

$$\Delta t \le
\frac{S}{c\sqrt{\frac{1}{\Delta x^2}+\frac{1}{\Delta y^2}+\frac{1}{\Delta z^2}}}$$

where $S$ is the Courant factor. A smaller spatial grid forces a smaller time step. This is why refining a 3D FDTD grid is expensive twice: the number of voxels increases, and the number of time steps needed for the same physical duration also increases.

If a simulation suddenly produces `NaN`, exploding fields, or nonphysical energy gain, check the time step, material model, and PML first.

## Grid resolution and numerical dispersion

The grid must resolve the shortest wavelength inside the highest-index material:

$$\Delta \le \frac{\lambda_0}{n_\text{max} N_\text{ppw}}$$

where $N_\text{ppw}$ is the target points per wavelength. A common starting point is 15-20 points per wavelength for qualitative work, with finer checks for sign-off.

For silicon at $\lambda_0 = 400$ nm and $n \approx 4$:

$$\Delta \le \frac{0.4\ \mu\text{m}}{4 \times 20} = 5\ \text{nm}$$

That single estimate explains most FDTD cost in image sensors. Blue light inside silicon demands very fine cells, and a 2x2 Bayer domain can reach hundreds of millions of Yee samples if every dimension is refined uniformly.

### What under-resolution looks like

| Symptom | Likely cause |
|---|---|
| QE changes strongly when `dx` changes | Grid dispersion or geometry staircasing |
| Metal-grid effect disappears | Skin depth or metal edge under-resolved |
| DTI crosstalk looks too optimistic | Trench width or sidewall location shifted by voxelization |
| Field hot spots move with resolution | Interface interpolation artifact |
| Flux balance drifts | Monitors too close, grid too coarse, or material loss mismatch |

## Sources and monitors

FDTD results depend heavily on source and monitor setup.

### Sources

| Source type | Use it when | Caveat |
|---|---|---|
| Continuous wave (CW) | One wavelength, steady-state field maps | Must run long enough to reach steady state |
| Gaussian pulse | Broadband spectra | Needs frequency-domain monitor normalization |
| Planewave/TFSF | Incident plane wave on a finite scatterer | Source box must not intersect scatterers or PML |
| Bloch-periodic planewave | Periodic pixel array at oblique CRA | Boundary phase must match the incident wavevector |

### Broadband oblique incidence

Oblique broadband planewaves need extra care. With Bloch-periodic boundaries, the lateral phase is tied to the incident wavevector. If a short pulse spans many wavelengths, the same nominal source setup can represent slightly different polar angles across the spectrum unless the backend explicitly handles broadband oblique injection.

There are three common strategies:

| Strategy | Use when | Caveat |
|---|---|---|
| Single-frequency Bloch sweep | Highest accuracy per angle/wavelength | Many runs |
| Broadband Bloch run plus interpolation | Moderate bandwidth and smooth angular response | Must interpolate angular response carefully |
| Specialized broadband oblique source | Backend supports it directly | Backend-specific assumptions must be documented |

For image-sensor angular response, a robust workflow is to solve a structured angular grid, store $\text{QE}(\lambda,\theta,\phi)$, and interpolate that data to the CRA/MRA or ray-file angles used by the camera model. This avoids pretending that a sparse, nonuniform set of lens rays is itself a good FDTD sweep grid.

When reporting broadband oblique FDTD, include the angular grid, wavelength grid, source type, boundary phase convention, and interpolation method.

### Monitors

Flux monitors measure the Poynting vector through a surface:

$$\mathbf{S} = \frac{1}{2}\operatorname{Re}(\mathbf{E} \times \mathbf{H}^*)$$

For spectra, fields are Fourier-transformed at monitor points and then combined into flux. Do not Fourier-transform the time-domain power trace directly unless the backend explicitly documents that workflow.

For image-sensor QE, absorption is normally integrated in silicon or photodiode regions:

$$P_\text{abs}(\omega) =
\frac{1}{2}\omega\varepsilon_0\varepsilon_r''(\omega)
\int_V |\mathbf{E}(\mathbf{r}, \omega)|^2\,dV$$

The integration volume is part of the model. Optical absorption in silicon is not automatically the same as collected charge if the electrical collection region is smaller than the absorbing region.

## Boundary conditions

### Periodic and Bloch boundaries

For a repeated unit cell, lateral boundaries can be periodic:

$$\mathbf{E}(x+\Lambda_x,y,z)=\mathbf{E}(x,y,z)$$

For oblique illumination, the periodic boundary needs a Bloch phase:

$$\mathbf{E}(x+\Lambda_x,y,z)=\mathbf{E}(x,y,z)e^{ik_x\Lambda_x}$$

Use Bloch periodicity when comparing FDTD against RCWA for the same infinite pixel array.

### PML absorbing boundaries

Open boundaries are usually terminated with PML (Perfectly Matched Layers). PML is an artificial absorbing layer placed outside the physical region so outgoing waves leave the simulation cell with minimal reflection.

Practical PML rules:

- Keep PML away from high-index geometry and strong near fields.
- Increase PML thickness for grazing incidence, high-Q resonances, or evanescent-rich fields.
- Check reflection by moving the PML farther away and rerunning.
- Do not place sources or flux monitors inside the PML.

## Practical FDTD workflow for image sensors

1. Start from a simple 1D stack and match TMM or zero-order RCWA.
2. Add the periodic pixel geometry with coarse but valid grid spacing.
3. Verify the voxelized geometry before running expensive sweeps.
4. Normalize against an incident-field reference run.
5. Place reflection/transmission monitors away from sources, scatterers, and PML.
6. Run until fields decay below the target threshold.
7. Sweep grid spacing: for example 20 nm, 10 nm, 5 nm.
8. Sweep PML thickness and monitor offsets.
9. Compare integrated $R + T + A$ against 1.
10. Only then interpret photodiode QE and crosstalk.

## Runtime and memory scaling

The memory footprint scales roughly with the number of Yee cells:

$$N_\text{cells}=N_xN_yN_z$$

Each cell stores multiple electric and magnetic components, material coefficients, and sometimes DFT monitor accumulators. Runtime scales with:

$$\text{work} \propto N_xN_yN_zN_t$$

and $N_t$ increases as the grid spacing shrinks. Halving $\Delta x$, $\Delta y$, and $\Delta z$ can increase memory by about $8\times$ and runtime by more than $16\times$ for the same physical duration.

## Common failure modes

### Fields blow up

Check the Courant factor, negative or inconsistent material coefficients, dispersive material setup, and whether the source overlaps a lossy or PML region.

### PML reflects too much

Move the PML farther from the pixel stack, thicken it, and rerun with the same monitor placement. This is especially important for oblique CRA, high-index silicon, and guided or evanescent fields.

### Broadband spectrum is noisy

Increase the run time, reduce the source bandwidth, use a smoother pulse, or use frequency-domain convergence criteria. A short time trace cannot resolve narrow spectral features.

### RCWA and FDTD disagree

First match the physical problem: same unit cell, same materials, same incident angle, same polarization, same absorption volume, and same boundary conditions. Then converge RCWA order and FDTD grid independently.

## FDTD vs RCWA vs TMM

| Method | Best at | Weak at |
|---|---|---|
| TMM | 1D thin-film stacks | No lateral diffraction |
| RCWA | Periodic layered pixels and wavelength sweeps | Aperiodic finite features |
| FDTD | Real-space fields, finite geometry, broadband checks | Fine-grid memory and runtime |

For normal periodic BSI pixels, RCWA is usually the fastest primary solver. Use FDTD for validation, finite-layout studies, broadband response, and field intuition.

## COMPASS FDTD solvers

| Solver | Library | GPU support | Notes |
|---|---|---|---|
| `fdtd_flaport` | fdtd (flaport) | CUDA (PyTorch) | Lightweight backend for prototyping and quick checks. |
| `fdtdz` | fdtdz | CUDA/JAX depending install | Experimental high-performance workflow for structured grids. |
| `meep` | Meep | CPU/MPI | Mature reference backend with broad material and monitor support. |

These backends share the `SolverBase` interface, but their feature coverage is not identical. Treat backend changes as cross-validation, not as a drop-in guarantee.

## Practical setup for image sensors

```yaml
solver:
  name: fdtd_flaport
  type: fdtd
  params:
    grid_spacing: 0.01        # um; start coarse, then converge
    runtime_fs: 300
    courant_factor: 0.5
    pml_thickness: 20         # cells
    source:
      type: gaussian_pulse
      normalize_reference: true
    monitors:
      flux_offset: 0.2        # um from patterned stack
      dft_fields: true
  convergence:
    grid_spacing_um: [0.02, 0.01, 0.005]
    energy_tolerance: 0.02
```

For final reports, include grid spacing, time step or Courant factor, physical runtime, PML thickness, monitor locations, and the final energy balance. Without those, an FDTD number is hard to reproduce.

## Further reading

- K. S. Yee, [Numerical solution of initial boundary value problems involving Maxwell's equations in isotropic media](https://ieeexplore.ieee.org/document/1138693), IEEE Transactions on Antennas and Propagation 14, 302-307 (1966).
- J. P. Berenger, [A perfectly matched layer for the absorption of electromagnetic waves](https://doi.org/10.1006/jcph.1994.1159), Journal of Computational Physics 114, 185-200 (1994).
- Meep documentation, [Introduction](https://meep.readthedocs.io/en/stable/Introduction/), [Perfectly Matched Layers](https://meep.readthedocs.io/en/stable/Perfectly_Matched_Layer/), [Materials](https://meep.readthedocs.io/en/stable/Materials/), and [Subpixel Smoothing](https://meep.readthedocs.io/en/stable/Subpixel_Smoothing/).
