# FDTD Explained

::: tip 선수 지식 | Prerequisites
[전자기파](/theory/electromagnetic-waves) → 이 페이지
FDTD가 처음이라면 먼저 [솔버 선택 가이드](/guide/choosing-solver)에서 개요를 확인하세요.
:::

FDTD (Finite-Difference Time-Domain) is the other main solver type in COMPASS. Unlike RCWA which works in frequency space, FDTD directly simulates how electromagnetic waves propagate through space over time. It divides space into a 3D grid and marches forward in time step by step, like watching a movie of light passing through the pixel.

The Finite-Difference Time-Domain (FDTD) method is the second major solver approach in COMPASS. It directly solves Maxwell's equations on a discrete spatial and temporal grid.

## Core idea

FDTD replaces the continuous derivatives in Maxwell's equations with finite differences on a staggered grid (the **Yee lattice**). The electric and magnetic fields are updated alternately in a leapfrog time-stepping scheme:

1. Update $\mathbf{H}$ from $\mathbf{E}$ (half time step).
2. Update $\mathbf{E}$ from $\mathbf{H}$ (half time step).
3. Repeat until the simulation reaches steady state or a pulse has fully propagated.

## The Yee lattice

The Yee cell places the six field components ($E_x, E_y, E_z, H_x, H_y, H_z$) at staggered positions within each grid cell:

```
        Hz ---- Ey
        |       |
        Ex      |
        |       |
        Ey ---- Hz
```

This staggering ensures that every finite-difference curl approximation is second-order accurate and naturally satisfies the divergence conditions ($\nabla \cdot \mathbf{B} = 0$).

<YeeCellViewer />

## Update equations

The update equations for a single component (e.g., $E_x$) in a non-magnetic medium:

$$E_x^{n+1}(i,j,k) = E_x^n(i,j,k) + \frac{\Delta t}{\varepsilon_0 \varepsilon_r(i,j,k)} \left( \frac{H_z^{n+1/2}(i,j,k) - H_z^{n+1/2}(i,j-1,k)}{\Delta y} - \frac{H_y^{n+1/2}(i,j,k) - H_y^{n+1/2}(i,j,k-1)}{\Delta z} \right)$$

The magnetic field components are updated analogously.

## Stability: the Courant condition

The time step $\Delta t$ must satisfy the Courant-Friedrichs-Lewy (CFL) condition to prevent numerical instability:

$$\Delta t \leq \frac{1}{c \sqrt{\frac{1}{\Delta x^2} + \frac{1}{\Delta y^2} + \frac{1}{\Delta z^2}}}$$

where $c$ is the speed of light. COMPASS FDTD solvers automatically compute the maximum stable time step from the grid spacing.

## Boundary conditions

### Periodic boundaries (Bloch)

For pixel simulations with plane wave excitation, COMPASS uses **Bloch periodic boundary conditions** in the lateral (xy) directions:

$$\mathbf{E}(x + \Lambda_x, y, z) = \mathbf{E}(x, y, z) \, e^{ik_x \Lambda_x}$$

This models an infinite array of identical pixels, matching the assumption in RCWA.

### Absorbing boundaries (PML)

In the vertical (z) direction, **Perfectly Matched Layers (PML)** absorb outgoing waves without reflection. PML is a lossy medium with conductivity that increases gradually into the boundary:

$$\sigma(z) = \sigma_\text{max} \left(\frac{z}{d_\text{PML}}\right)^p$$

The exponent $p$ (typically 3-4) and the PML thickness (typically 10-20 cells) control the absorption quality. Abrupt PML can cause spurious reflections.

## Source injection

COMPASS injects a plane wave using the **total-field/scattered-field (TFSF)** technique or by specifying field values on a source plane. For a monochromatic simulation, a CW (continuous-wave) source is used and the simulation runs until the fields reach steady state.

## Extracting results

After the FDTD simulation reaches steady state, COMPASS extracts:

- **Reflection and transmission**: Poynting flux through monitor planes above and below the structure.
- **Field distributions**: Snapshot of $|\mathbf{E}|^2$ on any plane.
- **Absorption per pixel**: Volume integral of $\frac{1}{2}\omega\varepsilon'' |\mathbf{E}|^2$ within each photodiode region.

## Grid resolution

The spatial grid must resolve both the smallest geometric features and the shortest wavelength in any material:

$$\Delta x \leq \frac{\lambda_\text{min}}{n_\text{max} \cdot N_\text{ppw}}$$

where $N_\text{ppw}$ is the number of points per wavelength (typically 15-20 for second-order FDTD). For silicon ($n \approx 4$) at 400 nm:

$$\Delta x \leq \frac{0.4 \text{ um}}{4 \times 20} = 5 \text{ nm}$$

This can lead to large grids and long run times compared to RCWA.

## FDTD solvers in COMPASS

| Solver | Library | GPU support | Notes |
|--------|---------|-------------|-------|
| `fdtd_flaport` | fdtd (flaport) | CUDA (PyTorch) | Lightweight, good for prototyping. |
| `fdtdz` | fdtdz | CUDA | Specialized for layered 2.5D structures. |
| `meep` | Meep | CPU (MPI) | Full-featured, supports dispersive materials natively. |

## When to choose FDTD

FDTD is the better choice when:

- The structure has **non-periodic** or **aperiodic** features.
- You need a **broadband** response from a single simulation run (pulse excitation).
- You want to **visualize time-domain** field propagation.
- You are studying **near-field** effects that are hard to extract from Fourier-space data.

For standard periodic pixel simulations, RCWA is usually faster and more accurate. See [RCWA vs FDTD](./rcwa-vs-fdtd) for a detailed comparison.
