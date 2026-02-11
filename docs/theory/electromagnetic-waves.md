# Electromagnetic Waves

::: tip 선수 지식 | Prerequisites
이 페이지를 읽기 전에 [광학 기초 입문](/introduction/optics-primer)과 [빛의 기초](/theory/light-basics)를 먼저 읽어보세요.
Before reading this page, check out the [Optics Primer](/introduction/optics-primer) and [Light Basics](/theory/light-basics).
:::

Why do we need Maxwell's equations? Because they tell us exactly how light behaves when it encounters the tiny structures inside a pixel. When pixel features are smaller than the wavelength of light (~0.5 um), we can't use simple ray tracing -- we need the full wave picture that Maxwell's equations provide. The solvers in COMPASS (RCWA and FDTD) are both methods for solving these equations numerically.

This page introduces Maxwell's equations and the wave formalism that RCWA and FDTD solvers use internally.

<EMWaveAnimation />

## Maxwell's equations

All electromagnetic phenomena are governed by four equations. In a linear, isotropic, non-magnetic medium with no free charges:

$$\nabla \times \mathbf{E} = -\mu_0 \frac{\partial \mathbf{H}}{\partial t}$$

$$\nabla \times \mathbf{H} = \varepsilon_0 \varepsilon_r \frac{\partial \mathbf{E}}{\partial t}$$

$$\nabla \cdot (\varepsilon_r \mathbf{E}) = 0$$

$$\nabla \cdot \mathbf{H} = 0$$

Here $\mathbf{E}$ is the electric field, $\mathbf{H}$ is the magnetic field, $\varepsilon_r$ is the relative permittivity (which may be complex and spatially varying), $\varepsilon_0$ is the vacuum permittivity, and $\mu_0$ is the vacuum permeability.

## Time-harmonic form

For monochromatic (single-frequency) light with time dependence $e^{-i\omega t}$, the curl equations become:

$$\nabla \times \mathbf{E} = i\omega \mu_0 \mathbf{H}$$

$$\nabla \times \mathbf{H} = -i\omega \varepsilon_0 \varepsilon_r \mathbf{E}$$

This is the starting point for **RCWA**, which solves the time-harmonic equations in the frequency domain. **FDTD** instead solves the time-domain equations directly on a grid.

## Plane waves

The simplest solution to Maxwell's equations in a uniform medium is a plane wave:

$$\mathbf{E}(\mathbf{r}, t) = \mathbf{E}_0 \, e^{i(\mathbf{k} \cdot \mathbf{r} - \omega t)}$$

where the wave vector $\mathbf{k}$ satisfies the dispersion relation:

$$|\mathbf{k}|^2 = k_0^2 \varepsilon_r, \qquad k_0 = \frac{2\pi}{\lambda}$$

In COMPASS, the incident light is always a plane wave (or a weighted sum of plane waves for cone illumination). The solver computes how this plane wave interacts with the layered pixel structure.

## Incidence geometry

COMPASS uses a spherical coordinate convention for the incident wave direction:

- $\theta$: Polar angle measured from the surface normal (z-axis). $\theta = 0$ is normal incidence.
- $\phi$: Azimuthal angle in the xy-plane. $\phi = 0$ is along the x-axis.

The transverse components of the wave vector in the incidence medium ($n_\text{inc}$) are:

$$k_x = k_0 n_\text{inc} \sin\theta \cos\phi$$

$$k_y = k_0 n_\text{inc} \sin\theta \sin\phi$$

These components are conserved at every interface (Snell's law generalized to 3D), which is how both RCWA and FDTD enforce the incidence angle.

## Boundary conditions

At an interface between two media, the tangential components of $\mathbf{E}$ and $\mathbf{H}$ must be continuous:

$$\mathbf{E}_{t,1} = \mathbf{E}_{t,2}$$

$$\mathbf{H}_{t,1} = \mathbf{H}_{t,2}$$

These conditions lead to the Fresnel reflection and transmission coefficients for a single interface:

$$r_\text{TE} = \frac{n_1 \cos\theta_1 - n_2 \cos\theta_2}{n_1 \cos\theta_1 + n_2 \cos\theta_2}$$

$$r_\text{TM} = \frac{n_2 \cos\theta_1 - n_1 \cos\theta_2}{n_2 \cos\theta_1 + n_1 \cos\theta_2}$$

For a multi-layer stack with lateral patterning, these conditions must be solved numerically -- which is exactly what RCWA and FDTD do.

## Energy flow: the Poynting vector

The time-averaged power flow per unit area is given by the Poynting vector:

$$\langle \mathbf{S} \rangle = \frac{1}{2} \text{Re}(\mathbf{E} \times \mathbf{H}^*)$$

The z-component $S_z$ tells us how much power flows through a horizontal plane. COMPASS uses the Poynting vector to compute:

- **Reflection** ($R$): power reflected back above the structure.
- **Transmission** ($T$): power transmitted below the structure.
- **Absorption** ($A$): power absorbed within the structure, computed as $A = 1 - R - T$.
- **QE per pixel**: power absorbed specifically within each photodiode region.

## Why two solver approaches?

Maxwell's equations can be solved in different ways, each with trade-offs:

| Approach | Method | Strengths |
|----------|--------|-----------|
| **Frequency domain** | RCWA | Fast for periodic structures, exact periodicity, efficient wavelength sweeps |
| **Time domain** | FDTD | Handles arbitrary geometry, broadband in one run, intuitive field visualization |

COMPASS supports both so you can choose the best tool for each problem and cross-validate results. See [RCWA Explained](./rcwa-explained) and [FDTD Explained](./fdtd-explained) for details.
