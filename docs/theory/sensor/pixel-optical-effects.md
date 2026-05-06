---
title: Pixel Optical Effects
description: Design-focused guide to microlens CRA matching, CFA spectral response, BARL tuning, silicon absorption depth, DTI crosstalk, angular response, and polarization sensitivity in CMOS image sensor pixels.
---

# Pixel Optical Effects

::: tip Prerequisites
[Image Sensor Optics](./image-sensor-optics.md) gives the stack map. This page explains the design effects that move QE, crosstalk, angular response, and color accuracy.
:::

Modern BSI pixels are not limited by one component. QE and crosstalk emerge from the interaction between the microlens, CFA, metal grid, BARL, silicon, photodiode, and DTI. This page organizes those effects by the design question they answer.

## Design Effect Map

| Effect | Main design variables | Primary metric | Best first check |
|---|---|---|---|
| Microlens focusing and CRA | Lens height, radius, shape exponent, lateral offset, incident angle | Center QE, corner shading, chief-ray response | [Microlens & CRA](/cookbook/microlens-optimization) |
| CFA spectral response | Filter passband, out-of-band absorption, thickness, metal-grid aperture | Channel QE, color separation | [Color Filter Designer](/simulator/color-filter) |
| BARL and ARC behavior | Layer materials, thicknesses, order, target wavelength band | Reflection, QE ripple | [BARL Design](/cookbook/barl-design) |
| Silicon absorption depth | Silicon thickness, photodiode depth, wavelength range | Red/NIR QE, transmission loss | [Si Absorption Depth](/simulator/si-absorption) |
| DTI and optical crosstalk | Trench depth, width, fill material, placement | Crosstalk matrix, MTF proxy | [DTI Crosstalk](/cookbook/dti-crosstalk) |
| Angular response | Incident angle, azimuth, CRA offset, aperture stop assumptions | QE vs angle, lens shading | [Angular Response](/simulator/angular-response) |
| Polarization sensitivity | TE/TM response, metal grid direction, oblique incidence | TE/TM QE split, color nonuniformity | [RCWA vs FDTD](../simulation/rcwa-vs-fdtd.md) |

## Microlens and CRA Matching

The microlens is the first active optical element in the pixel. It concentrates light toward the photodiode and compensates for the chief ray angle at off-axis sensor locations.

The main trade-off is focus strength versus angular tolerance:

- A taller or more curved lens focuses light more aggressively, which can improve normal-incidence QE.
- The same lens can be less tolerant to large CRA if its focal spot shifts outside the photodiode.
- A laterally shifted lens can recover corner QE, but the optimum shift depends on wavelength, stack height, and the photodiode target location.

In COMPASS, this effect is best studied by sweeping lens height and lateral offset while measuring QE for several incident angles. RCWA is efficient for periodic sweeps; FDTD is useful when the geometry or source model breaks periodic assumptions.

## CFA Spectral Response and Color Separation

The color filter is both a desired spectral selector and an unavoidable loss element. A red filter should transmit red light and absorb blue/green light, but it also absorbs some red light inside the passband. That passband loss directly lowers red-channel QE.

Two crosstalk paths are important:

| Path | Mechanism | Symptom |
|---|---|---|
| Spectral leakage | Filter transmits wavelengths outside the intended band | Poor color separation even if light stays in the right pixel |
| Spatial leakage | Light passes through one color filter and is absorbed in a neighboring photodiode | Off-diagonal crosstalk matrix terms |

The metal grid between CFA cells improves spatial isolation, but it adds diffraction and absorption. The right aperture is therefore a compromise between isolation, throughput, and angular response.

## BARL and ARC Tuning

The BARL stack reduces reflection at the color-filter-to-silicon interface. Its optimum is not independent of the rest of the pixel:

- CFA thickness changes the phase of light before it reaches BARL.
- Silicon thickness changes the spectral ripple caused by back-reflections.
- Incident angle and polarization shift the effective optical path in every film.
- The target band differs by channel, so a stack optimized for green may not be optimal for red or blue.

For 1D planar stacks, TMM is the right first approximation. Once the CFA grid, metal grid, microlens, or DTI geometry matters, switch to RCWA or FDTD and compare the full energy balance.

## Silicon Absorption Depth

Silicon absorption is strongly wavelength-dependent. Blue light is absorbed near the surface, green light within the first few micrometers, and red/NIR light can require several micrometers or more.

This creates a three-way trade-off:

| Increase silicon thickness | Benefit | Cost |
|---|---|---|
| More red/NIR absorption | Higher long-wavelength QE | Higher chance of lateral diffusion and crosstalk |
| More optical path length | More opportunities for absorption | More spectral ripple if reflections are not controlled |
| Deeper photodiode volume | Better collection for red light | More dependence on DTI and electrical isolation |

The optical model can estimate absorbed power, but collected charge also depends on photodiode geometry and carrier transport. Treat optical QE as an upper bound unless the electrical collection model is included.

## DTI and Crosstalk

Deep Trench Isolation (DTI) is both an optical structure and an electrical isolation structure. Optically, the low-index trench wall reflects light back into the intended pixel. Electrically, it blocks carriers generated near pixel boundaries from diffusing into neighbors.

DTI design affects different wavelengths differently:

- Blue light is absorbed near the top, so DTI mainly controls diffraction and near-surface leakage.
- Green light is sensitive to both focusing and sidewall reflection.
- Red/NIR light penetrates deeply, so DTI depth and bottom leakage become more important.

Use per-pixel QE and the crosstalk matrix together. A design can raise total silicon absorption while also increasing off-diagonal crosstalk, which may hurt color accuracy even if total QE looks better.

## Angular and Polarization Response

At normal incidence, TE and TM behavior can be similar for symmetric pixels. At oblique incidence, the stack sees different effective optical paths and boundary conditions:

- Reflection from films changes with angle and polarization.
- Diffraction orders shift laterally, changing where power lands.
- Metal grid and DTI edges can favor one field orientation.
- The microlens focal spot moves relative to the photodiode.

For production-style evaluation, report QE as a function of wavelength, polar angle, azimuth, and polarization. For natural illumination, COMPASS averages TE and TM for unpolarized light, but inspecting the split is useful when debugging angular color shading.

## Solver Implications

| Question | Solver family | Notes |
|---|---|---|
| Is my BARL stack roughly tuned? | TMM | Fast and exact for 1D planar layers |
| How does a periodic 2x2 Bayer cell behave? | RCWA | Efficient for wavelength and parameter sweeps |
| How sensitive is the design to non-periodic details? | FDTD | More flexible, but slower and grid-dependent |
| Are results physically plausible? | Cross-validation | Compare R/T/A, energy balance, and convergence ladders |

The practical workflow is: use the fastest model that captures the effect, then validate the final candidate with a different solver family. The [RCWA vs FDTD](../simulation/rcwa-vs-fdtd.md) page explains how to make that comparison without mixing incompatible observables.
