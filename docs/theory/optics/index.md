# Optics Chapter Overview

This chapter covers the wave-optics fundamentals you need before reading the simulation or image sensor chapters. It is solver-agnostic: the equations and intuitions here hold for *any* electromagnetic method.

## Why wave optics?

Geometric (ray) optics treats light as straight rays that refract and reflect at interfaces. It works well when every feature in the system is much larger than the wavelength. Modern CMOS image sensors break this assumption:

- Pixel pitches of 0.6–1.0 um are comparable to visible wavelengths (0.38–0.78 um).
- Color filter arrays, microlens arrays, and Deep Trench Isolation (DTI) form periodic sub-wavelength structures.
- Anti-reflection coatings and BARL stacks rely on **interference**, which ray optics cannot describe.

To predict quantum efficiency, crosstalk, and angular response correctly, we have to solve Maxwell's equations directly. That is what this chapter prepares you for.

## What this chapter covers

| Page | Topic | Key takeaway |
|---|---|---|
| [Light Basics](./light-basics.md) | Wavelength, refractive index, polarization, absorption | The vocabulary used by every later page |
| [Electromagnetic Waves](./electromagnetic-waves.md) | Maxwell's equations, plane waves, boundary conditions | The governing equations all solvers approximate |
| [Thin Film Optics](./thin-film-optics.md) | Fresnel coefficients, multilayer interference, ARC design | Why BARL stacks and ARCs work |
| [Diffraction](./diffraction.md) | Periodic gratings, diffraction orders, Bloch waves | Why sub-wavelength pixel features require modal methods |

## How this connects to the rest of theory

- The **image sensor** chapter applies these concepts to the layered BSI pixel structure.
- The **simulation** chapter shows how each solver discretizes Maxwell's equations into a numerical scheme.

::: tip
If you only need a refresher, [Light Basics](./light-basics.md) and the first sections of [Electromagnetic Waves](./electromagnetic-waves.md) are enough to follow the rest of the documentation.
:::
