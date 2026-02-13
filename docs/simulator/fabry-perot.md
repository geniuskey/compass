---
title: "Fabry-P\xE9rot Visualizer"
---

# Fabry-P&eacute;rot Visualizer

Explore single-layer thin film interference with a phasor diagram showing how multiple reflected beams combine to determine overall reflectance and transmittance.

<FabryPerotVisualizer />

## Physics

When light encounters a thin film of thickness d and refractive index n, partial reflections occur at each interface. The reflected beams interfere, with the result determined by the Fresnel coefficients and the phase accumulated during each round trip:

**delta = 4 pi n d cos(theta) / lambda**

### Fresnel Coefficients

At each interface, the reflection and transmission amplitudes are given by the Fresnel equations. The overall reflectance is the coherent sum of all partially reflected beams, which the phasor diagram shows as a spiral of decreasing-amplitude vectors.

### Quarter-Wave Condition

Minimum reflectance occurs when the film optical thickness equals lambda/4, making the round-trip phase shift pi. If the film index satisfies **n_film = sqrt(n_substrate)**, the first two reflected beams cancel perfectly, giving zero reflectance at the design wavelength.

### Application to Image Sensors

Anti-reflection coatings on BSI pixels use this principle. The BARL (bottom anti-reflective layer) stack is a multilayer extension that achieves broadband low reflectance across the visible spectrum.

::: tip
Drag the wavelength slider to watch the phasor spiral rotate — constructive and destructive interference alternate as the optical path length changes.
:::
