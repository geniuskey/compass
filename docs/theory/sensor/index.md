---
title: Image Sensor Chapter Overview
description: Overview of the image sensor theory chapter covering BSI pixel architecture, pixel optical effects, quantum efficiency, crosstalk, signal chain modeling, and color reproduction.
---

# Image Sensor Chapter Overview

This chapter explains the optical structure of a modern CMOS image sensor (CIS) pixel and the chain of physical effects that turn incident photons into a measurable signal.

## What an image sensor pixel is

A CIS pixel is a layered optical stack on top of a silicon photodiode. Light entering from the top passes through a microlens, planarization, color filter, anti-reflection layer (BARL), and finally into silicon, where it is absorbed and generates electron-hole pairs. The collected charge is read out as the pixel signal.

In COMPASS the layer order from bottom to top is:

```
silicon → BARL → color filter → planarization → microlens → air
```

Light propagates in the **−z** direction (air → silicon), and z = 0 is at the silicon bottom.

## What this chapter covers

| Page | Topic | Key takeaway |
|---|---|---|
| [Image Sensor Optics](./image-sensor-optics.md) | BSI architecture, microlens, CFA, BARL, DTI | Anatomy of the optical stack and what each layer does |
| [Pixel Optical Effects](./pixel-optical-effects.md) | CRA, CFA spectral response, BARL, silicon absorption, DTI, angular and polarization response | Design trade-offs that shape QE and crosstalk |
| [Quantum Efficiency](./quantum-efficiency.md) | QE definition, computation methods, crosstalk | The headline metric and how COMPASS measures it |
| [Signal Chain](./signal-chain.md) | Illuminant → scene → lens → sensor signal | How simulated QE turns into a radiometric pixel value |
| [Color Reproduction](./color-reproduction.md) | Camera RGB, CIE XYZ, Lab, sRGB, CCM, color error | How spectral sensor response becomes standard color metrics |
| [Noise, SNR, and Dynamic Range](./noise-and-snr.md) | Shot/read/dark/FPN noise, FWC, DR, PTC, responsivity, linearity | Full noise model behind every signal-chain prediction |
| [EMVA 1288 Characterization](./emva1288.md) | EMVA 1288 parameter set, SNR curve, quality grades | Standard reporting format for sensor characterization |
| [Resolution, MTF, and Pixel Scaling](./resolution-and-mtf.md) | Pixel aperture MTF, optical diffraction, Nyquist, pitch scaling | Spatial frequency response and pitch trade-offs |

## Boundary with nearby pages

| Nearby page | Difference |
|---|---|
| [Pixel Anatomy](../basics/pixel-anatomy.md) | A guided beginner tour of the same stack; this chapter is the compact technical model. |
| [Optics](../optics/) | Defines the wave-optics laws; this chapter maps them onto sensor components. |
| [Simulation](../simulation/) | Explains the numerical methods that compute the fields and absorbed power. |

## How to read this chapter

- If you only care about the optical design of the pixel, [Image Sensor Optics](./image-sensor-optics.md) is enough.
- If you are evaluating sensor performance, continue with [Pixel Optical Effects](./pixel-optical-effects.md) and [Quantum Efficiency](./quantum-efficiency.md).
- If you need end-to-end image quality predictions, read [Signal Chain](./signal-chain.md), [Color Reproduction](./color-reproduction.md), and [Noise, SNR, and Dynamic Range](./noise-and-snr.md).
- If you are reporting against an external standard, see [EMVA 1288 Characterization](./emva1288.md).
- If you are choosing a pixel pitch or evaluating sharpness, see [Resolution, MTF, and Pixel Scaling](./resolution-and-mtf.md).

::: tip Prerequisites
This chapter assumes you are comfortable with the [Optics](../optics/) chapter — at minimum, refractive index, absorption, and polarization.
:::
