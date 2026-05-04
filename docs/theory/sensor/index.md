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
| [Quantum Efficiency](./quantum-efficiency.md) | QE definition, computation methods, crosstalk | The headline metric and how COMPASS measures it |
| [Signal Chain](./signal-chain.md) | Illuminant → scene → lens → sensor signal | How simulated QE turns into a radiometric pixel value |

## How to read this chapter

- If you only care about the optical design of the pixel, [Image Sensor Optics](./image-sensor-optics.md) is enough.
- If you are evaluating sensor performance, continue with [Quantum Efficiency](./quantum-efficiency.md).
- If you need end-to-end image quality predictions (color accuracy, SNR), read [Signal Chain](./signal-chain.md) as well.

::: tip Prerequisites
This chapter assumes you are comfortable with the [Optics](../optics/) chapter — at minimum, refractive index, absorption, and polarization.
:::
