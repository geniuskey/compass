---
title: Resolution, MTF, and Pixel Scaling
description: Formal reference for spatial resolution metrics — pixel aperture MTF, optical diffraction MTF, Nyquist sampling, aliasing, and pixel-pitch scaling trade-offs.
---

# Resolution, MTF, and Pixel Scaling

::: tip Prerequisites
[Image Sensor Optics](./image-sensor-optics.md) -> [Quantum Efficiency](./quantum-efficiency.md) -> [Noise, SNR, and Dynamic Range](./noise-and-snr.md) -> this page.
:::

::: info Scope of this page
This page is the formal reference for spatial resolution in COMPASS — modulation transfer function (MTF) of the pixel aperture and the optical system, sampling and aliasing through the Nyquist frequency, and how pitch scaling propagates to FWC, SNR, diffraction, and resolution. Interactive visualizations live in the [Simulator](/simulator/) section.
:::

A sensor's QE alone says nothing about how sharp the image looks. Two pixel designs with the same QE but different aperture shapes, fill factors, or pitches produce visibly different images. The modulation transfer function (MTF) captures this — it measures how well sinusoidal scene contrast at spatial frequency $f$ survives the imaging chain.

## Spatial sampling and Nyquist frequency

A pixel grid with pitch $p$ samples the optical image at intervals of $p$ in each axis. By the sampling theorem, the highest spatial frequency that can be represented without aliasing is

$$f_\text{Nyquist} = \frac{1}{2p}$$

in cycles per micrometer (or line pairs per millimeter when scaled). Frequencies above $f_\text{Nyquist}$ in the optical image fold back into the passband as aliased lower-frequency content and cannot be removed after the fact. Optical low-pass filters (OLPF) attenuate above-Nyquist content; modern sensors often replace the OLPF with a demosaic pipeline that tolerates moderate aliasing.

## Pixel aperture MTF

A square pixel aperture of side $a$ acts as a spatial low-pass filter. Its MTF is the magnitude of the Fourier transform of a rect function:

$$\text{MTF}_\text{pixel}(f_x, f_y) = |\operatorname{sinc}(\pi a f_x)\,\operatorname{sinc}(\pi a f_y)|$$

For a fully filled square pixel ($a = p$) the MTF along one axis is $|\operatorname{sinc}(\pi p f)|$, which reaches $\approx 0.637$ at the Nyquist frequency. A reduced fill factor ($a < p$) increases the value of $\operatorname{sinc}$ at a given $f$ (the aperture is smaller, so its frequency response is broader), but at the cost of QE because less photosensitive area collects light. This is one of the central trade-offs in pixel design.

## Optical diffraction MTF

A diffraction-limited circular lens with f-number $F\#$ at wavelength $\lambda$ has a cutoff frequency

$$f_c = \frac{1}{\lambda F\#}$$

and a diffraction MTF given by the autocorrelation of the pupil (the "chat" function):

$$\text{MTF}_\text{optics}(f) = \frac{2}{\pi}\!\left[\arccos\!\left(\frac{f}{f_c}\right) - \frac{f}{f_c}\sqrt{1 - \left(\frac{f}{f_c}\right)^2}\right]\!, \quad 0 \le f \le f_c.$$

Above $f_c$ the optical MTF is identically zero — no information at those frequencies reaches the sensor. The Airy disk diameter for the first zero is $2.44\,\lambda F\#$; this is the natural length scale to compare against pixel pitch.

## System MTF

The pixel and optics are approximately separable, so the system MTF is the product:

$$\text{MTF}_\text{system}(f) = \text{MTF}_\text{pixel}(f) \cdot \text{MTF}_\text{optics}(f) \cdot \text{MTF}_\text{defocus}(f) \cdot \ldots$$

with additional factors for defocus, motion blur, and on-chip color filter array (CFA) demosaic. The headline number reported for a sensor is usually $\text{MTF}_\text{system}(f_\text{Nyquist})$ — a single value summarizing how much contrast remains at the spatial frequency where aliasing onsets. Values above $\approx 0.5$ are considered crisp; below $\approx 0.2$ the image looks soft.

## Pitch scaling trade-offs

Pixel pitch is the single most consequential design parameter. The relevant scaling laws, with pitch $p$ as the independent variable, are:

| Metric | Scaling | Origin |
|---|---|---|
| Full well capacity | $\text{FWC} \propto p^2$ | Photodiode area |
| Photon collection per pixel | $\propto p^2$ | Aperture area |
| Maximum SNR | $\text{SNR}_\text{max} \propto p$ | $\sqrt{\text{FWC}}$ |
| Nyquist frequency | $f_\text{Nyquist} \propto 1/p$ | Sampling rate |
| Diffraction-limited fraction | $\propto (p / (\lambda F\#))^2$ | Airy area vs pixel area |

The interpretation is direct: shrinking the pixel by $2\times$ quadruples spatial sampling density but loses 1 stop of SNR and rapidly enters the diffraction-limited regime. Below roughly $p \approx 0.6$ µm the Airy disk for green light at f/2.0 exceeds the pixel area, energy spreads to neighbors as **diffraction crosstalk**, and per-pixel QE drops.

Compensations available to sensor designers include:

- **Microlenses** that increase effective fill factor toward 1
- **BSI architecture** that moves wiring off the optical path
- **Deep trench isolation (DTI)** that reduces lateral diffusion crosstalk
- **Pixel binning** that trades spatial resolution back for SNR after the fact

None of these defeats the underlying physical scaling — they only push the practical floor lower.

## Boundary with adjacent pages

| Adjacent page | Difference |
|---|---|
| [Image Sensor Optics](./image-sensor-optics.md) | Describes the optical stack and how DTI, microlens, and BARL physically affect resolution. This page treats those effects as the spatial frequency response. |
| [Pixel Optical Effects](./pixel-optical-effects.md) | Covers angular and polarization response, CRA, and spectral crosstalk. The "spatial crosstalk" measured here is geometric; this page does not re-derive optical crosstalk. |
| [Quantum Efficiency](./quantum-efficiency.md) | QE is a per-wavelength scalar; MTF is a per-spatial-frequency function. The two describe orthogonal aspects of the same pixel. |

## Browser simulators

- [MTF Analyzer](/simulator/mtf-analyzer) — pixel-aperture MTF, optical diffraction MTF, and the combined system MTF
- [Pixel Scaling Trends](/simulator/pixel-scaling) — FWC, $f_\text{Nyquist}$, SNR, and diffraction-limited QE as pitch is swept
