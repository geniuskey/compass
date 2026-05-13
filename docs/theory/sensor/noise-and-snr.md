---
title: Noise, SNR, and Dynamic Range
description: Formal reference for image sensor noise sources, signal-to-noise ratio, full well capacity, dynamic range, photon transfer curve, PRNU/DSNU, dark current, and linearity.
---

# Noise, SNR, and Dynamic Range

::: tip Prerequisites
[Quantum Efficiency](./quantum-efficiency.md) -> [Signal Chain](./signal-chain.md) -> this page.
:::

::: info Scope of this page
This page is the formal noise reference for COMPASS. It defines each noise source, the signal-to-noise ratio, full well capacity, dynamic range, photon transfer curve, fixed pattern noise (PRNU/DSNU), dark current, and linearity. Interactive tools that visualize these relationships live in the [Simulator](/simulator/) section.
:::

A pixel's signal is meaningful only relative to the noise on top of it. The same QE design can look excellent or unusable depending on the read noise floor, the integration time, and the operating temperature. This page collects the standard noise model used throughout COMPASS and its browser simulators.

## Signal model

Let $N_\text{ph}$ be the number of incident photons per pixel during exposure time $t_\text{int}$. The collected signal in electrons is

$$N = \text{QE} \cdot N_\text{ph} + I_\text{d}(T)\,t_\text{int}$$

where $I_\text{d}(T)$ is the dark current in electrons per second at temperature $T$. The first term is photogenerated charge, the second is thermally generated charge. The full well capacity (FWC) is the upper limit on $N$; beyond it the pixel saturates.

## Noise sources

The total noise on the readout is the quadrature sum of statistically independent contributions:

$$\sigma_\text{total}^2 = \sigma_\text{shot}^2 + \sigma_\text{dark}^2 + \sigma_\text{read}^2 + \sigma_\text{PRNU}^2 + \sigma_\text{DSNU}^2$$

### Shot noise (photon noise)

Photon arrival is Poisson-distributed, so for an average signal of $N$ electrons:

$$\sigma_\text{shot} = \sqrt{N}$$

Shot noise is fundamental and cannot be reduced by sensor design — only by collecting more photons (larger pixel, longer exposure, higher QE).

### Dark current noise

Dark current is itself Poisson:

$$\sigma_\text{dark} = \sqrt{I_\text{d}(T)\,t_\text{int}}$$

Dark current follows an Arrhenius dependence on temperature:

$$I_\text{d}(T) \propto T^{3/2}\,\exp\!\left(-\frac{E_g}{2 k_B T}\right)$$

with $E_g \approx 1.12$ eV for silicon. A common engineering rule of thumb is that dark current approximately doubles for every 6–8 °C of temperature rise. Cooling by 20 °C therefore reduces dark current by roughly 8×.

### Read noise

Read noise $\sigma_\text{read}$ is the noise added by the readout chain (source follower, column amplifier, ADC) and is approximately signal-independent. It dominates at low light.

### Fixed pattern noise: PRNU and DSNU

Fixed pattern noise (FPN) is a pixel-to-pixel variation that does not change frame to frame. It splits into two pieces.

**PRNU (photo response non-uniformity)** is gain variation proportional to signal:

$$\sigma_\text{PRNU} = u_\text{PRNU}\,N$$

with $u_\text{PRNU} \approx 0.5\%$–$2\%$ in modern sensors. It comes from variations in microlens alignment, photodiode doping, and pixel geometry.

**DSNU (dark signal non-uniformity)** is offset variation independent of signal:

$$\sigma_\text{DSNU} = \text{constant in } e^-\text{ RMS}$$

caused by per-pixel variation in dark current from crystal defects and interface traps.

Total FPN is:

$$\sigma_\text{FPN} = \sqrt{\sigma_\text{PRNU}^2 + \sigma_\text{DSNU}^2}$$

FPN can be partially removed by per-pixel flat-field and dark-frame calibration; the residual sets a floor on uniformity-limited applications.

## Signal-to-noise ratio

The SNR at signal level $N$ is

$$\text{SNR}(N) = \frac{N}{\sqrt{\sigma_\text{read}^2 + N + (u_\text{PRNU}\,N)^2}}$$

(absorbing dark and DSNU into the read-noise term when relevant). It has three regimes:

| Regime | Dominant noise | SNR scaling |
|---|---|---|
| Low light | Read noise | $\text{SNR} \propto N$ |
| Mid range | Shot noise | $\text{SNR} \propto \sqrt{N}$ |
| High signal | PRNU | $\text{SNR}$ saturates at $1/u_\text{PRNU}$ |

Reported on a logarithmic scale:

$$\text{SNR}_\text{dB} = 20 \log_{10}\!\left(\frac{N}{\sigma_\text{total}}\right)$$

Common reference points: SNR = 0 dB (signal equals noise, the absolute detection limit), SNR = 20 dB (often used as the minimum for acceptable image quality), SNR$_\text{max} = \sqrt{\text{FWC}}$ in the shot-noise-limited limit ignoring PRNU.

## Full well capacity and dynamic range

FWC is the maximum signal in electrons a pixel can hold before saturating. It scales roughly with pixel area:

$$\text{FWC} \propto \text{pitch}^2$$

Dynamic range (DR) is the ratio of FWC to the noise floor at minimum integration:

$$\text{DR}_\text{dB} = 20 \log_{10}\!\left(\frac{\text{FWC}}{\sigma_\text{floor}}\right), \quad
\sigma_\text{floor} = \sqrt{\sigma_\text{read}^2 + I_\text{d}(T)\,t_\text{int}}$$

Equivalent expression in stops: $\text{DR}_\text{stops} = \text{DR}_\text{dB} / 6.02$. Multi-exposure HDR extends DR by combining a long exposure (for shadows) with a short exposure (for highlights):

$$\text{DR}_\text{HDR} = 20 \log_{10}\!\left(\frac{\text{FWC}\,r}{\sigma_\text{floor}}\right)$$

where $r$ is the exposure ratio. The achievable gain is limited by motion artifacts between frames.

## Photon transfer curve

The photon transfer curve (PTC) plots total noise versus mean signal on a log–log scale and is the canonical way to extract conversion gain, read noise, FWC, and PRNU from measurements. Three regions are visible:

| Region | Slope on log–log | Extracted parameter |
|---|---|---|
| Flat floor | 0 | Read noise $\sigma_\text{read}$ |
| Shot noise | $1/2$ | Conversion gain $K$ (from $\sigma^2 = K\,S$) |
| PRNU | $1$ | $u_\text{PRNU}$ |

The total variance model is:

$$\sigma_\text{total}^2 = \sigma_\text{read}^2 + N + (u_\text{PRNU}\,N)^2$$

The crossover between regions is itself diagnostic: a high read-to-shot crossover means excessive read noise; a low shot-to-PRNU crossover means poor manufacturing uniformity.

## Responsivity

Spectral responsivity converts QE to an electrical metric:

$$R(\lambda) = \text{QE}(\lambda)\,\frac{q\,\lambda}{h c} \approx \text{QE}(\lambda)\,\frac{\lambda_\text{nm}}{1240}\ \ [\text{A/W}]$$

The $\lambda/1240$ factor shifts the responsivity peak to longer wavelengths than the QE peak. The ideal silicon photodiode (QE = 1) has $R_\text{ideal}(\lambda) = \lambda_\text{nm}/1240$.

## Linearity

The ideal transfer function is linear: $\text{DN}_\text{ideal} = (N/\text{FWC})\,\text{DN}_\text{max}$. Real sensors deviate due to source-follower nonlinearity, voltage-dependent junction capacitance, and ADC INL. The standard metric is

$$\text{NL}(\%) = \frac{\max |\text{DN}_\text{actual} - \text{DN}_\text{ideal}|}{\text{DN}_\text{max}} \times 100$$

Most machine vision applications require NL < 1%; HDR merging and photometric work need < 0.5%.

## Browser simulators

These interactive tools visualize the equations above:

- [SNR Calculator](/simulator/snr-calculator) — total noise breakdown and SNR vs operating point
- [Photon Transfer Curve](/simulator/photon-transfer-curve) — extract read noise, conversion gain, and PRNU
- [SNR vs Illuminance](/simulator/pixel-snr-vs-illuminance) — three noise regimes against the ideal shot-noise limit
- [Dynamic Range](/simulator/dynamic-range) — FWC / noise-floor and HDR extension
- [Responsivity Calculator](/simulator/responsivity-calculator) — QE-to-A/W conversion per channel
- [Dark Current](/simulator/dark-current) — Arrhenius model and dark-frame visualization
- [PRNU / DSNU](/simulator/prnu-visualizer) — fixed pattern noise spatial maps
- [Linearity Analyzer](/simulator/linearity-analyzer) — transfer-curve deviation and knee point
