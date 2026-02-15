---
title: "Dynamic Range Calculator"
---

# Dynamic Range Calculator

Calculate sensor dynamic range from full well capacity, read noise, dark current, and exposure settings. Compare single exposure vs HDR modes across operating temperatures.

<DynamicRangeCalculator />

## Dynamic Range Definition

Dynamic range is the ratio between the maximum signal a sensor can capture (saturation) and the minimum detectable signal (noise floor):

**DR = 20 x log10(N_sat / sigma_floor)** [dB]

**DR_stops = DR / 6.02** [EV]

### Noise Floor

The minimum detectable signal is limited by the combined noise floor:

**sigma_floor = sqrt(sigma_read^2 + I_dark x t_exp)**

### Temperature Dependence

Dark current follows an Arrhenius model, approximately doubling every 5.5 degrees C. This makes dynamic range strongly temperature-dependent, especially for long exposures.

### HDR Extension

Multi-exposure HDR extends dynamic range by combining a long exposure (for shadows) with a short exposure (for highlights):

**DR_HDR = 20 x log10(N_sat x ratio / sigma_floor)**

::: warning
HDR gain is limited by the exposure ratio and may introduce motion artifacts between frames.
:::
