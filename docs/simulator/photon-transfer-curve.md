---
title: "Photon Transfer Curve (PTC)"
---

# Photon Transfer Curve (PTC)

Visualize the fundamental noise-vs-signal relationship to extract read noise, conversion gain, full well capacity, and PRNU from a single log-log plot.

<PhotonTransferCurve />

## Physics

The Photon Transfer Curve plots total noise (standard deviation) against mean signal on a log-log scale. Three distinct noise regimes emerge:

### Read Noise Region (flat)
At low signal levels, noise is dominated by the constant read noise floor of the readout electronics. The PTC is flat at sigma = sigma_read.

### Shot Noise Region (slope 0.5)
As signal increases, photon shot noise (Poisson statistics) dominates. Since sigma_shot = sqrt(N), the log-log slope is exactly 0.5. The conversion gain is extracted from this region.

### PRNU Region (slope 1.0)
At high signal levels, pixel-to-pixel gain variation (PRNU) dominates. Since sigma_PRNU = PRNU x N, the slope becomes 1.0, and SNR saturates.

### Total Noise Model

The total noise variance combines all sources:

**sigma_total^2 = sigma_read^2 + N_signal + (PRNU x N_signal)^2**

::: tip
The crossover points between regions are key diagnostic parameters. A high shot/read crossover indicates excessive read noise, while a low PRNU/shot crossover indicates poor manufacturing uniformity.
:::
