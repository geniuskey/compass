---
outline: deep
---

# CRA Cone Illumination Sweep

_Generated on 2026-06-11 from `ConeIlluminationRunner` and the real PixelStack path._

This report validates the cone-illumination workflow before using it for larger edge-of-sensor studies. It separates cheap planar-stack integration checks from a low-order patterned `torcwa` smoke run.

## Executive summary

- TMM cone integration was swept over 5 sampling methods and 4 sample counts; the worst sampled max |A-A_ref| was 0.0068, and the best 49-point result was 2.72e-05.
- CRA/F-number maps were generated for CRA 0, 10, 20, and 30 deg across F/1.4, F/2.0, F/2.8, and F/4.0 using a 49-point Hammersley cone.
- The patterned `torcwa` smoke run used F/2.0, five angular samples, TE polarization, and 550 nm. The auto-shift mean-QE delta ranged from -0.0025 to 7.54e-05 in this low-order check.

::: warning Scope
The `torcwa` section is a low-order path check, not a converged edge-pixel design result. Use it to verify that CRA and microlens shift are wired into the solver path, then increase Fourier order and cone samples for production.
:::

## Cone Sampling Maps

![Cone sampling maps](/reports/cra-cone/01_cone_sampling_maps.png)

The red cross marks the chief ray. Marker area follows the normalized cone integration weight.

## TMM Integration Convergence

![TMM cone convergence](/reports/cra-cone/02_tmm_cone_convergence.png)

Reference: TMM, CRA 20 deg, F/2.0, 181-point Hammersley cone, wavelengths 450/550/650 nm.

| sampling | 5 pts | 13 pts | 25 pts | 49 pts |
| --- | --- | --- | --- | --- |
| fibonacci | 0.0026 | 0.0013 | 0.0009 | 2.72e-05 |
| rings | 0.0067 | 0.0059 | 0.0011 | 0.0005 |
| halton | 0.0051 | 0.0057 | 0.003 | 0.0012 |
| hammersley | 0.0068 | 0.0024 | 0.0002 | 0.0002 |
| grid | 0.0009 | 0.0009 | 0.0001 | 2.77e-05 |

## CRA and F-number Response

![TMM CRA F-number response](/reports/cra-cone/03_tmm_cra_fnumber_response.png)

| CRA | min A@550 over F/# | max A@550 over F/# | range |
| --- | --- | --- | --- |
| 0 | 0.2012 | 0.2031 | 0.0019 |
| 10 | 0.2013 | 0.2109 | 0.0097 |
| 20 | 0.2128 | 0.2309 | 0.0181 |
| 30 | 0.2462 | 0.2574 | 0.0112 |

## Patterned torcwa Smoke

![torcwa CRA shift smoke](/reports/cra-cone/04_torcwa_cra_shift_smoke.png)

| CRA | shift | R@550 | T@550 | A@550 | mean QE@550 | QE_R | QE_G | QE_B | energy residual | runtime s | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | none | 0.0317 | 0.0007 | 0.9676 | 0.0058 | 0.0053 | 0.0061 | 0.0056 | 0 | 3.2284 | - |
| 10 | none | 0.0306 | 0.0007 | 0.9687 | 0.0058 | 0.0053 | 0.0061 | 0.0056 | 0 | 1.3253 | - |
| 20 | none | 0.0282 | 0.0007 | 0.9711 | 0.0057 | 0.0053 | 0.006 | 0.0056 | 0 | 1.3186 | - |
| 30 | none | 0.0231 | 0.0007 | 0.9762 | 0.0056 | 0.0052 | 0.0059 | 0.0055 | 2.22e-16 | 1.317 | - |
| 0 | auto_cra | 0.0317 | 0.0007 | 0.9676 | 0.0058 | 0.0053 | 0.0061 | 0.0056 | 0 | 1.365 | - |
| 10 | auto_cra | 0.0265 | 0.0007 | 0.9728 | 0.0058 | 0.0054 | 0.0062 | 0.0056 | 0 | 1.3738 | - |
| 20 | auto_cra | 0.0215 | 0.0006 | 0.9778 | 0.0052 | 0.0052 | 0.0055 | 0.0045 | 2.22e-16 | 1.3694 | - |
| 30 | auto_cra | 0.0193 | 0.0004 | 0.9803 | 0.0031 | 0.0032 | 0.0036 | 0.0021 | 2.22e-16 | 1.2832 | - |

## Interpretation

- In this symmetric planar TMM gate, grid and Fibonacci both converge tightly by 49 samples. For patterned RCWA workflows, low-discrepancy sampling remains the safer default because it avoids structured angular bias.
- TMM isolates the angular-integration behavior from lateral pixel geometry. That makes it useful for convergence gates, but it does not model microlens focus or crosstalk.
- The `torcwa` smoke run exercises the actual patterned PixelStack path. The numbers are intentionally labeled as QE proxies because the Fourier order and sample count are deliberately small.

## Regeneration

```powershell
uv run python scripts\generate_cra_cone_report.py
```

Generated metrics are stored at `docs/public/reports/cra-cone/cra_cone_metrics.json`.
