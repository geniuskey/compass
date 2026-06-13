---
outline: deep
---

# RCWA Backend Parity Report

_Generated on 2026-06-11 from the real `PixelStack` -> RCWA adapter path._

This report runs a small 1.0 um BSI PixelStack through each RCWA backend adapter with the same TE source and wavelength list. It is a backend-health gate, not a converged optical benchmark: the Fourier orders are deliberately small so adapter failures and normalization mismatches are visible quickly.

## Executive summary

- 2 of 4 RCWA adapters produced bounded, energy-consistent R/T/A values in this smoke test.
- 1 of 4 adapters met the parity target against `torcwa` (max R/T/A delta <= 0.05).
- Current result: `torcwa` is the usable reference path for patterned PixelStack runs; the other adapters need normalization/API work before they should be used for production pixel comparisons.

::: warning Low-order smoke test
This report does not claim converged QE. It checks whether adapters can run the same patterned stack and return plausible R/T/A numbers. Converged solver settings belong in the convergence report.
:::

## R/T/A spectra

![RCWA backend spectra](/reports/rcwa-backend-parity/01_backend_rta_spectra.png)

## Health and runtime

![RCWA backend health](/reports/rcwa-backend-parity/02_backend_health_runtime.png)

## Backend table

| Backend | physical ok | parity ok | R@550 | T@550 | A@550 | mean QE@550 | energy residual | max delta vs torcwa | runtime s | first warning/error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| torcwa | yes | yes | 0.0342 | 0.0007 | 0.9652 | 0.0057 | 0 | 0 | 0.9607 | - |
| grcwa | yes | no | 0 | 0 | 1 | 0.9101 | 0 | 0.1137 | 0.05 | - |
| meent/numpy | no | no | 180.8335 | 0.8235 | 0 | 0 | 332.3061 | n/a | 3.7173 | meent: R+T=16.5237 > 1 at λ=0.4500um (numerical instability for multi-layer 2... |
| fmmax | no | no | 0 | 0 | 0 | 0 | 1 | n/a | 0.4479 | fmmax failed at lambda=0.4500um, pol=TE: 'Expansion' object has no attribute ... |

## Interpretation

- `physical ok` means finite, bounded R/T/A values and |R+T+A-1| <= 0.05.
- `parity ok` means physical output and max R/T/A delta against `torcwa` <= 0.05.
- `mean QE` is the adapter's photodiode-allocation proxy averaged over the four Bayer pixels. It is useful for detecting broken absorption allocation, not for design decisions at this low order.
- A backend can import successfully but still fail this report if the adapter uses stale upstream APIs or incompatible normalization.

## Regeneration

```powershell
uv run python scripts\generate_rcwa_backend_parity_report.py
```

Generated metrics are stored at `docs/public/reports/rcwa-backend-parity/rcwa_backend_parity_metrics.json`.
