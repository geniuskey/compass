---
outline: deep
---

# RCWA Backend Parity 리포트

_생성일: 2026-06-11. 실제 `PixelStack` -> RCWA adapter 경로에서 생성._

이 리포트는 작은 1.0 um BSI PixelStack을 동일한 TE source와 wavelength list로 각 RCWA backend adapter에 실행한다. 목적은 backend-health gate다. Fourier order는 의도적으로 낮게 잡아 adapter failure와 normalization mismatch가 빨리 보이게 했다.

## 요약

- 4개 RCWA adapter 중 2개가 bounded 및 energy-consistent R/T/A 값을 냈다.
- 4개 중 1개가 `torcwa` 대비 parity target (max R/T/A delta <= 0.05)을 만족했다.
- 현재 결과에서 patterned PixelStack run의 usable reference path는 `torcwa`다. 나머지 adapter는 production pixel comparison 전에 normalization/API 보수가 필요하다.

::: warning Low-order smoke test
이 리포트는 converged QE를 주장하지 않는다. adapter가 같은 patterned stack을 실행하고 그럴듯한 R/T/A 값을 반환하는지 확인한다. 수렴 설정은 convergence report에서 다룬다.
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

## 해석

- `physical ok`는 finite, bounded R/T/A 값과 |R+T+A-1| <= 0.05를 의미한다.
- `parity ok`는 physical output이면서 `torcwa` 대비 max R/T/A delta <= 0.05임을 의미한다.
- `mean QE`는 네 Bayer pixel에 대한 adapter의 photodiode-allocation proxy 평균이다. 이 낮은 order에서는 설계 판단이 아니라 broken absorption allocation 감지용이다.
- backend import가 성공해도 upstream API 변화나 normalization mismatch가 있으면 이 리포트에서 실패할 수 있다.

## 재생성

```powershell
uv run python scripts\generate_rcwa_backend_parity_report.py
```

생성 metric은 `docs/public/reports/rcwa-backend-parity/rcwa_backend_parity_metrics.json`에 저장된다.
