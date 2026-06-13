---
outline: deep
---

# CRA Cone Illumination Sweep 리포트

_생성일: 2026-06-11. `ConeIlluminationRunner`와 실제 PixelStack 경로에서 생성._

이 리포트는 더 큰 sensor edge 연구에 쓰기 전에 cone illumination workflow를 검증한다. 빠른 planar-stack integration check와 낮은 order의 patterned `torcwa` smoke run을 분리했다.

## 요약

- TMM cone integration은 5개 sampling method와 4개 sample count로 sweep했다. sampled max |A-A_ref| 최대값은 0.0068, 49-point 최선 결과는 2.72e-05다.
- CRA/F-number map은 CRA 0, 10, 20, 30 deg와 F/1.4, F/2.0, F/2.8, F/4.0에 대해 49-point Hammersley cone으로 생성했다.
- patterned `torcwa` smoke run은 F/2.0, angular sample 5개, TE, 550 nm 조건이다. 이 low-order check에서 auto-shift mean-QE delta는 -0.0025부터 7.54e-05까지였다.

::: warning 범위
`torcwa` 섹션은 low-order path check이며, converged edge-pixel 설계 결과가 아니다. CRA와 microlens shift가 solver path에 연결되는지 확인한 뒤, production에서는 Fourier order와 cone sample을 늘려야 한다.
:::

## Cone sampling map

![Cone sampling maps](/reports/cra-cone/01_cone_sampling_maps.png)

빨간 십자는 chief ray다. marker 면적은 normalized cone integration weight를 따른다.

## TMM integration convergence

![TMM cone convergence](/reports/cra-cone/02_tmm_cone_convergence.png)

Reference: TMM, CRA 20 deg, F/2.0, 181-point Hammersley cone, wavelength 450/550/650 nm.

| sampling | 5 pts | 13 pts | 25 pts | 49 pts |
| --- | --- | --- | --- | --- |
| fibonacci | 0.0026 | 0.0013 | 0.0009 | 2.72e-05 |
| rings | 0.0067 | 0.0059 | 0.0011 | 0.0005 |
| halton | 0.0051 | 0.0057 | 0.003 | 0.0012 |
| hammersley | 0.0068 | 0.0024 | 0.0002 | 0.0002 |
| grid | 0.0009 | 0.0009 | 0.0001 | 2.77e-05 |

## CRA 및 F-number 응답

![TMM CRA F-number response](/reports/cra-cone/03_tmm_cra_fnumber_response.png)

| CRA | min A@550 over F/# | max A@550 over F/# | range |
| --- | --- | --- | --- |
| 0 | 0.2012 | 0.2031 | 0.0019 |
| 10 | 0.2013 | 0.2109 | 0.0097 |
| 20 | 0.2128 | 0.2309 | 0.0181 |
| 30 | 0.2462 | 0.2574 | 0.0112 |

## Patterned torcwa smoke

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

## 해석

- 이 대칭 planar TMM gate에서는 grid와 Fibonacci가 49 sample에서 모두 잘 수렴한다. Patterned RCWA workflow에서는 structured angular bias를 피하기 위해 low-discrepancy sampling을 기본값으로 두는 편이 안전하다.
- TMM은 angular integration behavior를 lateral pixel geometry에서 분리한다. 따라서 convergence gate에는 유용하지만 microlens focus나 crosstalk는 모델링하지 않는다.
- `torcwa` smoke run은 실제 patterned PixelStack 경로를 실행한다. Fourier order와 sample count가 의도적으로 낮기 때문에 숫자는 QE proxy로 표시한다.

## 재생성

```powershell
uv run python scripts\generate_cra_cone_report.py
```

생성 metric은 `docs/public/reports/cra-cone/cra_cone_metrics.json`에 저장된다.
