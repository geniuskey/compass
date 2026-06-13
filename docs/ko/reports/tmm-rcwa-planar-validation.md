---
outline: deep
---

# TMM vs RCWA 평면 스택 검증

_생성일: 2026-06-11. `transfer_matrix_1d`와 direct `torcwa` zero-order RCWA solve에서 생성._

이 리포트는 평면 스택 한계만 분리한다. 횡방향 패턴이 없으면 zero-order RCWA는 TMM이 푸는 1D optics와 같은 해로 수렴해야 한다. 따라서 Bayer 패턴, metal grid, DTI trench, microlens를 쓰기 전에 확인하는 첫 번째 검증 단계다.

## 요약

- 네 개의 평면 검증 케이스가 모두 5e-5 R/T/A agreement target을 통과했다. 가장 큰 spectral difference는 **2.59e-06**이다.
- ideal quarter-wave ARC는 550 nm bare silicon reflectance를 **0.36**에서 TMM **2.11e-33**, RCWA **4.63e-15**로 낮춘다.
- 이 리포트는 normal-incidence planar-only 검증이다. lateral diffraction, color-filter relief, DTI crosstalk, photodiode collection은 의도적으로 검증하지 않는다.

## R/T/A alignment

![TMM vs RCWA RTA alignment](/reports/tmm-rcwa-planar/01_rta_alignment.png)

## Error summary

![TMM vs RCWA error summary](/reports/tmm-rcwa-planar/02_error_summary.png)

## Quarter-wave ARC sanity check

![Quarter-wave ARC reflectance](/reports/tmm-rcwa-planar/03_arc_reflectance.png)

## Validation table

| Case | layers | thickness um | max \|dR\| | max \|dT\| | max \|dA\| | RCWA energy residual | passes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Air / glass interface | 0 | 0 | 2.83e-09 | 9.78e-08 | 1.01e-07 | 0 | yes |
| Ideal quarter-wave ARC on silicon | 1 | 0.0688 | 6.15e-08 | 6.85e-07 | 6.93e-07 | 7.45e-08 | yes |
| Lossless pixel-like multilayer | 7 | 0.99 | 1.14e-06 | 1.62e-06 | 2.59e-06 | 4.47e-08 | yes |
| Lossy pixel-like multilayer | 7 | 0.99 | 7.26e-07 | 1.06e-06 | 1.50e-06 | 0 | yes |

## 해석

- single-interface row는 finite film 없이 Fresnel normalization을 확인한다.
- ARC row는 interference phase와 quarter-wave reflectance null을 확인한다.
- lossless multilayer는 pixel-like dielectric ladder의 phase accumulation을 확인한다.
- lossy multilayer는 complex refractive index와 absorption accounting이 맞는지 확인한다.

## 재생성

```powershell
uv run python scripts\generate_tmm_rcwa_planar_report.py
```

생성 metric은 `docs/public/reports/tmm-rcwa-planar/tmm_rcwa_planar_metrics.json`에 저장된다.
