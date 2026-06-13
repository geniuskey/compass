---
outline: deep
---

# BARL Optimization Benchmark 리포트

_생성일: 2026-06-11. generic 1.0 um BSI stack에서 COMPASS TMM solver로 생성._

이 리포트는 BARL cookbook recipe를 재현 가능한 benchmark로 승격한다. single-layer Si3N4 두께, two-layer SiO2/HfO2 grid를 sweep하고, 대표 design을 같은 400-700 nm wavelength grid에서 비교한다.

## 요약

- `cf_green`과 silicon 사이의 550 nm ideal ARC index는 2.5148이고, Si3N4 quarter-wave thickness는 67.9612 nm다.
- single-layer Si3N4 sweep 최적점은 70 nm이며 mean R=0.0604다.
- two-layer SiO2/HfO2 grid 최적점은 SiO2 5 nm / HfO2 65 nm이며 mean R=0.0627다.
- 이 candidate set의 최선 design은 no BARL 대비 mean reflection을 0.029 absolute 줄인다.

::: warning Planar proxy
이 리포트는 TMM planar green-stack benchmark다. BARL screening과 reflection trend에는 적합하지만 lateral Bayer geometry, microlens focusing, metal-grid diffraction, crosstalk는 포함하지 않는다.
:::

## Candidate spectra

![BARL design spectra](/reports/barl-optimization/01_barl_design_spectra.png)

## Single-layer sweep

![Single-layer BARL sweep](/reports/barl-optimization/02_single_layer_sweep.png)

## Two-layer sweep

![Two-layer BARL heatmap](/reports/barl-optimization/03_hfo2_sio2_heatmap.png)

## Design scorecard

![BARL scorecard](/reports/barl-optimization/04_barl_design_scorecard.png)

| Design | role | layers | total nm | mean R | max R | R@550 | mean A | energy residual |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| No BARL | baseline | none | 0 | 0.0895 | 0.3656 | 0.0091 | 0.6492 | 1.11e-16 |
| Default 4-layer | sample default | sio2 10 nm / hfo2 25 nm / sio2 15 nm / si3n4 30 nm | 80 | 0.0698 | 0.2213 | 0.2213 | 0.6345 | 1.11e-16 |
| Si3N4 quarter-wave | analytic | si3n4 68 nm | 67.9612 | 0.0605 | 0.1788 | 0.1788 | 0.6274 | 1.11e-16 |
| Best single Si3N4 | sweep best | si3n4 70 nm | 70 | 0.0604 | 0.1766 | 0.1766 | 0.6272 | 0 |
| Best SiO2/HfO2 | sweep best | sio2 5 nm / hfo2 65 nm | 70 | 0.0627 | 0.195 | 0.195 | 0.6294 | 1.11e-16 |

## 해석

- BARL tuning은 R@550 하나가 아니라 broadband reflection을 최적화해야 한다. Quarter-wave layer는 좋은 seed지만 color-filter와 planarization phase가 포함되면 자동으로 broadband optimum이 되지는 않는다.
- 기본 sample BARL은 예시 process stack이지 보장된 optimum이 아니다. 이 리포트는 간단한 sweep alternative와 비교해 그 점을 명시한다.
- two-layer optimum은 SiO2 sweep의 lower boundary에 걸려 있다. 다음 local search에서는 더 얇은 SiO2 또는 HfO2-only variant를 확인하는 것이 좋다.
- TMM BARL candidate를 고른 뒤에는 patterned RCWA check를 돌려야 한다. metal grid와 microlens가 apparent optimum을 이동시킬 수 있기 때문이다.

## 재생성

```powershell
uv run python scripts\generate_barl_optimization_report.py
```

생성 metric은 `docs/public/reports/barl-optimization/barl_optimization_metrics.json`에 저장된다.
