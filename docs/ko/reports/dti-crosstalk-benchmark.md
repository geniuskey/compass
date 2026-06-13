---
outline: deep
---

# DTI Crosstalk Benchmark 리포트

_생성일: 2026-06-11. PixelStack geometry sweep와 representative scalar FDTD snapshot에서 생성._

이 리포트는 빠르게 재생성 가능한 DTI geometry evidence와 비용이 큰 localized-source crosstalk benchmark를 분리한다. 긴 vector FDTD 또는 high-order RCWA를 돌리기 전 design-space gate로 사용하기 위한 문서다.

## 요약

- 100 nm full-depth DTI는 generic 1.0 um 2x2 BSI PixelStack geometry에서 silicon volume의 0.19를 차지한다.
- 100 nm, 0.6 um BDTI는 backside/top 일부만 trench 처리되므로 silicon volume의 0.0393를 차지한다.
- representative 44x44x118, 3500-step scalar FDTD snapshot의 max neighbor crosstalk는 FDTI 0.2582, BDTI 0.2609이며 gap은 0.0027 absolute다.
- periodic trench RCWA/FDTD alignment snapshot은 FDTI와 BDTI 모두에서 R/T/A가 대략 3 percentage point 이내로 맞는다.

::: warning 범위
crosstalk matrix는 scalar FDTD visual benchmark이지 production full-vector FDTD solve가 아니다. geometry와 normalization path 비교용으로 사용하고, 최종 isolation claim에는 더 긴 vector run이 필요하다.
:::

## Geometry sweep

![DTI geometry sweeps](/reports/dti-crosstalk/01_dti_geometry_sweeps.png)

### FDTI width sweep

| width nm | max XY DTI area | effective DTI volume | open Si volume |
| --- | --- | --- | --- |
| 0 | 0 | 0 | 1 |
| 40 | 0.0784 | 0.0784 | 0.9216 |
| 60 | 0.1164 | 0.1164 | 0.8836 |
| 80 | 0.1536 | 0.1536 | 0.8464 |
| 100 | 0.19 | 0.19 | 0.81 |
| 120 | 0.2256 | 0.2256 | 0.7744 |
| 150 | 0.2775 | 0.2775 | 0.7225 |

### BDTI depth sweep

| BDTI depth um | active depth um | effective DTI volume | open Si volume |
| --- | --- | --- | --- |
| 0 | 0 | 0 | 1 |
| 0.3 | 0.3 | 0.0197 | 0.9803 |
| 0.6 | 0.6 | 0.0393 | 0.9607 |
| 1.2 | 1.2 | 0.0786 | 0.9214 |
| 1.8 | 1.8 | 0.1179 | 0.8821 |
| 2.4 | 2.4 | 0.1572 | 0.8428 |
| 2.9 | 2.9 | 0.19 | 0.81 |

## Silicon DTI mask

![DTI XY and XZ masks](/reports/dti-crosstalk/02_dti_xz_masks.png)

## Representative crosstalk snapshot

![DTI crosstalk matrices](/reports/dti-crosstalk/03_dti_crosstalk_matrices.png)

![DTI crosstalk summary](/reports/dti-crosstalk/04_dti_crosstalk_summary.png)

| mode | mean self collection | max neighbor crosstalk | mean PD signal | energy tail change |
| --- | --- | --- | --- | --- |
| fdti | 0.5831 | 0.2582 | 15320.3545 | 0.0471 |
| bdti_0p6um | 0.5827 | 0.2609 | 15249.1798 | 0.047 |

## Periodic trench alignment snapshot

| mode | max abs dR | max abs dT | max abs dA | Si absorption proxy | trench field leakage |
| --- | --- | --- | --- | --- | --- |
| fdti | 0.0266 | 0.027 | 0.0278 | 0.6087 | 0.1305 |
| bdti | 0.0183 | 0.0088 | 0.019 | 0.6105 | 0.096 |

## 해석

- 대표 coarse scalar snapshot에서 FDTI와 BDTI의 crosstalk가 비슷한 이유는 이 run이 주로 path와 normalization check 역할을 하기 때문이다.
- geometry sweep은 의도한 monotonic control을 보여준다. FDTI가 넓어질수록 silicon trench volume이 증가하고, BDTI가 깊어질수록 FDTI에 가까워진다.
- production DTI report는 wavelength-resolved localized-source vector FDTD와 실제 width/depth/material crosstalk sweep으로 확장해야 한다.

## 재생성

```powershell
uv run python scripts\generate_dti_crosstalk_report.py
```

생성 metric은 `docs/public/reports/dti-crosstalk/dti_crosstalk_metrics.json`에 저장된다.
