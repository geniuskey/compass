---
outline: deep
---

# 픽셀 스택 Geometry 감사

_생성일: 2026-06-06. `compass.geometry.sample_pixels`와 `PixelStack`에서 생성._

이 리포트는 광학 성능이 아니라 geometry evidence를 게시한다. 대표 sample pixel preset이 color-filter relief, metal-grid thickness, DTI, microlens, photodiode window를 포함한 solver 입력 stack으로 실제 확장되는지 확인한다.

## 요약

- 모든 감사 대상 preset은 단일 flat slab가 아니라 color-filter relief slice를 생성한다.
- 모든 preset에서 color-filter stack 높이는 가장 높은 RGB channel과 metal grid를 덮는다.
- photodiode x-y window는 모든 preset에서 pixel pitch 안에 있다.
- 이 리포트는 QE 또는 crosstalk 변화를 주장하지 않는다. optical sweep 전에 시뮬레이션되는 geometry를 확인하는 용도다.

## Geometry overview

![PixelStack geometry overview](/reports/geometry/pixel-stack-audit/sample_stack_overview.png)

## 감사 대상 preset

| Preset | pitch um | unit cell | CF stack um | grid um | R/G/B CF um | min angle | CF slices | PD xy fill |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Generic 1.0 um BSI | 1 | 2x2 | 0.648 | 0.468 | 0.624/0.6/0.648 | 62 | 11 | 0.49 |
| 0.56 um 4x4 OCL | 0.56 | 8x8 | 0.594 | 0.429 | 0.572/0.55/0.594 | 62 | 11 | 0.49 |
| 1.0 um Quad Bayer | 1 | 4x4 | 0.648 | 0.468 | 0.624/0.6/0.648 | 62 | 11 | 0.49 |
| 1.22 um 2x2 OCL | 1.22 | 4x4 | 0.756 | 0.546 | 0.728/0.7/0.756 | 62 | 11 | 0.49 |
| 1.6 um split PD | 1.6 | 2x2 | 0.94 | 0.679 | 0.905/0.87/0.94 | 62 | 11 | 0.774 |
| 1.2 um LOFIC | 1.2 | 4x4 | 0.745 | 0.538 | 0.718/0.69/0.745 | 62 | 11 | 0.423 |
| 1.12 um NIR (IPA + lined DTI) | 1.12 | 2x2 | 0.702 | 0.507 | 0.676/0.65/0.702 | 62 | 11 | 0.49 |

## Checks

| Preset | CF covers channels | grid <= stack | multi-slice relief | PD inside pixel |
| --- | --- | --- | --- | --- |
| Generic 1.0 um BSI | yes | yes | yes | yes |
| 0.56 um 4x4 OCL | yes | yes | yes | yes |
| 1.0 um Quad Bayer | yes | yes | yes | yes |
| 1.22 um 2x2 OCL | yes | yes | yes | yes |
| 1.6 um split PD | yes | yes | yes | yes |
| 1.2 um LOFIC | yes | yes | yes | yes |
| 1.12 um NIR (IPA + lined DTI) | yes | yes | yes | yes |

## 재생성

```powershell
uv run python scripts\generate_geometry_reports.py
```

생성 metric은 `docs/public/reports/geometry/pixel-stack-audit/geometry_metrics.json`에 저장된다.
