---
outline: deep
---

# 컬러 필터 Relief 민감도 리포트

_생성일: 2026-05-08. generic 1.0 um BSI `PixelStack`에서 생성._

이 리포트는 색별 컬러 필터 모델의 geometry 민감도 리포트다. `grid.thickness`, `red/green/blue.thickness`, `red/green/blue.contact_angle`이 z-sliced solver geometry를 어떻게 바꾸는지 보여준다.

::: info 범위
아래 그림은 geometry evidence다. 아직 optical QE 또는 crosstalk delta를 보고하지 않는다. 다음 optical 리포트에서는 이 geometry variant에 대해 RCWA order sweep을 실행하는 것이 좋다.
:::

## 단면 variant

![Color filter relief cross sections](/reports/geometry/color-filter-relief/color_filter_relief_sections.png)

## Contact-angle sweep

![Contact angle sweep](/reports/geometry/color-filter-relief/contact_angle_sweep.png)

## 기본 색별 geometry

| Color | material | thickness um | above grid um | contact angle | top area / pitch area |
| --- | --- | --- | --- | --- | --- |
| R | cf_red | 0.624 | 0.156 | 66 | 0.61 |
| G | cf_green | 0.6 | 0.132 | 72 | 0.696 |
| B | cf_blue | 0.648 | 0.18 | 62 | 0.531 |

## 해석

- `grid.thickness`는 metal-grid 영역의 수직 높이를 정한다.
- 색별 `thickness`는 각 color resist의 최대 높이를 정한다.
- `contact_angle`은 grid 위 돌출부의 사다리꼴 taper를 제어한다. 각도가 낮을수록 같은 돌출 높이에서 top footprint가 작아진다.
- red, green, blue가 서로 다른 높이와 각도를 가지므로 microlens slice를 고려하기 전에도 RCWA에는 여러 color-filter z slice가 들어간다.

## 재생성

```powershell
uv run python scripts\generate_geometry_reports.py
```

생성 metric은 `docs/public/reports/geometry/color-filter-relief/color_filter_relief_metrics.json`에 저장된다.
