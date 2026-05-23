---
title: 마이크로렌즈 공정 형상 예측
---

# 마이크로렌즈 공정 형상 예측

Lithography layout, thermal reflow, etch-transfer 조건이 CIS 마이크로렌즈의 최종 gap, height, curvature, 3D profile에 주는 영향을 추정합니다.

<MicrolensProcessShape />

::: info 모델 범위
이 브라우저 도구는 **surrogate process model**이며 calibrated foundry recipe가 아닙니다. 공정 방향성, 민감도, failure mode를 이해하는 용도로 사용하세요. 정량 공정 판단에 쓰려면 wafer metrology 데이터로 계수를 다시 보정해야 합니다.
:::

## 이 도구가 추가하는 것

기존 COMPASS 마이크로렌즈 도구는 주어진 lens geometry에서 시작합니다. 이 도구는 그보다 한 단계 앞의 공정 입력에서 시작합니다.

- **Layout**: pixel pitch, lithographic island width, aperture footprint shape, 그리고 1x1, 2x1, 1x2, 2x2 on-chip-lens unit 배치 선택기 (대표 preset과 함께, die 안에서 2x2 OCL + 1x1 같은 이종 혼합을 표현할 수 있는 custom 4x4 편집기 제공).
- **Reflow**: 온도/시간을 normalized thermal budget으로 보고, lens spread와 volume conservation 기반 height 변화를 계산. 비대칭(2x1/1x2) 마스크에는 reflow 중 photoresist가 isotropy로 이완하는 경향을 반영한 surface-tension 보정을 적용 — 장축은 덜 퍼지고, 단축은 더 퍼집니다.
- **Etch transfer**: etch time은 residual lens gap closure를 키우고, polymerization은 height 보존에, mask thickness는 transfer robustness에 영향을 준다고 모델링.
- **Calibration**: reflow spread, volume retention, lateral etch, vertical height-loss gain을 AFM/SEM 또는 DOE 데이터에 맞춰 보정할 수 있습니다.
- **혼합 레이아웃에서의 인접 결합**: 면별 proximity coupling(mass-flow 비대칭)과 plasma microloading 효과로, 작은 lens는 큰 이웃 쪽으로 부풀고 큰 OCL을 마주보는 boundary에는 잔류 gap이 살짝 더 큽니다. 동일 크기 레이아웃(전체 1x1 또는 전체 2x2)에서는 두 효과 모두 0이 됩니다.
- **Outputs**: final gap (X/Y 중 worst), height, 장단축비(WX:WY), vertex radius of curvature, f-number, fill factor, zero-gap etch-time estimate, profile exponent, cross-section, **lens group마다 면별 비대칭 outline과 height heat-map을 포함한 top-down (XY) footprint view**, 3D wireframe, etch response curve.

## 문헌 기반

- Ristoiu et al., **A DOE study of plasma etched microlens shape for CMOS image sensors**, SPIE 2020. CIS 공정 질문에 가장 직접적인 자료입니다. Reflowed microlens를 plasma transfer하고, mask thickness, polymerizing gas flow, etch time에 따른 gap/height evolution을 DOE로 모델링합니다. DOI: [10.1117/12.2551857](https://doi.org/10.1117/12.2551857)
- Baillie and Gendler, **Zero-space microlenses for CMOS image sensors: optical modeling and lithographic process development**, SPIE 2004. Layout gap과 zero-space 문제의 배경이 됩니다. Lithographic space가 너무 작으면 melt/reflow 중 merger 위험이 있고, residual space는 fill factor를 낮춥니다. DOI: [10.1117/12.533453](https://doi.org/10.1117/12.533453)
- Jin, Liu, and Yang, **Design, characterization and evaluation of high performance 2.8 μm pitch zero space microlens**, *Optics Communications*, 2011. Zero-space microlens geometry를 AFM characterization 및 silicon-level sensitivity/crosstalk test와 연결합니다. DOI: [10.1016/j.optcom.2010.11.073](https://doi.org/10.1016/j.optcom.2010.11.073)
- Tan, Goh, and Kim, **Microfabrication of Microlens by Timed-Development-and-Thermal-Reflow**, *Micromachines*, 2020. Parabolic/superellipse profile 표현, aperture geometry, development time, diameter, thickness, radius of curvature, focal length를 regression으로 연결하는 관점을 제공합니다. DOI: [10.3390/mi11030277](https://doi.org/10.3390/mi11030277)
- Choi *et al.*, **Profile control of asymmetric reflowed microlens for CMOS image sensors**, *Microelectronic Engineering* (2014). 2x1, 1x2 unit에 적용한 "장축은 덜 퍼지고 단축은 더 퍼진다"는 surface-tension 보정의 근거 문헌입니다. 비대칭 reflowed photoresist는 isotropy로 이완하므로, as-printed boundary gap이 같아도 reflow 이후 X gap과 Y gap이 달라집니다.
- Y. Oike *et al.* (Sony Semiconductor Solutions), **All-pixel phase-detection autofocus with 2x1 on-chip lens (Dual Pixel CMOS Image Sensor)**, ISSCC / IEEE JSSC. 2x1 OCL은 인접한 두 photodiode 위에 가로로 긴 하나의 lens를 공유하며, lens 장축이 PDAF 분리 방향입니다. **All 2x1 (Sony 2PD)** preset의 근거입니다.
- J. Park *et al.* (Samsung), **A 1/2.55-inch 1.0 μm-pixel 64-Mpixel CMOS image sensor with Tetracell color filter array**, ISSCC. 같은 색 2x2 sub-pixel이 하나의 2x2 OCL을 공유하여 저조도 binning에 사용되며, 인접한 다른 색 2x2 cell과의 boundary가 merger-risk 면이 됩니다. **All 2x2 (Tetracell OCL)** preset의 근거입니다.
- Lee *et al.*, **Covered Microlens Structure for Quad Color Filter Array of CMOS Image Sensor**, *Current Optics and Photonics*, 2023. Mixed-size lens 배치(서브픽셀 4개 위 covered microlens 1개)를 직접 다루며, 주변 lens geometry가 광학 효율을 실질적으로 바꾸는 것을 보입니다. Top view의 면별 proximity coupling의 경험적 근거입니다. URL: <https://opg.optica.org/copp/abstract.cfm?uri=copp-7-5-485>
- Mogab, **The Loading Effect in Plasma Etching**, *J. Electrochem. Soc.*, 1977. Plasma etch loading 효과의 고전 레퍼런스 — dense pattern 영역에서 국소 etchant 소모로 lateral etch rate가 감소합니다. 큰 OCL과 맞닿은 boundary의 잔류 gap을 넓히는 microloading 항의 근거. DOI: [10.1149/1.2133486](https://doi.org/10.1149/1.2133486)
- Gottscho, Jurgensen, Vitkavage, **Microscopic uniformity in plasma etching**, *J. Vac. Sci. Technol. B*, 1992. ARDE/microloading의 기초 문헌으로, feature 간 etch-rate 차이가 국소 pattern density와 aspect ratio에 의존한다는 점이 surrogate의 면별 density 점수의 근거입니다.
- Sony/Samsung/OmniVision의 on-chip-lens 설계 규칙 특허 (**US8941766**, **US11765476**, **US12022217**, **US12069384** 등) — 작은 이웃을 마주보는 OCL의 bottom을 명시적으로 넓히고 국소 CRA에 맞춰 shift합니다. 우리 surrogate의 면별 결합이 경험적으로 검증됨을 보여줍니다.

::: warning
공개 논문은 범용 CIS recipe 계수를 제공하지 않습니다. 이 시뮬레이터는 문헌의 **방향성 관계**를 보존하고, 나중에 DOE 또는 AFM/SEM 데이터로 치환할 수 있도록 계수를 명시한 형태입니다.
:::

::: tip 관련 도구
[마이크로렌즈 광선 추적](./microlens-raytrace) · [MLA 어레이 시각화](./mla-array) · [마이크로렌즈 & CRA 레시피](/ko/cookbook/microlens-optimization)
:::

<SimulatorTheory slug="microlens-process-shape" />
