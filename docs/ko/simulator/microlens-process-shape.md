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

- **Layout**: pixel pitch, lithographic island width, aperture footprint shape.
- **Reflow**: 온도/시간을 normalized thermal budget으로 보고, lens spread와 volume conservation 기반 height 변화를 계산.
- **Etch transfer**: etch time은 residual lens gap closure를 키우고, polymerization은 height 보존에, mask thickness는 transfer robustness에 영향을 준다고 모델링.
- **Outputs**: final gap, height, vertex radius of curvature, f-number, fill factor, cross-section, 3D wireframe, etch response curve.

## 문헌 기반

- Ristoiu et al., **A DOE study of plasma etched microlens shape for CMOS image sensors**, SPIE 2020. CIS 공정 질문에 가장 직접적인 자료입니다. Reflowed microlens를 plasma transfer하고, mask thickness, polymerizing gas flow, etch time에 따른 gap/height evolution을 DOE로 모델링합니다. DOI: [10.1117/12.2551857](https://doi.org/10.1117/12.2551857)
- Baillie and Gendler, **Zero-space microlenses for CMOS image sensors: optical modeling and lithographic process development**, SPIE 2004. Layout gap과 zero-space 문제의 배경이 됩니다. Lithographic space가 너무 작으면 melt/reflow 중 merger 위험이 있고, residual space는 fill factor를 낮춥니다. DOI: [10.1117/12.533453](https://doi.org/10.1117/12.533453)
- Jin, Liu, and Yang, **Design, characterization and evaluation of high performance 2.8 μm pitch zero space microlens**, *Optics Communications*, 2011. Zero-space microlens geometry를 AFM characterization 및 silicon-level sensitivity/crosstalk test와 연결합니다. DOI: [10.1016/j.optcom.2010.11.073](https://doi.org/10.1016/j.optcom.2010.11.073)
- Tan, Goh, and Kim, **Microfabrication of Microlens by Timed-Development-and-Thermal-Reflow**, *Micromachines*, 2020. Parabolic/superellipse profile 표현, aperture geometry, development time, diameter, thickness, radius of curvature, focal length를 regression으로 연결하는 관점을 제공합니다. DOI: [10.3390/mi11030277](https://doi.org/10.3390/mi11030277)

::: warning
공개 논문은 범용 CIS recipe 계수를 제공하지 않습니다. 이 시뮬레이터는 문헌의 **방향성 관계**를 보존하고, 나중에 DOE 또는 AFM/SEM 데이터로 치환할 수 있도록 계수를 명시한 형태입니다.
:::

::: tip 관련 도구
[마이크로렌즈 광선 추적](./microlens-raytrace) · [MLA 어레이 시각화](./mla-array) · [마이크로렌즈 & CRA 레시피](/ko/cookbook/microlens-optimization)
:::
