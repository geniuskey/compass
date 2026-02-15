---
title: 다이나믹 레인지 계산기
---

# 다이나믹 레인지 계산기

풀 웰 용량, 읽기 노이즈, 암전류, 노출 설정으로 센서 다이나믹 레인지를 계산합니다. 단일 노출과 HDR 모드를 온도별로 비교합니다.

<DynamicRangeCalculator />

## 다이나믹 레인지 정의

최대 신호(포화)와 최소 검출 가능 신호(노이즈 플로어)의 비율:

**DR = 20 x log10(N_sat / sigma_floor)** [dB]

### 노이즈 플로어

**sigma_floor = sqrt(sigma_read^2 + I_dark x t_exp)**

### 온도 의존성

암전류는 약 5.5°C마다 2배 증가하여 장시간 노출에서 DR이 크게 감소합니다.

### HDR 확장

다중 노출 HDR은 긴 노출(그림자)과 짧은 노출(하이라이트)을 결합합니다:

**DR_HDR = 20 x log10(N_sat x ratio / sigma_floor)**

::: warning
HDR 이득은 노출 비율에 의해 제한되며, 프레임 간 모션 아티팩트가 발생할 수 있습니다.
:::
