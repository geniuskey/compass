---
title: 노이즈, SNR, 다이나믹 레인지
description: 이미지 센서 노이즈 소스, 신호 대 잡음비, 풀웰 용량, 다이나믹 레인지, 광자 전달 곡선(PTC), PRNU/DSNU, 암전류, 선형성의 공식 기준 문서.
---

# 노이즈, SNR, 다이나믹 레인지

::: tip 선수 지식
[양자 효율](./quantum-efficiency.md) → [신호 체인](./signal-chain.md) → 이 페이지.
:::

::: info 이 페이지의 범위
이 페이지는 COMPASS의 공식 노이즈 기준 문서입니다. 각 노이즈 소스, SNR, 풀웰 용량, 다이나믹 레인지, 광자 전달 곡선, 고정 패턴 노이즈(PRNU/DSNU), 암전류, 선형성을 정의합니다. 이 관계들을 시각화하는 인터랙티브 도구는 [시뮬레이터](/ko/simulator/) 섹션에 있습니다.
:::

픽셀의 신호는 그 위에 얹힌 노이즈에 상대적인 값일 때만 의미가 있습니다. 같은 QE 설계가 읽기 잡음, 적분 시간, 동작 온도에 따라 훌륭하게도, 사용 불가능하게도 보일 수 있습니다. 이 페이지는 COMPASS와 브라우저 시뮬레이터 전반에서 사용되는 표준 노이즈 모델을 정리합니다.

## 신호 모델

적분 시간 $t_\text{int}$ 동안 픽셀에 입사한 광자 수를 $N_\text{ph}$라 하면, 수집된 신호(전자 단위)는

$$N = \text{QE} \cdot N_\text{ph} + I_\text{d}(T)\,t_\text{int}$$

이고, $I_\text{d}(T)$는 온도 $T$에서의 암전류(전자/초)입니다. 첫 항은 광생성 전하, 둘째 항은 열적 생성 전하입니다. 풀웰 용량(FWC)은 $N$의 상한이며 이를 넘으면 픽셀이 포화됩니다.

## 노이즈 소스

총 노이즈는 통계적으로 독립인 기여의 직교 합입니다:

$$\sigma_\text{total}^2 = \sigma_\text{shot}^2 + \sigma_\text{dark}^2 + \sigma_\text{read}^2 + \sigma_\text{PRNU}^2 + \sigma_\text{DSNU}^2$$

### 샷 노이즈 (광자 노이즈)

광자 도착은 Poisson 분포이므로 평균 신호 $N$ 전자에 대해

$$\sigma_\text{shot} = \sqrt{N}$$

샷 노이즈는 근본적이며 센서 설계로 줄일 수 없습니다 — 광자를 더 모으는 방법(픽셀 크기 증가, 노출 시간 증가, QE 향상)만 가능합니다.

### 암전류 노이즈

암전류도 Poisson:

$$\sigma_\text{dark} = \sqrt{I_\text{d}(T)\,t_\text{int}}$$

암전류는 Arrhenius 의존성을 가집니다:

$$I_\text{d}(T) \propto T^{3/2}\,\exp\!\left(-\frac{E_g}{2 k_B T}\right)$$

여기서 $E_g \approx 1.12$ eV(실리콘). 흔히 쓰는 실무 규칙은 온도가 6–8 °C 오를 때마다 암전류가 대략 두 배가 된다는 것입니다. 따라서 20 °C 냉각 시 암전류는 약 8× 감소합니다.

### 읽기 잡음

읽기 잡음 $\sigma_\text{read}$은 readout 체인(소스 팔로워, 컬럼 앰프, ADC)이 더하는 잡음이며 신호와 거의 무관합니다. 저조도에서 지배적입니다.

### 고정 패턴 노이즈: PRNU와 DSNU

고정 패턴 노이즈(FPN)는 프레임마다 변하지 않는 픽셀-픽셀 변동입니다.

**PRNU (광응답 비균일성)** 은 신호에 비례하는 게인 변동:

$$\sigma_\text{PRNU} = u_\text{PRNU}\,N$$

으로 현대 센서에서 $u_\text{PRNU} \approx 0.5\%$–$2\%$. 마이크로렌즈 정렬, 포토다이오드 도핑, 픽셀 형상의 변동에서 발생합니다.

**DSNU (암신호 비균일성)** 은 신호와 무관한 오프셋 변동:

$$\sigma_\text{DSNU} = \text{constant in } e^-\text{ RMS}$$

결정 결함이나 계면 트랩에 의한 픽셀별 암전류 편차로 발생합니다.

총 FPN:

$$\sigma_\text{FPN} = \sqrt{\sigma_\text{PRNU}^2 + \sigma_\text{DSNU}^2}$$

FPN은 픽셀별 플랫 필드와 다크 프레임 보정으로 부분적으로 제거 가능하며, 잔차가 균일도가 중요한 응용의 하한이 됩니다.

## 신호 대 잡음비

신호 레벨 $N$에서의 SNR:

$$\text{SNR}(N) = \frac{N}{\sqrt{\sigma_\text{read}^2 + N + (u_\text{PRNU}\,N)^2}}$$

(필요 시 암전류와 DSNU를 read 항에 흡수). 세 영역으로 나뉩니다:

| 영역 | 지배 잡음 | SNR 스케일링 |
|---|---|---|
| 저조도 | 읽기 잡음 | $\text{SNR} \propto N$ |
| 중간 | 샷 노이즈 | $\text{SNR} \propto \sqrt{N}$ |
| 고신호 | PRNU | $\text{SNR}$ 포화 ($1/u_\text{PRNU}$) |

로그 스케일 표기:

$$\text{SNR}_\text{dB} = 20 \log_{10}\!\left(\frac{N}{\sigma_\text{total}}\right)$$

대표 기준점: SNR = 0 dB(신호=잡음, 절대 검출 한계), SNR = 20 dB(허용 화질 최소 기준으로 자주 사용), 샷 노이즈 한계 시 $\text{SNR}_\text{max} = \sqrt{\text{FWC}}$ (PRNU 무시).

## 풀웰 용량과 다이나믹 레인지

FWC는 픽셀이 포화 전에 보유할 수 있는 최대 전자 수이며 픽셀 면적에 비례:

$$\text{FWC} \propto \text{pitch}^2$$

다이나믹 레인지(DR)는 FWC와 최소 적분 시 노이즈 플로어의 비:

$$\text{DR}_\text{dB} = 20 \log_{10}\!\left(\frac{\text{FWC}}{\sigma_\text{floor}}\right), \quad
\sigma_\text{floor} = \sqrt{\sigma_\text{read}^2 + I_\text{d}(T)\,t_\text{int}}$$

스톱 단위: $\text{DR}_\text{stops} = \text{DR}_\text{dB} / 6.02$. 다중 노출 HDR은 긴 노출(섀도용)과 짧은 노출(하이라이트용)을 결합하여 DR을 확장합니다:

$$\text{DR}_\text{HDR} = 20 \log_{10}\!\left(\frac{\text{FWC}\,r}{\sigma_\text{floor}}\right)$$

$r$은 노출 비. 확장폭은 프레임 간 모션 아티팩트로 제한됩니다.

## 광자 전달 곡선 (PTC)

PTC는 총 노이즈와 평균 신호를 로그-로그 스케일로 플롯한 것으로, 측정으로부터 컨버전 게인, 읽기 잡음, FWC, PRNU를 추출하는 표준 방법입니다. 세 영역이 보입니다:

| 영역 | 로그-로그 기울기 | 추출 파라미터 |
|---|---|---|
| 평평한 플로어 | 0 | 읽기 잡음 $\sigma_\text{read}$ |
| 샷 노이즈 | $1/2$ | 컨버전 게인 $K$ ($\sigma^2 = K\,S$에서) |
| PRNU | $1$ | $u_\text{PRNU}$ |

총 분산 모델:

$$\sigma_\text{total}^2 = \sigma_\text{read}^2 + N + (u_\text{PRNU}\,N)^2$$

영역 간 교차점도 진단 정보입니다: read–shot 교차점이 높으면 읽기 잡음 과다, shot–PRNU 교차점이 낮으면 제조 균일도 부족.

## 응답도

분광 응답도는 QE를 전기적 지표로 변환:

$$R(\lambda) = \text{QE}(\lambda)\,\frac{q\,\lambda}{h c} \approx \text{QE}(\lambda)\,\frac{\lambda_\text{nm}}{1240}\ \ [\text{A/W}]$$

$\lambda/1240$ 인자로 인해 응답도 피크가 QE 피크보다 장파장 쪽으로 이동합니다. 이상적 실리콘 포토다이오드(QE = 1)는 $R_\text{ideal}(\lambda) = \lambda_\text{nm}/1240$.

## 선형성

이상적 전달 함수는 선형: $\text{DN}_\text{ideal} = (N/\text{FWC})\,\text{DN}_\text{max}$. 실제 센서는 소스 팔로워 비선형성, 전압 의존 접합 캐패시턴스, ADC INL로 인해 편차가 발생합니다. 표준 지표:

$$\text{NL}(\%) = \frac{\max |\text{DN}_\text{actual} - \text{DN}_\text{ideal}|}{\text{DN}_\text{max}} \times 100$$

대부분의 머신 비전 응용은 NL < 1%, HDR 머징과 광량 측정은 < 0.5% 요구.

## 브라우저 시뮬레이터

다음 인터랙티브 도구가 위 식을 시각화합니다:

- [SNR 계산기](/ko/simulator/snr-calculator) — 총 노이즈 분해와 동작점별 SNR
- [광자 전달 곡선](/ko/simulator/photon-transfer-curve) — 읽기 잡음, 컨버전 게인, PRNU 추출
- [SNR vs 조도](/ko/simulator/pixel-snr-vs-illuminance) — 세 노이즈 영역과 이상적 샷 한계 비교
- [다이나믹 레인지](/ko/simulator/dynamic-range) — FWC/노이즈 플로어와 HDR 확장
- [응답도 계산기](/ko/simulator/responsivity-calculator) — 채널별 QE-A/W 변환
- [암전류 & 온도](/ko/simulator/dark-current) — Arrhenius 모델과 다크 프레임 시각화
- [PRNU / DSNU](/ko/simulator/prnu-visualizer) — 고정 패턴 노이즈 공간 맵
- [선형성 분석기](/ko/simulator/linearity-analyzer) — 전달 곡선 편차와 니 포인트
