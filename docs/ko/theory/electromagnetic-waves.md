# 전자기파

::: tip 선수 지식
이 페이지를 읽기 전에 [광학 기초 입문](/ko/introduction/optics-primer)과 [빛의 기초](/ko/theory/light-basics)를 먼저 읽어보세요.
:::

왜 맥스웰 방정식이 필요할까요? 맥스웰 방정식은 빛이 픽셀 내부의 미세한 구조와 만날 때 정확히 어떻게 행동하는지를 알려주기 때문입니다. 픽셀 구조의 크기가 빛의 파장(~0.5 um)보다 작아지면, 단순한 광선 추적(Ray Tracing)으로는 충분하지 않으며 맥스웰 방정식이 제공하는 완전한 파동 해석이 필요합니다. COMPASS의 솔버(RCWA와 FDTD)는 모두 이 방정식을 수치적으로 푸는 방법입니다.

이 페이지에서는 RCWA 및 FDTD 솔버가 내부적으로 사용하는 맥스웰 방정식(Maxwell's Equations)과 파동 형식론을 소개합니다.

<EMWaveAnimation />

## 맥스웰 방정식

모든 전자기 현상은 네 개의 방정식으로 지배됩니다. 선형, 등방성, 비자성 매질에서 자유 전하가 없는 경우:

$$\nabla \times \mathbf{E} = -\mu_0 \frac{\partial \mathbf{H}}{\partial t}$$

$$\nabla \times \mathbf{H} = \varepsilon_0 \varepsilon_r \frac{\partial \mathbf{E}}{\partial t}$$

$$\nabla \cdot (\varepsilon_r \mathbf{E}) = 0$$

$$\nabla \cdot \mathbf{H} = 0$$

여기서 $\mathbf{E}$는 전기장(Electric Field), $\mathbf{H}$는 자기장(Magnetic Field), $\varepsilon_r$은 비유전율(Relative Permittivity, 복소수이며 공간적으로 변할 수 있음), $\varepsilon_0$는 진공 유전율(Vacuum Permittivity), $\mu_0$는 진공 투자율(Vacuum Permeability)입니다.

## 시간 조화 형식

단색(단일 주파수) 빛의 시간 의존성이 $e^{-i\omega t}$인 경우, 회전(Curl) 방정식은 다음과 같이 됩니다:

$$\nabla \times \mathbf{E} = i\omega \mu_0 \mathbf{H}$$

$$\nabla \times \mathbf{H} = -i\omega \varepsilon_0 \varepsilon_r \mathbf{E}$$

이것이 **RCWA**의 출발점으로, 주파수 영역(Frequency Domain)에서 시간 조화 방정식을 풉니다. **FDTD**는 이와 달리 시간 영역(Time Domain) 방정식을 격자 위에서 직접 풉니다.

## 평면파

균일한 매질에서 맥스웰 방정식의 가장 단순한 해는 평면파(Plane Wave)입니다:

$$\mathbf{E}(\mathbf{r}, t) = \mathbf{E}_0 \, e^{i(\mathbf{k} \cdot \mathbf{r} - \omega t)}$$

여기서 파수 벡터(Wave Vector) $\mathbf{k}$는 분산 관계(Dispersion Relation)를 만족합니다:

$$|\mathbf{k}|^2 = k_0^2 \varepsilon_r, \qquad k_0 = \frac{2\pi}{\lambda}$$

COMPASS에서 입사광은 항상 평면파(또는 원추 조명을 위한 평면파의 가중 합)입니다. 솔버는 이 평면파가 적층된 픽셀 구조와 어떻게 상호작용하는지를 계산합니다.

## 입사 기하학

COMPASS는 입사파 방향에 대해 구면 좌표계(Spherical Coordinate) 규약을 사용합니다:

- $\theta$: 표면 법선(z축)으로부터 측정한 극각(Polar Angle)입니다. $\theta = 0$은 수직 입사(Normal Incidence)입니다.
- $\phi$: xy 평면에서의 방위각(Azimuthal Angle)입니다. $\phi = 0$은 x축 방향입니다.

입사 매질($n_\text{inc}$)에서 파수 벡터의 횡방향 성분은 다음과 같습니다:

$$k_x = k_0 n_\text{inc} \sin\theta \cos\phi$$

$$k_y = k_0 n_\text{inc} \sin\theta \sin\phi$$

이 성분들은 모든 계면에서 보존되며(스넬의 법칙을 3D로 일반화한 것), RCWA와 FDTD 모두 이 방법으로 입사각을 적용합니다.

## 경계 조건

두 매질의 경계면에서 $\mathbf{E}$와 $\mathbf{H}$의 접선 성분은 연속이어야 합니다:

$$\mathbf{E}_{t,1} = \mathbf{E}_{t,2}$$

$$\mathbf{H}_{t,1} = \mathbf{H}_{t,2}$$

이 조건들로부터 단일 계면에서의 프레넬 반사 및 투과 계수(Fresnel Reflection and Transmission Coefficients)가 도출됩니다:

$$r_\text{TE} = \frac{n_1 \cos\theta_1 - n_2 \cos\theta_2}{n_1 \cos\theta_1 + n_2 \cos\theta_2}$$

$$r_\text{TM} = \frac{n_2 \cos\theta_1 - n_1 \cos\theta_2}{n_2 \cos\theta_1 + n_1 \cos\theta_2}$$

측면 패턴이 있는 다층 스택의 경우, 이 조건들은 수치적으로 풀어야 하며, 이것이 바로 RCWA와 FDTD가 수행하는 작업입니다.

## 에너지 흐름: 포인팅 벡터

시간 평균 단위 면적당 전력 흐름은 포인팅 벡터(Poynting Vector)로 주어집니다:

$$\langle \mathbf{S} \rangle = \frac{1}{2} \text{Re}(\mathbf{E} \times \mathbf{H}^*)$$

z 성분 $S_z$는 수평면을 통과하는 전력의 크기를 나타냅니다. COMPASS는 포인팅 벡터를 사용하여 다음을 계산합니다:

- **반사율(Reflection)** ($R$): 구조 위로 반사되는 전력.
- **투과율(Transmission)** ($T$): 구조 아래로 투과되는 전력.
- **흡수율(Absorption)** ($A$): 구조 내에서 흡수되는 전력으로, $A = 1 - R - T$로 계산됩니다.
- **픽셀별 QE**: 각 포토다이오드 영역에서 흡수되는 전력.

## 두 가지 솔버 접근법의 필요성

맥스웰 방정식은 다양한 방법으로 풀 수 있으며, 각각 장단점이 있습니다:

| 접근법 | 방법 | 강점 |
|----------|--------|-----------|
| **주파수 영역** | RCWA | 주기 구조에 대해 빠르고, 정확한 주기성 처리, 효율적인 파장 스윕 |
| **시간 영역** | FDTD | 임의의 형상 처리 가능, 단일 실행으로 광대역 응답 획득, 직관적인 전기장 시각화 |

COMPASS는 두 솔버를 모두 지원하므로, 각 문제에 가장 적합한 도구를 선택하고 결과를 상호 검증할 수 있습니다. 자세한 내용은 [RCWA 상세 설명](./rcwa-explained)과 [FDTD 상세 설명](./fdtd-explained)을 참조하십시오.
