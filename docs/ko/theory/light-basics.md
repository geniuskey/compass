# 빛의 기초

이 페이지에서는 COMPASS의 모든 시뮬레이션을 뒷받침하는 빛의 기본 성질을 다룹니다.

## 빛이란 무엇인가?

빛은 전자기파(Electromagnetic Radiation)로, 진동하는 전기장과 자기장이 공간을 통해 전파되는 현상입니다. 이미지 센서 시뮬레이션에서는 가시광선(Visible Light) 및 근적외선(Near-Infrared, NIR) 영역의 파장(약 380 nm ~ 1100 nm)을 주로 다룹니다.

빛은 이중성(Wave-Particle Duality)을 가집니다:

- **파동(Wave)**: 빛은 간섭(Interference)과 회절(Diffraction) 현상을 나타내며, 이는 픽셀 구조의 크기가 파장과 비슷할 때 매우 중요합니다.
- **입자(Photon)**: 각 광자는 $E = h\nu = hc/\lambda$의 에너지를 가지며, 여기서 $h$는 플랑크 상수(Planck's Constant), $c$는 빛의 속도, $\lambda$는 파장입니다.

## 주요 물리량

### 파장과 주파수

파장 $\lambda$과 주파수(Frequency) $\nu$는 다음과 같은 관계를 가집니다:

$$\lambda = \frac{c}{n \cdot \nu}$$

여기서 $n$은 매질의 굴절률(Refractive Index)이고 $c = 3 \times 10^8$ m/s입니다. COMPASS는 마이크로미터(um) 단위를 사용하므로, 가시광선은 $0.38$ ~ $0.78$ um 범위에 해당합니다.

### 굴절률

재료의 광학적 특성은 **복소 굴절률(Complex Refractive Index)**로 표현됩니다:

$$\tilde{n} = n + ik$$

| 구성 요소 | 기호 | 의미 |
|------|--------|---------|
| 실수부 | $n$ | 진공에서의 위상 속도 대비 매질에서의 위상 속도의 비율입니다. 굴절과 간섭을 결정합니다. |
| 허수부 | $k$ | 소광 계수(Extinction Coefficient)입니다. 빛이 전파될 때 강도가 얼마나 빠르게 감쇠하는지를 결정합니다. |

예를 들어, 550 nm에서 실리콘(Silicon)의 $n \approx 4.08$, $k \approx 0.028$이며, 이는 빛을 강하게 굴절시키고 수 마이크로미터 이내에서 흡수한다는 것을 의미합니다.

### 유전율

복소 유전율(Permittivity) $\varepsilon$은 복소 굴절률의 제곱입니다:

$$\varepsilon = \tilde{n}^2 = (n + ik)^2 = (n^2 - k^2) + 2ink$$

RCWA 및 FDTD 솔버는 내부적으로 유전율을 사용합니다. COMPASS는 재료 데이터를 $(n, k)$ 형식으로 저장하고, 시뮬레이션 구조를 구성할 때 $\varepsilon$로 변환합니다.

### 흡수

빛이 흡수 매질을 거리 $d$만큼 통과할 때, 강도는 베어-람베르트 법칙(Beer-Lambert Law)에 따라 감쇠합니다:

$$I(d) = I_0 \, e^{-\alpha d}$$

여기서 $\alpha$는 흡수 계수(Absorption Coefficient)이며, $k$와 다음과 같은 관계를 가집니다:

$$\alpha = \frac{4\pi k}{\lambda}$$

이것이 실리콘에서 단파장(청색) 빛은 표면 부근에서 흡수되지만, 장파장(적색/근적외선) 빛은 수 마이크로미터 더 깊이 침투하는 이유입니다. 파장에 따른 흡수 깊이의 차이는 이미지 센서 설계의 핵심 과제입니다.

<WavelengthSlider />

## 굴절: 스넬의 법칙

빛이 한 매질에서 다른 매질로 진행할 때, 스넬의 법칙에 따라 방향이 변합니다: $n_1 \sin\theta_1 = n_2 \sin\theta_2$. 입사각이 임계각을 초과하면 (밀한 매질에서 소한 매질로 진행하는 경우) 전반사가 발생합니다.

<SnellCalculator />

## 편광

빛은 횡파(Transverse Wave)로, 전기장이 전파 방향에 수직으로 진동합니다. 이 진동의 방향을 **편광 상태(Polarization State)**라고 합니다.

- **TE (s-편광)**: 전기장이 입사면에 수직입니다.
- **TM (p-편광)**: 전기장이 입사면 내에 있습니다.
- **무편광(Unpolarized)**: TE와 TM이 동일하게 혼합된 상태입니다. 자연 태양광과 대부분의 주변 광원은 무편광입니다.

COMPASS는 TE, TM, 무편광 여기(Excitation)를 모두 지원합니다. 무편광의 경우, TE와 TM 시뮬레이션을 각각 수행한 후 결과를 평균합니다:

$$\text{QE}_\text{unpol} = \frac{1}{2}(\text{QE}_\text{TE} + \text{QE}_\text{TM})$$

## COMPASS와의 관련성

모든 COMPASS 시뮬레이션은 소스 설정(Source Configuration)을 통해 파장 범위, 입사각, 편광 상태를 정의하는 것에서 시작됩니다. 이 매개변수들은 다음을 결정합니다:

1. 각 파장에서 각 재료의 유전율 (`MaterialDB`를 통해 계산).
2. 박막 스택(Anti-Reflection Coating 등)에서의 간섭 조건.
3. 서브파장 구조(컬러 필터 격자, DTI)에서의 회절 거동.
4. 실리콘에서의 흡수 깊이로, 이는 QE에 직접적인 영향을 미칩니다.

::: tip
대부분의 이미지 센서 시뮬레이션에서는 `polarization: "unpolarized"`를 사용하고, 0.38 ~ 0.78 um 범위의 파장 스윕(Wavelength Sweep)을 설정하여 전체 가시광 스펙트럼을 포괄하는 것을 권장합니다.
:::
