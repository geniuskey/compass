---
title: 색 재현과 색공간
description: CMOS 이미지 센서의 분광 응답을 camera RGB, CIE XYZ, Lab, sRGB, 색 보정 행렬, 색차 지표로 연결하는 센서 관점 색과학 이론입니다.
---

# 색 재현과 색공간

::: tip 선수 지식
[신호 체인](/ko/theory/sensor/signal-chain) -> 이 페이지 -> [색 정확도 분석기](/ko/simulator/color-accuracy)
:::

이미지 센서에서 색공간을 다루는 이유는 센서가 사람의 눈이나 sRGB 디스플레이가 정의하는 "빨강", "초록", "파랑"을 직접 측정하지 않기 때문입니다. CIS 픽셀은 세 개의 장치 의존적 분광 적분값을 측정합니다. 색 재현은 이 적분값을 표준 색도 공간, 보통 CIE XYZ로 옮긴 뒤 sRGB나 CIE Lab 같은 표시 또는 평가 공간으로 변환하는 과정입니다.

COMPASS에서 이 페이지는 광학 시뮬레이션과 화질 지표를 잇는 다리입니다:

1. 전자기 시뮬레이션 또는 브라우저 모델이 채널별 분광 응답을 만듭니다.
2. 장면 스펙트럼과 광원을 그 응답으로 적분합니다.
3. 화이트 밸런스와 색 보정 행렬(CCM)로 camera RGB를 XYZ에 맞춥니다.
4. Lab과 색차 지표로 남은 오차를 정량화합니다.

## 센서 색은 분광 매칭 문제

채널 $c \in \{R,G,B\}$에 대해 간단한 센서 응답 모델은 다음과 같습니다:

$$
Q_c(\lambda) =
T_{\text{lens}}(\lambda)
T_{\text{IR}}(\lambda)
T_{\text{CF},c}(\lambda)
\eta_{\text{abs},c}(\lambda)
\eta_{\text{col},c}(\lambda)
$$

여기서:

- $Q_c(\lambda)$는 채널 $c$의 종단 간 분광 응답입니다.
- $T_{\text{lens}}(\lambda)$는 카메라 렌즈 투과율입니다.
- $T_{\text{IR}}(\lambda)$는 IR-cut 필터 투과율입니다.
- $T_{\text{CF},c}(\lambda)$는 컬러 필터 투과율입니다.
- $\eta_{\text{abs},c}(\lambda)$는 목표 포토다이오드 영역에서의 광 흡수 효율입니다.
- $\eta_{\text{col},c}(\lambda)$는 흡수 이후의 캐리어 수집 효율입니다.

광원 $E(\lambda)$ 아래에서 반사율 $\rho_j(\lambda)$를 가진 표면 패치 $j$의 raw 채널 응답은:

$$
r_{c,j} =
\int_{\lambda_1}^{\lambda_2}
E(\lambda)\rho_j(\lambda)Q_c(\lambda)\,d\lambda
$$

이 식 때문에 색은 센서 설계의 일부입니다. CFA 두께, 염료 스펙트럼, 마이크로렌즈 초점, DTI 누설, IR-cut 기울기, 실리콘 흡수 깊이는 모두 $Q_c(\lambda)$를 바꾸며, 따라서 ISP 보정 이전의 camera RGB triplet도 바꿉니다.

::: info Camera RGB는 display RGB가 아닙니다
Raw camera RGB는 장치 의존 좌표계입니다. 그 축은 센서 채널 응답이지, CIE 표준 관측자 기본색이나 sRGB 디스플레이 기본색이 아닙니다. raw RGB를 곧바로 sRGB처럼 취급하면 광원 의존성, 메타메릭 오차, CFA 트레이드오프가 가려집니다.
:::

## CIE XYZ 기준색

CIE XYZ는 색 재현에서 가장 흔히 쓰는 장치 독립 기준입니다. 같은 광원과 표면 반사율에 대한 기준 삼자극값은:

$$
\begin{bmatrix}
X_j \\
Y_j \\
Z_j
\end{bmatrix}
=
k
\int_{\lambda_1}^{\lambda_2}
E(\lambda)\rho_j(\lambda)
\begin{bmatrix}
\bar{x}(\lambda) \\
\bar{y}(\lambda) \\
\bar{z}(\lambda)
\end{bmatrix}
d\lambda
$$

여기서:

- $\bar{x}(\lambda)$, $\bar{y}(\lambda)$, $\bar{z}(\lambda)$는 CIE 표준 관측자 색일치 함수입니다.
- $k$는 정규화 상수이며, 보통 기준 백색의 $Y$가 100 또는 1이 되도록 둡니다.
- $Y$는 CIE 체계에서 명소시 휘도에 대응합니다.

색도는 절대 휘도를 제거합니다:

$$
x = \frac{X}{X+Y+Z}, \qquad
y = \frac{Y}{X+Y+Z}
$$

컬러 필터 시뮬레이터에서 쓰는 CIE 1931 색도도는 이 $(x,y)$ 좌표를 표시합니다. 색역을 보는 데 유용하지만, 그 자체로 낮은 색 오차를 보장하지는 않습니다. 휘도와 비선형 지각 균일성이 2D 도표에 충분히 표현되지 않기 때문입니다.

## 화이트 밸런스

화이트 밸런스는 중립 타겟이 채널 스케일링 이후에도 중립이 되도록 만들어 광원 색 편향의 1차 성분을 제거합니다. 중립 패치의 raw 응답이 $\mathbf{r}_{w}=[r_R,r_G,r_B]^T$라면, green 기준 gain은 보통:

$$
g_R = \frac{r_G}{r_R}, \qquad
g_G = 1, \qquad
g_B = \frac{r_G}{r_B}
$$

화이트 밸런스 이후의 camera vector는:

$$
\mathbf{c}_j =
\begin{bmatrix}
g_R & 0 & 0 \\
0 & g_G & 0 \\
0 & 0 & g_B
\end{bmatrix}
\left(\mathbf{r}_j-\mathbf{b}\right)
$$

여기서 $\mathbf{b}$는 black-level 또는 dark offset입니다. 시뮬레이션 센서에서는 $\mathbf{b}$가 0일 수 있지만, 실제 보정 데이터에서는 색 보정 전에 제거해야 합니다.

화이트 밸런스만으로는 센서가 사람의 표준 관측자와 같아지지 않습니다. 하나의 white point를 맞출 뿐, 유색 물체 전반의 분광 불일치를 고치지는 못합니다. 그 잔여 불일치를 색 보정 행렬이 처리합니다.

## 색 보정 행렬(CCM)

3x3 색 보정 행렬(CCM)은 white-balanced camera RGB를 목표 색공간으로 매핑합니다. 센서 평가에서는 목표를 보통 XYZ로 둡니다:

$$
\hat{\mathbf{x}}_j =
\begin{bmatrix}
\hat{X}_j \\
\hat{Y}_j \\
\hat{Z}_j
\end{bmatrix}
=
M
\mathbf{c}_j
$$

행렬은 보정 패치들로부터 fitting합니다:

$$
M^\star =
\arg\min_M
\sum_j
w_j
\left\|
M\mathbf{c}_j - \mathbf{x}_j
\right\|_2^2
$$

여기서:

- $\mathbf{c}_j$는 패치 $j$의 white-balanced camera vector입니다.
- $\mathbf{x}_j=[X_j,Y_j,Z_j]^T$는 기준 CIE XYZ vector입니다.
- $w_j$는 선택적 패치 가중치입니다.
- $M^\star$는 least-squares CCM입니다.

Camera sample을 $C=[\mathbf{c}_1,\dots,\mathbf{c}_n]$, 기준값을 $X=[\mathbf{x}_1,\dots,\mathbf{x}_n]$로 두면 regularization 없는 해는:

$$
M^\star = XC^T(CC^T)^{-1}
$$

보정 세트에 noise가 있거나 조건수가 나쁘면 ridge regularization이 더 안정적입니다:

$$
M^\star_\alpha = XC^T(CC^T+\alpha I)^{-1}
$$

CCM은 일부 광학적 타협을 가릴 수 있지만, 센서가 측정하지 못한 정보를 복원할 수는 없습니다. 서로 다른 두 스펙트럼이 같은 camera RGB를 만들지만 서로 다른 XYZ를 가진다면, 어떤 3x3 행렬도 둘을 분리할 수 없습니다.

## 메타메리즘과 분광 불일치

두 스펙트럼이 센서에 대해 metamer라면 모든 채널에서 같은 camera response를 만듭니다:

$$
\int E(\lambda)\rho_a(\lambda)Q_c(\lambda)\,d\lambda
=
\int E(\lambda)\rho_b(\lambda)Q_c(\lambda)\,d\lambda
\quad \text{for all } c
$$

사람에게도 같게 보이려면 같은 동등성이 CIE 색일치 함수에 대해서도 성립해야 합니다. 어떤 쌍이 camera metamer이지만 human-observer metamer가 아니거나, 그 반대일 때 색 오차가 발생합니다.

그래서 CFA 스펙트럼은 좁고 포화된 기본색만 보고 최적화할 수 없습니다. 매우 좁은 RGB 필터는 색 분리를 키울 수 있지만 신호를 줄이고, LED/형광등/혼합 조명에서 색 보정을 불안정하게 만들 수 있습니다. 넓은 필터는 감도를 높이지만 채널 독립성을 낮춥니다. 실무 목표는 예상 광원과 재료 범위에서 안정적인 CCM fitting을 지원하는 균형 잡힌 분광 응답입니다.

## XYZ에서 sRGB로

sRGB는 display-referred encoding입니다. XYZ가 D65 기준으로 적응되어 있다면 linear sRGB는 다음 행렬로 얻습니다:

$$
\begin{bmatrix}
R_{\text{lin}} \\
G_{\text{lin}} \\
B_{\text{lin}}
\end{bmatrix}
=
\begin{bmatrix}
3.2406 & -1.5372 & -0.4986 \\
-0.9689 & 1.8758 & 0.0415 \\
0.0557 & -0.2040 & 1.0570
\end{bmatrix}
\begin{bmatrix}
X \\
Y \\
Z
\end{bmatrix}
$$

클리핑된 linear component $u$에 대한 nonlinear sRGB code value는:

$$
v =
\begin{cases}
12.92u, & u \le 0.0031308 \\
1.055u^{1/2.4} - 0.055, & u > 0.0031308
\end{cases}
$$

여기서 $u$는 linear-light display RGB component이고 $v$는 nonlinear encoded component입니다. 이 인코딩은 표시와 파일 교환을 위한 것입니다. 모델이 display-coded 값을 명시적으로 요구하지 않는 한, CCM fitting이나 Lab error 계산 전에 적용하지 마세요.

## Lab과 색차

CIE Lab은 기준 백색 $(X_n,Y_n,Z_n)$에 대한 XYZ로부터 계산됩니다:

$$
L^\star = 116 f\!\left(\frac{Y}{Y_n}\right)-16
$$

$$
a^\star = 500\left[
f\!\left(\frac{X}{X_n}\right) -
f\!\left(\frac{Y}{Y_n}\right)
\right]
$$

$$
b^\star = 200\left[
f\!\left(\frac{Y}{Y_n}\right) -
f\!\left(\frac{Z}{Z_n}\right)
\right]
$$

여기서:

$$
f(t)=
\begin{cases}
t^{1/3}, & t > \delta^3 \\
\dfrac{t}{3\delta^2} + \dfrac{4}{29}, & t \le \delta^3
\end{cases}
\qquad
\delta=\frac{6}{29}
$$

단순한 1976 색차는:

$$
\Delta E_{ab}^\star =
\sqrt{
(\Delta L^\star)^2 +
(\Delta a^\star)^2 +
(\Delta b^\star)^2
}
$$

CIEDE2000, 보통 $\Delta E_{00}$로 쓰는 지표는 lightness, chroma, hue 항의 가중을 수정하여 시각적 차이를 더 잘 반영합니다. 수식은 더 복잡하지만, Lab 공간에서 같은 $\Delta E_{ab}^\star$가 항상 같은 가시 차이를 뜻하지 않기 때문에 카메라 색 정확도 비교에서는 더 실용적인 지표입니다.

## 센서 설계에 주는 의미

색 재현은 광학 스택 설계를 여러 방향에서 제약합니다:

| 설계 선택 | 색 재현 결과 |
|---|---|
| CFA 중심 파장과 대역폭 | 채널 분리, 감도, CCM conditioning을 결정합니다. |
| CFA 두께와 relief | 분광 passband와 각도 응답을 함께 바꾸므로, 화면 중심과 주변부의 색이 달라질 수 있습니다. |
| 마이크로렌즈 초점과 CRA shift | 각 파장이 어느 포토다이오드에 모이는지 바꾸며, 이미지 edge에서 color shading을 만들 수 있습니다. |
| IR-cut transition | 잔여 NIR이 모든 채널에 섞이면 채도가 낮아지고 white balance가 틀어집니다. |
| 실리콘 두께와 흡수 깊이 | 청색은 얕게, 적색/NIR은 깊게 흡수되어 crosstalk와 channel balance가 달라집니다. |
| Noise와 clipping | CCM coefficient가 noisy channel을 증폭할 수 있고, saturation된 채널은 chromatic 정보를 잃습니다. |

따라서 견고한 색 설계는 RGB endpoint 직관만으로 평가하면 안 되고, 스펙트럼으로 평가해야 합니다. 최소 평가 세트는 다음과 같습니다:

1. D65와 Illuminant A, 그리고 제품이 실제로 만날 LED 또는 형광등 스펙트럼.
2. 화이트 밸런스와 노출을 위한 neutral scale.
3. ColorChecker 계열처럼 반사율 스펙트럼이 알려진 patch set.
4. CCM 전후의 color error.
5. CCM 이후의 SNR 또는 noise amplification 점검.

## COMPASS 사용 흐름

광학 시뮬레이션을 색으로 연결할 때는 다음 순서가 자연스럽습니다:

1. [픽셀 광학 효과](/ko/theory/sensor/pixel-optical-effects)에서 CFA, CRA, crosstalk, stack-level effect를 봅니다.
2. [양자 효율](/ko/theory/sensor/quantum-efficiency)에서 absorbed power를 channel spectral response로 변환합니다.
3. [신호 체인](/ko/theory/sensor/signal-chain)에서 illuminant, scene reflectance, lens, IR filter, electron signal을 연결합니다.
4. [컬러 필터 설계](/ko/simulator/color-filter)에서 빠르게 spectral shape와 gamut을 탐색합니다.
5. [색 정확도 분석기](/ko/simulator/color-accuracy)에서 CCM fitting과 color error 직관을 확인합니다.
6. [신호 체인 색 정확도](/ko/cookbook/signal-chain-color-accuracy)에서 종단 간 레시피를 따릅니다.

양산 센서에서 브라우저 시뮬레이터는 초기 설계와 커뮤니케이션 도구입니다. Sign-off에는 측정된 CFA spectrum, module transmittance, dark/flat-field calibration, 실제 illuminant set, silicon image 검증이 필요합니다.

## 참고문헌

- [CIE 015:2018, Colorimetry, 4th Edition](https://www.cie.co.at/publications/colorimetry-4th-edition): 여기서 사용하는 표준 관측자, 표준 광원, 삼자극값 계산, 색공간, 색차 실무의 기준입니다.
- [IEC 61966-2-1:1999, Default RGB colour space - sRGB](https://webstore.iec.ch/en/publication/6169): display-referred RGB로 쓰는 sRGB 색공간과 인코딩을 정의합니다.
- [Sharma, Wu, Dalal, "The CIEDE2000 Color-Difference Formula: Implementation Notes, Supplementary Test Data, and Mathematical Observations"](https://doi.org/10.1002/col.20070): $\Delta E_{00}$ 구현과 검증에 널리 쓰이는 실무 레퍼런스입니다.
- [NIST, "Color By Numbers"](https://www.nist.gov/publications/color-numbers-using-calibration-source-spectrally-matched-your-test-source-key): 분광 responsivity mismatch가 색 측정 정확도에 왜 영향을 주는지 설명합니다.
