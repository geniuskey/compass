# 이미지 센서 광학

이 페이지에서는 COMPASS의 주요 시뮬레이션 대상인 후면 조사(Backside-Illuminated, BSI) CMOS 이미지 센서 픽셀의 광학 구조를 설명합니다.

## BSI 픽셀 구조

BSI 픽셀에서 빛은 실리콘 후면(배선측의 반대편)을 통해 입사합니다. 픽셀 스택은 상단(빛 입사측)에서 하단 순으로 다음과 같이 구성됩니다:

```
              Incident light
                   |
                   v
    +---------------------------------+
    |             Air                  |
    +---------------------------------+
    |          Microlens               |   Focuses light into pixel center
    +---------------------------------+
    |       Planarization (SiO2)       |   Uniform dielectric
    +---------------------------------+
    |  Color Filter (Bayer pattern)    |   Wavelength-selective absorption
    |  + Metal Grid (W)               |   Optical isolation between pixels
    +---------------------------------+
    |   BARL (anti-reflection stack)   |   Minimizes reflection at CF/Si
    +---------------------------------+
    |         Silicon                  |   Absorbs photons, generates e-h pairs
    |   [Photodiode regions]           |   Collects charge
    |   [DTI trenches]                 |   Prevents optical/electrical crosstalk
    +---------------------------------+
```

<StackVisualizer />

## 마이크로렌즈

마이크로렌즈(Microlens)는 입사광을 픽셀 중심으로 집광하는 곡면 폴리머 구조입니다. COMPASS에서 마이크로렌즈는 초타원(Superellipse) 프로파일로 모델링됩니다:

$$h(x,y) = H \left(1 - \left|\frac{x - c_x}{r_x}\right|^n - \left|\frac{y - c_y}{r_y}\right|^n \right)^{1/\alpha}$$

매개변수:
- $H$: 렌즈 높이 (일반적으로 0.4~0.8 um)
- $r_x, r_y$: 반축 (일반적으로 피치의 절반보다 약간 작음)
- $n$: 사각성 매개변수 (2.0 = 타원, 값이 클수록 더 사각형)
- $\alpha$: 곡률 제어

마이크로렌즈 중심은 **주광선각(Chief Ray Angle, CRA)**을 고려하여 이동시킬 수 있습니다. CRA는 카메라 렌즈로부터 빛이 도달하는 각도입니다. 이미지 센서 가장자리에 위치한 픽셀은 더 큰 CRA에서 빛을 받으므로, 양호한 집광 효율을 유지하기 위해 마이크로렌즈를 이동시켜야 합니다.

## 컬러 필터 배열

컬러 필터 배열(Color Filter Array, CFA)은 선택적으로 빛을 흡수하여 색 감도를 만들어냅니다. 가장 일반적인 패턴은 **베이어 RGGB(Bayer RGGB)** 배열입니다:

```
  +---+---+
  | R | G |
  +---+---+
  | G | B |
  +---+---+
```

각 컬러 필터 재료는 파장에 따른 복소 굴절률을 가지며, 통과 대역 밖에서는 흡수($k > 0$)가 크고 통과 대역 내에서는 흡수가 작습니다. 필터는 원하지 않는 파장을 흡수하고 대상 색상을 투과시킵니다.

<BayerPatternViewer />

컬러 필터 서브픽셀 사이의 **금속 격자(Metal Grid)**(일반적으로 텅스텐, 40~80 nm 폭)는 광학적 격리를 제공하여, 인접한 색상 채널 간의 빛 누설을 방지합니다.

## BARL: 하부 반사 방지층

BARL(Bottom Anti-Reflection Layer)은 컬러 필터와 실리콘 사이의 계면에 위치한 얇은 다층 유전체 스택입니다. 이 고대비 계면에서의 반사를 최소화하는 것이 목적입니다.

BARL이 없으면, 컬러 필터/실리콘 계면($n \approx 1.55$에서 $n \approx 4.0$)에서의 반사율은 다음과 같습니다:

$$R = \left(\frac{n_\text{Si} - n_\text{CF}}{n_\text{Si} + n_\text{CF}}\right)^2 \approx 20\%$$

잘 설계된 BARL 스택(예: SiO2/HfO2/Si3N4)은 가시광 스펙트럼 전체에서 이 값을 5% 이하로 줄일 수 있습니다.

## 실리콘과 포토다이오드

실리콘(Silicon)은 광자가 전자로 변환되는 흡수 매질입니다. 흡수 깊이(Absorption Depth)는 파장에 크게 의존합니다:

| 파장 | 색상 | Si에서의 흡수 깊이 |
|------------|-------|------------------------|
| 400 nm | 보라색 | ~0.1 um |
| 450 nm | 파란색 | ~0.4 um |
| 550 nm | 녹색 | ~1.7 um |
| 650 nm | 빨간색 | ~3.3 um |
| 800 nm | 근적외선 | ~10 um |

이는 청색 빛은 표면 근처에서 흡수되지만, 적색/근적외선 빛은 수 마이크로미터의 실리콘이 필요함을 의미합니다. 일반적인 BSI 픽셀의 실리콘 두께는 2~4 um입니다.

**포토다이오드(Photodiode)**는 실리콘 내의 정해진 영역을 차지합니다. 포토다이오드 체적 내에서 흡수된 광자만이 광전류에 기여합니다. COMPASS는 포토다이오드 경계 박스 내의 흡수 전력을 적분하여 이를 모델링합니다.

## 심부 트렌치 격리 (DTI)

심부 트렌치 격리(Deep Trench Isolation, DTI)는 저굴절률 재료(일반적으로 SiO2, $n \approx 1.46$)로 채워진 수직 트렌치로, 실리콘 내에서 인접 픽셀을 광학적으로 격리합니다. 실리콘($n \approx 3.5$~$4.0$)과 SiO2 사이의 큰 굴절률 차이로 트렌치 벽에서 전반사(Total Internal Reflection)가 발생하여, 빛이 인접 픽셀로 건너가는 것을 방지합니다.

DTI의 핵심 역할:
- 광학적 크로스토크(Crosstalk) 감소
- 색 재현성 향상
- MTF(Modulation Transfer Function, 변조 전달 함수) 유지

## BSI 픽셀에서의 광학 현상

| 효과 | 메커니즘 | QE에 대한 영향 |
|--------|-----------|--------------|
| 박막 간섭 | BARL/평탄화층에서의 다중빔 간섭 | 스펙트럼 리플 |
| 회절 | 서브파장 금속 격자 | 각도 의존적 광 재분배 |
| 도파 모드 | DTI가 실리콘 도파관 형성 | 빛을 가두어 QE 향상 또는 저하 가능 |
| 마이크로렌즈 집광 | 굴절 | 픽셀 중심에 빛을 집중 |
| 광학적 크로스토크 | 인접 픽셀로의 빛 누설 | 색 정확도 저하 |
| 전반사 | 고굴절률 Si / 저굴절률 주변부 | 빛을 가두어 유효 광경로 증가 |

이 모든 효과는 COMPASS의 전파 전자기(Full-Wave EM) 솔버(RCWA, FDTD)에 의해 자동으로 포착됩니다.
