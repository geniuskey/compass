# 박막 광학

박막 간섭(Thin Film Interference)은 이미지 센서 설계에서 가장 중요한 광학 효과 중 하나입니다. 반사 방지 코팅(Anti-Reflection Coating), BARL 층, 그리고 평탄화층(Planarization Layer)까지 모두 박막 원리에 기반합니다.

## 단일 박막

두께 $d$, 굴절률(Refractive Index) $n_f$인 박막이 기판 위에 있는 경우를 생각해 봅시다. 박막의 상면과 하면에서 반사된 빛이 간섭합니다:

$$\text{Path difference} = 2 n_f d \cos\theta_f$$

여기서 $\theta_f$는 박막 내부에서의 전파각입니다. 최소 반사(두 반사빔의 상쇄 간섭)를 위한 간섭 조건은 다음과 같습니다:

$$2 n_f d \cos\theta_f = \left(m + \frac{1}{2}\right) \lambda, \quad m = 0, 1, 2, \ldots$$

수직 입사($m=0$)에서의 4분의 1 파장 반사 방지(AR) 코팅의 경우:

$$d = \frac{\lambda}{4 n_f}$$

그리고 반사가 0이 되려면 박막의 굴절률이 다음 조건을 만족해야 합니다:

$$n_f = \sqrt{n_\text{air} \cdot n_\text{substrate}}$$

<FresnelCalculator />

## 다층 스택

실제 이미지 센서는 여러 층의 박막을 사용합니다. 일반적인 COMPASS 구성에서 BARL(Bottom Anti-Reflection Layer, 하부 반사 방지층)은 SiO2, HfO2, Si3N4 층이 교대로 배치됩니다:

```yaml
barl:
  layers:
    - thickness: 0.010   # SiO2
      material: "sio2"
    - thickness: 0.025   # HfO2
      material: "hfo2"
    - thickness: 0.015   # SiO2
      material: "sio2"
    - thickness: 0.030   # Si3N4
      material: "si3n4"
```

다층 스택의 경우, 전달 행렬법(Transfer Matrix Method, TMM)이 정확한 해를 제공합니다. 각 층은 2x2 행렬로 표현됩니다:

$$M_j = \begin{pmatrix} \cos\delta_j & -\frac{i}{\eta_j}\sin\delta_j \\ -i\eta_j\sin\delta_j & \cos\delta_j \end{pmatrix}$$

여기서 $\delta_j = k_0 n_j d_j \cos\theta_j$는 위상 두께(Phase Thickness)이고 $\eta_j$는 어드미턴스(Admittance, TE와 TM에서 다른 값)입니다. 전체 시스템 행렬은 각 층 행렬의 곱입니다:

$$M = M_1 \cdot M_2 \cdots M_N$$

전체 반사 및 투과 계수는 이 행렬의 요소로부터 구할 수 있습니다.

<ThinFilmReflectance />

## BSI 픽셀에서의 역할

후면 조사(Backside-Illuminated, BSI) 픽셀에서 빛은 실리콘 후면을 통해 입사하며, 포토다이오드에 도달하기 전에 여러 층을 통과해야 합니다:

```
Incident light
      |
      v
  [Air]
  [Microlens]         -- focuses light onto pixel
  [Planarization]     -- uniform dielectric
  [Color filter]      -- wavelength-selective absorption
  [BARL layers]       -- anti-reflection at color-filter/silicon interface
  [Silicon + DTI]     -- photodiode region
```

BARL 스택은 컬러 필터-실리콘 계면에서의 반사를 최소화하도록 설계됩니다. BARL이 없으면 큰 굴절률 차이(컬러 필터 $n \approx 1.55$, 실리콘 $n \approx 4$)로 인해 약 30~40%의 반사가 발생하여 양자 효율(Quantum Efficiency, QE)이 크게 감소합니다.

## 스펙트럼 응답에 대한 영향

박막 간섭은 QE 스펙트럼에 파장 의존적인 진동을 만들어냅니다. 이러한 "파브리-페로(Fabry-Perot)" 프린지는 이미지 센서 시뮬레이션에서 흔히 관찰되는 특징입니다:

- **보강 간섭(Constructive Interference)**: 특정 파장에서 QE를 증가시킵니다.
- **상쇄 간섭(Destructive Interference)**: 다른 파장에서 QE 감소(딥)를 만듭니다.

프린지 간격은 대략 다음과 같습니다:

$$\Delta\lambda \approx \frac{\lambda^2}{2 n d}$$

3 um 실리콘 층에서 600 nm 기준으로, $\Delta\lambda \approx 15$ nm입니다. 이는 프린지를 분해하기 위해 최소 5 nm 이하의 파장 간격이 필요하다는 것을 의미합니다(나이퀴스트 기준).

::: warning
QE 스펙트럼이 들쭉날쭉하거나 예상치 못한 진동을 보이는 경우, 파장 간격이 박막 프린지를 분해하기에 충분히 작은지 확인하십시오. 10 nm 이하의 간격을 권장합니다.
:::

## COMPASS 구현

COMPASS는 솔버에 따라 박막을 다르게 처리합니다:

- **RCWA**: 각 균일 박막 층을 두께 $d$와 유전율 $\varepsilon(\lambda)$를 가진 단일 레이어 슬라이스로 정확하게 표현합니다. 근사가 필요하지 않습니다.
- **FDTD**: 박막은 공간 격자로 분해되어야 합니다. 10 nm BARL 층의 경우 격자 간격이 5 nm 이하여야 하므로, 메모리와 연산 시간이 증가할 수 있습니다.

`PixelStack` 클래스는 YAML 구성에서 모든 층을 자동으로 구성하고, 각 솔버 유형에 적합한 표현을 제공합니다.
