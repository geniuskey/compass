# FDTD 상세 설명

유한차분 시간영역법(Finite-Difference Time-Domain, FDTD)은 COMPASS의 두 번째 주요 솔버 방법입니다. 이산화된 공간 및 시간 격자 위에서 맥스웰 방정식을 직접 풉니다.

## 핵심 개념

FDTD는 맥스웰 방정식의 연속 미분을 엇갈린 격자(Staggered Grid), 즉 **이 격자(Yee Lattice)** 위에서의 유한 차분으로 대체합니다. 전기장과 자기장은 도약 개구리(Leapfrog) 시간 전진 기법으로 교대로 업데이트됩니다:

1. $\mathbf{E}$로부터 $\mathbf{H}$를 업데이트합니다 (반 시간 단계).
2. $\mathbf{H}$로부터 $\mathbf{E}$를 업데이트합니다 (반 시간 단계).
3. 시뮬레이션이 정상 상태(Steady State)에 도달하거나 펄스가 완전히 전파될 때까지 반복합니다.

## 이 격자

이 셀(Yee Cell)은 각 격자 셀 내에서 여섯 개의 전기장 성분($E_x, E_y, E_z, H_x, H_y, H_z$)을 엇갈린 위치에 배치합니다:

```
        Hz ---- Ey
        |       |
        Ex      |
        |       |
        Ey ---- Hz
```

이러한 엇갈림 배치 덕분에 모든 유한 차분 회전(Curl) 근사가 2차 정확도를 가지며, 발산 조건($\nabla \cdot \mathbf{B} = 0$)을 자연스럽게 만족합니다.

## 업데이트 방정식

비자성 매질에서 단일 성분(예: $E_x$)에 대한 업데이트 방정식은 다음과 같습니다:

$$E_x^{n+1}(i,j,k) = E_x^n(i,j,k) + \frac{\Delta t}{\varepsilon_0 \varepsilon_r(i,j,k)} \left( \frac{H_z^{n+1/2}(i,j,k) - H_z^{n+1/2}(i,j-1,k)}{\Delta y} - \frac{H_y^{n+1/2}(i,j,k) - H_y^{n+1/2}(i,j,k-1)}{\Delta z} \right)$$

자기장 성분도 유사하게 업데이트됩니다.

## 안정성: 쿠랑 조건

시간 간격 $\Delta t$은 수치적 불안정을 방지하기 위해 쿠랑-프리드리히스-레비(Courant-Friedrichs-Lewy, CFL) 조건을 만족해야 합니다:

$$\Delta t \leq \frac{1}{c \sqrt{\frac{1}{\Delta x^2} + \frac{1}{\Delta y^2} + \frac{1}{\Delta z^2}}}$$

여기서 $c$는 빛의 속도입니다. COMPASS FDTD 솔버는 격자 간격으로부터 최대 안정 시간 간격을 자동으로 계산합니다.

## 경계 조건

### 주기 경계 조건 (블로흐)

평면파 여기를 사용하는 픽셀 시뮬레이션에서, COMPASS는 수평(xy) 방향으로 **블로흐 주기 경계 조건(Bloch Periodic Boundary Conditions)**을 사용합니다:

$$\mathbf{E}(x + \Lambda_x, y, z) = \mathbf{E}(x, y, z) \, e^{ik_x \Lambda_x}$$

이는 동일한 픽셀의 무한 배열을 모델링하며, RCWA의 가정과 일치합니다.

### 흡수 경계 조건 (PML)

수직(z) 방향에서는 **완전 정합 흡수층(Perfectly Matched Layer, PML)**이 나가는 파동을 반사 없이 흡수합니다. PML은 경계 내부로 점진적으로 증가하는 전도도를 가진 손실 매질입니다:

$$\sigma(z) = \sigma_\text{max} \left(\frac{z}{d_\text{PML}}\right)^p$$

지수 $p$(일반적으로 3~4)와 PML 두께(일반적으로 10~20 셀)가 흡수 품질을 제어합니다. 급격한 PML은 허위 반사(Spurious Reflection)를 유발할 수 있습니다.

## 소스 주입

COMPASS는 **전체장/산란장(Total-Field/Scattered-Field, TFSF)** 기법 또는 소스 평면에서의 전기장 값 지정을 통해 평면파를 주입합니다. 단색 시뮬레이션의 경우, 연속파(Continuous-Wave, CW) 소스를 사용하며 전기장이 정상 상태에 도달할 때까지 시뮬레이션을 실행합니다.

## 결과 추출

FDTD 시뮬레이션이 정상 상태에 도달한 후, COMPASS는 다음을 추출합니다:

- **반사율 및 투과율**: 구조 위와 아래의 모니터 평면을 통과하는 포인팅 플럭스(Poynting Flux).
- **전기장 분포**: 임의의 평면에서 $|\mathbf{E}|^2$의 스냅샷.
- **픽셀별 흡수**: 각 포토다이오드(Photodiode) 영역 내에서 $\frac{1}{2}\omega\varepsilon'' |\mathbf{E}|^2$의 체적 적분.

## 격자 해상도

공간 격자는 가장 작은 기하학적 구조와 모든 재료에서의 최단 파장을 모두 분해해야 합니다:

$$\Delta x \leq \frac{\lambda_\text{min}}{n_\text{max} \cdot N_\text{ppw}}$$

여기서 $N_\text{ppw}$는 파장당 격자점 수(일반적으로 2차 FDTD에서 15~20)입니다. 400 nm에서 실리콘($n \approx 4$)의 경우:

$$\Delta x \leq \frac{0.4 \text{ um}}{4 \times 20} = 5 \text{ nm}$$

이로 인해 RCWA에 비해 큰 격자와 긴 실행 시간이 필요할 수 있습니다.

## COMPASS의 FDTD 솔버

| 솔버 | 라이브러리 | GPU 지원 | 비고 |
|--------|---------|-------------|-------|
| `fdtd_flaport` | fdtd (flaport) | CUDA (PyTorch) | 경량, 프로토타이핑에 적합. |
| `fdtdz` | fdtdz | CUDA | 적층된 2.5D 구조에 특화. |
| `meep` | Meep | CPU (MPI) | 완전한 기능, 분산 재료를 기본 지원. |

## FDTD를 선택해야 하는 경우

FDTD가 더 적합한 경우는 다음과 같습니다:

- 구조에 **비주기적(Non-periodic)** 또는 **비반복적(Aperiodic)** 특징이 있는 경우.
- 파장 루프 없이 단일 시뮬레이션 실행으로 **광대역(Broadband)** 응답이 필요한 경우(펄스 여기).
- **시간 영역** 전기장 전파를 시각화하고자 하는 경우.
- 푸리에 공간 데이터에서 추출하기 어려운 **근접장(Near-field)** 효과를 연구하는 경우.

표준적인 주기 픽셀 시뮬레이션에서는 RCWA가 일반적으로 더 빠르고 정확합니다. 자세한 비교는 [RCWA vs FDTD](./rcwa-vs-fdtd)를 참조하십시오.
