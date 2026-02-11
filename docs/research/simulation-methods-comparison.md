# 전자기 시뮬레이션 방법론 비교 (EM Simulation Methods Comparison)

> 작성일: 2026-02-11 | COMPASS 프로젝트 연구 문서

---

## 1. 개요 (Overview)

CMOS 이미지 센서(CIS) 픽셀의 광학 시뮬레이션은 맥스웰 방정식(Maxwell's equations)의 수치적 풀이를 필요로 한다. 픽셀 피치가 가시광 파장(0.38--0.78 um)과 동일한 스케일(0.6--1.4 um)에 도달하면서, 기하광학만으로는 회절, 간섭, 근접장 결합 등의 파동 효과를 포착할 수 없게 되었다.

시뮬레이션 방법론의 선택은 세 가지 근본적 트레이드오프에 의해 결정된다:

| 축 | 설명 |
|---|---|
| **정확도 (Accuracy)** | 물리적 현실에 대한 충실도. 맥스웰 방정식의 근사 수준 |
| **속도 (Speed)** | 단일 시뮬레이션의 벽시계 시간(wall-clock time) |
| **범용성 (Generality)** | 다룰 수 있는 기하 구조 및 재료의 범위 |

어떤 단일 방법도 세 축을 동시에 최적화할 수 없으며, 이것이 다양한 수치 방법이 공존하는 근본적 이유다. COMPASS는 RCWA와 FDTD를 주력 솔버로 채택하고 교차 검증(cross-validation)으로 결과 신뢰성을 확보한다.

| 방법론 | 영역 | 차원 | 주기성 | 파장대역 | CIS 적합도 |
|--------|------|------|--------|----------|-----------|
| RCWA | 주파수 | 3D | 필수 | 단일 | ★★★★★ |
| FDTD | 시간 | 3D | 선택 | 광대역 | ★★★★☆ |
| FEM | 주파수 | 3D | 선택 | 단일 | ★★★☆☆ |
| BEM | 주파수 | 표면 | 불필요 | 단일 | ★★☆☆☆ |
| TMM | 주파수 | 1D | N/A | 단일 | ★★★☆☆ |
| Ray Tracing | N/A | 3D | 불필요 | 광대역 | ★★☆☆☆ |

---

## 2. RCWA (Rigorous Coupled-Wave Analysis)

### 2.1 수학적 기초 (Mathematical Foundation)

RCWA(엄밀 결합파 해석법)는 주기 구조에 대한 맥스웰 방정식을 주파수 영역에서 푸리에 급수로 전개하여 푸는 반해석적(semi-analytical) 방법이다. 핵심 단계:

**(1) 유전율의 푸리에 전개:** 주기 $\Lambda_x, \Lambda_y$ 구조에서 각 층의 유전율을 전개한다:

$$\varepsilon(x, y) = \sum_{p=-N}^{N} \sum_{q=-N}^{N} \hat{\varepsilon}_{pq} \, e^{i(G_{px} x + G_{qy} y)}$$

여기서 역격자 벡터(reciprocal lattice vector) $G_{px} = 2\pi p / \Lambda_x$이며, 총 고조파 수는 $M = (2N+1)^2$이다.

**(2) 고유값 문제 (Eigenvalue problem)**

각 층에서 푸리에 전개를 맥스웰 방정식에 대입하면 $2M \times 2M$ 고유값 문제가 도출된다:

$$\Omega \mathbf{w}_j = \gamma_j^2 \mathbf{w}_j$$

여기서 $\gamma_j$는 각 모드의 z방향 전파 상수(propagation constant), $\mathbf{w}_j$는 모드 프로파일이다.

**(3) S 행렬 연쇄 (S-matrix cascading)**

개별 층의 산란 행렬을 레드헤퍼 스타곱(Redheffer star product)으로 결합한다:

$$S_\text{total} = S_1 \star S_2 \star \cdots \star S_L$$

이 방법은 전달 행렬(T-matrix)과 달리 에바네센트 모드에 대해 수치적으로 안정적이다.

### 2.2 강점 (Strengths)

| 강점 | 설명 |
|------|------|
| **주기 구조 최적** | 이미지 센서의 픽셀 배열은 본질적으로 주기적이므로 RCWA에 이상적 |
| **박막 처리 정확** | 각 층을 공간 이산화 없이 정확히 처리 (anti-reflection coating, color filter) |
| **스펙트럼 분석** | 단일 파장 계산이 매우 빠름 → QE 스펙트럼을 파장별로 효율적 산출 |
| **지수 수렴** | 푸리에 차수 증가에 따라 지수적(exponential) 수렴 (매끄러운 프로파일) |
| **GPU 가속** | 행렬 연산 기반이므로 GPU 가속에 적합 (PyTorch/JAX 백엔드) |

### 2.3 약점 (Weaknesses)

| 약점 | 설명 |
|------|------|
| **비주기 구조 불가** | 유한 구조, 결함, 비주기 패턴은 원리적으로 처리 불가 |
| **곡면의 계단 근사** | 마이크로렌즈 등 곡면은 staircase approximation 필요 → 수렴 저하 |
| **메모리 스케일링** | 고유값 분해의 메모리 $O(M^2)$, 연산 $O(M^3)$. $N=15$이면 $M=961$, 행렬 크기 $1922 \times 1922$ |
| **분산 재료** | 각 파장마다 별도 계산 필요 (광대역 sweep 시 반복 비용) |

### 2.4 핵심 파라미터 (Key Parameters)

| 파라미터 | 역할 | COMPASS 기본값 |
|----------|------|---------------|
| **푸리에 차수** (Fourier order, $N$) | 공간 해상도 결정. 높을수록 정확하나 $O(N^6)$ 비용 증가 | `[9, 9]` |
| **리 인수분해** (Li's factorization) | 불연속 경계에서의 수렴성 개선. 역규칙(inverse rule), 법선 벡터법(normal vector method) | `li_inverse` |
| **편광** (Polarization) | TE/TM 또는 임의 편광. TM에서 Li 규칙이 특히 중요 | 무편광 (평균) |

**리의 푸리에 인수분해 규칙 (Li's Fourier factorization rules):**

Lifeng Li (1996)가 도입한 세 가지 규칙은 RCWA 수렴성의 핵심이다:

1. **로랑 규칙 (Laurent's rule)**: 두 함수에 동시 불연속이 없을 때 → $[\![f \cdot g]\!] = [\![f]\!] \cdot [\![g]\!]$
2. **역규칙 (Inverse rule)**: 모든 불연속이 상보적(complementary)일 때 → $[\![f \cdot g]\!] = [\![f^{-1}]\!]^{-1} \cdot [\![g]\!]$
3. **불가 조건**: 비상보적 동시 불연속이 존재하면 로랑/역규칙 모두 수렴 실패

### 2.5 CIS 적용 시나리오 (When to Use for CIS)

- **컬러 필터 (Color filter)**: 주기적 배열, 평면 층 → RCWA 최적
- **BARL (Bottom Anti-Reflection Layer)**: 박막 스택 최적화, 파장 sweep → RCWA 최적
- **마이크로렌즈 (Microlens)**: staircase 근사 필요하나, 차수 15+ 에서 충분한 정확도
- **금속 격자 (Metal grid) / DTI**: 급격한 유전율 불연속 → Li 역규칙 필수, 고차수 필요
- **파라미터 sweep**: 단일 파장 계산이 빠르므로 두께/피치/각도 sweep에 유리

---

## 3. FDTD (Finite-Difference Time-Domain)

### 3.1 수학적 기초 (Mathematical Foundation)

FDTD(유한차분 시간영역법)는 맥스웰 방정식의 회전(curl) 방정식을 시간과 공간에서 직접 이산화하는 방법이다.

**이 격자 (Yee lattice):**

Kane Yee (1966)가 제안한 엇갈린 격자(staggered grid)에서 전기장($\mathbf{E}$)과 자기장($\mathbf{H}$)의 6개 성분을 공간적으로 반 격자점만큼 엇갈려 배치한다. 이 배치는 2차 정확도의 중심차분을 자연스럽게 보장한다.

**갱신 방정식 (Leapfrog time-stepping):**

전기장과 자기장을 교대로 반 시간 스텝씩 갱신한다:

$$H_x^{n+1/2} = H_x^{n-1/2} + \frac{\Delta t}{\mu_0} \left( \frac{E_y^n|_{k+1} - E_y^n|_k}{\Delta z} - \frac{E_z^n|_{j+1} - E_z^n|_j}{\Delta y} \right)$$

$$E_x^{n+1} = E_x^n + \frac{\Delta t}{\varepsilon_0 \varepsilon_r} \left( \frac{H_z^{n+1/2}|_{j} - H_z^{n+1/2}|_{j-1}}{\Delta y} - \frac{H_y^{n+1/2}|_{k} - H_y^{n+1/2}|_{k-1}}{\Delta z} \right)$$

**쿠랑 안정성 조건 (Courant-Friedrichs-Lewy condition):**

시간 스텝은 다음 조건을 만족해야 수치적으로 안정적이다:

$$\Delta t \leq \frac{1}{c \sqrt{\frac{1}{\Delta x^2} + \frac{1}{\Delta y^2} + \frac{1}{\Delta z^2}}}$$

### 3.2 강점 (Strengths)

| 강점 | 설명 |
|------|------|
| **광대역 응답** | 단일 시뮬레이션으로 전체 가시광 스펙트럼 획득 (Fourier transform of impulse response) |
| **임의 기하 구조** | 격자 해상도 내에서 임의의 3D 구조 표현 가능. 주기성 불필요 |
| **시간 영역 정보** | 펄스 전파, 과도 응답(transient response)을 직접 관찰 가능 |
| **직관적 구현** | 갱신 방정식이 단순한 산술 연산 → 병렬화 및 GPU 가속 용이 |
| **비선형 재료** | 시간 영역에서 비선형 응답을 자연스럽게 포함 가능 |

### 3.3 약점 (Weaknesses)

| 약점 | 설명 |
|------|------|
| **CFL 제약** | 미세 격자 → 작은 시간 스텝 → 긴 시뮬레이션 시간. 박막에서 치명적 |
| **메모리** | 3D 격자 전체를 메모리에 유지. 5 nm 격자, 1 um 픽셀 → $200^3 \approx 8 \times 10^6$ 셀 × 6 성분 |
| **분산 재료** | 금속, 반도체의 주파수 의존 유전율을 보조 미분방정식(ADE)으로 처리해야 함 |
| **박막 비효율** | 수 nm 두께의 BARL도 전체 격자로 분해해야 함 (RCWA는 정확 해석) |
| **수치 분산** | 격자 해상도가 불충분하면 위상 속도에 인공적 분산 발생 ($\Delta x \leq \lambda / 20$ 권장) |

### 3.4 핵심 파라미터 (Key Parameters)

| 파라미터 | 역할 | CIS 시뮬레이션 권장값 |
|----------|------|---------------------|
| **격자 간격** ($\Delta x, \Delta y, \Delta z$) | 공간 해상도. $\lambda/(20n)$ 이하 권장 | 5--10 nm (가시광, Si) |
| **시간 스텝** ($\Delta t$) | CFL로 자동 결정 | ~0.01 fs (5 nm 격자) |
| **PML 층수** | 완전 정합층(Perfectly Matched Layer)의 두께 | 8--16 층 |
| **총 시간 스텝** | 정상 상태 도달까지 | 수천--수만 스텝 |
| **소스 유형** | 광대역 펄스 또는 연속파(CW) | Gaussian pulse (380--780 nm) |

**PML (Perfectly Matched Layer):** Berenger (1994)가 제안한 PML은 계산 영역 경계에서 나가는 파동을 무반사 흡수하는 인공 층이다. UPML과 CPML이 현재 주류 구현이다.

### 3.5 CIS 적용 시나리오 (When to Use for CIS)

- **복잡한 3D 구조**: 비대칭 DTI, 불규칙한 금속 배선 → FDTD 유리
- **RCWA 결과 검증**: 선택된 파장에서 FDTD spot-check로 교차 검증
- **광대역 QE**: 전체 가시광 대역을 단일 실행으로 획득 시
- **시간 영역 분석**: 마이크로렌즈를 통과하는 빛의 전파 과정 시각화
- **비주기 구조**: 단일 픽셀 분석, 에지 효과 연구

---

## 4. FEM (Finite Element Method)

유한요소법(FEM)은 맥스웰 방정식의 약형식(weak form)을 사면체/육면체 메시 상에서 푼다. 벡터 파동 방정식의 변분 공식화:

$$\int_\Omega \left[ \frac{1}{\mu_r} (\nabla \times \mathbf{E}) \cdot (\nabla \times \mathbf{F}) - k_0^2 \varepsilon_r \mathbf{E} \cdot \mathbf{F} \right] d\Omega = -ik_0 Z_0 \int_\Omega \mathbf{J} \cdot \mathbf{F} \, d\Omega$$

기하 구조를 사면체로 분할하며, 곡면 근처에서 적응 세분화(adaptive mesh refinement)가 가능한 것이 핵심 장점이다.

### 강점 및 약점

| 강점 | 약점 |
|------|------|
| 곡면 기하의 정확한 표현 (곡선 요소) | 대규모 희소 행렬 조립 및 풀이 비용 |
| 적응적 메시 세분화 (AMR) | 메시 생성 자체가 복잡 (특히 3D) |
| 다중물리 결합 용이 (열, 구조) | 주파수 영역 → 파장별 반복 필요 |
| 비균일/이방성 재료 처리 | 오픈소스 광학 FEM 솔버 제한적 |

COMPASS는 현재 FEM을 포함하지 않는다. CIS 픽셀의 주기성에는 RCWA가 더 효율적이며, 상용 FEM(COMSOL)은 라이선스 비용이 높다. 단, 마이크로렌즈 곡면이나 열-광학 다중물리 시뮬레이션에서는 FEM이 유일한 선택지가 될 수 있다.

---

## 5. BEM (Boundary Element Method)

경계요소법(BEM)은 체적 전체가 아닌 경계면(surface)에서만 미지수를 배치한다. 자유 공간 그린 함수 $G(\mathbf{r}, \mathbf{r}') = e^{ik|\mathbf{r}-\mathbf{r}'|} / (4\pi|\mathbf{r}-\mathbf{r}'|)$를 이용한 표면 적분 방정식으로 3D 체적 문제를 2D 표면 문제로 축소한다.

### 강점 및 약점

| 강점 | 약점 |
|------|------|
| 차원 축소: 3D 문제를 2D 표면 문제로 | 밀집 행렬(dense matrix) → $O(N^2)$ 메모리, $O(N^3)$ 풀이 |
| 개방 경계(open boundary) 자연스럽게 처리 | 비균일/비선형 재료 처리 어려움 |
| 원거리장(far-field) 계산에 효율적 | 적층 구조(layered media)에는 특수 그린 함수 필요 |
| 산란 문제에 최적화 | 다층 박막 스택에는 비효율적 |

BEM은 CIS 픽셀 시뮬레이션에 일반적으로 사용되지 않는다. 이미지 센서는 다층 구조이며 체적 전체에서의 전기장 분포가 중요하기 때문이다. 개별 마이크로렌즈의 산란 특성이나 금속 나노입자의 플라즈몬 응답 연구에는 유용할 수 있다.

---

## 6. TMM (Transfer Matrix Method)

### 6.1 수학적 기초 (Mathematical Foundation)

전달 행렬법(TMM)은 평면 다층 박막에서의 전자기파 전파를 행렬 곱으로 기술하는 해석적 방법이다.

**2x2 전달 행렬 (등방성, 수직 입사):**

각 층 $j$의 전달 행렬은:

$$M_j = \begin{pmatrix} \cos\delta_j & -\frac{i}{n_j}\sin\delta_j \\ -in_j\sin\delta_j & \cos\delta_j \end{pmatrix}$$

여기서 위상 두께(phase thickness) $\delta_j = \frac{2\pi}{\lambda} n_j d_j \cos\theta_j$이다.

전체 스택의 전달 행렬은 단순 곱이다:

$$M_\text{total} = M_1 \cdot M_2 \cdots M_L$$

**4x4 베레만 행렬 (Berreman matrix, 이방성):**

이방성 재료를 포함하는 경우, Berreman (1972)의 4x4 공식이 필요하다:

$$\frac{d}{dz} \boldsymbol{\Psi}(z) = \frac{i\omega}{c} \mathbf{D}(z) \boldsymbol{\Psi}(z)$$

여기서 $\boldsymbol{\Psi} = (E_x, H_y, E_y, -H_x)^T$이며, $\mathbf{D}$는 유전율 텐서로부터 구성되는 4x4 행렬이다.

### 6.2 강점 및 약점

| 강점 | 약점 |
|------|------|
| **극히 빠름**: 행렬 곱 몇 번으로 완료 | **1D 한정**: 수평 방향 패턴 처리 불가 |
| 해석적 정확도: 수치 이산화 오차 없음 | 두꺼운 흡수층에서 수치 불안정 가능 |
| 반사율/투과율/흡수율 즉시 산출 | 회절, 산란 현상 포착 불가 |
| 다층 박막 설계의 산업 표준 | 마이크로렌즈, 금속 격자 등 2D/3D 패턴 불가 |

### 6.3 CIS 적용 시나리오 (When to Use for CIS)

TMM은 COMPASS에서 직접 솔버로 사용되지 않지만, 다음 용도로 극히 유용하다:

- **초기 스택 설계**: BARL, ARC(Anti-Reflection Coating) 두께 최적화의 출발점
- **재료 스크리닝**: 컬러 필터 재료의 흡수/투과 특성 빠른 평가
- **해석적 검증**: 균일 층만으로 구성된 구조에서 RCWA 결과의 레퍼런스
- **1D 수렴 확인**: RCWA의 $N=0$ (0차만) 결과가 TMM과 일치해야 함

COMPASS의 `compass.materials.database.MaterialDB`는 TMM 방식의 박막 반사율 계산을 내부적으로 활용한다.

---

## 7. 레이 트레이싱 (Ray Tracing)

### 7.1 기하광학 근사 (Geometric Optics Approximation)

레이 트레이싱은 빛을 광선(ray)으로 취급하여 스넬의 법칙(Snell's law)과 프레넬 계수(Fresnel coefficients)로 전파를 추적한다. 맥스웰 방정식의 단파장 극한($\lambda \to 0$)에 해당한다.

**추적 방정식:**

$$\frac{d}{ds}\left(n \frac{d\mathbf{r}}{ds}\right) = \nabla n$$

여기서 $s$는 광선 경로를 따른 호 길이(arc length), $n(\mathbf{r})$은 굴절률 분포이다.

### 7.2 강점 및 약점

| 강점 | 약점 |
|------|------|
| **극히 빠름**: 수백만 광선을 초 단위로 추적 | 파장 스케일 구조에서 회절 무시 → 치명적 오류 |
| 직관적 물리: 광선 경로 시각화 용이 | 간섭 현상 포착 불가 (박막 효과 등) |
| 렌즈 시스템 설계의 표준 (Zemax, Code V) | 근접장(near-field) 결합 무시 |
| CRA(Chief Ray Angle) 분석에 적합 | 픽셀 피치 < 수 $\lambda$ 에서 급격히 부정확 |

### 7.3 CIS에서의 위치

현대 CIS 설계에서 레이 트레이싱은 파동 광학(wave optics)의 **전처리(pre-processing)** 단계로 주로 사용된다:

1. **카메라 렌즈 → 센서 면**: Zemax/Code V에서 CRA, 조사 분포(irradiance distribution) 계산
2. **핸드오프**: 센서 면에서의 입사 조건(각도, 진폭)을 추출
3. **파동 광학 시뮬레이션**: COMPASS의 RCWA/FDTD에서 해당 입사 조건으로 픽셀 시뮬레이션

COMPASS의 `compass.sources.ray_file_reader`와 `compass.sources.cone_illumination` 모듈이 이 핸드오프를 지원한다.

---

## 8. 하이브리드 방법 (Hybrid Methods)

단일 방법으로는 이미지 센서 시스템의 모든 스케일을 효율적으로 다룰 수 없다. 하이브리드 방법은 각 스케일에 최적인 방법을 조합한다.

### 8.1 레이 트레이싱 → RCWA 핸드오프 (Zemax → COMPASS)

```
카메라 렌즈 (mm 스케일)     →  Zemax (Ray Tracing)
    ↓ CRA, irradiance
픽셀 스택 (um 스케일)       →  COMPASS (RCWA/FDTD)
    ↓ QE, crosstalk
센서 성능 (pixel array)     →  시스템 분석
```

핵심 인터페이스: 입사각($\theta$, $\phi$), 편광 상태, 파워 분포.

### 8.2 FEM + 산란 행렬 (EMUstack 접근법)

EMUstack은 각 층을 2D FEM으로 풀고 층간 연결을 산란 행렬로 처리한다. FEM의 기하 유연성과 S-matrix의 수치 안정성을 결합하지만, 메시 생성의 복잡성은 여전히 존재한다.

### 8.3 다중 스케일 접근 (Multi-scale Approaches)

| 스케일 | 방법 | 대상 |
|--------|------|------|
| 수 mm | Ray Tracing | 카메라 렌즈, 마이크로렌즈 어레이 |
| 수 um | RCWA / FDTD | 컬러 필터, BARL, DTI |
| 수 nm | FEM / BEM | 플라즈모닉 나노구조, 표면 거칠기 |

미래 방향으로는 신경망(neural network) 기반 대리 모델(surrogate model)이 주목받고 있다. RCWA/FDTD의 학습 데이터로 훈련된 신경망이 실시간 예측을 제공하며, 정밀도가 필요한 지점에서만 정밀 솔버를 호출한다.

---

## 9. 방법론별 성능 비교표 (Comprehensive Comparison)

### 9.1 정성적 비교 (Qualitative Comparison)

| 특성 | RCWA | FDTD | FEM | BEM | TMM | Ray Tracing |
|------|------|------|-----|-----|-----|-------------|
| **정확도** | 높음 (주기) | 높음 | 매우 높음 | 높음 (표면) | 정확 (1D) | 낮음 (파동 효과 무시) |
| **속도 (단일 파장)** | 매우 빠름 | 느림 | 느림 | 중간 | 극히 빠름 | 극히 빠름 |
| **속도 (광대역)** | 중간 (반복) | 빠름 (단일 실행) | 느림 (반복) | 느림 (반복) | 극히 빠름 | 극히 빠름 |
| **메모리** | 중간 | 높음 | 높음 | 높음 (밀집) | 극히 낮음 | 낮음 |
| **기하 유연성** | 낮음 (주기만) | 높음 | 매우 높음 | 중간 | 없음 (1D만) | 높음 (매크로) |
| **재료 유연성** | 높음 | 중간 | 매우 높음 | 낮음 | 높음 | 중간 |
| **자동미분 (AD)** | 가능 | 가능 | 제한적 | 어려움 | 가능 | 어려움 |
| **GPU 가속** | 매우 적합 | 적합 | 제한적 | 제한적 | 불필요 | 적합 |

### 9.2 계산 복잡도 비교 (Computational Scaling)

| 방법 | 공간 자유도 | 시간 복잡도 | 메모리 복잡도 | 병목 |
|------|-----------|-----------|-------------|------|
| **RCWA** | $M = (2N+1)^2$ | $O(M^3)$ per layer | $O(M^2)$ | 고유값 분해 |
| **FDTD** | $N_x N_y N_z$ | $O(N_\text{total} \cdot T)$ | $O(N_\text{total})$ | 시간 스텝 수 $T$ |
| **FEM** | $N_\text{DOF}$ (메시 노드) | $O(N_\text{DOF}^{1.5})$ sparse | $O(N_\text{DOF})$ sparse | 행렬 풀이 |
| **BEM** | $N_\text{surface}$ | $O(N_\text{surface}^3)$ | $O(N_\text{surface}^2)$ | 밀집 행렬 |
| **TMM** | $L$ (층 수) | $O(L)$ | $O(1)$ | 없음 |
| **Ray Tracing** | $N_\text{rays}$ | $O(N_\text{rays} \cdot S)$ | $O(N_\text{rays})$ | 광선 수 $N_\text{rays}$, 표면 수 $S$ |

### 9.3 대표 실행 시간 (Typical Problem Sizes)

1 um 피치 BSI 픽셀, 2x2 Bayer 단위셀, 550 nm 기준:

| 방법 | 파라미터 설정 | 자유도 | 단일 파장 시간 | 41-파장 sweep |
|------|-------------|--------|-------------|--------------|
| **RCWA** (GPU) | $N = 9$, 10 layers | ~7,000 | **0.3 s** | 12 s |
| **RCWA** (GPU) | $N = 15$, 10 layers | ~19,000 | 2 s | 80 s |
| **FDTD** (GPU) | $\Delta x = 5$ nm, PML 12 | ~8M cells | 45 s | **45 s** (광대역) |
| **FDTD** (CPU) | $\Delta x = 10$ nm, PML 8 | ~1M cells | 300 s | 300 s |
| **FEM** | 적응 메시, $\lambda/10$ | ~500K DOF | 60 s | 2,460 s |
| **TMM** | 10 layers | 10 | **< 0.001 s** | 0.04 s |

> **주의**: 위 수치는 대표적 추정값이며, 하드웨어(GPU: NVIDIA A100, CPU: 8-core)와 구현에 따라 크게 달라질 수 있다.

### 9.4 미분가능 시뮬레이션 지원 (Differentiable Simulation Support)

역설계(inverse design)와 토폴로지 최적화를 위한 자동미분(AD) 지원 현황:

| 방법 | AD 프레임워크 | 그래디언트 방식 | 대표 솔버 |
|------|-------------|---------------|----------|
| **RCWA** | PyTorch, JAX | Forward/Reverse AD | meent, fmmax, torcwa |
| **FDTD** | PyTorch, JAX | Reverse AD, Adjoint | FDTDX, flaport, fdtdz |
| **FEM** | 제한적 | Adjoint method | EMOPT (FDFD) |
| **TMM** | 용이 | Analytical gradient | 자체 구현 |

---

## 10. COMPASS에서의 적용 (Application in COMPASS)

### 10.1 RCWA + FDTD 선택 이유

| 기준 | RCWA | FDTD | 선택 이유 |
|------|------|------|----------|
| CIS 픽셀의 주기성 | 완벽 적합 | 적합 | 픽셀 배열 = 주기 구조 |
| 박막 스택 처리 | 정확 해석 | 격자 이산화 | BARL/ARC 설계에 RCWA 우위 |
| 교차 검증 | - | - | 서로 다른 수학적 접근법으로 독립 검증 |
| GPU 가속 | 매우 적합 | 적합 | PyTorch/JAX 기반 오픈소스 활용 |
| 라이선스 | MIT 가능 | MIT 가능 | meent(MIT), flaport(MIT) |

### 10.2 교차 검증 철학 (Cross-Validation Philosophy)

동일한 물리 법칙을 서로 다른 수학적 접근으로 풀기 때문에, 두 솔버의 일치는 결과 신뢰도를 크게 높인다. 불일치 시 점검 사항:

1. **RCWA 수렴 부족** → 푸리에 차수 증가
2. **FDTD 해상도 부족** → 격자 미세화
3. **모델링 차이** → staircase 근사, 재료 모델, 경계 조건 검토
4. **에너지 보존 위반** ($R + T + A \neq 1$) → 구현 버그

`SolverComparison` 클래스가 QE 차이, 상대 오차, 에너지 보존 검증을 자동화한다.

### 10.3 솔버 선택 가이드 (Decision Guide)

```
시뮬레이션 시작
    │
    ├─ 구조가 주기적인가?
    │   ├─ YES → 박막만 있는가?
    │   │         ├─ YES → TMM (초기 설계) → RCWA (정밀)
    │   │         └─ NO  → RCWA (기본) + FDTD (검증)
    │   └─ NO  → FDTD
    │
    ├─ 광대역이 필요한가?
    │   ├─ YES, 50+ 파장 → FDTD (단일 실행이 효율적)
    │   └─ NO, < 50 파장 → RCWA (파장별 반복이 더 빠름)
    │
    └─ 시간 영역 정보가 필요한가?
        ├─ YES → FDTD
        └─ NO  → RCWA (기본 선택)
```

### 10.4 미래 확장 (Future Directions)

COMPASS의 솔버 확장 로드맵:

| 우선순위 | 솔버/방법 | 목적 |
|---------|----------|------|
| **높음** | fmmax (RCWA) 통합 | 벡터 FMM으로 수렴성 향상, JAX 배칭 |
| **높음** | FDTDX (FDTD) 통합 | 멀티GPU 3D, 대규모 역설계 |
| **중간** | TMM 모듈 내장 | 빠른 스택 사전 설계, 1D 레퍼런스 |
| **중간** | 신경망 대리 모델 | 실시간 파라미터 최적화 |
| **낮음** | FEM 통합 (EMUstack) | 플라즈모닉/곡면 특수 연구 |

---

## 참고 문헌 (References)

### 핵심 논문

- K. S. Yee, "Numerical solution of initial boundary value problems involving Maxwell's equations in isotropic media," *IEEE Trans. Antennas Propag.*, vol. 14, no. 3, pp. 302-307, 1966.
- M. G. Moharam and T. K. Gaylord, "Rigorous coupled-wave analysis of planar-grating diffraction," *J. Opt. Soc. Am.*, vol. 71, no. 7, pp. 811-818, 1981.
- L. Li, "Use of Fourier series in the analysis of discontinuous periodic structures," *J. Opt. Soc. Am. A*, vol. 13, no. 9, pp. 1870-1876, 1996.
- J.-P. Berenger, "A perfectly matched layer for the absorption of electromagnetic waves," *J. Comput. Phys.*, vol. 114, no. 2, pp. 185-200, 1994.
- D. W. Berreman, "Optics in stratified and anisotropic media: 4x4-matrix formulation," *J. Opt. Soc. Am.*, vol. 62, no. 4, pp. 502-510, 1972.

### 웹 자료

- [Planopsim: RCWA vs FDTD Benchmark](https://planopsim.com/design-example/getting-accurate-and-fast-nano-structure-simulations-a-benchmark-of-rcwa-and-fdtd-for-meta-surface-calculation/)
- [Ansys: CMOS Optical Simulation Methodology](https://optics.ansys.com/hc/en-us/articles/360042851793-CMOS-Optical-simulation-methodology)
- [Joint EM and Ray-Tracing Simulations for Quad-Pixel Sensor](https://opg.optica.org/oe/fulltext.cfm?uri=oe-27-21-30486&id=422094)
- [FDTD vs FEM vs MoM - Cadence](https://resources.system-analysis.cadence.com/blog/msa2021-fdtd-vs-fem-vs-mom-what-are-they-and-how-are-they-different)
- [3D Broadband FDTD Simulations of CMOS Image Sensor](https://arxiv.org/abs/2310.10302)
- [VarRCWA: Adaptive High-Order RCWA](https://pmc.ncbi.nlm.nih.gov/articles/PMC9589908/)
