# 수치 안정성

RCWA 시뮬레이션은 높은 푸리에 차수, 짧은 파장, 흡수 재료 등의 조건에서 수치적 불안정(Numerical Instability)이 발생할 수 있습니다. COMPASS는 이러한 문제를 해결하기 위해 5단계 방어 체계를 구현합니다.

## 불안정의 원인

<PrecisionComparison />

### 1. 고유값 분해 오류

RCWA의 핵심은 $2M \times 2M$ 고유값 문제(Eigenvalue Problem)를 푸는 것입니다($M$이 300 이상일 수 있음). 단정밀도(Float32)에서는 제한된 가수(Mantissa) 비트로 인해 다음과 같은 문제가 발생합니다:

- 고유벡터의 직교성 상실
- 고유값 순서 오류
- 거의 축퇴된 고유값 쌍의 혼합

### 2. T 행렬에서의 지수 오버플로

전통적인 전달 행렬(T-matrix) 접근법은 다음과 같이 층을 통해 전기장을 전파합니다:

$$T_j = \begin{pmatrix} \exp(+\lambda d) & \cdots \\ \cdots & \exp(-\lambda d) \end{pmatrix}$$

소멸 모드(Evanescent Mode)의 경우, $\lambda$가 허수이고 $\exp(+|\lambda|d)$가 지수적으로 증가합니다. 높은 푸리에 차수에서 이러한 발산 항은 오버플로와 파국적 소거(Catastrophic Cancellation)를 유발합니다.

### 3. NVIDIA GPU의 TF32

NVIDIA Ampere 이후 세대의 GPU는 float32 행렬 곱셈에 TF32(TensorFloat-32)를 기본으로 사용합니다. TF32는 가수 비트가 10비트에 불과하여(float32의 23비트 대비) RCWA의 정확도를 무선적으로 파괴합니다. **COMPASS는 기본적으로 TF32를 비활성화합니다.**

### 4. 불연속점에서의 푸리에 인수분해

급격한 재료 경계(금속 격자, DTI)에서 유전율의 표준 푸리에 표현은 수렴이 느리며 비물리적 결과를 생성할 수 있습니다(깁스 현상). 이는 주로 TM 편광에 영향을 미칩니다.

## 5단계 방어 체계

COMPASS는 `compass.solvers.rcwa.stability`의 전용 모듈로 각 불안정 원인에 대응합니다:

### 1단계: PrecisionManager

모든 단계에서의 부동소수점 정밀도를 제어합니다:

```yaml
solver:
  stability:
    precision_strategy: "mixed"    # float32 for most, float64 for eigendecomp
    allow_tf32: false              # CRITICAL: disable TF32
    eigendecomp_device: "cpu"      # CPU eigendecomp is more stable than GPU
```

`mixed` 전략은 대부분의 계산을 속도를 위해 float32로 유지하되, 고유값 분해를 float64로 승격시킵니다:

```python
# Internally, COMPASS does:
matrix_f64 = matrix.astype(np.complex128)
eigenvalues, eigenvectors = np.linalg.eig(matrix_f64)
# Then converts back to original precision
```

### 2단계: S 행렬 알고리즘

COMPASS는 T 행렬 대신 **레드헤퍼 스타곱(Redheffer Star Product)**을 사용합니다. 스타곱은 유계 수량만을 사용하여 S 행렬을 층별로 결합합니다:

$$S_\text{total} = S_1 \star S_2 \star \cdots \star S_N$$

핵심 특성: 모든 중간 지수 함수가 $\exp(-|\lambda|d)$ 형태이므로, 항상 $\leq 1$입니다. 오버플로가 발생할 수 없습니다.

### 3단계: 리 인수분해

급격한 유전체 경계가 있는 구조에 대해, COMPASS는 리의 역규칙(Li's Inverse Rule)을 적용합니다:

$$[\varepsilon]_\text{eff} = \left[\text{FT}\left(\frac{1}{\varepsilon}\right)\right]^{-1}$$

나이브한 $\text{FT}(\varepsilon)$ 대신 이를 사용하면 금속/유전체 계면에서 TM 편광의 수렴성이 크게 개선됩니다.

### 4단계: 고유값 안정화기

고유값에 대한 후처리로 다음을 처리합니다:

- **축퇴 고유값(Degenerate Eigenvalues)**: 두 고유값이 임계값(`eigenvalue_broadening: 1e-10`)보다 가까운 경우, 그람-슈미트(Gram-Schmidt) 과정으로 고유벡터를 직교화합니다.
- **분기 선택(Branch Selection)**: 고유값의 제곱근은 올바른 부호를 선택해야 합니다. COMPASS는 전파 모드에 대해 $\text{Re}(\sqrt{\lambda}) \geq 0$을 적용하고, 소멸 모드에 대해서는 올바른 감쇠 방향을 설정합니다.

### 5단계: 적응형 정밀도 실행기

에너지 균형 검사($|R + T + A - 1| > \text{tolerance}$)가 실패하면, COMPASS는 자동으로 더 높은 정밀도로 재시도합니다:

```
float32 (GPU)  --->  float64 (GPU)  --->  float64 (CPU)
     fast               slower              slowest but most stable
```

이는 다음과 같이 설정합니다:

```yaml
solver:
  stability:
    energy_check:
      enabled: true
      tolerance: 0.02
      auto_retry_float64: true
```

## 안정성 문제 진단

### 시뮬레이션 전 검사

`StabilityDiagnostics.pre_simulation_check` 메서드는 다음 상황에 대해 경고합니다:

- float32를 사용하는 대형 행렬 (위험: 고유값 분해 실패)
- S 행렬 없이 사용하는 두꺼운 층 (위험: T 행렬 오버플로)
- TF32 활성화 상태 (위험: 무선적 정확도 손실)
- 나이브 인수분해를 사용하는 패턴 층 (위험: 느린 수렴)

### 시뮬레이션 후 검사

시뮬레이션 후, `StabilityDiagnostics.post_simulation_check`는 다음을 검증합니다:

- 모든 픽셀에서 QE가 $[0, 1]$ 범위 내인지
- 결과에 NaN 또는 Inf가 없는지
- 에너지 보존: $|R + T + A - 1| < 0.05$

### 경고 징후

| 증상 | 가능한 원인 | 해결 방법 |
|---------|-------------|-----|
| QE > 100% 또는 < 0% | 고유값 분해 실패 | `precision_strategy: "float64"` 사용 |
| 에너지 균형 위반 | T 행렬 오버플로 | S 행렬 알고리즘 사용 확인 |
| 단파장에서의 노이즈가 있는 QE | float32 불충분 | `auto_retry_float64` 활성화 |
| TM 수렴 느림 | 나이브 인수분해 | `li_inverse`로 전환 |
| 조건수 경고 | 거의 특이한 행렬 | 푸리에 차수 감소 또는 브로드닝 증가 |

## 권장 설정

프로덕션 시뮬레이션의 경우:

```yaml
solver:
  stability:
    precision_strategy: "mixed"
    allow_tf32: false
    eigendecomp_device: "cpu"
    fourier_factorization: "li_inverse"
    energy_check:
      enabled: true
      tolerance: 0.02
      auto_retry_float64: true
    eigenvalue_broadening: 1.0e-10
    condition_number_warning: 1.0e+12
```

최대 정확도를 위한 설정(속도 희생):

```yaml
solver:
  stability:
    precision_strategy: "float64"
    allow_tf32: false
    eigendecomp_device: "cpu"
    fourier_factorization: "li_inverse"
```
