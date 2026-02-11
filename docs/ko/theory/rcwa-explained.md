# RCWA 상세 설명

::: tip 선수 지식
[전자기파](/ko/theory/electromagnetic-waves) → [회절](/ko/theory/diffraction) → 이 페이지
RCWA가 처음이라면 먼저 [솔버 선택 가이드](/ko/guide/choosing-solver)에서 개요를 확인하세요.
:::

RCWA(엄밀 결합파 해석법)는 COMPASS의 두 가지 주요 솔버 유형 중 하나입니다. 픽셀 구조를 공간 주파수 성분(푸리에 급수)으로 분해한 다음, 각 성분에 대해 맥스웰 방정식을 푸는 방법이라고 생각하면 됩니다. 픽셀 배열과 같은 주기 구조에 특히 빠릅니다.

엄밀 결합파 해석법(Rigorous Coupled-Wave Analysis, RCWA)은 COMPASS의 주요 솔버 방법입니다. 주파수 영역에서 전기장을 푸리에 고조파(Fourier Harmonics)로 전개하여 주기 구조에 대한 맥스웰 방정식을 풉니다.

## 핵심 개념

RCWA는 문제를 두 부분으로 분해합니다:

1. **수평 방향(xy)**: 주기적인 유전율과 전기장을 절단된 푸리에 급수로 전개합니다.
2. **수직 방향(z)**: 각 층 내에서 푸리에 계수가 만족하는 결합 상미분방정식(Coupled ODE) 계를 고유값 분해(Eigendecomposition)로 풉니다.

그런 다음 산란 행렬(S-matrix) 재귀를 사용하여 모든 계면에서의 경계 조건을 적용하며 층을 연결합니다.

## 단계별 알고리즘

### 1. 유전율의 푸리에 전개

주기 $\Lambda_x$와 $\Lambda_y$를 가진 2차원 주기 구조에서 각 층의 유전율을 전개합니다:

$$\varepsilon(x, y) = \sum_{p,q} \hat{\varepsilon}_{pq} \, e^{i(G_{px} x + G_{qy} y)}$$

여기서 $G_{px} = 2\pi p / \Lambda_x$, $G_{qy} = 2\pi q / \Lambda_y$는 역격자 벡터(Reciprocal Lattice Vector)입니다. 전개는 $|p| \leq N_x$, $|q| \leq N_y$로 절단되어, 총 $(2N_x+1)(2N_y+1)$개의 고조파를 사용합니다.

COMPASS에서 푸리에 차수는 솔버 설정에서 지정합니다:

```yaml
solver:
  name: torcwa
  params:
    fourier_order: [9, 9]   # Nx=Ny=9 -> 19x19 = 361 harmonics
```

### 2. 고유값 문제

각 층 내에서 푸리에 전개를 맥스웰 방정식에 대입하면, 차원이 $2M \times 2M$인 고유값 문제(Eigenvalue Problem)가 도출됩니다. 여기서 $M = (2N_x+1)(2N_y+1)$입니다:

$$\Omega \mathbf{w} = \lambda^2 \mathbf{w}$$

행렬 $\Omega$는 유전율 패턴으로 인한 푸리에 고조파 간의 결합을 나타냅니다. 고유값 $\lambda_j$는 전파 상수(Propagation Constant)를, 고유벡터 $\mathbf{w}_j$는 모드 프로파일(Mode Profile)을 나타냅니다.

균일한 층의 경우, $\Omega$는 대각 행렬이 되며 고유값은 단순한 평면파 전파 상수가 됩니다.

### 3. S 행렬 재귀

개별 층의 S 행렬은 **레드헤퍼 스타곱(Redheffer Star Product)**을 사용하여 결합되며, 이는 전달 행렬(T-matrix) 연쇄의 수치적 불안정성을 방지합니다:

$$S_\text{total} = S_1 \star S_2 \star \cdots \star S_N$$

$S^A$와 $S^B$를 결합하는 스타곱 공식은 다음과 같습니다:

$$S_{11} = S^A_{11} + S^A_{12} D S^B_{11} S^A_{21}$$

$$S_{12} = S^A_{12} D S^B_{12}$$

여기서 $D = (I - S^B_{11} S^A_{22})^{-1}$입니다. 이 공식은 모든 중간 행렬이 감쇠 지수 함수만 포함하므로 수치적으로 안정적입니다.

### 4. 전기장 계산

전역 S 행렬을 구하면, 반사 및 투과 회절 차수를 직접 구할 수 있습니다:

$$\begin{pmatrix} \mathbf{r} \\ \mathbf{t} \end{pmatrix} = \begin{pmatrix} S_{11} & S_{12} \\ S_{21} & S_{22} \end{pmatrix} \begin{pmatrix} \mathbf{a} \\ \mathbf{0} \end{pmatrix}$$

여기서 $\mathbf{a}$는 입사파 진폭입니다. 회절 효율(차수별 반사 및 투과 전력)은 다음과 같습니다:

$$R_m = \frac{\text{Re}(k_{z,m}^r)}{k_{z,0}^i} |r_m|^2$$

$$T_m = \frac{\text{Re}(k_{z,m}^t)}{k_{z,0}^i} |t_m|^2$$

## 푸리에 인수분해

RCWA의 수렴성은 유전율의 푸리에 계수를 계산하는 방법에 결정적으로 의존합니다. 급격한 재료 경계(예: 금속 격자)가 있는 구조에서는 기존 방법으로 TM 편광에 대한 수렴이 느립니다.

**리의 역규칙(Li's Inverse Rule)**은 적절한 전기장 성분에 대해 $\text{FT}(\varepsilon)$ 대신 $[\text{FT}(1/\varepsilon)]^{-1}$을 사용하여 수렴성을 크게 개선합니다:

| 인수분해 방법 | 적용 조건 | COMPASS 설정 |
|---|---|---|
| 나이브(로랑 규칙) | 연속 유전율 프로파일 | `fourier_factorization: "naive"` |
| 리 역규칙 | 불연속 경계 (금속 격자, DTI) | `fourier_factorization: "li_inverse"` |
| 법선 벡터법 | 복잡한 2D 패턴 | `fourier_factorization: "normal_vector"` |

## 수렴성

RCWA의 정확도는 푸리에 차수가 증가함에 따라 개선되지만, 연산 비용도 함께 증가합니다(고유값 문제의 스케일링이 $O(M^3)$). 일반적인 수렴 연구에서는 `fourier_order`를 3에서 25까지 변화시키며 QE 대 차수를 도시합니다:

<RCWAConvergenceDemo />

```yaml
solver:
  convergence:
    auto_converge: true
    order_range: [5, 25]
    qe_tolerance: 0.01
```

1 um 피치 픽셀에 간단한 컬러 필터 패턴이 있는 경우, 차수 9가 일반적으로 충분합니다. 더 미세한 구조(금속 격자, DTI)의 경우 차수 15~21이 필요할 수 있습니다.

## COMPASS의 RCWA 솔버

COMPASS는 세 개의 RCWA 라이브러리를 래핑합니다:

| 솔버 | 라이브러리 | GPU 지원 | 비고 |
|--------|---------|-------------|-------|
| `torcwa` | torcwa | CUDA (PyTorch) | 기본값. GPU 가속 스윕에 최적. |
| `grcwa` | grcwa | CUDA (JAX/PyTorch) | 대안 GPU 백엔드. |
| `meent` | meent | CUDA/CPU | 해석적 고유값 분해 지원. |

세 솔버 모두 동일한 `SolverBase` 인터페이스를 구현하므로, 설정 파일에서 `solver.name`만 변경하면 솔버를 전환할 수 있습니다.
