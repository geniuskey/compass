---
title: RCWA 상세 설명
description: 주기적 이미지 센서 픽셀을 위한 RCWA/FMM을 푸리에 고조파, 층 고유모드, S 행렬 재귀, 푸리에 인수분해, 수렴, COMPASS 설정 관점에서 설명합니다.
---

# RCWA 상세 설명

::: tip 선수 지식
[전자기파](/ko/theory/optics/electromagnetic-waves) -> [회절](/ko/theory/optics/diffraction) -> 이 페이지.
RCWA가 처음이라면 먼저 [솔버 선택 가이드](/ko/guide/choosing-solver)를 확인하세요.
:::

RCWA(Rigorous Coupled-Wave Analysis, 엄밀 결합파 해석)는 FMM(Fourier Modal Method)이라고도 부릅니다. 가로 방향으로는 주기적이고, 세로 방향으로는 층상 구조인 물체에 대해 맥스웰 방정식을 주파수 영역에서 풉니다. 이미지 센서 픽셀은 이 조건과 잘 맞습니다. 2x2 Bayer 또는 Quad Bayer 단위 셀이 x-y 방향으로 반복되고, z 방향으로 공기, 마이크로렌즈, 평탄화막, 컬러 필터, BARL, 실리콘이 쌓이기 때문입니다.

실무적으로는 다음 흐름입니다.

1. 각 z 층의 x-y 패턴을 푸리에 고조파로 표현합니다.
2. 그 층 안에서 결합된 전자기 모드를 풉니다.
3. 모든 층을 안정적인 산란 행렬(S-matrix)로 연결합니다.
4. 반사, 투과, 흡수를 QE와 crosstalk 지표로 변환합니다.

RCWA는 광선 추적도 아니고 스칼라 회절 근사도 아닙니다. 벡터 전자기장, 재료 손실, 사입사, 편광, 소멸파, 근접장 간섭을 모두 포함합니다. 대신 정확도는 푸리에 차수와 불연속 경계의 인수분해 방식에 크게 의존합니다.

## RCWA가 푸는 문제

RCWA는 한 파장에서 시간 조화장을 가정합니다.

$$\mathbf{E}(\mathbf{r}, t) = \operatorname{Re}\left[\mathbf{E}(\mathbf{r}) e^{-i\omega t}\right]$$

재료 패턴은 x, y 방향으로 주기적입니다.

$$\varepsilon(x + \Lambda_x, y, z) = \varepsilon(x, y, z), \quad
\varepsilon(x, y + \Lambda_y, z) = \varepsilon(x, y, z)$$

이미지 센서에서 $\Lambda_x$, $\Lambda_y$는 시뮬레이션 도메인 크기입니다.

$$\Lambda_x = \text{pitch} \times \text{unit\_cell.cols}, \quad
\Lambda_y = \text{pitch} \times \text{unit\_cell.rows}$$

각 z 층은 x-y 방향으로 패턴을 가질 수 있지만, 그 얇은 slice 안에서는 z 방향으로 균일하다고 봅니다. 마이크로렌즈나 돌출된 컬러 필터처럼 곡면 또는 테이퍼가 있는 구조는 여러 개의 얇은 계단형 slice로 근사합니다.

::: info BSI 픽셀과 잘 맞는 이유
BSI 픽셀은 본질적으로 주기적 다층 구조입니다. RCWA는 하나의 고립된 픽셀을 흉내 내는 것이 아니라, 반복 단위 셀과 그 사이를 오가는 모든 회절 차수를 함께 풉니다.
:::

## 직관적 모델

RCWA는 좌표계를 바꾸는 방법이라고 생각하면 쉽습니다.

- 실제 공간에서는 픽셀에 재료 경계, DTI 트렌치, 금속 grid, 컬러 필터, 마이크로렌즈가 있습니다.
- 푸리에 공간에서는 이 패턴들이 여러 공간 주파수 성분으로 바뀝니다.
- 맥스웰 방정식은 이 공간 주파수 성분들이 z 방향으로 전파되며 서로 에너지를 주고받는 행렬 방정식이 됩니다.

낮은 푸리에 차수는 큰 형태만 봅니다. 높은 차수는 금속 grid edge, DTI 벽, 컬러 필터 경계처럼 날카로운 특징을 더 잘 분해합니다.

## 픽셀 스택이 RCWA 층으로 바뀌는 방식

COMPASS는 `PixelStack`을 여러 개의 z slice로 변환합니다.

| 물리 구조 | RCWA 표현 | 수렴상 영향 |
|---|---|---|
| 공기, 평탄화막, BARL | 균일 층 | 빠름. 재료 행렬이 대각에 가까움 |
| Bayer 컬러 필터 | x-y 유전율 패턴 | 색 셀 경계를 보려면 충분한 차수가 필요 |
| 금속 grid | 고대비 패턴 층 | Li 인수분해와 높은 차수 필요 |
| 마이크로렌즈 | height map을 계단형 slice로 분할 | z slice 수와 x-y sampling이 중요 |
| 돌출/테이퍼 컬러 필터 | z-aware 컬러 필터 slice | slice 수는 늘지만 SEM에 가까운 형상 가능 |
| DTI가 있는 실리콘 | 패턴화된 실리콘 slice | 강한 굴절률 대비. 에너지 보존 확인 필요 |

RCWA backend가 실제로 보는 것은 z slice별 2D 유전율 배열입니다. 따라서 solver 설정만큼이나 geometry 확인이 중요합니다.

## 단계별 알고리즘

### 1. 회절 차수 선택

2D 주기 구조의 역격자 벡터는 다음과 같습니다.

$$G_{px} = \frac{2\pi p}{\Lambda_x}, \quad
G_{qy} = \frac{2\pi q}{\Lambda_y}$$

회절 차수 $(p,q)$의 면내 파수는 다음입니다.

$$k_{x,p} = k_{x,0} + G_{px}, \quad
k_{y,q} = k_{y,0} + G_{qy}$$

$(k_{x,0}, k_{y,0})$는 입사각과 방위각으로 정해집니다. COMPASS는 무한한 차수를 아래 범위로 잘라냅니다.

$$-N_x \le p \le N_x, \quad -N_y \le q \le N_y$$

따라서 고조파 수는 다음입니다.

$$M = (2N_x + 1)(2N_y + 1)$$

예시:

```yaml
solver:
  name: torcwa
  params:
    fourier_order: [9, 9]   # 19 x 19 = 361 harmonics
```

차수를 올리면 정확도는 좋아지지만 비용도 급격히 증가합니다. 조밀한 고유값 문제와 행렬 연산이 $M$에 대해 가파르게 증가하기 때문입니다.

### 2. 재료를 푸리에 전개

하나의 slice 안에서 상대 유전율은 다음처럼 전개됩니다.

$$\varepsilon(x,y) = \sum_{p,q}\hat{\varepsilon}_{pq}
e^{i(G_{px}x + G_{qy}y)}$$

행렬 관점에서 $\varepsilon(x,y)$를 곱하는 연산은 푸리에 공간의 convolution matrix가 됩니다. 이것이 RCWA의 핵심입니다. 실제 공간의 곱셈이 푸리에 공간의 convolution으로 바뀝니다.

부드러운 층에서는 푸리에 계수가 빠르게 감소합니다. 하지만 tungsten grid와 polymer color filter처럼 불연속 경계가 있으면 계수가 천천히 감소하고 Gibbs-like ringing이 생깁니다. 그래서 이미지 센서의 고대비 구조가 단순 박막보다 어렵습니다.

### 3. 층 고유모드 계산

푸리에 전개 후 맥스웰 curl 방정식은 z 방향의 결합 1차 시스템이 됩니다. 한 가지 표현은 다음과 같습니다.

$$\frac{d}{dz}
\begin{bmatrix}
\mathbf{s}_x \\
\mathbf{s}_y
\end{bmatrix}
= i k_0
\mathbf{A}
\begin{bmatrix}
\mathbf{s}_x \\
\mathbf{s}_y
\end{bmatrix}$$

$\mathbf{s}_x$, $\mathbf{s}_y$는 접선 전기장 성분의 푸리에 계수를 모은 벡터입니다. 이 시스템의 고유모드는 다음을 풉니다.

$$\mathbf{A}\mathbf{v}_m = \gamma_m \mathbf{v}_m$$

고유값 $\gamma_m$는 z 방향 전파 상수이고, 고유벡터는 푸리에 공간의 전기장 분포입니다. 균일 층에서는 서로 독립인 평면파가 되지만, 패턴 층에서는 여러 회절 차수가 섞인 모드가 됩니다.

### 4. 경계 조건 적용

모든 층 계면에서 맥스웰 경계 조건은 접선 성분의 연속성을 요구합니다.

$$E_x, E_y, H_x, H_y \quad \text{continuous across the interface}$$

RCWA는 잘라낸 푸리에 basis 안에서 이 조건을 강제합니다. 그 결과 인접한 두 층의 forward/backward 모드 진폭 사이 관계가 생깁니다.

### 5. S 행렬로 층 연결

전달 행렬을 단순히 곱하면 소멸파의 지수 증가/감소 때문에 수치적으로 불안정해질 수 있습니다. 안정적인 RCWA 구현은 scattering matrix를 cascade합니다. 이렇게 하면 아주 큰 수와 아주 작은 수가 같은 행렬 곱 안에서 섞이는 문제를 줄일 수 있습니다.

두 블록 $A$, $B$를 합친 산란 행렬은 Redheffer star product로 씁니다.

$$S^{AB} = S^A \star S^B$$

개념적으로 S 행렬은 들어오는 파동을 나가는 파동으로 변환합니다.

$$
\begin{bmatrix}
\mathbf{b}_\text{top} \\
\mathbf{b}_\text{bottom}
\end{bmatrix}
=
\begin{bmatrix}
S_{11} & S_{12} \\
S_{21} & S_{22}
\end{bmatrix}
\begin{bmatrix}
\mathbf{a}_\text{top} \\
\mathbf{a}_\text{bottom}
\end{bmatrix}
$$

위에서 조명되는 픽셀 스택에서는 $\mathbf{a}_\text{top}$이 입사 평면파이고, 아래쪽에서 들어오는 파가 없으므로 $\mathbf{a}_\text{bottom}=0$입니다.

### 6. 전력과 흡수 계산

전역 S 행렬을 구하면 반사 및 투과 회절 차수를 얻습니다. 전파 가능한 차수 $m$에 대해:

$$R_m \propto \operatorname{Re}(k_{z,m}^{r}) |r_m|^2, \quad
T_m \propto \operatorname{Re}(k_{z,m}^{t}) |t_m|^2$$

흡수는 에너지 보존 또는 손실 영역의 field integration으로 구합니다.

$$A = 1 - R - T$$

이미지 센서에서는 COMPASS가 실리콘/포토다이오드 영역의 흡수를 픽셀별 QE 또는 crosstalk 지표로 매핑합니다.

## 푸리에 인수분해

푸리에 인수분해는 RCWA에서 가장 중요한 세부사항 중 하나입니다. $\varepsilon E$처럼 불연속 함수의 곱을 다룰 때, 각 항을 푸리에 변환한 뒤 잘라낸 급수를 곱하는 것은 곱 자체를 푸리에 변환해 잘라내는 것과 같지 않습니다. 이 차이가 특히 금속 또는 고굴절률 경계에서 TM-like field의 느리거나 잘못된 수렴을 만듭니다.

Li의 인수분해 규칙은 다음을 구분합니다.

- **직접/로랑 규칙**: $\varepsilon$의 푸리에 행렬을 사용합니다.
- **역규칙**: $1/\varepsilon$의 푸리에 행렬을 만든 뒤 역행렬을 사용합니다.
- **법선 벡터 방법**: 불연속면에 수직/접선인 field 성분을 나눠 처리합니다.

COMPASS 설정:

```yaml
solver:
  stability:
    fourier_factorization: "li_inverse"  # 센서 픽셀의 권장 기본값
```

naive 규칙은 매끄럽거나 저대비 패턴에서만 사용하세요. 금속 grid, DTI, 컬러 필터 경계는 보통 inverse 또는 normal-vector 처리가 필요합니다.

## 수렴 workflow

RCWA 수렴은 추정하지 말고 측정해야 합니다.

<RCWAConvergenceDemo />

권장 순서:

1. `[5, 5]`처럼 작은 차수에서 시작합니다.
2. `[9, 9]`, `[13, 13]`, `[17, 17]`로 올립니다.
3. 평균 QE, 색별 QE, crosstalk, energy balance를 같이 봅니다.
4. 수렴이 단조인지, 진동하는지 확인합니다.
5. 형상 자체가 거칠면 x-y sampling을 올립니다.
6. 마이크로렌즈 또는 테이퍼 컬러 필터가 있으면 z slice 수를 늘립니다.

```bash
# 푸리에 차수를 스윕하며 QE가 포화되는지 확인:
PYTHONPATH=. python scripts/convergence_study.py --sweep fourier_order_torcwa
```

### 차수를 더 요구하는 구조

| 구조 | 어려운 이유 |
|---|---|
| 좁은 금속 grid | 고대비와 sharp corner |
| DTI trench | 실리콘 내부의 큰 굴절률 대비 |
| 매우 작은 pitch | 파장 대비 더 많은 구조 |
| 큰 CRA | 비대칭 회절 차수 증가 |
| blue wavelength | 짧은 파장으로 공간 detail 증가 |
| 테이퍼 컬러 필터 relief | z slicing과 푸리에 해상도를 동시에 요구 |

단순한 1 um Bayer 픽셀은 차수 9로 정성 경향을 볼 수 있습니다. 하지만 metal grid와 DTI가 있는 sign-off 성격의 결과는 차수 15-25가 더 현실적입니다. 정답은 “그 차수에서 metric이 더 이상 움직이지 않는가”로 결정됩니다.

## 흔한 실패 모드

### 에너지가 보존되지 않음

$R + T + A$가 1에서 크게 벗어나면 다음을 확인하세요.

- 재료 손실 부호와 파장 단위.
- 푸리에 인수분해 설정.
- 중요한 회절 차수를 너무 많이 잘라내지 않았는지.
- zero-thickness 또는 중복 층이 stack에 들어갔는지.

### 결과가 차수에 따라 크게 변함

대부분 버그가 아닙니다. 푸리에 basis가 부족하거나 실제 공간 유전율 grid가 거친 것입니다.

### 금속 grid 수렴이 느림

금속은 고대비와 손실을 동시에 가집니다. inverse factorization을 사용하고, 물리적으로 타당하다면 grid corner round를 반영하며, 파장별 수렴 sweep을 확인하세요.

### 마이크로렌즈가 blocky하게 보임

RCWA는 곡면을 staircase로 봅니다. `n_lens_slices`를 늘리거나 microlens height map을 확인하세요.

## RCWA, TMM, FDTD 비교

| 방법 | 강점 | 약점 |
|---|---|---|
| TMM | 1D 박막, BARL 직관 | lateral diffraction과 crosstalk 없음 |
| RCWA | 주기적 layered pixel, spectrum, parameter sweep | 비주기 finite layout, 매우 날카로운 3D 구조 |
| FDTD | 일반 time-domain field와 finite feature | fine grid 비용과 긴 수렴 run |

주기적 unit-cell 이미지 센서 광학에서는 RCWA를 기본값으로 두는 것이 좋습니다. 주기성이 깨지거나, 시간 영역 현상이 중요하거나, z-slice 주기층으로 잘 표현되지 않는 구조라면 FDTD를 사용하세요.

## COMPASS RCWA 솔버

COMPASS는 여러 RCWA/FMM 계열 backend를 래핑합니다.

| Solver | Library | GPU support | Notes |
|---|---|---|---|
| `torcwa` | torcwa | CUDA (PyTorch) | GPU 가속 sweep의 기본 선택. |
| `grcwa` | grcwa | CUDA/JAX depending install | cross-check용 backend. |
| `meent` | meent | CPU/CUDA depending install | 대안 RCWA 구현. |
| `fmmax` | fmmax | JAX accelerators | 선택 가능한 formulation을 가진 vector FMM. |

모두 같은 `SolverBase` interface를 따르므로, 첫 cross-check는 보통 `solver.name`만 바꾸면 됩니다.

## 이미지 센서용 실무 설정

```yaml
solver:
  name: torcwa
  type: rcwa
  params:
    fourier_order: [13, 13]
    dtype: complex64
  stability:
    precision_strategy: mixed
    fourier_factorization: li_inverse
    energy_check:
      enabled: true
      tolerance: 0.02
```

최종 비교는 더 높은 차수에서 다시 실행하고, 가능하다면 두 번째 backend로 확인하세요. solver agreement가 물리적 참값을 증명하지는 않지만, disagreement는 geometry, factorization, convergence를 다시 볼 강한 신호입니다.

## 더 읽을거리

- M. G. Moharam and T. K. Gaylord, [Rigorous coupled-wave analysis of planar-grating diffraction](https://opg.optica.org/josa/abstract.cfm?uri=josa-71-7-811), JOSA 71, 811-818 (1981).
- M. G. Moharam et al., [Stable implementation of the rigorous coupled-wave analysis for surface-relief gratings](https://opg.optica.org/josaa/abstract.cfm?URI=josaa-12-5-1077), JOSA A 12, 1077-1086 (1995).
- L. Li, [Use of Fourier series in the analysis of discontinuous periodic structures](https://opg.optica.org/abstract.cfm?uri=josaa-13-9-1870), JOSA A 13, 1870-1876 (1996).
- V. Liu and S. Fan, [S4: a free electromagnetic solver for layered periodic structures](https://www.sciencedirect.com/science/article/pii/S0010465512001658), Computer Physics Communications 183, 2233-2244 (2012).
- Stanford Fan Group, [S4 documentation](https://web.stanford.edu/group/fan/S4), RCWA/FMM reference implementation.
