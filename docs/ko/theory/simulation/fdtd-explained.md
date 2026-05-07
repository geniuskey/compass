---
title: FDTD 상세 설명
description: 이미지 센서 광학을 위한 FDTD를 Yee 격자, leapfrog update, CFL 안정성, PML 경계, source, monitor, 수렴, COMPASS 설정 관점에서 설명합니다.
---

# FDTD 상세 설명

::: tip 선수 지식
[전자기파](/ko/theory/optics/electromagnetic-waves) -> 이 페이지.
FDTD가 처음이라면 먼저 [솔버 선택 가이드](/ko/guide/choosing-solver)를 확인하세요.
:::

FDTD(Finite-Difference Time-Domain, 유한차분 시간영역법)는 맥스웰 방정식을 공간과 시간에서 직접 풉니다. RCWA처럼 주기 픽셀을 푸리에 고조파로 전개하는 대신, geometry를 voxel grid로 만들고, 시간에 따라 변하는 source를 넣은 뒤, 전기장과 자기장을 한 step씩 전진시킵니다.

이미지 센서에서는 FDTD가 “field를 실제 공간에서 보고 싶을 때” 특히 유용합니다. 마이크로렌즈가 빛을 어떻게 모으는지, 금속 grid에서 어떻게 산란되는지, DTI를 넘어 crosstalk가 어떻게 생기는지, 비주기/finite layout이 어떻게 동작하는지 볼 수 있습니다. 대신 grid는 가장 작은 구조와 가장 높은 굴절률 재료 내부의 최단 파장을 모두 분해해야 합니다.

## FDTD가 푸는 문제

비자성 optical stack에서 시간 영역 curl 방정식은 다음과 같습니다.

$$\frac{\partial \mathbf{H}}{\partial t} = -\frac{1}{\mu_0}\nabla \times \mathbf{E}$$

$$\frac{\partial \mathbf{E}}{\partial t} =
\frac{1}{\varepsilon_0 \varepsilon_r(\mathbf{r})}
\left(\nabla \times \mathbf{H} - \mathbf{J}\right)$$

FDTD는 연속 미분을 직교 grid 위의 유한 차분으로 바꿉니다.

$$x_i = i\Delta x,\quad y_j = j\Delta y,\quad z_k = k\Delta z,\quad t_n = n\Delta t$$

출력은 두 방식으로 해석할 수 있습니다.

- **시간 영역 field**: $\mathbf{E}(t)$, $\mathbf{H}(t)$ snapshot.
- **주파수 영역 관측값**: run 중 누적한 Fourier-transformed field로 구한 반사율, 투과율, 흡수, QE.

::: info 광대역이 공짜라는 뜻은 아닙니다
짧은 pulse는 한 번의 run으로 여러 파장을 포함할 수 있습니다. 하지만 모든 관심 주파수와 resonant tail이 충분히 감쇠할 만큼 긴 물리 시간이 필요합니다. broadband FDTD 결과도 grid convergence와 time-window convergence를 모두 확인해야 합니다.
:::

## 직관적 모델

FDTD는 맥스웰 방정식을 찍는 movie camera처럼 볼 수 있습니다.

1. 픽셀 stack을 3D material array로 변환합니다.
2. source가 pulse 또는 continuous wave를 주입합니다.
3. field가 전파, 산란, 간섭, 흡수됩니다.
4. monitor가 flux와 field data를 수집합니다.
5. source가 지나가고 residual energy가 충분히 줄면 run을 멈춥니다.

RCWA와 달리 FDTD는 모든 층이 lateral periodic일 필요가 없습니다. periodic boundary는 선택입니다. 이 유연성 때문에 FDTD는 RCWA cross-check 또는 어려운 geometry의 reference method로 자주 쓰입니다.

## 픽셀 스택이 FDTD grid로 바뀌는 방식

COMPASS는 `PixelStack`을 voxelized permittivity volume으로 변환합니다.

| 물리 구조 | FDTD 표현 | 주요 위험 |
|---|---|---|
| 공기, 평탄화막, BARL | 균일 voxel 영역 | 얇은 막이 grid에서 사라질 수 있음 |
| 마이크로렌즈 | staircase 또는 smoothed 3D shape | 곡률을 보려면 fine x-y-z resolution 필요 |
| 컬러 필터 | 흡수/분산 voxel 영역 | material loss 오류가 QE에 직접 반영됨 |
| 금속 grid | 고손실 고대비 voxel | skin depth와 sharp corner에 fine grid 필요 |
| DTI/BDTI | silicon/oxide/trench voxel boundary | crosstalk가 boundary 위치에 민감 |
| 포토다이오드 | absorption integration volume | optical absorption과 electrical collection model 구분 필요 |
| 상하 open space | PML absorbing layer | PML이 가까우면 near field를 반사할 수 있음 |

solver가 보는 것은 discretized grid뿐입니다. field map을 물리적 통찰로 해석하기 전에 반드시 voxelized geometry를 확인하세요.

## Yee 격자

FDTD는 보통 Yee lattice를 사용합니다. 전기장과 자기장 성분이 공간과 시간에서 서로 엇갈린 위치에 배치됩니다. 각 curl update는 맥스웰 방정식에 맞는 주변 field sample을 사용합니다.

<YeeCellViewer />

이 엇갈림에는 실무상 중요한 결과가 있습니다.

- field component들이 같은 점에 저장되지 않으므로 energy density와 flux monitor가 field를 보간할 수 있습니다.
- 유전체 경계가 field sample 사이에 놓일 수 있으므로 sharp boundary 주변에서는 staircasing과 subpixel averaging이 중요합니다.

## Leapfrog update

전기장과 자기장은 번갈아 업데이트됩니다.

1. $\mathbf{H}^{n+1/2}$를 $\mathbf{E}^{n}$에서 업데이트합니다.
2. $\mathbf{E}^{n+1}$를 $\mathbf{H}^{n+1/2}$에서 업데이트합니다.
3. 원하는 물리 시간까지 반복합니다.

예를 들어, 비자성/비분산 매질에서 한 전기장 성분은 다음처럼 갱신됩니다.

$$E_x^{n+1}(i,j,k) = E_x^n(i,j,k) +
\frac{\Delta t}{\varepsilon_0 \varepsilon_r(i,j,k)}
\left[
\frac{H_z^{n+1/2}(i,j,k) - H_z^{n+1/2}(i,j-1,k)}{\Delta y}
-
\frac{H_y^{n+1/2}(i,j,k) - H_y^{n+1/2}(i,j,k-1)}{\Delta z}
\right]$$

다른 성분도 같은 curl pattern을 따릅니다. 손실 또는 분산 재료에서는 $\varepsilon(\omega)$와 흡수를 올바르게 표현하기 위한 추가 material-update 항이 들어갑니다.

## 안정성: CFL 한계

시간 간격은 Courant-Friedrichs-Lewy(CFL) 조건을 만족해야 합니다. 3D Cartesian grid에서는:

$$\Delta t \le
\frac{S}{c\sqrt{\frac{1}{\Delta x^2}+\frac{1}{\Delta y^2}+\frac{1}{\Delta z^2}}}$$

$S$는 Courant factor입니다. 공간 grid를 더 작게 만들면 시간 step도 같이 작아집니다. 그래서 3D FDTD grid refinement는 두 번 비싸집니다. voxel 수가 늘고, 같은 물리 시간을 계산하는 데 필요한 time step 수도 늘어납니다.

simulation에서 `NaN`, 폭주하는 field, 비물리적인 energy gain이 나오면 time step, material model, PML을 먼저 확인하세요.

## Grid resolution과 numerical dispersion

grid는 가장 높은 굴절률 재료 안의 최단 파장을 분해해야 합니다.

$$\Delta \le \frac{\lambda_0}{n_\text{max} N_\text{ppw}}$$

$N_\text{ppw}$는 wavelength당 grid point 수입니다. 정성 확인은 15-20 points per wavelength에서 시작할 수 있지만, sign-off 성격의 결과는 더 촘촘한 확인이 필요합니다.

실리콘에서 $\lambda_0 = 400$ nm, $n \approx 4$라면:

$$\Delta \le \frac{0.4\ \mu\text{m}}{4 \times 20} = 5\ \text{nm}$$

이 계산 하나가 이미지 센서 FDTD 비용의 대부분을 설명합니다. blue light는 silicon 내부에서 매우 짧은 파장이 되고, 2x2 Bayer domain을 모든 방향으로 균일하게 refine하면 Yee sample 수가 매우 커질 수 있습니다.

### 해상도가 부족할 때 보이는 증상

| 증상 | 가능성 높은 원인 |
|---|---|
| `dx`를 바꾸면 QE가 크게 변함 | grid dispersion 또는 geometry staircasing |
| metal-grid effect가 약하게 보임 | skin depth 또는 metal edge under-resolution |
| DTI crosstalk가 지나치게 좋아 보임 | trench width 또는 sidewall 위치가 voxel화로 밀림 |
| field hot spot이 resolution에 따라 움직임 | interface interpolation artifact |
| flux balance가 흔들림 | monitor가 너무 가깝거나 grid/material loss가 맞지 않음 |

## Source와 monitor

FDTD 결과는 source와 monitor 배치에 크게 의존합니다.

### Source

| Source type | 쓰는 경우 | 주의점 |
|---|---|---|
| Continuous wave (CW) | 한 파장, steady-state field map | 정상 상태까지 충분히 길게 run |
| Gaussian pulse | broadband spectrum | frequency-domain monitor normalization 필요 |
| Planewave/TFSF | finite scatterer에 평면파 입사 | source box가 scatterer/PML과 겹치면 안 됨 |
| Bloch-periodic planewave | CRA가 있는 periodic pixel array | boundary phase가 incident wavevector와 맞아야 함 |

### Monitor

Flux monitor는 표면을 지나는 Poynting vector를 측정합니다.

$$\mathbf{S} = \frac{1}{2}\operatorname{Re}(\mathbf{E} \times \mathbf{H}^*)$$

spectrum에서는 monitor point마다 field를 Fourier transform한 뒤 flux를 계산합니다. backend가 명확히 문서화하지 않은 이상 time-domain power trace를 직접 Fourier transform해서 spectrum으로 쓰면 안 됩니다.

이미지 센서 QE는 보통 silicon 또는 photodiode 영역의 흡수 적분으로 계산합니다.

$$P_\text{abs}(\omega) =
\frac{1}{2}\omega\varepsilon_0\varepsilon_r''(\omega)
\int_V |\mathbf{E}(\mathbf{r}, \omega)|^2\,dV$$

integration volume도 model의 일부입니다. silicon에서 흡수된 광자 수가 항상 electrical collected charge와 같은 것은 아닙니다. 전기적 수집 영역이 optical absorption 영역보다 작을 수 있습니다.

## 경계 조건

### Periodic/Bloch boundary

반복 단위 셀에서는 lateral boundary를 periodic으로 둘 수 있습니다.

$$\mathbf{E}(x+\Lambda_x,y,z)=\mathbf{E}(x,y,z)$$

사입사에서는 Bloch phase가 필요합니다.

$$\mathbf{E}(x+\Lambda_x,y,z)=\mathbf{E}(x,y,z)e^{ik_x\Lambda_x}$$

RCWA와 같은 무한 periodic pixel array를 비교할 때는 Bloch periodicity를 사용하세요.

### PML absorbing boundary

open boundary는 보통 PML(Perfectly Matched Layer)로 끝냅니다. PML은 물리 영역 바깥에 두는 artificial absorbing layer이며, outgoing wave가 최소 반사로 simulation cell을 떠나게 합니다.

실무 규칙:

- PML을 고굴절률 geometry와 강한 near field에서 떨어뜨립니다.
- grazing incidence, high-Q resonance, evanescent-rich field에서는 PML을 두껍게 합니다.
- PML을 더 멀리 옮겨 reflection이 줄어드는지 확인합니다.
- source나 flux monitor를 PML 안에 두지 않습니다.

## 이미지 센서용 FDTD workflow

1. 단순 1D stack에서 TMM 또는 zero-order RCWA와 먼저 맞춥니다.
2. 주기 픽셀 geometry를 coarse하지만 안정적인 grid로 추가합니다.
3. 비싼 sweep 전에 voxelized geometry를 확인합니다.
4. incident-field reference run으로 normalization합니다.
5. reflection/transmission monitor를 source, scatterer, PML에서 떨어뜨립니다.
6. field가 target threshold 아래로 감쇠할 때까지 run합니다.
7. grid spacing을 sweep합니다. 예: 20 nm, 10 nm, 5 nm.
8. PML thickness와 monitor offset을 sweep합니다.
9. integrated $R + T + A$가 1에 가까운지 봅니다.
10. 그 다음 photodiode QE와 crosstalk를 해석합니다.

## Runtime과 memory scaling

memory footprint는 대략 Yee cell 수에 비례합니다.

$$N_\text{cells}=N_xN_yN_z$$

각 cell은 여러 전기장/자기장 성분, material coefficient, 때로는 DFT monitor accumulator를 저장합니다. runtime은 다음에 비례합니다.

$$\text{work} \propto N_xN_yN_zN_t$$

그리고 grid spacing이 작아질수록 $N_t$도 증가합니다. $\Delta x$, $\Delta y$, $\Delta z$를 절반으로 줄이면 같은 물리 시간 기준 memory는 약 $8\times$, runtime은 $16\times$ 이상 증가할 수 있습니다.

## 흔한 실패 모드

### Field가 폭주함

Courant factor, 음수 또는 불일치 material coefficient, dispersive material 설정, source가 손실/PML 영역과 겹치는지 확인하세요.

### PML reflection이 큼

PML을 pixel stack에서 더 멀리 두고, 두껍게 만들고, monitor 위치는 그대로 둔 채 다시 실행하세요. 큰 CRA, 고굴절률 silicon, guided/evanescent field가 있을 때 특히 중요합니다.

### Broadband spectrum이 noisy함

run time을 늘리거나, source bandwidth를 줄이거나, 더 부드러운 pulse를 쓰거나, frequency-domain convergence 기준을 사용하세요. 짧은 time trace는 좁은 spectral feature를 분해할 수 없습니다.

### RCWA와 FDTD가 맞지 않음

먼저 같은 물리 문제인지 확인하세요. unit cell, material, incident angle, polarization, absorption volume, boundary condition이 모두 같아야 합니다. 그 다음 RCWA order와 FDTD grid를 각각 독립적으로 수렴시킵니다.

## FDTD, RCWA, TMM 비교

| 방법 | 강점 | 약점 |
|---|---|---|
| TMM | 1D thin-film stack | lateral diffraction 없음 |
| RCWA | periodic layered pixel과 wavelength sweep | aperiodic finite feature |
| FDTD | real-space field, finite geometry, broadband check | fine-grid memory/runtime |

일반적인 주기적 BSI 픽셀에서는 RCWA가 보통 가장 빠른 primary solver입니다. FDTD는 validation, finite-layout study, broadband response, field intuition에 사용하세요.

## COMPASS FDTD 솔버

| Solver | Library | GPU support | Notes |
|---|---|---|---|
| `fdtd_flaport` | fdtd (flaport) | CUDA (PyTorch) | prototyping과 quick check용 lightweight backend. |
| `fdtdz` | fdtdz | CUDA/JAX depending install | structured grid용 experimental high-performance workflow. |
| `meep` | Meep | CPU/MPI | 성숙한 reference backend. material/monitor 지원 폭이 넓음. |

이 backend들은 같은 `SolverBase` interface를 공유하지만 feature coverage가 완전히 같지는 않습니다. backend 변경은 drop-in guarantee가 아니라 cross-validation으로 다루세요.

## 이미지 센서용 실무 설정

```yaml
solver:
  name: fdtd_flaport
  type: fdtd
  params:
    grid_spacing: 0.01        # um; coarse에서 시작 후 수렴 확인
    runtime_fs: 300
    courant_factor: 0.5
    pml_thickness: 20         # cells
    source:
      type: gaussian_pulse
      normalize_reference: true
    monitors:
      flux_offset: 0.2        # patterned stack에서 떨어진 거리, um
      dft_fields: true
  convergence:
    grid_spacing_um: [0.02, 0.01, 0.005]
    energy_tolerance: 0.02
```

최종 report에는 grid spacing, time step 또는 Courant factor, physical runtime, PML thickness, monitor location, final energy balance를 포함하세요. 이 정보가 없으면 FDTD 숫자는 재현하기 어렵습니다.

## 더 읽을거리

- K. S. Yee, [Numerical solution of initial boundary value problems involving Maxwell's equations in isotropic media](https://ieeexplore.ieee.org/document/1138693), IEEE Transactions on Antennas and Propagation 14, 302-307 (1966).
- J. P. Berenger, [A perfectly matched layer for the absorption of electromagnetic waves](https://doi.org/10.1006/jcph.1994.1159), Journal of Computational Physics 114, 185-200 (1994).
- Meep documentation, [Introduction](https://meep.readthedocs.io/en/stable/Introduction/), [Perfectly Matched Layers](https://meep.readthedocs.io/en/stable/Perfectly_Matched_Layer/), [Materials](https://meep.readthedocs.io/en/stable/Materials/), [Subpixel Smoothing](https://meep.readthedocs.io/en/stable/Subpixel_Smoothing/).
