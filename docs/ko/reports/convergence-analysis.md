---
outline: deep
---

# RCWA/FDTD 수렴 분석 리포트

_생성일: 2026-05-01. 로컬 `outputs/` 벤치마크 산출물에서 생성됨._

이 페이지는 Python 레벨 벤치마크 결과를 GitHub Pages에서 볼 수 있는 리포트 형태로 정리한다. 큰 원본 산출물은 `outputs/`에 두고, 선별된 그림과 표만 `docs/public/reports/convergence/`로 복사한다.

## 현재 판단

- 1D 정합성 사다리는 통과했다. torcwa RCWA는 TMM과 수치 오차 수준으로 맞고, 1D FDTD도 lossy pixel-like multilayer에서 sub-percent 범위다.
- 2D periodic trench는 현재 coarse 설정에서 FDTI/BDTI 모두 R/T/A 차이가 대략 3 percentage point 이하로 맞는다.
- 전체 2x2 pixel scalar FDTD는 아직 최종 물리 결과가 아니라 시각적 수렴 테스트다. 현재 안정적으로 볼 기준점은 44x44x118, 3500 steps이고, 64x64x170 및 128x128x340 결과는 더 긴 물리 시간이 필요하다.

::: warning 고해상도 FDTD 행 해석 주의
`--fdtd-steps`를 고정하면 격자가 촘촘해질수록 실제 물리 시간이 짧아진다. 고해상도 행은 grid 크기만 보지 말고 `c*time`과 energy tail 값을 함께 봐야 한다.
:::

## 재생성 명령

```powershell
uv run python scripts\rcwa_fdtd_alignment.py --structure lossy-multilayer --outdir outputs\rcwa_fdtd_alignment_lossy
uv run python scripts\rcwa_fdtd_trench_benchmark.py --convergence --outdir outputs\rcwa_fdtd_trench_benchmark
uv run python scripts\rcwa_fdtd_pixel_benchmark.py --fdtd-steps 3500 --outdir outputs\rcwa_fdtd_pixel_benchmark_steps3500
uv run python scripts\generate_convergence_report.py
```

고해상도 pixel 체크는 다음 명령으로 다시 만들 수 있다.

```powershell
uv run python scripts\rcwa_fdtd_pixel_benchmark.py --nx 64 --ny 64 --nz 170 --fdtd-steps 5200 --outdir outputs\rcwa_fdtd_pixel_benchmark_64x64x170_steps5200
uv run python scripts\rcwa_fdtd_pixel_benchmark.py --nx 128 --ny 128 --nz 340 --source-set single --fdtd-steps 10400 --outdir outputs\rcwa_fdtd_pixel_benchmark_128x128x340_single_steps10400
uv run python scripts\generate_convergence_report.py
```

## 1D 솔버 정합성 사다리

| 구조 | RCWA 통과 | FDTD 통과 | max \|Rrcwa-Rtmm\| | max \|Rfdtd-Rtmm\| | max \|Tfdtd-Ttmm\| |
| --- | --- | --- | --- | --- | --- |
| lossless_slab | yes | yes | 1.57e-08 | 0.0002 | 0.004 |
| lossless_pixel_like_multilayer | yes | yes | 1.14e-06 | 0.0039 | 0.0026 |
| lossy_pixel_like_multilayer | yes | yes | 7.26e-07 | 0.0026 | 0.0011 |

### 1D slab: TMM vs torcwa RCWA vs 1D FDTD

![01 Slab Spectrum Alignment](/reports/convergence/alignment_slab/01_slab_spectrum_alignment.png)

*01 Slab Spectrum Alignment*

![01 Spectrum Alignment](/reports/convergence/alignment_slab/01_spectrum_alignment.png)

*01 Spectrum Alignment*

![02 Alignment Errors](/reports/convergence/alignment_slab/02_alignment_errors.png)

*02 Alignment Errors*

### 1D lossless multilayer: TMM vs torcwa RCWA vs 1D FDTD

![01 Spectrum Alignment](/reports/convergence/alignment_multilayer/01_spectrum_alignment.png)

*01 Spectrum Alignment*

![02 Alignment Errors](/reports/convergence/alignment_multilayer/02_alignment_errors.png)

*02 Alignment Errors*

![03 Fdtd Grid Convergence](/reports/convergence/alignment_multilayer/03_fdtd_grid_convergence.png)

*03 Fdtd Grid Convergence*

### 1D lossy pixel-like multilayer: TMM vs torcwa RCWA vs 1D FDTD

![01 Spectrum Alignment](/reports/convergence/alignment_lossy_multilayer/01_spectrum_alignment.png)

*01 Spectrum Alignment*

![02 Alignment Errors](/reports/convergence/alignment_lossy_multilayer/02_alignment_errors.png)

*02 Alignment Errors*

![03 Fdtd Grid Convergence](/reports/convergence/alignment_lossy_multilayer/03_fdtd_grid_convergence.png)

*03 Fdtd Grid Convergence*

## 2D FDTI/BDTI periodic trench

FDTI와 BDTI 방향성이 처음으로 실제 오차에 영향을 주는 단계다. 두 솔버가 같은 periodic trench 구조와 같은 R/T/A 정의를 사용한다.

| 모드 | RCWA order | FDTD dx um | max \|dR\| | max \|dT\| | max \|dA\| | field leakage | 정합 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FDTI | 3 | 0.015 | 0.0266 | 0.027 | 0.0278 | 0.1305 | yes |
| BDTI | 3 | 0.015 | 0.0183 | 0.0088 | 0.019 | 0.096 | yes |

### 2D periodic FDTI/BDTI trench: torcwa RCWA vs 2D TE FDTD

![01 Shared Geometry](/reports/convergence/periodic_trench/01_shared_geometry.png)

*01 Shared Geometry*

![02 Spectrum Alignment](/reports/convergence/periodic_trench/02_spectrum_alignment.png)

*02 Spectrum Alignment*

![03 Rta Error](/reports/convergence/periodic_trench/03_rta_error.png)

*03 Rta Error*

![04 Fdtd Field Maps](/reports/convergence/periodic_trench/04_fdtd_field_maps.png)

*04 Fdtd Field Maps*

![05 Convergence](/reports/convergence/periodic_trench/05_convergence.png)

*05 Convergence*

## 전체 2x2 Bayer pixel 수렴

Pixel benchmark는 실제 `PixelStack` 경로를 사용한다. FDTI/BDTI 옵션, 복소 굴절률 재료, BARL, microlens slice, color filter, photodiode integration window가 모두 들어간다. RCWA는 full-supercell R/T/A를, scalar FDTD는 localized-source collection 및 crosstalk proxy를 보고한다.

![Pixel convergence summary](/reports/convergence/pixel_convergence_summary.png)

*사용 가능한 모든 pixel benchmark metric에서 생성한 수렴 요약.*

| Grid | steps | sources | mode | dx um | c*time um | self frac | max neighbor | tail | warnings |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 44x44x118 | 950 | all | FDTI | 0.0455 | 9.5962 | 0.7426 | 0.2003 | 0.278 | 1 |
| 44x44x118 | 950 | all | BDTI | 0.0455 | 9.5962 | 0.7426 | 0.2003 | 0.278 | 1 |
| 44x44x118 | 2200 | all | FDTI | 0.0455 | 22.2228 | 0.6041 | 0.2724 | 0.1077 | 0 |
| 44x44x118 | 2200 | all | BDTI | 0.0455 | 22.2228 | 0.6041 | 0.2728 | 0.1077 | 0 |
| 44x44x118 | 3500 | all | FDTI | 0.0455 | 35.3544 | 0.5831 | 0.2582 | 0.0471 | 0 |
| 44x44x118 | 3500 | all | BDTI | 0.0455 | 35.3544 | 0.5827 | 0.2609 | 0.047 | 0 |
| 64x64x170 | 3500 | all | FDTI | 0.0312 | 24.3794 | 0.5639 | 0.4434 | 0.3352 | 1 |
| 64x64x170 | 3500 | all | BDTI | 0.0312 | 24.3794 | 0.5677 | 0.4377 | 0.3353 | 1 |
| 64x64x170 | 5200 | all | FDTI | 0.0312 | 36.2208 | 0.5155 | 0.4367 | 0.2222 | 1 |
| 64x64x170 | 5200 | all | BDTI | 0.0312 | 36.2208 | 0.5192 | 0.4293 | 0.2223 | 1 |
| 128x128x340 | 10400 | single | FDTI | 0.0156 | 36.2208 | 0.0061 | 0.4945 | 0.22 | 1 |
| 128x128x340 | 10400 | single | BDTI | 0.0156 | 36.2208 | 0.0071 | 0.4933 | 0.2201 | 1 |

### Pixel plots

아래 이미지는 benchmark output 폴더에서 복사한 것이다. geometry와 field slice 그림은 숫자 metric만으로 놓치기 쉬운 방향, indexing, source-placement 문제를 확인하는 데 중요하다.

#### 2x2 pixel scalar FDTD, 44x44x118, 950 steps

![01 Geometry Slices](/reports/convergence/pixel_44x44x118_steps950/01_geometry_slices.png)

*01 Geometry Slices*

![02 Rcwa Rta](/reports/convergence/pixel_44x44x118_steps950/02_rcwa_rta.png)

*02 Rcwa Rta*

![03 Fdtd Crosstalk Matrix](/reports/convergence/pixel_44x44x118_steps950/03_fdtd_crosstalk_matrix.png)

*03 Fdtd Crosstalk Matrix*

![04 Fdtd Field Slices](/reports/convergence/pixel_44x44x118_steps950/04_fdtd_field_slices.png)

*04 Fdtd Field Slices*

#### 2x2 pixel scalar FDTD, 44x44x118, 2200 steps

![01 Geometry Slices](/reports/convergence/pixel_44x44x118_steps2200/01_geometry_slices.png)

*01 Geometry Slices*

![02 Rcwa Rta](/reports/convergence/pixel_44x44x118_steps2200/02_rcwa_rta.png)

*02 Rcwa Rta*

![03 Fdtd Crosstalk Matrix](/reports/convergence/pixel_44x44x118_steps2200/03_fdtd_crosstalk_matrix.png)

*03 Fdtd Crosstalk Matrix*

![04 Fdtd Field Slices](/reports/convergence/pixel_44x44x118_steps2200/04_fdtd_field_slices.png)

*04 Fdtd Field Slices*

#### 2x2 pixel scalar FDTD, 44x44x118, 3500 steps

![01 Geometry Slices](/reports/convergence/pixel_44x44x118_steps3500/01_geometry_slices.png)

*01 Geometry Slices*

![02 Rcwa Rta](/reports/convergence/pixel_44x44x118_steps3500/02_rcwa_rta.png)

*02 Rcwa Rta*

![03 Fdtd Crosstalk Matrix](/reports/convergence/pixel_44x44x118_steps3500/03_fdtd_crosstalk_matrix.png)

*03 Fdtd Crosstalk Matrix*

![04 Fdtd Field Slices](/reports/convergence/pixel_44x44x118_steps3500/04_fdtd_field_slices.png)

*04 Fdtd Field Slices*

#### 2x2 pixel scalar FDTD, 64x64x170, 3500 steps

![01 Geometry Slices](/reports/convergence/pixel_64x64x170_steps3500/01_geometry_slices.png)

*01 Geometry Slices*

![02 Rcwa Rta](/reports/convergence/pixel_64x64x170_steps3500/02_rcwa_rta.png)

*02 Rcwa Rta*

![03 Fdtd Crosstalk Matrix](/reports/convergence/pixel_64x64x170_steps3500/03_fdtd_crosstalk_matrix.png)

*03 Fdtd Crosstalk Matrix*

![04 Fdtd Field Slices](/reports/convergence/pixel_64x64x170_steps3500/04_fdtd_field_slices.png)

*04 Fdtd Field Slices*

#### 2x2 pixel scalar FDTD, 64x64x170, 5200 steps

![01 Geometry Slices](/reports/convergence/pixel_64x64x170_steps5200/01_geometry_slices.png)

*01 Geometry Slices*

![02 Rcwa Rta](/reports/convergence/pixel_64x64x170_steps5200/02_rcwa_rta.png)

*02 Rcwa Rta*

![03 Fdtd Crosstalk Matrix](/reports/convergence/pixel_64x64x170_steps5200/03_fdtd_crosstalk_matrix.png)

*03 Fdtd Crosstalk Matrix*

![04 Fdtd Field Slices](/reports/convergence/pixel_64x64x170_steps5200/04_fdtd_field_slices.png)

*04 Fdtd Field Slices*

#### 2x2 pixel scalar FDTD, 128x128x340, single source, 10400 steps

![01 Geometry Slices](/reports/convergence/pixel_128x128x340_single_steps10400/01_geometry_slices.png)

*01 Geometry Slices*

![02 Rcwa Rta](/reports/convergence/pixel_128x128x340_single_steps10400/02_rcwa_rta.png)

*02 Rcwa Rta*

![03 Fdtd Crosstalk Matrix](/reports/convergence/pixel_128x128x340_single_steps10400/03_fdtd_crosstalk_matrix.png)

*03 Fdtd Crosstalk Matrix*

![04 Fdtd Field Slices](/reports/convergence/pixel_128x128x340_single_steps10400/04_fdtd_field_slices.png)

*04 Fdtd Field Slices*

## 시각적 smoke test

이 그림들은 엄밀한 solver evidence가 아니다. FDTI/BDTI geometry, photodiode window, plotting pipeline을 빠르게 눈으로 확인하기 위한 테스트로 유지한다.

### Visual CMOS pixel smoke test

![01 Geometry Overview](/reports/convergence/visual_cmos_pixel/01_geometry_overview.png)

*01 Geometry Overview*

![02 Rcwa Visual Test](/reports/convergence/visual_cmos_pixel/02_rcwa_visual_test.png)

*02 Rcwa Visual Test*

![03 Fdtd Visual Test](/reports/convergence/visual_cmos_pixel/03_fdtd_visual_test.png)

*03 Fdtd Visual Test*

![04 Rcwa Fdtd Comparison](/reports/convergence/visual_cmos_pixel/04_rcwa_fdtd_comparison.png)

*04 Rcwa Fdtd Comparison*

## 해석 기준

- 전체 pixel을 보기 전에 1D 사다리로 normalization, material loss, monitor 계산을 먼저 검증한다.
- FDTI/BDTI 비교는 periodic trench benchmark에서 같은 geometry와 R/T/A 정의로 먼저 맞춘다.
- full-pixel scalar FDTD는 목표 grid에서 energy tail이 기준 이하로 내려갈 때까지 시각적 수렴 및 crosstalk proxy로 해석한다.
- 최종 고정밀 평가는 FDTD step을 grid refinement에 맞춰 늘리고, 네 source를 모두 돌리며, RCWA도 Fourier order와 permittivity grid sweep을 같이 수행해야 한다.
