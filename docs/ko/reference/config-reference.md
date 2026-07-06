# 설정 레퍼런스(Config Reference)

COMPASS 설정 스키마(configuration schema)의 전체 레퍼런스입니다. 모든 설정은 `compass.core.config_schema`에 정의된 Pydantic 모델로 검증됩니다.

## 최상위: CompassConfig

```yaml
pixel: ...          # PixelConfig
solver: ...         # SolverConfig
source: ...         # SourceConfig
compute: ...        # ComputeConfig
experiment_name: "default"
output_dir: "./outputs"
seed: 42
```

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `experiment_name` | str | `"default"` | 출력 디렉터리용 실험 식별자 |
| `output_dir` | str | `"./outputs"` | 기본 출력 디렉터리 |
| `seed` | int | `42` | 재현성을 위한 랜덤 시드 |

## pixel: PixelConfig

```yaml
pixel:
  pitch: 1.0
  unit_cell: [2, 2]
  bayer_map: [["R", "G"], ["G", "B"]]
  layers: ...       # LayersConfig
```

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `pitch` | float | `1.0` | 픽셀 피치 (um) |
| `unit_cell` | [int, int] | `[2, 2]` | 유닛 셀 크기 [행, 열] |
| `bayer_map` | list[list[str]] | `[["R","G"],["G","B"]]` | 색상 채널 맵 |

### pixel.layers: LayersConfig

```yaml
layers:
  air: {thickness: 1.0, material: "air"}
  microlens: ...
  planarization: {thickness: 0.3, material: "sio2"}
  color_filter: ...
  barl: ...
  silicon: ...
```

<PixelStackBuilder />

### pixel.layers.microlens: MicrolensConfig

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `enabled` | bool | `true` | 마이크로렌즈 활성화 |
| `height` | float | `0.6` | 렌즈 새그 높이(sag height) (um) |
| `radius_x` | float | `0.48` | x 방향 반축 (um) |
| `radius_y` | float | `0.48` | y 방향 반축 (um) |
| `material` | str | `"polymer_n1p56"` | 렌즈 재료 |
| `profile.type` | str | `"superellipse"` | 프로파일 모델 |
| `profile.n` | float | `2.5` | 사각도 매개변수 |
| `profile.alpha` | float | `1.0` | 곡률 매개변수 |
| `shift.mode` | str | `"auto_cra"` | 시프트 모드: `"none"`, `"manual"`, `"auto_cra"` |
| `shift.cra_deg` | float | `0.0` | 자동 시프트용 CRA (도) |
| `shift.shift_x` | float | `0.0` | 수동 x 시프트 (um) |
| `shift.shift_y` | float | `0.0` | 수동 y 시프트 (um) |
| `gap` | float | `0.0` | 렌즈 간 간격 (um) |

### pixel.layers.color_filter: ColorFilterConfig

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `thickness` | float | `0.6` | legacy flat CFA 두께(um), 색별 두께가 없을 때 사용 |
| `pattern` | str | `"bayer_rggb"` | 필터 패턴 |
| `materials` | dict | `{"R":"cf_red","G":"cf_green","B":"cf_blue"}` | legacy 색상-재료 매핑 |
| `red.material`, `green.material`, `blue.material` | str | `cf_*` | 색별 재료명 |
| `red.thickness`, `green.thickness`, `blue.thickness` | float | `thickness` | 색별 CFA 높이(um) |
| `red.contact_angle`, `green.contact_angle`, `blue.contact_angle` | float | `90.0` | grid 위 사다리꼴 돌출부 sidewall 각도(deg) |
| `grid.enabled` | bool | `true` | 금속 그리드 활성화 |
| `grid.width` | float | `0.05` | 그리드 선 폭 (um) |
| `grid.thickness` | float | `thickness` | 금속 grid 높이(um) |
| `grid.height` | float | `0.6` | `grid.thickness`의 legacy alias |
| `grid.material` | str | `"tungsten"` | 그리드 재료 |
| `grid.corner_radius` | float | `0.0` | 각 CF 셀의 rounded rectangle 모서리 반경 `r`(um). 네 모서리 모두 동일. `0` = 직각, `> 0`이면 CF를 rounded rectangle로 모델링하고 격자는 그 보집합. `(pitch - grid.width) / 2`로 클램프. |
| `n_slices` | int | taper 시 `8` | taper된 컬러 필터 relief를 계단 근사할 z-slice 수 |

### pixel.layers.barl: BarlConfig

```yaml
barl:
  layers:
    - {thickness: 0.010, material: "sio2"}
    - {thickness: 0.025, material: "hfo2"}
```

`{thickness, material}` 쌍의 목록이며, 상단에서 하단 순으로 정렬됩니다.

### pixel.layers.silicon: SiliconConfig

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `thickness` | float | `3.0` | 실리콘 두께 (um) |
| `material` | str | `"silicon"` | 기판 재료 |
| `photodiode.position` | [float, float, float] | `[0, 0, 0.5]` | PD 오프셋 (x, y, z) um |
| `photodiode.size` | [float, float, float] | `[0.7, 0.7, 2.0]` | PD 크기 (dx, dy, dz) um |
| `dti.enabled` | bool | `true` | DTI 활성화 |
| `dti.mode` | str | `"fdti"` | `"fdti"`(전체) 또는 `"bdti"`(후면 부분) |
| `dti.width` | float | `0.1` | 트렌치 개구부 폭 (um) |
| `dti.depth` | float | `3.0` | 트렌치 깊이 (um) |
| `dti.material` | str | `"sio2"` | 코어 충전 재료 |
| `dti.liner.enabled` | bool | `false` | 트렌치 측벽 컨포멀 high-k 라이너 |
| `dti.liner.material` | str | `"al2o3"` | 라이너 재료 |
| `dti.liner.thickness` | float | `0.0` | 라이너 두께 (um) |
| `dti.taper_angle` | float | `90.0` | 기판면 기준 측벽 각도 (90 = 수직) |
| `dti.n_slices` | int | `6` | 테이퍼 트렌치의 계단형 z-슬라이스 수 |
| `surface_texture.enabled` | bool | `false` | NIR 광포획용 후면 역피라미드 어레이 |
| `surface_texture.height` | float | `0.3` | 피라미드 높이 (um) |
| `surface_texture.period` | float or null | `null` | 피라미드 주기 (um); 기본은 픽셀 피치 |
| `surface_texture.fill_material` | str | `"sio2"` | 피트 충전 재료 |
| `surface_texture.n_slices` | int | `8` | 피라미드 계단형 z-슬라이스 수 |

## solver: SolverConfig

```yaml
solver:
  name: torcwa
  type: rcwa
  params:
    fourier_order: [9, 9]
    dtype: "complex64"
  stability: ...
```

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `name` | str | `"torcwa"` | 솔버 백엔드 이름 |
| `type` | str | `"rcwa"` | `"rcwa"`, `"fdtd"`, 또는 `"tmm"` |
| `params` | dict | `{"fourier_order": [9,9]}` | 솔버 고유 매개변수 (아래 참조) |

### solver.params 의미

`params`는 솔버 어댑터에 그대로 전달되므로 각 키의 의미는 솔버마다 다릅니다.
가장 중요한 차이:

- **torcwa / meent / fmmax**: 축별 푸리에 차수 `fourier_order: [m, m]` →
  총 평면파 수 `(2m+1)²`.
- **grcwa**: **총 평면파 수**로 절단합니다. `nG`를 직접 지정하세요(예: `nG: 49`).
  레거시 폴백으로 `fourier_order[0]`도 경고와 함께 허용되지만, 다른 RCWA
  솔버의 같은 숫자와 **동등하지 않습니다**.
- **FDTD 솔버**: `grid_spacing`(um) 또는 `resolution`(pixels/um, meep) 사용.

모든 결과에는 `metadata["qe_method"]`(`field_integration`, `eps_imag_weight`,
`tmm_1d_analytic`)가 기록되므로, 솔버 간 QE 차이가 솔버 정확도 차이인지
후처리 방법 차이인지 구분할 수 있습니다.

### solver.stability: StabilityConfig

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `precision_strategy` | str | `"mixed"` | `"float32"`, `"float64"`, `"mixed"`, `"adaptive"` — 진단(pre-simulation check)에서 사용 |
| `allow_tf32` | bool | `false` | Ampere+ GPU에서 TF32 허용 (RCWA에서는 `false` 유지) |
| `fourier_factorization` | str | `"li_inverse"` | `"naive"`, `"li_inverse"`, `"normal_vector"` |
| `energy_check.enabled` | bool | `true` | 실행 후 R+T+A ≈ 1 검증 |
| `energy_check.tolerance` | float | `0.02` | 최대 허용 \|R+T+A-1\| |
| `energy_check.auto_retry_float64` | bool | `true` | 위반 시 dtype을 승격해 1회 재실행 (complex64→complex128, float32→float64); 재시도는 `metadata["energy_retry_dtype"]`로 표시 |

## source: SourceConfig

```yaml
source:
  type: planewave
  wavelength:
    mode: single
    value: 0.55
  angle:
    theta_deg: 0.0
    phi_deg: 0.0
  polarization: unpolarized
```

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `type` | str | `"planewave"` | `"planewave"` 또는 `"cone_illumination"` |
| `wavelength.mode` | str | `"single"` | `"single"`, `"sweep"`, 또는 `"list"` |
| `wavelength.value` | float | `0.55` | 단일 파장 (um) |
| `wavelength.sweep.start` | float | `0.38` | 스윕 시작 (um) |
| `wavelength.sweep.stop` | float | `0.78` | 스윕 종료 (um) |
| `wavelength.sweep.step` | float | `0.01` | 스윕 간격 (um) |
| `wavelength.values` | list[float] | null | 명시적 파장 목록 |
| `angle.theta_deg` | float | `0.0` | 극각 (도) |
| `angle.phi_deg` | float | `0.0` | 방위각 (도) |
| `polarization` | str | `"unpolarized"` | `"TE"`, `"TM"`, 또는 `"unpolarized"` |

## compute: ComputeConfig

```yaml
compute:
  backend: auto
  gpu_id: 0
  num_workers: 4
```

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `backend` | str | `"auto"` | `"auto"`, `"cuda"`, `"cpu"`, `"mps"` |
| `gpu_id` | int | `0` | GPU 장치 인덱스 |
| `num_workers` | int | `4` | 병렬 작업용 워커 스레드 수 |

## Hydra 설정 구조

COMPASS는 모듈식 설정을 위해 Hydra를 사용합니다:

```
configs/
  config.yaml           # 기본값이 포함된 메인 설정
  pixel/
    default_bsi_1um.yaml
    default_bsi_0p8um.yaml
  solver/
    torcwa.yaml
    grcwa.yaml            # + grcwa_fast.yaml, grcwa_converged.yaml 프리셋
    meent.yaml
    fmmax.yaml
    fdtd_flaport.yaml
    fdtdz.yaml
    fdtdx.yaml
    meep.yaml
    tmm.yaml
  source/
    planewave.yaml
    wavelength_sweep.yaml
    cone_illumination.yaml
  compute/
    cuda.yaml
    cpu.yaml
    mps.yaml
  experiment/
    solver_comparison.yaml
    qe_benchmark.yaml
    roi_sweep.yaml
    optimize_microlens.yaml
```

명령줄에서 모든 매개변수를 오버라이드할 수 있습니다:

```bash
python scripts/run_simulation.py \
    pixel.pitch=0.8 \
    solver.params.fourier_order=[11,11] \
    source.wavelength.mode=sweep
```
