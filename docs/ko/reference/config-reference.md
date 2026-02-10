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
| `thickness` | float | `0.6` | CFA 두께 (um) |
| `pattern` | str | `"bayer_rggb"` | 필터 패턴 |
| `materials` | dict | `{"R":"cf_red","G":"cf_green","B":"cf_blue"}` | 색상-재료 매핑 |
| `grid.enabled` | bool | `true` | 금속 그리드 활성화 |
| `grid.width` | float | `0.05` | 그리드 선 폭 (um) |
| `grid.height` | float | `0.6` | 그리드 높이 (um) |
| `grid.material` | str | `"tungsten"` | 그리드 재료 |

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
| `dti.width` | float | `0.1` | 트렌치 폭 (um) |
| `dti.depth` | float | `3.0` | 트렌치 깊이 (um) |
| `dti.material` | str | `"sio2"` | 충전 재료 |

## solver: SolverConfig

```yaml
solver:
  name: torcwa
  type: rcwa
  params:
    fourier_order: [9, 9]
    dtype: "complex64"
  convergence: ...
  stability: ...
```

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `name` | str | `"torcwa"` | 솔버 백엔드 이름 |
| `type` | str | `"rcwa"` | `"rcwa"` 또는 `"fdtd"` |
| `params` | dict | `{"fourier_order": [9,9]}` | 솔버 고유 매개변수 |

### solver.convergence: ConvergenceConfig

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `auto_converge` | bool | `false` | 자동 푸리에 차수 스윕 |
| `order_range` | [int, int] | `[5, 25]` | 최소/최대 푸리에 차수 |
| `qe_tolerance` | float | `0.01` | 수렴 임계값 |
| `spacing_range` | [float, float] or null | `null` | FDTD용 그리드 간격 범위 |

### solver.stability: StabilityConfig

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `precision_strategy` | str | `"mixed"` | `"float32"`, `"float64"`, `"mixed"`, `"adaptive"` |
| `allow_tf32` | bool | `false` | Ampere+ GPU에서 TF32 허용 |
| `eigendecomp_device` | str | `"cpu"` | `"cpu"` 또는 `"gpu"` |
| `fourier_factorization` | str | `"li_inverse"` | `"naive"`, `"li_inverse"`, `"normal_vector"` |
| `energy_check.enabled` | bool | `true` | 에너지 밸런스 검사 활성화 |
| `energy_check.tolerance` | float | `0.02` | 최대 허용 |R+T+A-1| |
| `energy_check.auto_retry_float64` | bool | `true` | 실패 시 float64로 자동 재시도 |
| `eigenvalue_broadening` | float | `1e-10` | 축퇴 감지 임계값 |
| `condition_number_warning` | float | `1e12` | 병조건 행렬 경고 기준 |

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
    grcwa.yaml
    meent.yaml
    fdtd_flaport.yaml
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
```

명령줄에서 모든 매개변수를 오버라이드할 수 있습니다:

```bash
python scripts/run_simulation.py \
    pixel.pitch=0.8 \
    solver.params.fourier_order=[11,11] \
    source.wavelength.mode=sweep
```
