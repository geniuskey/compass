# 설치

## 요구 사항

- Python 3.10 이상
- RCWA 솔버(Solver)에는 CUDA 지원 GPU를 권장하지만 필수는 아닙니다

## 소스에서 설치

저장소를 클론하고 편집 가능 모드로 설치합니다:

```bash
git clone https://github.com/compass-sim/compass.git
cd compass
pip install -e .
```

이 명령은 기본 의존성인 numpy, torch, hydra-core, omegaconf, pydantic, h5py, matplotlib, scipy, pyyaml, tqdm과 함께 코어 패키지를 설치합니다.

## 선택적 의존성

COMPASS는 솔버 및 시각화(Visualization) 백엔드를 선택적 의존성 그룹으로 구성합니다. 필요한 것만 설치하십시오:

```bash
# RCWA solvers (torcwa, grcwa, meent)
pip install -e ".[rcwa]"

# FDTD solver (flaport)
pip install -e ".[fdtd]"

# 3D visualization (plotly, pyvista)
pip install -e ".[viz]"

# Everything
pip install -e ".[all]"

# Development tools (pytest, mypy, ruff)
pip install -e ".[dev]"
```

| 그룹    | 패키지                  | 용도                            |
|---------|-------------------------|---------------------------------|
| `rcwa`  | torcwa                  | RCWA 솔버 (torcwa, grcwa, meent) |
| `fdtd`  | fdtd                    | FDTD 솔버 (flaport 백엔드)      |
| `viz`   | pyvista, plotly          | 인터랙티브 3D 시각화             |
| `all`   | rcwa + fdtd + viz       | 전체 설치                        |
| `dev`   | pytest, pytest-cov, mypy, ruff | 테스트 및 린팅              |

## CUDA 설정

RCWA 솔버는 PyTorch CUDA를 통한 GPU 가속의 효과가 매우 큽니다. NVIDIA GPU가 있는 경우:

1. CUDA 호환 PyTorch 빌드를 설치합니다. 사용 중인 CUDA 버전에 맞는 올바른 명령은 [pytorch.org](https://pytorch.org/get-started/locally/)에서 확인하십시오:

```bash
# Example for CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

2. CUDA 사용 가능 여부를 확인합니다:

```python
import torch
print(torch.cuda.is_available())  # Should print True
print(torch.cuda.get_device_name(0))
```

3. COMPASS는 `compute.backend`가 `"auto"`(기본값)로 설정되어 있을 때 자동으로 GPU를 감지합니다. 특정 장치를 강제로 지정하려면 설정에서 `compute.backend`를 설정하십시오:

```yaml
compute:
  backend: "cuda"   # Force CUDA
  gpu_id: 0         # GPU index
```

### TF32 정밀도

COMPASS는 기본적으로 RCWA 계산에서 **TF32를 비활성화**합니다. TF32는 Ampere 이상 GPU에서 행렬 곱셈의 부동소수점 정밀도를 낮추어 S-행렬(S-matrix) 계산에서 수치 불안정성을 유발할 수 있습니다. 이는 다음과 같이 제어됩니다:

```yaml
solver:
  stability:
    allow_tf32: false  # Default; do not change unless benchmarked
```

## Apple Silicon (MPS)

PyTorch는 M1/M2/M3 Mac에서 Apple Metal Performance Shaders를 지원합니다. COMPASS는 MPS를 컴퓨트 백엔드(Compute Backend)로 사용할 수 있습니다:

```yaml
compute:
  backend: "mps"
```

**제한 사항:**
- 일부 PyTorch 버전에서 복소수 지원이 불완전합니다
- 고유값 분해(Eigendecomposition)가 자동으로 CPU로 폴백될 수 있습니다
- RCWA 워크로드에서 성능은 일반적으로 CUDA보다 느립니다
- MPS 관련 오류가 발생하면 먼저 `compute.backend: "cpu"`로 테스트하십시오

## CPU 폴백

모든 솔버는 GPU 의존성 없이 CPU에서 작동합니다:

```yaml
compute:
  backend: "cpu"
  num_workers: 4
```

CPU 모드는 느리지만 완전히 기능합니다. 디버깅, 소형 피치 픽셀, CI/CD 파이프라인에 유용합니다.

## 환경 변수

| 변수                    | 설명                                         | 기본값    |
|-----------------------|----------------------------------------------|---------|
| `COMPASS_MATERIALS`   | 사용자 정의 재료 디렉토리 경로                   | `./materials/` |
| `COMPASS_OUTPUT_DIR`  | 결과의 기본 출력 디렉토리                        | `./outputs/` |
| `CUDA_VISIBLE_DEVICES`| 표준 PyTorch GPU 선택                          | 모든 GPU  |

## 설치 확인

테스트 스위트를 실행하여 모든 것이 정상적으로 작동하는지 확인합니다:

```bash
# Run all tests
PYTHONPATH=. python3.11 -m pytest tests/ -v

# Run only unit tests (no GPU required)
PYTHONPATH=. python3.11 -m pytest tests/unit/ -v

# Skip slow benchmarks
PYTHONPATH=. python3.11 -m pytest tests/ -v -m "not slow"
```

간단한 Python 확인:

```python
from compass.core.config_schema import CompassConfig
from compass.materials.database import MaterialDB
from compass.solvers.base import SolverFactory

# Config loads with defaults
config = CompassConfig()
print(f"Default solver: {config.solver.name}")

# Material database loads built-in materials
mat_db = MaterialDB()
print(f"Available materials: {mat_db.list_materials()}")

# Check which solvers are importable
print(f"Registered solvers: {SolverFactory.list_solvers()}")
```

솔버 패키지에 대해 `ModuleNotFoundError`가 표시되면 해당하는 선택적 의존성 그룹을 설치하십시오(위 표 참조).

## 시스템 아키텍처 개요

아래 다이어그램은 코어, 지오메트리, 재료, 솔버, 분석 컴포넌트가 어떻게 연결되는지를 보여주는 COMPASS 모듈 아키텍처 전체 구조입니다:

<ModuleArchitectureDiagram />
