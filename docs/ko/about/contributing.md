---
title: 기여 가이드
description: 개발 환경 설정, 솔버 및 물질 추가, 코드 스타일, 테스트 요구사항 등 COMPASS에 기여하는 방법입니다.
---

# 기여 가이드

COMPASS에 기여해 주셔서 감사합니다. 이 가이드에서는 개발 환경 설정 방법, 기여 워크플로, 특정 유형의 기여에 대한 지침을 다룹니다.

## 개발 환경 설정

### 사전 요구사항

- Python 3.10 이상
- Git
- (선택) GPU 가속 솔버를 위한 CUDA 지원 NVIDIA GPU

### 클론 및 설치

```bash
git clone https://github.com/compass-team/compass.git
cd compass

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in development mode with all optional dependencies
pip install -e ".[all,dev]"
```

### 설치 확인

```bash
# Run the test suite
PYTHONPATH=. python -m pytest tests/ -v

# Run linting
ruff check compass/

# Run type checking
mypy compass/
```

## 코드 스타일

COMPASS는 다음 규칙을 따릅니다:

- 모든 공개 함수 및 메서드에 타입 힌트를 적용한 **Python 3.10+**
- 설정 유효성 검증을 위한 **Pydantic**
- 모든 공개 클래스 및 함수에 **Google 스타일 독스트링**
- 린팅을 위한 **ruff** (규칙: E, F, W, I)
- **줄 길이**: 최대 100자
- **임포트**: ruff에 의한 정렬 (isort 호환)

올바른 스타일의 함수 예시:

```python
def compute_qe(
    result: SimulationResult,
    pixel_stack: PixelStack,
    wavelength: float,
) -> dict[str, float]:
    """Compute quantum efficiency per pixel from simulation result.

    Extracts the absorbed power in each photodiode region and normalizes
    by the incident power to obtain QE.

    Args:
        result: Completed simulation result with field data.
        pixel_stack: Pixel geometry for photodiode region identification.
        wavelength: Wavelength in micrometers.

    Returns:
        Dictionary mapping pixel names (e.g., "R_0_0") to QE values.

    Raises:
        ValueError: If result does not contain absorption data.
    """
    ...
```

## 새 솔버 추가

COMPASS는 솔버에 플러그인 아키텍처(plug-in architecture)를 사용합니다. 새 솔버를 추가하려면 다음을 수행하십시오:

### 1. 솔버 모듈 생성

적절한 디렉터리 아래에 새 파일을 생성합니다:

- RCWA 솔버: `compass/solvers/rcwa/your_solver.py`
- FDTD 솔버: `compass/solvers/fdtd/your_solver.py`

### 2. SolverBase 인터페이스 구현

```python
from compass.solvers.base import SolverBase, SolverFactory
from compass.core.types import SimulationResult
from compass.geometry.pixel_stack import PixelStack
import numpy as np


class YourSolver(SolverBase):
    """Your solver adapter for COMPASS."""

    def setup_geometry(self, pixel_stack: PixelStack) -> None:
        """Convert PixelStack to your solver's geometry format."""
        self._pixel_stack = pixel_stack
        # Convert layers, materials, and geometry to your solver's format
        ...

    def setup_source(self, source_config: dict) -> None:
        """Configure the excitation source."""
        self._source_config = source_config
        # Set wavelength, angle, polarization
        ...

    def run(self) -> SimulationResult:
        """Execute the simulation."""
        # Run your solver
        # Extract R, T, A, QE per pixel, field data
        # Return standardized SimulationResult
        ...

    def get_field_distribution(
        self, component: str, plane: str, position: float
    ) -> np.ndarray:
        """Extract a 2D field slice."""
        ...


# Register with the factory
SolverFactory.register("your_solver", YourSolver)
```

### 3. 임포트 맵에 추가

`compass/solvers/base.py`에서 솔버를 `_try_import` 맵에 추가합니다:

```python
import_map = {
    ...
    "your_solver": "compass.solvers.rcwa.your_solver",  # or fdtd
}
```

### 4. 테스트 추가

최소한 다음을 포함하는 `tests/test_your_solver.py`를 생성합니다:

- 솔버 생성 테스트
- 기하학 설정 테스트
- 단일 파장 실행 테스트
- 에너지 수지 검증 테스트

## 새 물질 추가

### 내장 물질 (CSV)

`materials/` 디렉터리 아래에 `wavelength_um`, `n`, `k` 열을 가진 CSV 파일을 추가합니다:

```csv
wavelength_um,n,k
0.380,1.520,0.000
0.400,1.518,0.000
0.420,1.516,0.000
...
```

그런 다음 물질 데이터베이스 설정에 등록합니다.

### 해석적 분산 모델

셀마이어 또는 코시 방정식으로 정의된 물질의 경우, 물질 데이터베이스 YAML에 계수를 추가합니다:

```yaml
your_material:
  model: "sellmeier"
  coefficients:
    B: [1.0396, 0.2318, 1.0105]
    C: [0.00600, 0.02002, 103.56]
```

## 테스트

### 테스트 실행

```bash
# All tests
PYTHONPATH=. python -m pytest tests/ -v

# Specific test file
PYTHONPATH=. python -m pytest tests/test_geometry.py -v

# Skip slow tests (FDTD, large sweeps)
PYTHONPATH=. python -m pytest tests/ -v -m "not slow"

# With coverage
PYTHONPATH=. python -m pytest tests/ --cov=compass --cov-report=html
```

### 테스트 작성

- `tests/` 디렉터리에 `test_<module>.py` 명명 규칙으로 테스트를 배치합니다
- 공유 설정(설정, 물질 데이터베이스, 픽셀 스택)에 pytest 픽스처를 사용합니다
- 느린 테스트는 `@pytest.mark.slow`로 표시합니다
- 물리적 제약을 테스트합니다: QE가 [0, 1] 범위, 에너지 수지 R + T + A = 1

테스트 예시:

```python
import numpy as np
import pytest
from compass.solvers.base import SolverFactory


def test_solver_energy_balance(pixel_stack, solver_config):
    """Verify that R + T + A = 1 within tolerance."""
    solver = SolverFactory.create("torcwa", solver_config)
    solver.setup_geometry(pixel_stack)
    solver.setup_source({"wavelength": 0.55, "theta": 0.0,
                         "phi": 0.0, "polarization": "unpolarized"})
    result = solver.run()

    total = result.reflection + result.transmission + result.absorption
    assert np.allclose(total, 1.0, atol=0.02), (
        f"Energy not conserved: R+T+A = {total}"
    )
```

## 변경사항 제출

1. 저장소를 포크(fork)하고 `main`에서 기능 브랜치를 생성합니다
2. 위의 코드 스타일 지침에 따라 변경합니다
3. 변경 사항을 커버하는 테스트를 추가하거나 업데이트합니다
4. 전체 테스트 스위트와 린터를 실행하여 아무것도 손상되지 않았는지 확인합니다
5. 변경 내용을 설명하는 명확한 커밋 메시지를 작성합니다
6. 변경이 무엇을 하고 왜 하는지에 대한 설명과 함께 풀 리퀘스트를 엽니다

### 커밋 메시지 형식

간결한 명령형 제목줄을 사용합니다:

```
Add meep FDTD solver adapter

Implement SolverBase interface for MIT Meep, supporting:
- 3D simulation with periodic/PML boundaries
- Broadband source with DFT monitors
- Field extraction on arbitrary planes
```

## 도움 받기

- 버그 리포트나 기능 요청은 이슈를 등록하십시오
- 사용법이나 설계 결정에 대한 질문은 토론(discussions)을 이용하십시오
- 이슈에 적절한 레이블을 태그하십시오: `bug`, `enhancement`, `solver`, `material`, `docs`
