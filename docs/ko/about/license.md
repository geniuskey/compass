---
title: 라이선스
description: COMPASS 시뮬레이션 플랫폼의 MIT 라이선스입니다.
---

# 라이선스

COMPASS는 MIT 라이선스(MIT License)로 배포됩니다.

```
MIT License

Copyright (c) 2025 COMPASS Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 서드파티 의존성

COMPASS는 각각 고유한 라이선스를 가진 여러 오픈소스 라이브러리에 의존합니다:

| 패키지      | 라이선스     | 용도                          |
|------------|-------------|-------------------------------|
| NumPy      | BSD-3       | 배열 연산                      |
| PyTorch    | BSD-3       | GPU 텐서, 자동 미분            |
| SciPy      | BSD-3       | 과학 연산                      |
| Matplotlib | PSF (BSD)   | 시각화 및 플로팅               |
| Pydantic   | MIT         | 설정 유효성 검증               |
| Hydra      | MIT         | 설정 관리                      |
| OmegaConf  | BSD-3       | YAML 설정 처리                 |
| h5py       | BSD-3       | HDF5 파일 입출력               |
| tqdm       | MIT/MPL-2.0 | 진행률 표시                    |
| PyYAML     | MIT         | YAML 파싱                      |

### 선택적 의존성

| 패키지    | 라이선스  | 용도                          |
|----------|---------|-------------------------------|
| torcwa   | MIT     | RCWA 솔버 (PyTorch)           |
| fdtd     | MIT     | FDTD 솔버 (flaport)           |
| PyVista  | MIT     | 3D 시각화                      |
| Plotly   | MIT     | 대화형 플롯                    |
| pytest   | MIT     | 테스트 프레임워크               |
| ruff     | MIT     | 린팅                           |
| mypy     | MIT     | 정적 타입 검사                  |

## 데이터 및 물질

내장 물질 데이터베이스에는 발표된 문헌에서 파생된 광학 상수가 포함되어 있습니다:

- 실리콘 광학 상수: M.A. Green, "Self-consistent optical parameters of intrinsic silicon at 300 K," Solar Energy Materials and Solar Cells, 2008. 과학 연산을 위한 공정 사용(fair use)에 따라 사용된 데이터입니다.
- SiO2, Si3N4 및 기타 유전체 상수: E.D. Palik, "Handbook of Optical Constants of Solids," Academic Press, 1998.

컬러 필터 염료 스펙트럼은 시뮬레이션 목적의 대표 값이며 특정 상용 제품에 해당하지 않습니다.
