---
title: License
description: MIT License for the COMPASS simulation platform.
---

# License

COMPASS is released under the MIT License.

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

## Third-party dependencies

COMPASS depends on several open-source libraries, each with their own licenses:

| Package    | License      | Usage                          |
|------------|-------------|--------------------------------|
| NumPy      | BSD-3       | Array computation              |
| PyTorch    | BSD-3       | GPU tensors, autograd          |
| SciPy      | BSD-3       | Scientific computation         |
| Matplotlib | PSF (BSD)   | Plotting and visualization     |
| Pydantic   | MIT         | Configuration validation       |
| Hydra      | MIT         | Configuration management       |
| OmegaConf  | BSD-3       | YAML config handling           |
| h5py       | BSD-3       | HDF5 file I/O                  |
| tqdm       | MIT/MPL-2.0 | Progress bars                  |
| PyYAML     | MIT         | YAML parsing                   |

### Optional dependencies

| Package  | License  | Usage                          |
|----------|---------|--------------------------------|
| torcwa   | MIT     | RCWA solver (PyTorch)          |
| fdtd     | MIT     | FDTD solver (flaport)          |
| PyVista  | MIT     | 3D visualization               |
| Plotly   | MIT     | Interactive plots              |
| pytest   | MIT     | Testing framework              |
| ruff     | MIT     | Linting                        |
| mypy     | MIT     | Static type checking           |

## Data and materials

The built-in material database includes optical constants derived from published literature:

- Silicon optical constants: M.A. Green, "Self-consistent optical parameters of intrinsic silicon at 300 K," Solar Energy Materials and Solar Cells, 2008. Data used under fair use for scientific computation.
- SiO2, Si3N4, and other dielectric constants: E.D. Palik, "Handbook of Optical Constants of Solids," Academic Press, 1998.

Color filter dye spectra are representative values for simulation purposes and do not correspond to any specific commercial product.
