---
title: RCWA Explained
description: Rigorous coupled-wave analysis for periodic image-sensor pixels, including Fourier harmonics, layer eigenmodes, S-matrix recursion, factorization, convergence, and practical COMPASS settings.
---

# RCWA Explained

::: tip Prerequisites
[Electromagnetic Waves](/theory/optics/electromagnetic-waves) -> [Diffraction](/theory/optics/diffraction) -> this page.
If RCWA is new to you, start with the [solver selection guide](/guide/choosing-solver).
:::

RCWA (Rigorous Coupled-Wave Analysis), also called the Fourier Modal Method (FMM), solves Maxwell's equations in the frequency domain for structures that are periodic in the lateral directions and layered in the vertical direction. That is exactly the shape of many image-sensor simulation cells: a repeated Bayer or Quad Bayer unit cell in x-y, sliced into air, microlens, planarization, color filter, BARL, and silicon layers along z.

The practical idea is:

1. Represent each x-y layer pattern as Fourier harmonics.
2. Solve the coupled electromagnetic modes inside that layer.
3. Connect all layers with stable scattering matrices.
4. Convert reflected, transmitted, and absorbed power into QE and crosstalk metrics.

RCWA is not a ray model and not a scalar diffraction approximation. It keeps the vector nature of the fields, material loss, oblique incidence, polarization, evanescent orders, and near-field interference. The tradeoff is that convergence depends strongly on Fourier order and on how discontinuous material boundaries are factorized.

## What problem RCWA solves

RCWA assumes a time-harmonic field at one wavelength:

$$\mathbf{E}(\mathbf{r}, t) = \operatorname{Re}\left[\mathbf{E}(\mathbf{r}) e^{-i\omega t}\right]$$

The material pattern is periodic in x and y:

$$\varepsilon(x + \Lambda_x, y, z) = \varepsilon(x, y, z), \quad
\varepsilon(x, y + \Lambda_y, z) = \varepsilon(x, y, z)$$

For image sensors, $\Lambda_x$ and $\Lambda_y$ are the simulation domain dimensions:

$$\Lambda_x = \text{pitch} \times \text{unit\_cell.cols}, \quad
\Lambda_y = \text{pitch} \times \text{unit\_cell.rows}$$

Each z layer is laterally patterned but vertically uniform within that slice. Curved or tapered structures, such as microlenses or protruding color filters, are represented as a staircase of thin slices.

::: info Why this matches BSI pixels
BSI pixels are naturally multilayer periodic structures. RCWA is efficient because it does not simulate one isolated finite sensor pixel; it solves one periodic unit cell and includes all diffraction orders that couple between neighboring periods.
:::

## Mental model

Think of RCWA as a basis change:

- In real space, a pixel has material boundaries, DTI trenches, metal grids, color-filter cells, and microlens shapes.
- In Fourier space, those patterns become a set of spatial frequencies.
- Maxwell's equations become matrix equations that describe how those spatial frequencies exchange energy as light propagates through z.

Low Fourier order sees only the coarse shape. Higher Fourier order resolves sharper features such as metal-grid edges and DTI walls.

## From pixel stack to RCWA layers

COMPASS turns a `PixelStack` into layer slices:

| Physical feature | RCWA representation | Convergence consequence |
|---|---|---|
| Air, planarization, BARL sublayers | Uniform layers | Fast, diagonal material matrices |
| Bayer color filter | Patterned x-y permittivity grid | Needs enough Fourier order for color-cell boundaries |
| Metal grid | High-contrast patterned grid | Needs Li factorization and higher order |
| Microlens | Staircase slices from a height map | Needs enough z slices and x-y sampling |
| Protruding/tapered color filter | z-aware color-filter slices | More slices, but closer to SEM-like geometry |
| Silicon with DTI | Patterned silicon slices | Strong index contrast; check energy balance |

The RCWA backend only sees a stack of 2D permittivity arrays, one per z slice. That is why geometry verification is as important as solver settings.

## Step-by-step algorithm

### 1. Choose diffraction orders

For a 2D periodic structure, the reciprocal lattice vectors are:

$$G_{px} = \frac{2\pi p}{\Lambda_x}, \quad
G_{qy} = \frac{2\pi q}{\Lambda_y}$$

The in-plane wavevector of order $(p,q)$ is:

$$k_{x,p} = k_{x,0} + G_{px}, \quad
k_{y,q} = k_{y,0} + G_{qy}$$

where $(k_{x,0}, k_{y,0})$ is set by the incident angle and azimuth. COMPASS truncates the infinite set to:

$$-N_x \le p \le N_x, \quad -N_y \le q \le N_y$$

so the number of harmonics is:

$$M = (2N_x + 1)(2N_y + 1)$$

Example:

```yaml
solver:
  name: torcwa
  params:
    fourier_order: [9, 9]   # 19 x 19 = 361 harmonics
```

Increasing order improves accuracy but quickly increases cost because dense eigensolves and matrix products scale steeply with $M$.

### 2. Fourier-expand the material

Within a slice, the relative permittivity is expanded as:

$$\varepsilon(x,y) = \sum_{p,q}\hat{\varepsilon}_{pq}
e^{i(G_{px}x + G_{qy}y)}$$

In matrix form, multiplication by $\varepsilon(x,y)$ becomes a convolution matrix. This is the core RCWA operation: real-space multiplication becomes Fourier-space convolution.

For smooth layers, the Fourier coefficients decay quickly. For discontinuous boundaries, such as tungsten grid to polymer color filter, Fourier coefficients decay slowly and Gibbs-like ringing appears. This is why high-contrast sensor structures are harder than smooth dielectric films.

### 3. Solve the layer eigenmodes

After Fourier expansion, Maxwell's curl equations become a coupled first-order system along z. A common form is:

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
\end{bmatrix}
$$

where $\mathbf{s}_x$ and $\mathbf{s}_y$ collect tangential field Fourier coefficients. Solving this system gives eigenmodes:

$$\mathbf{A}\mathbf{v}_m = \gamma_m \mathbf{v}_m$$

Each eigenvalue $\gamma_m$ is a z propagation constant. Each eigenvector is a Fourier-space field profile. In a uniform layer, these modes reduce to independent plane waves; in a patterned layer, the modes are mixtures of many diffraction orders.

### 4. Match boundaries

At every layer interface, Maxwell boundary conditions require continuity of tangential fields:

$$E_x, E_y, H_x, H_y \quad \text{continuous across the interface}$$

RCWA enforces these conditions in the truncated Fourier basis. The result is a relation between forward and backward modal amplitudes in adjacent layers.

### 5. Cascade layers with an S-matrix

Naively multiplying transfer matrices can become unstable because evanescent modes include exponentially growing and decaying factors. Stable RCWA implementations cascade scattering matrices instead. This avoids carrying very large and very small numbers in the same matrix product.

For two adjacent blocks $A$ and $B$, the combined scattering matrix is written with the Redheffer star product:

$$S^{AB} = S^A \star S^B$$

Conceptually, the S-matrix maps incoming waves to outgoing waves:

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

For a normally illuminated pixel stack, $\mathbf{a}_\text{top}$ contains the incident plane wave and $\mathbf{a}_\text{bottom}=0$.

### 6. Compute power and absorption

After solving the global S-matrix, RCWA obtains reflected and transmitted diffraction orders. For propagating order $m$:

$$R_m \propto \operatorname{Re}(k_{z,m}^{r}) |r_m|^2, \quad
T_m \propto \operatorname{Re}(k_{z,m}^{t}) |t_m|^2$$

Absorption follows from energy conservation or from field integration in lossy regions:

$$A = 1 - R - T$$

For image sensors, COMPASS maps absorption into silicon/photodiode regions and reports per-pixel QE or crosstalk metrics.

## Fourier factorization

Fourier factorization is one of the most important RCWA details. If a product such as $\varepsilon E$ is discontinuous, taking the Fourier transform of each factor and multiplying truncated series is not equivalent to truncating the Fourier transform of the product. This causes slow or wrong convergence, especially for TM-like fields at metal or high-index boundaries.

Li's factorization rules explain when to use:

- **Direct/Laurent rule**: use the Fourier matrix of $\varepsilon$.
- **Inverse rule**: use the inverse of the Fourier matrix of $1/\varepsilon$.
- **Normal-vector methods**: decompose fields into components normal/tangential to discontinuity surfaces.

In COMPASS:

```yaml
solver:
  stability:
    fourier_factorization: "li_inverse"  # recommended default for sensor pixels
```

Use the naive rule only for smooth or low-contrast patterns. Metal grids, DTI, and color-filter boundaries generally need inverse or normal-vector treatment.

## Convergence workflow

RCWA convergence should be measured, not assumed.

<RCWAConvergenceDemo />

Recommended workflow:

1. Start with a small order, such as `[5, 5]`.
2. Sweep to `[9, 9]`, `[13, 13]`, `[17, 17]`, and higher if needed.
3. Watch target metrics: average QE, per-color QE, crosstalk, and energy balance.
4. Check whether convergence is monotonic or oscillatory.
5. Increase x-y sampling if the geometry itself is under-resolved.
6. Increase z slices for curved microlenses or tapered color filters.

```bash
# Sweep the Fourier order and watch QE saturate:
PYTHONPATH=. python scripts/convergence_study.py --sweep fourier_order_torcwa
```

### What usually needs more order

| Feature | Why it is hard |
|---|---|
| Narrow metal grid | High contrast and sharp corners |
| DTI trenches | High index contrast in silicon |
| Very small pitch | More features per wavelength |
| Oblique CRA | More asymmetric diffraction orders |
| Blue wavelength | Shorter wavelength, more spatial detail |
| Tapered color filter relief | Needs z slicing plus Fourier resolution |

For a simple 1 um Bayer pixel, order 9 may be enough for qualitative trends. For sign-off-style results with metal grid and DTI, order 15-25 is more realistic. The right number is the one where your metric stops moving.

## Common failure modes

### Energy is not conserved

If $R + T + A$ is far from 1, check:

- Material loss signs and wavelength units.
- Fourier factorization setting.
- Whether too few harmonics truncate important diffracted orders.
- Whether the layer stack contains zero-thickness or duplicated layers.

### Results change strongly with order

This is usually not a bug. It means the Fourier basis is not large enough, or the real-space permittivity grid is too coarse.

### Metallic grids converge slowly

Metals combine high contrast and loss. Use inverse factorization, rounded grid corners when physically justified, and convergence sweeps by wavelength.

### Microlens shape looks too blocky

RCWA sees a staircase. Increase `n_lens_slices` or verify the microlens height map.

## RCWA vs TMM vs FDTD

| Method | Best at | Weak at |
|---|---|---|
| TMM | 1D thin films, BARL intuition | No lateral diffraction or crosstalk |
| RCWA | Periodic layered pixels, spectra, parameter sweeps | Aperiodic finite layouts, very sharp 3D objects |
| FDTD | General time-domain fields and finite features | Expensive fine grids, long convergence runs |

Use RCWA as the default for periodic unit-cell image-sensor optics. Use FDTD when periodicity breaks down, when time-domain behavior matters, or when the geometry cannot be represented well as z-sliced periodic layers.

## COMPASS RCWA solvers

COMPASS wraps several RCWA/FMM-style backends:

| Solver | Library | GPU support | Notes |
|---|---|---|---|
| `torcwa` | torcwa | CUDA (PyTorch) | Default for GPU-accelerated sweeps. |
| `grcwa` | grcwa | CUDA/JAX depending install | Useful cross-check backend. |
| `meent` | meent | CPU/CUDA depending install | Alternative RCWA implementation. |
| `fmmax` | fmmax | JAX accelerators | Vector FMM with selectable formulations. |

All implement the same `SolverBase` interface, so the first cross-check is often as simple as changing `solver.name`.

## Practical setup for image sensors

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

For final comparisons, rerun at a higher order and, if possible, with a second backend. Solver agreement is not proof of physical truth, but disagreement is a strong signal to inspect geometry, factorization, and convergence.

## Further reading

- M. G. Moharam and T. K. Gaylord, [Rigorous coupled-wave analysis of planar-grating diffraction](https://opg.optica.org/josa/abstract.cfm?uri=josa-71-7-811), JOSA 71, 811-818 (1981).
- M. G. Moharam et al., [Stable implementation of the rigorous coupled-wave analysis for surface-relief gratings](https://opg.optica.org/josaa/abstract.cfm?URI=josaa-12-5-1077), JOSA A 12, 1077-1086 (1995).
- L. Li, [Use of Fourier series in the analysis of discontinuous periodic structures](https://opg.optica.org/abstract.cfm?uri=josaa-13-9-1870), JOSA A 13, 1870-1876 (1996).
- V. Liu and S. Fan, [S4: a free electromagnetic solver for layered periodic structures](https://www.sciencedirect.com/science/article/pii/S0010465512001658), Computer Physics Communications 183, 2233-2244 (2012).
- Stanford Fan Group, [S4 documentation](https://web.stanford.edu/group/fan/S4), a practical RCWA/FMM reference implementation.
