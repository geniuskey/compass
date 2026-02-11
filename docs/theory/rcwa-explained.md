# RCWA Explained

::: tip 선수 지식 | Prerequisites
[전자기파](/theory/electromagnetic-waves) → [회절](/theory/diffraction) → 이 페이지
RCWA가 처음이라면 먼저 [솔버 선택 가이드](/guide/choosing-solver)에서 개요를 확인하세요.
:::

RCWA (Rigorous Coupled-Wave Analysis) is one of the two main solver types in COMPASS. Think of it as a method that breaks down the pixel structure into its spatial frequency components (Fourier series), then solves Maxwell's equations for each component. It's particularly fast for periodic structures like pixel arrays.

Rigorous Coupled-Wave Analysis (RCWA) is the primary solver method in COMPASS. It solves Maxwell's equations in the frequency domain for periodic structures by expanding fields in Fourier harmonics.

## Core idea

RCWA decomposes the problem into two parts:

1. **Lateral (xy)**: The periodic permittivity and fields are expanded in a truncated Fourier series.
2. **Vertical (z)**: Within each layer, the Fourier coefficients satisfy a system of coupled ODEs that can be solved by eigendecomposition.

The layers are then connected using scattering-matrix (S-matrix) recursion to enforce boundary conditions at every interface.

## Step-by-step algorithm

### 1. Fourier expansion of permittivity

For a 2D periodic structure with periods $\Lambda_x$ and $\Lambda_y$, the permittivity in each layer is expanded:

$$\varepsilon(x, y) = \sum_{p,q} \hat{\varepsilon}_{pq} \, e^{i(G_{px} x + G_{qy} y)}$$

where $G_{px} = 2\pi p / \Lambda_x$ and $G_{qy} = 2\pi q / \Lambda_y$ are reciprocal lattice vectors. The expansion is truncated to $|p| \leq N_x$ and $|q| \leq N_y$, giving a total of $(2N_x+1)(2N_y+1)$ harmonics.

In COMPASS, the Fourier order is set in the solver config:

```yaml
solver:
  name: torcwa
  params:
    fourier_order: [9, 9]   # Nx=Ny=9 -> 19x19 = 361 harmonics
```

### 2. Eigenvalue problem

Within each layer, substituting the Fourier expansion into Maxwell's equations yields an eigenvalue problem of dimension $2M \times 2M$, where $M = (2N_x+1)(2N_y+1)$:

$$\Omega \mathbf{w} = \lambda^2 \mathbf{w}$$

The matrix $\Omega$ encodes the coupling between Fourier harmonics due to the permittivity pattern. The eigenvalues $\lambda_j$ give the propagation constants and the eigenvectors $\mathbf{w}_j$ give the mode profiles.

For a uniform layer, $\Omega$ is diagonal and the eigenvalues are simply the plane-wave propagation constants.

### 3. S-matrix recursion

Individual layer S-matrices are combined using the **Redheffer star product**, which avoids the numerical instability of transfer-matrix (T-matrix) cascading:

$$S_\text{total} = S_1 \star S_2 \star \cdots \star S_N$$

The star product formula for combining $S^A$ and $S^B$:

$$S_{11} = S^A_{11} + S^A_{12} D S^B_{11} S^A_{21}$$

$$S_{12} = S^A_{12} D S^B_{12}$$

where $D = (I - S^B_{11} S^A_{22})^{-1}$. This formulation is numerically stable because all intermediate matrices involve only decaying exponentials.

### 4. Field computation

Once the global S-matrix is known, the reflected and transmitted diffraction orders are obtained directly:

$$\begin{pmatrix} \mathbf{r} \\ \mathbf{t} \end{pmatrix} = \begin{pmatrix} S_{11} & S_{12} \\ S_{21} & S_{22} \end{pmatrix} \begin{pmatrix} \mathbf{a} \\ \mathbf{0} \end{pmatrix}$$

where $\mathbf{a}$ is the incident wave amplitude. The diffraction efficiencies (reflected and transmitted power per order) are then:

$$R_m = \frac{\text{Re}(k_{z,m}^r)}{k_{z,0}^i} |r_m|^2$$

$$T_m = \frac{\text{Re}(k_{z,m}^t)}{k_{z,0}^i} |t_m|^2$$

## Fourier factorization

The convergence of RCWA depends critically on how the permittivity Fourier coefficients are computed. For structures with sharp material boundaries (e.g., metal grids), the standard approach converges slowly for TM polarization.

**Li's inverse rule** dramatically improves convergence by using $[\text{FT}(1/\varepsilon)]^{-1}$ instead of $\text{FT}(\varepsilon)$ for the appropriate field components:

| Factorization | When to use | COMPASS setting |
|---|---|---|
| Naive (Laurent rule) | Continuous permittivity profiles | `fourier_factorization: "naive"` |
| Li inverse rule | Discontinuous boundaries (metal grids, DTI) | `fourier_factorization: "li_inverse"` |
| Normal vector method | Complex 2D patterns | `fourier_factorization: "normal_vector"` |

## Convergence

The accuracy of RCWA improves as the Fourier order increases, but so does the computational cost (the eigenvalue problem scales as $O(M^3)$). A typical convergence study sweeps `fourier_order` from 3 to 25 and plots QE vs order:

<RCWAConvergenceDemo />

```yaml
solver:
  convergence:
    auto_converge: true
    order_range: [5, 25]
    qe_tolerance: 0.01
```

For a 1 um pitch pixel with a simple color filter pattern, order 9 is usually sufficient. For finer features (metal grids, DTI), order 15-21 may be needed.

## RCWA solvers in COMPASS

COMPASS wraps three RCWA libraries:

| Solver | Library | GPU support | Notes |
|--------|---------|-------------|-------|
| `torcwa` | torcwa | CUDA (PyTorch) | Default. Best for GPU-accelerated sweeps. |
| `grcwa` | grcwa | CUDA (JAX/PyTorch) | Alternative GPU backend. |
| `meent` | meent | CUDA/CPU | Supports analytic eigendecomposition. |

All three implement the same `SolverBase` interface, so switching between them requires only changing `solver.name` in the config.
