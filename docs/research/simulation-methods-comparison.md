# EM Simulation Methods Comparison

> Written: 2026-02-11 | COMPASS project research document

---

## 1. Overview

Optical simulation of CMOS image sensor (CIS) pixels requires numerical solutions of Maxwell's equations. As pixel pitch has reached the same scale (0.6--1.4 um) as visible light wavelengths (0.38--0.78 um), geometric optics alone can no longer capture wave effects such as diffraction, interference, and near-field coupling.

The choice of simulation methodology is determined by three fundamental trade-offs:

| Axis | Description |
|---|---|
| **Accuracy** | Fidelity to physical reality. Level of approximation to Maxwell's equations |
| **Speed** | Wall-clock time for a single simulation |
| **Generality** | Range of geometries and materials that can be handled |

No single method can simultaneously optimize all three axes, which is the fundamental reason why diverse numerical methods coexist. COMPASS adopts RCWA and FDTD as its primary solvers and ensures result reliability through cross-validation.

| Method | Domain | Dimension | Periodicity | Wavelength Band | CIS Suitability |
|--------|------|------|--------|----------|-----------|
| RCWA | Frequency | 3D | Required | Single | ★★★★★ |
| FDTD | Time | 3D | Optional | Broadband | ★★★★☆ |
| FEM | Frequency | 3D | Optional | Single | ★★★☆☆ |
| BEM | Frequency | Surface | Not required | Single | ★★☆☆☆ |
| TMM | Frequency | 1D | N/A | Single | ★★★☆☆ |
| Ray Tracing | N/A | 3D | Not required | Broadband | ★★☆☆☆ |

---

## 2. RCWA (Rigorous Coupled-Wave Analysis)

### 2.1 Mathematical Foundation

RCWA is a semi-analytical method that solves Maxwell's equations for periodic structures by expanding them in Fourier series in the frequency domain. Key steps:

**(1) Fourier expansion of permittivity:** For a structure with periods $\Lambda_x, \Lambda_y$, the permittivity of each layer is expanded as:

$$\varepsilon(x, y) = \sum_{p=-N}^{N} \sum_{q=-N}^{N} \hat{\varepsilon}_{pq} \, e^{i(G_{px} x + G_{qy} y)}$$

where the reciprocal lattice vector $G_{px} = 2\pi p / \Lambda_x$, and the total number of harmonics is $M = (2N+1)^2$.

**(2) Eigenvalue problem**

Substituting the Fourier expansion into Maxwell's equations for each layer yields a $2M \times 2M$ eigenvalue problem:

$$\Omega \mathbf{w}_j = \gamma_j^2 \mathbf{w}_j$$

where $\gamma_j$ is the z-direction propagation constant for each mode, and $\mathbf{w}_j$ is the mode profile.

**(3) S-matrix cascading**

The scattering matrices of individual layers are combined using the Redheffer star product:

$$S_\text{total} = S_1 \star S_2 \star \cdots \star S_L$$

This method is numerically stable for evanescent modes, unlike the transfer matrix (T-matrix) approach.

### 2.2 Strengths

| Strength | Description |
|------|------|
| **Optimal for periodic structures** | Image sensor pixel arrays are inherently periodic, making them ideal for RCWA |
| **Accurate thin-film handling** | Each layer is treated exactly without spatial discretization (anti-reflection coating, color filter) |
| **Spectral analysis** | Single-wavelength computation is very fast → efficient QE spectrum calculation per wavelength |
| **Exponential convergence** | Exponential convergence with increasing Fourier order (for smooth profiles) |
| **GPU acceleration** | Matrix operation-based, well-suited for GPU acceleration (PyTorch/JAX backends) |

### 2.3 Weaknesses

| Weakness | Description |
|------|------|
| **Cannot handle non-periodic structures** | Finite structures, defects, and non-periodic patterns are fundamentally impossible to treat |
| **Staircase approximation of curved surfaces** | Curved surfaces such as microlenses require staircase approximation → degraded convergence |
| **Memory scaling** | Memory $O(M^2)$ and computation $O(M^3)$ for eigenvalue decomposition. At $N=15$, $M=961$, matrix size $1922 \times 1922$ |
| **Dispersive materials** | Separate computation required for each wavelength (repeated cost for broadband sweeps) |

### 2.4 Key Parameters

| Parameter | Role | COMPASS Default |
|----------|------|---------------|
| **Fourier order** ($N$) | Determines spatial resolution. Higher is more accurate but cost scales as $O(N^6)$ | `[9, 9]` |
| **Li's factorization** | Improves convergence at discontinuous boundaries. Inverse rule, normal vector method | `li_inverse` |
| **Polarization** | TE/TM or arbitrary polarization. Li's rule is particularly important for TM | Unpolarized (averaged) |

**Li's Fourier factorization rules:**

The three rules introduced by Lifeng Li (1996) are central to RCWA convergence:

1. **Laurent's rule**: When two functions have no simultaneous discontinuities → $[\![f \cdot g]\!] = [\![f]\!] \cdot [\![g]\!]$
2. **Inverse rule**: When all discontinuities are complementary → $[\![f \cdot g]\!] = [\![f^{-1}]\!]^{-1} \cdot [\![g]\!]$
3. **Impossible condition**: When non-complementary simultaneous discontinuities exist, both Laurent and inverse rules fail to converge

### 2.5 CIS Application Scenarios (When to Use for CIS)

- **Color filter**: Periodic array, planar layers → optimal for RCWA
- **BARL (Bottom Anti-Reflection Layer)**: Thin-film stack optimization, wavelength sweep → optimal for RCWA
- **Microlens**: Staircase approximation required, but sufficient accuracy at order 15+
- **Metal grid / DTI**: Sharp permittivity discontinuities → Li inverse rule essential, high order required
- **Parameter sweep**: Single-wavelength computation is fast, advantageous for thickness/pitch/angle sweeps

---

## 3. FDTD (Finite-Difference Time-Domain)

### 3.1 Mathematical Foundation

FDTD directly discretizes the curl equations of Maxwell's equations in both time and space.

**Yee lattice:**

The staggered grid proposed by Kane Yee (1966) places the six components of the electric field ($\mathbf{E}$) and magnetic field ($\mathbf{H}$) offset by half a grid point in space. This arrangement naturally ensures second-order accurate central differences.

**Update equations (Leapfrog time-stepping):**

The electric and magnetic fields are updated alternately in half time steps:

$$H_x^{n+1/2} = H_x^{n-1/2} + \frac{\Delta t}{\mu_0} \left( \frac{E_y^n|_{k+1} - E_y^n|_k}{\Delta z} - \frac{E_z^n|_{j+1} - E_z^n|_j}{\Delta y} \right)$$

$$E_x^{n+1} = E_x^n + \frac{\Delta t}{\varepsilon_0 \varepsilon_r} \left( \frac{H_z^{n+1/2}|_{j} - H_z^{n+1/2}|_{j-1}}{\Delta y} - \frac{H_y^{n+1/2}|_{k} - H_y^{n+1/2}|_{k-1}}{\Delta z} \right)$$

**Courant-Friedrichs-Lewy (CFL) stability condition:**

The time step must satisfy the following condition for numerical stability:

$$\Delta t \leq \frac{1}{c \sqrt{\frac{1}{\Delta x^2} + \frac{1}{\Delta y^2} + \frac{1}{\Delta z^2}}}$$

### 3.2 Strengths

| Strength | Description |
|------|------|
| **Broadband response** | Entire visible spectrum obtained from a single simulation (Fourier transform of impulse response) |
| **Arbitrary geometry** | Any 3D structure can be represented within grid resolution. No periodicity required |
| **Time-domain information** | Pulse propagation and transient response can be directly observed |
| **Intuitive implementation** | Update equations are simple arithmetic operations → easy parallelization and GPU acceleration |
| **Nonlinear materials** | Nonlinear response can be naturally included in the time domain |

### 3.3 Weaknesses

| Weakness | Description |
|------|------|
| **CFL constraint** | Fine grid → small time step → long simulation time. Critical for thin films |
| **Memory** | Entire 3D grid must be kept in memory. 5 nm grid, 1 um pixel → $200^3 \approx 8 \times 10^6$ cells x 6 components |
| **Dispersive materials** | Frequency-dependent permittivity of metals and semiconductors must be handled via auxiliary differential equations (ADE) |
| **Thin-film inefficiency** | Even BARL layers of a few nm thickness must be resolved by the full grid (RCWA handles them analytically) |
| **Numerical dispersion** | Insufficient grid resolution causes artificial dispersion in phase velocity ($\Delta x \leq \lambda / 20$ recommended) |

### 3.4 Key Parameters

| Parameter | Role | Recommended for CIS Simulation |
|----------|------|---------------------|
| **Grid spacing** ($\Delta x, \Delta y, \Delta z$) | Spatial resolution. $\lambda/(20n)$ or below recommended | 5--10 nm (visible, Si) |
| **Time step** ($\Delta t$) | Automatically determined by CFL | ~0.01 fs (5 nm grid) |
| **PML layer count** | Thickness of Perfectly Matched Layer | 8--16 layers |
| **Total time steps** | Until steady state is reached | Thousands to tens of thousands of steps |
| **Source type** | Broadband pulse or continuous wave (CW) | Gaussian pulse (380--780 nm) |

**PML (Perfectly Matched Layer):** The PML proposed by Berenger (1994) is an artificial layer that absorbs outgoing waves at the computational domain boundary without reflection. UPML and CPML are the current mainstream implementations.

### 3.5 CIS Application Scenarios (When to Use for CIS)

- **Complex 3D structures**: Asymmetric DTI, irregular metal interconnects → FDTD advantageous
- **RCWA result verification**: Cross-validation via FDTD spot-check at selected wavelengths
- **Broadband QE**: When the entire visible band needs to be obtained in a single run
- **Time-domain analysis**: Visualization of light propagation through microlenses
- **Non-periodic structures**: Single-pixel analysis, edge effect studies

---

## 4. FEM (Finite Element Method)

The Finite Element Method (FEM) solves the weak form of Maxwell's equations on tetrahedral/hexahedral meshes. The variational formulation of the vector wave equation:

$$\int_\Omega \left[ \frac{1}{\mu_r} (\nabla \times \mathbf{E}) \cdot (\nabla \times \mathbf{F}) - k_0^2 \varepsilon_r \mathbf{E} \cdot \mathbf{F} \right] d\Omega = -ik_0 Z_0 \int_\Omega \mathbf{J} \cdot \mathbf{F} \, d\Omega$$

The geometry is subdivided into tetrahedra, and the key advantage is the ability to perform adaptive mesh refinement near curved surfaces.

### Strengths and Weaknesses

| Strengths | Weaknesses |
|------|------|
| Accurate representation of curved geometry (curvilinear elements) | Cost of assembling and solving large sparse matrices |
| Adaptive mesh refinement (AMR) | Mesh generation itself is complex (especially in 3D) |
| Easy multiphysics coupling (thermal, structural) | Frequency domain → repeated computation per wavelength required |
| Non-uniform/anisotropic material handling | Limited open-source optical FEM solvers |

COMPASS does not currently include FEM. For the periodicity of CIS pixels, RCWA is more efficient, and commercial FEM (COMSOL) has high licensing costs. However, FEM may be the only option for microlens curved surfaces or thermo-optical multiphysics simulations.

---

## 5. BEM (Boundary Element Method)

The Boundary Element Method (BEM) places unknowns only on boundary surfaces rather than throughout the volume. Using the free-space Green's function $G(\mathbf{r}, \mathbf{r}') = e^{ik|\mathbf{r}-\mathbf{r}'|} / (4\pi|\mathbf{r}-\mathbf{r}'|)$, surface integral equations reduce a 3D volume problem to a 2D surface problem.

### Strengths and Weaknesses

| Strengths | Weaknesses |
|------|------|
| Dimension reduction: 3D problem to 2D surface problem | Dense matrix → $O(N^2)$ memory, $O(N^3)$ solve |
| Open boundary naturally handled | Difficulty with non-uniform/nonlinear materials |
| Efficient for far-field calculations | Specialized Green's function needed for layered media |
| Optimized for scattering problems | Inefficient for multilayer thin-film stacks |

BEM is not commonly used for CIS pixel simulation. Image sensors are multilayer structures where the electric field distribution throughout the entire volume is important. It may be useful for studying scattering characteristics of individual microlenses or plasmonic responses of metal nanoparticles.

---

## 6. TMM (Transfer Matrix Method)

### 6.1 Mathematical Foundation

The Transfer Matrix Method (TMM) is an analytical method that describes electromagnetic wave propagation in planar multilayer thin films as matrix products.

**2x2 transfer matrix (isotropic, normal incidence):**

The transfer matrix for each layer $j$ is:

$$M_j = \begin{pmatrix} \cos\delta_j & -\frac{i}{n_j}\sin\delta_j \\ -in_j\sin\delta_j & \cos\delta_j \end{pmatrix}$$

where the phase thickness $\delta_j = \frac{2\pi}{\lambda} n_j d_j \cos\theta_j$.

The transfer matrix for the entire stack is a simple product:

$$M_\text{total} = M_1 \cdot M_2 \cdots M_L$$

**4x4 Berreman matrix (anisotropic):**

For cases involving anisotropic materials, Berreman's (1972) 4x4 formulation is required:

$$\frac{d}{dz} \boldsymbol{\Psi}(z) = \frac{i\omega}{c} \mathbf{D}(z) \boldsymbol{\Psi}(z)$$

where $\boldsymbol{\Psi} = (E_x, H_y, E_y, -H_x)^T$ and $\mathbf{D}$ is a 4x4 matrix constructed from the permittivity tensor.

### 6.2 Strengths and Weaknesses

| Strengths | Weaknesses |
|------|------|
| **Extremely fast**: Completed in just a few matrix multiplications | **Limited to 1D**: Cannot handle lateral patterns |
| Analytical accuracy: No numerical discretization error | Possible numerical instability in thick absorbing layers |
| Immediate calculation of reflectance/transmittance/absorptance | Cannot capture diffraction or scattering phenomena |
| Industry standard for multilayer thin-film design | Cannot handle 2D/3D patterns such as microlenses and metal grids |

### 6.3 CIS Application Scenarios (When to Use for CIS)

TMM is not used as a direct solver in COMPASS, but is extremely useful for the following purposes:

- **Initial stack design**: Starting point for BARL and ARC (Anti-Reflection Coating) thickness optimization
- **Material screening**: Rapid evaluation of absorption/transmission characteristics of color filter materials
- **Analytical verification**: Reference for RCWA results in structures composed only of uniform layers
- **1D convergence check**: RCWA results at $N=0$ (zeroth order only) should match TMM

COMPASS's `compass.materials.database.MaterialDB` internally uses TMM-based thin-film reflectance calculations.

---

## 7. Ray Tracing

### 7.1 Geometric Optics Approximation

Ray tracing treats light as rays and traces propagation using Snell's law and Fresnel coefficients. It corresponds to the short-wavelength limit ($\lambda \to 0$) of Maxwell's equations.

**Ray equation:**

$$\frac{d}{ds}\left(n \frac{d\mathbf{r}}{ds}\right) = \nabla n$$

where $s$ is the arc length along the ray path, and $n(\mathbf{r})$ is the refractive index distribution.

### 7.2 Strengths and Weaknesses

| Strengths | Weaknesses |
|------|------|
| **Extremely fast**: Millions of rays traced in seconds | Ignoring diffraction at wavelength-scale structures → critical errors |
| Intuitive physics: Easy visualization of ray paths | Cannot capture interference phenomena (thin-film effects, etc.) |
| Standard for lens system design (Zemax, Code V) | Ignores near-field coupling |
| Suitable for CRA (Chief Ray Angle) analysis | Rapidly inaccurate when pixel pitch < several $\lambda$ |

### 7.3 Role in CIS

In modern CIS design, ray tracing is primarily used as a **pre-processing** stage for wave optics:

1. **Camera lens → sensor plane**: Compute CRA and irradiance distribution in Zemax/Code V
2. **Handoff**: Extract incidence conditions (angle, amplitude) at the sensor plane
3. **Wave optics simulation**: Pixel simulation with the corresponding incidence conditions in COMPASS's RCWA/FDTD

COMPASS's `compass.sources.ray_file_reader` and `compass.sources.cone_illumination` modules support this handoff.

---

## 8. Hybrid Methods

No single method can efficiently handle all scales of an image sensor system. Hybrid methods combine the optimal method for each scale.

### 8.1 Ray Tracing → RCWA Handoff (Zemax → COMPASS)

```
Camera lens (mm scale)          →  Zemax (Ray Tracing)
    ↓ CRA, irradiance
Pixel stack (um scale)          →  COMPASS (RCWA/FDTD)
    ↓ QE, crosstalk
Sensor performance (pixel array) →  System analysis
```

Key interface: angle of incidence ($\theta$, $\phi$), polarization state, power distribution.

### 8.2 FEM + Scattering Matrix (EMUstack Approach)

EMUstack solves each layer with 2D FEM and handles inter-layer coupling via scattering matrices. It combines the geometric flexibility of FEM with the numerical stability of S-matrices, but the complexity of mesh generation remains.

### 8.3 Multi-scale Approaches

| Scale | Method | Target |
|--------|------|------|
| Several mm | Ray Tracing | Camera lens, microlens array |
| Several um | RCWA / FDTD | Color filter, BARL, DTI |
| Several nm | FEM / BEM | Plasmonic nanostructures, surface roughness |

As a future direction, neural network-based surrogate models are gaining attention. Neural networks trained on RCWA/FDTD data provide real-time predictions, and precise solvers are called only at points where high accuracy is needed.

---

## 9. Comprehensive Performance Comparison

### 9.1 Qualitative Comparison

| Characteristic | RCWA | FDTD | FEM | BEM | TMM | Ray Tracing |
|------|------|------|-----|-----|-----|-------------|
| **Accuracy** | High (periodic) | High | Very high | High (surface) | Exact (1D) | Low (ignores wave effects) |
| **Speed (single wavelength)** | Very fast | Slow | Slow | Medium | Extremely fast | Extremely fast |
| **Speed (broadband)** | Medium (repeated) | Fast (single run) | Slow (repeated) | Slow (repeated) | Extremely fast | Extremely fast |
| **Memory** | Medium | High | High | High (dense) | Extremely low | Low |
| **Geometric flexibility** | Low (periodic only) | High | Very high | Medium | None (1D only) | High (macro) |
| **Material flexibility** | High | Medium | Very high | Low | High | Medium |
| **Automatic differentiation (AD)** | Possible | Possible | Limited | Difficult | Possible | Difficult |
| **GPU acceleration** | Very suitable | Suitable | Limited | Limited | Unnecessary | Suitable |

### 9.2 Computational Scaling

| Method | Spatial DOF | Time Complexity | Memory Complexity | Bottleneck |
|------|-----------|-----------|-------------|------|
| **RCWA** | $M = (2N+1)^2$ | $O(M^3)$ per layer | $O(M^2)$ | Eigenvalue decomposition |
| **FDTD** | $N_x N_y N_z$ | $O(N_\text{total} \cdot T)$ | $O(N_\text{total})$ | Number of time steps $T$ |
| **FEM** | $N_\text{DOF}$ (mesh nodes) | $O(N_\text{DOF}^{1.5})$ sparse | $O(N_\text{DOF})$ sparse | Matrix solve |
| **BEM** | $N_\text{surface}$ | $O(N_\text{surface}^3)$ | $O(N_\text{surface}^2)$ | Dense matrix |
| **TMM** | $L$ (number of layers) | $O(L)$ | $O(1)$ | None |
| **Ray Tracing** | $N_\text{rays}$ | $O(N_\text{rays} \cdot S)$ | $O(N_\text{rays})$ | Number of rays $N_\text{rays}$, number of surfaces $S$ |

### 9.3 Typical Execution Times

1 um pitch BSI pixel, 2x2 Bayer unit cell, at 550 nm:

| Method | Parameter Settings | DOF | Single Wavelength Time | 41-Wavelength Sweep |
|------|-------------|--------|-------------|--------------|
| **RCWA** (GPU) | $N = 9$, 10 layers | ~7,000 | **0.3 s** | 12 s |
| **RCWA** (GPU) | $N = 15$, 10 layers | ~19,000 | 2 s | 80 s |
| **FDTD** (GPU) | $\Delta x = 5$ nm, PML 12 | ~8M cells | 45 s | **45 s** (broadband) |
| **FDTD** (CPU) | $\Delta x = 10$ nm, PML 8 | ~1M cells | 300 s | 300 s |
| **FEM** | Adaptive mesh, $\lambda/10$ | ~500K DOF | 60 s | 2,460 s |
| **TMM** | 10 layers | 10 | **< 0.001 s** | 0.04 s |

> **Note**: The above figures are representative estimates and can vary significantly depending on hardware (GPU: NVIDIA A100, CPU: 8-core) and implementation.

### 9.4 Differentiable Simulation Support

Automatic differentiation (AD) support for inverse design and topology optimization:

| Method | AD Framework | Gradient Method | Representative Solvers |
|------|-------------|---------------|----------|
| **RCWA** | PyTorch, JAX | Forward/Reverse AD | meent, fmmax, torcwa |
| **FDTD** | PyTorch, JAX | Reverse AD, Adjoint | FDTDX, flaport, fdtdz |
| **FEM** | Limited | Adjoint method | EMOPT (FDFD) |
| **TMM** | Easy | Analytical gradient | Custom implementation |

<SolverComparisonChart />

---

## 10. Application in COMPASS

### 10.1 Rationale for RCWA + FDTD Selection

| Criterion | RCWA | FDTD | Rationale |
|------|------|------|----------|
| CIS pixel periodicity | Perfect fit | Suitable | Pixel array = periodic structure |
| Thin-film stack handling | Analytical treatment | Grid discretization | RCWA advantage for BARL/ARC design |
| Cross-validation | - | - | Independent verification via different mathematical approaches |
| GPU acceleration | Very suitable | Suitable | Leveraging PyTorch/JAX-based open source |
| License | MIT available | MIT available | meent (MIT), flaport (MIT) |

### 10.2 Cross-Validation Philosophy

Because the same physical laws are solved with different mathematical approaches, agreement between the two solvers significantly increases result confidence. Checklist when discrepancies arise:

1. **Insufficient RCWA convergence** → Increase Fourier order
2. **Insufficient FDTD resolution** → Refine grid
3. **Modeling differences** → Review staircase approximation, material models, boundary conditions
4. **Energy conservation violation** ($R + T + A \neq 1$) → Implementation bug

The `SolverComparison` class automates QE difference, relative error, and energy conservation verification.

### 10.3 Solver Selection Guide (Decision Guide)

```
Start simulation
    │
    ├─ Is the structure periodic?
    │   ├─ YES → Only thin films?
    │   │         ├─ YES → TMM (initial design) → RCWA (precise)
    │   │         └─ NO  → RCWA (default) + FDTD (verification)
    │   └─ NO  → FDTD
    │
    ├─ Is broadband needed?
    │   ├─ YES, 50+ wavelengths → FDTD (single run is more efficient)
    │   └─ NO, < 50 wavelengths → RCWA (per-wavelength iteration is faster)
    │
    └─ Is time-domain information needed?
        ├─ YES → FDTD
        └─ NO  → RCWA (default choice)
```

### 10.4 Future Directions

COMPASS solver expansion roadmap:

| Priority | Solver/Method | Purpose |
|---------|----------|------|
| **High** | fmmax (RCWA) integration | Improved convergence via vector FMM, JAX batching |
| **High** | FDTDX (FDTD) integration | Multi-GPU 3D, large-scale inverse design |
| **Medium** | Built-in TMM module | Fast stack pre-design, 1D reference |
| **Medium** | Neural network surrogate model | Real-time parameter optimization |
| **Low** | FEM integration (EMUstack) | Plasmonic/curved surface specialized research |

---

## References

### Key Papers

- K. S. Yee, "Numerical solution of initial boundary value problems involving Maxwell's equations in isotropic media," *IEEE Trans. Antennas Propag.*, vol. 14, no. 3, pp. 302-307, 1966.
- M. G. Moharam and T. K. Gaylord, "Rigorous coupled-wave analysis of planar-grating diffraction," *J. Opt. Soc. Am.*, vol. 71, no. 7, pp. 811-818, 1981.
- L. Li, "Use of Fourier series in the analysis of discontinuous periodic structures," *J. Opt. Soc. Am. A*, vol. 13, no. 9, pp. 1870-1876, 1996.
- J.-P. Berenger, "A perfectly matched layer for the absorption of electromagnetic waves," *J. Comput. Phys.*, vol. 114, no. 2, pp. 185-200, 1994.
- D. W. Berreman, "Optics in stratified and anisotropic media: 4x4-matrix formulation," *J. Opt. Soc. Am.*, vol. 62, no. 4, pp. 502-510, 1972.

### Web Resources

- [Planopsim: RCWA vs FDTD Benchmark](https://planopsim.com/design-example/getting-accurate-and-fast-nano-structure-simulations-a-benchmark-of-rcwa-and-fdtd-for-meta-surface-calculation/)
- [Ansys: CMOS Optical Simulation Methodology](https://optics.ansys.com/hc/en-us/articles/360042851793-CMOS-Optical-simulation-methodology)
- [Joint EM and Ray-Tracing Simulations for Quad-Pixel Sensor](https://opg.optica.org/oe/fulltext.cfm?uri=oe-27-21-30486&id=422094)
- [FDTD vs FEM vs MoM - Cadence](https://resources.system-analysis.cadence.com/blog/msa2021-fdtd-vs-fem-vs-mom-what-are-they-and-how-are-they-different)
- [3D Broadband FDTD Simulations of CMOS Image Sensor](https://arxiv.org/abs/2310.10302)
- [VarRCWA: Adaptive High-Order RCWA](https://pmc.ncbi.nlm.nih.gov/articles/PMC9589908/)
