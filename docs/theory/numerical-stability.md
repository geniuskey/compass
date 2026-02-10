# Numerical Stability

RCWA simulations can suffer from numerical instability, especially at high Fourier orders, short wavelengths, and with absorbing materials. COMPASS implements a five-layer defense system to address these issues.

## Sources of instability

### 1. Eigenvalue decomposition errors

The core of RCWA is solving a $2M \times 2M$ eigenvalue problem (where $M$ can be 300+). In single precision (float32), the limited mantissa bits cause:

- Loss of orthogonality in eigenvectors
- Incorrect eigenvalue ordering
- Near-degenerate eigenvalue pairs mixing

### 2. Exponential overflow in T-matrix

The traditional transfer-matrix (T-matrix) approach propagates fields through layers using:

$$T_j = \begin{pmatrix} \exp(+\lambda d) & \cdots \\ \cdots & \exp(-\lambda d) \end{pmatrix}$$

For evanescent modes, $\lambda$ is imaginary and $\exp(+|\lambda|d)$ grows exponentially. At high Fourier order, these divergent terms cause overflow and catastrophic cancellation.

### 3. TF32 on NVIDIA GPUs

NVIDIA Ampere and newer GPUs use TF32 (TensorFloat-32) by default for float32 matrix multiplications. TF32 has only 10 mantissa bits (vs. 23 for float32), which destroys RCWA accuracy silently. **COMPASS disables TF32 by default.**

### 4. Fourier factorization at discontinuities

At sharp material boundaries (metal grids, DTI), the standard Fourier representation of permittivity converges slowly and can produce unphysical results (Gibbs phenomenon). This primarily affects TM polarization.

## The five-layer defense

COMPASS addresses each instability source with a dedicated module in `compass.solvers.rcwa.stability`:

### Layer 1: PrecisionManager

Controls floating-point precision at every stage:

```yaml
solver:
  stability:
    precision_strategy: "mixed"    # float32 for most, float64 for eigendecomp
    allow_tf32: false              # CRITICAL: disable TF32
    eigendecomp_device: "cpu"      # CPU eigendecomp is more stable than GPU
```

The `mixed` strategy keeps most computation in float32 for speed but promotes the eigendecomposition to float64:

```python
# Internally, COMPASS does:
matrix_f64 = matrix.astype(np.complex128)
eigenvalues, eigenvectors = np.linalg.eig(matrix_f64)
# Then converts back to original precision
```

### Layer 2: S-Matrix Algorithm

COMPASS uses the **Redheffer star product** instead of the T-matrix. The star product combines S-matrices layer by layer using only bounded quantities:

$$S_\text{total} = S_1 \star S_2 \star \cdots \star S_N$$

The key property: all intermediate exponentials are of the form $\exp(-|\lambda|d)$, which are always $\leq 1$. No overflow is possible.

### Layer 3: Li Factorization

For structures with sharp dielectric boundaries, COMPASS applies Li's inverse rule:

$$[\varepsilon]_\text{eff} = \left[\text{FT}\left(\frac{1}{\varepsilon}\right)\right]^{-1}$$

instead of the naive $\text{FT}(\varepsilon)$. This dramatically improves convergence for TM polarization at metal/dielectric interfaces.

### Layer 4: Eigenvalue Stabilizer

Post-processing of eigenvalues to handle:

- **Degenerate eigenvalues**: When two eigenvalues are closer than a threshold (`eigenvalue_broadening: 1e-10`), their eigenvectors are orthogonalized via Gram-Schmidt.
- **Branch selection**: The square root of eigenvalues must choose the correct sign. COMPASS enforces $\text{Re}(\sqrt{\lambda}) \geq 0$ for propagating modes and the correct decay direction for evanescent modes.

### Layer 5: Adaptive Precision Runner

If the energy balance check fails ($|R + T + A - 1| > \text{tolerance}$), COMPASS automatically retries with higher precision:

```
float32 (GPU)  --->  float64 (GPU)  --->  float64 (CPU)
     fast               slower              slowest but most stable
```

This is configured via:

```yaml
solver:
  stability:
    energy_check:
      enabled: true
      tolerance: 0.02
      auto_retry_float64: true
```

## Diagnosing stability issues

### Pre-simulation checks

The `StabilityDiagnostics.pre_simulation_check` method warns about:

- Large matrices with float32 (risk: eigendecomp failure)
- Thick layers without S-matrix (risk: T-matrix overflow)
- TF32 enabled (risk: silent accuracy loss)
- Patterned layers with naive factorization (risk: slow convergence)

### Post-simulation checks

After simulation, `StabilityDiagnostics.post_simulation_check` validates:

- QE in range $[0, 1]$ for all pixels
- No NaN or Inf in results
- Energy conservation: $|R + T + A - 1| < 0.05$

### Warning signs

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| QE > 100% or < 0% | Eigendecomp failure | Use `precision_strategy: "float64"` |
| Energy balance violation | T-matrix overflow | Ensure S-matrix algorithm is used |
| Noisy QE at short wavelengths | Float32 insufficient | Enable `auto_retry_float64` |
| Slow TM convergence | Naive factorization | Switch to `li_inverse` |
| Condition number warning | Nearly singular matrix | Reduce Fourier order or increase broadening |

## Recommended settings

For production simulations:

```yaml
solver:
  stability:
    precision_strategy: "mixed"
    allow_tf32: false
    eigendecomp_device: "cpu"
    fourier_factorization: "li_inverse"
    energy_check:
      enabled: true
      tolerance: 0.02
      auto_retry_float64: true
    eigenvalue_broadening: 1.0e-10
    condition_number_warning: 1.0e+12
```

For maximum accuracy (at the cost of speed):

```yaml
solver:
  stability:
    precision_strategy: "float64"
    allow_tf32: false
    eigendecomp_device: "cpu"
    fourier_factorization: "li_inverse"
```
