"""RCWA Numerical Stability Module — 5-Layer Defense.

Implements comprehensive numerical stability measures for RCWA solvers:
1. PrecisionManager: TF32 control, mixed precision eigendecomp (CPU / cuSOLVER)
2. StableSMatrixAlgorithm: Redheffer star product (no T-matrix)
3. LiFactorization: Inverse rule for high-contrast boundaries
4. EigenvalueStabilizer: Degenerate eigenvalue handling, branch selection
5. AdaptivePrecisionRunner: Automatic fallback strategy

Eigendecomposition backends:
    - CPU (default): torch.linalg.eig on CPU with float64
    - cuSOLVER (via CuPy): GPU-resident eigendecomp via NVIDIA cuSOLVER,
      zero-copy DLPack transfer between PyTorch ↔ CuPy. Eliminates
      GPU→CPU→GPU round-trip. Requires: pip install cupy-cuda12x

References:
    - Moharam et al., JOSA A 12(5), 1995 (S-matrix)
    - Li, JOSA A 13(9), 1996 (Fourier factorization)
    - Schuster et al., JOSA A 24(9), 2007 (Normal Vector Method)
    - Kim & Lee, CPC 282, 2023 (eigenvalue broadening)
"""

from __future__ import annotations

import logging

import numpy as np

from compass.core.types import SimulationResult
from compass.geometry.pixel_stack import PixelStack

logger = logging.getLogger(__name__)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


def has_cusolver() -> bool:
    """Check if cuSOLVER is available via CuPy with a CUDA device."""
    if not HAS_CUPY:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


class PrecisionManager:
    """RCWA numerical precision management."""

    @staticmethod
    def configure(config: dict) -> None:
        """Configure precision settings from config."""
        if not HAS_TORCH:
            return

        stability = config.get("stability", {})

        # TF32 must be disabled for RCWA
        torch.backends.cuda.matmul.allow_tf32 = stability.get("allow_tf32", False)
        torch.backends.cudnn.allow_tf32 = stability.get("allow_tf32", False)

        if torch.backends.cuda.matmul.allow_tf32:
            logger.warning(
                "TF32 is ENABLED — this WILL cause RCWA instability. "
                "Set allow_tf32: false in solver.stability config."
            )

    @staticmethod
    def mixed_precision_eigen(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Eigendecomposition with float64 precision.

        Performs eigendecomp in float64 regardless of input precision,
        then returns results in original precision.

        Args:
            matrix: Square complex matrix.

        Returns:
            (eigenvalues, eigenvectors) tuple.
        """
        orig_dtype = matrix.dtype
        matrix_f64 = matrix.astype(np.complex128)
        eigenvalues, eigenvectors = np.linalg.eig(matrix_f64)

        if orig_dtype != np.complex128:
            eigenvalues = eigenvalues.astype(orig_dtype)
            eigenvectors = eigenvectors.astype(orig_dtype)

        return eigenvalues, eigenvectors

    @staticmethod
    def mixed_precision_eigen_torch(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Eigendecomposition with float64 precision (PyTorch version).

        Tries cuSOLVER via CuPy for GPU-resident computation first,
        then falls back to CPU torch.linalg.eig if CuPy is unavailable
        or the tensor is not on CUDA.
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch required for torch eigendecomp")

        orig_dtype = matrix.dtype
        orig_device = matrix.device

        # Try GPU path via cuSOLVER (CuPy) — avoids GPU→CPU→GPU round-trip
        if HAS_CUPY and matrix.is_cuda:
            try:
                return PrecisionManager._cusolver_eig(matrix, orig_dtype, orig_device)
            except Exception as e:
                logger.debug(f"cuSOLVER eigendecomp failed, falling back to CPU: {e}")

        # CPU fallback
        matrix_f64 = matrix.to(torch.complex128).cpu()
        eigenvalues, eigenvectors = torch.linalg.eig(matrix_f64)

        return (
            eigenvalues.to(dtype=orig_dtype, device=orig_device),
            eigenvectors.to(dtype=orig_dtype, device=orig_device),
        )

    @staticmethod
    def _cusolver_eig(
        matrix: torch.Tensor,
        orig_dtype: torch.dtype,
        orig_device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Eigendecomposition on GPU via cuSOLVER (CuPy).

        Uses DLPack for zero-copy memory sharing between PyTorch and CuPy.
        Computation stays entirely on GPU — no host transfer needed.
        """
        # PyTorch → CuPy (zero-copy via DLPack)
        matrix_f64 = matrix.to(torch.complex128)
        cp_matrix = cp.from_dlpack(matrix_f64)

        # cuSOLVER geev — runs on same GPU stream
        eigenvalues_cp, eigenvectors_cp = cp.linalg.eig(cp_matrix)

        # CuPy → PyTorch (zero-copy via DLPack)
        eigenvalues = torch.from_dlpack(eigenvalues_cp)
        eigenvectors = torch.from_dlpack(eigenvectors_cp)

        return (
            eigenvalues.to(dtype=orig_dtype, device=orig_device),
            eigenvectors.to(dtype=orig_dtype, device=orig_device),
        )


class StableSMatrixAlgorithm:
    """Redheffer Star Product based S-matrix recursion.

    Eliminates T-matrix exp(+λd) divergence by using only
    exp(-|λ|d) terms (decaying exponentials).
    """

    @staticmethod
    def redheffer_star_product(
        SA: dict[str, np.ndarray],
        SB: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Redheffer star product of two S-matrices.

        All intermediate matrices remain bounded → no overflow.

        Args:
            SA: First S-matrix with keys S11, S12, S21, S22.
            SB: Second S-matrix.

        Returns:
            Combined S-matrix.
        """
        n = SA["S11"].shape[0]
        I = np.eye(n, dtype=SA["S11"].dtype)

        # D = (I - SB_S11 @ SA_S22)^(-1)
        D = np.linalg.solve(I - SB["S11"] @ SA["S22"], I)
        # F = (I - SA_S22 @ SB_S11)^(-1)
        F = np.linalg.solve(I - SA["S22"] @ SB["S11"], I)

        S = {
            "S11": SA["S11"] + SA["S12"] @ D @ SB["S11"] @ SA["S21"],
            "S12": SA["S12"] @ D @ SB["S12"],
            "S21": SB["S21"] @ F @ SA["S21"],
            "S22": SB["S22"] + SB["S21"] @ F @ SA["S22"] @ SB["S12"],
        }
        return S

    @staticmethod
    def identity_smatrix(n: int, dtype=np.complex128) -> dict[str, np.ndarray]:
        """Create identity S-matrix (no scattering)."""
        I = np.eye(n, dtype=dtype)
        Z = np.zeros((n, n), dtype=dtype)
        return {"S11": Z.copy(), "S12": I.copy(), "S21": I.copy(), "S22": Z.copy()}

    @staticmethod
    def layer_smatrix(
        eigenvalues: np.ndarray,
        W: np.ndarray,
        V: np.ndarray,
        thickness: float,
        k0: float,
    ) -> dict[str, np.ndarray]:
        """Compute single layer S-matrix.

        Uses exp(-j*λ*k0*d) which is bounded for all eigenvalues:
        - Propagating modes: |exp| = 1
        - Evanescent modes: |exp| < 1

        Args:
            eigenvalues: Eigenvalues of the layer.
            W: Eigenvector matrix for E-field.
            V: Eigenvector matrix for H-field.
            thickness: Layer thickness in um.
            k0: Free-space wavenumber.

        Returns:
            Layer S-matrix.
        """
        X = np.diag(np.exp(-1j * eigenvalues * k0 * thickness))
        n = len(eigenvalues)
        I = np.eye(n, dtype=W.dtype)

        # Interface matrices
        A = np.linalg.solve(W, I) + np.linalg.solve(V, I)
        B = np.linalg.solve(W, I) - np.linalg.solve(V, I)

        A_inv = np.linalg.inv(A)
        D = A - X @ B @ A_inv @ X @ B

        S11 = np.linalg.solve(D, X @ B @ A_inv @ X @ A - B)
        S12 = np.linalg.solve(D, X @ (A - B @ A_inv @ B))
        S: dict[str, np.ndarray] = {
            "S11": S11,
            "S12": S12,
            "S21": S12,  # Reciprocity
            "S22": S11,
        }
        return S


class LiFactorization:
    """Li's Fourier factorization rules for improved convergence.

    Applies inverse rule at discontinuous permittivity boundaries
    to dramatically improve TM convergence.
    """

    @staticmethod
    def toeplitz_from_fft(f: np.ndarray, n_harmonics: int) -> np.ndarray:
        """Build Toeplitz matrix from FFT coefficients of a 1D function.

        Args:
            f: 1D array of function values.
            n_harmonics: Number of harmonics (matrix will be (2N+1)x(2N+1)).

        Returns:
            Toeplitz matrix of Fourier coefficients.
        """
        N = len(f)
        coeffs = np.fft.fft(f) / N

        size = 2 * n_harmonics + 1
        T = np.zeros((size, size), dtype=complex)

        for i in range(size):
            for j in range(size):
                idx = (i - j) % N
                T[i, j] = coeffs[idx]

        return T

    @staticmethod
    def convolution_matrix_naive(
        eps_grid: np.ndarray,
        n_harmonics: int,
    ) -> np.ndarray:
        """Standard Laurent rule convolution matrix (naive factorization).

        Direct Fourier transform of epsilon. Works well for continuous
        permittivity but converges slowly at discontinuities.
        """
        return LiFactorization.toeplitz_from_fft(eps_grid, n_harmonics)

    @staticmethod
    def convolution_matrix_inverse_rule(
        eps_grid: np.ndarray,
        n_harmonics: int,
    ) -> np.ndarray:
        """Li's inverse rule convolution matrix.

        Uses inv(FT(1/eps)) instead of FT(eps) at material boundaries.
        Critical for TM polarization convergence with metal structures.

        Args:
            eps_grid: 1D permittivity profile.
            n_harmonics: Number of Fourier harmonics.

        Returns:
            Convolution matrix with inverse rule applied.
        """
        eps_safe = np.where(np.abs(eps_grid) > 1e-30, eps_grid, 1e-30)
        eps_inv = 1.0 / eps_safe
        convmat_inv = LiFactorization.toeplitz_from_fft(eps_inv, n_harmonics)
        return np.linalg.inv(convmat_inv)

    @staticmethod
    def convolution_matrix_2d(
        eps_grid_2d: np.ndarray,
        nx_harmonics: int,
        ny_harmonics: int,
        use_inverse_rule: bool = True,
    ) -> np.ndarray:
        """2D convolution matrix for crossed gratings.

        Args:
            eps_grid_2d: 2D permittivity array (ny, nx).
            nx_harmonics: Number of harmonics in x.
            ny_harmonics: Number of harmonics in y.
            use_inverse_rule: Whether to apply Li's inverse rule.

        Returns:
            2D convolution matrix.
        """
        ny, nx = eps_grid_2d.shape
        Nx = 2 * nx_harmonics + 1
        Ny = 2 * ny_harmonics + 1

        if use_inverse_rule:
            # Apply inverse rule row by row, then column by column
            eps_safe_2d = np.where(np.abs(eps_grid_2d) > 1e-30, eps_grid_2d, 1e-30)
            eps_inv = 1.0 / eps_safe_2d

            # 2D FFT approach
            coeffs = np.fft.fft2(eps_inv) / (nx * ny)

            size = Nx * Ny
            T = np.zeros((size, size), dtype=complex)

            for i in range(size):
                pi, qi = divmod(i, Nx)
                for j in range(size):
                    pj, qj = divmod(j, Nx)
                    dp = (pi - pj) % ny
                    dq = (qi - qj) % nx
                    T[i, j] = coeffs[dp, dq]

            return np.linalg.inv(T)
        else:
            # Naive: direct FFT of eps
            coeffs = np.fft.fft2(eps_grid_2d) / (nx * ny)

            size = Nx * Ny
            T = np.zeros((size, size), dtype=complex)

            for i in range(size):
                pi, qi = divmod(i, Nx)
                for j in range(size):
                    pj, qj = divmod(j, Nx)
                    dp = (pi - pj) % ny
                    dq = (qi - qj) % nx
                    T[i, j] = coeffs[dp, dq]

            return T


class EigenvalueStabilizer:
    """Eigenvalue post-processing and validation."""

    @staticmethod
    def fix_degenerate_eigenvalues(
        eigenvalues: np.ndarray,
        eigenvectors: np.ndarray,
        broadening: float = 1e-10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Handle near-degenerate eigenvalues.

        Detects eigenvalue pairs that are too close and applies
        Gram-Schmidt orthogonalization to stabilize eigenvectors.

        Args:
            eigenvalues: Complex eigenvalue array.
            eigenvectors: Complex eigenvector matrix.
            broadening: Threshold for detecting degeneracy.

        Returns:
            Stabilized (eigenvalues, eigenvectors).
        """
        n = len(eigenvalues)

        # Detect degenerate pairs
        for i in range(n):
            for j in range(i + 1, n):
                if abs(eigenvalues[i] - eigenvalues[j]) < broadening:
                    # Gram-Schmidt orthogonalization
                    vi = eigenvectors[:, i].copy()
                    vj = eigenvectors[:, j].copy()

                    # Orthogonalize vj against vi
                    proj = np.dot(np.conj(vi), vj) / np.dot(np.conj(vi), vi)
                    vj -= proj * vi

                    # Normalize
                    norm_j = np.linalg.norm(vj)
                    if norm_j > 1e-15:
                        eigenvectors[:, j] = vj / norm_j

        return eigenvalues, eigenvectors

    @staticmethod
    def select_propagation_direction(eigenvalues: np.ndarray) -> np.ndarray:
        """Select correct branch of sqrt(eigenvalues).

        Rules (time convention e^{-iwt}):
        - Re(sqrt_λ) > 0: forward propagating
        - Re(sqrt_λ) ≈ 0 and Im(sqrt_λ) > 0: evanescent (correct decay)

        Args:
            eigenvalues: Raw eigenvalues from eigendecomp.

        Returns:
            Square roots with correct sign.
        """
        sqrt_ev = np.sqrt(eigenvalues.astype(np.complex128))

        # Fix sign: want Im(sqrt_ev) <= 0 for exp(-i*k*z) convention
        # or Re(sqrt_ev) >= 0 for propagating modes
        wrong_sign = np.real(sqrt_ev) < 0
        near_zero_real = np.abs(np.real(sqrt_ev)) < 1e-10
        wrong_evanescent = near_zero_real & (np.imag(sqrt_ev) < 0)

        flip_mask = wrong_sign | wrong_evanescent
        sqrt_ev[flip_mask] *= -1

        return sqrt_ev

    @staticmethod
    def validate_energy_conservation(
        R: np.ndarray,
        T: np.ndarray,
        tolerance: float = 0.05,
    ) -> dict:
        """Validate energy conservation from R, T values.

        Args:
            R: Reflection coefficients.
            T: Transmission coefficients.
            tolerance: Acceptable deviation from 1.0 for non-absorbing case.

        Returns:
            Validation result dict.
        """
        total = np.asarray(R) + np.asarray(T)
        max_violation = float(np.max(np.abs(total - 1.0)))

        # For absorbing media, R+T < 1 is expected
        over_unity = total > 1.0 + tolerance

        return {
            "valid": not np.any(over_unity),
            "max_violation": max_violation,
            "violation_indices": np.where(over_unity)[0].tolist() if np.any(over_unity) else [],
        }


class AdaptivePrecisionRunner:
    """Adaptive precision fallback for unstable simulations.

    Strategy: GPU-f32 → GPU-f64-cuSOLVER → GPU-f64 → CPU-f64 (4-level fallback).
    When CuPy is available, cuSOLVER eigendecomposition on GPU is attempted
    before falling back to CPU, eliminating GPU↔CPU transfer overhead.
    """

    def __init__(self, tolerance: float = 0.02):
        self.tolerance = tolerance

    def run_with_fallback(
        self,
        solver,
        wavelength: float,
        config: dict,
    ) -> dict:
        """Run simulation with automatic precision fallback.

        Args:
            solver: Solver instance with run_single method.
            wavelength: Wavelength in um.
            config: Solver config.

        Returns:
            Simulation result dict.
        """
        strategies = [
            {"dtype": "complex64", "device": "cuda", "label": "GPU-f32"},
            {"dtype": "complex128", "device": "cuda", "label": "GPU-f64-cuSOLVER"},
            {"dtype": "complex128", "device": "cuda", "label": "GPU-f64"},
            {"dtype": "complex128", "device": "cpu", "label": "CPU-f64"},
        ]

        for strategy in strategies:
            try:
                result = solver.run_single_wavelength(
                    wavelength=wavelength,
                    dtype=strategy["dtype"],
                    device=strategy["device"],
                )

                # Validate
                R = result.get("R", 0)
                T = result.get("T", 0)
                check = EigenvalueStabilizer.validate_energy_conservation(
                    np.array([R]), np.array([T]), self.tolerance,
                )

                if check["valid"]:
                    if strategy["label"] != "GPU-f32":
                        logger.warning(
                            f"λ={wavelength:.3f}um: fallback to {strategy['label']}"
                        )
                    return dict(result)
                else:
                    logger.warning(
                        f"λ={wavelength:.3f}um: energy violation "
                        f"{check['max_violation']:.4f} with {strategy['label']}"
                    )

            except Exception as e:
                logger.warning(
                    f"λ={wavelength:.3f}um: {strategy['label']} failed: {e}"
                )
                continue

        raise RuntimeError(
            f"All precision strategies failed for λ={wavelength:.3f}um. "
            f"Consider reducing Fourier order."
        )


class StabilityDiagnostics:
    """Pre/post simulation stability diagnostics."""

    @staticmethod
    def pre_simulation_check(
        pixel_stack: PixelStack,
        solver_config: dict,
    ) -> list[str]:
        """Check for potential stability issues before simulation.

        Args:
            pixel_stack: Pixel stack structure.
            solver_config: Solver configuration.

        Returns:
            List of warning messages.
        """
        warnings = []

        # 1. Check Fourier order vs precision
        params = solver_config.get("params", {})
        order = params.get("fourier_order", [9, 9])
        matrix_size = (2 * order[0] + 1) * (2 * order[1] + 1)

        stability = solver_config.get("stability", {})
        strategy = stability.get("precision_strategy", "mixed")

        if matrix_size > 200 and strategy == "float32":
            warnings.append(
                f"Large matrix ({matrix_size}x{matrix_size}) with float32. "
                f"High risk of eigendecomp instability. "
                f"Recommend precision_strategy: 'mixed'"
            )

        # 2. Check for thick layers
        for layer in pixel_stack.layers:
            if layer.thickness > 2.0:
                warnings.append(
                    f"Thick layer '{layer.name}' ({layer.thickness}um). "
                    f"Ensure S-matrix algorithm is used."
                )

        # 3. TF32 check
        if HAS_TORCH and torch.backends.cuda.matmul.allow_tf32:
            warnings.append(
                "TF32 is ENABLED. Set allow_tf32: false for RCWA stability."
            )

        # 4. Check Fourier factorization for patterned layers
        has_patterned = any(l.is_patterned for l in pixel_stack.layers)
        fact = stability.get("fourier_factorization", "naive")
        if has_patterned and fact == "naive":
            warnings.append(
                "Patterned layers detected with naive factorization. "
                "Recommend fourier_factorization: 'li_inverse' for better convergence."
            )

        for w in warnings:
            logger.warning(f"Stability check: {w}")

        return warnings

    @staticmethod
    def post_simulation_check(result: SimulationResult) -> dict:
        """Validate simulation results for physical plausibility.

        Args:
            result: Completed simulation result.

        Returns:
            Validation report dict.
        """
        report = {}

        # QE range check
        for pixel_name, qe in result.qe_per_pixel.items():
            qe_arr = np.asarray(qe)
            if np.any(qe_arr < -0.01) or np.any(qe_arr > 1.01):
                report[pixel_name] = {
                    "status": "FAILED",
                    "issue": f"QE out of range [{qe_arr.min():.4f}, {qe_arr.max():.4f}]",
                }

        # NaN/Inf check
        for name in ["reflection", "transmission", "absorption"]:
            arr = getattr(result, name, None)
            if arr is not None:
                arr = np.asarray(arr)
                if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                    report[name] = {"status": "FAILED", "issue": "NaN or Inf detected"}

        # Energy conservation
        if result.reflection is not None and result.transmission is not None:
            total = np.asarray(result.reflection) + np.asarray(result.transmission)
            if result.absorption is not None:
                total += np.asarray(result.absorption)
            max_viol = float(np.max(np.abs(total - 1.0)))
            if max_viol > 0.05:
                report["energy_balance"] = {
                    "status": "WARNING",
                    "issue": f"Max |R+T+A-1| = {max_viol:.4f}",
                }

        return report
