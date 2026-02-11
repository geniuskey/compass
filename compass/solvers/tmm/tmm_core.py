"""Transfer Matrix Method (TMM) for 1D planar multilayer stacks.

Pure numpy implementation of the 2x2 transfer matrix method for computing
reflection, transmission, and absorption spectra of thin-film stacks.
No external dependencies beyond numpy.

Uses the characteristic matrix approach (also known as the "transfer matrix"
or "Abeles matrix" method) where each layer contributes a 2x2 matrix that
relates the tangential field components at the two boundaries of the layer.

References:
    - Hecht, "Optics" (5th ed.), Chapters 9 & 11
    - Born & Wolf, "Principles of Optics", Chapter 1
    - Centurioni, "Generalized matrix method for calculation of internal
      light energy flux in mixed coherent and incoherent multilayers",
      Applied Optics 44(35), 2005
    - Byrnes, "Multilayer optical calculations" (tmm package documentation)
"""

from __future__ import annotations

import numpy as np


def _snell_cos_theta(
    n_layers: np.ndarray,
    theta_inc: float,
) -> np.ndarray:
    """Compute cos(theta) in each layer via Snell's law.

    n_0 * sin(theta_0) = n_j * sin(theta_j) for all j.

    Args:
        n_layers: Complex refractive indices array [n_0, n_1, ..., n_N].
        theta_inc: Incidence angle in radians (real).

    Returns:
        Array of complex cos(theta) in each layer.
    """
    n0_sin_theta0 = n_layers[0] * np.sin(theta_inc)
    sin_theta = n0_sin_theta0 / n_layers
    cos_theta = np.sqrt(1.0 - sin_theta**2 + 0j)
    # Choose correct branch: Re(cos_theta) >= 0 for forward propagation
    cos_theta = np.where(np.real(cos_theta) < 0, -cos_theta, cos_theta)
    # For evanescent waves where Re(cos)==0, ensure Im(cos) > 0
    cos_theta = np.where(
        (np.real(cos_theta) == 0) & (np.imag(cos_theta) < 0),
        -cos_theta,
        cos_theta,
    )
    return cos_theta


def _eta(n: complex, cos_theta: complex, polarization: str) -> complex:
    """Compute the effective admittance (eta) for a given polarization.

    For TE: eta = n * cos(theta)
    For TM: eta = n / cos(theta)

    These quantities are the "tilted optical admittances" used in the
    characteristic matrix formulation, ensuring that the Poynting vector
    formulation gives correct R, T, A including absorbing media.

    Args:
        n: Complex refractive index.
        cos_theta: Complex cosine of propagation angle.
        polarization: "TE" or "TM".

    Returns:
        Effective admittance.
    """
    if polarization == "TE":
        return n * cos_theta
    else:  # TM
        return n / cos_theta


def transfer_matrix_1d(
    n_layers: np.ndarray,
    d_layers: np.ndarray,
    wavelength: float,
    theta_inc: float = 0.0,
    polarization: str = "TE",
) -> tuple[float, float, float]:
    """Compute reflection, transmission, and absorption for a planar multilayer stack.

    Uses the characteristic matrix (transfer matrix) method. The first and last
    entries in n_layers/d_layers correspond to the semi-infinite incident and
    substrate media, respectively. Their thicknesses are ignored.

    The characteristic matrix for each film layer j is:

        M_j = [[cos(delta_j),        -i*sin(delta_j)/eta_j],
               [-i*eta_j*sin(delta_j), cos(delta_j)       ]]

    where delta_j = (2*pi/lambda) * n_j * cos(theta_j) * d_j
    and eta_j is the tilted admittance.

    The total system matrix M = M_1 * M_2 * ... * M_{N-1} relates the
    tangential field components at the first and last interfaces.

    Then:
        r = (eta_0 * M[0,0] + eta_0*eta_s * M[0,1] - M[1,0] - eta_s * M[1,1]) /
            (eta_0 * M[0,0] + eta_0*eta_s * M[0,1] + M[1,0] + eta_s * M[1,1])
        t = 2 * eta_0 /
            (eta_0 * M[0,0] + eta_0*eta_s * M[0,1] + M[1,0] + eta_s * M[1,1])

    R = |r|^2
    T = |t|^2 * Re(eta_s) / Re(eta_0)
    A = 1 - R - T

    Args:
        n_layers: Complex refractive indices [n_inc, n_1, n_2, ..., n_sub].
            Length N >= 2.
        d_layers: Layer thicknesses in um [inf, d_1, d_2, ..., inf].
            Length N, same as n_layers. First and last are semi-infinite.
        wavelength: Free-space wavelength in um (must be > 0).
        theta_inc: Incidence angle in radians (0 = normal incidence).
        polarization: "TE" or "TM".

    Returns:
        Tuple (R, T, A) where R + T + A = 1 (within numerical precision).
        R = reflectance, T = transmittance, A = absorbance.
    """
    if polarization not in ("TE", "TM"):
        raise ValueError(f"polarization must be 'TE' or 'TM', got '{polarization}'")

    n_layers = np.asarray(n_layers, dtype=complex)
    d_layers = np.asarray(d_layers, dtype=float)

    if len(n_layers) < 2:
        raise ValueError("Need at least 2 layers (incident + substrate)")
    if len(n_layers) != len(d_layers):
        raise ValueError("n_layers and d_layers must have the same length")
    if wavelength <= 0:
        raise ValueError("wavelength must be positive")

    n_total = len(n_layers)
    k0 = 2.0 * np.pi / wavelength

    # Compute cos(theta) in each layer via Snell's law
    cos_theta = _snell_cos_theta(n_layers, theta_inc)

    # Compute effective admittances
    eta_0 = _eta(n_layers[0], cos_theta[0], polarization)
    eta_s = _eta(n_layers[-1], cos_theta[-1], polarization)

    # Build the total characteristic matrix M = product of layer matrices
    # Only for the internal layers (indices 1 to n_total-2)
    M = np.eye(2, dtype=complex)

    for j in range(1, n_total - 1):
        d_j = d_layers[j]
        delta_j = k0 * n_layers[j] * cos_theta[j] * d_j
        eta_j = _eta(n_layers[j], cos_theta[j], polarization)

        cos_d = np.cos(delta_j)
        sin_d = np.sin(delta_j)

        # Characteristic matrix for layer j
        M_j = np.array(
            [
                [cos_d, -1j * sin_d / eta_j],
                [-1j * eta_j * sin_d, cos_d],
            ],
            dtype=complex,
        )
        M = M @ M_j

    # Extract r and t from the system matrix
    # Using boundary conditions at first and last interfaces
    denom = (
        eta_0 * M[0, 0]
        + eta_0 * eta_s * M[0, 1]
        + M[1, 0]
        + eta_s * M[1, 1]
    )

    r = (
        eta_0 * M[0, 0]
        + eta_0 * eta_s * M[0, 1]
        - M[1, 0]
        - eta_s * M[1, 1]
    ) / denom

    t = 2.0 * eta_0 / denom

    R = float(np.abs(r) ** 2)
    T = float(np.abs(t) ** 2 * np.real(eta_s) / np.real(eta_0))
    A = 1.0 - R - T

    # Clamp small negative values from floating-point noise
    if A < 0 and A > -1e-10:
        A = 0.0

    return R, T, A


def tmm_spectrum(
    n_layers_func: callable,
    d_layers: np.ndarray,
    wavelengths: np.ndarray,
    theta_inc: float = 0.0,
    polarization: str = "TE",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute R, T, A spectra over multiple wavelengths.

    Args:
        n_layers_func: Callable(wavelength) -> np.ndarray of complex refractive
            indices for each layer at that wavelength.
        d_layers: Layer thicknesses in um (same for all wavelengths).
        wavelengths: Array of wavelengths in um.
        theta_inc: Incidence angle in radians.
        polarization: "TE" or "TM".

    Returns:
        Tuple of (R_spectrum, T_spectrum, A_spectrum), each shape (len(wavelengths),).
    """
    wavelengths = np.asarray(wavelengths)
    n_wl = len(wavelengths)
    R_arr = np.zeros(n_wl)
    T_arr = np.zeros(n_wl)
    A_arr = np.zeros(n_wl)

    for i, wl in enumerate(wavelengths):
        n_layers = n_layers_func(wl)
        R_arr[i], T_arr[i], A_arr[i] = transfer_matrix_1d(
            n_layers, d_layers, wl, theta_inc, polarization,
        )

    return R_arr, T_arr, A_arr


def tmm_field_profile(
    n_layers: np.ndarray,
    d_layers: np.ndarray,
    wavelength: float,
    theta_inc: float,
    polarization: str,
    z_points: np.ndarray,
) -> np.ndarray:
    """Compute |E|^2 field intensity profile through the stack.

    Uses a numerically stable approach based on the S-matrix (scattering
    matrix) formulation to avoid exponential overflow in absorbing layers.
    For each z-point inside layer j, the forward and backward wave amplitudes
    are computed from both sides (incident and substrate) and matched.

    The z-axis is defined with z=0 at the first interface (between the
    incident medium and the first film layer). Positive z goes into the stack
    toward the substrate.

    Args:
        n_layers: Complex refractive indices [n_inc, n_1, ..., n_sub].
        d_layers: Layer thicknesses in um [inf, d1, ..., inf].
        wavelength: Wavelength in um.
        theta_inc: Incidence angle in radians.
        polarization: "TE" or "TM".
        z_points: Array of z positions (um) at which to evaluate |E|^2.
            z=0 is at the first interface.

    Returns:
        Array of |E|^2 values at each z_point, normalized to incident intensity.
    """
    if polarization not in ("TE", "TM"):
        raise ValueError(f"polarization must be 'TE' or 'TM', got '{polarization}'")

    n_layers = np.asarray(n_layers, dtype=complex)
    d_layers = np.asarray(d_layers, dtype=float)
    z_points = np.asarray(z_points, dtype=float)

    n_total = len(n_layers)
    k0 = 2.0 * np.pi / wavelength
    cos_theta = _snell_cos_theta(n_layers, theta_inc)

    # Compute admittances
    eta_vals = np.array([
        _eta(n_layers[j], cos_theta[j], polarization) for j in range(n_total)
    ])
    # Interface positions: z=0 at first interface
    interface_z = np.zeros(n_total)
    for j in range(2, n_total):
        interface_z[j] = interface_z[j - 1] + d_layers[j - 1]

    # Compute kz for each layer
    kz = k0 * n_layers * cos_theta

    # --- Fresnel coefficients at each interface ---
    r_intf = np.zeros(n_total - 1, dtype=complex)
    t_intf = np.zeros(n_total - 1, dtype=complex)
    for j in range(n_total - 1):
        r_intf[j] = (eta_vals[j] - eta_vals[j + 1]) / (eta_vals[j] + eta_vals[j + 1])
        t_intf[j] = 2.0 * eta_vals[j] / (eta_vals[j] + eta_vals[j + 1])

    # --- Numerically stable S-matrix (scattering matrix) approach ---
    # For each internal layer j, compute the reflection coefficient r_j
    # "looking right" from the left boundary of layer j toward the substrate.
    # This is done by a recursion from the substrate backward.
    #
    # At the last interface (n_total-2 -> n_total-1):
    #   r_right[n_total-2] = r_{N-2,N-1}
    #
    # For layer j, the reflection looking right from the left of layer j:
    #   r_right_after = r_{j,j+1} combined with propagation through j+1 and r_right[j+1]
    #   Specifically, from the right side of interface j:
    #     r_prime = r_{j,j+1} + t_{j,j+1} * r_next * exp(2i*kz_{j+1}*d_{j+1}) * t_{j+1,j}
    #              / (1 - r_{j+1,j} * r_next * exp(2i*kz_{j+1}*d_{j+1}))
    #
    # Actually, we use a simpler approach: for each layer j (internal),
    # compute the local reflection coefficient from the RIGHT side of layer j
    # looking rightward (toward substrate). This uses the standard Airy formula.

    # r_right[j] = effective Fresnel reflection looking right from the
    # right boundary of layer j (j = 0..n_total-2)
    # We build from the substrate side backward.
    r_right = np.zeros(n_total, dtype=complex)
    # At the last interface: r_right[n_total-2] = r_{n_total-2, n_total-1}
    # Looking from layer n_total-2 into substrate
    r_right[n_total - 2] = (eta_vals[n_total - 2] - eta_vals[n_total - 1]) / (
        eta_vals[n_total - 2] + eta_vals[n_total - 1]
    )

    for j in range(n_total - 3, -1, -1):
        # r_right[j] looks from layer j into the rest of the stack to the right
        # First: effective r at the interface from j to j+1, combined with
        # propagation through layer j+1 and r_right[j+1]
        r_jk = (eta_vals[j] - eta_vals[j + 1]) / (eta_vals[j] + eta_vals[j + 1])
        if j + 1 < n_total - 1:
            # Layer j+1 is an internal layer with finite thickness
            phase = np.exp(2j * kz[j + 1] * d_layers[j + 1])
            r_next = r_right[j + 1]
            # Airy formula for combining interface with slab
            r_right[j] = r_jk + (1 - r_jk**2) * r_next * phase / (
                1 - r_jk * r_next * phase  # note: r_jk here is r_{j+1,j} = -r_jk
            )
            # Correction: r_{j+1,j} = -r_jk, so:
            r_right[j] = (r_jk + r_next * phase) / (1 + r_jk * r_next * phase)
        else:
            r_right[j] = r_jk

    # System reflection coefficient (looking from incident medium)
    r_sys = r_right[0]

    # --- Compute the transmission coefficient into each layer ---
    # t_into[j] = amplitude of the forward wave at the LEFT boundary of layer j,
    # normalized to incident amplitude = 1.
    # For the incident medium: t_into[0] = 1 (forward) and r_sys (backward)
    # For layer j+1: t_into[j+1] = t_into[j] * t_{j,j+1} / (1 + r_{j,j+1} * r_right[j+1] * phase)
    # where phase accounts for propagation through layer j (if j is internal).

    t_into = np.zeros(n_total, dtype=complex)
    t_into[0] = 1.0  # incident amplitude

    for j in range(n_total - 1):
        r_jk = (eta_vals[j] - eta_vals[j + 1]) / (eta_vals[j] + eta_vals[j + 1])
        t_jk = 2.0 * eta_vals[j] / (eta_vals[j] + eta_vals[j + 1])

        if j == 0:
            # Propagation through incident medium is trivial (semi-infinite)
            # At the first interface, the forward amplitude arriving is t_into[0] = 1
            # Transmitted into layer 1:
            if j + 1 < n_total - 1:
                r_next_phase = r_right[j + 1] * np.exp(2j * kz[j + 1] * d_layers[j + 1])
            elif j + 1 == n_total - 1:
                r_next_phase = 0.0  # substrate, no reflection from right
            else:
                r_next_phase = 0.0
            t_into[j + 1] = t_into[j] * t_jk / (1 + r_jk * r_next_phase)
        else:
            # Layer j is internal: propagate through it first
            phase_j = np.exp(1j * kz[j] * d_layers[j])
            # The forward amplitude at the right boundary of layer j
            # (after propagation through layer j):
            a_right = t_into[j] * phase_j

            if j + 1 < n_total - 1:
                r_next_phase = r_right[j + 1] * np.exp(2j * kz[j + 1] * d_layers[j + 1])
            elif j + 1 == n_total - 1:
                r_next_phase = 0.0
            else:
                r_next_phase = 0.0
            t_into[j + 1] = a_right * t_jk / (1 + r_jk * r_next_phase)

    # Now for each layer j, the field at local position dz from the left boundary:
    # Forward wave: A_fwd = t_into[j]
    # Backward wave: B_bwd at the left boundary due to reflection from the right
    # B_bwd = t_into[j] * r_right[j] * exp(2i*kz_j*d_j) when at left boundary
    # But more carefully: the backward wave at position dz is
    #   B_bwd(dz) = t_into[j] * r_right_from_dz * exp(...)
    # For numerical stability, compute from the right side:
    #   At the right boundary of layer j (dz = d_j):
    #     A_fwd_right = t_into[j] * exp(i*kz_j*d_j)
    #     B_bwd_right = A_fwd_right * r_right[j]  (reflection from right)
    #   At arbitrary dz:
    #     E(dz) = t_into[j] * exp(i*kz_j*dz) + B_bwd_left * exp(-i*kz_j*dz)
    #   where B_bwd_left = t_into[j] * r_right_eff * exp(2i*kz_j*d_j) and
    #   r_right_eff is the effective reflection at the right side of layer j.
    #
    # For numerical stability with absorbing media, compute the backward wave
    # by propagating FROM the right boundary.

    intensity = np.zeros(len(z_points))

    for idx, z in enumerate(z_points):
        if z <= 0:
            # Incident medium: E = exp(ikz*z) + r_sys * exp(-ikz*z)
            E = np.exp(1j * kz[0] * z) + r_sys * np.exp(-1j * kz[0] * z)
            intensity[idx] = float(np.abs(E) ** 2)
            continue

        # Find which layer
        layer_j = n_total - 1
        dz = 0.0
        for j in range(1, n_total - 1):
            layer_left = interface_z[j]
            layer_right = layer_left + d_layers[j]
            if z <= layer_right:
                layer_j = j
                dz = z - layer_left
                break
        else:
            dz = z - (interface_z[n_total - 1] if n_total > 2 else 0.0)

        if layer_j == n_total - 1:
            # Substrate: only forward wave
            E = t_into[layer_j] * np.exp(1j * kz[layer_j] * dz)
            intensity[idx] = float(np.abs(E) ** 2)
            continue

        # Internal layer j: compute field at position dz from left boundary.
        # Forward wave at dz: A_fwd(dz) = t_into[j] * exp(i*kz*dz)
        # Backward wave at dz: the backward wave at the RIGHT boundary of j is
        #   b_right = (forward at right boundary) * r_right[j]
        #           = t_into[j] * exp(i*kz*d_j) * r_right[j]
        # Then b at dz = b_right * exp(-i*kz*(d_j - dz))  [propagate backward]
        #             = t_into[j] * r_right[j] * exp(i*kz*(2*d_j - d_j + dz - d_j))
        # Wait, let me be more careful.
        #
        # Let the forward wave be A*exp(i*kz*x) and backward be B*exp(-i*kz*x)
        # where x is measured from the left boundary of layer j.
        # At x=0 (left boundary): forward = A = t_into[j]
        # At x=d_j (right boundary): forward arriving = A*exp(i*kz*d_j)
        # The backward wave at x=d_j is: B_right = A*exp(i*kz*d_j) * r_right[j]
        # Propagating the backward wave from x=d_j back to x:
        #   B(x) = B_right * exp(-i*kz*(d_j - x)) * ... NO, the backward wave
        #   is B*exp(-i*kz*x). At x=d_j: B*exp(-i*kz*d_j) = B_right
        #   So B = B_right * exp(i*kz*d_j) = A*exp(2i*kz*d_j)*r_right[j]
        #
        # Then: E(x) = A*exp(i*kz*x) + B*exp(-i*kz*x)
        #            = t_into[j]*exp(i*kz*x) + t_into[j]*r_right[j]*exp(2i*kz*d_j)*exp(-i*kz*x)
        #            = t_into[j] * [exp(i*kz*x) + r_right[j]*exp(i*kz*(2*d_j - x))]
        #
        # For numerical stability in absorbing media (Im(kz) > 0):
        # - exp(i*kz*x) has factor exp(-Im(kz)*x) which decays -> OK
        # - exp(i*kz*(2*d_j - x)) has factor exp(-Im(kz)*(2*d_j - x))
        #   which also decays as long as x < 2*d_j -> OK for x in [0, d_j]
        #
        # So this formulation is numerically stable!

        d_j = d_layers[layer_j]
        fwd = np.exp(1j * kz[layer_j] * dz)
        bwd = r_right[layer_j] * np.exp(1j * kz[layer_j] * (2 * d_j - dz))
        E = t_into[layer_j] * (fwd + bwd)
        intensity[idx] = float(np.abs(E) ** 2)

    return intensity
