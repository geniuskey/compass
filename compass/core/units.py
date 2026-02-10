"""Unit system and coordinate conventions for COMPASS.

Coordinate Convention:
    - x, y: lateral directions (in-plane)
    - z: stack direction (vertical)
    - Light propagates in -z direction (from air at z_max toward silicon at z_min)
    - Air = top = z_max, Silicon = bottom = z_min

Internal Units:
    - Length: micrometers (um)
    - Wavelength: micrometers (um)
    - Angle: degrees (external API), radians (internal computation)
    - Frequency: c / lambda (Hz)
    - Permittivity: dimensionless complex (epsilon = (n + ik)^2)

Physical Constants:
    - c = 2.998e14 um/s (speed of light)
    - h = 6.626e-34 J*s (Planck constant)
"""

from __future__ import annotations

import numpy as np

# Speed of light in um/s
C_UM_PER_S = 2.99792458e14

# Planck constant in J*s
H_JS = 6.62607015e-34

# Electron charge in C
Q_C = 1.602176634e-19


def um_to_nm(um: float) -> float:
    """Convert micrometers to nanometers."""
    return um * 1000.0


def nm_to_um(nm: float) -> float:
    """Convert nanometers to micrometers."""
    return nm / 1000.0


def um_to_m(um: float) -> float:
    """Convert micrometers to meters."""
    return um * 1e-6


def m_to_um(m: float) -> float:
    """Convert meters to micrometers."""
    return m * 1e6


def wavelength_to_frequency(wavelength_um: float) -> float:
    """Convert wavelength (um) to frequency (Hz)."""
    return C_UM_PER_S / wavelength_um


def frequency_to_wavelength(frequency_hz: float) -> float:
    """Convert frequency (Hz) to wavelength (um)."""
    return C_UM_PER_S / frequency_hz


def eV_to_um(eV: float) -> float:
    """Convert photon energy (eV) to wavelength (um)."""
    # λ(um) = h(J·s) * c(um/s) / (E(eV) * q(C))
    return H_JS * C_UM_PER_S / (eV * Q_C)


def um_to_eV(wavelength_um: float) -> float:
    """Convert wavelength (um) to photon energy (eV)."""
    return H_JS * C_UM_PER_S / (wavelength_um * Q_C)


def deg_to_rad(deg: float) -> float:
    """Convert degrees to radians."""
    return np.deg2rad(deg)


def rad_to_deg(rad: float) -> float:
    """Convert radians to degrees."""
    return np.rad2deg(rad)


def wavelength_to_k0(wavelength_um: float) -> float:
    """Convert wavelength (um) to free-space wavenumber k0 (1/um)."""
    return 2.0 * np.pi / wavelength_um
