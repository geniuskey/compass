"""Unit tests for grcwa eigenvalue branch repair."""

import numpy as np

from compass.solvers.rcwa.grcwa_solver import fix_propagation_branch


def test_flipped_propagating_mode_restored():
    # Round-off Im(q) > 0 made grcwa keep Re(q) < 0 (backward wave).
    q = np.array([-5.72 + 6.9e-32j, -10.8 + 9.9e-17j])
    np.testing.assert_allclose(fix_propagation_branch(q), [5.72 - 6.9e-32j, 10.8 - 9.9e-17j])


def test_propagating_mode_with_negative_roundoff_kept_forward():
    q = np.array([3.0 - 1e-17j])
    out = fix_propagation_branch(q)
    assert out[0].real > 0


def test_evanescent_and_lossy_modes_decay():
    q = np.array([-2j, 0.5 - 0.1j, -0.5 - 0.1j, 1.0 + 0.01j])
    out = fix_propagation_branch(q)
    assert np.all(out.imag > 0)
    np.testing.assert_allclose(out, [2j, -0.5 + 0.1j, 0.5 + 0.1j, 1.0 + 0.01j])


def test_squared_eigenvalue_preserved():
    rng = np.random.default_rng(0)
    q = rng.normal(size=50) + 1j * rng.normal(size=50) * rng.choice([1e-18, 1.0], size=50)
    np.testing.assert_allclose(fix_propagation_branch(q) ** 2, q**2)
