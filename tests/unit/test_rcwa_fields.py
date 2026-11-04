"""Unit tests for the solver-independent RCWA field helpers (no backend needed)."""

from types import SimpleNamespace

import numpy as np
import pytest

from compass.core.types import LayerSlice, PhotodiodeSpec
from compass.solvers.rcwa.fields import (
    approximate_field,
    eps_imag_weight_qe,
    integrate_photodiode_qe,
)


def _stack(n_cols=2):
    return SimpleNamespace(
        domain_size=(float(n_cols), 1.0),
        pitch=1.0,
        unit_cell=(1, n_cols),
        bayer_map=[["R", "G"][:n_cols]],
        layers=[SimpleNamespace(name="silicon", z_end=1.0)],
        photodiodes=[
            PhotodiodeSpec((0.0, 0.0, 0.5), (1.0, 1.0, 1.0), (0, c), "RG"[c])
            for c in range(n_cols)
        ],
    )


def _slices():
    lossy = np.full((4, 8), 1.0 + 0j)
    lossy[:, :4] = 16.0 + 1.0j  # absorbing under the left pixel only
    return [
        LayerSlice(1.0, 2.0, 1.0, np.full((4, 8), 2.25 + 0j), "top", "glass"),
        LayerSlice(0.0, 1.0, 1.0, lossy, "silicon", "silicon"),
    ]


class TestApproximateField:
    def test_xz_uses_domain_coordinates_and_decays_from_top(self):
        out = approximate_field(_stack(), _slices(), 0.55, "|E|2", "xz", 0.5)
        assert out.shape == (64, 2)
        # column 0 is the bottom (silicon), column 1 the top (lossless glass)
        np.testing.assert_allclose(out[:, 1], 1.0)
        left, right = out[:16, 0], out[48:, 0]
        kappa = np.imag(np.sqrt(16.0 + 1.0j))
        np.testing.assert_allclose(left, np.exp(-2 * (2 * np.pi / 0.55) * kappa * 1.0))
        np.testing.assert_allclose(right, 1.0)

    def test_xz_other_components_return_extinction(self):
        out = approximate_field(_stack(), _slices(), 0.55, "Ex", "xz", 0.5)
        np.testing.assert_allclose(out[0, 0], np.imag(np.sqrt(16.0 + 1.0j)))

    def test_xy_picks_slice_by_height(self):
        out = approximate_field(_stack(), _slices(), 0.55, "|E|2", "xy", 1.5)
        np.testing.assert_allclose(out, 2.25)
        out = approximate_field(_stack(), _slices(), 0.55, "|E|2", "xy", 0.5)
        assert out.shape == (64, 64)
        assert out[0, 0] == pytest.approx(16.0) and out[-1, 0] == pytest.approx(1.0)

    def test_rejects_unknown_plane(self):
        with pytest.raises(ValueError):
            approximate_field(_stack(), _slices(), 0.55, "|E|2", "zz", 0.0)


class TestEpsImagWeightQE:
    def test_absorption_goes_to_absorbing_pixel(self):
        qe = eps_imag_weight_qe(_stack(), _slices(), 0.4)
        assert qe["R_0_0"] == pytest.approx(0.8)  # 0.4 * all weight * 2 pixels
        assert qe["G_0_1"] == 0.0

    def test_weights_scale_with_photodiode_area(self):
        stack = _stack()
        stack.photodiodes[1] = PhotodiodeSpec((0.0, 0.0, 0.5), (0.5, 1.0, 1.0), (0, 1), "G")
        slices = [LayerSlice(0.0, 1.0, 1.0, np.full((4, 8), 16.0 + 1.0j), "silicon")]
        qe = eps_imag_weight_qe(stack, slices, 0.3)
        assert qe["R_0_0"] == pytest.approx(2 * qe["G_0_1"])

    def test_lossless_stack_reports_cell_average(self):
        slices = [LayerSlice(0.0, 1.0, 1.0, np.full((4, 8), 2.0 + 0j), "silicon")]
        assert eps_imag_weight_qe(_stack(), slices, 0.25) == {"R_0_0": 0.25, "G_0_1": 0.25}


class TestSliceAbsorption:
    def test_density_is_rescaled_to_exact_slice_absorption(self):
        stack = _stack()
        slices = _slices()[::-1][::-1]  # backend order: top first
        e2 = np.ones((4, 8))

        def raw(_li, _s):
            return e2

        k0 = 2 * np.pi / 0.55
        unscaled = integrate_photodiode_qe(stack, slices, 0.55, 0.0, raw, 0.5, "t")
        # analytic: k0 * eps'' * |E|^2 * dz over the left pixel, per pixel area
        assert unscaled["R_0_0"] == pytest.approx(min(1.0, k0 * 1.0))
        pinned = integrate_photodiode_qe(
            stack, slices, 0.55, 0.0, raw, 0.5, "t", slice_absorption=lambda _li, _s: 0.2
        )
        # all of the 0.2 cell absorption sits under the left pixel (half the cell)
        assert pinned["R_0_0"] == pytest.approx(0.4)
        assert pinned["G_0_1"] == pytest.approx(0.0)

    def test_negative_flux_noise_is_clamped(self):
        pinned = integrate_photodiode_qe(
            _stack(), _slices(), 0.55, 0.0, lambda _li, _s: np.ones((4, 8)), 0.0, "t",
            slice_absorption=lambda _li, _s: -1e-9,
        )
        assert pinned == {"R_0_0": 0.0, "G_0_1": 0.0}
