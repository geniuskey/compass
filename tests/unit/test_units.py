"""Unit tests for units module."""

import numpy as np
import pytest

from compass.core import units


class TestUnitConversions:
    """Tests for unit conversion functions."""

    def test_um_to_nm(self):
        assert units.um_to_nm(1.0) == 1000.0
        assert units.um_to_nm(0.55) == 550.0

    def test_nm_to_um(self):
        assert units.nm_to_um(550.0) == 0.55

    def test_um_nm_roundtrip(self):
        assert units.nm_to_um(units.um_to_nm(0.55)) == pytest.approx(0.55)

    def test_um_to_m(self):
        assert units.um_to_m(1.0) == 1e-6

    def test_m_to_um(self):
        assert units.m_to_um(1e-6) == 1.0

    def test_wavelength_to_frequency(self):
        freq = units.wavelength_to_frequency(0.55)
        # c/lambda = 2.998e14 / 0.55 ≈ 5.45e14 Hz
        assert 5e14 < freq < 6e14

    def test_frequency_to_wavelength_roundtrip(self):
        wl = 0.55
        freq = units.wavelength_to_frequency(wl)
        wl_back = units.frequency_to_wavelength(freq)
        assert wl_back == pytest.approx(wl)

    def test_eV_to_um(self):
        # 1 eV ≈ 1.24 um
        wl = units.eV_to_um(1.0)
        assert 1.2 < wl < 1.3

    def test_deg_to_rad(self):
        assert units.deg_to_rad(0.0) == 0.0
        assert units.deg_to_rad(90.0) == pytest.approx(np.pi / 2)
        assert units.deg_to_rad(180.0) == pytest.approx(np.pi)

    def test_rad_to_deg(self):
        assert units.rad_to_deg(np.pi) == pytest.approx(180.0)

    def test_wavelength_to_k0(self):
        k0 = units.wavelength_to_k0(0.55)
        # k0 = 2*pi/lambda
        assert k0 == pytest.approx(2 * np.pi / 0.55)
