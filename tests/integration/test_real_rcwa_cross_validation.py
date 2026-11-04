"""Cross-validate torcwa and grcwa on a patterned Bayer pixel.

The planar-slab TMM comparison cannot catch errors that only appear with
lateral patterning (diffraction orders, eigenmode branch selection). Both
backends must agree on R/T/A for the default BSI pixel, including oblique
incidence where grcwa previously diverged (R ~ 1e60) because round-off
flipped propagating eigenmodes to the backward branch.
"""

import numpy as np
import pytest
import yaml

from compass.geometry.pixel_stack import PixelStack
from compass.solvers.base import SolverFactory

pytest.importorskip("torcwa")
pytest.importorskip("grcwa")
import compass.solvers.rcwa.grcwa_solver
import compass.solvers.rcwa.torcwa_solver  # noqa: F401

WAVELENGTHS = [0.45, 0.6, 0.65]


def _run(name, params, theta_deg):
    with open("configs/pixel/default_bsi_1um.yaml") as f:
        cfg = yaml.safe_load(f)
    solver = SolverFactory.create(
        name, {"name": name, "params": {"n_lens_slices": 5, "dtype": "complex128", **params}}
    )
    solver.setup_geometry(PixelStack(cfg if "pixel" in cfg else {"pixel": cfg}))
    solver.setup_source({
        "wavelength": {"mode": "list", "values": WAVELENGTHS},
        "angle": {"theta_deg": theta_deg, "phi_deg": 0.0},
        "polarization": "TE",
    })
    return solver.run()


@pytest.mark.parametrize("theta_deg", [0.0, 10.0])
def test_torcwa_grcwa_agree_on_bsi_pixel(theta_deg):
    # torcwa [3,3] -> 7x7 = 49 orders; grcwa nG=49 (circular truncation).
    t = _run("torcwa", {"fourier_order": [3, 3]}, theta_deg)
    g = _run("grcwa", {"nG": 49}, theta_deg)
    for arr in (g.reflection, g.transmission, g.absorption):
        assert np.all(np.isfinite(arr))
    assert np.all((g.reflection >= 0) & (g.reflection <= 1))
    np.testing.assert_allclose(g.reflection, t.reflection, atol=0.01)
    np.testing.assert_allclose(g.transmission, t.transmission, atol=0.01)
    np.testing.assert_allclose(g.absorption, t.absorption, atol=0.01)
    # Both use field-integrated QE; residual differences come from the
    # different (circular vs. rectangular) Fourier truncation.
    assert g.metadata["qe_method"] == t.metadata["qe_method"] == "field_integration"
    for key in t.qe_per_pixel:
        np.testing.assert_allclose(g.qe_per_pixel[key], t.qe_per_pixel[key], atol=0.04)


@pytest.mark.parametrize("grid", [63, 297])
def test_grcwa_odd_grid_oblique_is_stable(grid):
    """Regression: odd epsilon grids at oblique incidence used to diverge."""
    g = _run("grcwa", {"nG": 49, "epsilon_grid": grid, "grid_multiplier": 0}, 10.0)
    assert np.all((g.reflection >= 0) & (g.reflection <= 1))
    assert np.all((g.transmission >= 0) & (g.reflection + g.transmission <= 1 + 1e-6))
