"""Meep microlens geometry, built against a stand-in ``mp`` (meep is conda-only)."""

from types import SimpleNamespace

import numpy as np

from compass.geometry.pixel_stack import PixelStack
from compass.solvers.fdtd.meep_solver import MeepSolver


class _V:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = x, y, z


def _mp():
    return SimpleNamespace(
        Vector3=_V,
        Prism=lambda verts, height, material: ("prism", verts, height),
        Block=lambda center, size, material: ("block", center, size),
    )


def _geometry(shift_x=0.0, base=0.0):
    stack = PixelStack({"pixel": {"pitch": 1.0, "unit_cell": [1, 1], "layers": {"microlens": {
        "height": 0.6, "radius_x": 0.45, "radius_y": 0.45, "base_thickness": base,
        "profile": {"n": 4.0}, "shift": {"mode": "manual", "shift_x": shift_x}}}}})
    solver = MeepSolver({"name": "meep"})
    solver.setup_geometry(stack)
    layer = next(lyr for lyr in stack.layers if lyr.name == "microlens")
    return solver._microlens_geometry(_mp(), layer, None, 1.0, 1.0, 0.0), layer


def test_prisms_fill_superellipse_slices():
    geom, layer = _geometry()
    prisms = [g for g in geom if g[0] == "prism"]
    assert len(prisms) == 20  # one full-height prism per slice, no gaps
    assert all(abs(p[2] - 0.6 / 20) < 1e-12 for p in prisms)
    verts = prisms[0][1]
    pts = np.array([[v.x, v.y] for v in verts])
    # n = 4 superellipse: |x/r|^4 + |y/r|^4 = 1 (squarer than a circle)
    r0 = np.abs(pts).max()
    norm = (np.abs(pts / r0) ** 4).sum(axis=1)
    np.testing.assert_allclose(norm, 1.0, rtol=1e-6)
    assert verts[0].z == layer.z_start


def test_base_block_and_periodic_images():
    geom, layer = _geometry(shift_x=0.3, base=0.05)
    assert geom[0][0] == "block"
    prisms = [g for g in geom if g[0] == "prism"]
    # the lens shifted by 0.3 crosses x = +0.5, so its -lx image is added too
    centres = {round(float(np.mean([v.x for v in p[1]])), 3) for p in prisms}
    assert {0.3, -0.7} <= centres
    assert min(v.z for p in prisms for v in p[1]) == layer.z_start + 0.05
