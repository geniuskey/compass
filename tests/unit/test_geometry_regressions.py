"""Regressions for geometry / material / source bugs found in the 2026-11 audit."""

import numpy as np
import pytest

from compass.geometry.pixel_stack import PixelStack
from compass.materials.database import MaterialDB
from compass.sources.planewave import PlanewaveSource


def _stack(layers=None, **pixel):
    cfg = {"pitch": 1.0, "unit_cell": [2, 2], "layers": layers or {}, **pixel}
    return PixelStack({"pixel": cfg})


class TestMicrolensShift:
    def test_shifted_lens_wraps_across_the_period(self):
        ml = {"radius_x": 0.5, "radius_y": 0.5}
        base = _stack({"microlens": dict(ml)}, unit_cell=[1, 1])._get_height_map(64, 64)
        shifted = _stack(
            {"microlens": {**ml, "shift": {"mode": "manual", "shift_x": 0.3}}}, unit_cell=[1, 1]
        )._get_height_map(64, 64)
        assert shifted.mean() == pytest.approx(base.mean(), rel=1e-3)  # no volume lost
        # periodic continuity: first and last columns are neighbours
        assert abs(shifted[:, 0].max() - shifted[:, -1].max()) < 0.1

    def test_sharing_must_divide_unit_cell(self):
        with pytest.raises(ValueError, match="sharing"):
            _stack({"microlens": {"sharing": 2}}, unit_cell=[2, 3])


class TestBayerMap:
    def test_pattern_is_used_without_explicit_bayer_map(self):
        stack = _stack({"color_filter": {"pattern": "quad_bayer"}}, unit_cell=[4, 4])
        assert stack.bayer_map[0][:2] == ["R", "R"]
        assert len(stack.bayer_map) == 4 and len(stack.bayer_map[0]) == 4
        stack.get_layer_slices(0.55, 32, 32)  # used to raise IndexError

    def test_small_explicit_map_is_tiled(self):
        stack = _stack(unit_cell=[4, 4], bayer_map=[["R", "G"], ["G", "B"]])
        assert stack.bayer_map[3][3] == "B"


class TestDtiDepthIsPathIndependent:
    @pytest.mark.parametrize("mode,expected_span", [("fdti", (0.0, 3.0)), ("bdti", (1.5, 3.0))])
    def test_liner_does_not_change_trench_depth(self, mode, expected_span):
        def trench_span(liner):
            dti = {"enabled": True, "width": 0.1, "depth": 1.5, "mode": mode}
            if liner:
                dti["liner"] = {"enabled": True, "thickness": 0.01, "material": "al2o3"}
            stack = _stack({"silicon": {"thickness": 3.0, "dti": dti}})
            spans = [
                (s.z_start, s.z_end)
                for s in stack.get_layer_slices(0.55, 64, 64)
                if s.name.startswith("silicon") and np.any(np.real(s.eps_grid) < 10)
            ]
            return min(a for a, _ in spans), max(b for _, b in spans)

        assert trench_span(False) == pytest.approx(expected_span)
        assert trench_span(True) == pytest.approx(expected_span)


class TestMaterials:
    def test_si3n4_has_no_pole_at_one_micron(self):
        n, k = MaterialDB().get_nk("si3n4", 1.0)
        assert 1.9 < n < 2.1 and k == 0.0

    def test_reregistering_invalidates_epsilon_cache(self):
        db = MaterialDB()
        db.register_constant("probe", 1.5)
        assert db.get_epsilon("probe", 0.55) == pytest.approx(2.25)
        db.register_constant("probe", 2.0)
        assert db.get_epsilon("probe", 0.55) == pytest.approx(4.0)

    def test_aliases_follow_reregistered_material(self):
        db = MaterialDB()
        db.register_constant("aluminum_oxide", 1.7)
        assert db.get_nk("al2o3", 0.55) == (1.7, 0.0)

    def test_tabulated_k_does_not_ring_below_absorption_edge(self, tmp_path):
        path = tmp_path / "edge.csv"
        wl = np.linspace(0.8, 1.1, 13)
        k = np.where(wl < 0.95, 0.3, 0.0)
        np.savetxt(path, np.column_stack([wl, np.full_like(wl, 3.5), k]), delimiter=",")
        db = MaterialDB()
        db.load_csv("edge", str(path))
        fine = np.linspace(0.95, 1.1, 200)
        assert max(db.get_nk("edge", float(w))[1] for w in fine) == 0.0


class TestPlanewaveConfig:
    def test_none_sections_use_defaults(self):
        src = PlanewaveSource.from_config(
            {"wavelength": {"mode": "sweep", "sweep": None}, "angle": None, "polarization": None}
        )
        assert src.n_wavelengths > 1 and src.polarization == "unpolarized"

    def test_list_mode_requires_values(self):
        with pytest.raises(ValueError, match="must not be empty"):
            PlanewaveSource.from_config({"wavelength": {"mode": "list", "values": None}})

    def test_rejects_unknown_polarization(self):
        with pytest.raises(ValueError, match="polarization"):
            PlanewaveSource(wavelengths=np.array([0.55]), polarization="te")  # type: ignore[arg-type]


class TestMicrolensConfigFields:
    def test_gap_narrows_default_radius(self):
        ml = _stack({"microlens": {"gap": 0.1}}).microlenses[0]
        assert ml.radius_x == pytest.approx(0.45) and ml.radius_y == pytest.approx(0.45)

    def test_spherical_profile_is_n2_alpha1(self):
        ml = _stack({"microlens": {"profile": {"type": "spherical", "n": 5.0}}}).microlenses[0]
        assert (ml.n_param, ml.alpha_param) == (2.0, 1.0)

    def test_unknown_profile_raises(self):
        with pytest.raises(ValueError, match="profile type"):
            _stack({"microlens": {"profile": {"type": "conical"}}})


@pytest.mark.parametrize("source", [
    {"type": "cone_illumination", "cone": {"pupil_shape": "elliptical"}},
    {"type": "cone_illumination", "cone": {}, "ray_file": {"enabled": True}},
])
def test_unimplemented_cone_options_raise(source):
    from compass.runners.cone_runner import ConeIlluminationRunner

    source = {"wavelength": {"mode": "single", "value": 0.55}, **source}
    with pytest.raises(NotImplementedError):
        ConeIlluminationRunner.run({"pixel": {"pitch": 1.0}, "solver": {"name": "tmm"},
                                    "source": source})
