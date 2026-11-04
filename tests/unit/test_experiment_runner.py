"""Experiment dispatch: configs/experiment/*.yaml must reach the right runner."""

import pytest
import yaml

from compass.optimization.objectives import CompositeObjective, MinimizeCrosstalk
from compass.optimization.parameters import MicrolensHeight
from compass.runners import experiment_runner as er


def _load(name):
    with open(f"configs/experiment/{name}.yaml") as f:
        return yaml.safe_load(f)


@pytest.mark.parametrize("name,kind", [
    ("solver_comparison", "comparison"),
    ("roi_sweep", "roi_sweep"),
    ("optimize_microlens", "optimization"),
    ("qe_benchmark", "single"),
])
def test_shipped_experiments_dispatch(name, kind):
    assert er.experiment_kind(_load(name)) == kind


def test_comparison_entries_name_their_solver():
    for entry in _load("solver_comparison")["experiment"]["solvers"]:
        assert entry.get("name"), entry


def test_builders_follow_yaml_specs():
    opt = _load("optimize_microlens")["optimization"]
    config = {"pixel": {"layers": {"microlens": {"height": 0.6}}}}
    space = er.build_parameter_space(config, opt["parameters"])
    assert isinstance(space.params[0], MicrolensHeight)
    assert space.get_bounds()[0][0] == pytest.approx(0.2)
    space.from_vector(space.to_vector() * 0 + 0.9)
    assert config["pixel"]["layers"]["microlens"]["height"] == pytest.approx(0.9)

    objective = er.build_objective(opt["objective"])
    assert isinstance(objective, CompositeObjective)
    assert any(isinstance(o, MinimizeCrosstalk) for _, o in objective.objectives)


def test_unknown_types_raise():
    with pytest.raises(ValueError):
        er.build_parameter_space({}, [{"type": "nope"}])
    with pytest.raises(ValueError):
        er.build_objective({"type": "nope"})


def test_comparison_runner_requires_solver_name():
    from compass.runners.comparison_runner import ComparisonRunner

    with pytest.raises(ValueError, match="name"):
        ComparisonRunner._run_single_solver({}, {"params": {}})
