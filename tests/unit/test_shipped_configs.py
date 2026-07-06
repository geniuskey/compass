"""Every shipped Hydra config must pass CompassConfig validation.

scripts/run_simulation.py validates the composed config with
CompassConfig.model_validate() before running, so any configs/ YAML that
drifts from the schema (or vice versa) breaks the CLI. This test composes
each shipped option group against the defaults and validates it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from compass.core import CompassConfig

CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs")


def _options(group: str) -> list[str]:
    return sorted(p.stem for p in (Path(CONFIG_DIR) / group).glob("*.yaml"))


def _compose_and_validate(overrides: list[str]) -> None:
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config", overrides=overrides)
    # Match the entry point: resolve=True, but stub the launch-time ${now:...}
    # interpolation that only exists inside a real Hydra app run.
    container = OmegaConf.to_container(cfg, resolve=False)
    container["output_dir"] = "./outputs/test"
    CompassConfig.model_validate(container)


def test_default_config_validates():
    _compose_and_validate([])


@pytest.mark.parametrize("solver", _options("solver"))
def test_solver_configs_validate(solver):
    _compose_and_validate([f"solver={solver}"])


@pytest.mark.parametrize("source", _options("source"))
def test_source_configs_validate(source):
    _compose_and_validate([f"source={source}"])


@pytest.mark.parametrize("pixel", _options("pixel"))
def test_pixel_configs_validate(pixel):
    _compose_and_validate([f"pixel={pixel}"])


@pytest.mark.parametrize("compute", _options("compute"))
def test_compute_configs_validate(compute):
    _compose_and_validate([f"compute={compute}"])


@pytest.mark.parametrize("experiment", _options("experiment"))
def test_experiment_configs_validate(experiment):
    _compose_and_validate([f"+experiment={experiment}"])


def test_unknown_nested_key_is_rejected():
    with pytest.raises(Exception, match="thicknes"):
        CompassConfig.model_validate({"pixel": {"layers": {"silicon": {"thicknes": 3.0}}}})
