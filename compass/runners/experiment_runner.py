"""Dispatch a composed config to the runner its experiment section asks for.

Experiment overlays (``configs/experiment/*.yaml``) add one of:

* ``experiment.solvers``  -> :class:`ComparisonRunner`
* ``experiment.roi``      -> :class:`ROISweepRunner`
* ``optimization``        -> :class:`PixelOptimizer`

Configs without these sections run a single simulation.
"""

from __future__ import annotations

import logging
from typing import Any

from compass.core.types import SimulationResult

logger = logging.getLogger(__name__)

_PARAMETER_TYPES = {
    "microlens_height": "MicrolensHeight",
    "microlens_squareness": "MicrolensSquareness",
    "microlens_radii": "MicrolensRadii",
    "barl_thicknesses": "BARLThicknesses",
    "color_filter_thickness": "ColorFilterThickness",
}


def experiment_kind(config: dict) -> str:
    """Return "comparison", "roi_sweep", "optimization" or "single"."""
    experiment = config.get("experiment") or {}
    if experiment.get("solvers"):
        return "comparison"
    if experiment.get("roi"):
        return "roi_sweep"
    if config.get("optimization"):
        return "optimization"
    return "single"


def build_parameter_space(config: dict, specs: list[dict]):
    """Build a ParameterSpace bound to ``config`` from YAML parameter specs."""
    from compass.optimization import parameters as params_mod

    params = []
    for spec in specs:
        kind = spec.get("type")
        if kind not in _PARAMETER_TYPES:
            raise ValueError(
                f"Unknown optimization parameter type {kind!r}; "
                f"expected one of {sorted(_PARAMETER_TYPES)}"
            )
        kwargs: dict[str, Any] = {}
        if "bounds" in spec:
            lo, hi = spec["bounds"]
            kwargs = {"min_val": float(lo), "max_val": float(hi)}
        params.append(getattr(params_mod, _PARAMETER_TYPES[kind])(config, **kwargs))
    return params_mod.ParameterSpace(params)


def build_objective(spec: dict):
    """Build an ObjectiveFunction from a YAML objective spec."""
    from compass.optimization import objectives as obj_mod

    kind = spec.get("type")
    wl_range = spec.get("wavelength_range")
    wl_range = tuple(wl_range) if wl_range is not None else None
    if kind == "maximize_qe":
        return obj_mod.MaximizeQE(target_pixels=spec.get("target_pixels"), wavelength_range=wl_range)
    if kind == "minimize_crosstalk":
        return obj_mod.MinimizeCrosstalk(target_wavelength_range=wl_range)
    if kind == "maximize_peak_qe":
        return obj_mod.MaximizePeakQE(channel=spec.get("channel", "G"))
    if kind == "energy_balance_regularizer":
        return obj_mod.EnergyBalanceRegularizer(
            tolerance=float(spec.get("tolerance", 0.01)),
            penalty_weight=float(spec.get("penalty_weight", 10.0)),
        )
    if kind == "composite":
        components = spec.get("components") or []
        return obj_mod.CompositeObjective(
            [(float(c.get("weight", 1.0)), build_objective(c)) for c in components]
        )
    raise ValueError(f"Unknown objective type {kind!r}")


def run_experiment(config: dict) -> dict:
    """Run the experiment described by ``config``.

    Returns:
        ``{"kind": ..., "results": {label: SimulationResult}, ...}`` with
        extra keys per kind ("summary" for comparisons, "optimization" for
        optimizations).
    """
    kind = experiment_kind(config)
    if kind == "comparison":
        from compass.runners.comparison_runner import ComparisonRunner

        out = ComparisonRunner.run(config, list(config["experiment"]["solvers"]))
        return {
            "kind": kind,
            "results": dict(zip(out["labels"], out["results"], strict=True)),
            "summary": out["summary"],
        }
    if kind == "roi_sweep":
        from compass.runners.roi_sweep_runner import ROISweepRunner

        roi_results = ROISweepRunner.run(config, dict(config["experiment"]["roi"]))
        return {"kind": kind, "results": dict(sorted(roi_results.items()))}
    if kind == "optimization":
        import copy

        from compass.optimization.optimizer import PixelOptimizer

        opt = dict(config["optimization"])
        base = copy.deepcopy(config)
        solver_name = opt.get("solver") or base.get("solver", {}).get("name", "torcwa")
        if base.get("solver", {}).get("name") not in (None, solver_name):
            # Params of a different top-level solver do not apply.
            logger.warning(
                f"optimization.solver={solver_name!r} overrides solver "
                f"{base['solver']['name']!r}; its params are dropped"
            )
            base["solver"] = {"name": solver_name}
        space = build_parameter_space(base, list(opt.get("parameters") or []))
        objective = build_objective(dict(opt.get("objective") or {}))
        optimizer = PixelOptimizer(
            base_config=base,
            parameter_space=space,
            objective=objective,
            solver_name=solver_name,
            method=opt.get("method", "nelder-mead"),
            max_iterations=int(opt.get("max_iterations", 100)),
            tolerance=float(opt.get("tolerance", 1e-4)),
        )
        res = optimizer.optimize()
        results: dict[str, SimulationResult] = {}
        if res.final_result is not None:
            results["best"] = res.final_result
        return {
            "kind": kind,
            "results": results,
            "optimization": {
                "parameters": space.names,
                "best_params": res.best_params.tolist(),
                "best_objective": res.best_objective,
                "n_evaluations": res.n_evaluations,
                "converged": bool(res.converged),
            },
        }

    from compass.runners.single_run import SingleRunner

    return {"kind": kind, "results": {"result": SingleRunner.run(config)}}
