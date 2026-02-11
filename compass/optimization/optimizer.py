"""Main optimization engine for COMPASS inverse design.

Wraps scipy.optimize methods with COMPASS-specific parameter handling,
objective evaluation through the solver pipeline, and iteration tracking.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass

import numpy as np
from scipy import optimize as sp_optimize

from compass.core.types import SimulationResult
from compass.optimization.history import OptimizationHistory
from compass.optimization.objectives import ObjectiveFunction
from compass.optimization.parameters import ParameterSpace
from compass.runners.single_run import SingleRunner

logger = logging.getLogger(__name__)

# Methods that support bound constraints
_BOUNDED_METHODS = {"l-bfgs-b", "differential-evolution", "powell", "nelder-mead"}

# Methods implemented via scipy.optimize.minimize
_MINIMIZE_METHODS = {"nelder-mead", "l-bfgs-b", "powell"}


@dataclass
class OptimizationResult:
    """Container for optimization run results.

    Attributes:
        best_params: Parameter vector at the optimum.
        best_objective: Objective value at the optimum.
        history: Full iteration history.
        n_evaluations: Total number of objective evaluations.
        converged: Whether the optimizer reported convergence.
        final_config: Config dict with best parameters applied.
        final_result: SimulationResult from the best configuration.
    """

    best_params: np.ndarray
    best_objective: float
    history: OptimizationHistory
    n_evaluations: int
    converged: bool
    final_config: dict
    final_result: SimulationResult | None = None


class PixelOptimizer:
    """Gradient-free and gradient-based optimization for pixel structures.

    Runs COMPASS simulations in a loop, adjusting pixel geometry parameters
    to minimize a user-defined objective function.

    Supported methods:
        - ``nelder-mead``: Gradient-free simplex (robust default).
        - ``l-bfgs-b``: Quasi-Newton with bound constraints.
        - ``powell``: Gradient-free directional search.
        - ``differential-evolution``: Global stochastic optimizer.

    Args:
        base_config: Full COMPASS configuration dict (deep-copied internally).
        parameter_space: ParameterSpace defining what to optimize.
        objective: Objective function to minimize.
        solver_name: Solver backend name (e.g. "meent", "torcwa").
        method: Optimization algorithm name.
        max_iterations: Maximum number of optimizer iterations.
        tolerance: Convergence tolerance for the optimizer.
    """

    def __init__(
        self,
        base_config: dict,
        parameter_space: ParameterSpace,
        objective: ObjectiveFunction,
        solver_name: str = "meent",
        method: str = "nelder-mead",
        max_iterations: int = 100,
        tolerance: float = 1e-4,
    ):
        self.base_config = copy.deepcopy(base_config)
        self.parameter_space = parameter_space
        self.objective = objective
        self.solver_name = solver_name
        self.method = method.lower()
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        self._history = OptimizationHistory()
        self._n_evals = 0
        self._best_objective = float("inf")
        self._best_params: np.ndarray | None = None
        self._best_result: SimulationResult | None = None
        self._start_time: float = 0.0

    def optimize(self) -> OptimizationResult:
        """Run the optimization loop.

        Returns:
            OptimizationResult with best parameters, history, and final config.
        """
        x0 = self.parameter_space.to_vector()
        lo, hi = self.parameter_space.get_bounds()
        bounds = list(zip(lo.tolist(), hi.tolist()))

        logger.info(
            f"Starting optimization: method={self.method}, "
            f"n_params={len(x0)}, max_iter={self.max_iterations}"
        )
        logger.info(f"Objective: {self.objective.name()}")
        logger.info(f"Initial params: {x0.tolist()}")

        self._start_time = time.perf_counter()

        if self.method == "differential-evolution":
            result = sp_optimize.differential_evolution(
                self._evaluate,
                bounds=bounds,
                maxiter=self.max_iterations,
                tol=self.tolerance,
                callback=self._callback_de,
                seed=self.base_config.get("seed", 42),
            )
            converged = result.success
            best_x = result.x

        elif self.method in _MINIMIZE_METHODS:
            options: dict = {
                "maxiter": self.max_iterations,
            }
            if self.method == "nelder-mead":
                options["xatol"] = self.tolerance
                options["fatol"] = self.tolerance
                scipy_bounds = None  # Nelder-Mead handles bounds via clipping in _evaluate
            elif self.method in ("l-bfgs-b", "powell"):
                scipy_bounds = bounds

            result = sp_optimize.minimize(
                self._evaluate,
                x0,
                method=self.method,
                bounds=scipy_bounds if self.method != "nelder-mead" else None,
                options=options,
                callback=self._callback_minimize,
            )
            converged = result.success
            best_x = result.x

        else:
            raise ValueError(
                f"Unknown optimization method: '{self.method}'. "
                f"Supported: nelder-mead, l-bfgs-b, powell, differential-evolution"
            )

        elapsed = time.perf_counter() - self._start_time

        # Use tracked best (may differ from scipy's result due to clipping)
        if self._best_params is not None:
            best_x = self._best_params

        # Re-create parameter space on the final config to apply best values
        self.parameter_space.from_vector(best_x)

        logger.info(
            f"Optimization complete: {self._n_evals} evaluations in {elapsed:.1f}s"
        )
        logger.info(f"Best objective: {self._best_objective:.6f}")
        logger.info(f"Best params: {best_x.tolist()}")
        logger.info(f"Converged: {converged}")

        return OptimizationResult(
            best_params=best_x,
            best_objective=self._best_objective,
            history=self._history,
            n_evaluations=self._n_evals,
            converged=converged,
            final_config=self.base_config,
            final_result=self._best_result,
        )

    def _evaluate(self, x: np.ndarray) -> float:
        """Run simulation with parameter vector x and compute objective.

        Args:
            x: Flat parameter vector from the optimizer.

        Returns:
            Scalar objective value.
        """
        # Clip to bounds
        lo, hi = self.parameter_space.get_bounds()
        x_clipped = np.clip(x, lo, hi)

        # Apply parameters to config
        self.parameter_space.from_vector(x_clipped)

        # Ensure solver name is set
        self.base_config.setdefault("solver", {})["name"] = self.solver_name

        # Run simulation
        try:
            result = SingleRunner.run(self.base_config)
        except Exception as e:
            logger.warning(f"Simulation failed for params {x_clipped.tolist()}: {e}")
            return 1e6  # Large penalty for failed simulations

        # Evaluate objective
        obj_value = self.objective.evaluate(result)
        self._n_evals += 1

        # Track best
        if obj_value < self._best_objective:
            self._best_objective = obj_value
            self._best_params = x_clipped.copy()
            self._best_result = result

        # Record history
        elapsed = time.perf_counter() - self._start_time
        self._history.record(
            iteration=self._n_evals - 1,
            params=x_clipped,
            objective=obj_value,
            metadata={"elapsed_seconds": elapsed},
        )

        if self._n_evals % 10 == 0:
            logger.info(
                f"  eval {self._n_evals}: obj={obj_value:.6f} "
                f"best={self._best_objective:.6f}"
            )

        return obj_value

    def _callback_minimize(self, xk: np.ndarray) -> None:
        """Callback for scipy.optimize.minimize methods.

        Args:
            xk: Current parameter vector after an iteration.
        """
        logger.debug(
            f"Iteration callback: n_evals={self._n_evals}, "
            f"best={self._best_objective:.6f}"
        )

    def _callback_de(self, xk: np.ndarray, convergence: float) -> None:
        """Callback for differential_evolution.

        Args:
            xk: Best parameter vector so far.
            convergence: Convergence metric (0 = not converged, 1 = converged).
        """
        logger.debug(
            f"DE callback: convergence={convergence:.4f}, "
            f"best={self._best_objective:.6f}"
        )
