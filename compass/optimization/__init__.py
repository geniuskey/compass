"""Inverse design / optimization module for COMPASS pixel structures.

Provides gradient-free and gradient-based optimization of pixel geometry
parameters (microlens shape, BARL thicknesses, color filter thickness, etc.)
to maximize QE, minimize crosstalk, or achieve custom objective targets.
"""

from compass.optimization.history import OptimizationHistory
from compass.optimization.objectives import (
    CompositeObjective,
    EnergyBalanceRegularizer,
    MaximizePeakQE,
    MaximizeQE,
    MinimizeCrosstalk,
    ObjectiveFunction,
)
from compass.optimization.optimizer import OptimizationResult, PixelOptimizer
from compass.optimization.parameters import (
    BARLThicknesses,
    ColorFilterThickness,
    MicrolensHeight,
    MicrolensRadii,
    MicrolensSquareness,
    OptimizableParameter,
    ParameterSpace,
)

__all__ = [
    "BARLThicknesses",
    "ColorFilterThickness",
    "CompositeObjective",
    "EnergyBalanceRegularizer",
    "MaximizePeakQE",
    "MaximizeQE",
    "MicrolensHeight",
    "MicrolensRadii",
    "MicrolensSquareness",
    "MinimizeCrosstalk",
    "ObjectiveFunction",
    "OptimizableParameter",
    "OptimizationHistory",
    "OptimizationResult",
    "ParameterSpace",
    "PixelOptimizer",
]
