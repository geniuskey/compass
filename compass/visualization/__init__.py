"""Visualization modules for COMPASS CMOS image sensor simulation.

Provides matplotlib-based 2D plotting utilities for pixel stack structures,
electromagnetic field distributions, quantum efficiency spectra, and crosstalk
analysis, as well as interactive 3D visualization via plotly.
"""

from compass.visualization.field_plot_2d import (
    plot_field_2d,
    plot_field_multi_wavelength,
)
from compass.visualization.qe_plot import (
    plot_angular_response,
    plot_crosstalk_heatmap,
    plot_qe_comparison,
    plot_qe_spectrum,
)
from compass.visualization.structure_plot_2d import plot_pixel_cross_section
from compass.visualization.viewer_3d import view_pixel_3d
from compass.visualization.report_generator import ReportGenerator

__all__ = [
    "plot_pixel_cross_section",
    "plot_field_2d",
    "plot_field_multi_wavelength",
    "plot_qe_spectrum",
    "plot_qe_comparison",
    "plot_crosstalk_heatmap",
    "plot_angular_response",
    "view_pixel_3d",
    "ReportGenerator",
]
