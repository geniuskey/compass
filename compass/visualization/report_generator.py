"""Automated comparison report generator for COMPASS."""
from __future__ import annotations

import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from compass.analysis.crosstalk import CrosstalkAnalyzer
from compass.analysis.solver_comparison import SolverComparison
from compass.core.types import SimulationResult
from compass.visualization.qe_plot import (
    plot_crosstalk_heatmap,
    plot_qe_comparison,
)

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate HTML comparison reports from multiple simulation results.

    The generator produces:
    * QE comparison plots (per-channel overlay across solvers/configs)
    * Crosstalk heatmaps for each result
    * Energy-balance bar charts (R + T + A = 1)
    * An HTML file that embeds or links to all figures

    Args:
        results: List of SimulationResult objects to compare.
        labels: Human-readable labels for each result.
        output_dir: Directory where report artefacts are written.
    """

    def __init__(
        self,
        results: list[SimulationResult],
        labels: list[str],
        output_dir: str,
    ) -> None:
        if len(results) != len(labels):
            raise ValueError(
                f"Number of results ({len(results)}) must match number of "
                f"labels ({len(labels)})."
            )
        self.results = results
        self.labels = labels
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Figure generation
    # ------------------------------------------------------------------

    def generate_figures(self) -> list[str]:
        """Save QE comparison, crosstalk, and energy balance figures.

        Returns:
            List of file paths for the saved figure images.
        """
        saved: list[str] = []

        # 1. QE comparison figure
        saved.append(self._make_qe_comparison_figure())

        # 2. Crosstalk heatmap per result
        for idx, (result, label) in enumerate(zip(self.results, self.labels)):
            saved.append(self._make_crosstalk_figure(result, label, idx))

        # 3. Energy balance figure
        saved.append(self._make_energy_balance_figure())

        return [p for p in saved if p]

    def _make_qe_comparison_figure(self) -> str:
        """Create and save a QE comparison overlay plot."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        try:
            plot_qe_comparison(self.results, self.labels, ax=ax)
        except Exception:
            logger.exception("Failed to create QE comparison figure.")
            plt.close(fig)
            return ""
        path = os.path.join(self.output_dir, "qe_comparison.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("Saved QE comparison figure to %s", path)
        return path

    def _make_crosstalk_figure(
        self, result: SimulationResult, label: str, idx: int,
    ) -> str:
        """Create and save a crosstalk heatmap for a single result."""
        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        try:
            plot_crosstalk_heatmap(result, ax=ax)
            ax.set_title(f"Crosstalk -- {label}")
        except Exception:
            logger.exception("Failed to create crosstalk figure for '%s'.", label)
            plt.close(fig)
            return ""
        path = os.path.join(self.output_dir, f"crosstalk_{idx}.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("Saved crosstalk figure to %s", path)
        return path

    def _make_energy_balance_figure(self) -> str:
        """Create and save a grouped bar chart of R, T, A per result."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        n = len(self.results)
        x = np.arange(n)
        width = 0.25

        r_vals, t_vals, a_vals = [], [], []
        for result in self.results:
            r_vals.append(float(np.mean(result.reflection)) if result.reflection is not None else 0.0)
            t_vals.append(float(np.mean(result.transmission)) if result.transmission is not None else 0.0)
            a_vals.append(float(np.mean(result.absorption)) if result.absorption is not None else 0.0)

        ax.bar(x - width, r_vals, width, label="Reflection", color="steelblue")
        ax.bar(x, t_vals, width, label="Transmission", color="darkorange")
        ax.bar(x + width, a_vals, width, label="Absorption", color="forestgreen")

        ax.set_xticks(x)
        ax.set_xticklabels(self.labels, rotation=30, ha="right")
        ax.set_ylabel("Mean value")
        ax.set_title("Energy Balance Comparison")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3, axis="y")

        path = os.path.join(self.output_dir, "energy_balance.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("Saved energy balance figure to %s", path)
        return path

    # ------------------------------------------------------------------
    # HTML report
    # ------------------------------------------------------------------

    def generate_html(self) -> str:
        """Generate an HTML comparison report as a string.

        The report includes embedded images (relative paths) and summary
        tables produced by :class:`SolverComparison` and
        :class:`CrosstalkAnalyzer`.

        Returns:
            HTML string.
        """
        # Solver comparison summary
        comparison = SolverComparison(self.results, self.labels)
        comp_summary = comparison.summary()

        # Crosstalk summaries
        ct_summaries: list[dict] = []
        for result in self.results:
            ct_summaries.append(CrosstalkAnalyzer.summarize(result))

        lines: list[str] = [
            "<!DOCTYPE html>",
            "<html><head>",
            "<meta charset='utf-8'>",
            "<title>COMPASS Comparison Report</title>",
            "<style>",
            "  body { font-family: sans-serif; margin: 2em; }",
            "  table { border-collapse: collapse; margin: 1em 0; }",
            "  th, td { border: 1px solid #ccc; padding: 4px 8px; text-align: right; }",
            "  th { background-color: #f0f0f0; }",
            "  img { max-width: 100%; height: auto; }",
            "  h1, h2, h3 { color: #333; }",
            "</style>",
            "</head><body>",
            "<h1>COMPASS Comparison Report</h1>",
        ]

        # QE comparison section
        lines.append("<h2>QE Comparison</h2>")
        lines.append("<img src='qe_comparison.png' alt='QE comparison'>")

        # Solver comparison table
        lines.append("<h2>Solver Comparison</h2>")
        lines.append("<h3>Runtime</h3>")
        lines.append("<table><tr><th>Solver</th><th>Runtime (s)</th></tr>")
        for label, rt in comp_summary.get("runtimes_seconds", {}).items():
            lines.append(f"<tr><td>{label}</td><td>{rt:.3f}</td></tr>")
        lines.append("</table>")

        if comp_summary.get("max_qe_diff"):
            lines.append("<h3>Max QE Difference</h3>")
            lines.append("<table><tr><th>Pair</th><th>Max |dQE|</th></tr>")
            for key, val in comp_summary["max_qe_diff"].items():
                lines.append(f"<tr><td>{key}</td><td>{val:.6f}</td></tr>")
            lines.append("</table>")

        # Crosstalk section
        lines.append("<h2>Crosstalk Analysis</h2>")
        for idx, (label, ct_s) in enumerate(zip(self.labels, ct_summaries)):
            lines.append(f"<h3>{label}</h3>")
            lines.append(f"<img src='crosstalk_{idx}.png' alt='Crosstalk {label}'>")
            lines.append("<table>")
            lines.append("<tr><th>Metric</th><th>Value</th></tr>")
            lines.append(f"<tr><td>Peak crosstalk</td><td>{ct_s['peak_crosstalk']:.6f}</td></tr>")
            lines.append(f"<tr><td>Peak wavelength (um)</td><td>{ct_s['peak_wavelength_um']:.4f}</td></tr>")
            lines.append(f"<tr><td>Mean crosstalk</td><td>{ct_s['mean_crosstalk']:.6f}</td></tr>")
            lines.append("</table>")

        # Energy balance section
        lines.append("<h2>Energy Balance</h2>")
        lines.append("<img src='energy_balance.png' alt='Energy balance'>")

        lines.append("</body></html>")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, filepath: str) -> None:
        """Save complete report (figures + HTML) to disk.

        This method first generates all figures, then writes the HTML
        report to *filepath*.

        Args:
            filepath: Path for the HTML file. If it does not end with
                '.html' it will be appended.
        """
        self.generate_figures()
        html = self.generate_html()

        if not filepath.endswith(".html"):
            filepath = filepath + ".html"

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")
        logger.info("Saved report to %s", filepath)
