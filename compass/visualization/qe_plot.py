"""Quantum efficiency and crosstalk visualization for COMPASS.

Provides plotting functions for QE spectra, multi-solver comparisons,
pixel-to-pixel crosstalk heatmaps, and angular response curves.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np

from compass.core.types import SimulationResult
from compass.core.units import um_to_nm

logger = logging.getLogger(__name__)

# Default color mapping for Bayer channels.
_CHANNEL_COLORS: dict[str, str] = {
    "R": "red",
    "G": "green",
    "B": "blue",
}

# Line style rotation for multi-solver comparison.
_LINE_STYLES: list = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]


def _classify_pixels(
    qe_per_pixel: dict[str, np.ndarray],
) -> dict[str, list[tuple[str, np.ndarray]]]:
    """Group pixel QE arrays by color channel.

    Pixel keys are expected to follow the pattern '{Color}_{row}_{col}'
    (e.g. 'R_0_0', 'G_0_1'). The first character is used as the channel
    identifier.

    Args:
        qe_per_pixel: Mapping of pixel name to QE spectrum array.

    Returns:
        Dictionary mapping color character ('R', 'G', 'B') to a list of
        (pixel_name, qe_array) tuples.
    """
    channels: dict[str, list[tuple[str, np.ndarray]]] = {}
    for pname, qe_arr in qe_per_pixel.items():
        channel = pname[0].upper() if pname else "?"
        if channel not in channels:
            channels[channel] = []
        channels[channel].append((pname, qe_arr))
    return channels


def plot_qe_spectrum(
    result: SimulationResult,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (8, 5),
) -> plt.Axes:
    """Plot quantum efficiency vs wavelength for R, G, B channels.

    For each color channel, the QE spectrum is averaged across all pixels
    of that channel in the unit cell. Individual pixels are plotted as thin
    semi-transparent lines.

    Args:
        result: SimulationResult containing qe_per_pixel and wavelengths.
        ax: Optional matplotlib Axes. If None, a new figure is created.
        figsize: Figure size in inches (used when ax is None).

    Returns:
        The matplotlib Axes containing the plot.
    """
    if ax is None:
        _fig, ax = plt.subplots(1, 1, figsize=figsize)

    wavelengths_nm = np.asarray(um_to_nm(result.wavelengths))

    if not result.qe_per_pixel:
        logger.warning("No QE data available in SimulationResult.")
        ax.set_title("No QE data available")
        return ax

    channels = _classify_pixels(result.qe_per_pixel)

    for channel in ("R", "G", "B"):
        if channel not in channels:
            continue

        pixels = channels[channel]
        color = _CHANNEL_COLORS.get(channel, "gray")

        # Plot individual pixels as thin lines
        for pname, qe_arr in pixels:
            if len(qe_arr) != len(wavelengths_nm):
                logger.warning(
                    "Pixel '%s' QE length (%d) does not match wavelength "
                    "array length (%d); skipping.",
                    pname, len(qe_arr), len(wavelengths_nm),
                )
                continue
            ax.plot(
                wavelengths_nm, qe_arr,
                color=color, alpha=0.3, linewidth=0.8,
            )

        # Average across all pixels in this channel
        valid_arrays = [
            qe for _, qe in pixels if len(qe) == len(wavelengths_nm)
        ]
        if valid_arrays:
            mean_qe = np.mean(valid_arrays, axis=0)
            ax.plot(
                wavelengths_nm, mean_qe,
                color=color, linewidth=2.0, label=f"{channel} (mean)",
            )

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("QE")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Quantum Efficiency Spectrum")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    return ax


def plot_qe_comparison(
    results: Sequence[SimulationResult],
    labels: Sequence[str],
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (10, 6),
    show_difference: bool = False,
) -> plt.Axes | tuple[plt.Axes, plt.Axes]:
    """Overlay QE spectra from multiple solver results for comparison.

    Each solver result is rendered with a distinct line style while
    maintaining channel colors (R=red, G=green, B=blue).

    Args:
        results: Sequence of SimulationResult objects to compare.
        labels: Display labels for each result (e.g. solver names).
        ax: Optional matplotlib Axes. If None, a new figure is created.
        figsize: Figure size in inches (used when ax is None).
        show_difference: If True, adds a second subplot below showing the
            difference between the first result and each subsequent result.

    Returns:
        The matplotlib Axes (or tuple of main and difference Axes if
        show_difference is True).
    """
    if len(results) != len(labels):
        raise ValueError(
            f"Number of results ({len(results)}) must match number of "
            f"labels ({len(labels)})."
        )

    if not results:
        logger.warning("No results provided for comparison.")
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_title("No results to compare")
        return ax

    if show_difference and len(results) >= 2:
        if ax is not None:
            logger.warning(
                "show_difference=True requires creating new figure; "
                "ignoring provided ax."
            )
        fig, (ax_main, ax_diff) = plt.subplots(
            2, 1, figsize=figsize, height_ratios=[3, 1], sharex=True,
        )
    else:
        ax_diff = None
        if ax is None:
            fig, ax_main = plt.subplots(1, 1, figsize=figsize)
        else:
            ax_main = ax

    for res_idx, (result, label) in enumerate(zip(results, labels)):
        ls = _LINE_STYLES[res_idx % len(_LINE_STYLES)]
        wavelengths_nm = np.asarray(um_to_nm(result.wavelengths))

        if not result.qe_per_pixel:
            logger.warning("Result '%s' has no QE data; skipping.", label)
            continue

        channels = _classify_pixels(result.qe_per_pixel)

        for channel in ("R", "G", "B"):
            if channel not in channels:
                continue

            color = _CHANNEL_COLORS.get(channel, "gray")
            valid_arrays = [
                qe for _, qe in channels[channel]
                if len(qe) == len(wavelengths_nm)
            ]
            if not valid_arrays:
                continue

            mean_qe = np.mean(valid_arrays, axis=0)
            ax_main.plot(
                wavelengths_nm, mean_qe,
                color=color, linestyle=ls, linewidth=1.8,
                label=f"{channel} -- {label}",
            )

            # Difference subplot
            if ax_diff is not None and res_idx > 0:
                ref_channels = _classify_pixels(results[0].qe_per_pixel)
                if channel in ref_channels:
                    ref_wl_nm = np.asarray(um_to_nm(results[0].wavelengths))
                    ref_arrays = [
                        qe for _, qe in ref_channels[channel]
                        if len(qe) == len(ref_wl_nm)
                    ]
                    if ref_arrays:
                        ref_mean = np.mean(ref_arrays, axis=0)
                        # Interpolate if wavelength grids differ
                        if len(ref_mean) == len(mean_qe):
                            diff = mean_qe - ref_mean
                        else:
                            diff = np.interp(
                                wavelengths_nm,
                                ref_wl_nm,
                                ref_mean,
                            )
                            diff = mean_qe - diff
                        ax_diff.plot(
                            wavelengths_nm, diff,
                            color=color, linestyle=ls, linewidth=1.2,
                            label=f"{channel}: {label} - {labels[0]}",
                        )

    ax_main.set_ylabel("QE")
    ax_main.set_ylim(-0.02, 1.02)
    ax_main.set_title("QE Comparison")
    ax_main.legend(loc="best", fontsize=7, ncol=2)
    ax_main.grid(True, alpha=0.3)

    if ax_diff is not None:
        ax_diff.set_xlabel("Wavelength (nm)")
        ax_diff.set_ylabel("Delta QE")
        ax_diff.axhline(y=0, color="black", linewidth=0.5)
        ax_diff.legend(loc="best", fontsize=7, ncol=2)
        ax_diff.grid(True, alpha=0.3)
        fig.tight_layout()
        result_axes: plt.Axes | tuple[plt.Axes, plt.Axes] = (ax_main, ax_diff)
        return result_axes
    else:
        ax_main.set_xlabel("Wavelength (nm)")
        result_ax: plt.Axes = ax_main
        return result_ax


def plot_crosstalk_heatmap(
    result: SimulationResult,
    wavelength_idx: int | None = None,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (7, 6),
) -> plt.Axes:
    """Plot pixel-to-pixel crosstalk as a heatmap matrix.

    Constructs a matrix where entry (i, j) represents the fraction of light
    intended for pixel j that is detected by pixel i. The diagonal represents
    the correctly detected signal, and off-diagonal elements represent
    crosstalk.

    If wavelength_idx is None, the crosstalk is averaged over all wavelengths.

    Args:
        result: SimulationResult with qe_per_pixel data.
        wavelength_idx: Index into the wavelength array. If None, averages
            over all wavelengths.
        ax: Optional matplotlib Axes. If None, a new figure is created.
        figsize: Figure size in inches (used when ax is None).

    Returns:
        The matplotlib Axes containing the heatmap.
    """
    if ax is None:
        _fig, ax = plt.subplots(1, 1, figsize=figsize)

    if not result.qe_per_pixel:
        logger.warning("No QE data available for crosstalk computation.")
        ax.set_title("No QE data available")
        return ax

    pixel_names = sorted(result.qe_per_pixel.keys())
    n_pixels = len(pixel_names)

    if n_pixels == 0:
        ax.set_title("No pixel data")
        return ax

    # Build QE matrix: rows = pixels, cols = wavelength indices
    qe_matrix = np.zeros((n_pixels, len(result.wavelengths)))
    for i, pname in enumerate(pixel_names):
        qe = result.qe_per_pixel[pname]
        if len(qe) == len(result.wavelengths):
            qe_matrix[i, :] = qe
        else:
            logger.warning(
                "Pixel '%s' QE length mismatch; padding with zeros.", pname
            )

    # Compute crosstalk matrix
    if wavelength_idx is not None:
        if wavelength_idx < 0 or wavelength_idx >= len(result.wavelengths):
            logger.warning(
                "wavelength_idx=%d out of range [0, %d); using average.",
                wavelength_idx, len(result.wavelengths),
            )
            qe_vec = np.mean(qe_matrix, axis=1)
            wl_label = "avg"
        else:
            qe_vec = qe_matrix[:, wavelength_idx]
            wl_nm = um_to_nm(result.wavelengths[wavelength_idx])
            wl_label = f"{wl_nm:.0f} nm"
    else:
        qe_vec = np.mean(qe_matrix, axis=1)
        wl_label = "avg over all wavelengths"

    # Normalize: crosstalk_ij = qe_i / sum(qe) for each intended target pixel
    total_qe = np.sum(qe_vec)
    if total_qe > 0:
        xtalk_matrix = np.outer(qe_vec, qe_vec) / (total_qe * total_qe) * n_pixels
    else:
        xtalk_matrix = np.zeros((n_pixels, n_pixels))

    # Simple diagonal-dominant approximation:
    # Self-signal on diagonal, crosstalk estimated from QE ratios
    for i in range(n_pixels):
        row_sum = np.sum(qe_vec)
        if row_sum > 0:
            xtalk_matrix[i, :] = qe_vec / row_sum
            xtalk_matrix[i, i] = qe_vec[i] / row_sum

    im = ax.imshow(
        xtalk_matrix,
        cmap="YlOrRd",
        aspect="equal",
        vmin=0,
        vmax=max(np.max(xtalk_matrix), 1e-10),
        origin="lower",
    )
    plt.colorbar(im, ax=ax, label="Relative signal")

    ax.set_xticks(range(n_pixels))
    ax.set_yticks(range(n_pixels))
    ax.set_xticklabels(pixel_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(pixel_names, fontsize=7)
    ax.set_xlabel("Source pixel")
    ax.set_ylabel("Detector pixel")
    ax.set_title(f"Crosstalk Matrix ({wl_label})")

    # Annotate cell values
    for i in range(n_pixels):
        for j in range(n_pixels):
            val = xtalk_matrix[i, j]
            text_color = "white" if val > 0.5 * np.max(xtalk_matrix) else "black"
            ax.text(
                j, i, f"{val:.2f}",
                ha="center", va="center",
                fontsize=7, color=text_color,
            )

    return ax


def plot_angular_response(
    results_vs_angle: Sequence[SimulationResult],
    angles: Sequence[float],
    wavelength_idx: int | None = None,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (8, 5),
) -> plt.Axes:
    """Plot QE vs incidence angle at a fixed wavelength.

    Each result in the sequence corresponds to a simulation at a different
    chief ray angle. The mean QE per color channel is extracted and plotted
    as a function of angle.

    Args:
        results_vs_angle: Sequence of SimulationResult objects, one per angle.
        angles: Incidence angles in degrees, matching the order of
            results_vs_angle.
        wavelength_idx: Index into the wavelength array to select a single
            wavelength. If None, the QE is averaged over all wavelengths.
        ax: Optional matplotlib Axes. If None, a new figure is created.
        figsize: Figure size in inches (used when ax is None).

    Returns:
        The matplotlib Axes containing the plot.
    """
    if len(results_vs_angle) != len(angles):
        raise ValueError(
            f"Number of results ({len(results_vs_angle)}) must match "
            f"number of angles ({len(angles)})."
        )

    if ax is None:
        _fig, ax = plt.subplots(1, 1, figsize=figsize)

    if not results_vs_angle:
        logger.warning("No angular results provided.")
        ax.set_title("No angular data")
        return ax

    angles_arr = np.array(angles, dtype=float)

    # Collect mean QE per channel at each angle
    channel_data: dict[str, list[float]] = {}

    for result in results_vs_angle:
        if not result.qe_per_pixel:
            # Append NaN for missing data
            for ch in channel_data:
                channel_data[ch].append(np.nan)
            continue

        channels = _classify_pixels(result.qe_per_pixel)
        seen_channels = set()

        for channel, pixels in channels.items():
            if channel not in channel_data:
                # Back-fill previous angles with NaN
                channel_data[channel] = [np.nan] * (
                    len(angles_arr) - len(results_vs_angle) + len(channel_data.get(channel, []))
                )
                channel_data[channel] = [np.nan] * (
                    results_vs_angle.index(result)
                )

            valid_arrays = [
                qe for _, qe in pixels
                if len(qe) == len(result.wavelengths)
            ]
            if not valid_arrays:
                channel_data.setdefault(channel, []).append(np.nan)
                seen_channels.add(channel)
                continue

            mean_qe_spectrum = np.mean(valid_arrays, axis=0)

            if wavelength_idx is not None and 0 <= wavelength_idx < len(result.wavelengths):
                qe_val = float(mean_qe_spectrum[wavelength_idx])
            else:
                qe_val = float(np.mean(mean_qe_spectrum))

            channel_data.setdefault(channel, []).append(qe_val)
            seen_channels.add(channel)

        # Fill channels not present in this result
        for ch in channel_data:
            if ch not in seen_channels:
                channel_data[ch].append(np.nan)

    # Plot each channel
    for channel in ("R", "G", "B"):
        if channel not in channel_data:
            continue
        qe_vs_angle = np.array(channel_data[channel])
        if len(qe_vs_angle) != len(angles_arr):
            logger.warning(
                "Channel '%s' has %d entries but %d angles; skipping.",
                channel, len(qe_vs_angle), len(angles_arr),
            )
            continue
        color = _CHANNEL_COLORS.get(channel, "gray")
        ax.plot(
            angles_arr, qe_vs_angle,
            color=color, linewidth=2.0, marker="o", markersize=4,
            label=channel,
        )

    # Determine wavelength label for title
    if wavelength_idx is not None and results_vs_angle:
        first_result = results_vs_angle[0]
        if 0 <= wavelength_idx < len(first_result.wavelengths):
            wl_nm = um_to_nm(first_result.wavelengths[wavelength_idx])
            wl_label = f"at {wl_nm:.0f} nm"
        else:
            wl_label = "(averaged)"
    else:
        wl_label = "(averaged over wavelength)"

    ax.set_xlabel("Angle (degrees)")
    ax.set_ylabel("QE")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"Angular Response {wl_label}")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    return ax
