"""Export report figures from measured WSL results; no simulation or interpolation."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

FIGURES = ("spectra", "differences", "performance", "convergence", "geometry")
COLORS = ["#2563eb", "#0891b2", "#16a34a", "#7c3aed", "#db2777", "#ea580c", "#64748b"]


def setting(row):
    if row["solver"] == "fdtdx":
        duration = row["params"].get("duration", row["params"].get("runtime"))
        return f"dx={row['params']['grid_spacing']:g}, t={duration:g}"
    return f"N={row['order']}, grid={row['grid']}"


def label(row):
    return f"{row['solver']} / {row['device']}\n{setting(row)}"


def export(fig, output, name):
    fig.savefig(output / f"{name}.png", dpi=180, facecolor="white")
    svg = output / f"{name}.svg"
    fig.savefig(svg, facecolor="white")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def generate_figures(output: Path, highest, paired, best, previous, selection):
    """Use the same selected runs as the report tables and reference gates."""
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "#334155",
            "text.color": "#0f172a",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "axes.edgecolor": "#cbd5e1",
            "axes.titleweight": "bold",
            "svg.fonttype": "none",
        }
    ):
        reference = None
        if best:
            spectra = sorted(best["spectra"], key=lambda s: s["wavelength_um"])
            reference = np.asarray([s["rta"] for s in spectra])
        _spectra(output, highest, best, reference)
        if reference is not None:
            _differences(output, highest, reference, selection, best["resolution"])
        if paired:
            _performance(output, paired)
        _convergence(output, best, previous, selection)
        _geometry(output)


def _spectra(output, highest, best, reference):
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True, layout="constrained")
    handles = []
    names = []
    for row, color in zip(highest, COLORS):
        for i, ax in enumerate(axs):
            (line,) = ax.plot(
                np.asarray(row["wavelength_um"]) * 1000,
                np.asarray(row["rta"])[:, i] * 100,
                "o-",
                color=color,
                lw=2,
                markersize=6,
            )
        handles.append(line)
        names.append(label(row).replace("\n", " · "))
    if reference is not None:
        wavelengths = [
            s["wavelength_um"] * 1000
            for s in sorted(best["spectra"], key=lambda s: s["wavelength_um"])
        ]
        for i, ax in enumerate(axs):
            (line,) = ax.plot(
                wavelengths,
                reference[:, i] * 100,
                "s--",
                color="#0f172a",
                lw=3,
                markersize=7,
                zorder=5,
            )
        handles.append(line)
        names.append(f"Meep res={best['resolution']} (provisional reference)")
    for ax, title in zip(
        axs, ["Reflected power R (%)", "Transmitted power T (%)", "Residual absorption A (%)"]
    ):
        ax.set_ylabel(title)
        ax.grid(alpha=0.18)
        ax.set_xticks([450, 550, 650])
    axs[0].set_title("Measured optical response at three wavelengths", loc="left", pad=14)
    axs[-1].set_xlabel("Wavelength (nm) · segments connect measured samples only")
    fig.legend(handles, names, loc="outside lower center", ncol=2, fontsize=10, frameon=False)
    export(fig, output, "spectra")


def _differences(output, highest, reference, selection, resolution):
    changes = (
        np.asarray([np.max(np.abs(np.asarray(row["rta"]) - reference), axis=1) for row in highest])
        * 100
    )
    fig, (ax, heat) = plt.subplots(
        1, 2, figsize=(11, 6), layout="constrained", gridspec_kw={"width_ratios": [1.55, 1]}
    )
    positions = np.arange(len(highest))
    maxima = changes.max(axis=1)
    ax.barh(positions, maxima, color=COLORS[: len(highest)], height=0.6)
    for i, value in enumerate(maxima):
        ax.text(value + 0.06, i, f"{value:.2f}", va="center", fontsize=11)
    if selection["grid_delta"] is not None:
        grid_pp = selection["grid_delta"] * 100
        ax.axvline(
            grid_pp, color="#b45309", ls="--", lw=1.8, label=f"Meep grid change: {grid_pp:.2f} pp"
        )
        ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.set(
        yticks=positions,
        yticklabels=[label(row) for row in highest],
        xlabel="Maximum |solver - Meep| (percentage points)",
        title=f"Difference from Meep res={resolution}",
    )
    ax.set_xlim(0, max(maxima) * 1.24)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.15)
    image = heat.imshow(changes, cmap="YlOrRd", aspect="auto", vmin=0, vmax=maxima.max())
    heat.set(
        xticks=[0, 1, 2],
        xticklabels=["450", "550", "650"],
        yticks=[],
        xlabel="Wavelength (nm)",
        title="Largest R/T/A difference",
    )
    for i in range(len(highest)):
        for j in range(3):
            heat.text(
                j,
                i,
                f"{changes[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if changes[i, j] > maxima.max() * 0.6 else "#0f172a",
            )
    fig.colorbar(image, ax=heat, label="Percentage points", shrink=0.65)
    fig.suptitle(
        "Provisional comparison · highest measured settings, not an accuracy ranking", fontsize=13
    )
    export(fig, output, "differences")


def _performance(output, paired):
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    y = np.arange(len(paired))
    cpu = np.asarray([a["warm_median_s"] for a, _ in paired])
    gpu = np.asarray([b["warm_median_s"] for _, b in paired])
    ax.barh(y - 0.18, cpu, 0.32, color="#64748b", label="CPU")
    ax.barh(y + 0.18, gpu, 0.32, color="#2563eb", label="GPU requested")
    for i in range(len(paired)):
        ax.text(cpu[i] + 2, i - 0.18, f"{cpu[i]:.2f} s", va="center", fontsize=11)
        ax.text(
            gpu[i] + 2,
            i + 0.18,
            f"{gpu[i]:.2f} s · {cpu[i] / gpu[i]:.2f}x",
            va="center",
            fontsize=11,
        )
    names = [
        label(a) + ("\nGPU + CPU eigensolver" if a["solver"] == "meent_jax" else "")
        for a, _ in paired
    ]
    ax.set(
        yticks=y,
        yticklabels=names,
        xlabel="Seconds per run · all three wavelengths · lower is faster",
        title="CPU / GPU time at identical settings",
    )
    ax.set_xlim(0, max(cpu.max(), gpu.max()) * 1.30)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.15)
    ax.legend(loc="lower right", frameon=False)
    fig.suptitle(
        "Median of 3 warm runs · low-order timing does not establish accuracy", fontsize=13
    )
    export(fig, output, "performance")


def _convergence(output, best, previous, selection):
    fig, (ax, detail) = plt.subplots(
        2, 1, figsize=(10, 8), layout="constrained", gridspec_kw={"height_ratios": [1, 1.4]}
    )
    values = [
        selection["grid_delta"],
        max(selection["time_delta"], selection["coarse_time_delta"])
        if selection["time_delta"] is not None and selection["coarse_time_delta"] is not None
        else None,
    ]
    for i, (value, limit) in enumerate(zip(values, [0.005, 0.0005])):
        if value is None:
            ax.text(0.05, i, "Not measured", va="center")
            continue
        ratio = value / limit
        ax.barh(i, ratio, color="#16a34a" if ratio <= 1 else "#ea580c", height=0.45)
        ax.text(
            max(ratio + 0.07, 1.08),
            i,
            f"{value * 100:.4f} pp / limit {limit * 100:g} pp",
            va="center",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
        )
    ax.axvline(1, color="#0f172a", ls="--", lw=2)
    ax.set(
        yticks=[0, 1, 2],
        yticklabels=["Spatial grid", "Time window", "PML width"],
        xlabel="Measured change / allowed change · ≤1 passes",
        title="Is the Meep reference converged?",
    )
    ax.text(0.05, 2, "Not measured — reference remains unapproved", color="#92400e", va="center")
    ax.set_xlim(
        0, max([v / t for v, t in zip(values, [0.005, 0.0005]) if v is not None] + [1]) + 2.5
    )
    ax.set_ylim(2.5, -0.5)
    ax.grid(axis="x", alpha=0.15)
    if best and previous:
        fine = {s["wavelength_um"]: s for s in best["spectra"]}
        coarse = {s["wavelength_um"]: s for s in previous["spectra"]}
        x = np.arange(3)
        for i, (name, color) in enumerate(zip(["R", "T", "A"], COLORS)):
            changes = [(fine[w]["rta"][i] - coarse[w]["rta"][i]) * 100 for w in sorted(fine)]
            detail.bar(x + (i - 1) * 0.24, changes, 0.23, color=color, label=name)
        detail.axhline(0.5, color="#ea580c", ls="--", label="±0.5 pp grid criterion")
        detail.axhline(-0.5, color="#ea580c", ls="--")
        detail.axhline(0, color="#94a3b8", lw=1)
        detail.set(
            xticks=x,
            xticklabels=["450", "550", "650"],
            xlabel="Wavelength (nm)",
            ylabel="Signed change (percentage points)",
            title=f"What changes from Meep res={previous['resolution']} to res={best['resolution']}?",
        )
        detail.legend(frameon=False, ncol=4, fontsize=10, loc="lower left")
        detail.grid(axis="y", alpha=0.15)
    export(fig, output, "convergence")


def _geometry(output):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    top, side = axs
    top.add_patch(Rectangle((-0.4, -0.4), 0.8, 0.8, facecolor="#eff6ff", edgecolor="#64748b"))
    for x, y, w, h in [(-0.2, -0.15, 0.25, 0.2), (0.175, 0.175, 0.2, 0.25)]:
        top.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, facecolor="#2563eb", alpha=0.85))
    top.axhline(-0.15, color="#ea580c", ls="--", label="XZ slice at y=-0.15 µm")
    top.set(
        xlim=(-0.44, 0.44),
        ylim=(-0.44, 0.44),
        xlabel="x (µm)",
        ylabel="y (µm)",
        title="XY: one periodic unit cell",
    )
    top.set_aspect("equal")
    top.legend(loc="lower left", fontsize=9)
    for z, height, color, text in [
        (0, 0.12, "#cbd5e1", "Spacer n=1.45"),
        (0.12, 0.24, "#eff6ff", "Air around pillars"),
        (0.36, 0.08, "#93c5fd", "Cap n=1.6"),
    ]:
        side.add_patch(Rectangle((-0.4, z), 0.8, height, facecolor=color, edgecolor="white"))
        side.text(0.42, z + height / 2, text, va="center", fontsize=10)
    side.add_patch(Rectangle((-0.325, 0.12), 0.25, 0.24, facecolor="#2563eb"))
    side.text(-0.2, 0.24, "n=2.4\n+0.02i", color="white", ha="center", va="center", fontsize=10)
    side.annotate(
        "Incident Ey",
        xy=(0, 0.46),
        xytext=(0, 0.66),
        ha="center",
        arrowprops={"arrowstyle": "->", "color": "#ea580c", "lw": 2},
        color="#c2410c",
    )
    side.text(0, -0.075, "Air output", ha="center", fontsize=11)
    side.set(
        xlim=(-0.44, 1.1),
        ylim=(-0.12, 0.75),
        xlabel="x (µm)",
        ylabel="z above spacer bottom (µm)",
        title="XZ: slice through the lower-left pillar",
    )
    side.set_aspect("equal")
    export(fig, output, "geometry")
