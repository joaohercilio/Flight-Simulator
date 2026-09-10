from __future__ import annotations

import pathlib

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from flightsim.analysis.channels import TRAJECTORY_3D, Group, build_groups
from flightsim.core.simulation import SimulationResult

MIN_SPAN = {"deg/s": 2.0, "deg": 2.0, "m/s": 1.0, "[m]": 10.0}


def _min_span(label: str) -> float:
    return next((span for key, span in MIN_SPAN.items() if key in label), 1.0)


def _pad(ax, data: NDArray, span: float) -> None:
    lo, hi = float(np.min(data)), float(np.max(data))
    if hi - lo < span:
        mid = 0.5 * (lo + hi)
        ax.set_ylim(mid - span / 2, mid + span / 2)


def draw_groups(fig: Figure, t: NDArray, groups: dict[str, Group], names: list[str]) -> None:
    valid = [n for n in names if n in groups]
    fig.clear()
    if not valid:
        return
    n_cols = max(len(groups[n]) for n in valid)
    axs = fig.subplots(len(valid), n_cols, squeeze=False)
    fig.subplots_adjust(hspace=0.5, wspace=0.35, left=0.08, right=0.98, top=0.95, bottom=0.1)
    for row, name in enumerate(valid):
        axs[row, 0].annotate(name, (0, 0.5), xycoords="axes fraction", xytext=(-52, 0), textcoords="offset points",
                             rotation=90, va="center", ha="center", fontsize=9, color="gray")
        for col in range(n_cols):
            ax = axs[row, col]
            if col >= len(groups[name]):
                ax.axis("off")
                continue
            label, data = groups[name][col]
            ax.plot(t, data, linewidth=1.2)
            ax.set_ylabel(label, fontsize=8)
            ax.set_xlabel("Time [s]", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, linestyle="--", alpha=0.5)
            _pad(ax, data, _min_span(label))


def draw_trajectory_3d(fig: Figure, groups: dict[str, Group]) -> None:
    north, east, alt = (d for _, d in groups["Position"])
    fig.clear()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(east, north, alt, color="#1f77b4", linewidth=1.5, label="Flight path")
    ax.scatter(east[0], north[0], alt[0], color="green", s=40, label="Start")
    ax.scatter(east[-1], north[-1], alt[-1], color="red", marker="x", s=40, label="End")
    ax.set_xlabel("East [m]", fontsize=9)
    ax.set_ylabel("North [m]", fontsize=9)
    ax.set_zlabel("Altitude [m]", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8)
    for data, setter in ((east, ax.set_xlim), (north, ax.set_ylim), (alt, ax.set_zlim)):
        lo, hi = float(np.min(data)), float(np.max(data))
        if hi - lo < 10.0:
            mid = 0.5 * (lo + hi)
            setter(mid - 5.0, mid + 5.0)


def draw_figure(fig: Figure, res: SimulationResult, names: list[str], groups: dict[str, Group] | None = None) -> None:
    groups = groups or build_groups(res)
    if TRAJECTORY_3D in names and len(names) == 1:
        draw_trajectory_3d(fig, groups)
    else:
        draw_groups(fig, res.t, groups, [n for n in names if n != TRAJECTORY_3D])


def generate_figures(res: SimulationResult, figures: list[list[str]], output_dir: pathlib.Path | None = None,
                     save: bool = False, show: bool = True) -> list[Figure]:
    import matplotlib.pyplot as plt
    groups = build_groups(res)
    out = []
    for i, names in enumerate(figures):
        fig = plt.figure(figsize=(14, 3.2 * len(names)) if names != [TRAJECTORY_3D] else (8, 8))
        draw_figure(fig, res, names, groups)
        fig.canvas.manager.set_window_title(" / ".join(names))
        if save and output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            name = "_".join(names).lower().replace(" ", "_")
            fig.savefig(output_dir / f"fig_{i + 1:02d}_{name}.png", dpi=150, bbox_inches="tight")
        out.append(fig)
    if show:
        plt.show()
    return out
