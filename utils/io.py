# utils/io.py
"""I/O utilities: loading models and generating plots."""

from __future__ import annotations

import dataclasses
import pathlib
import tomllib

import numpy as np


@dataclasses.dataclass(frozen=True)
class AircraftModel:
    """Aircraft inertia and geometry parameters.

    Attributes:
        mass: Total mass (kg).
        Ix: Moment of inertia about x (kg·m²).
        Iy: Moment of inertia about y (kg·m²).
        Iz: Moment of inertia about z (kg·m²).
        Ixz: Product of inertia xz (kg·m²).
        s: Wing reference area (m²).
        b: Wingspan (m).
        c: Mean aerodynamic chord (m).
        aero_tables_dir: Path to aerodynamic coefficient tables.
    """

    mass: float
    Ix: float
    Iy: float
    Iz: float
    Ixz: float
    s: float
    b: float
    c: float
    aero_tables_dir: pathlib.Path

    def report(self) -> None:
        """Prints a summary of the aircraft model parameters."""
        print("--- inertia ---")
        print(f"  mass : {self.mass} kg")
        print(f"  Ix   : {self.Ix} kg·m²")
        print(f"  Iy   : {self.Iy} kg·m²")
        print(f"  Iz   : {self.Iz} kg·m²")
        print(f"  Ixz  : {self.Ixz} kg·m²")
        print("--- geometry ---")
        print(f"  S    : {self.s} m²")
        print(f"  b    : {self.b} m")
        print(f"  c    : {self.c} m")
        print(f"--- aero tables ---")
        print(f"  dir  : {self.aero_tables_dir}")


def load_model(model_file: pathlib.Path) -> AircraftModel:
    """Loads an aircraft model from a TOML file.

    Args:
        model_file: Path to the aircraft_model.toml file.

    Returns:
        Populated AircraftModel instance.

    Raises:
        FileNotFoundError: If the file does not exist.
        KeyError: If any required field is missing.
    """
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    with open(model_file, "rb") as f:
        data = tomllib.load(f)

    inertia  = data["inertia"]
    geometry = data["geometry"]

    tables_dir = model_file.parent / data["aero"]["tables_dir"]

    return AircraftModel(
        mass=inertia["mass"],
        Ix=inertia["Ix"],
        Iy=inertia["Iy"],
        Iz=inertia["Iz"],
        Ixz=inertia["Ixz"],
        s=geometry["S"],
        b=geometry["b"],
        c=geometry["c"],
        aero_tables_dir=tables_dir,
    )


def _build_plot_groups(
    x: NDArray,
    dx: NDArray,
) -> dict[str, list[tuple[str, NDArray]]]:
    """Builds the dict of plottable variable groups from state arrays.

    Args:
        x: State history, shape (12, N).
        dx: State derivative history, shape (12, N).

    Returns:
        Dict mapping group name to list of (y-axis label, data array).
    """
    u = x[StateIndex.U]
    v = x[StateIndex.V]
    w = x[StateIndex.W]

    speed = np.sqrt(u**2 + v**2 + w**2)
    speed_safe = np.where(speed < 1e-8, 1e-8, speed)
    alpha = np.arctan2(w, u)
    beta  = np.arcsin(np.clip(v / speed_safe, -1.0, 1.0))

    rad2deg = 180.0 / np.pi

    def _clean(arr: NDArray, tol: float = 1e-6) -> NDArray:
        out = arr.copy()
        out[np.abs(out) < tol] = 0.0
        return out

    def _wrap(angle: NDArray) -> NDArray:
        return np.arctan2(np.sin(angle), np.cos(angle))

    return {
        "Position": [
            ("North [m]",    _clean(x[StateIndex.X_E])),
            ("East [m]",     _clean(x[StateIndex.Y_E])),
            ("Altitude [m]", _clean(-x[StateIndex.Z_E])),
        ],
        "Velocity NED": [
            ("Vel. North [m/s]", _clean(dx[StateIndex.X_E])),
            ("Vel. East [m/s]",  _clean(dx[StateIndex.Y_E])),
            ("Vel. Down [m/s]",  _clean(dx[StateIndex.Z_E])),
        ],
        "Euler angles": [
            ("phi [deg]",   x[StateIndex.PHI]   * rad2deg),
            ("theta [deg]", x[StateIndex.THETA] * rad2deg),
            ("psi [deg]",   _wrap(x[StateIndex.PSI]) * rad2deg),
        ],
        "Euler rates": [
            ("phi_dot [deg/s]",   dx[StateIndex.PHI]   * rad2deg),
            ("theta_dot [deg/s]", dx[StateIndex.THETA] * rad2deg),
            ("psi_dot [deg/s]",   dx[StateIndex.PSI]   * rad2deg),
        ],
        "Angular velocity": [
            ("p [deg/s]", x[StateIndex.P] * rad2deg),
            ("q [deg/s]", x[StateIndex.Q] * rad2deg),
            ("r [deg/s]", x[StateIndex.R] * rad2deg),
        ],
        "Aerodynamics": [
            ("alpha [deg]",    alpha * rad2deg),
            ("beta [deg]",     beta  * rad2deg),
            ("Airspeed [m/s]", _clean(speed)),
        ],
        "Body velocity": [
            ("u [m/s]", _clean(u)),
            ("v [m/s]", _clean(v)),
            ("w [m/s]", _clean(w)),
        ],
    }


def _load_plot_config(path: pathlib.Path) -> list[list[str]]:
    """Parses a plots.toml file into a list of figure definitions.

    Each [[figure]] entry defines one figure window, listing which
    groups of variables to include as subplot rows.

    Args:
        path: Path to plots.toml.

    Returns:
        List of figures, each a list of group name strings.

    Raises:
        FileNotFoundError: If the file does not exist.
        KeyError: If the TOML structure is invalid.
    """
    if not path.exists():
        raise FileNotFoundError(f"Plot config not found: {path}")

    with open(path, "rb") as f:
        data = tomllib.load(f)

    return [figure["groups"] for figure in data["figure"]]


def _plot_figure(
    t: NDArray,
    groups: dict[str, list[tuple[str, NDArray]]],
    group_names: list[str],
) -> None:
    """Renders one figure window with the requested groups as rows.

    Args:
        t: Time vector, shape (N,).
        groups: Dict from _build_plot_groups.
        group_names: Which groups to include as rows in this figure.
    """
    valid = [name for name in group_names if name in groups]
    if not valid:
        return

    n_rows = len(valid)
    fig, axs = plt.subplots(n_rows, 3, figsize=(14, 3 * n_rows), squeeze=False)
    fig.subplots_adjust(hspace=0.45, wspace=0.35)

    for row, group_name in enumerate(valid):
        variables = groups[group_name]
        axs[row, 0].set_ylabel(
            group_name,
            fontsize=9,
            labelpad=40,
            rotation=90,
            va="center",
            color="gray",
        )
        for col in range(3):
            ax = axs[row, col]
            if col < len(variables):
                label, data = variables[col]
                ax.plot(t, data, linewidth=1.2)
                ax.set_ylabel(label, fontsize=8)
                ax.set_xlabel("Time [s]", fontsize=8)
                ax.tick_params(labelsize=7)
                ax.grid(True, linestyle="--", alpha=0.5)
            else:
                ax.axis("off")


def generate_plots(
    t: NDArray,
    x: NDArray,
    dx: NDArray,
    plot_config: pathlib.Path,
    output_dir: pathlib.Path | None = None,
    save_figures: bool = False,
    show_gui: bool = True,
) -> None:
    """Generates simulation plots from a TOML plot configuration.

    Args:
        t: Time vector, shape (N,).
        x: State history, shape (12, N).
        dx: State derivative history, shape (12, N).
        plot_config: Path to plots.toml.
        output_dir: Directory to save figures. Required if save_figures=True.
        save_figures: Whether to save figures to disk as PNG.
        show_gui: Whether to open interactive plot windows.
    """
    groups = _build_plot_groups(x, dx)
    figures = _load_plot_config(plot_config)

    for i, group_names in enumerate(figures):
        _plot_figure(t, groups, group_names)

        if save_figures:
            if output_dir is None:
                raise ValueError("output_dir must be set when save_figures=True")
            output_dir.mkdir(parents=True, exist_ok=True)
            name = "_".join(group_names).lower().replace(" ", "_")
            plt.savefig(output_dir / f"fig_{i+1:02d}_{name}.png", dpi=150, bbox_inches="tight")

    if show_gui:
        plt.show()
