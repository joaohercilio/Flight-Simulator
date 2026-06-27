# utils/io.py
"""I/O utilities: the bridge between files/UI and the simulation.

Two functions form the seam:

* ``load_aircraft(path)`` — turns an aircraft .toml (the file the UI loads or
  the New-aircraft editor saves) into an ``AircraftModel``, which is exactly
  what the simulation engine consumes. This is how the interface hands an
  aircraft to the physics: it only ever passes a path; ``load_aircraft`` does
  the translation.
* ``generate_plots(...)`` — turns the engine's output histories into figures.

It also has ``save_aircraft(path, model)`` so an ``AircraftModel`` round-trips
back to disk in the same schema the editor writes.

``AircraftModel`` itself is the domain entity and lives in ``flightsim.aircraft``
(infrastructure depends on the domain, not the reverse). It is re-exported here
so existing ``from utils.io import AircraftModel`` imports keep working.
"""

from __future__ import annotations

import pathlib
import tomllib

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

from flightsim.core.state import StateIndex
from flightsim.aircraft import AircraftModel
from flightsim.case import Case


def _resolve_tables_dir(model_file: pathlib.Path, configured: str) -> pathlib.Path:
    """Resolves the aero tables directory, with a sane fallback.
    """
    tables_dir = model_file.parent / configured
    if not tables_dir.is_dir():
        fallback = model_file.parent / "aero_tables"
        if fallback.is_dir():
            tables_dir = fallback
    return tables_dir


def load_aircraft(model_file: pathlib.Path) -> AircraftModel:
    """Loads an aircraft file into an AircraftModel.

    Args:
        model_file: Path to the aircraft file.

    Returns:
        Populated AircraftModel.

    Raises:
        FileNotFoundError: If the file does not exist.
        KeyError: If a required inertia/geometry value is missing.
    """
    model_file = pathlib.Path(model_file)
    if not model_file.exists():
        raise FileNotFoundError(f"Aircraft file not found: {model_file}")

    with open(model_file, "rb") as f:
        data = tomllib.load(f)

    aircraft = data.get("aircraft", {})
    inertia  = data["inertia"]
    geometry = data["geometry"]
    prop     = data.get("propulsion", {})
    ctl      = data.get("control_limits", {})
    gear     = data.get("landing_gear", {})
    aero     = data.get("aero", {})

    tables_dir = _resolve_tables_dir(model_file, aero.get("tables_dir", "aero_tables"))
    ground_altitude = data.get("ground_altitude", {}).get("alt", 0.0)

    return AircraftModel(
        mass=inertia["mass"],
        ix=inertia["Ix"], iy=inertia["Iy"], iz=inertia["Iz"], ixz=inertia["Ixz"],
        x_cg=inertia["x_cg"], y_cg=inertia["y_cg"], z_cg=inertia["z_cg"],
        s=geometry["S"], b=geometry["b"], c=geometry["c"],
        arm_z_engine=prop.get("arm_z", prop.get("arm_z_engine", 0.0)),
        elevator_max=ctl.get("elevator_max", 25.0),
        aileron_max=ctl.get("aileron_max", 20.0),
        rudder_max=ctl.get("rudder_max", 30.0),
        aero_tables_dir=tables_dir,
        ground_altitude=ground_altitude,
        elevator_min=ctl.get("elevator_min", -25.0),
        aileron_min=ctl.get("aileron_min", -20.0),
        rudder_min=ctl.get("rudder_min", -30.0),
        thrust_a=prop.get("a", 0.001274),
        thrust_b=prop.get("b", -0.07204),
        thrust_c=prop.get("c", -0.5428),
        thrust_d=prop.get("d", 40.89),
        main_gear_x=gear.get("main_gear_x", 0.44),
        main_gear_y=gear.get("main_gear_y", 0.2),
        wheelbase=gear.get("nose_gear", 0.306),
        gear_height=gear.get("gear_height", 0.15),
        design_deflection=gear.get("design_deflection", 0.03),
        damping_ratio=gear.get("damping_ratio", 1.0),
        ground_effect=geometry.get("ground_effect", 0.0),
        name=aircraft.get("name", model_file.stem),
    )


def load_case(case_file: pathlib.Path) -> Case:
    """Loads an case file into a Case.

    Args:
        case_file: Path to the case file.

    Returns:
        Populated Case object.

    Raises:
        FileNotFoundError: If the file does not exist.
        KeyError: If a required value is missing.
    """
    case_file = pathlib.Path(case_file)
    if not case_file.exists():
        raise FileNotFoundError(f"Case file not found: {case_file}")

    with open(case_file, "rb") as f:
        data = tomllib.load(f)

    case               = data["case"]
    time               = data["time"]
    gravity            = data["gravity"]
    density            = data["density"]
    trim               = data["trim_options"]
    initial_conditions = data["initial_conditions"]

    return Case(
        name=case["name"],
        total_time = time["total_time"],
        time_step = time["time_step"],
        gravity_model = gravity["gravity_model"],
        gravity = gravity["gravity"],
        density_model = density["density_model"],
        density = density["density"],
        enable_trim = trim["enable"],
        target_speed = trim["target_speed"],
        trim_name = trim["trim_name"],
        trim_alt =  trim["trim_alt"],
        trim_gamma = trim["trim_gamma"],
        trim_radius = trim["trim_radius"],
        u = initial_conditions["u"],
        v = initial_conditions["v"],
        w = initial_conditions["w"],
        x = initial_conditions["x"],
        y = initial_conditions["y"],
        height = initial_conditions["height"],
        p = initial_conditions["p"],
        q = initial_conditions["q"],
        r = initial_conditions["r"],
        phi = initial_conditions["phi"],
        theta = initial_conditions["theta"],
        psi = initial_conditions["psi"]
    )


def report_model(model: AircraftModel) -> str:
    text = f"""# aircraft model (SI units; angles in degrees)

[aircraft]
name = "{model.name}"

[inertia]
mass = {model.mass}
Ix   = {model.ix}
Iy   = {model.iy}
Iz   = {model.iz}
Ixz  = {model.ixz}
x_cg = {model.x_cg}
y_cg = {model.y_cg}
z_cg = {model.z_cg}

[geometry]
S = {model.s}
b = {model.b}
c = {model.c}
ground_effect = {model.ground_effect}

[control_limits]
elevator_min = {model.elevator_min}
elevator_max = {model.elevator_max}
aileron_min  = {model.aileron_min}
aileron_max  = {model.aileron_max}
rudder_min   = {model.rudder_min}
rudder_max   = {model.rudder_max}

[propulsion]
arm_z = {model.arm_z_engine}
# thrust = a*v^3 + b*v^2 + c*v + d
a = {model.thrust_a}
b = {model.thrust_b}
c = {model.thrust_c}
d = {model.thrust_d}

[landing_gear]
main_gear_x = {model.main_gear_x}
main_gear_y = {model.main_gear_y}
nose_gear   = {model.wheelbase}
gear_height = {model.gear_height}
design_deflection = {model.design_deflection}
damping_ratio     = {model.damping_ratio}

[aero]
tables_dir = "{model.aero_tables_dir}"

"""
    return text


def report_case(case: Case) -> str:
    text = f"""# case (SI units; angular rates in deg/s)

[case]
name = "{case.name}"

[gravity]
gravity_model = "{case.gravity_model}"
gravity = {case.gravity}

[density]
density_model = "{case.density_model}"
density = {case.density}

[initial_conditions]
u = {case.u}
v = {case.v}
w = {case.w}

x = {case.x}
y = {case.y}
height = {case.height}

p = {case.p}
q = {case.q}
r = {case.r}

phi = {case.phi}
theta = {case.theta}
psi = {case.psi}

[trim_options]
enable = {case.enable_trim}
trim_name = "{case.trim_name}"
target_speed = {case.target_speed}
trim_alt = {case.trim_alt}
trim_gamma = {case.trim_gamma}
trim_radius = {case.trim_radius}

"""
    return text


def save_aircraft(model_file: pathlib.Path, model: AircraftModel) -> None:
    """Writes an AircraftModel file.

    Args:
        model_file: Destination path for the file.
        model: Aircraft to serialise.
    """
    model_file = pathlib.Path(model_file)
    aero_dir = pathlib.Path(model.aero_tables_dir)
    try:
        aero_str = aero_dir.relative_to(model_file.parent).as_posix()
    except ValueError:
        aero_str = aero_dir.as_posix()

    text = f"""# aircraft model (SI units; angles in degrees)

[aircraft]
name = "{model.name}"

[inertia]
mass = {model.mass}
Ix   = {model.ix}
Iy   = {model.iy}
Iz   = {model.iz}
Ixz  = {model.ixz}
x_cg = {model.x_cg}
y_cg = {model.y_cg}
z_cg = {model.z_cg}

[geometry]
S = {model.s}
b = {model.b}
c = {model.c}
ground_effect = {model.ground_effect}

[control_limits]
elevator_min = {model.elevator_min}
elevator_max = {model.elevator_max}
aileron_min  = {model.aileron_min}
aileron_max  = {model.aileron_max}
rudder_min   = {model.rudder_min}
rudder_max   = {model.rudder_max}

[propulsion]
arm_z = {model.arm_z_engine}
# thrust = a*v^3 + b*v^2 + c*v + d
a = {model.thrust_a}
b = {model.thrust_b}
c = {model.thrust_c}
d = {model.thrust_d}

[landing_gear]
main_gear_x = {model.main_gear_x}
main_gear_y = {model.main_gear_y}
nose_gear   = {model.wheelbase}
gear_height = {model.gear_height}
design_deflection = {model.design_deflection}
damping_ratio     = {model.damping_ratio}

[aero]
tables_dir = "{aero_str}"
"""
    model_file.write_text(text, encoding="utf-8")


def save_case(case_file: pathlib.Path, case: Case) -> None:
    """Writes an Case file.

    Args:
        case_file: Destination path for the file.
        case: Case to serialise.
    """
    case_file = pathlib.Path(case_file)

    text = f"""# case model (SI units; angular velocity in deg/s)
[time]
total_time = {case.total_time}
time_step = {case.time_step}

[case]
name = "{case.name}"

[gravity]
gravity_model = "{case.gravity_model}"
gravity = {case.gravity}

[density]
density_model = "{case.density_model}"
density = {case.density}

[initial_conditions]
u = {case.u}
v = {case.v}
w = {case.w}

x = {case.x}
y = {case.y}
height = {case.height}

p = {case.p}
q = {case.q}
r = {case.r}

phi = {case.phi}
theta = {case.theta}
psi = {case.psi}

[trim_options]
enable = {str(case.enable_trim).lower()}
trim_name = "{case.trim_name}"
target_speed = {case.target_speed}
trim_alt = {case.trim_alt}
trim_gamma = {case.trim_gamma}
trim_radius = {case.trim_radius}

"""
    case_file.write_text(text, encoding="utf-8")


def _plot_trajectory_3d(
    position_data: list[tuple[str, NDArray]],
    output_dir: pathlib.Path | None,
    save_figures: bool,
    fig_index: int,
) -> None:
    """Renders a 3D trajectory plot from position data.

    Args:
        position_data: List of (label, data) tuples for North, East, Altitude.
        output_dir: Directory to save figures.
        save_figures: Whether to save to disk.
        fig_index: Figure number for file naming.
    """
    north = position_data[0][1]
    east = position_data[1][1]
    altitude = position_data[2][1]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(east, north, altitude, label="Flight Path", color="#1f77b4", linewidth=1.5)
    ax.scatter(east[0], north[0], altitude[0], color="green", marker="o", s=40, label="Start")
    ax.scatter(east[-1], north[-1], altitude[-1], color="red", marker="x", s=40, label="End")

    ax.set_xlabel("East [m]", fontsize=9, labelpad=10)
    ax.set_ylabel("North [m]", fontsize=9, labelpad=10)
    ax.set_zlabel("Altitude [m]", fontsize=9, labelpad=10)
    ax.set_title("3D Aircraft Trajectory", fontsize=12, pad=20)

    ax.tick_params(labelsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    min_span = 10.0

    for data, set_lim_func in [
        (east, ax.set_xlim),
        (north, ax.set_ylim),
        (altitude, ax.set_zlim),
    ]:
        d_min, d_max = np.min(data), np.max(data)
        if (d_max - d_min) < min_span:
            center = (d_max + d_min) / 2.0
            set_lim_func(center - min_span / 2.0, center + min_span / 2.0)

    if save_figures and output_dir is not None:
        plt.savefig(
            output_dir / f"fig_{fig_index+1:02d}_trajectory_3d.png",
            dpi=150,
            bbox_inches="tight"
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


_DEFAULT_FIGURES: list[list[str]] = [
    ["Position", "Velocity NED"],
    ["Euler angles", "Angular velocity"],
    ["Aerodynamics", "Body velocity"],
    ["Trajectory 3D"],
]


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

                y_min, y_max = np.min(data), np.max(data)
                y_span = y_max - y_min


                label_lower = label.lower()
                if "deg/s" in label_lower:
                    min_span = 2.0
                elif "deg" in label_lower:
                    min_span = 2.0
                elif "m/s" in label_lower:
                    min_span = 1.0
                elif "[m]" in label_lower:
                    min_span = 10.0
                else:
                    min_span = 1.0


                if y_span < min_span:
                    y_center = (y_max + y_min) / 2.0
                    ax.set_ylim(y_center - min_span / 2.0, y_center + min_span / 2.0)

            else:
                ax.axis("off")



