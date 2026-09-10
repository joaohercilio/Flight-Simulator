from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from flightsim.core.simulation import SimulationResult
from flightsim.core.state import StateIndex as I

Group = list[tuple[str, NDArray]]
TRAJECTORY_3D = "Trajectory 3D"


def _clean(arr: NDArray, tol: float = 1e-9) -> NDArray:
    out = np.array(arr, dtype=float)
    out[np.abs(out) < tol] = 0.0
    return out


def _wrap_deg(angle: NDArray) -> NDArray:
    return np.degrees(np.arctan2(np.sin(angle), np.cos(angle)))


def build_groups(res: SimulationResult) -> dict[str, Group]:
    x, dx, u = res.x, res.dx, res.u
    deg = np.degrees
    return {
        "Position": [("North [m]", _clean(x[I.X_E])), ("East [m]", _clean(x[I.Y_E])), ("Altitude [m]", _clean(-x[I.Z_E]))],
        "Velocity NED": [("Vel. North [m/s]", _clean(dx[I.X_E])), ("Vel. East [m/s]", _clean(dx[I.Y_E])),
                         ("Vel. Down [m/s]", _clean(dx[I.Z_E]))],
        "Euler angles": [("phi [deg]", _clean(deg(x[I.PHI]))), ("theta [deg]", _clean(deg(x[I.THETA]))),
                         ("psi [deg]", _clean(_wrap_deg(x[I.PSI])))],
        "Euler rates": [("phi_dot [deg/s]", _clean(deg(dx[I.PHI]))), ("theta_dot [deg/s]", _clean(deg(dx[I.THETA]))),
                        ("psi_dot [deg/s]", _clean(deg(dx[I.PSI])))],
        "Angular velocity": [("p [deg/s]", _clean(deg(x[I.P]))), ("q [deg/s]", _clean(deg(x[I.Q]))),
                             ("r [deg/s]", _clean(deg(x[I.R])))],
        "Angular acceleration": [("p_dot [deg/s²]", _clean(deg(dx[I.P]))), ("q_dot [deg/s²]", _clean(deg(dx[I.Q]))),
                                 ("r_dot [deg/s²]", _clean(deg(dx[I.R])))],
        "Aerodynamics": [("alpha [deg]", _clean(deg(res.alpha))), ("beta [deg]", _clean(deg(res.beta))),
                         ("Airspeed [m/s]", _clean(res.airspeed))],
        "Body velocity": [("u [m/s]", _clean(x[I.U])), ("v [m/s]", _clean(x[I.V])), ("w [m/s]", _clean(x[I.W]))],
        "Body acceleration": [("u_dot [m/s²]", _clean(dx[I.U])), ("v_dot [m/s²]", _clean(dx[I.V])),
                              ("w_dot [m/s²]", _clean(dx[I.W]))],
        "Controls": [("Elevator [deg]", u[0]), ("Aileron [deg]", u[1]), ("Rudder [deg]", u[2]), ("Throttle [-]", u[3])],
    }


GROUP_NAMES = ["Position", "Velocity NED", "Euler angles", "Euler rates", "Angular velocity", "Angular acceleration",
               "Aerodynamics", "Body velocity", "Body acceleration", "Controls", TRAJECTORY_3D]
