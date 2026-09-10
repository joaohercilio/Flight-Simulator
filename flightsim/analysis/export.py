from __future__ import annotations

import pathlib

import numpy as np

from flightsim.core.simulation import SimulationResult
from flightsim.core.state import StateIndex

STATE_COLUMNS = ["x_e_m", "y_e_m", "z_e_m", "phi_rad", "theta_rad", "psi_rad", "u_m_s", "v_m_s", "w_m_s",
                 "p_rad_s", "q_rad_s", "r_rad_s"]
CONTROL_COLUMNS = ["elevator_deg", "aileron_deg", "rudder_deg", "throttle", "brake"]


def export_csv(res: SimulationResult, path: pathlib.Path, t_start: float | None = None, t_end: float | None = None) -> int:
    if t_start is not None or t_end is not None:
        res = res.window(t_start if t_start is not None else -np.inf, t_end if t_end is not None else np.inf)
    columns = (["time_s"] + STATE_COLUMNS + [c + "_dot" for c in STATE_COLUMNS[6:]]
               + ["alpha_rad", "beta_rad", "airspeed_m_s"] + CONTROL_COLUMNS)
    data = np.column_stack([res.t, res.x.T, res.dx[StateIndex.U:StateIndex.R + 1].T,
                            res.alpha, res.beta, res.airspeed, res.u.T])
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, data, delimiter=",", header=",".join(columns), comments="", fmt="%.10g")
    return len(res.t)
