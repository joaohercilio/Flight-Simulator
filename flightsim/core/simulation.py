# flightsim/core/simulation.py
"""Top-level simulation runner."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from flightsim.core.state_eq import make_state_eq
from flightsim.core.integrator import rk4, rk4_step
from flightsim.atmosphere.model import AtmosphereModel
from flightsim.aero.database import AeroDatabase
from flightsim.core.state import StateIndex

from utils.io import AircraftModel

from scipy.optimize import fsolve

import numpy as np
import pandas as pd
from flightsim.core.state import StateIndex

def sample_and_save_loads(t, x, dx, t_start, t_end, filename="loads_data.csv"):
    """
    Slices the simulation data for a specific time window and saves the 
    linear/angular velocities and accelerations to a CSV for structural analysis.
    """
    # 1. Find the indices where time is within our target window
    window_mask = (t >= t_start) & (t <= t_end)
    
    t_window = t[window_mask]
    x_window = x[:, window_mask]
    dx_window = dx[:, window_mask]
    
    # 2. Extract velocities from the state array (x)
    # Adjust these StateIndex names if they differ in your actual class
    u = x_window[StateIndex.U]
    v = x_window[StateIndex.V]
    w = x_window[StateIndex.W]
    p = x_window[StateIndex.P]
    q = x_window[StateIndex.Q]
    r = x_window[StateIndex.R]
    
    # 3. Extract accelerations from the derivative array (dx)
    u_dot = dx_window[StateIndex.U]
    v_dot = dx_window[StateIndex.V]
    w_dot = dx_window[StateIndex.W]
    p_dot = dx_window[StateIndex.P]
    q_dot = dx_window[StateIndex.Q]
    r_dot = dx_window[StateIndex.R]
    
    # 4. Package it into a DataFrame
    data = {
        'time_s': t_window,
        'u_m_s': u, 'v_m_s': v, 'w_m_s': w,
        'p_rad_s': p, 'q_rad_s': q, 'r_rad_s': r,
        'u_dot_m_s2': u_dot, 'v_dot_m_s2': v_dot, 'w_dot_m_s2': w_dot,
        'p_dot_rad_s2': p_dot, 'q_dot_rad_s2': q_dot, 'r_dot_rad_s2': r_dot
    }
    
    df = pd.DataFrame(data)
    
    # 5. Save to disk
    df.to_csv(filename, index=False)
    print(f"Saved {len(df)} samples from t={t_start}s to t={t_end}s into '{filename}'")
    
    return df


def run_simulation(
    model: AircraftModel,
    x0: NDArray,
    trim_controls: dict[str, float],
    t_start: float,
    t_end: float,
    dt: float,
    atmosphere: AtmosphereModel,
) -> tuple[NDArray, NDArray, NDArray]:
    """Runs the 6DOF simulation using RK4 integration.

    Args:
        model: Aircraft model dataclass.
        x0: Initial state vector, shape (12,).
        t_start: Start time (s).
        t_end: End time (s).
        dt: Time step (s).
        atmosphere: AtmosphereModel instance.

    Returns:
        Tuple (t, x, dx) where:
            t:  Time vector, shape (N,).
            x:  State history, shape (12, N).
            dx: State derivative history, shape (12, N).
    """
    t = np.arange(t_start, t_end + dt, dt)
    n = len(t)

    x  = np.zeros((12, n))
    dx = np.zeros((12, n))
    x[:, 0] = x0

    aero_db = AeroDatabase(model.aero_tables_dir)

    elevator_trim, aileron_trim, rudder_trim, throttle_trim = trim_controls["elevator"], trim_controls["aileron"], trim_controls["rudder"], trim_controls["throttle"]

    ail_start   = 5.0
    ail_end     = 6.0
    ail_deflect = 2*0 #4.821695697645064*0

    ele_start   = 5.0
    ele_mid     = 6.0
    ele_end     = 6.0
    ele_deflect = -2.0*0

    rud_start = 5.0
    rud_end = 5.1
    rud_deflect = 10.0*0

    

    current_t = t_start

    def timed_control():

        """if current_t > ele_start and current_t < ele_mid:
            ele = elevator_trim + ele_deflect
        elif current_t > ele_mid and current_t < ele_end:
            ele = elevator_trim -ele_deflect
        else:
            ele = elevator_trim"""
        ele = elevator_trim + ele_deflect if ele_start <= current_t <= ele_end else elevator_trim
        ail = ail_deflect + aileron_trim if ail_start <= current_t <= ail_end else aileron_trim
        rud = rud_deflect + rudder_trim if rud_start <= current_t <= rud_end else rudder_trim

        return ele, ail, rud, throttle_trim, 0.0

    f = make_state_eq(model, aero_db, timed_control, atmosphere)
    
    
    xi  = x0.copy()
    

    dxi = np.zeros(12)

    for i in range(n - 1):
        current_t = t[i]
        rk4_step(f, xi, dxi, dt)
        x[:, i + 1] = xi
        dx[:, i]    = dxi

    dx[:, -1] = dxi

    return t, x, dx
