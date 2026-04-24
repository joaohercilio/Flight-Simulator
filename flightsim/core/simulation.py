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
from flightsim.core.state import StateIndex # Assuming this is where it lives

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

# --- How to use it after your simulation runs ---
# t, x, dx = run_simulation(...)
# df_loads = sample_and_save_loads(t, x, dx, t_start=5.0, t_end=6.5, filename="aileron_kick_loads.csv")

def compute_trim(model, aero_db, g, V, rho):
    """
    Find trim (alpha, elevator, thrust) such that
    u_dot = w_dot = q_dot = 0 at straight and level flight.
    
    Returns alpha_trim (rad), elevator_trim (deg), thrust_trim (N)
    """
    
    W   = model.mass * g
    Sref = model.s
    dyn_pres = 0.5 * rho * V**2

    def residuals(x):
        alpha_rad = x[0]
        el_deg    = x[1]
        thrust    = x[2]

        u = V * np.cos(alpha_rad)
        w = V * np.sin(alpha_rad)

        half_c_v = model.c / (2 * V)

        # Aero coefficients at trim (p=q=r=0, beta=0)
        cl = (aero_db.get_coeff("CL0",  alpha_rad, 0.0)
            + aero_db.get_coeff("CL_el", alpha_rad, 0.0) * el_deg)
        cd = (aero_db.get_coeff("CD0",  alpha_rad, 0.0)
            + aero_db.get_coeff("CD_el", alpha_rad, 0.0) * el_deg)
        cm = (aero_db.get_coeff("Cm0",  alpha_rad, 0.0)
            + aero_db.get_coeff("Cm_el", alpha_rad, 0.0) * el_deg)
        # q=0 so no Cmq term

        lift = cl * dyn_pres * Sref
        drag = cd * dyn_pres * Sref
        pitch_moment = cm * dyn_pres * Sref * model.c

        # Wind to body axis forces
        sin_a, cos_a = np.sin(alpha_rad), np.cos(alpha_rad)
        fx_aero = -(drag * cos_a - lift * sin_a)
        fz_aero = -(drag * sin_a + lift * cos_a)

        # Thrust along body x-axis
        fx_total = fx_aero + thrust
        fz_total = fz_aero

        # Pitch moment from thrust arm
        pitch_total = pitch_moment + model.arm_z_engine * thrust

        # Trim residuals: u_dot=0, w_dot=0, q_dot=0
        # translational (gravity projected into body axes, theta=alpha at trim)
        sin_tht, cos_tht = sin_a, cos_a   # theta = alpha in level flight
        res_u = fx_total / model.mass - g * sin_tht          # u_dot = 0
        res_w = fz_total / model.mass + g * cos_tht          # w_dot = 0
        res_q = pitch_total / model.iy                        # q_dot = 0

        return [res_u, res_w, res_q]

    # Initial guess from AVL or rough estimate
    alpha0   = np.deg2rad(2.0)
    el0      = 2.0          # deg
    thrust0  = W            # rough: thrust ~ weight for slow UAV

    x_trim, info, ier, msg = fsolve(
        residuals,
        x0=[alpha0, el0, thrust0],
        full_output=True,
        xtol=1e-10,
        
    )

    if ier != 1:
        print(f"WARNING: trim did not converge — {msg}")

    alpha_trim  = x_trim[0]
    el_trim     = x_trim[1]
    thrust_trim = x_trim[2]

    res = residuals(x_trim)
    print(f"\n── Trim solution ──────────────────────────────")
    print(f"  α_trim    = {np.rad2deg(alpha_trim):.4f} °")
    print(f"  δe_trim   = {el_trim:.4f} °")
    print(f"  T_trim    = {thrust_trim:.4f} N")
    print(f"  Residuals : u_dot={res[0]:.2e}  w_dot={res[1]:.2e}  q_dot={res[2]:.2e}")

    return alpha_trim, el_trim, thrust_trim

def run_simulation(
    model: AircraftModel,
    x0: NDArray,
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
    Va = 16.7126 #Manobra, m/s, valor definido por cargas (?)
    Vc = 18.4860 #Cruzeiro, m/s, valor definido por cargas (?)
    Vd = 23.1075 #Mergulho, m/s, valor definido por cargas (?)
    V = Vd

    alpha_trim, elevator_trim, thrust_trim = compute_trim(
        model,
        aero_db=AeroDatabase(model.aero_tables_dir),
        g=9.81,
        V=V,
        rho=1.1
        )



    x0[StateIndex.U] = V *np.cos(alpha_trim)
    x0[StateIndex.W] = V *np.sin(alpha_trim)
    x0[StateIndex.THETA] = alpha_trim
    

    x[:, 0] = x0

    aero_db = AeroDatabase(model.aero_tables_dir)

    ail_start   = 5.0
    ail_end     = 6.0
    ail_deflect = 4.821695697645064 #18.081358866168987*0 20*0


    ele_start   = 5.0
    ele_mid     = 6.0
    ele_end     = 6.0
    ele_deflect = -25.0*0

    rud_start = 5.0
    rud_end = 6.0
    rud_deflect = 30.0*0

    current_t = t_start

    def timed_control():

        """if current_t > ele_start and current_t < ele_mid:
            ele = elevator_trim + ele_deflect
        elif current_t > ele_mid and current_t < ele_end:
            ele = elevator_trim -ele_deflect
        else:
            ele = elevator_trim"""
        ele = elevator_trim + ele_deflect if ele_start <= current_t <= ele_end else elevator_trim
        ail = ail_deflect if ail_start <= current_t <= ail_end else 0.0
        rud = rud_deflect if rud_start <= current_t <= rud_end else 0.0

        return ele, ail, rud, thrust_trim, 0.0

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
