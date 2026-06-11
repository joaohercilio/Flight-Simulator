# utils/recording.py
"""Post-processing helpers: export simulation histories to disk."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

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
