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

    #x0[StateIndex.U] = 10

    x[:, 0] = x0

    aero_db = AeroDatabase(model.aero_tables_dir)

    ail_start = 1.0
    ail_end   = 3.0
    ail_deflect = 0.0

    ele_start = 1.0
    ele_end   = 10.0
    ele_deflect = 20.0

    current_t = t_start

    def timed_control():
        ele = ele_deflect if ele_start <= current_t <= ele_end else 0.0
        ail = ail_deflect if ail_start <= current_t <= ail_end else 0.0

        return ele, ail, 0.0, 0.0, 0.0

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
