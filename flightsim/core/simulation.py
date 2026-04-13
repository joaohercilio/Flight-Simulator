# flightsim/core/simulation.py
"""Top-level simulation runner."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from flightsim.core.state_eq import make_state_eq
from flightsim.core.integrator import rk4
from flightsim.atmosphere.model import AtmosphereModel
from flightsim.aero.database import AeroDatabase


def run_simulation(
    model: dict,
    x0: NDArray,
    t_start: float,
    t_end: float,
    dt: float,
    atmosphere: AtmosphereModel,
) -> tuple[NDArray, NDArray, NDArray]:
    """Runs the 6DOF simulation using RK4 integration.

    Args:
        model: Aircraft model dict.
        x0: Initial state vector, shape (12,).
        t_start: Start time (s).
        t_end: End time (s).
        dt: Time step (s).
        atmosphere: AtmosphereModel instance. Creates a default one if None.

    Returns:
        Tuple (t, x, dx) where:
            t:  Time vector, shape (N,).
            x:  State history, shape (12, N).
            dx: State derivative history, shape (12, N).
    """

    t  = np.arange(t_start, t_end + dt, dt)
    x  = np.zeros((12, len(t)))
    dx = np.zeros((12, len(t)))

    x[:, 0] = x0

    def zero_control():
        return 0.0, 0.0, 0.0, 0.0, 0.0

    aero_db = AeroDatabase(model.aero_tables_dir)

    f = make_state_eq(model, aero_db, zero_control, atmosphere)

    rk4(f, x, dx, t, dt)

    return t, x, dx
