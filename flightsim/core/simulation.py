# flightsim/core/simulation.py
"""Top-level simulation engine.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from flightsim.control.source import ControlSource
from flightsim.core.dynamics import Dynamics
from flightsim.core.integrator import rk4_step
from flightsim.environment.density import DensityModel
from flightsim.environment.gravity import GravityModel
from flightsim.aero.database import AeroDatabase
from flightsim.core.state import StateIndex, StateVector
from flightsim.aircraft import AircraftModel


class Simulator:


    def __init__(
        self,
        model:         AircraftModel,
        controls:      ControlSource,
        aero_db:       AeroDatabase,
        density_model: DensityModel,
        gravity_model: GravityModel,
    ) -> None:
        self.dynamics = Dynamics(model, aero_db, controls, density_model, gravity_model)
        self.state    = np.zeros(StateIndex.SIZE)
        self.deriv    = np.zeros(StateIndex.SIZE)
        self.t        = 0.0


    def step(self, dt: float) -> tuple[NDArray, NDArray]:
        rk4_step(self.dynamics, self.state, self.deriv, dt, self.t)
        self.t += dt
        return self.state, self.deriv


    def run(self, t_end: float, dt: float, x0: StateVector) -> tuple[NDArray, StateVector, StateVector]:
        t = np.arange(0, t_end + dt, dt)
        n = len(t)
        x  = np.zeros((StateIndex.SIZE, n))
        dx = np.zeros((StateIndex.SIZE, n))

        x[:, 0] = self.state

        for i in range(n - 1):
            self.step(dt)
            x[:, i + 1] = self.state
            dx[:, i]    = self.deriv

        dx[:, -1] = self.deriv
        return t, x, dx
