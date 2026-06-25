# flightsim/core/simulation.py
"""Top-level simulation engine.

``Simulator`` owns the model, dynamics and current state. It exposes
two modes that share the exact same physics:


"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from flightsim.control.source import ControlSource
from flightsim.core.dynamics import Dynamics
from flightsim.core.integrator import rk4_step
from flightsim.atmosphere.model import AtmosphereModel
from flightsim.aero.database import AeroDatabase
from flightsim.core.state import StateIndex

from flightsim.aircraft import AircraftModel


class Simulator:
    """Stateful 6DOF simulator integrated with RK4.

    Args:
        model: Aircraft model dataclass.
        atmosphere: AtmosphereModel instance.
        controls: Control source queried each step.
        aero_db: Optional pre-loaded aero database. Built from the model
            if omitted (loading the .dat tables is expensive, so reuse one
            instance across trim + simulation when you can).
    """

    def __init__(
        self,
        model: AircraftModel,
        atmosphere: AtmosphereModel,
        controls: ControlSource,
        aero_db: AeroDatabase | None = None,
    ) -> None:
        self.model = model
        self.atmosphere = atmosphere
        self.aero_db = aero_db or AeroDatabase(model.aero_tables_dir)
        self.dynamics = Dynamics(model, self.aero_db, controls, atmosphere)

        self.state = np.zeros(StateIndex.SIZE)
        self.deriv = np.zeros(StateIndex.SIZE)
        self.t = 0.0

    @property
    def controls(self) -> ControlSource:
        """The active control source. Reassign to swap inputs live."""
        return self.dynamics.controls

    @controls.setter
    def controls(self, source: ControlSource) -> None:
        self.dynamics.controls = source

    def reset(self, x0: NDArray, t0: float = 0.0) -> None:
        """Resets the engine to an initial state.

        Args:
            x0: Initial state vector, shape (12,).
            t0: Initial simulation time (s).
        """
        self.state = np.asarray(x0, dtype=float).copy()
        self.deriv = np.zeros(StateIndex.SIZE)
        self.t = t0

    def step(self, dt: float) -> tuple[NDArray, NDArray]:
        """Advances the simulation by one RK4 step in place.

        Args:
            dt: Time step (s).

        Returns:
            Tuple (state, deriv) at the new time. Both are views into
            the engine's internal arrays — copy if you need to retain them.
        """
        rk4_step(self.dynamics, self.state, self.deriv, dt, self.t)
        self.t += dt
        return self.state, self.deriv

    def run(
        self,
        t_start: float,
        t_end: float,
        dt: float,
        x0: NDArray | None = None,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Integrates over a window and returns full histories.

        Args:
            t_start: Start time (s).
            t_end: End time (s).
            dt: Time step (s).
            x0: Optional initial state. If given, the engine is reset to it.

        Returns:
            Tuple (t, x, dx) where:
                t:  Time vector, shape (N,).
                x:  State history, shape (12, N).
                dx: State derivative history, shape (12, N).
        """
        if x0 is not None:
            self.reset(x0, t_start)
        else:
            self.t = t_start

        t = np.arange(t_start, t_end + dt, dt)
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
