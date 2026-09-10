from __future__ import annotations

import dataclasses
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from flightsim.core.dynamics import FORCE_NAMES, Dynamics
from flightsim.core.integrator import rk4_step
from flightsim.core.state import StateIndex


@dataclasses.dataclass
class SimulationResult:
    t: NDArray
    x: NDArray
    dx: NDArray
    u: NDArray
    f: NDArray

    @property
    def airspeed(self) -> NDArray:
        return np.sqrt(np.sum(self.x[StateIndex.U:StateIndex.W + 1] ** 2, axis=0))

    @property
    def alpha(self) -> NDArray:
        return np.arctan2(self.x[StateIndex.W], self.x[StateIndex.U])

    @property
    def beta(self) -> NDArray:
        return np.arcsin(np.clip(self.x[StateIndex.V] / np.maximum(self.airspeed, 1e-8), -1.0, 1.0))

    def window(self, t_start: float, t_end: float) -> SimulationResult:
        mask = (self.t >= t_start) & (self.t <= t_end)
        return SimulationResult(self.t[mask], self.x[:, mask], self.dx[:, mask], self.u[:, mask], self.f[:, mask])

    def force(self, name: str) -> NDArray:
        return self.f[FORCE_NAMES.index(name)]


class Simulator:
    def __init__(self, dynamics: Dynamics, x0: NDArray, t0: float = 0.0) -> None:
        self.dynamics = dynamics
        self.x = np.array(x0, dtype=float)
        self.dx = np.zeros(StateIndex.SIZE)
        self.forces = np.zeros(len(FORCE_NAMES))
        self.t = t0

    def evaluate(self) -> None:
        self.dx[:] = self.dynamics(self.x, self.t)
        self.forces[:] = self.dynamics.forces

    def step(self, dt: float) -> None:
        self.dynamics.env.wind.update(self.t)
        self.dynamics.controls.poll()
        self.evaluate()
        rk4_step(self.dynamics, self.x, self.dx, self.t, dt)
        self.t += dt

    def run(self, t_end: float, dt: float, progress: Callable[[float], None] | None = None) -> SimulationResult:
        n = int(round((t_end - self.t) / dt)) + 1
        t = self.t + dt * np.arange(n)
        x = np.zeros((StateIndex.SIZE, n))
        dx = np.zeros((StateIndex.SIZE, n))
        u = np.zeros((5, n))
        f = np.zeros((len(FORCE_NAMES), n))
        x[:, 0] = self.x
        for i in range(n - 1):
            u[:, i] = self.dynamics.limit(self.dynamics.controls.get(self.t)).as_tuple()
            self.step(dt)
            x[:, i + 1] = self.x
            dx[:, i] = self.dx
            f[:, i] = self.forces
            if progress and i % 200 == 0:
                progress(i / (n - 1))
        self.evaluate()
        dx[:, -1], f[:, -1] = self.dx, self.forces
        u[:, -1] = self.dynamics.limit(self.dynamics.controls.get(self.t)).as_tuple()
        if progress:
            progress(1.0)
        return SimulationResult(t, x, dx, u, f)
