from __future__ import annotations

import dataclasses
from typing import Callable

import numpy as np
from scipy.optimize import fsolve

from flightsim.control.source import ControlInput, LiveControl
from flightsim.core.dynamics import Dynamics
from flightsim.core.state import StateIndex as I


@dataclasses.dataclass(frozen=True)
class CeilingResult:
    ceiling: float
    alpha_deg: float
    elevator_deg: float
    gamma_deg: float
    table: list[tuple[float, float]]

    def summary(self) -> str:
        return (f"Ceiling ≈ {self.ceiling:.1f} m   alpha {self.alpha_deg:.2f}°   elevator {self.elevator_deg:.2f}°   "
                f"gamma {self.gamma_deg:.2f}°")


def climb_trim(dynamics: Dynamics, V: float, h: float, throttle: float):
    live = LiveControl()
    previous, dynamics.controls = dynamics.controls, live
    try:
        def residuals(g):
            alpha, el, gamma = g
            x = np.zeros(I.SIZE)
            x[I.U], x[I.W], x[I.THETA], x[I.Z_E] = V * np.cos(alpha), V * np.sin(alpha), alpha + gamma, -h
            live.set(ControlInput(elevator=el, throttle=throttle))
            dx = dynamics(x, 0.0)
            return [dx[I.U], dx[I.W], dx[I.Q]]
        sol, _, ier, _ = fsolve(residuals, [0.05, 0.0, 0.0], full_output=True, xtol=1e-10)
    finally:
        dynamics.controls = previous
    return (sol if ier == 1 else None)


def ceiling_sweep(dynamics: Dynamics, V: float, throttle: float, h_max: float = 6000.0, step: float = 25.0,
                  log: Callable[[str], None] | None = None) -> CeilingResult:
    m = dynamics.model
    log = log or (lambda _: None)
    best = CeilingResult(0.0, 0.0, 0.0, 0.0, [])
    table: list[tuple[float, float]] = []
    gear, dynamics.gear = dynamics.gear, None
    try:
        with dynamics.env.still_air():
            for h in np.arange(0.0, h_max + step, step):
                sol = climb_trim(dynamics, V, h, throttle)
                if sol is None or abs(sol[1]) > m.elevator_max or sol[0] > np.radians(m.stall_alpha):
                    log(f"  [{h:7.1f} m] limits exceeded, stopping sweep")
                    break
                alpha, el, gamma = sol
                climb = V * np.sin(gamma)
                table.append((float(h), float(climb)))
                log(f"  [{h:7.1f} m] climb rate {climb:+.3f} m/s")
                best = CeilingResult(float(h), float(np.degrees(alpha)), float(el), float(np.degrees(gamma)), table)
                if climb <= 0.0:
                    break
    finally:
        dynamics.gear = gear
    return best
