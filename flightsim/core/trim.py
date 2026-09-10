from __future__ import annotations

import dataclasses

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from flightsim.control.source import ControlInput, LiveControl
from flightsim.core.dynamics import Dynamics
from flightsim.core.state import StateIndex, StateVector
from flightsim.environment import Wind


@dataclasses.dataclass(frozen=True)
class TrimResult:
    condition: str
    x0: NDArray
    controls: ControlInput
    cost: float

    def summary(self) -> str:
        s = StateVector(self.x0)
        alpha = np.degrees(np.arctan2(s.w, s.u))
        return (f"Trim ({self.condition}) cost {self.cost:.2e}\n"
                f"  alpha {alpha:.3f}°  theta {np.degrees(s.theta):.3f}°  phi {np.degrees(s.phi):.3f}°  "
                f"V {s.airspeed:.2f} m/s  h {s.altitude:.1f} m\n"
                f"  {self.controls.summary()}")


class TrimSolver:
    def __init__(self, dynamics: Dynamics) -> None:
        self.dynamics = dynamics
        self._live = LiveControl()
        self._previous = dynamics.controls

    def _state(self, V, h, alpha, phi=0.0, theta=None, psi_dot=0.0) -> NDArray:
        theta = alpha if theta is None else theta
        x = np.zeros(StateIndex.SIZE)
        x[StateIndex.U], x[StateIndex.W] = V * np.cos(alpha), V * np.sin(alpha)
        x[StateIndex.PHI], x[StateIndex.THETA] = phi, theta
        x[StateIndex.P] = -psi_dot * np.sin(theta)
        x[StateIndex.Q] = psi_dot * np.sin(phi) * np.cos(theta)
        x[StateIndex.R] = psi_dot * np.cos(phi) * np.cos(theta)
        x[StateIndex.Z_E] = -h
        return x

    def residual(self, x: NDArray, controls: ControlInput) -> float:
        self._live.set(controls)
        dx = self.dynamics(x, 0.0)
        return float(np.sum(dx[StateIndex.U:StateIndex.R + 1] ** 2))

    def solve(self, condition: str, V: float, h: float, gamma_deg: float = 0.0, radius: float = 0.0) -> TrimResult:
        m = self.dynamics.model
        gamma = np.radians(gamma_deg)
        env = self.dynamics.env
        self.dynamics.controls, wind, env.wind = self._live, env.wind, Wind()
        try:
            if condition == "steady_level_flight":
                gamma = 0.0
            if condition in ("steady_level_flight", "steady_climb"):
                def cost(g):
                    return self.residual(self._state(V, h, g[0], theta=g[0] + gamma), ControlInput(elevator=g[1], throttle=g[2]))
                res = self._minimize(cost, [0.05, 0.0, 0.5], [(-0.17, 0.35), (-m.elevator_max, m.elevator_max), (0.0, 1.0)])
                a, de, thr = res.x
                return TrimResult(condition, self._state(V, h, a, theta=a + gamma), ControlInput(elevator=de, throttle=thr), res.fun)
            if condition == "glide":
                def cost(g):
                    return self.residual(self._state(V, h, g[0], theta=g[2]), ControlInput(elevator=g[1]))
                res = self._minimize(cost, [0.05, 0.0, -0.05], [(-0.17, 0.35), (-m.elevator_max, m.elevator_max), (-0.6, 0.6)])
                a, de, th = res.x
                return TrimResult(condition, self._state(V, h, a, theta=th), ControlInput(elevator=de), res.fun)
            if condition == "coordinated_turn":
                if radius <= 0:
                    raise ValueError("Turn radius must be positive for a coordinated turn")
                psi_dot = V / radius
                def cost(g):
                    a, phi, de, da, dr, thr = g
                    theta = np.arctan(np.tan(a) * np.cos(phi))
                    return self.residual(self._state(V, h, a, phi, theta, psi_dot), ControlInput(de, da, dr, thr))
                phi0 = np.arctan(V**2 / (self.dynamics.env.gravity * radius))
                res = self._minimize(cost, [0.05, phi0, 0.0, 0.0, 0.0, 0.5],
                                     [(-0.1, 0.3), (-1.05, 1.05), (-m.elevator_max, m.elevator_max),
                                      (-m.aileron_max, m.aileron_max), (-m.rudder_max, m.rudder_max), (0.0, 1.0)])
                a, phi, de, da, dr, thr = res.x
                theta = np.arctan(np.tan(a) * np.cos(phi))
                return TrimResult(condition, self._state(V, h, a, phi, theta, psi_dot), ControlInput(de, da, dr, thr), res.fun)
            raise ValueError(f"Unknown trim condition '{condition}'")
        finally:
            self.dynamics.controls, env.wind = self._previous, wind

    @staticmethod
    def _minimize(cost, guess, bounds):
        res = minimize(cost, np.array(guess), method="SLSQP", bounds=bounds, tol=1e-12, options={"maxiter": 500})
        if not res.success and res.fun > 1e-6:
            raise RuntimeError(f"Trim failed to converge: {res.message}")
        return res
