from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from flightsim.aero.database import AeroDatabase
from flightsim.aero.forces import aerodynamic_force_body, aerodynamic_force_wind
from flightsim.aircraft import AircraftModel
from flightsim.control.source import ControlInput, ControlSource
from flightsim.core.equations import (
    kinematic_equations,
    navigation_equations,
    rotational_equations,
    translational_equations,
)
from flightsim.core.gear import LandingGear
from flightsim.core.state import StateIndex, StateVector
from flightsim.environment import Environment


class Dynamics:
    def __init__(self, model: AircraftModel, aero_db: AeroDatabase, controls: ControlSource,
                 env: Environment) -> None:
        self.model = model
        self.aero_db = aero_db
        self.controls = controls
        self.env = env
        self.gear = LandingGear(model, env.gravity) if env.ground_contact else None
        self.on_ground = False

    def limit(self, cmd: ControlInput) -> ControlInput:
        m = self.model
        return ControlInput(
            float(np.clip(cmd.elevator, -m.elevator_max, m.elevator_max)),
            float(np.clip(cmd.aileron, -m.aileron_max, m.aileron_max)),
            float(np.clip(cmd.rudder, -m.rudder_max, m.rudder_max)),
            float(np.clip(cmd.throttle, 0.0, 1.0)),
            float(np.clip(cmd.brake, 0.0, 1.0)),
        )

    def thrust(self, speed: float, rho: float, throttle: float) -> float:
        a, b, c, d = self.model.thrust_coeffs
        return throttle * (a * speed**3 + b * speed**2 + c * speed + d) * rho / 1.225

    def airdata(self, s: StateVector, t: float, sin_phi, cos_phi, sin_tht, cos_tht, sin_psi, cos_psi):
        u, v, w = s.u, s.v, s.w
        wind = self.env.wind.ned(t)
        if wind.any():
            c_bn = np.array([
                [cos_tht * cos_psi, cos_tht * sin_psi, -sin_tht],
                [sin_phi * sin_tht * cos_psi - cos_phi * sin_psi, sin_phi * sin_tht * sin_psi + cos_phi * cos_psi, sin_phi * cos_tht],
                [cos_phi * sin_tht * cos_psi + sin_phi * sin_psi, cos_phi * sin_tht * sin_psi - sin_phi * cos_psi, cos_phi * cos_tht],
            ])
            u, v, w = np.array([u, v, w]) - c_bn @ wind
        speed = max(np.sqrt(u**2 + v**2 + w**2), 1e-8)
        alpha = np.arctan2(w, u)
        beta = np.arcsin(np.clip(v / speed, -1.0, 1.0))
        return speed, alpha, beta

    def __call__(self, raw: NDArray, t: float) -> NDArray:
        s = StateVector(raw)
        model = self.model
        sin_phi, cos_phi = np.sin(s.phi), np.cos(s.phi)
        sin_tht, cos_tht = np.sin(s.theta), np.cos(s.theta)
        tan_tht = np.tan(s.theta)
        sin_psi, cos_psi = np.sin(s.psi), np.cos(s.psi)

        rho = self.env.density.get_density(s.altitude)
        g = self.env.gravity
        speed, alpha, beta = self.airdata(s, t, sin_phi, cos_phi, sin_tht, cos_tht, sin_psi, cos_psi)
        dyn_pres = 0.5 * rho * speed**2

        cmd = self.limit(self.controls.get(t))
        drag, lift, side, l, m, n = aerodynamic_force_wind(
            model, self.aero_db, alpha, beta, s.p, s.q, s.r,
            cmd.elevator, cmd.aileron, cmd.rudder, speed, dyn_pres)
        fx, fy, fz = aerodynamic_force_body(drag, lift, side, np.sin(alpha), np.cos(alpha), np.sin(beta), np.cos(beta))

        thrust = self.thrust(speed, rho, cmd.throttle)
        fx += thrust
        m += model.arm_z_engine * thrust

        if self.gear is not None:
            gfx, gfy, gfz, gl, gm, gn, self.on_ground = self.gear.forces(
                s, self.env.ground_z, cmd.brake, sin_phi, cos_phi, sin_tht, cos_tht)
            fx, fy, fz, l, m, n = fx + gfx, fy + gfy, fz + gfz, l + gl, m + gm, n + gn

        dx = np.zeros(StateIndex.SIZE)
        dx[0:3] = navigation_equations(s.u, s.v, s.w, sin_phi, cos_phi, sin_tht, cos_tht, sin_psi, cos_psi)
        dx[3:6] = kinematic_equations(s.p, s.q, s.r, sin_phi, cos_phi, cos_tht, tan_tht)
        dx[6:9] = translational_equations(model.mass, g, fx, fy, fz, s.u, s.v, s.w, s.p, s.q, s.r,
                                          sin_phi, cos_phi, sin_tht, cos_tht)
        dx[9:12] = rotational_equations(model.ix, model.iy, model.iz, model.ixz, l, m, n, s.p, s.q, s.r)
        return dx
