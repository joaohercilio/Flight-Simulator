from __future__ import annotations

import numpy as np

from flightsim.aircraft import AircraftModel


class LandingGear:
    def __init__(self, model: AircraftModel, gravity: float) -> None:
        self.model = model
        x_ng = model.x_cg - (model.main_gear_x - model.wheelbase)
        x_mg = model.x_cg - model.main_gear_x
        weight = model.mass * gravity
        w_ng = weight * abs(x_mg) / model.wheelbase
        w_mg = weight * x_ng / model.wheelbase / 2.0
        k = model.gear_stiffness
        self.wheels = [
            (x_ng, 0.0, k, 2.0 * model.damping_ratio * np.sqrt(k * w_ng / gravity)),
            (x_mg, -model.main_gear_y, k, 2.0 * model.damping_ratio * np.sqrt(k * w_mg / gravity)),
            (x_mg, model.main_gear_y, k, 2.0 * model.damping_ratio * np.sqrt(k * w_mg / gravity)),
        ]

    def forces(self, s, ground_z: float, brake: float, sin_phi, cos_phi, sin_tht, cos_tht):
        z_gear = self.model.gear_height
        fx = fy = fz = l = m = n = 0.0
        normal = 0.0
        zdot_body = -s.u * sin_tht + s.v * sin_phi * cos_tht + s.w * cos_phi * cos_tht
        for x, y, k, c in self.wheels:
            z_earth = s.z_e - x * sin_tht + y * sin_phi * cos_tht + z_gear * cos_phi * cos_tht
            penetration = z_earth - ground_z
            if penetration <= 0.0:
                continue
            zdot = zdot_body - s.q * x * cos_tht + s.p * y * cos_phi * cos_tht
            f_earth = -k * penetration - c * zdot
            if f_earth >= 0.0:
                continue
            fx_b = -f_earth * sin_tht
            fy_b = f_earth * sin_phi * cos_tht
            fz_b = f_earth * cos_phi * cos_tht
            fx += fx_b
            fy += fy_b
            fz += fz_b
            l += y * fz_b - z_gear * fy_b
            m += z_gear * fx_b - x * fz_b
            n += x * fy_b - y * fx_b
            normal += abs(f_earth)
        if normal > 0.0:
            mu = self.model.rolling_friction + self.model.brake_friction * brake
            fx += -mu * normal * np.tanh(s.u / 0.5) * cos_tht
        return fx, fy, fz, l, m, n, normal > 0.0
