# flightsim/core/dynamics.py
"""6DOF equations of motion as a callable RHS, dx/dt = f(x, t).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from flightsim.control.source import ControlSource
from flightsim.core.state import StateIndex, StateVector
from flightsim.core.equations import (
    navigation_equations,
    kinematic_equations,
    translational_equations,
    rotational_equations,
)
from flightsim.aero.forces import aerodynamic_force_wind, aerodynamic_force_body
from flightsim.environment.density import DensityModel
from flightsim.environment.gravity import GravityModel
from flightsim.aero.database import AeroDatabase
from flightsim.aircraft import AircraftModel


class Dynamics:
    """Right-hand side of the 6DOF state equation.
    """

    def __init__(
        self,
        model: AircraftModel,
        aero_db: AeroDatabase,
        controls: ControlSource,
        density_model: DensityModel,
        gravity_model: GravityModel
    ) -> None:
        self._model = model
        self._aero_db = aero_db
        self._controls = controls
        self._density_model = density_model
        self._gravity_model = gravity_model

        # ---------------------------------------------------------------
        # 1. Fixed Landing Gear Structural Constants & Baseline Tuning
        # ---------------------------------------------------------------
        self.POS_MG    = model.main_gear_x
        self.WHEELBASE = model.wheelbase  # This is your B_mg
        self.Y_MG      = model.main_gear_y
        self.Z_GEAR    = model.gear_height


        # Calculate physical spring/damping constants ONCE using a fixed nominal CG.
        # This prevents the airframe from "auto-softening" when you shift weight.
        NOMINAL_X_CG = model.x_cg
        x_ng_nominal = NOMINAL_X_CG - (self.POS_MG - self.WHEELBASE)
        x_mg_nominal = NOMINAL_X_CG - self.POS_MG

        total_weight_nominal = model.mass * 9.81
        weight_ng_nominal    = total_weight_nominal * (abs(x_mg_nominal) / self.WHEELBASE)
        weight_per_mg_nominal = (total_weight_nominal * (x_ng_nominal / self.WHEELBASE)) / 2.0

        self.k_ng = weight_ng_nominal / 0.03
        self.k_ng = 30.7e3
        self.k_mg = weight_per_mg_nominal / 0.03
        self.k_mg = 30.7e3
        self.c_ng = 2.0 * 1.0 * np.sqrt(self.k_ng * (weight_ng_nominal / 9.81))
        self.c_mg = 2.0 * 1.0 * np.sqrt(self.k_mg * (weight_per_mg_nominal / 9.81))

        self.arm_z_engine = model.arm_z_engine
        self.ground       = model.ground_altitude


    @property
    def controls(self) -> ControlSource:
        """The active control source. Reassign to swap inputs at runtime."""
        return self._controls

    @controls.setter
    def controls(self, source: ControlSource) -> None:
        self._controls = source

    def __call__(self, raw: NDArray, t: float) -> NDArray:
        """Evaluates dx/dt at state ``raw`` and time ``t``.
        """
        model        = self._model
        aero_db      = self._aero_db
        density_model   = self._density_model
        gravity_model = self._gravity_model

        # arm_z_engine = self.arm_z_engine
        # ground       = self.ground
        # k_ng, k_mg   = self.k_ng, self.k_mg
        # c_ng, c_mg   = self.c_ng, self.c_mg
        # POS_MG, WHEELBASE = self.POS_MG, self.WHEELBASE
        # Y_MG, Z_GEAR      = self.Y_MG, self.Z_GEAR

        s = StateVector(raw)

        # --- Atmosphere ---
        rho = density_model.get_density(s.altitude)
        g   = gravity_model.get_gravity(s.altitude)

        # --- Airspeed and Dynamic Pressure ---
        speed = max(np.sqrt(s.u**2 + s.v**2 + s.w**2), 1e-8)
        dyn_pres = 0.5 * rho * speed**2

        # --- Aerodynamic Angles ---
        if speed < 1e-8:
            alpha, beta = 0.0, 0.0
        else:
            alpha = np.arctan2(s.w, s.u)
            beta  = np.arcsin(s.v / speed)

        # --- Trig Pre-computation ---
        sin_phi, cos_phi = np.sin(s.phi), np.cos(s.phi)
        sin_tht, cos_tht = np.sin(s.theta), np.cos(s.theta)
        tan_tht          = np.tan(s.theta)
        sin_psi, cos_psi = np.sin(s.psi), np.cos(s.psi)
        sin_alpha, cos_alpha = np.sin(alpha), np.cos(alpha)
        sin_beta,  cos_beta  = np.sin(beta),  np.cos(beta)

        # --- Control Surface Deflections ---
        ele_cmd, ail_cmd, rud_cmd, throttle_cmd = self._controls.get(t).as_tuple()

        # --- Aerodynamic Forces and Moments (Wind Axes) ---
        drag, lift, side, roll_moment, pitch_moment, yaw_moment = aerodynamic_force_wind(
            model, aero_db, alpha, beta,
            s.p, s.q, s.r,
            ele_cmd, ail_cmd, rud_cmd,
            speed, dyn_pres,
        )

        # --- Body-Axis Aerodynamic Forces ---
        fx, fy, fz = aerodynamic_force_body(
            drag, lift, side,
            sin_alpha, cos_alpha,
            sin_beta, cos_beta,
        )

        # --- Propulsion Forces ---
        # a, b, c, d = self._model.thrust_a, self._model.thrust_a, self._model.thrust_a, self._model.thrust_a
        # thrust = throttle_cmd * (a * speed**3 + b * speed**2 + c * speed + d) * rho / 1.225
        #
        # fx += thrust
        # pitch_moment += arm_z_engine * thrust
        #
        # # --- Kinematic Wheel Positions Relative to CURRENT Flight CG ---
        # x_ng = model.x_cg - (POS_MG - WHEELBASE)
        # x_mg = model.x_cg - POS_MG

        # --- Landing Gear Interaction ---
        # gear_fx, gear_fy, gear_fz = 0.0, 0.0, 0.0
        # gear_L,  gear_M,  gear_N  = 0.0, 0.0, 0.0
        #
        # z_earth_ng       = s.z_e + (-x_ng * sin_tht + Z_GEAR * cos_phi * cos_tht)
        # z_earth_mg_left  = s.z_e + (-x_mg * sin_tht - Y_MG * sin_phi * cos_tht + Z_GEAR * cos_phi * cos_tht)
        # z_earth_mg_right = s.z_e + (-x_mg * sin_tht + Y_MG * sin_phi * cos_tht + Z_GEAR * cos_phi * cos_tht)
        #
        # zdot_body = -s.u * sin_tht + s.v * sin_phi * cos_tht + s.w * cos_phi * cos_tht
        #
        # zdot_ng  = zdot_body - s.q * x_ng * cos_tht
        # zdot_mgl = zdot_body - s.q * x_mg * cos_tht + s.p * (-Y_MG) * cos_phi * cos_tht
        # zdot_mgr = zdot_body - s.q * x_mg * cos_tht + s.p * ( Y_MG) * cos_phi * cos_tht
        #
        # def apply_gear_force(x, y, z, fz_earth):
        #     if fz_earth >= 0:
        #         return
        #     fx_b =  -fz_earth * sin_tht
        #     fy_b =   fz_earth * sin_phi * cos_tht
        #     fz_b =   fz_earth * cos_phi * cos_tht
        #     nonlocal gear_fx, gear_fy, gear_fz, gear_L, gear_M, gear_N
        #     gear_fx += fx_b
        #     gear_fy += fy_b
        #     gear_fz += fz_b
        #     gear_L  += y * fz_b - z * fy_b
        #     gear_M  += z * fx_b - x * fz_b
        #     gear_N  += x * fy_b - y * fx_b
        #
        # fz_total_earth = 0.0
        #
        # if z_earth_ng - ground > 0:
        #     penetration_ng = z_earth_ng - ground
        #     fz_ng_earth = -k_ng * penetration_ng - c_ng * zdot_ng
        #     apply_gear_force(x_ng, 0.0, Z_GEAR, fz_ng_earth)
        #     fz_total_earth += abs(fz_ng_earth)
        #
        # if z_earth_mg_left - ground > 0:
        #     penetration_mgl = z_earth_mg_left - ground
        #     fz_mgl_earth = -k_mg * penetration_mgl - c_mg * zdot_mgl
        #     apply_gear_force(x_mg, -Y_MG, Z_GEAR, fz_mgl_earth)
        #     fz_total_earth += abs(fz_mgl_earth)
        #
        # if z_earth_mg_right - ground > 0:
        #     penetration_mgr = z_earth_mg_right - ground
        #     fz_mgr_earth = -k_mg * penetration_mgr - c_mg * zdot_mgr
        #     apply_gear_force(x_mg, Y_MG, Z_GEAR, fz_mgr_earth)
        #     fz_total_earth += abs(fz_mgr_earth)
        #
        # mu_roll = 0.04
        # mu_brake = 0.0
        # mu_eff = mu_roll + mu_brake * (brake_cmd)
        # friction_force = -mu_eff * fz_total_earth * np.sign(s.u)
        # gear_fx += friction_force * cos_tht
        #
        # fx += gear_fx
        # fy += gear_fy
        # fz += gear_fz
        # roll_moment  += gear_L
        # pitch_moment += gear_M
        # yaw_moment   += gear_N

        # --- State Derivative Compilation ---
        dx = np.zeros(StateIndex.SIZE)

        dx[StateIndex.X_E], dx[StateIndex.Y_E], dx[StateIndex.Z_E] = navigation_equations(
                s.u, s.v, s.w, sin_phi, cos_phi, sin_tht, cos_tht, sin_psi, cos_psi,
        )

        dx[StateIndex.PHI], dx[StateIndex.THETA], dx[StateIndex.PSI] = kinematic_equations(
            s.p, s.q, s.r, sin_phi, cos_phi, cos_tht, tan_tht
        )

        dx[StateIndex.U], dx[StateIndex.V], dx[StateIndex.W] = translational_equations(
            model.mass, g, fx, fy, fz, s.u, s.v, s.w, s.p, s.q, s.r,
            sin_phi, cos_phi, sin_tht, cos_tht,
        )

        dx[StateIndex.P], dx[StateIndex.Q], dx[StateIndex.R] = rotational_equations(
            model.ix, model.iy, model.iz, model.ixz,
            roll_moment, pitch_moment, yaw_moment, s.p, s.q, s.r,
        )

        return dx
