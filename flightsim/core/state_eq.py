# flightsim/core/state_eq.py
"""Builds the 6DOF state equation closure."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from flightsim.core.state import StateIndex, StateVector
from flightsim.core.equations import (
    navigation_equations,
    kinematic_equations,
    translational_equations,
    rotational_equations,
)
from flightsim.aero.forces import aerodynamic_force_wind, aerodynamic_force_body
from flightsim.atmosphere.model import AtmosphereModel


def make_state_eq(model: AircraftModel, aero_db: AeroDatabase, control_input, atmosphere: AtmosphereModel):
    """Builds the RHS of the 6DOF state equation dx/dt = f(x, t).

    Args:
        model: Aircraft model dict with keys: mass, Ix, Iy, Iz, Ixz, b, c, S.
        control_input: Callable returning
            (ele_cmd, ail_cmd, rud_cmd, throttle_cmd, brake_cmd).
        atmosphere: AtmosphereModel instance.

    Returns:
        Callable f(x, t) -> dx where x and dx are NDArray of shape (12,).
    """
    arm_z_engine = -0.05  # distance from CG to thrust line (m), positive down

    def f(raw: NDArray, t: float) -> NDArray:
        s = StateVector(raw)

        # --- atmosphere ---
        rho = atmosphere.get_density(s.altitude)
        g   = atmosphere.get_gravity(s.altitude)

        # --- airspeed and dynamic pressure ---
        speed = max(np.sqrt(s.u**2 + s.v**2 + s.w**2), 1e-8)
        dyn_pres = 0.5 * rho * speed**2

        # --- aerodynamic angles ---
        if speed < 1e-8:
            alpha, beta = 0.0, 0.0
        else:
            alpha = np.arctan2(s.w, s.u)
            beta  = np.arcsin(s.v / speed)

        # --- trig pre-computation ---
        sin_phi, cos_phi = np.sin(s.phi), np.cos(s.phi)
        sin_tht, cos_tht = np.sin(s.theta), np.cos(s.theta)
        tan_tht          = np.tan(s.theta)
        sin_psi, cos_psi = np.sin(s.psi), np.cos(s.psi)
        sin_alpha, cos_alpha = np.sin(alpha), np.cos(alpha)
        sin_beta,  cos_beta  = np.sin(beta),  np.cos(beta)

        # --- control surface deflections ---
        ele_cmd, ail_cmd, rud_cmd, throttle_cmd, brake_cmd = control_input()
        el  = 10.0 * ele_cmd
        ail = 30.0 * ail_cmd
        rud = 10.0 * rud_cmd

        # --- aerodynamic forces and moments (wind axes) ---
        drag, lift, side, roll_moment, pitch_moment, yaw_moment = aerodynamic_force_wind(
            model, aero_db, alpha, beta,
            s.p, s.q, s.r,
            el, ail, rud,
            speed, dyn_pres,
        )

        # --- body-axis forces ---
        fx, fy, fz = aerodynamic_force_body(
            drag, lift, side,
            sin_alpha, cos_alpha,
            sin_beta, cos_beta,
        )

        # --- propulsion and braking ---
        thrust = throttle_cmd*(-0.0005 * speed**3 - 0.0053 * speed**2 - 0.928 * speed + 38.37) * rho / 1.225
        max_brake = 200.0
        bm = 100.0 / max_brake**2
        brake  = (brake_cmd * max_brake) ** 2
        fx = fx + thrust - brake
        pitch_moment = pitch_moment + arm_z_engine * thrust

        # --- state derivatives ---
        dx = np.zeros(StateIndex.SIZE)

        dx[StateIndex.X_E], dx[StateIndex.Y_E], dx[StateIndex.Z_E] = navigation_equations(
                s.u, s.v, s.w,
                sin_phi, cos_phi,
                sin_tht, cos_tht,
                sin_psi, cos_psi,
        )

        dx[StateIndex.PHI], dx[StateIndex.THETA], dx[StateIndex.PSI] = kinematic_equations(
            s.p, s.q, s.r,
            sin_phi, cos_phi,
            cos_tht, tan_tht
        )

        dx[StateIndex.U], dx[StateIndex.V], dx[StateIndex.W] = translational_equations(
            model["mass"], g,
            fx, fy, fz,
            s.u, s.v, s.w,
            s.p, s.q, s.r,
            sin_phi, cos_phi, sin_tht, cos_tht,
        )

        dx[StateIndex.P], dx[StateIndex.Q], dx[StateIndex.R] = rotational_equations(
            model["Ix"], model["Iy"], model["Iz"], model["Ixz"],
            roll_moment, pitch_moment, yaw_moment,
            s.p, s.q, s.r,
        )

        return dx

    return f
