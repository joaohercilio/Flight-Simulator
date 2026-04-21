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
from flightsim.aero.database import AeroDatabase
from utils.io import AircraftModel


def make_state_eq(
    model: AircraftModel,
    aero_db: AeroDatabase,
    control_input,
    atmosphere: AtmosphereModel
):
    """Builds the RHS of the 6DOF state equation dx/dt = f(x, t).

    Args:
        model: Aircraft model dataclass with inertia and geometry.
        aero_db: Aerodynamic coefficient database.
        control_input: Callable returning
            (ele_cmd, ail_cmd, rud_cmd, throttle_cmd, brake_cmd).
        atmosphere: AtmosphereModel instance.

    Returns:
        Callable f(x, t) -> dx where x and dx are NDArray of shape (12,).
    """

    arm_z_engine = model.arm_z_engine
    max_brake    = model.brake_max
    bm           = 100.0 / max_brake**2

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
        el  = ele_cmd
        ail = ail_cmd
        rud = rud_cmd

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
        thrust = throttle_cmd*(0.0010482548 * speed**3 - 0.0715234262 * speed**2 - 0.7276455480 * speed + 44.085638000) * rho / 1.225
        thrust = throttle_cmd

        fx = fx + thrust
        pitch_moment = pitch_moment + arm_z_engine * thrust

        # --- gear geometry (relative to CG, body frame) ---
        x_ng = 0.2313
        x_mg = -0.0187
        y_mg = 0.2455

        # Spring/Damper constants (you will need to tune these based on mass)
        k_spring = 5000.0  # N/m
        c_damper = 500.0   # N*s/m

        # --- gear geometry (relative to CG, body frame) ---
        x_ng = 0.2313
        x_mg = -0.0187
        y_mg = 0.2455
        z_gear = 0.15 # Ensure this matches your model!

        # --- DYNAMIC SPRING & DAMPER CALCULATION ---
        total_weight = model.mass * 9.81
        wheelbase = x_ng - x_mg
        
        # Calculate actual weight resting on each gear using lever arms
        weight_ng = total_weight * (abs(x_mg) / wheelbase)
        weight_per_mg = (total_weight * (x_ng / wheelbase)) / 2.0
        
        # Springs (tuned for 3cm / 0.03m compression under static weight)
        k_ng = weight_ng / 0.03
        k_mg = weight_per_mg / 0.03
        
        # Dampers (Critical damping ratio zeta = 1.0)
        c_ng = 2.0 * 1.0 * np.sqrt(k_ng * (weight_ng / 9.81))
        c_mg = 2.0 * 1.0 * np.sqrt(k_mg * (weight_per_mg / 9.81))

        # --- NEW: GROUND INTERACTION ---
        gear_fx, gear_fy, gear_fz = 0.0, 0.0, 0.0
        gear_L, gear_M, gear_N = 0.0, 0.0, 0.0

        # 1. Transform Z-positions to Earth Frame
        z_earth_ng = s.z_e + (-x_ng * sin_tht + z_gear * cos_phi * cos_tht) 
        z_earth_mg_left = s.z_e + (-x_mg * sin_tht - y_mg * sin_phi * cos_tht + z_gear * cos_phi * cos_tht)
        z_earth_mg_right = s.z_e + (-x_mg * sin_tht + y_mg * sin_phi * cos_tht + z_gear * cos_phi * cos_tht)

        # 2. Local Z velocities for damping (V_z = w + p*y - q*x)
        w_ng = s.w - s.q * x_ng
        w_mgl = s.w + s.p * (-y_mg) - s.q * x_mg  # FIXED: + p * y
        w_mgr = s.w + s.p * (y_mg)  - s.q * x_mg  # FIXED: + p * y

        # 3. Calculate Forces and Moments (Nose Gear)
        if z_earth_ng > 0:  
            fz_ng = -k_ng * z_earth_ng - c_ng * w_ng 
            fz_ng = min(fz_ng, 0.0)  
            
            gear_fz += fz_ng
            gear_M += -x_ng * fz_ng  

        # 4. Calculate Forces and Moments (Main Gears)
        # Left
        if z_earth_mg_left > 0:
            fz_mgl = -k_mg * z_earth_mg_left - c_mg * w_mgl
            fz_mgl = min(fz_mgl, 0.0)
            
            gear_fz += fz_mgl
            gear_M += -x_mg * fz_mgl 
            gear_L += -y_mg * fz_mgl  

        # Right
        if z_earth_mg_right > 0:
            fz_mgr = -k_mg * z_earth_mg_right - c_mg * w_mgr
            fz_mgr = min(fz_mgr, 0.0)
            
            gear_fz += fz_mgr
            gear_M += -x_mg * fz_mgr
            gear_L += y_mg * fz_mgr  

        # --- ADD TO EXISTING FORCES ---
        fx += gear_fx
        fz += gear_fz
        fy += gear_fy
        roll_moment += gear_L
        pitch_moment += gear_M
        yaw_moment += gear_N
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
            model.mass, g,
            fx, fy, fz,
            s.u, s.v, s.w,
            s.p, s.q, s.r,
            sin_phi, cos_phi, sin_tht, cos_tht,
        )

        dx[StateIndex.P], dx[StateIndex.Q], dx[StateIndex.R] = rotational_equations(
            model.ix, model.iy, model.iz, model.ixz,
            roll_moment, pitch_moment, yaw_moment,
            s.p, s.q, s.r,
        )

        return dx

    return f
