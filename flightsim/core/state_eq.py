# flightsim/core/state_eq.py
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
    """Builds the RHS of the 6DOF state equation dx/dt = f(x, t)."""

    # ---------------------------------------------------------------
    # 1. Fixed Landing Gear Structural Constants & Baseline Tuning
    # ---------------------------------------------------------------
    POS_MG    = 0.44
    WHEELBASE = 0.306  # This is your B_mg
    Y_MG      = 0.2
    Z_GEAR    = 0.15
    
    # Calculate physical spring/damping constants ONCE using a fixed nominal CG.
    # This prevents the airframe from "auto-softening" when you shift weight.
    NOMINAL_X_CG = 0.3913
    x_ng_nominal = NOMINAL_X_CG - (POS_MG - WHEELBASE)
    x_mg_nominal = NOMINAL_X_CG - POS_MG
    
    total_weight_nominal = model.mass * 9.81
    weight_ng_nominal    = total_weight_nominal * (abs(x_mg_nominal) / WHEELBASE)
    weight_per_mg_nominal = (total_weight_nominal * (x_ng_nominal / WHEELBASE)) / 2.0
    
    k_ng = weight_ng_nominal / 0.03
    k_ng = 30.7e3     
    k_mg = weight_per_mg_nominal / 0.03
    k_mg = 30.7e3     
    c_ng = 2.0 * 1.0 * np.sqrt(k_ng * (weight_ng_nominal / 9.81))
    c_mg = 2.0 * 1.0 * np.sqrt(k_mg * (weight_per_mg_nominal / 9.81))

    # ---------------------------------------------------------------
    # 2. Engine and Brake Configuration
    # ---------------------------------------------------------------
    arm_z_engine = model.arm_z_engine
    max_brake    = model.brake_max
    ground       = model.ground_altitude 

    # Aerodynamic Reference Point (where wind tunnel tables are centered)
    # Fallback to nominal if your TOML doesn't explicitly store x_ref/z_ref
    x_ref = 0.3913
    z_ref = 0.315

    # ---------------------------------------------------------------
    # 3. Runtime State Equation Loop
    # ---------------------------------------------------------------
    def f(raw: NDArray, t: float) -> NDArray:
        s = StateVector(raw)

        # --- Atmosphere ---
        rho = atmosphere.get_density(s.altitude)
        g   = atmosphere.get_gravity(s.altitude)

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
        ele_cmd, ail_cmd, rud_cmd, throttle_cmd, brake_cmd = control_input()

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

        # --- Aerodynamic Reference Point Transformation (Transport Theorem) ---
        # Accounts for the distance between actual flight CG and the aero reference center
        dx_ref = x_ref - model.x_cg  
        dz_ref = z_ref - model.z_cg  
        pitch_moment += (dz_ref * fx - dx_ref * fz)

        # --- Propulsion Forces ---
        a, b, c, d = 0.001274, -0.07204, -0.5428, 40.89
        thrust = throttle_cmd * (a * speed**3 + b * speed**2 + c * speed + d) * rho / 1.225

        fx += thrust
        pitch_moment += arm_z_engine * thrust
        
        # --- Kinematic Wheel Positions Relative to CURRENT Flight CG ---
        x_ng = model.x_cg - (POS_MG - WHEELBASE)
        x_mg = model.x_cg - POS_MG

        # --- Landing Gear Interaction ---
        gear_fx, gear_fy, gear_fz = 0.0, 0.0, 0.0
        gear_L,  gear_M,  gear_N  = 0.0, 0.0, 0.0

        z_earth_ng       = s.z_e + (-x_ng * sin_tht + Z_GEAR * cos_phi * cos_tht)
        z_earth_mg_left  = s.z_e + (-x_mg * sin_tht - Y_MG * sin_phi * cos_tht + Z_GEAR * cos_phi * cos_tht)
        z_earth_mg_right = s.z_e + (-x_mg * sin_tht + Y_MG * sin_phi * cos_tht + Z_GEAR * cos_phi * cos_tht)

        zdot_body = -s.u * sin_tht + s.v * sin_phi * cos_tht + s.w * cos_phi * cos_tht

        zdot_ng  = zdot_body - s.q * x_ng * cos_tht
        zdot_mgl = zdot_body - s.q * x_mg * cos_tht + s.p * (-Y_MG) * cos_phi * cos_tht
        zdot_mgr = zdot_body - s.q * x_mg * cos_tht + s.p * ( Y_MG) * cos_phi * cos_tht

        def apply_gear_force(x, y, z, fz_earth):
            if fz_earth >= 0:
                return
            fx_b =  -fz_earth * sin_tht
            fy_b =   fz_earth * sin_phi * cos_tht
            fz_b =   fz_earth * cos_phi * cos_tht
            nonlocal gear_fx, gear_fy, gear_fz, gear_L, gear_M, gear_N
            gear_fx += fx_b
            gear_fy += fy_b
            gear_fz += fz_b
            gear_L  += y * fz_b - z * fy_b
            gear_M  += z * fx_b - x * fz_b
            gear_N  += x * fy_b - y * fx_b

        fz_total_earth = 0.0

        if z_earth_ng - ground > 0:
            penetration_ng = z_earth_ng - ground
            fz_ng_earth = -k_ng * penetration_ng - c_ng * zdot_ng
            apply_gear_force(x_ng, 0.0, Z_GEAR, fz_ng_earth)
            fz_total_earth += abs(fz_ng_earth)

        if z_earth_mg_left - ground > 0:
            penetration_mgl = z_earth_mg_left - ground
            fz_mgl_earth = -k_mg * penetration_mgl - c_mg * zdot_mgl
            apply_gear_force(x_mg, -Y_MG, Z_GEAR, fz_mgl_earth)
            fz_total_earth += abs(fz_mgl_earth)

        if z_earth_mg_right - ground > 0:
            penetration_mgr = z_earth_mg_right - ground
            fz_mgr_earth = -k_mg * penetration_mgr - c_mg * zdot_mgr
            apply_gear_force(x_mg, Y_MG, Z_GEAR, fz_mgr_earth)
            fz_total_earth += abs(fz_mgr_earth)

        mu_roll = 0.04
        mu_brake = 0.0
        mu_eff = mu_roll + mu_brake * (brake_cmd / max_brake)
        friction_force = -mu_eff * fz_total_earth * np.sign(s.u)
        gear_fx += friction_force * cos_tht

        fx += gear_fx
        fy += gear_fy
        fz += gear_fz
        roll_moment  += gear_L
        pitch_moment += gear_M
        yaw_moment   += gear_N
        
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

    return f