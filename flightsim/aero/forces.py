# flightsim/aero/forces.py
"""Aerodynamic force and moment calculations."""

from __future__ import annotations

from flightsim.aero.database import AeroDatabase
from utils.io import AircraftModel


def aerodynamic_force_body(
    drag: float,
    lift: float,
    side: float,
    sin_alpha: float,
    cos_alpha: float,
    sin_beta: float,
    cos_beta: float,
) -> tuple[float, float, float]:
    """Converts aerodynamic forces from wind to body axes.

    Args:
        drag: Drag force (N).
        lift: Lift force (N).
        side: Side force (N).
        sin_alpha: Sine of angle of attack.
        cos_alpha: Cosine of angle of attack.
        sin_beta: Sine of sideslip angle.
        cos_beta: Cosine of sideslip angle.

    Returns:
        Tuple (fx, fy, fz) of body-axis forces in Newtons.
    """
    fx = -(drag * cos_alpha * cos_beta - lift * sin_alpha + side * cos_alpha * sin_beta)
    fy = -(drag * sin_beta) + side * cos_beta
    fz = -(drag * sin_alpha * cos_beta - side * sin_alpha * sin_beta + lift * cos_alpha)
    return fx, fy, fz


def aerodynamic_force_wind(
    model: AircraftModel,
    aero_db: AeroDatabase,
    alpha: float,
    beta: float,
    p: float,
    q: float,
    r: float,
    el: float,
    ail: float,
    rud: float,
    speed: float,
    dyn_pres: float,
) -> tuple[float, float, float, float, float, float]:
    """Computes aerodynamic forces and moments in wind axes.

    Args:
        model: Aircraft geometry parameters.
        aero_db: Aerodynamic coefficient database.
        alpha: Angle of attack (rad).
        beta: Sideslip angle (rad).
        p: Roll rate (rad/s).
        q: Pitch rate (rad/s).
        r: Yaw rate (rad/s).
        el: Elevator deflection (deg).
        ail: Aileron deflection (deg).
        rud: Rudder deflection (deg).
        speed: Airspeed (m/s).
        dyn_pres: Dynamic pressure (Pa).

    Returns:
        Tuple (drag, lift, side, roll_moment, pitch_moment, yaw_moment).
        Forces in N, moments in N·m.
    """
    half_b_v = model.b / (2 * speed)
    half_c_v = model.c / (2 * speed)
    gc = aero_db.get_coeff
    cl0 = gc("CL0", alpha, beta)
    clmax = 2.00945
    #if cl0 > clmax:
    #    cl0 = 0.0
    cl = cl0 + gc("CL_el", alpha, beta) * el + gc("CL_q",  alpha, beta) * q * half_c_v

    cd = gc("CD0", alpha, beta) + gc("CD_el", alpha, beta) * el

    cy = (
        gc("CY0",    alpha, beta)
        + gc("CY_p",   alpha, beta) * p * half_b_v
        + gc("CY_r",   alpha, beta) * r * half_b_v
        + gc("CY_ail", alpha, beta) * ail
        + gc("CY_rud", alpha, beta) * rud
    )

    c_roll = (
        gc("Croll0",    alpha, beta)
        + gc("Croll_p",   alpha, beta) * p * half_b_v
        + gc("Croll_r",   alpha, beta) * r * half_b_v
        + gc("Croll_ail", alpha, beta) * ail
        + gc("Croll_rud", alpha, beta) * rud
    )

    cm = (
        gc("Cm0",  alpha, beta)
        + gc("Cm_q",  alpha, beta) * q * half_c_v
        + gc("Cm_el", alpha, beta) * el
    )

    cn = (
        gc("Cn0",    alpha, beta)
        + gc("Cn_p",   alpha, beta) * p * half_b_v
        + gc("Cn_r",   alpha, beta) * r * half_b_v
        + gc("Cn_ail", alpha, beta) * ail
        + gc("Cn_rud", alpha, beta) * rud
    )

    drag         = cd * dyn_pres * model.s
    lift         = cl * dyn_pres * model.s
    side         = cy * dyn_pres * model.s
    roll_moment  = c_roll * dyn_pres * model.s * model.b
    pitch_moment = cm * dyn_pres * model.s * model.c
    yaw_moment   = cn * dyn_pres * model.s * model.b

    return drag, lift, side, roll_moment, pitch_moment, yaw_moment
