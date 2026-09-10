from __future__ import annotations

import numpy as np

from flightsim.aero.database import AeroDatabase
from flightsim.aircraft import AircraftModel


def aerodynamic_force_body(drag, lift, side, sin_alpha, cos_alpha, sin_beta, cos_beta):
    fx = -(drag * cos_alpha * cos_beta - lift * sin_alpha + side * cos_alpha * sin_beta)
    fy = -(drag * sin_beta) + side * cos_beta
    fz = -(drag * sin_alpha * cos_beta + side * sin_alpha * sin_beta + lift * cos_alpha)
    return fx, fy, fz


def aerodynamic_force_wind(model: AircraftModel, aero_db: AeroDatabase, alpha, beta, p, q, r, el, ail, rud,
                           speed, dyn_pres):
    half_b_v = model.b / (2 * speed)
    half_c_v = model.c / (2 * speed)
    k = aero_db.coefficients(alpha, beta)
    gc = k.get

    if alpha > np.radians(model.stall_alpha):
        cl = 0.0
    else:
        cl = gc("CL0", 0.0) + gc("CL_el", 0.0) * el + gc("CL_q", 0.0) * q * half_c_v

    cd = gc("CD0", 0.0) + gc("CD_el", 0.0) * el

    cy = (gc("CY0", 0.0) + gc("CY_p", 0.0) * p * half_b_v + gc("CY_r", 0.0) * r * half_b_v
          + gc("CY_ail", 0.0) * ail + gc("CY_rud", 0.0) * rud)

    c_roll = (gc("Croll0", 0.0) + gc("Croll_p", 0.0) * p * half_b_v + gc("Croll_r", 0.0) * r * half_b_v
              + gc("Croll_ail", 0.0) * ail + gc("Croll_rud", 0.0) * rud)

    cm = gc("Cm0", 0.0) + gc("Cm_q", 0.0) * q * half_c_v + gc("Cm_el", 0.0) * el

    cn = (gc("Cn0", 0.0) + gc("Cn_p", 0.0) * p * half_b_v + gc("Cn_r", 0.0) * r * half_b_v
          + gc("Cn_ail", 0.0) * ail + gc("Cn_rud", 0.0) * rud)

    qs = dyn_pres * model.s
    return cd * qs, cl * qs, cy * qs, c_roll * qs * model.b, cm * qs * model.c, cn * qs * model.b
