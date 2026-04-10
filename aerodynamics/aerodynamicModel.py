from .aeroDatabase import AeroDatabase
from scipy.interpolate import RegularGridInterpolator
import numpy as np

aero_db = AeroDatabase()

def aerodynamicForceBody(D, L, C, salfa, calfa, sbeta, cbeta):
    """
    Computes aerodynamic forces and moments in body axes

    :param D: Drag (N)
    :param L: Lift (N)
    :param C: Side force (N)
    :param salfa: sin of alfa
    :param calfa: cos of alfa
    :param sbeta: sin of beta
    :param cbeta: cos of Beta
    :return: X, Y, Z (forces in N)
    """

    X = -( D*calfa*cbeta - L*salfa + C*calfa*sbeta )
    Y = -( D*sbeta) + C*cbeta
    Z = -( D*salfa*cbeta - C*salfa*sbeta + L*calfa )
        
    return X, Y, Z

def aerodynamicForceWind(model, alpha, beta, p, q, r, el, ail, rud, V, dynPres):
    """
    Computes aerodynamic forces and moments in wind axes.

    All coefficients are looked up from the aero database (interpolated at
    current alpha, beta). If a table is not present, falls back to 0.

    :param model:       Aircraft model dict (geometry only)
    :param alpha:       Angle of attack (rad)
    :param beta:        Sideslip angle (rad)
    :param p, q, r:     Body angular rates (rad/s)
    :param el:          Elevator deflection (rad)
    :param ail:         Aileron deflection (rad)
    :param rud:         Rudder deflection (rad)
    :param V:           Airspeed (m/s)
    :param dynPres:     Dynamic pressure (Pa)
    :return: D, L, C, l, m, n  (forces in N, moments in N·m)
    """

    b = model["b"]
    c = model["c"]
    S = model["S"]

    half_b_V = b / (2 * V)
    half_c_V = c / (2 * V)

    getCoeff = aero_db.getCoeff

    # --- LIFT ---
    CL = (getCoeff("CL0",    alpha, beta)
        + getCoeff("CL_el",  alpha, beta) * el)

    # --- DRAG ---
    CD = (getCoeff("CD0",    alpha, beta)
        + getCoeff("CD_el",  alpha, beta) * el)

    # --- SIDE FORCE ---
    CY = (getCoeff("CY0",    alpha, beta)
        + getCoeff("CY_p",   alpha, beta) * p  * half_b_V
        + getCoeff("CY_r",   alpha, beta) * r  * half_b_V
        + getCoeff("CY_ail", alpha, beta) * ail
        + getCoeff("CY_rud", alpha, beta) * rud)

    # --- ROLLING MOMENT ---
    Cl = (getCoeff("Croll0",    alpha, beta)
    + getCoeff("Croll_p",   alpha, beta) * p  * half_b_V
    + getCoeff("Croll_r",   alpha, beta) * r  * half_b_V
    + getCoeff("Croll_ail", alpha, beta) * ail
    + getCoeff("Croll_rud", alpha, beta) * rud)

    # --- PITCHING MOMENT ---
    Cm = (getCoeff("Cm0",    alpha, beta)
        + getCoeff("Cm_q",   alpha, beta) * q  * half_c_V
        + getCoeff("Cm_el",  alpha, beta) * el)

    # --- YAWING MOMENT ---
    Cn = (getCoeff("Cn0",    alpha, beta)
        + getCoeff("Cn_p",   alpha, beta) * p  * half_b_V
        + getCoeff("Cn_r",   alpha, beta) * r  * half_b_V
        + getCoeff("Cn_ail", alpha, beta) * ail
        + getCoeff("Cn_rud", alpha, beta) * rud)

    # --- FORCES AND MOMENTS ---
    D = CD * dynPres * S
    L = CL * dynPres * S
    C = CY * dynPres * S

    l = dynPres * S * b * Cl
    m = dynPres * S * c * Cm
    n = dynPres * S * b * Cn

    return D, L, C, l, m, n