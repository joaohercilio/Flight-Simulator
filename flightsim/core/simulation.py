# flightsim/core/simulation.py
"""Top-level simulation runner."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from flightsim.core.state_eq import make_state_eq
from flightsim.core.integrator import rk4, rk4_step
from flightsim.atmosphere.model import AtmosphereModel
from flightsim.aero.database import AeroDatabase
from flightsim.core.state import StateIndex

from utils.io import AircraftModel

import optvl
import os


def levelflight(mass, rho, V, ixx, iyy, izz, izx):
    os.chdir(r"C:\Users\enzo_\Desktop\AVL\Design Files-20260310T235542Z-3-001\Design Files\466 - 2C\Avl_00001")
    ovl = optvl.OVLSolver(geo_file="Setup_cruzeiro _subido_ev 015_cortado.avl")
    #DADOS DE REFERÊNCIA
    ref_data = ovl.get_reference_data()

    Sref = ref_data['Sref']
    Bref = ref_data['Bref']
    Cref = ref_data['Cref']
    XYZref = ref_data['XYZref']
    xcg = XYZref[0]
    zcg = XYZref[2]
    CD0  = 0.02567
    g=9.81
    W = mass*g
    q=0.5*rho*V**2
    CL_required = W / (q * Sref)
    ovl.set_constraint("alpha", "CL", CL_required)
    ovl.set_constraint("D2", "Cm", 0.0)
    ovl.set_parameter("velocity",   V)
    ovl.set_parameter("mass",       mass)
    ovl.set_parameter("Ixx",        ixx)
    ovl.set_parameter("Iyy",        iyy)
    ovl.set_parameter("Izz",        izz)
    ovl.set_parameter("Izx",        izx)
    ovl.set_parameter("grav.acc.",  g)
    ovl.set_parameter("density",    rho)
    ovl.set_parameter("CD0",        CD0)
    ovl.set_parameter("X cg",       xcg)
    ovl.set_parameter("Z cg",       zcg)

    ovl.execute_run()

    alpha_trim=ovl.get_variable("alpha")
    elevator_trim = ovl.get_control_deflection("profundor")
    return alpha_trim, elevator_trim


def run_simulation(
    model: AircraftModel,
    x0: NDArray,
    t_start: float,
    t_end: float,
    dt: float,
    atmosphere: AtmosphereModel,
) -> tuple[NDArray, NDArray, NDArray]:
    """Runs the 6DOF simulation using RK4 integration.

    Args:
        model: Aircraft model dataclass.
        x0: Initial state vector, shape (12,).
        t_start: Start time (s).
        t_end: End time (s).
        dt: Time step (s).
        atmosphere: AtmosphereModel instance.

    Returns:
        Tuple (t, x, dx) where:
            t:  Time vector, shape (N,).
            x:  State history, shape (12, N).
            dx: State derivative history, shape (12, N).
    """
    t = np.arange(t_start, t_end + dt, dt)
    n = len(t)

    x  = np.zeros((12, n))
    dx = np.zeros((12, n))

    V = 14

    alpha_trim, elevator_trim = levelflight(
        model.mass,
        1.1,
        V,
        model.ix,
        model.iy,
        model.iz,
        model.ixz)


    alpha_trim = 0.8096761099146154
    elevator_trim = 1.7341185187967543
    print (alpha_trim)
    print (elevator_trim)

    x0[StateIndex.U] = V *np.cos(np.deg2rad(alpha_trim))
    x0[StateIndex.W] = V *np.sin(np.deg2rad(alpha_trim))
    x0[StateIndex.THETA] = np.deg2rad(alpha_trim)

    x[:, 0] = x0

    aero_db = AeroDatabase(model.aero_tables_dir)

    ail_start   = 1.0
    ail_end     = 3.0
    ail_deflect = 0.0

    ele_start   = 5.0
    ele_end     = 6.0
    ele_deflect = 25

    current_t = t_start

    def timed_control():

        ele = ele_deflect if ele_start <= current_t <= ele_end else elevator_trim
        ail = ail_deflect if ail_start <= current_t <= ail_end else 0.0

        return ele, ail, 0.0, 0.0, 0.0

    f = make_state_eq(model, aero_db, timed_control, atmosphere)

    xi  = x0.copy()
    dxi = np.zeros(12)

    for i in range(n - 1):
        current_t = t[i]
        rk4_step(f, xi, dxi, dt)
        x[:, i + 1] = xi
        dx[:, i]    = dxi

    dx[:, -1] = dxi

    return t, x, dx
