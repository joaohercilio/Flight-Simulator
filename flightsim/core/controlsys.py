from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from flightsim.core.state_eq import make_state_eq
from flightsim.core.integrator import rk4, rk4_step
from flightsim.atmosphere.model import AtmosphereModel
from flightsim.aero.database import AeroDatabase
from flightsim.core.state import StateIndex

from utils.io import AircraftModel

from scipy.optimize import fsolve

import numpy as np
import pandas as pd
from flightsim.core.state import StateIndex



def trim_opt(Vdes, hdes, gammades, atmosphere: AtmosphereModel, model: AircraftModel,condition="steady_level_flight"):
    
    if condition == "steady_level_flight":
        def slfres(guess):
            alpha, delta_e, throttle = guess
            
            u = Vdes*np.cos(alpha)
            w = Vdes*np.sin(alpha)
            v = 0.0
            p,q,r = 0.0, 0.0, 0.0
            phi,theta,psi = 0.0, alpha+gammades, 0.0
            x = np.zeros(12)
            x[StateIndex.U] = u
            x[StateIndex.V] = v
            x[StateIndex.W] = w
            x[StateIndex.PHI] = phi
            x[StateIndex.THETA] = theta
            x[StateIndex.PSI] = psi
            x[StateIndex.P] = p
            x[StateIndex.Q] = q
            x[StateIndex.R] = r
            x[StateIndex.Z_E] = -hdes
            #a, b, c, d = 0.001274, -0.07204, -0.5428, 40.89
            #throttle = thrust/((a * Vdes**3 + b * Vdes**2 + c * Vdes + d)*(rho / 1.225))

            controls = lambda: (delta_e, 0.0, 0.0, throttle, 0.0) #ele, ail, rud, thr, brk

            f_eq = make_state_eq(model, AeroDatabase(model.aero_tables_dir), controls, atmosphere)
            dx_dt = f_eq(x, 0.0)

            u_dot = dx_dt[StateIndex.U]
            
            w_dot = dx_dt[StateIndex.W]
            
            q_dot = dx_dt[StateIndex.Q]
            

            return [u_dot, w_dot, q_dot]
            
        guess = np.array([0.1, 0.0, 0.5]) #alpha, delta_e, throttle
        sol, infodict, ier, mesg = fsolve(slfres, guess, full_output=True)
        if ier != 1:
            raise RuntimeError(f"Trim optimization failed to converge: {mesg}")
        
        alpha_trim, delta_e_trim, throttle_trim = sol

        initcond = np.zeros(12)
        initcond[StateIndex.U] = Vdes * np.cos(alpha_trim)
        
        initcond[StateIndex.W] = Vdes * np.sin(alpha_trim)
        
        initcond[StateIndex.THETA] = alpha_trim + gammades
        
        initcond[StateIndex.Z_E] = -hdes

        trim_controls = {
            "elevator": delta_e_trim,
            "aileron": 0.0,
            "rudder": 0.0,
            "throttle": throttle_trim,
            "brake": 0.0
        }

        return initcond, trim_controls
        
        
    elif condition == "coordinated_turn":
        pass
    elif condition == "steady_climb":
        pass
    elif condition == "steady_descent":
        pass
    elif condition == "glide":
        pass
    elif condition == "turn":
        pass

        