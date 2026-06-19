from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from flightsim.core.state_eq import make_state_eq
from flightsim.core.integrator import rk4, rk4_step
from flightsim.atmosphere.model import AtmosphereModel
from flightsim.aero.database import AeroDatabase
from flightsim.core.state import StateIndex

from utils.io import AircraftModel

from scipy.optimize import fsolve, minimize

import numpy as np
import pandas as pd
from flightsim.core.state import StateIndex



def trim_opt(Vdes, hdes, gammades, radiusdes, atmosphere: AtmosphereModel, model: AircraftModel, condition="steady_level_flight"):
    
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

            f_eq, _= make_state_eq(model, AeroDatabase(model.aero_tables_dir), controls, atmosphere)
            dx_dt = f_eq(x, 0.0)

            u_dot = dx_dt[StateIndex.U]
            v_dot = dx_dt[StateIndex.V]
            w_dot = dx_dt[StateIndex.W]
            p_dot = dx_dt[StateIndex.P]
            q_dot = dx_dt[StateIndex.Q]
            r_dot = dx_dt[StateIndex.R]

            J= u_dot**2 + v_dot**2 + w_dot**2 + p_dot**2 + q_dot**2 + r_dot**2
            return J
            
        guess = np.array([0.1, 0.0, 0.5]) #alpha, delta_e, throttle
        bounds = [(np.deg2rad(-10.0), np.deg2rad(20.0)), 
                (-25, 25), 
                (0.0, 1.0)]
        res = minimize(slfres, guess, method='SLSQP', bounds=bounds, tol=1e-12)\
    
        if not res.success:
            raise RuntimeError(f"Trim optimization failed to converge: {res.message}")
        print(f"Final Cost (J): {res.fun}")
        
        alpha_trim, delta_e_trim, throttle_trim = res.x

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
        if radiusdes is None:
                raise ValueError("Radius must be specified for coordinated turn trim condition.")
        psi_dot = Vdes/radiusdes
        def cturn(guess):
            alpha, phi, delta_e, delta_a, delta_r, throttle = guess
            theta = np.arctan((np.tan(alpha))*(np.cos(phi)))  #ze = -usin(theta) + w*cos(theta)*cos(phi) -> ze = 0
            psi = 0.0
            p = -psi_dot*np.sin(theta)
            q = psi_dot*np.sin(phi)*np.cos(theta)
            r = psi_dot*np.cos(phi)*np.cos(theta)

            u = Vdes*np.cos(alpha)
            w = Vdes*np.sin(alpha)
            v = 0.0

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

            controls = lambda: (delta_e, delta_a, delta_r, throttle, 0.0)
            f_eq, _= make_state_eq(model, AeroDatabase(model.aero_tables_dir), controls, atmosphere)
            dx_dt = f_eq(x, 0.0)

            u_dot = dx_dt[StateIndex.U]
            v_dot = dx_dt[StateIndex.V]
            w_dot = dx_dt[StateIndex.W]
            p_dot = dx_dt[StateIndex.P]
            q_dot = dx_dt[StateIndex.Q]
            r_dot = dx_dt[StateIndex.R]

            J = u_dot**2 + v_dot**2 + w_dot**2 + p_dot**2 + q_dot**2 + r_dot**2
            return J
        
        phi_guess = np.arctan((Vdes**2) / (9.81 * radiusdes))
        guess = np.array([np.deg2rad(2.0), phi_guess, 0.0, 0.0, 0.0, 0.5])
        
        bounds = [
            (np.deg2rad(-5.0), np.deg2rad(15.0)),   # alpha
            (np.deg2rad(-60.0), np.deg2rad(60.0)),  # phi (max 60 deg bank)
            (-25.0, 25.0),                          # elevator
            (-20.0, 20.0),                          # aileron
            (-30.0, 30.0),                          # rudder
            (0.0, 1.0)                              # throttle
        ]

        res = minimize(cturn, guess, method='SLSQP', bounds=bounds, tol=1e-8)
        print(f"Final Cost (J): {res.fun}")
        
        if not res.success:
            raise RuntimeError(f"Coordinated turn trim failed: {res.message}")
        
        alpha_tr, phi_tr, de_tr, da_tr, dr_tr, thr_tr = res.x
        theta_tr = np.arctan(np.tan(alpha_tr) * np.cos(phi_tr))

        initcond = np.zeros(12)
        initcond[StateIndex.U] = Vdes * np.cos(alpha_tr)
        initcond[StateIndex.W] = Vdes * np.sin(alpha_tr)
        initcond[StateIndex.PHI] = phi_tr
        initcond[StateIndex.THETA] = theta_tr

        initcond[StateIndex.P] = -psi_dot * np.sin(theta_tr)
        initcond[StateIndex.Q] =  psi_dot * np.sin(phi_tr) * np.cos(theta_tr)
        initcond[StateIndex.R] =  psi_dot * np.cos(phi_tr) * np.cos(theta_tr)

        initcond[StateIndex.Z_E] = -hdes

        trim_controls = {
            "elevator": de_tr,
            "aileron": da_tr,
            "rudder": dr_tr,
            "throttle": thr_tr,
            "brake": 0.0
        }

        return initcond, trim_controls

    elif condition == "steady_climb":
        def sclimb(guess):
            alpha, delta_e, throttle = guess
            theta = alpha + np.deg2rad(gammades)
            u = Vdes*np.cos(alpha)
            w = Vdes*np.sin(alpha)
            v = 0.0
            p,q,r = 0.0, 0.0, 0.0
            x = np.zeros(12)
            x[StateIndex.U] = u
            x[StateIndex.V] = v
            x[StateIndex.W] = w
            x[StateIndex.PHI] = 0.0
            x[StateIndex.THETA] = theta
            x[StateIndex.PSI] = 0.0
            x[StateIndex.P] = p
            x[StateIndex.Q] = q
            x[StateIndex.R] = r
            x[StateIndex.Z_E] = -hdes
            controls = lambda: (delta_e, 0.0, 0.0, throttle, 0.0) #ele, ail, rud, thr, brk
            f_eq, _= make_state_eq(model, AeroDatabase(model.aero_tables_dir), controls, atmosphere)
            dx_dt = f_eq(x, 0.0)
            u_dot = dx_dt[StateIndex.U]
            v_dot = dx_dt[StateIndex.V]
            w_dot = dx_dt[StateIndex.W]
            p_dot = dx_dt[StateIndex.P]
            q_dot = dx_dt[StateIndex.Q]
            r_dot = dx_dt[StateIndex.R]
            J= u_dot**2 + v_dot**2 + w_dot**2 + p_dot**2 + q_dot**2 + r_dot**2
            return J
        guess = np.array([0.1, 0.0, 0.5]) #alpha, delta_e, throttle
        bounds = [(np.deg2rad(-10.0), np.deg2rad(20.0)), 
                (-25, 25), 
                (0.0, 1.0)]
        res = minimize(sclimb, guess, method='SLSQP', bounds=bounds, tol=1e-12)
        print(f"Final Cost (J): {res.fun}")
        if not res.success:
            raise RuntimeError(f"Coordinated turn trim failed: {res.message}")
        alpha_trim, delta_e_trim, throttle_trim = res.x
        initcond = np.zeros(12)
        initcond[StateIndex.U] = Vdes * np.cos(alpha_trim)
        initcond[StateIndex.W] = Vdes * np.sin(alpha_trim)
        initcond[StateIndex.THETA] = alpha_trim + np.deg2rad(gammades)
        initcond[StateIndex.Z_E] = -hdes
        trim_controls = {
            "elevator": delta_e_trim,
            "aileron": 0.0,
            "rudder": 0.0,
            "throttle": throttle_trim,
            "brake": 0.0
        }
        return initcond, trim_controls
    
    elif condition == "glide":
        def glide(guess):
            alpha, delta_e, theta = guess
            phi, psi = 0.0, 0.0
            throttle = 0.0
            u = Vdes*np.cos(alpha)
            w = Vdes*np.sin(alpha)
            v = 0.0
            p,q,r = 0.0, 0.0, 0.0
            x = np.zeros(12)
            x[StateIndex.U] = u
            
            x[StateIndex.W] = w
            
            x[StateIndex.THETA] = theta
            x[StateIndex.Z_E] = -hdes
            controls = lambda: (delta_e, 0.0, 0.0, throttle, 0.0) #ele, ail, rud, thr, brk
            f_eq, _= make_state_eq(model, AeroDatabase(model.aero_tables_dir), controls, atmosphere)
            dx_dt = f_eq(x, 0.0)
            u_dot = dx_dt[StateIndex.U]
            v_dot = dx_dt[StateIndex.V]
            w_dot = dx_dt[StateIndex.W]
            p_dot = dx_dt[StateIndex.P]
            q_dot = dx_dt[StateIndex.Q]
            r_dot = dx_dt[StateIndex.R]
            J = u_dot**2 + v_dot**2 + w_dot**2 + p_dot**2 + q_dot**2 + r_dot**2
            return J
        guess = np.array([0.1, 0.0, 0.0])
        bounds = [(np.deg2rad(-10.0), np.deg2rad(15.0)), (-25.0, 25.0), (np.deg2rad(-30.0), np.deg2rad(30.0))]
        res = minimize(glide, guess, method='SLSQP', bounds=bounds, tol=1e-11)
        print(f"Final Cost (J): {res.fun}")
        if not res.success:
            raise RuntimeError(f"Coordinated turn trim failed: {res.message}")
        alpha_trim, delta_e_trim, theta_trim = res.x
        initcond = np.zeros(12)
        initcond[StateIndex.U] = Vdes * np.cos(alpha_trim)
        initcond[StateIndex.W] = Vdes * np.sin(alpha_trim)
        initcond[StateIndex.THETA] = theta_trim
        initcond[StateIndex.Z_E] = -hdes
        trim_controls = {
            "elevator": delta_e_trim,
            "aileron": 0.0,
            "rudder": 0.0,
            "throttle": 0.0,
            "brake": 0.0
        }
        return initcond, trim_controls
    elif condition == "turn":
        def turn(guess):
            pass
