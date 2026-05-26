from __future__ import annotations

import time
import pathlib
import numpy as np
import pygame
from flightgear_python.fg_if import FDMConnection
from scipy.optimize import fsolve

from config.settings import SimConfig
from utils.io import load_model
from flightsim.core.state_eq import make_state_eq
from flightsim.core.integrator import rk4_step
from flightsim.aero.database import AeroDatabase
from flightsim.core.simulation import compute_trim
from flightsim.core.state import StateIndex
from utils.io import AircraftModel

# ---------------------------------------------------------------
# Constants (Keep these global so the worker process can see them)
# ---------------------------------------------------------------
CASE_DIR = pathlib.Path("cases/mushu")

FDM_HZ         = 240          
SEND_HZ        = 60           
DT             = 1.0 / FDM_HZ
STEPS_PER_SEND = FDM_HZ // SEND_HZ   

_R_EARTH   = 6_378_137.0   
_LAT0_DEG  = -23.2292
_LON0_DEG  = -45.8615
_LAT0      = np.radians(_LAT0_DEG)
_LON0      = np.radians(_LON0_DEG)
groundSJC  = -635.3
PRINT_EACH = 60   

# ---------------------------------------------------------------
# Trim and Analysis Functions
# ---------------------------------------------------------------
def compute_climbing_trim(model, aero_db, atmosphere, g: float, V: float, altitude: float, throttle: float):
    """Finds steady-state flight conditions (alpha, elevator, gamma) for a given state."""
    rho = atmosphere.get_density(altitude)
    dyn_pres = 0.5 * rho * V**2
    Sref = model.s
    
    # Thrust polynomial adjusted by density altitude ratio
    a, b, c, d = 0.001274, -0.07204, -0.5428, 40.89
    thrust = throttle * (a * V**3 + b * V**2 + c * V + d) * rho / 1.225

    def residuals(x):
        alpha_rad = x[0]
        el_deg    = x[1]
        gamma_rad = x[2]
        
        theta_rad = alpha_rad + gamma_rad

        cl = aero_db.get_coeff("CL0", alpha_rad, 0.0) + aero_db.get_coeff("CL_el", alpha_rad, 0.0) * el_deg
        cd = aero_db.get_coeff("CD0", alpha_rad, 0.0) + aero_db.get_coeff("CD_el", alpha_rad, 0.0) * el_deg
        cm = aero_db.get_coeff("Cm0", alpha_rad, 0.0) + aero_db.get_coeff("Cm_el", alpha_rad, 0.0) * el_deg

        lift = cl * dyn_pres * Sref
        drag = cd * dyn_pres * Sref
        pitch_moment = cm * dyn_pres * Sref * model.c

        sin_a, cos_a = np.sin(alpha_rad), np.cos(alpha_rad)
        fx_aero = -(drag * cos_a - lift * sin_a)
        fz_aero = -(drag * sin_a + lift * cos_a)

        fx_total = fx_aero + thrust
        fz_total = fz_aero
        pitch_total = pitch_moment + model.arm_z_engine * thrust

        sin_tht, cos_tht = np.sin(theta_rad), np.cos(theta_rad)
        
        res_u = fx_total / model.mass - g * sin_tht
        res_w = fz_total / model.mass + g * cos_tht
        res_q = pitch_total / model.iy

        return [res_u, res_w, res_q]

    x0 = [np.deg2rad(2.0), 0.0, 0.0]
    x_trim, info, ier, msg = fsolve(residuals, x0, full_output=True, xtol=1e-10)

    if ier != 1:
        # If it completely fails to resolve forces, return a massive penalty climb rate
        return np.deg2rad(2.0), 0.0, np.deg2rad(-90.0), -999.0

    alpha_trim = x_trim[0]
    el_trim = x_trim[1]
    gamma_trim = x_trim[2]
    climb_rate = V * np.sin(gamma_trim)

    return alpha_trim, el_trim, gamma_trim, climb_rate


def estimate_cruise_altitude(model, aero_db, atmosphere, g, V_cruise, target_throttle=0.7):
    """Sweeps altitudes to find where steady level flight (climb rate = 0) occurs at 70% throttle."""
    alts = np.linspace(0, 6000, 200) # Slightly wider grid search 
    
    print(f"Sweeping altitudes at V = {V_cruise} m/s, Throttle Setting = {target_throttle}")
    print("-" * 60)
    
    closest_cruise_alt = 0.0
    
    for h in alts:
        alpha, el, gamma, climb_rate = compute_climbing_trim(
            model, aero_db, atmosphere, g, V_cruise, h, target_throttle
        )
        
        # Guardrails for geometric/aerodynamic stalls
        if abs(el) > 25.0 or np.rad2deg(alpha) > 15.0 or climb_rate == -999.0:
            print(f"  [{h:6.1f} m] Limits Exceeded (Aerodynamic / Trim failure)")
            break
            
        print(f"  [{h:6.1f} m] Calculated excess climb capacity: {climb_rate:+.3f} m/s")
        
        # When climb rate drops to or below 0, we have found our absolute cruise ceiling for this throttle
        if climb_rate <= 0.0:
            closest_cruise_alt = h
            print(f"\n>>> Level Flight Equilibrium Crossed near this altitude threshold!")
            break
            
        closest_cruise_alt = h
            
    return closest_cruise_alt


# ---------------------------------------------------------------
# Transmitter Core Classes & Kinematics Helper
# ---------------------------------------------------------------
def _ned_to_geodetic(x_e: float, y_e: float) -> tuple[float, float]:
    lat = _LAT0 + x_e / _R_EARTH
    lon = _LON0 + y_e / (_R_EARTH * np.cos(_LAT0))
    return lat, lon

class ScriptedTransmitter:
    def __init__(self, trim_ele: float, trim_thr: float):
        self.sim_time = 0.0  
        self._trim_ele = trim_ele
        self._trim_thr = trim_thr
        
    def read(self) -> tuple[float, float, float, float, float]:
        current_t = self.sim_time
        ele_init = self._trim_ele
        ail, rud, brake = 0.0, 0.0, 0.0
        throttle = self._trim_thr

        ele_start, ele_end, ele_deflect = 4.0, 30.0, 0.0
        ail_start, ail_end, ail_deflect = 5.0, 16.0, 0.0
        rud_start, rud_end, rud_deflect = 5.0, 16.0, 0.0

        ele = ele_init + ele_deflect if ele_start <= current_t <= ele_end else ele_init
        ail = ail_deflect if ail_start <= current_t <= ail_end else 0.0
        rud = rud_deflect if rud_start <= current_t <= rud_end else 0.0

        return ele, ail, rud, throttle, brake

class RCTransmitter:
    _AXIS_AILERON  = 0
    _AXIS_ELEVATOR = 1
    _AXIS_THROTTLE = 2
    _AXIS_RUDDER   = 3
    _AXIS_BRAKE    = 4

    def __init__(self, joystick_index: int = 0) -> None:
        self.joystick_index = joystick_index
        self._joystick = None 

    def _ensure_init(self):
        if self._joystick is None:
            pygame.init()
            pygame.joystick.init()
            if pygame.joystick.get_count() == 0:
                raise RuntimeError("No joystick detected. Plug it in!")
            self._joystick = pygame.joystick.Joystick(self.joystick_index)
            self._joystick.init()
            print(f"Joystick initialized: {self._joystick.get_name()}")

    def read(self) -> tuple[float, float, float, float, float]:
        self._ensure_init()
        pygame.event.pump()
        ail      =  self._joystick.get_axis(self._AXIS_AILERON)
        ele      = -self._joystick.get_axis(self._AXIS_ELEVATOR)
        rud      =  self._joystick.get_axis(self._AXIS_RUDDER)
        throttle = (self._joystick.get_axis(self._AXIS_THROTTLE) + 1.0) / 2.0
        brake    = (self._joystick.get_axis(self._AXIS_BRAKE)    + 1.0) / 2.0
        return 25*ele, -20*ail, 30*rud, throttle, 0.0

class FlightGearBridge:
    def __init__(self, case_dir: pathlib.Path, manual_control: bool = False, start_in_air: bool = False) -> None:
        cfg   = SimConfig.from_toml_file(case_dir / "sim_config.toml")
        model = load_model(case_dir / "aircraft_model.toml")
        aero_db = AeroDatabase(model.aero_tables_dir)

        if start_in_air:
            V = 14.9 
            alpha_trim, trim_elevator, trim_throttle = compute_trim(
                model, aero_db=aero_db, g=9.81, V=V, rho=1.1
            )
            trim_throttle = 1.0
            cfg.x0[StateIndex.U] = V * np.cos(alpha_trim)
            cfg.x0[StateIndex.W] = V * np.sin(alpha_trim)
            cfg.x0[StateIndex.THETA] = alpha_trim
            
            if cfg.x0[StateIndex.Z_E] - groundSJC == 0:
                cfg.x0[StateIndex.Z_E] = -50.0 + groundSJC  
        else:
            trim_elevator = 0.0
            trim_throttle = 0.0

        if manual_control:
            self._transmitter = RCTransmitter()
        else:
            self._transmitter = ScriptedTransmitter(trim_elevator, trim_throttle)

        self._x  = cfg.x0.copy()
        self._dx = np.zeros_like(self._x)
        self._f  = make_state_eq(model, aero_db, self._transmitter.read, cfg.atmosphere)
        self._frame = 0

    def _callback(self, fdm_data, event_pipe):
        for _ in range(STEPS_PER_SEND):
            rk4_step(self._f, self._x, self._dx, DT)
            if hasattr(self._transmitter, 'sim_time'):
                self._transmitter.sim_time += DT

        x_e, y_e, z_e     = self._x[0], self._x[1], self._x[2]
        phi, theta, psi   = self._x[3], self._x[4], self._x[5]
        lat, lon = _ned_to_geodetic(x_e, y_e)

        fdm_data.lon_rad   = lon
        fdm_data.lat_rad   = lat
        fdm_data.alt_m     = -z_e
        fdm_data.phi_rad   = phi
        fdm_data.theta_rad = theta
        fdm_data.psi_rad   = psi

        if self._frame % PRINT_EACH == 0:
            self._print_status()

        self._frame += 1
        return fdm_data

    def _print_status(self) -> None:
        ele, ail, rud, throttle, brake = self._transmitter.read()
        u, v, w  = self._x[6], self._x[7], self._x[8]
        v_air    = max(np.sqrt(u**2 + v**2 + w**2), 1e-8)
        alpha    = np.degrees(np.arctan2(w, u))
        beta     = np.degrees(np.arcsin(np.clip(v / v_air, -1.0, 1.0)))
        x_e, y_e = self._x[0], self._x[1]
        ground_track = np.sqrt(x_e**2 + y_e**2)

        sim_t_str = f" t={self._transmitter.sim_time:.1f}s " if hasattr(self._transmitter, 'sim_time') else " "
        print(
            f"[{sim_t_str}] ele: {ele:+.2f}  ail: {ail:+.2f}  rud: {rud:+.2f}  thr: {throttle:.2f}  "
            f"alt: {-(self._x[2]-(groundSJC)):.2f} m  alpha: {alpha:.1f}°  beta: {beta:.1f}°  v_air: {v_air:.1f} m/s  "
            f"track: {ground_track:.2f} m", end='\r'
        )

    def run(self) -> None:
        fdm_conn = FDMConnection()
        fdm_conn.connect_rx("localhost", 5501, self._callback)
        fdm_conn.connect_tx("localhost", 5502)
        fdm_conn.start()

        print("FlightGear bridge running. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopped.")


# ---------------------------------------------------------------
# Single Unified Execution Entrypoint
# ---------------------------------------------------------------
if __name__ == "__main__":
    print("Initializing aircraft configurations for pre-flight analysis...")
    from flightsim.atmosphere.model import AtmosphereModel  
    
    # Load environment datasets
    cfg = SimConfig.from_toml_file(CASE_DIR / "sim_config.toml")
    ac_model = load_model(CASE_DIR / "aircraft_model.toml")
    aero_database = AeroDatabase(ac_model.aero_tables_dir)
    atmos = cfg.atmosphere  
    
    # Targets configuration matching user parameters
    V_cruise = 14.9  # m/s
    g_gravity = 9.81
    target_throttle = 0.70
    
    print("\n==================================================")
    print(" RUNNING CRUISE ALTITUDE CALCULATION")
    print("==================================================")
    
    cruise_altitude = estimate_cruise_altitude(
        model=ac_model, 
        aero_db=aero_database, 
        atmosphere=atmos, 
        g=g_gravity, 
        V_cruise=V_cruise,
        target_throttle=target_throttle
    )
    
    print("==================================================")
    print(f" ANALYSIS COMPLETE")
    print(f" Target Speed: {V_cruise} m/s at {target_throttle*100:.0f}% Throttle")
    print(f" Steady Cruise Altitude (Climb Rate = 0): {cruise_altitude:.2f} meters")
    print("==================================================\n")
    
    # Hand off to standard active hardware loops
    print("Launching FlightGear bridge. Press Ctrl+C to stop.")
    bridge = FlightGearBridge(CASE_DIR, manual_control=False, start_in_air=True)
    bridge.run()