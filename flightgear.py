from __future__ import annotations

import time
import pathlib
import numpy as np
import pygame
from flightgear_python.fg_if import FDMConnection

from config.settings import SimConfig
from utils.io import load_model
from flightsim.core.state_eq import make_state_eq
from flightsim.core.integrator import rk4_step
from flightsim.aero.database import AeroDatabase
from flightsim.core.simulation import compute_trim

# ---------------------------------------------------------------
# Constants (Keep these global so the worker process can see them)
# ---------------------------------------------------------------
CASE_DIR = pathlib.Path("cases/mushu")

FDM_HZ       = 240          
SEND_HZ      = 60           
DT           = 1.0 / FDM_HZ
STEPS_PER_SEND = FDM_HZ // SEND_HZ   

_R_EARTH   = 6_378_137.0   
_LAT0_DEG  = 27.039
_LON0_DEG  = 49.405
_LAT0      = np.radians(_LAT0_DEG)
_LON0      = np.radians(_LON0_DEG)

PRINT_EACH = 60   

# ---------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------
def _ned_to_geodetic(x_e: float, y_e: float) -> tuple[float, float]:
    lat = _LAT0 + x_e / _R_EARTH
    lon = _LON0 + y_e / (_R_EARTH * np.cos(_LAT0))
    return lat, lon

class ScriptedTransmitter:
    """Provides pre-programmed control inputs for automated test cases."""
    
    def __init__(self, trim_ele: float, trim_thr: float):
        self.sim_time = 0.0  # <--- Time is now externally tracked
        self._trim_ele = trim_ele
        self._trim_thr = trim_thr
        
    def read(self) -> tuple[float, float, float, float, float]:
        current_t = self.sim_time
        
        ele = self._trim_ele
        ail = 0.0
        rud = 0.0
        throttle = self._trim_thr 
        brake = 0.0

        # Exact maneuver logic from simulation.py
        ail_start   = 5.0
        ail_end     = 30.0
        ail_deflect = 20.0*0 #4.821695697645064*0

        ele_start   = 5.0
        ele_end     = 6.0
        ele_deflect = 0.0 

        rud_start = 5.0
        rud_end = 10.0
        rud_deflect = 3.0*0

        ele = self._trim_ele + ele_deflect if ele_start <= current_t <= ele_end else self._trim_ele
        ail = ail_deflect if ail_start <= current_t <= ail_end else 0.0
        rud = rud_deflect if rud_start <= current_t <= rud_end else 0.0

        return ele, ail, rud, throttle, brake

class RCTransmitter:
    """Reads control inputs. Pickle-safe for Windows multiprocessing."""
    _AXIS_AILERON  = 0
    _AXIS_ELEVATOR = 1
    _AXIS_THROTTLE = 2
    _AXIS_RUDDER   = 3
    _AXIS_BRAKE    = 4

    def __init__(self, joystick_index: int = 0) -> None:
        self.joystick_index = joystick_index
        self._joystick = None  # Don't init yet!

    def _ensure_init(self):
        """This runs ONLY when we actually start reading, inside the correct process."""
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
        return ele, ail, rud, throttle, 0.0

class FlightGearBridge:
    def __init__(self, case_dir: pathlib.Path, manual_control: bool = False) -> None:
        cfg   = SimConfig.from_toml_file(case_dir / "sim_config.toml")
        model = load_model(case_dir / "aircraft_model.toml")
        aero_db = AeroDatabase(model.aero_tables_dir)

        # 1. Compute Trim dynamically (just like in simulation.py)
        V = 16.74
        alpha_trim, trim_elevator, trim_throttle = compute_trim(
            model,
            aero_db=aero_db,
            g=9.81,
            V=V,
            rho=1.1
        )

        # 2. Update initial states to match trim
        # Indices: 4 is THETA, 6 is U, 8 is W
        cfg.x0[6] = V * np.cos(alpha_trim)
        cfg.x0[8] = V * np.sin(alpha_trim)
        cfg.x0[4] = alpha_trim

        if manual_control:
            self._transmitter = RCTransmitter()
        else:
            self._transmitter = ScriptedTransmitter(trim_elevator, trim_throttle)

        self._x  = cfg.x0.copy()
        self._dx = np.zeros_like(self._x)
        
        self._f  = make_state_eq(model, aero_db, self._transmitter.read, cfg.atmosphere)
        self._frame = 0

        if manual_control:
            self._transmitter = RCTransmitter()
        else:
            self._transmitter = ScriptedTransmitter(trim_elevator, trim_throttle)

        self._x  = cfg.x0.copy()
        self._dx = np.zeros_like(self._x)
        
        self._f  = make_state_eq(model, aero_db, self._transmitter.read, cfg.atmosphere)
        self._frame = 0

    def _callback(self, fdm_data, event_pipe):
        # This runs in the background process spawned by flightgear-python
        for _ in range(STEPS_PER_SEND):
            rk4_step(self._f, self._x, self._dx, DT)
            
            # Advance simulation time ONLY after the full step is complete
            if hasattr(self._transmitter, 'sim_time'):
                self._transmitter.sim_time += DT

        x_e, y_e, z_e     = self._x[0], self._x[1], self._x[2]
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

        # Grab the exact simulation time if it's the scripted transmitter
        sim_t_str = f" t={self._transmitter.sim_time:.1f}s " if hasattr(self._transmitter, 'sim_time') else " "

        print(
            f"[{sim_t_str}] ele: {ele:+.2f}  ail: {ail:+.2f}  rud: {rud:+.2f}  thr: {throttle:.2f}  "
            f"alt: {-self._x[2]:.0f} m  alpha: {alpha:.1f}°  beta: {beta:.1f}° v_air: {v_air:.1f} m/s" , end='\r'
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

if __name__ == "__main__":
    # On Windows, all code MUST be under this if __name__ block
    # to avoid recursive process spawning.
    
    # Set manual_control=False to watch the automated maneuver
    bridge = FlightGearBridge(CASE_DIR, manual_control=False)
    bridge.run()