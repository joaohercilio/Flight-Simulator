from __future__ import annotations

import time
import pathlib
import tomllib
import dataclasses
import numpy as np
import pygame
from flightgear_python.fg_if import FDMConnection
from scipy.optimize import fsolve

from config.settings import SimConfig
from utils.io import load_model
from flightsim.core.state_eq import make_state_eq
from flightsim.core.integrator import rk4_step
from flightsim.aero.database import AeroDatabase
from flightsim.core.controlsys import trim_opt

# ---------------------------------------------------------------
# Constants
# ---------------------------------------------------------------
CASE_DIR = pathlib.Path("cases/mushu")

FDM_HZ         = 240
SEND_HZ        = 60
DT             = 1.0 / FDM_HZ
STEPS_PER_SEND = FDM_HZ // SEND_HZ

_R_EARTH  = 6_378_137.0
_LAT0_DEG = -23.2292
_LON0_DEG = -45.8615
_LAT0     = np.radians(_LAT0_DEG)
_LON0     = np.radians(_LON0_DEG)

groundSJC  = load_model(CASE_DIR / "aircraft_model.toml").ground_altitude
PRINT_EACH = 60

# ---------------------------------------------------------------
# Configuration Loader
# ---------------------------------------------------------------
@dataclasses.dataclass
class FlightGearConfig:
    """Holds FlightGear bridge toggles loaded from TOML."""
    find_cruise_ceiling: bool
    start_in_air: bool
    start_trimmed: bool
    manual_control: bool
    throttleceiling: float

    @classmethod
    def from_toml_file(cls, path: pathlib.Path) -> FlightGearConfig:
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "rb") as f:
            data = tomllib.load(f)
        
        fg = data.get("flightgear", {})
        return cls(
            find_cruise_ceiling=fg.get("find_cruise_ceiling", False),
            start_in_air=fg.get("start_in_air", True),
            start_trimmed=fg.get("start_trimmed", True),
            manual_control=fg.get("manual_control", False),
            throttleceiling=fg.get("throttleceiling", 1.0),
        )

# ---------------------------------------------------------------
# Kinematics helper
# ---------------------------------------------------------------
def _ned_to_geodetic(x_e: float, y_e: float) -> tuple[float, float]:
    lat = _LAT0 + x_e / _R_EARTH
    lon = _LON0 + y_e / (_R_EARTH * np.cos(_LAT0))
    return lat, lon


# ---------------------------------------------------------------
# Performance analysis — ceiling sweep
# (independent of trim_opt; used for offline analysis only)
# ---------------------------------------------------------------
def _climbing_trim(model, aero_db, atmosphere, g: float, V: float, altitude: float, throttle: float):
    """
    Solves for (alpha, elevator, gamma) at a fixed throttle and airspeed.
    Used exclusively by estimate_cruise_ceiling — not for flight initialisation.
    Returns (alpha_rad, el_deg, gamma_rad, climb_rate_ms).
    climb_rate == -999 signals solver failure.
    """
    true_asl = altitude - groundSJC
    rho      = atmosphere.get_density(true_asl)
    q        = 0.5 * rho * V**2
    Sref     = model.s
    a, b, c, d = 0.001274, -0.07204, -0.5428, 40.89
    thrust = throttle * (a * V**3 + b * V**2 + c * V + d) * rho / 1.225

    def residuals(x):
        alpha, el_deg, gamma = x
        theta = alpha + gamma
        cl = aero_db.get_coeff("CL0", alpha, 0.0) + aero_db.get_coeff("CL_el", alpha, 0.0) * el_deg
        cd = aero_db.get_coeff("CD0", alpha, 0.0) + aero_db.get_coeff("CD_el", alpha, 0.0) * el_deg
        cm = aero_db.get_coeff("Cm0", alpha, 0.0) + aero_db.get_coeff("Cm_el", alpha, 0.0) * el_deg
        lift  = cl * q * Sref
        drag  = cd * q * Sref
        mom   = cm * q * Sref * model.c
        sa, ca = np.sin(alpha), np.cos(alpha)
        fx = -(drag * ca - lift * sa) + thrust
        fz = -(drag * sa + lift * ca)
        st, ct = np.sin(theta), np.cos(theta)
        return [
            fx / model.mass - g * st,
            fz / model.mass + g * ct,
            (mom + model.arm_z_engine * thrust) / model.iy,
        ]

    sol, _, ier, _ = fsolve(residuals, [np.deg2rad(2.0), 0.0, 0.0],
                            full_output=True, xtol=1e-10)
    if ier != 1:
        return np.deg2rad(2.0), 0.0, np.deg2rad(-90.0), -999.0

    alpha, el, gamma = sol
    return alpha, el, gamma, V * np.sin(gamma)


def estimate_cruise_ceiling(
    model, aero_db, atmosphere, g: float,
    V: float, throttle: float,
    h_max: float = 6000.0, n_points: int = 2000,
) -> tuple[float, float, float, float]:
    """
    Sweeps altitudes from 0 to h_max and returns the highest point at which
    the aircraft can still sustain level flight at the given throttle and airspeed.

    Returns (ceiling_m, alpha_rad, elevator_deg, gamma_rad).
    Prints a progress table to stdout.
    """
    print(f"Ceiling sweep — V = {V} m/s  throttle = {throttle:.0%}")
    print("-" * 60)

    best = (0.0, 0.0, 0.0, 0.0)

    for h in np.linspace(0, h_max, n_points):
        alpha, el, gamma, climb_rate = _climbing_trim(
            model, aero_db, atmosphere, g, V, h, throttle
        )
        if climb_rate == -999.0 or abs(el) > 25.0 or np.rad2deg(alpha) > 15.0:
            print(f"  [{h:6.1f} m] Limits exceeded — stopping sweep")
            break

        print(f"  [{h:6.1f} m] Excess climb rate: {climb_rate:+.3f} m/s")

        if climb_rate <= 0.0:
            print(f"\n>>> Cruise ceiling crossed near {h:.1f} m")
            best = (h, alpha, el, gamma)
            break

        best = (h, alpha, el, gamma)

    return best   # (ceiling_m, alpha_rad, el_deg, gamma_rad)


# ---------------------------------------------------------------
# Transmitters
# ---------------------------------------------------------------
class ScriptedTransmitter:
    """Holds the aircraft at trim, with optional scripted deflections."""

    def __init__(self, trim_controls: dict) -> None:
        self.sim_time = 0.0
        self._trim_controls = trim_controls

    def read(self) -> tuple[float, float, float, float, float]:
        t = self.sim_time
        
        # 1. Grab ALL trim values safely
        ele_trim = self._trim_controls.get("elevator", 0.0)
        ail_trim = self._trim_controls.get("aileron", 0.0)
        rud_trim = self._trim_controls.get("rudder", 0.0)
        throttle = self._trim_controls.get("throttle", 0.0)

        # --- Scripted manoeuvre windows (edit as needed) ---
        ele_start, ele_end, ele_deflect = 4.0, 30.0, 0.0
        ail_start, ail_end, ail_deflect = 5.0, 16.0, 0.0
        rud_start, rud_end, rud_deflect = 5.0, 16.0, 0.0
        # ---------------------------------------------------

        # 2. Add deflections on top of the TRIM baseline, not zero
        ele = ele_trim + ele_deflect if ele_start <= t <= ele_end else ele_trim
        ail = ail_trim + ail_deflect if ail_start <= t <= ail_end else ail_trim
        rud = rud_trim + rud_deflect if rud_start <= t <= rud_end else rud_trim

        return ele, ail, rud, throttle, 0.0


class RCTransmitter:
    _AXIS_AILERON  = 0
    _AXIS_ELEVATOR = 1
    _AXIS_RUDDER   = 3

    def __init__(self, joystick_index: int = 0) -> None:
        self.joystick_index = joystick_index
        self._joystick      = None
        self._throttle_axis = None

    def _ensure_init(self) -> None:
        if self._joystick is not None:
            return

        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            raise RuntimeError("No joystick detected.")

        self._joystick = pygame.joystick.Joystick(self.joystick_index)
        self._joystick.init()
        print(f"Joystick initialised: {self._joystick.get_name()}")
        print(f"Number of axes: {self._joystick.get_numaxes()}")

        time.sleep(2)
        baseline = [self._joystick.get_axis(i) for i in range(self._joystick.get_numaxes())]
        detected = None

        for _ in range(200):
            pygame.event.pump()
            for i in range(self._joystick.get_numaxes()):
                if abs(self._joystick.get_axis(i) - baseline[i]) > 0.4:
                    detected = i
                    break
            if detected is not None:
                break
            time.sleep(0.01)

        if detected is None:
            raise RuntimeError("Could not detect R2 axis.")

        self._throttle_axis = detected
        print(f"Detected R2 throttle axis: {detected}")

    def read(self) -> tuple[float, float, float, float, float]:
        self._ensure_init()
        pygame.event.pump()

        ail = self._joystick.get_axis(self._AXIS_AILERON)
        ele = -self._joystick.get_axis(self._AXIS_ELEVATOR)
        rud = self._joystick.get_axis(self._AXIS_RUDDER)
        throttle = (self._joystick.get_axis(self._throttle_axis) + 1.0) / 2.0

        return 25 * ele, -20 * ail, 30 * rud, throttle, 0.0


# ---------------------------------------------------------------
# FlightGear Bridge
# ---------------------------------------------------------------
class FlightGearBridge:
    """
    Drives the 6-DOF integrator and streams state to FlightGear.

    Parameters
    ----------
    case_dir       : path to the case folder (must contain sim_config.toml + aircraft_model.toml)
    x0             : full initial state vector (from trim_opt or cfg.x0)
    trim_controls  : dict with keys elevator / aileron / rudder / throttle / brake
    manual_control : True -> RCTransmitter, False -> ScriptedTransmitter
    """

    def __init__(
        self,
        case_dir: pathlib.Path,
        x0: np.ndarray,
        trim_controls: dict,
        manual_control: bool = False,
    ) -> None:
        cfg     = SimConfig.from_toml_file(case_dir / "sim_config.toml")
        model   = load_model(case_dir / "aircraft_model.toml")
        aero_db = AeroDatabase(model.aero_tables_dir)

        self._transmitter = (
            RCTransmitter() if manual_control else ScriptedTransmitter(trim_controls)
        )

        self._x     = x0.copy()
        self._dx    = np.zeros_like(self._x)
        self._f     = make_state_eq(model, aero_db, self._transmitter.read, cfg.atmosphere)
        self._frame = 0

    # ------------------------------------------------------------------
    def _callback(self, fdm_data, event_pipe):
        for _ in range(STEPS_PER_SEND):
            rk4_step(self._f, self._x, self._dx, DT)
            if hasattr(self._transmitter, "sim_time"):
                self._transmitter.sim_time += DT

        x_e, y_e, z_e   = self._x[0], self._x[1], self._x[2]
        phi, theta, psi  = self._x[3], self._x[4], self._x[5]
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
        ele, ail, rud, throttle, _ = self._transmitter.read()
        u, v, w  = self._x[6], self._x[7], self._x[8]
        v_air    = max(np.sqrt(u**2 + v**2 + w**2), 1e-8)
        alpha    = np.degrees(np.arctan2(w, u))
        beta     = np.degrees(np.arcsin(np.clip(v / v_air, -1.0, 1.0)))
        x_e, y_e = self._x[0], self._x[1]
        track    = np.sqrt(x_e**2 + y_e**2)

        t_str = (
            f" t={self._transmitter.sim_time:.1f}s "
            if hasattr(self._transmitter, "sim_time")
            else " "
        )
        print(
            f"[{t_str}] ele: {ele:+.2f}  ail: {ail:+.2f}  rud: {rud:+.2f}  thr: {throttle:.2f}  "
            f"alt: {-(self._x[2] - groundSJC):.2f} m  alpha: {alpha:.1f}°  beta: {beta:.1f}°  "
            f"v_air: {v_air:.1f} m/s  track: {track:.2f} m",
            end="\r",
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
# Entry point
# ---------------------------------------------------------------
if __name__ == "__main__":
    # =========================================================
    # MASTER SIMULATION TOGGLES
    # =========================================================
    fg_cfg = FlightGearConfig.from_toml_file(CASE_DIR / "flightgear_config.toml")
    
    FIND_CRUISE_CEILING = fg_cfg.find_cruise_ceiling
    START_IN_AIR        = fg_cfg.start_in_air
    START_TRIMMED       = fg_cfg.start_trimmed
    MANUAL_CONTROL      = fg_cfg.manual_control
    # =========================================================

    cfg      = SimConfig.from_toml_file(CASE_DIR / "sim_config.toml")
    ac_model = load_model(CASE_DIR / "aircraft_model.toml")

    # ----------------------------------------------------------
    # ANALYSIS ONLY: ceiling sweep — prints results and exits
    # ----------------------------------------------------------
    if FIND_CRUISE_CEILING:
        from flightsim.aero.database import AeroDatabase as _ADB
        _aero_db = _ADB(ac_model.aero_tables_dir)
        ceiling, a_trim, el_trim, g_trim = estimate_cruise_ceiling(
            model     = ac_model,
            aero_db   = _aero_db,
            atmosphere= cfg.atmosphere,
            g         = 9.81,
            V         = cfg.v_des,
            throttle  = fg_cfg.throttleceiling,          # adjust as needed
        )
        print(f"\nResult — Ceiling: {ceiling:.1f} m  |  alpha: {np.rad2deg(a_trim):.2f}°  "
              f"elevator: {el_trim:.2f}°  gamma: {np.rad2deg(g_trim):.2f}°")
        raise SystemExit(0)

    print("\n==================================================")

    if not START_IN_AIR:
        # ----------------------------------------------------------
        # GROUND START — takeoff roll from TOML initial conditions
        # ----------------------------------------------------------
        print(" MODE: GROUND START  (takeoff roll)")
        x0 = cfg.x0.copy()
        trim_controls = {
            "elevator": 0.0,
            "aileron":  0.0,
            "rudder":   0.0,
            "throttle": 1.0,   # full throttle for the run
            "brake":    0.0,
        }
        print(f" -> Throttle: {trim_controls['throttle'] * 100:.0f}%  |  Elevator: neutral")

    elif START_TRIMMED:
        # ----------------------------------------------------------
        # AIRBORNE + TRIMMED — mirrors main.py's trim_opt call
        # ----------------------------------------------------------
        print(" MODE: AIRBORNE START — TRIMMED")
        print(
            f" -> Condition : {cfg.trim_condition}\n"
            f"    V = {cfg.v_des} m/s  |  h = {cfg.h_des} m  |  γ = {np.rad2deg(cfg.gamma_des):.1f}°"
        )
        x0, trim_controls = trim_opt(
            cfg.v_des,
            cfg.h_des,
            cfg.gamma_des,
            cfg.radiusdes,
            cfg.atmosphere,
            ac_model,
            condition=cfg.trim_condition,
        )
        print(
            f" -> Elevator : {trim_controls['elevator']:.4f}°  "
            f"Throttle : {trim_controls['throttle'] * 100:.1f}%"
        )

    else:
        # ----------------------------------------------------------
        # AIRBORNE + UNTRIMMED — TOML x0, neutral controls
        # ----------------------------------------------------------
        print(" MODE: AIRBORNE START — UNTRIMMED  (TOML initial conditions)")
        x0 = cfg.x0.copy()
        trim_controls = {
            "elevator": 0.0,
            "aileron":  0.0,
            "rudder":   0.0,
            "throttle": 0.5,
            "brake":    0.0,
        }
        print(f" -> Throttle: {trim_controls['throttle'] * 100:.0f}%  |  All surfaces: neutral")

    print("==================================================\n")

    bridge = FlightGearBridge(
        case_dir=CASE_DIR,
        x0=x0,
        trim_controls=trim_controls,
        manual_control=MANUAL_CONTROL,
    )
    bridge.run()