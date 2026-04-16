# main_flightgear.py
"""FlightGear real-time entry point for the 6DOF flight simulator.

Connects to FlightGear via FDM protocol, reads joystick input from
a RC transmitter, and integrates the equations of motion in real time.

FlightGear UDP configuration (add to your .xml or command line):
    --generic=socket,out,60,localhost,5501,udp,generic
    --generic=socket,in,60,localhost,5502,udp,generic
"""

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

# ---------------------------------------------------------------
# Escolha o caso (deve coincidir com main.py)
# ---------------------------------------------------------------
CASE_DIR = pathlib.Path("cases/mushu")

# ---------------------------------------------------------------
# Integração
# ---------------------------------------------------------------
FDM_HZ       = 240          # frequência interna de integração (Hz)
SEND_HZ      = 60           # frequência de envio ao FlightGear (Hz)
DT           = 1.0 / FDM_HZ
STEPS_PER_SEND = FDM_HZ // SEND_HZ   # passos internos por callback

# ---------------------------------------------------------------
# Origem geodésica (NED → lat/lon)
# ---------------------------------------------------------------
_R_EARTH   = 6_378_137.0   # m
_LAT0_DEG  = 19.7205
_LON0_DEG  = -155.0610
_LAT0      = np.radians(_LAT0_DEG)
_LON0      = np.radians(_LON0_DEG)

# ---------------------------------------------------------------
# Debug
# ---------------------------------------------------------------
PRINT_EACH = 60   # imprime a cada N callbacks


def _ned_to_geodetic(x_e: float, y_e: float) -> tuple[float, float]:
    """Converts NED position to geodetic coordinates.

    Args:
        x_e: North displacement from origin (m).
        y_e: East displacement from origin (m).

    Returns:
        Tuple (lat_rad, lon_rad).
    """
    lat = _LAT0 + x_e / _R_EARTH
    lon = _LON0 + y_e / (_R_EARTH * np.cos(_LAT0))
    return lat, lon


class RCTransmitter:
    """Reads control inputs from a RC transmitter via pygame joystick.

    Axis and button mapping should be adjusted to match the specific
    transmitter model. Current mapping is for a generic RC USB adapter
    with 4 primary axes.

    Channel layout (adjust indices to match your transmitter):
        Axis 0 — aileron   (roll)
        Axis 1 — elevator  (pitch, inverted)
        Axis 2 — throttle  (rescaled from [-1, 1] to [0, 1])
        Axis 3 — rudder
        Axis 4 — brake     (rescaled from [-1, 1] to [0, 1])
    """

    # --- ajuste os índices dos eixos aqui ---
    _AXIS_AILERON  = 0
    _AXIS_ELEVATOR = 1
    _AXIS_THROTTLE = 2
    _AXIS_RUDDER   = 3
    _AXIS_BRAKE    = 4

    def __init__(self, joystick_index: int = 0) -> None:
        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            raise RuntimeError(
                "No joystick detected. "
                "Connect the RC transmitter USB adapter and try again."
            )

        self._joystick = pygame.joystick.Joystick(joystick_index)
        self._joystick.init()
        print(f"Joystick: {self._joystick.get_name()}")

    def read(self) -> tuple[float, float, float, float, float]:
        """Reads current control inputs.

        Returns:
            Tuple (ele, ail, rud, throttle, brake), all in range [-1, 1]
            except throttle and brake which are in [0, 1].
        """
        pygame.event.pump()

        ail      =  self._joystick.get_axis(self._AXIS_AILERON)
        ele      = -self._joystick.get_axis(self._AXIS_ELEVATOR)   # inverted
        rud      =  self._joystick.get_axis(self._AXIS_RUDDER)
        throttle = (self._joystick.get_axis(self._AXIS_THROTTLE) + 1.0) / 2.0
        brake    = (self._joystick.get_axis(self._AXIS_BRAKE)    + 1.0) / 2.0
        brake = 0
        return ele, ail, rud, throttle, brake


class FlightGearBridge:
    """Manages real-time integration and FlightGear data exchange.

    Args:
        case_dir: Path to the simulation case directory.
    """

    def __init__(self, case_dir: pathlib.Path) -> None:
        cfg   = SimConfig.from_toml_file(case_dir / "sim_config.toml")
        model = load_model(case_dir / "aircraft_model.toml")
        aero_db = AeroDatabase(model.aero_tables_dir)

        self._transmitter = RCTransmitter()
        self._x  = cfg.x0.copy()
        self._dx = np.zeros_like(self._x)
        self._f  = make_state_eq(model, aero_db, self._transmitter.read, cfg.atmosphere)
        self._frame = 0

    def _callback(self, fdm_data, event_pipe):
        """FDM callback called by FlightGear at SEND_HZ.

        Integrates STEPS_PER_SEND internal steps then updates fdm_data.

        Args:
            fdm_data: FlightGear FDM data object (modified in-place).
            event_pipe: Unused event pipe from flightgear_python.

        Returns:
            Updated fdm_data.
        """
        for _ in range(STEPS_PER_SEND):
            rk4_step(self._f, self._x, self._dx, DT)

        x_e, y_e, z_e     = self._x[0], self._x[1], self._x[2]
        phi, theta, psi   = self._x[3], self._x[4], self._x[5]
        altitude          = -z_e

        lat, lon = _ned_to_geodetic(x_e, y_e)

        fdm_data.lon_rad   = lon
        fdm_data.lat_rad   = lat
        fdm_data.alt_m     = altitude
        fdm_data.phi_rad   = phi
        fdm_data.theta_rad = theta
        fdm_data.psi_rad   = psi

        if self._frame % PRINT_EACH == 0:
            self._print_status()

        self._frame += 1
        return fdm_data

    def _print_status(self) -> None:
        """Prints a one-line status summary to stdout."""
        ele, ail, rud, throttle, brake = self._transmitter.read()

        u, v, w  = self._x[6], self._x[7], self._x[8]
        altitude = -self._x[2]
        v_air    = np.sqrt(u**2 + v**2 + w**2)
        v_air    = max(v_air, 1e-8)
        alpha    = np.degrees(np.arctan2(w, u))
        beta     = np.degrees(np.arcsin(np.clip(v / v_air, -1.0, 1.0)))

        print(
            f"ele: {ele:+.2f}  ail: {ail:+.2f}  rud: {rud:+.2f}  "
            f"thr: {throttle:.2f}  brk: {brake:.2f}  "
            f"alt: {altitude:.0f} m  "
            f"alpha: {alpha:.1f}°  beta: {beta:.1f}°"
        )

    def run(self) -> None:
        """Starts the FlightGear connection and blocks until interrupted."""
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
    bridge = FlightGearBridge(CASE_DIR)
    bridge.run()
