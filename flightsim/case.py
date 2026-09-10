from __future__ import annotations

import dataclasses
import pathlib

import numpy as np
from numpy.typing import NDArray

from flightsim.config import load_toml, save_toml, spec
from flightsim.control.source import ControlInput, Maneuver
from flightsim.core.state import StateVector

TRIM_CONDITIONS = ["steady_level_flight", "steady_climb", "coordinated_turn", "glide"]
DENSITY_MODELS = ["constant", "isa"]
START_MODES = ["trimmed", "initial", "ground"]
CONTROL_SOURCES = ["joystick", "keyboard", "scripted"]
DEFAULT_FIGURES = [
    ["Position", "Velocity NED"],
    ["Euler angles", "Euler rates", "Angular velocity"],
    ["Aerodynamics", "Body velocity", "Controls"],
    ["Trajectory 3D"],
]


@dataclasses.dataclass
class SimCase:
    name: str = spec("case", "case", "Name")
    aircraft: str = spec("case", "aircraft.toml", "Aircraft file", path=True)
    t_end: float = spec("simulation", 30.0, "Duration", "s", min=0.01, max=1e5)
    dt: float = spec("simulation", 0.01, "Time step", "s", decimals=4, min=1e-4, max=1.0)
    density_model: str = spec("environment", "constant", "Density model", choices=DENSITY_MODELS)
    density: float = spec("environment", 1.225, "Air density (constant model)", "kg/m³", decimals=4)
    gravity: float = spec("environment", 9.81, "Gravity", "m/s²", decimals=3)
    ground_elevation: float = spec("environment", 0.0, "Ground elevation", "m", min=-1e3)
    ground_contact: bool = spec("environment", True, "Landing gear / ground contact")
    wind_north: float = spec("wind", 0.0, "Steady wind north", "m/s", min=-100.0)
    wind_east: float = spec("wind", 0.0, "Steady wind east", "m/s", min=-100.0)
    wind_down: float = spec("wind", 0.0, "Steady wind down", "m/s", min=-100.0)
    gust_enable: bool = spec("wind", False, "Random gusts")
    gust_amplitude: float = spec("wind", 6.0, "Max gust amplitude", "m/s")
    gust_duration_min: float = spec("wind", 1.0, "Gust duration min", "s")
    gust_duration_max: float = spec("wind", 5.0, "Gust duration max", "s")
    gust_interval_min: float = spec("wind", 2.0, "Interval between gusts min", "s")
    gust_interval_max: float = spec("wind", 4.0, "Interval between gusts max", "s")
    gust_seed: int = spec("wind", 0, "Random seed (0 = random)")
    x: float = spec("initial_condition", 0.0, "North x", "m", min=-1e6, max=1e6)
    y: float = spec("initial_condition", 0.0, "East y", "m", min=-1e6, max=1e6)
    altitude: float = spec("initial_condition", 100.0, "Altitude", "m", min=-1e3, max=1e5)
    u: float = spec("initial_condition", 15.0, "u", "m/s", decimals=3, min=-500.0, max=500.0)
    v: float = spec("initial_condition", 0.0, "v", "m/s", decimals=3, min=-500.0, max=500.0)
    w: float = spec("initial_condition", 0.0, "w", "m/s", decimals=3, min=-500.0, max=500.0)
    phi: float = spec("initial_condition", 0.0, "Roll φ", "deg", min=-180.0, max=180.0)
    theta: float = spec("initial_condition", 0.0, "Pitch θ", "deg", min=-89.0, max=89.0)
    psi: float = spec("initial_condition", 0.0, "Yaw ψ", "deg", min=-360.0, max=360.0)
    p: float = spec("initial_condition", 0.0, "Roll rate p", "deg/s", min=-1e3, max=1e3)
    q: float = spec("initial_condition", 0.0, "Pitch rate q", "deg/s", min=-1e3, max=1e3)
    r: float = spec("initial_condition", 0.0, "Yaw rate r", "deg/s", min=-1e3, max=1e3)
    trim_enable: bool = spec("trim", True, "Start from trimmed condition")
    trim_condition: str = spec("trim", "steady_level_flight", "Condition", choices=TRIM_CONDITIONS)
    trim_airspeed: float = spec("trim", 15.0, "Airspeed", "m/s")
    trim_altitude: float = spec("trim", 100.0, "Altitude", "m", min=-1e3, max=1e5)
    trim_gamma: float = spec("trim", 0.0, "Flight path angle γ (climb)", "deg", min=-60.0, max=60.0)
    trim_radius: float = spec("trim", 50.0, "Turn radius (coordinated turn)", "m", max=1e6)
    elevator: float = spec("control", 0.0, "Elevator", "deg", min=-90.0, max=90.0)
    aileron: float = spec("control", 0.0, "Aileron", "deg", min=-90.0, max=90.0)
    rudder: float = spec("control", 0.0, "Rudder", "deg", min=-90.0, max=90.0)
    throttle: float = spec("control", 0.5, "Throttle", "", decimals=3, max=1.0)
    brake: float = spec("control", 0.0, "Brake", "", decimals=3, max=1.0)
    maneuvers: list = spec("maneuver", list, "Maneuvers", table=True)
    fg_host: str = spec("flightgear", "localhost", "Host", key="host")
    fg_port_in: int = spec("flightgear", 5501, "Port FlightGear → sim", key="port_in", max=65535)
    fg_port_out: int = spec("flightgear", 5502, "Port sim → FlightGear", key="port_out", max=65535)
    fg_packet_hz: int = spec("flightgear", 60, "Packet rate", "Hz", key="packet_hz", min=1, max=1000)
    fg_fdm_hz: int = spec("flightgear", 240, "Integration rate", "Hz", key="fdm_hz", min=1, max=10000)
    fg_start_mode: str = spec("flightgear", "trimmed", "Start mode", key="start_mode", choices=START_MODES)
    fg_control: str = spec("flightgear", "joystick", "Pilot input", key="control", choices=CONTROL_SOURCES)
    fg_latitude: float = spec("flightgear", -23.2292, "Reference latitude", "deg", key="latitude", decimals=6, min=-90.0, max=90.0)
    fg_longitude: float = spec("flightgear", -45.8615, "Reference longitude", "deg", key="longitude", decimals=6, min=-180.0, max=180.0)
    fg_heading: float = spec("flightgear", 0.0, "Initial heading", "deg", key="heading", min=-360.0, max=360.0)
    fg_aircraft: str = spec("flightgear", "c172p", "FlightGear visual model", key="aircraft")
    fg_executable: str = spec("flightgear", "fgfs", "fgfs executable", key="executable", path=True)
    fg_extra_args: str = spec("flightgear", "--timeofday=noon --disable-ai-traffic", "Extra fgfs arguments", key="extra_args")
    js_index: int = spec("joystick", 0, "Device index", key="index", max=16)
    js_aileron: int = spec("joystick", 0, "Aileron axis", key="aileron_axis", max=32)
    js_elevator: int = spec("joystick", 1, "Elevator axis", key="elevator_axis", max=32)
    js_rudder: int = spec("joystick", 3, "Rudder axis", key="rudder_axis", max=32)
    js_throttle: int = spec("joystick", 2, "Throttle axis", key="throttle_axis", max=32)
    js_brake: int = spec("joystick", -1, "Brake axis (-1 = none)", key="brake_axis", min=-1, max=32)
    js_invert_aileron: bool = spec("joystick", False, "Invert aileron", key="invert_aileron")
    js_invert_elevator: bool = spec("joystick", True, "Invert elevator", key="invert_elevator")
    js_invert_rudder: bool = spec("joystick", False, "Invert rudder", key="invert_rudder")
    js_invert_throttle: bool = spec("joystick", False, "Invert throttle", key="invert_throttle")
    js_deadband: float = spec("joystick", 0.05, "Deadband", "", key="deadband", decimals=3, max=0.5)
    save_figures: bool = spec("plots", False, "Save figures")
    output_dir: str = spec("plots", "results", "Output directory", path=True)
    figures: list = spec("plots", lambda: [list(f) for f in DEFAULT_FIGURES], "Figures")

    def initial_state(self) -> NDArray:
        s = StateVector(np.zeros(12))
        s.x_e, s.y_e, s.z_e = self.x, self.y, -self.altitude
        s.phi, s.theta, s.psi = np.radians([self.phi, self.theta, self.psi])
        s.u, s.v, s.w = self.u, self.v, self.w
        s.p, s.q, s.r = np.radians([self.p, self.q, self.r])
        return s.to_array()

    def baseline_controls(self) -> ControlInput:
        return ControlInput(self.elevator, self.aileron, self.rudder, self.throttle, self.brake)

    def maneuver_list(self) -> list[Maneuver]:
        return [Maneuver(**m) for m in self.maneuvers]

    def aircraft_path(self, case_dir: pathlib.Path) -> pathlib.Path:
        return pathlib.Path(case_dir) / self.aircraft

    @classmethod
    def load(cls, path: pathlib.Path) -> SimCase:
        return load_toml(cls, path)

    def save(self, path: pathlib.Path) -> None:
        save_toml(self, path)
