from __future__ import annotations

import os
import shlex
import signal
import subprocess
import time
from typing import Callable

import numpy as np

from flightsim.case import SimCase
from flightsim.control.source import ControlInput, ControlSource, ScriptedControl
from flightsim.core.simulation import Simulator
from flightsim.core.state import StateIndex as I
from flightsim.session import Session

R_EARTH = 6_378_137.0
M_TO_FT = 3.28084


def join_command(args: list[str]) -> str:
    return subprocess.list2cmdline(args) if os.name == "nt" else shlex.join(args)


def fgfs_command(case: SimCase) -> list[str]:
    fg = case
    altitude = {"trimmed": case.trim_altitude, "initial": case.altitude}.get(fg.fg_start_mode, case.ground_elevation)
    heading = case.psi if fg.fg_start_mode == "initial" else fg.fg_heading
    return [fg.fg_executable, "--fdm=external",
            f"--native-fdm=socket,out,{fg.fg_packet_hz},{fg.fg_host},{fg.fg_port_in},udp",
            f"--native-fdm=socket,in,{fg.fg_packet_hz},{fg.fg_host},{fg.fg_port_out},udp",
            f"--aircraft={fg.fg_aircraft}", f"--lat={fg.fg_latitude}", f"--lon={fg.fg_longitude}",
            f"--heading={heading % 360:.1f}", f"--altitude={altitude * M_TO_FT:.0f}",
            *shlex.split(fg.fg_extra_args, posix=os.name != "nt")]


def make_controls(session: Session, base: ControlInput, kind: str | None = None) -> ControlSource:
    c = session.case
    kind = kind or c.fg_control
    if kind == "joystick":
        from flightsim.control.devices import AxisMap, JoystickControl
        axes = AxisMap(c.js_index, c.js_aileron, c.js_elevator, c.js_rudder, c.js_throttle, c.js_brake,
                       c.js_invert_aileron, c.js_invert_elevator, c.js_invert_rudder, c.js_invert_throttle, c.js_deadband)
        return JoystickControl(axes, session.control_limits())
    if kind == "keyboard":
        from flightsim.control.devices import KeyboardControl
        return KeyboardControl(session.control_limits())
    if kind == "scripted":
        return ScriptedControl(base, c.maneuver_list())
    raise ValueError(f"Unknown control source '{kind}'")


class FlightGearBridge:
    def __init__(self, session: Session, log: Callable[[str], None] = print, control_kind: str | None = None) -> None:
        self.session = session
        self.case = session.case
        self.log = log
        self.dynamics = session.dynamics(ScriptedControl(self.case.baseline_controls()))
        self.x0, base = self._start_state()
        self.controls = make_controls(session, base, control_kind)
        self.dynamics.controls = self.controls
        self.log(f"Pilot input: {control_kind or self.case.fg_control}" + (f" ({self.controls.name})" if hasattr(self.controls, "name") else ""))
        self.sim = Simulator(self.dynamics, self.x0)
        self.dt = 1.0 / self.case.fg_fdm_hz
        self.steps_per_packet = max(1, round(self.case.fg_fdm_hz / self.case.fg_packet_hz))
        self.lat0 = np.radians(self.case.fg_latitude)
        self.lon0 = np.radians(self.case.fg_longitude)
        self._last_print = 0.0

    def _start_state(self):
        mode = self.case.fg_start_mode
        base = self.case.baseline_controls()
        if mode == "initial":
            self.log("Start: airborne from case initial conditions")
            self.session.check_altitude(self.case.altitude, "Initial altitude")
            return self.case.initial_state(), base
        if mode == "ground":
            if not self.case.ground_contact:
                raise ValueError("Ground start requires 'Landing gear / ground contact' enabled in the case environment")
            self.log(f"Start: on ground at {self.case.ground_elevation} m elevation, heading {self.case.fg_heading}°")
            x0 = self.session.ground_state()
        else:
            result = self.session.trim(self.dynamics)
            self.log(result.summary())
            x0, base = result.x0, result.controls
        x0[I.PSI] = np.radians(self.case.fg_heading)
        return x0, base

    def geodetic(self, x_e: float, y_e: float) -> tuple[float, float]:
        return self.lat0 + x_e / R_EARTH, self.lon0 + y_e / (R_EARTH * np.cos(self.lat0))

    def callback(self, fdm, _pipe=None):
        for _ in range(self.steps_per_packet):
            self.sim.step(self.dt)
        x = self.sim.x
        lat, lon = self.geodetic(x[I.X_E], x[I.Y_E])
        u, v, w = x[I.U:I.W + 1]
        speed = max(np.sqrt(u**2 + v**2 + w**2), 1e-8)
        fdm.lat_rad, fdm.lon_rad, fdm.alt_m = lat, lon, -x[I.Z_E]
        fdm.agl_m = -x[I.Z_E] - self.case.ground_elevation
        fdm.phi_rad, fdm.theta_rad, fdm.psi_rad = x[I.PHI], x[I.THETA], x[I.PSI] % (2 * np.pi)
        fdm.alpha_rad, fdm.beta_rad = np.arctan2(w, u), np.arcsin(np.clip(v / speed, -1, 1))
        fdm.phidot_rad_per_s, fdm.thetadot_rad_per_s, fdm.psidot_rad_per_s = self.sim.dx[I.PHI:I.PSI + 1]
        fdm.v_body_u, fdm.v_body_v, fdm.v_body_w = u * M_TO_FT, v * M_TO_FT, w * M_TO_FT
        fdm.v_north_ft_per_s, fdm.v_east_ft_per_s, fdm.v_down_ft_per_s = self.sim.dx[I.X_E:I.Z_E + 1] * M_TO_FT
        fdm.vcas = speed * 1.94384
        fdm.climb_rate_ft_per_s = -self.sim.dx[I.Z_E] * M_TO_FT
        cmd = self.dynamics.limit(self.controls.get(self.sim.t))
        m = self.dynamics.model
        fdm.elevator = cmd.elevator / m.elevator_max
        fdm.left_aileron, fdm.right_aileron = cmd.aileron / m.aileron_max, -cmd.aileron / m.aileron_max
        fdm.rudder = cmd.rudder / m.rudder_max
        fdm.cur_time_s = int(time.time())
        if time.monotonic() - self._last_print > 0.5:
            self._last_print = time.monotonic()
            self.log(self.status(cmd, speed))
        return fdm

    def status(self, cmd, speed) -> str:
        x = self.sim.x
        return (f"t {self.sim.t:7.1f}s | ele {cmd.elevator:+6.1f} ail {cmd.aileron:+6.1f} rud {cmd.rudder:+6.1f} "
                f"thr {cmd.throttle:4.2f} | agl {-x[I.Z_E] - self.case.ground_elevation:7.1f} m | V {speed:5.1f} m/s | "
                f"alpha {np.degrees(np.arctan2(x[I.W], x[I.U])):+5.1f}° theta {np.degrees(x[I.THETA]):+5.1f}° "
                f"phi {np.degrees(x[I.PHI]):+6.1f}° psi {np.degrees(x[I.PSI]) % 360:5.1f}°"
                + ("  [ground]" if self.dynamics.on_ground else ""))

    def run(self) -> None:
        from flightgear_python.fg_if import FDMConnection
        c = self.case
        conn = FDMConnection(rx_timeout_s=5.0)
        conn.connect_rx(c.fg_host, c.fg_port_in, self.callback)
        conn.connect_tx(c.fg_host, c.fg_port_out)
        self.log(f"Waiting for FlightGear packets on {c.fg_host}:{c.fg_port_in} (sending to :{c.fg_port_out}) ...")
        self.log("Launch FlightGear with: " + join_command(fgfs_command(c)))

        def stop(*_):
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, stop)
        try:
            while True:
                try:
                    conn._fg_packet_roundtrip()
                except Exception as exc:
                    if "Timeout" in str(exc):
                        self.log("No packets from FlightGear yet, still waiting ...")
                        continue
                    raise
        except KeyboardInterrupt:
            self.log("Bridge stopped.")
        finally:
            self.controls.close()
