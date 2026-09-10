from __future__ import annotations

import pathlib
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from flightsim.aero.database import AeroDatabase
from flightsim.aircraft import AircraftModel
from flightsim.case import SimCase
from flightsim.control.source import ControlInput, ControlSource, ScriptedControl
from flightsim.core.dynamics import Dynamics
from flightsim.core.simulation import SimulationResult, Simulator
from flightsim.core.state import StateIndex
from flightsim.core.trim import TrimResult, TrimSolver
from flightsim.environment import Environment

CASE_FILE = "case.toml"
Logger = Callable[[str], None]


class Session:
    def __init__(self, case_dir: pathlib.Path, case: SimCase | None = None, aircraft: AircraftModel | None = None) -> None:
        self.case_dir = pathlib.Path(case_dir)
        self.case = case or SimCase.load(self.case_dir / CASE_FILE)
        self.aircraft = aircraft or AircraftModel.load(self.case.aircraft_path(self.case_dir))
        self._aero_db: AeroDatabase | None = None

    @property
    def aero_db(self) -> AeroDatabase:
        if self._aero_db is None or self._aero_db.tables_dir != self.aircraft.tables_path(self.case_dir):
            self._aero_db = AeroDatabase(self.aircraft.tables_path(self.case_dir))
        return self._aero_db

    @property
    def output_dir(self) -> pathlib.Path:
        return self.case_dir / self.case.output_dir

    def control_limits(self) -> tuple[float, float, float]:
        return self.aircraft.elevator_max, self.aircraft.aileron_max, self.aircraft.rudder_max

    def dynamics(self, controls: ControlSource, env: Environment | None = None) -> Dynamics:
        return Dynamics(self.aircraft, self.aero_db, controls, env or Environment.from_case(self.case))

    def check_altitude(self, altitude: float, what: str) -> None:
        if self.case.ground_contact and altitude <= self.case.ground_elevation:
            raise ValueError(f"{what} ({altitude} m) is at or below the ground elevation ({self.case.ground_elevation} m); "
                             "raise it or disable ground contact")

    def trim(self, dynamics: Dynamics | None = None) -> TrimResult:
        c = self.case
        self.check_altitude(c.trim_altitude, "Trim altitude")
        dyn = dynamics or self.dynamics(ScriptedControl(c.baseline_controls()))
        return TrimSolver(dyn).solve(c.trim_condition, c.trim_airspeed, c.trim_altitude, c.trim_gamma, c.trim_radius)

    def initial_conditions(self, log: Logger = print, dynamics: Dynamics | None = None) -> tuple[NDArray, ControlInput]:
        if self.case.trim_enable:
            result = self.trim(dynamics)
            log(result.summary())
            return result.x0, result.controls
        log("Using initial conditions from case (no trim).")
        self.check_altitude(self.case.altitude, "Initial altitude")
        return self.case.initial_state(), self.case.baseline_controls()

    def ground_state(self) -> NDArray:
        x0 = self.case.initial_state()
        x0[StateIndex.Z_E] = -(self.case.ground_elevation + self.aircraft.z_cg)
        x0[StateIndex.U:StateIndex.R + 1] = 0.0
        x0[StateIndex.PHI:StateIndex.THETA + 1] = 0.0
        return x0

    def run(self, log: Logger = print, progress: Callable[[float], None] | None = None) -> SimulationResult:
        log(self.aircraft.report())
        scripted = ScriptedControl(ControlInput(), self.case.maneuver_list())
        dyn = self.dynamics(scripted)
        x0, base = self.initial_conditions(log, dyn)
        dyn.controls = ScriptedControl(base, self.case.maneuver_list())
        if self.case.maneuvers:
            log("Maneuvers: " + "; ".join(f"{m.surface} {m.deflection:+g} @ {m.start}-{m.end}s" for m in self.case.maneuver_list()))
        log(f"Simulating {self.case.t_end} s at dt = {self.case.dt} s ...")
        result = Simulator(dyn, x0).run(self.case.t_end, self.case.dt, progress)
        s = result.x[:, -1]
        log(f"Done. Final: h {-s[StateIndex.Z_E]:.1f} m  V {result.airspeed[-1]:.2f} m/s  "
            f"alpha {np.degrees(result.alpha[-1]):.2f}°  theta {np.degrees(s[StateIndex.THETA]):.2f}°")
        return result
