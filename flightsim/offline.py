# flightsim/offline.py
"""Application layer: orchestrates a full offline run.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from flightsim.case import Case
from flightsim.aircraft import AircraftModel
from flightsim.control.source import ConstantControl, ControlInput, ControlSource
from flightsim.core.simulation import Simulator
from flightsim.core.state import StateVector
from flightsim.core.trim import TrimSolver
from flightsim.aero.database import AeroDatabase
from flightsim.environment.density import DensityModel
from flightsim.environment.gravity import GravityModel


class Application:


    def __init__(self, case: Case, aircraft: AircraftModel, console: Callable[[str], None]) -> None:
        self.case = case
        self.aircraft = aircraft
        self.console = console


    def resolve_initial_conditions(self) -> tuple[NDArray, ControlInput]:
        if self.case.enable_trim:
            self.console(f"Performing trim optimization ({case.trim_name})...")
            trimSolver = TrimSolver(self.model, self.case)
            trimResult = trimSolver.solve()
            self.console(trimResult.summary())
            return trimResult.x0, trimResult.controls

        self.console("Bypassing trim optimization. Using manual initial conditions.")
        x0 = StateVector.build_state_from_case(self.case)
        control = ControlInput()
        return x0, control


    def build_aero_db(self) -> AeroDatabase:
        aero_db = AeroDatabase(self.aircraft.aero_tables_dir)
        return aero_db


    def build_environment(self) -> tuple[DensityModel, GravityModel]:
        density_model = DensityModel.build(self.case.density_model, self.case.density)
        gravity_model = GravityModel.build(self.case.gravity_model, self.case.gravity)
        self.console (f"Density model: {self.case.density_model} ")
        self.console (f"Gravity model: {self.case.gravity_model} ")
        return density_model, gravity_model


    def run_offline(self) -> tuple[NDArray, NDArray, NDArray]:
        """Runs the full offline pipeline and generates plots.
        """

        x0, trim = self.resolve_initial_conditions()

        controls = ConstantControl(trim)

        aero_db = self.build_aero_db()

        density_model, gravity_model = self.build_environment()

        sim = Simulator(self.aircraft, controls, aero_db, density_model, gravity_model)

        t, x, dx = sim.run(t_end = self.case.total_time, dt = self.case.time_step, x0 = x0)

        return t, x, dx
















