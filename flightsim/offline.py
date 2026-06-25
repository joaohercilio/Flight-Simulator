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
from flightsim.core.trim import TrimSolver
from utils.io import generate_plots


class Application:


    def __init__(self, case: Case, aircraft: AircraftModel) -> None:
        self.case = case
        self.aircraft = aircraft


    def resolve_initial_conditions(self) -> tuple[NDArray, ControlInput]:
        """Returns (x0, controls) from trim or the manual initial condition.


        Returns:
            Tuple (initial_state, controls).
        """

        if self.case.enable_trim:
            self._report(f"Performing trim optimization ({case.trim_condition})...")
            trimSolver = TrimSolver(self.model, self.case)
            trimResult = trimSolver.solve()
            self._report(trimResult.summary())
            return trimResult.x0, trimResult.controls

        self._report("Bypassing trim optimization. Using manual initial conditions.")
        x0 =
        return ,

    def build_simulator(self, controls: ControlSource) -> Simulator:
        """Builds a Simulator wired to the given control source."""
        return Simulator(
            self.case.model, self.config.atmosphere, controls, self.case.aero_db,
        )

    def run_offline(
        self, force_no_trim: bool = False,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Runs the full offline pipeline and generates plots.

        Args:
            force_no_trim: If True, skip trim optimisation.

        Returns:
            Tuple (t, x, dx) — see Simulator.run.
        """
        self._report(self.case.model.summary())

        x0, trim = self.resolve_initial_conditions(force_no_trim)
        controls = ConstantControl(trim)

        sim = self.build_simulator(controls)
        cfg = self.config
        t, x, dx = sim.run(cfg.t_start, cfg.t_end, cfg.dt, x0)

        generate_plots(
            t, x, dx,
            plot_config=cfg.plot_config,
            output_dir=cfg.output_dir,
            save_figures=cfg.save_figures,
            show_gui=cfg.show_gui,
        )
        return t, x, dx
