from __future__ import annotations

import dataclasses
import pathlib

from PySide6 import QtWidgets

from flightsim.case import SimCase
from gui.forms import SchemaForm
from gui.widgets import ManeuverTable

TITLES = {
    "simulation": "Time",
    "environment": "Environment",
    "trim": "Trim (initial condition solved for steady flight)",
    "initial_condition": "Initial condition (used when trim is disabled)",
    "control": "Baseline controls (used when trim is disabled)",
    "wind": "Wind and gusts",
}


class SimulationTab(QtWidgets.QScrollArea):
    def __init__(self, case_dir: pathlib.Path) -> None:
        super().__init__()
        self.form = SchemaForm(SimCase, TITLES, case_dir)
        self.maneuvers = ManeuverTable()
        box = QtWidgets.QGroupBox("Scripted maneuvers")
        QtWidgets.QVBoxLayout(box).addWidget(self.maneuvers)
        grid = self.form.layout()
        grid.addWidget(box, 3, 0, 1, 2)
        grid.setRowStretch(3, 0)
        grid.setRowStretch(4, 1)
        self.form.widgets["trim_enable"].toggled.connect(self._trim_toggled)
        self.setWidget(self.form)
        self.setWidgetResizable(True)

    def _trim_toggled(self, enabled: bool) -> None:
        for name in ("trim_condition", "trim_airspeed", "trim_altitude", "trim_gamma", "trim_radius"):
            self.form.widgets[name].setEnabled(enabled)
        self.form.groups["initial_condition"].setEnabled(not enabled)
        self.form.groups["control"].setEnabled(not enabled)

    def load(self, case: SimCase, case_dir: pathlib.Path) -> None:
        self.form.set_base_dir(case_dir)
        self.form.load(case)
        self.maneuvers.load(case.maneuvers)
        self._trim_toggled(case.trim_enable)

    def apply(self, case: SimCase) -> SimCase:
        return dataclasses.replace(self.form.apply(case), maneuvers=self.maneuvers.values())
