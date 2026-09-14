from __future__ import annotations

import dataclasses
import pathlib

from PySide6 import QtWidgets

from flightsim.case import SimCase
from gui.forms import SchemaForm
from gui.widgets import ManeuverTable

COLUMNS = [
    {"simulation": "Time", "trim": "Trim (initial condition solved for steady flight)",
     "initial_condition": "Initial condition (trim disabled, or FlightGear 'initial' start)"},
    {"environment": "Environment", "control": "Baseline controls (trim disabled, or FlightGear ground start)",
     "wind": "Wind and gusts"},
]
TRIM_FIELDS = ["trim_condition", "trim_airspeed", "trim_altitude", "trim_gamma", "trim_radius"]
INITIAL_CONDITION_FIELDS = ["x", "y", "altitude", "u", "v", "w", "phi", "theta", "psi", "p", "q", "r"]
GUST_FIELDS = ["gust_amplitude", "gust_duration_min", "gust_duration_max", "gust_interval_min", "gust_interval_max", "gust_seed"]
TRIM_HINTS = {
    "steady_level_flight": "Solves α, elevator and throttle for level flight at the given airspeed and altitude.",
    "steady_climb": "Solves α, elevator and throttle for a steady climb (γ > 0) or descent (γ < 0).",
    "coordinated_turn": "Solves α, bank angle, elevator, aileron, rudder and throttle for a level turn of the given radius.",
    "glide": "Throttle = 0: solves α, elevator and pitch attitude for a steady glide (γ is a result).",
}


class SimulationTab(QtWidgets.QScrollArea):
    def __init__(self, case_dir: pathlib.Path) -> None:
        super().__init__()
        self.form = SchemaForm(SimCase, COLUMNS, case_dir)
        self.maneuvers = ManeuverTable()
        box = QtWidgets.QGroupBox("Scripted maneuvers")
        QtWidgets.QVBoxLayout(box).addWidget(self.maneuvers)
        self.form.bottom.addWidget(box)

        self.trim_hint = QtWidgets.QLabel()
        self.trim_hint.setWordWrap(True)
        self.trim_hint.setStyleSheet("color: gray")
        self.form.groups["trim"].layout().addRow(self.trim_hint)

        f = self.form
        f.enable_when("trim_enable", TRIM_FIELDS, {True})
        f.enable_when("trim_enable", INITIAL_CONDITION_FIELDS, {False})
        f.enable_when("trim_condition", ["trim_gamma"], {"steady_climb"})
        f.enable_when("trim_condition", ["trim_radius"], {"coordinated_turn"})
        f.enable_when("density_model", ["density"], {"constant"})
        f.enable_when("gust_enable", GUST_FIELDS, {True})
        f.widgets["trim_condition"].currentTextChanged.connect(self._update_hint)
        self.setWidget(self.form)
        self.setWidgetResizable(True)

    def _update_hint(self, condition: str) -> None:
        self.trim_hint.setText(TRIM_HINTS.get(condition, ""))

    def load(self, case: SimCase, case_dir: pathlib.Path) -> None:
        self.form.set_base_dir(case_dir)
        self.form.load(case)
        self.maneuvers.load(case.maneuvers)
        self._update_hint(case.trim_condition)

    def apply(self, case: SimCase) -> SimCase:
        return dataclasses.replace(self.form.apply(case), maneuvers=self.maneuvers.values())
