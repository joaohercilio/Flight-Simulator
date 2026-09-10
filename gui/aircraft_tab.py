from __future__ import annotations

import pathlib

from PySide6 import QtWidgets

from flightsim.aero.database import AeroDatabase
from flightsim.aircraft import AircraftModel
from gui.forms import SchemaForm

TITLES = {
    "aircraft": "Identification",
    "inertia": "Mass and inertia",
    "geometry": "Reference geometry",
    "control_limits": "Control surface limits",
    "propulsion": "Propulsion  (T = throttle · (aV³ + bV² + cV + d) · ρ/ρ₀)",
    "landing_gear": "Landing gear (tricycle)",
    "aero": "Aerodynamic tables",
}


class AircraftTab(QtWidgets.QScrollArea):
    def __init__(self, case_dir: pathlib.Path) -> None:
        super().__init__()
        self.case_dir = case_dir
        self.form = SchemaForm(AircraftModel, TITLES, case_dir)
        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        self.form.groups["aero"].layout().addRow(self.status)
        self.form.changed.connect(self.refresh_tables)
        self.setWidget(self.form)
        self.setWidgetResizable(True)

    def load(self, aircraft: AircraftModel, case_dir: pathlib.Path) -> None:
        self.case_dir = case_dir
        self.form.set_base_dir(case_dir)
        self.form.load(aircraft)
        self.refresh_tables()

    def apply(self, aircraft: AircraftModel) -> AircraftModel:
        return self.form.apply(aircraft)

    def refresh_tables(self) -> None:
        path = self.case_dir / self.form.widgets["tables_dir"].text()
        try:
            db = AeroDatabase(path)
            lo, hi = db.alpha_range
            self.status.setText(f"{len(db.names)} tables loaded (α {lo:.0f}° … {hi:.0f}°): " + ", ".join(db.names))
            self.status.setStyleSheet("color: gray")
        except Exception as exc:
            self.status.setText(str(exc))
            self.status.setStyleSheet("color: #c62828")
