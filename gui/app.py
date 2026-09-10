from __future__ import annotations

import pathlib
import shutil
import sys

from PySide6 import QtWidgets

from flightsim.aircraft import AircraftModel
from flightsim.case import SimCase
from flightsim.session import CASE_FILE, Session
from gui.aircraft_tab import AircraftTab
from gui.analysis_tab import AnalysisTab
from gui.flightgear_tab import FlightGearTab
from gui.simulation_tab import SimulationTab

ROOT = pathlib.Path(__file__).resolve().parents[1]
CASES_DIR = ROOT / "cases"


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, case_dir: pathlib.Path) -> None:
        super().__init__()
        self.setWindowTitle("Flight Simulator")
        self.resize(1400, 900)
        self.session = Session(case_dir)

        self.case_combo = QtWidgets.QComboBox()
        self.case_combo.setMinimumWidth(220)
        open_button = QtWidgets.QPushButton("Open folder…")
        new_button = QtWidgets.QPushButton("New case…")
        self.save_button = QtWidgets.QPushButton("Save")
        self.save_button.setShortcut("Ctrl+S")
        self.path_label = QtWidgets.QLabel()
        self.path_label.setStyleSheet("color: gray")
        header = QtWidgets.QHBoxLayout()
        header.addWidget(QtWidgets.QLabel("<b>Case:</b>"))
        header.addWidget(self.case_combo)
        header.addWidget(open_button)
        header.addWidget(new_button)
        header.addWidget(self.save_button)
        header.addWidget(self.path_label, 1)

        self.aircraft_tab = AircraftTab(case_dir)
        self.simulation_tab = SimulationTab(case_dir)
        self.analysis_tab = AnalysisTab(lambda: self.get_session(False))
        self.flightgear_tab = FlightGearTab(case_dir, self.get_session)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self.aircraft_tab, "Aircraft")
        self.tabs.addTab(self.simulation_tab, "Simulation")
        self.tabs.addTab(self.analysis_tab, "Analysis")
        self.tabs.addTab(self.flightgear_tab, "FlightGear")

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        layout.addLayout(header)
        layout.addWidget(self.tabs)
        self.setCentralWidget(central)

        open_button.clicked.connect(self.open_folder)
        new_button.clicked.connect(self.new_case)
        self.save_button.clicked.connect(lambda: self.get_session(True))
        self.case_combo.activated.connect(self._combo_selected)
        self.refresh_cases()
        self.load_forms()

    def refresh_cases(self) -> None:
        self.case_combo.blockSignals(True)
        self.case_combo.clear()
        dirs = sorted(p for p in CASES_DIR.glob("*") if (p / CASE_FILE).exists()) if CASES_DIR.exists() else []
        for d in dirs:
            self.case_combo.addItem(d.name, str(d))
        current = str(self.session.case_dir.resolve())
        idx = next((i for i in range(self.case_combo.count()) if pathlib.Path(self.case_combo.itemData(i)).resolve() == pathlib.Path(current)), -1)
        if idx < 0:
            self.case_combo.addItem(self.session.case_dir.name, str(self.session.case_dir))
            idx = self.case_combo.count() - 1
        self.case_combo.setCurrentIndex(idx)
        self.case_combo.blockSignals(False)

    def load_forms(self) -> None:
        d = self.session.case_dir
        self.aircraft_tab.load(self.session.aircraft, d)
        self.simulation_tab.load(self.session.case, d)
        self.flightgear_tab.load(self.session.case, d)
        self.path_label.setText(str(d.resolve()))
        self.setWindowTitle(f"Flight Simulator — {self.session.case.name}")

    def get_session(self, save: bool = False) -> Session:
        s = self.session
        s.aircraft = self.aircraft_tab.apply(s.aircraft)
        s.case = self.flightgear_tab.apply(self.simulation_tab.apply(s.case))
        if save:
            s.case.save(s.case_dir / CASE_FILE)
            s.aircraft.save(s.case.aircraft_path(s.case_dir))
            self.statusBar().showMessage(f"Saved {s.case_dir / CASE_FILE} and {s.case.aircraft}", 4000)
            self.setWindowTitle(f"Flight Simulator — {s.case.name}")
        return s

    def load_case(self, case_dir: pathlib.Path) -> None:
        try:
            self.session = Session(case_dir)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Load case", str(exc))
            return
        self.refresh_cases()
        self.load_forms()

    def _combo_selected(self, index: int) -> None:
        self.load_case(pathlib.Path(self.case_combo.itemData(index)))

    def open_folder(self) -> None:
        chosen = QtWidgets.QFileDialog.getExistingDirectory(self, "Open case folder", str(CASES_DIR))
        if chosen:
            self.load_case(pathlib.Path(chosen))

    def new_case(self) -> None:
        name, ok = QtWidgets.QInputDialog.getText(self, "New case", "Name (a copy of the current case is created in cases/):")
        if not ok or not name.strip():
            return
        target = CASES_DIR / name.strip()
        if target.exists():
            QtWidgets.QMessageBox.warning(self, "New case", f"{target} already exists.")
            return
        s = self.get_session(False)
        shutil.copytree(s.case_dir, target, ignore=shutil.ignore_patterns("results", "__pycache__"))
        case = SimCase(**{**s.case.__dict__, "name": name.strip()})
        case.save(target / CASE_FILE)
        AircraftModel(**s.aircraft.__dict__).save(case.aircraft_path(target))
        self.load_case(target)

    def closeEvent(self, event) -> None:
        self.flightgear_tab.closing()
        super().closeEvent(event)


def run(case_dir: pathlib.Path) -> None:
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow(pathlib.Path(case_dir))
    window.show()
    sys.exit(app.exec())
