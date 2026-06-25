# gui/main_window.py

from __future__ import annotations

import pathlib

from PySide6 import QtWidgets
from PySide6.QtCore import QFile
from PySide6.QtUiTools import QUiLoader

from gui.new_aircraft import NewAircraftWindow
from gui.new_case import NewCaseWindow

from flightsim.aircraft import AircraftModel
from flightsim.case import Case

from utils.io import load_aircraft, load_case, report_model, report_case

UI_FILE = pathlib.Path(__file__).resolve().parent / "mainwindow.ui"


def load_ui(path: pathlib.Path) -> QtWidgets.QWidget:
    """Loads a .ui file at runtime and returns the top-level widget."""
    loader = QUiLoader()
    f = QFile(str(path))
    f.open(QFile.ReadOnly)
    widget = loader.load(f)
    f.close()
    return widget


class MainWindow:


    def __init__(self) -> None:
        self.win = load_ui(UI_FILE)

        self.loadAircraftButton = self.win.findChild(
            QtWidgets.QPushButton, "loadAircraftButton")
        self.newAircraftButton = self.win.findChild(
            QtWidgets.QPushButton, "newAircraftButton")
        self.editAircraftButton = self.win.findChild(
            QtWidgets.QPushButton, "editAircraftButton")
        self.loadedAircraftLabel = self.win.findChild(
            QtWidgets.QLabel, "loadedAircraftLabel")

        self.loadCaseButton = self.win.findChild(
            QtWidgets.QPushButton, "loadCaseButton")
        self.newCaseButton = self.win.findChild(
            QtWidgets.QPushButton, "newCaseButton")
        self.editCaseButton = self.win.findChild(
            QtWidgets.QPushButton, "editCaseButton")
        self.loadedCaseLabel = self.win.findChild(
            QtWidgets.QLabel, "loadedCaseLabel")

        self.terminalPlainTextEdit = self.win.findChild(
            QtWidgets.QPlainTextEdit, "terminalPlainTextEdit")

        self.runSimulationButton = self.win.findChild(
            QtWidgets.QPushButton, "runSimulationButton")
        self.exportResultsButton = self.win.findChild(
            QtWidgets.QPushButton, "exportResultsButton")

        # Keep a reference to the editor window so it is not garbage-collected
        # while open.
        self._editor: NewAircraftWindow | None = None

        self.aircraft_path: pathlib.Path | None = None
        self.aircraft_model: AircraftModel | None = None

        self.case_path: pathlib.Path | None = None
        self.case: Case | None = None

        self.loadAircraftButton.clicked.connect(self._on_load_aircraft)
        self.newAircraftButton.clicked.connect(self._on_new_aircraft)
        self.editAircraftButton.clicked.connect(self._on_edit_aircraft)

        self.loadCaseButton.clicked.connect(self._on_load_case)
        self.newCaseButton.clicked.connect(self._on_new_case)
        self.editCaseButton.clicked.connect(self._on_edit_case)

        self.runSimulationButton.clicked.connect(self._on_run_simulation)

# ============== SIMULATION ========================

    def _on_run_simulation(self) -> None:


# ============== MODEL =============================

    def _on_load_aircraft(self) -> None:
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.win, "Load aircraft", "")
        if not fn:
            return
        model = load_aircraft(pathlib.Path(fn))
        self._set_aircraft(pathlib.Path(fn), model)

    def _on_new_aircraft(self) -> None:
        self._editor = NewAircraftWindow(on_saved=self._set_aircraft)
        self._editor.show()

    def _on_edit_aircraft(self) -> None:
        if(self.aircraft_model == None):
            self.log("No aircraft model loaded")
        else:
            self._editor = NewAircraftWindow(on_saved=self._set_aircraft)
            self._editor.fill_from_model(self.aircraft_model)
            self._editor.show()

    def _set_aircraft(self, path: pathlib.Path | str, model: AircraftModel) -> None:
        self.aircraft_path = pathlib.Path(path)
        self.aircraft_model = model
        self.loadedAircraftLabel.setText(self.aircraft_path.name)
        self.log("==============================")
        self.log(f"Loaded aircraft: {model.name}")
        self.log(report_model(model))

# ============== CASE =============================

    def _on_load_case(self) -> None:
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.win, "Load case", "")
        if not fn:
            return
        case = load_case(pathlib.Path(fn))
        self._set_case(pathlib.Path(fn), case)

    def _on_new_case(self) -> None:
        self._editor = NewCaseWindow(on_saved=self._set_case)
        self._editor.show()

    def _on_edit_case(self) -> None:
        if(self.case == None):
            self.log("No case loaded")
        else:
            self._editor = NewCaseWindow(on_saved=self._set_case)
            self._editor.fill_from_case(self.case)
            self._editor.show()

    def _set_case(self, path: pathlib.Path | str, case: Case) -> None:
        self.case_path = pathlib.Path(path)
        self.case = case
        self.loadedCaseLabel.setText(self.case_path.name)
        self.log("==============================")
        self.log(f"Loaded case: {case.name}")
        self.log(report_case(case))

    def log(self, msg: str) -> None:
        self.terminalPlainTextEdit.appendPlainText(msg)
        self.terminalPlainTextEdit.verticalScrollBar().setValue(
            self.terminalPlainTextEdit.verticalScrollBar().maximum()
        )
    def show(self) -> None:
        self.win.show()


def main() -> None:
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
