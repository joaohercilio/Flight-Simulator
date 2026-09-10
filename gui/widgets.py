from __future__ import annotations

import traceback
from typing import Callable

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6 import QtCore, QtGui, QtWidgets

from flightsim.control.source import SURFACES
from gui.forms import NoWheelFilter


class Console(QtWidgets.QPlainTextEdit):
    def __init__(self) -> None:
        super().__init__()
        self.setReadOnly(True)
        self.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        self.setMaximumBlockCount(5000)

    @QtCore.Slot(str)
    def log(self, text: str) -> None:
        self.appendPlainText(text)
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())


class PlotCanvas(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(NavigationToolbar2QT(self.canvas, self))
        layout.addWidget(self.canvas)

    def draw(self) -> None:
        self.canvas.draw_idle()


class ManeuverTable(QtWidgets.QWidget):
    COLUMNS = ("surface", "start", "end", "deflection")

    def __init__(self) -> None:
        super().__init__()
        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Surface", "Start [s]", "End [s]", "Deflection [deg / -]"])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        add = QtWidgets.QPushButton("Add")
        remove = QtWidgets.QPushButton("Remove")
        add.clicked.connect(lambda: self.add_row())
        remove.clicked.connect(self._remove)
        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(add)
        buttons.addWidget(remove)
        buttons.addStretch()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QtWidgets.QLabel("Deflections added to the baseline/trim command inside each time window:"))
        layout.addWidget(self.table)
        layout.addLayout(buttons)

    def add_row(self, m: dict | None = None) -> None:
        m = m or {"surface": "elevator", "start": 5.0, "end": 6.0, "deflection": 0.0}
        row = self.table.rowCount()
        self.table.insertRow(row)
        combo = QtWidgets.QComboBox()
        combo.addItems(SURFACES)
        combo.setCurrentText(m["surface"])
        self._no_wheel(combo)
        self.table.setCellWidget(row, 0, combo)
        for col, key in enumerate(self.COLUMNS[1:], start=1):
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(-1e4, 1e4)
            spin.setDecimals(3)
            spin.setValue(float(m[key]))
            self._no_wheel(spin)
            self.table.setCellWidget(row, col, spin)

    @staticmethod
    def _no_wheel(w: QtWidgets.QWidget) -> None:
        w.setFocusPolicy(QtCore.Qt.StrongFocus)
        w.installEventFilter(NoWheelFilter(w))

    def _remove(self) -> None:
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True) or ([self.table.rowCount() - 1] if self.table.rowCount() else [])
        for row in rows:
            self.table.removeRow(row)

    def load(self, maneuvers: list[dict]) -> None:
        self.table.setRowCount(0)
        for m in maneuvers:
            self.add_row(m)

    def values(self) -> list[dict]:
        out = []
        for row in range(self.table.rowCount()):
            out.append({"surface": self.table.cellWidget(row, 0).currentText(),
                        **{key: self.table.cellWidget(row, col).value() for col, key in enumerate(self.COLUMNS[1:], start=1)}})
        return out


class Task(QtCore.QThread):
    log = QtCore.Signal(str)
    progress = QtCore.Signal(float)
    finished_ok = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, func: Callable, parent=None) -> None:
        super().__init__(parent)
        self.func = func

    def run(self) -> None:
        try:
            self.finished_ok.emit(self.func(self.log.emit, self.progress.emit))
        except Exception as exc:
            self.failed.emit(f"{exc}\n{traceback.format_exc()}")
