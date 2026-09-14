from __future__ import annotations

import pathlib
from typing import Callable

from PySide6 import QtCore, QtWidgets

from flightsim.analysis.channels import GROUP_NAMES, build_groups
from flightsim.analysis.export import export_csv
from flightsim.analysis.linear import linearize, modes_report
from flightsim.analysis.performance import ceiling_sweep
from flightsim.analysis.plots import draw_figure
from flightsim.control.source import ScriptedControl
from flightsim.core.simulation import SimulationResult
from flightsim.environment import Environment, ISADensity
from flightsim.session import Session
from gui.widgets import Console, PlotCanvas, Task


class AnalysisTab(QtWidgets.QWidget):
    busy_changed = QtCore.Signal(bool)

    def __init__(self, get_session: Callable[[], Session]) -> None:
        super().__init__()
        self.get_session = get_session
        self.result: SimulationResult | None = None
        self._groups = None
        self._drawn: set[str] = set()
        self._task: Task | None = None

        self.run_button = QtWidgets.QPushButton("▶  Run simulation")
        self.run_button.setStyleSheet("font-weight: bold; padding: 8px")
        self.run_button.setShortcut("Ctrl+R")
        self.run_button.setToolTip("Ctrl+R")
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setTextVisible(False)
        run_box = QtWidgets.QGroupBox("Simulation")
        run_layout = QtWidgets.QVBoxLayout(run_box)
        run_layout.addWidget(self.run_button)
        run_layout.addWidget(self.progress)

        self.trim_button = QtWidgets.QPushButton("Solve trim")
        self.modes_button = QtWidgets.QPushButton("Linear modes (eigenvalues)")
        self.ceiling_button = QtWidgets.QPushButton("Ceiling sweep (ISA)")
        self.ceiling_throttle = QtWidgets.QDoubleSpinBox()
        self.ceiling_throttle.setRange(0.05, 1.0)
        self.ceiling_throttle.setSingleStep(0.05)
        self.ceiling_throttle.setValue(1.0)
        self.ceiling_hmax = QtWidgets.QDoubleSpinBox()
        self.ceiling_hmax.setRange(100, 30000)
        self.ceiling_hmax.setValue(6000)
        self.ceiling_hmax.setSuffix(" m")
        self.ceiling_speed = QtWidgets.QDoubleSpinBox()
        self.ceiling_speed.setRange(0, 1000)
        self.ceiling_speed.setSuffix(" m/s")
        self.ceiling_speed.setSpecialValueText("trim airspeed")
        analysis_box = QtWidgets.QGroupBox("Analyses (use the current trim / initial condition)")
        form = QtWidgets.QFormLayout(analysis_box)
        form.addRow(self.trim_button)
        form.addRow(self.modes_button)
        form.addRow("Airspeed:", self.ceiling_speed)
        form.addRow("Throttle:", self.ceiling_throttle)
        form.addRow("Max altitude:", self.ceiling_hmax)
        form.addRow(self.ceiling_button)

        self.csv_button = QtWidgets.QPushButton("Export CSV (full history)…")
        self.window_button = QtWidgets.QPushButton("Export CSV (time window)…")
        self.t0 = QtWidgets.QDoubleSpinBox()
        self.t1 = QtWidgets.QDoubleSpinBox()
        for spin, value in ((self.t0, 5.0), (self.t1, 6.0)):
            spin.setRange(0.0, 1e5)
            spin.setSuffix(" s")
            spin.setValue(value)
        self.figures_button = QtWidgets.QPushButton("Save all figures (PNG)")
        export_box = QtWidgets.QGroupBox("Export")
        export_form = QtWidgets.QFormLayout(export_box)
        export_form.addRow(self.csv_button)
        window = QtWidgets.QHBoxLayout()
        window.addWidget(self.t0)
        window.addWidget(QtWidgets.QLabel("→"))
        window.addWidget(self.t1)
        export_form.addRow("Window:", window)
        export_form.addRow(self.window_button)
        export_form.addRow(self.figures_button)

        self.console = Console()
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(run_box)
        left_layout.addWidget(analysis_box)
        left_layout.addWidget(export_box)
        left_layout.addWidget(QtWidgets.QLabel("Output"))
        left_layout.addWidget(self.console, 1)

        self.plots = QtWidgets.QTabWidget()
        self.canvases: dict[str, PlotCanvas] = {}
        for name in GROUP_NAMES:
            canvas = PlotCanvas()
            self.canvases[name] = canvas
            self.plots.addTab(canvas, name)
        self.plots.currentChanged.connect(self._draw_current)

        splitter = QtWidgets.QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(self.plots)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([380, 900])
        QtWidgets.QHBoxLayout(self).addWidget(splitter)

        self.run_button.clicked.connect(self.run_simulation)
        self.trim_button.clicked.connect(self.solve_trim)
        self.modes_button.clicked.connect(self.linear_modes)
        self.ceiling_button.clicked.connect(self.ceiling)
        self.csv_button.clicked.connect(lambda: self.export(False))
        self.window_button.clicked.connect(lambda: self.export(True))
        self.figures_button.clicked.connect(self.save_figures)
        self._set_result(None)

    def _start(self, func) -> None:
        if self._task is not None and self._task.isRunning():
            return
        try:
            session = self.get_session()
        except Exception as exc:
            self.console.log(f"Error: {exc}")
            return
        self.console.log("─" * 60)
        self.progress.setValue(0)
        self._busy(True)
        self._task = Task(lambda log, progress: func(session, log, progress), self)
        self._task.log.connect(self.console.log)
        self._task.progress.connect(lambda p: self.progress.setValue(int(100 * p)))
        self._task.finished_ok.connect(self._finished)
        self._task.failed.connect(self._failed)
        self._task.start()

    def is_busy(self) -> bool:
        return self._task is not None and self._task.isRunning()

    def _busy(self, busy: bool) -> None:
        for b in (self.run_button, self.trim_button, self.modes_button, self.ceiling_button):
            b.setEnabled(not busy)
        self.busy_changed.emit(busy)

    def _finished(self, value) -> None:
        self._busy(False)
        if isinstance(value, SimulationResult):
            self._set_result(value)

    def _failed(self, message: str) -> None:
        self._busy(False)
        self.console.log("Error: " + message)

    def _set_result(self, result: SimulationResult | None) -> None:
        self.result = result
        self._groups = build_groups(result) if result is not None else None
        self._drawn.clear()
        for b in (self.csv_button, self.window_button, self.figures_button):
            b.setEnabled(result is not None)
        if result is not None:
            self.t1.setValue(min(self.t1.value(), float(result.t[-1])))
            self._draw_current()

    def _draw_current(self) -> None:
        name = self.plots.tabText(self.plots.currentIndex())
        if self.result is None or name in self._drawn:
            return
        canvas = self.canvases[name]
        draw_figure(canvas.figure, self.result, [name], self._groups)
        canvas.draw()
        self._drawn.add(name)

    def run_simulation(self) -> None:
        self._start(lambda s, log, progress: s.run(log, progress))

    def solve_trim(self) -> None:
        self._start(lambda s, log, _: log(s.trim().summary()))

    def linear_modes(self) -> None:
        def task(s: Session, log, _):
            dyn = s.dynamics(ScriptedControl(s.case.baseline_controls()))
            x0, u0 = s.initial_conditions(log, dyn)
            log(modes_report(linearize(dyn, x0, u0)))
        self._start(task)

    def ceiling(self) -> None:
        throttle, h_max, speed = self.ceiling_throttle.value(), self.ceiling_hmax.value(), self.ceiling_speed.value()

        def task(s: Session, log, _):
            env = Environment.from_case(s.case)
            env.density = ISADensity()
            dyn = s.dynamics(ScriptedControl(s.case.baseline_controls()), env)
            V = speed or s.case.trim_airspeed
            log(f"Ceiling sweep: V = {V} m/s, throttle = {throttle:.0%}, ISA density")
            log(ceiling_sweep(dyn, V, throttle, h_max, log=log).summary())
        self._start(task)

    def export(self, windowed: bool) -> None:
        if self.result is None:
            return
        suggested = str(self.get_session().output_dir / ("loads_window.csv" if windowed else "history.csv"))
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export CSV", suggested, "CSV (*.csv)")
        if not path:
            return
        t0, t1 = (self.t0.value(), self.t1.value()) if windowed else (None, None)
        n = export_csv(self.result, pathlib.Path(path), t0, t1)
        self.console.log(f"Exported {n} samples to {path}")

    def save_figures(self) -> None:
        if self.result is None:
            return
        out = self.get_session().output_dir
        out.mkdir(parents=True, exist_ok=True)
        for i, name in enumerate(GROUP_NAMES):
            canvas = self.canvases[name]
            draw_figure(canvas.figure, self.result, [name], self._groups)
            canvas.figure.savefig(out / f"fig_{i + 1:02d}_{name.lower().replace(' ', '_')}.png", dpi=150, bbox_inches="tight")
            self._drawn.add(name)
        self.console.log(f"Saved {len(GROUP_NAMES)} figures to {out}")
