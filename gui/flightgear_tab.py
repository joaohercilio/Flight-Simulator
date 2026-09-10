from __future__ import annotations

import pathlib
import sys
from typing import Callable

from PySide6 import QtCore, QtGui, QtWidgets

from flightsim.case import SimCase
from flightsim.flightgear.bridge import fgfs_command, join_command
from flightsim.session import Session
from gui.forms import SchemaForm
from gui.widgets import Console

ROOT = pathlib.Path(__file__).resolve().parents[1]
COLUMNS = [{"flightgear": "Connection and start"}, {"joystick": "Joystick mapping"}]
STEPS = ("1. Configure the connection, start mode and pilot input below, then save the case.\n"
         "2. Launch FlightGear (button or the command shown) and wait until the scenery is loaded.\n"
         "3. Start the bridge: the 6DOF model integrates in real time and drives the FlightGear aircraft.")


class JoystickMonitor(QtWidgets.QGroupBox):
    def __init__(self) -> None:
        super().__init__("Joystick monitor (move an axis to identify its index)")
        self.devices = QtWidgets.QComboBox()
        self.refresh = QtWidgets.QPushButton("Refresh")
        self.toggle = QtWidgets.QPushButton("Start monitor")
        self.toggle.setCheckable(True)
        self.bars_layout = QtWidgets.QFormLayout()
        self.bars: list[QtWidgets.QProgressBar] = []
        self._joystick = None
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._poll)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Device:"))
        row.addWidget(self.devices, 1)
        row.addWidget(self.refresh)
        row.addWidget(self.toggle)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(row)
        layout.addLayout(self.bars_layout)
        self.refresh.clicked.connect(self.refresh_devices)
        self.toggle.toggled.connect(self._toggled)

    def refresh_devices(self) -> None:
        from flightsim.control.devices import list_joysticks
        self.devices.clear()
        names = list_joysticks()
        self.devices.addItems([f"[{i}] {n}" for i, n in enumerate(names)] or ["No joystick detected"])
        self.devices.setEnabled(bool(names))

    def _toggled(self, on: bool) -> None:
        from flightsim.control.devices import AxisMap, Joystick
        self._clear()
        if on:
            try:
                self._joystick = Joystick(AxisMap(index=max(self.devices.currentIndex(), 0)))
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "Joystick", str(exc))
                self.toggle.setChecked(False)
                return
            for i in range(len(self._joystick.axes())):
                bar = QtWidgets.QProgressBar()
                bar.setRange(-100, 100)
                bar.setFormat("%v")
                self.bars.append(bar)
                self.bars_layout.addRow(f"Axis {i}:", bar)
            self._timer.start()
            self.toggle.setText("Stop monitor")
        else:
            self.toggle.setText("Start monitor")

    def _clear(self) -> None:
        self._timer.stop()
        if self._joystick is not None:
            self._joystick.close()
            self._joystick = None
        while self.bars_layout.rowCount():
            self.bars_layout.removeRow(0)
        self.bars.clear()

    def _poll(self) -> None:
        for bar, value in zip(self.bars, self._joystick.axes()):
            bar.setValue(int(100 * value))


class FlightGearTab(QtWidgets.QScrollArea):
    def __init__(self, case_dir: pathlib.Path, get_session: Callable[[bool], Session]) -> None:
        super().__init__()
        self.get_session = get_session
        self.case_dir = case_dir
        self.form = SchemaForm(SimCase, COLUMNS, case_dir)
        self.form.changed.connect(self.refresh_command)
        self.monitor = JoystickMonitor()
        self.form.widgets["fg_control"].currentTextChanged.connect(self._control_changed)
        self.form.enable_when("fg_start_mode", ["fg_heading"], {"trimmed", "ground"})

        self.command = QtWidgets.QPlainTextEdit()
        self.command.setReadOnly(True)
        self.command.setMaximumHeight(70)
        self.copy_button = QtWidgets.QPushButton("Copy command")
        self.launch_button = QtWidgets.QPushButton("Launch FlightGear")
        self.start_button = QtWidgets.QPushButton("▶  Start bridge")
        self.start_button.setStyleSheet("font-weight: bold")
        self.stop_button = QtWidgets.QPushButton("■  Stop bridge")
        self.stop_button.setEnabled(False)
        self.status = QtWidgets.QLabel("Bridge not running.")
        self.status.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        self.console = Console()
        self.console.setMinimumHeight(180)

        launch_box = QtWidgets.QGroupBox("Launch")
        launch = QtWidgets.QVBoxLayout(launch_box)
        launch.addWidget(QtWidgets.QLabel("FlightGear command (generated from the settings above):"))
        launch.addWidget(self.command)
        buttons = QtWidgets.QHBoxLayout()
        for b in (self.copy_button, self.launch_button, self.start_button, self.stop_button):
            buttons.addWidget(b)
        launch.addLayout(buttons)
        launch.addWidget(self.status)
        launch.addWidget(self.console)

        steps = QtWidgets.QLabel(STEPS)
        steps.setStyleSheet("color: gray")
        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)
        layout.addWidget(steps)
        layout.addWidget(self.form)
        layout.addWidget(self.monitor)
        layout.addWidget(launch_box)
        self.setWidget(content)
        self.setWidgetResizable(True)

        self.process = QtCore.QProcess(self)
        self.process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        env = QtCore.QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUTF8", "1")
        self.process.setProcessEnvironment(env)
        self.process.readyReadStandardOutput.connect(self._read)
        self.process.finished.connect(self._finished)
        self.copy_button.clicked.connect(lambda: QtWidgets.QApplication.clipboard().setText(self.command.toPlainText()))
        self.launch_button.clicked.connect(self.launch_flightgear)
        self.start_button.clicked.connect(self.start_bridge)
        self.stop_button.clicked.connect(self.stop_bridge)

    def _control_changed(self, kind: str) -> None:
        self.form.groups["joystick"].setEnabled(kind == "joystick")
        self.monitor.setEnabled(kind == "joystick")

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self.monitor.devices.count():
            self.monitor.refresh_devices()

    def load(self, case: SimCase, case_dir: pathlib.Path) -> None:
        self.case_dir = case_dir
        self.form.set_base_dir(case_dir)
        self.form.load(case)
        self._control_changed(case.fg_control)
        self.refresh_command()

    def apply(self, case: SimCase) -> SimCase:
        return self.form.apply(case)

    def refresh_command(self) -> None:
        try:
            self.command.setPlainText(join_command(fgfs_command(self.get_session(False).case)))
        except Exception as exc:
            self.command.setPlainText(str(exc))

    def launch_flightgear(self) -> None:
        cmd = fgfs_command(self.get_session(False).case)
        ok, _ = QtCore.QProcess.startDetached(cmd[0], cmd[1:])
        self.console.log(("Launched: " if ok else "Failed to launch: ") + join_command(cmd))

    def start_bridge(self) -> None:
        try:
            session = self.get_session(True)
        except Exception as exc:
            self.console.log(f"Error: {exc}")
            return
        self.console.clear()
        self.console.log(f"Case saved. Starting bridge for {session.case_dir} ...")
        self.process.setWorkingDirectory(str(ROOT))
        self.process.start(sys.executable, ["-u", "main.py", "flightgear", str(session.case_dir.resolve())])
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop_bridge(self) -> None:
        if self.process.state() != QtCore.QProcess.NotRunning:
            self.process.terminate()
            if not self.process.waitForFinished(2000):
                self.process.kill()

    def _read(self) -> None:
        for line in bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace").splitlines():
            if line.startswith("t ") and "|" in line:
                self.status.setText(line)
            elif line.strip():
                self.console.log(line)

    def _finished(self, code: int, _status) -> None:
        self.console.log(f"Bridge process finished (exit code {code}).")
        self.status.setText("Bridge not running.")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def closing(self) -> None:
        self.monitor.toggle.setChecked(False)
        self.stop_bridge()
