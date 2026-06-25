# gui/newaircraft.py
"""New-aircraft editor (newaircraft.ui).
"""

from __future__ import annotations

import pathlib
from typing import Callable

from PySide6 import QtWidgets
from PySide6.QtCore import QFile
from PySide6.QtUiTools import QUiLoader

from flightsim.aircraft import AircraftModel
from utils.io import save_aircraft, load_aircraft, report_case

UI_FILE = pathlib.Path(__file__).resolve().parent / "newaircraft.ui"


class NewAircraftWindow:
    """Controller for newaircraft.ui."""

    def __init__(self, on_saved: Callable[[pathlib.Path], None] | None = None, model: AircraftModel = None ) -> None:
        """Builds the editor.

        Args:
            on_saved: Optional callback invoked with the saved file path
        """
        self._on_saved = on_saved

        loader = QUiLoader()
        f = QFile(str(UI_FILE))
        f.open(QFile.ReadOnly)
        self.win = loader.load(f)
        f.close()

        self._grab_widgets()
        self.browseButton.clicked.connect(self._on_browse)
        self.saveAircraftButton.clicked.connect(self._on_save)


    def _grab_widgets(self) -> None:
        w = self.win
        S, E, B = QtWidgets.QDoubleSpinBox, QtWidgets.QLineEdit, QtWidgets.QPushButton
        # name
        self.aircraftNameEdit = w.findChild(E, "aircraftNameEdit")
        # inertia
        self.massSpinBox = w.findChild(S, "massSpinBox")
        self.IxxSpinBox  = w.findChild(S, "IxxSpinBox")
        self.IyySpinBox  = w.findChild(S, "IyySpinBox")
        self.IzzSpinBox  = w.findChild(S, "IzzSpinBox")
        self.IxzSpinBox  = w.findChild(S, "IxzSpinBox")
        self.xcgSpinBox  = w.findChild(S, "xcgSpinBox")
        self.ycgSpinBox  = w.findChild(S, "ycgSpinBox")
        self.zcgSpinBox  = w.findChild(S, "zcgSpinBox")
        # geometry
        self.SrefSpinBox = w.findChild(S, "SrefSpinBox")
        self.bRefSpinBox = w.findChild(S, "bRefSpinBox")
        self.cRefSpinBox = w.findChild(S, "cRefSpinBox")
        self.groundEffectSpinBox = w.findChild(S, "groundEffectSpinBox")
        # control limits
        self.eleMinSpinBox = w.findChild(S, "eleMinSpinBox")
        self.eleMaxSpinBox = w.findChild(S, "eleMaxSpinBox")
        self.ailMinSpinBox = w.findChild(S, "ailMinSpinBox")
        self.ailMaxSpinBox = w.findChild(S, "ailMaxSpinBox")
        self.rudMinSpinBox = w.findChild(S, "rudMinSpinBox")
        self.rudMaxSpinBox = w.findChild(S, "rudMaxSpinBox")
        # propulsion
        self.propArmZSpinBox = w.findChild(S, "propArmZSpinBox")
        self.aSpinBox = w.findChild(S, "aSpinBox")
        self.bSpinBox = w.findChild(S, "bSpinBox")
        self.cSpinBox = w.findChild(S, "cSpinBox")
        self.dSpinBox = w.findChild(S, "dSpinBox")
        # landing gear
        self.mainGearXSpinBox = w.findChild(S, "mainGearXSpinBox")
        self.mainGearYSpinBox = w.findChild(S, "mainGearYSpinBox")
        self.noseGearSpinBox  = w.findChild(S, "noseGearSpinBox")
        self.gearHeightSpinBox = w.findChild(S, "gearHeightSpinBox")
        self.deflectionSpinBox = w.findChild(S, "deflectionSpinBox")
        self.dampingSpinBox    = w.findChild(S, "dampingSpinBox")
        # aero tables
        self.aeroTablesEdit = w.findChild(E, "aeroTablesEdit")
        self.browseButton   = w.findChild(B, "browseButton")
        # actions
        self.saveAircraftButton = w.findChild(B, "saveAircraftButton")


    def _on_browse(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self.win, "Select aero tables folder")
        if folder:
            self.aeroTablesEdit.setText(pathlib.Path(folder).as_posix())


    def _build_model(self) -> AircraftModel:
        """Reads the form and returns an AircraftModel
        """
        aero_dir = pathlib.Path(self.aeroTablesEdit.text() or "aero_tables")
        return AircraftModel(
            mass=self.massSpinBox.value(),
            ix=self.IxxSpinBox.value(),
            iy=self.IyySpinBox.value(),
            iz=self.IzzSpinBox.value(),
            ixz=self.IxzSpinBox.value(),
            x_cg=self.xcgSpinBox.value(),
            y_cg=self.ycgSpinBox.value(),
            z_cg=self.zcgSpinBox.value(),
            s=self.SrefSpinBox.value(),
            b=self.bRefSpinBox.value(),
            c=self.cRefSpinBox.value(),
            arm_z_engine=self.propArmZSpinBox.value(),
            elevator_max=self.eleMaxSpinBox.value(),
            aileron_max=self.ailMaxSpinBox.value(),
            rudder_max=self.rudMaxSpinBox.value(),
            aero_tables_dir=aero_dir,
            elevator_min=self.eleMinSpinBox.value(),
            aileron_min=self.ailMinSpinBox.value(),
            rudder_min=self.rudMinSpinBox.value(),
            thrust_a=self.aSpinBox.value(),
            thrust_b=self.bSpinBox.value(),
            thrust_c=self.cSpinBox.value(),
            thrust_d=self.dSpinBox.value(),
            main_gear_x=self.mainGearXSpinBox.value(),
            main_gear_y=self.mainGearYSpinBox.value(),
            wheelbase=self.noseGearSpinBox.value(),
            gear_height=self.gearHeightSpinBox.value(),
            design_deflection=self.deflectionSpinBox.value(),
            damping_ratio=self.dampingSpinBox.value(),
            ground_effect=self.groundEffectSpinBox.value(),
            name=self.aircraftNameEdit.text().strip() or "aircraft",
        )


    def _on_save(self) -> None:
        name = self.aircraftNameEdit.text().strip() or "aircraft"
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.win, "Save aircraft", f"{name}.model")
        if not fn:
            return
        path = pathlib.Path(fn)
        try:
            save_aircraft(path, self._build_model())
        except OSError as exc:
            QtWidgets.QMessageBox.warning(self.win, "Save failed", str(exc))
            return
        if self._on_saved is not None:
            self._on_saved(path, self._build_model())

        self.win.close()


    def fill_from_model(self, model: AircraftModel) -> None:
        """Fills the form with the model
        """
        self.massSpinBox.setValue(model.mass)
        self.IxxSpinBox.setValue(model.ix)
        self.IyySpinBox.setValue(model.iy)
        self.IzzSpinBox.setValue(model.iz)
        self.IxzSpinBox.setValue(model.ixz)
        self.xcgSpinBox.setValue(model.x_cg)
        self.ycgSpinBox.setValue(model.y_cg)
        self.zcgSpinBox.setValue(model.z_cg)
        self.SrefSpinBox.setValue(model.s)
        self.bRefSpinBox.setValue(model.b)
        self.cRefSpinBox.setValue(model.c)
        self.propArmZSpinBox.setValue(model.arm_z_engine)
        self.eleMaxSpinBox.setValue(model.elevator_max)
        self.ailMaxSpinBox.setValue(model.aileron_max)
        self.rudMaxSpinBox.setValue(model.rudder_max)
        self.aeroTablesEdit.setText(model.aero_tables_dir.as_posix())
        self.eleMinSpinBox.setValue(model.elevator_min)
        self.ailMinSpinBox.setValue(model.aileron_min)
        self.rudMinSpinBox.setValue(model.rudder_min)
        self.aSpinBox.setValue(model.thrust_a)
        self.bSpinBox.setValue(model.thrust_b)
        self.cSpinBox.setValue(model.thrust_c)
        self.dSpinBox.setValue(model.thrust_d)
        self.mainGearXSpinBox.setValue(model.main_gear_x)
        self.mainGearYSpinBox.setValue(model.main_gear_y)
        self.noseGearSpinBox.setValue(model.wheelbase)
        self.gearHeightSpinBox.setValue(model.gear_height)
        self.deflectionSpinBox.setValue(model.design_deflection)
        self.dampingSpinBox.setValue(model.damping_ratio)
        self.groundEffectSpinBox.setValue(model.ground_effect)
        self.aircraftNameEdit.setText(model.name)


    def show(self) -> None:
        self.win.show()
