#gui/new_case.py
"""" New case window editor (newcase.ui)
"""

from __future__ import annotations

import pathlib
from typing import Callable

from PySide6 import QtWidgets
from PySide6.QtCore import QFile
from PySide6.QtUiTools import QUiLoader

from flightsim.case import Case
from utils.io import save_case, load_case, report_case

UI_FILE = pathlib.Path(__file__).resolve().parent / "newcase.ui"


class NewCaseWindow:
    """ Controller for neweditor.ui """

    def __init__(self, on_saved: Callable[[pathlib.Path], None] | None = None,
    case: Case = None) -> None:
        """ Builds the editor

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
        self._on_trim_toggled(self.trimCheckBox.isChecked())

        self.saveCaseButton.clicked.connect(self._on_save)
        self.trimCheckBox.toggled.connect(self._on_trim_toggled)

    def _on_trim_toggled(self, checked: bool) -> None:
        self.trimComboBox.setEnabled(checked)
        self.targetVSpinBox.setEnabled(checked)
        self.trimAltSpinBox.setEnabled(checked)
        self.trimGammaSpinBox.setEnabled(checked)
        self.trimRadiusSpinBox.setEnabled(checked)


    def _grab_widgets(self) -> None:
        w = self.win
        S, E, B, C = QtWidgets.QDoubleSpinBox, QtWidgets.QLineEdit, QtWidgets.QPushButton, QtWidgets.QComboBox

        #buttons
        self.saveCaseButton = w.findChild(B, "saveCaseButton")
        # name
        self.caseNameEdit = w.findChild(E, "caseNameEdit")
        # time
        self.totalTimeSpinBox = w.findChild(S, "timeSpinBox")
        self.timeStepSpinBox = w.findChild(S, "timeStepSpinBox")
        # gravity
        self.gravityComboBox = w.findChild(C, "gravityComboBox")
        self.gravitySpinBox = w.findChild(S, "gravitySpinBox")
        # density
        self.densityComboBox = w.findChild(C, "densityComboBox")
        self.densitySpinBox = w.findChild(S, "densitySpinBox")
        # initial conditions
        self.uSpinBox = w.findChild(S, "uSpinBox")
        self.vSpinBox = w.findChild(S, "vSpinBox")
        self.wSpinBox = w.findChild(S, "wSpinBox")
        self.xSpinBox = w.findChild(S, "xSpinBox")
        self.ySpinBox = w.findChild(S, "ySpinBox")
        self.heightSpinBox = w.findChild(S, "hSpinBox")
        self.pSpinBox = w.findChild(S, "pSpinBox")
        self.qSpinBox = w.findChild(S, "qSpinBox")
        self.rSpinBox = w.findChild(S, "rSpinBox")
        self.phiSpinBox = w.findChild(S, "phiSpinBox")
        self.thetaSpinBox = w.findChild(S, "thetaSpinBox")
        self.psiSpinBox = w.findChild(S, "psiSpinBox")
        # trim
        self.trimGroupBox = w.findChild(QtWidgets.QGroupBox, "trimGroupBox")
        self.trimCheckBox = w.findChild(QtWidgets.QCheckBox, "trimCheckBox")
        self.trimComboBox = w.findChild(C, "trimComboBox")
        self.targetVSpinBox = w.findChild(S, "targetVSpinBox")
        self.trimAltSpinBox = w.findChild(S, "trimAltSpinBox")
        self.trimGammaSpinBox = w.findChild(S, "trimGammaSpinBox")
        self.trimRadiusSpinBox = w.findChild(S, "trimRadiusSpinBox")

    def _build_case(self) -> Case:

        return Case(
            name = self.caseNameEdit.text().strip(),
            total_time = self.totalTimeSpinBox.value(),
            time_step = self.timeStepSpinBox.value(),
            gravity_model = self.gravityComboBox.currentText(),
            gravity = self.gravitySpinBox.value(),
            density_model = self.densityComboBox.currentText(),
            density = self.densitySpinBox.value(),
            enable_trim = self.trimCheckBox.isChecked(),
            trim_name = self.trimComboBox.currentText(),
            target_speed = self.targetVSpinBox.value(),
            trim_alt = self.trimAltSpinBox.value(),
            trim_gamma = self.trimGammaSpinBox.value(),
            trim_radius = self.trimRadiusSpinBox.value(),
            u = self.uSpinBox.value(),
            v = self.vSpinBox.value(),
            w = self.wSpinBox.value(),
            x = self.xSpinBox.value(),
            y = self.ySpinBox.value(),
            height = self.heightSpinBox.value(),
            p = self.pSpinBox.value(),
            q = self.qSpinBox.value(),
            r = self.rSpinBox.value(),
            phi = self.phiSpinBox.value(),
            theta = self.thetaSpinBox.value(),
            psi = self.psiSpinBox.value()
        )


    def _on_save(self) -> None:
        name = self.caseNameEdit.text().strip() or "case"
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.win, "Save case", f"{name}.case")
        if not fn:
            return
        path = pathlib.Path(fn)
        try:
            save_case(path, self._build_case())
        except OSError as exc:
            QtWidgets.QMessageBox.warning(self.win, "Save failed", str(exc))
            return
        if self._on_saved is not None:
            self._on_saved(path, self._build_case())

        self.win.close()


    def fill_from_case(self, case: Case) -> None:
        """Fills the form with the case
        """
        self.caseNameEdit.setText(case.name),
        self.totalTimeSpinBox.setValue(case.total_time),
        self.timeStepSpinBox.setValue(case.time_step),
        self.gravityComboBox.setCurrentText(case.gravity_model),
        self.gravitySpinBox.setValue(case.gravity),
        self.densityComboBox.setCurrentText(case.density_model),
        self.densitySpinBox.setValue(case.density),
        self.trimCheckBox.setChecked(case.enable_trim),
        self.trimComboBox.setCurrentText(case.trim_name),
        self.targetVSpinBox.setValue(case.target_speed),
        self.trimAltSpinBox.setValue(case.trim_alt),
        self.trimGammaSpinBox.setValue(case.trim_gamma),
        self.trimRadiusSpinBox.setValue(case.trim_radius),
        self.uSpinBox.setValue(case.u),
        self.vSpinBox.setValue(case.v),
        self.wSpinBox.setValue(case.w),
        self.xSpinBox.setValue(case.x),
        self.ySpinBox.setValue(case.y),
        self.heightSpinBox.setValue(case.height),
        self.pSpinBox.setValue(case.p),
        self.qSpinBox.setValue(case.q),
        self.rSpinBox.setValue(case.r)
        self.phiSpinBox.setValue(case.phi)
        self.thetaSpinBox.setValue(case.theta)
        self.psiSpinBox.setValue(case.psi)

    def show(self) -> None:
        self.win.show()








