from __future__ import annotations

import dataclasses
import pathlib

from PySide6 import QtCore, QtWidgets

from flightsim.config import sections


class PathEdit(QtWidgets.QWidget):
    changed = QtCore.Signal()

    def __init__(self, base_dir: pathlib.Path | None = None, directory: bool = True) -> None:
        super().__init__()
        self.base_dir = base_dir
        self.directory = directory
        self.edit = QtWidgets.QLineEdit()
        button = QtWidgets.QToolButton()
        button.setText("…")
        button.clicked.connect(self._browse)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.edit)
        layout.addWidget(button)
        self.edit.editingFinished.connect(self.changed)

    def _browse(self) -> None:
        start = str(self.base_dir or pathlib.Path.cwd())
        if self.directory:
            chosen = QtWidgets.QFileDialog.getExistingDirectory(self, "Select directory", start)
        else:
            chosen, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select file", start)
        if not chosen:
            return
        path = pathlib.Path(chosen)
        if self.base_dir is not None:
            try:
                path = path.relative_to(self.base_dir.resolve())
            except ValueError:
                pass
        self.edit.setText(path.as_posix())
        self.changed.emit()

    def text(self) -> str:
        return self.edit.text()

    def setText(self, value: str) -> None:
        self.edit.setText(value)


class NoWheelFilter(QtCore.QObject):
    def eventFilter(self, obj, event) -> bool:
        return event.type() == QtCore.QEvent.Wheel and not obj.hasFocus()


def make_widget(field: dataclasses.Field, base_dir: pathlib.Path | None) -> QtWidgets.QWidget:
    meta = field.metadata
    default = field.default
    if isinstance(default, bool):
        w = QtWidgets.QCheckBox()
    elif isinstance(default, int):
        w = QtWidgets.QSpinBox()
        w.setRange(int(meta.get("min", 0)), int(meta.get("max", 1_000_000)))
        if meta.get("unit"):
            w.setSuffix(" " + meta["unit"])
    elif isinstance(default, float):
        w = QtWidgets.QDoubleSpinBox()
        w.setDecimals(meta.get("decimals", 2))
        w.setRange(meta.get("min", 0.0), meta.get("max", 1e6))
        w.setSingleStep(10 ** -min(meta.get("decimals", 2), 2))
        if meta.get("unit"):
            w.setSuffix(" " + meta["unit"])
    elif "choices" in meta:
        w = QtWidgets.QComboBox()
        w.addItems(meta["choices"])
    elif meta.get("path"):
        w = PathEdit(base_dir, directory=not str(default).endswith(".toml"))
    else:
        w = QtWidgets.QLineEdit()
    w.setToolTip(f"[{meta['section']}] {meta.get('key') or field.name}")
    if isinstance(w, (QtWidgets.QAbstractSpinBox, QtWidgets.QComboBox)):
        w.setFocusPolicy(QtCore.Qt.StrongFocus)
        w.installEventFilter(NoWheelFilter(w))
    return w


def get_value(w: QtWidgets.QWidget):
    if isinstance(w, QtWidgets.QCheckBox):
        return w.isChecked()
    if isinstance(w, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
        return w.value()
    if isinstance(w, QtWidgets.QComboBox):
        return w.currentText()
    return w.text()


def set_value(w: QtWidgets.QWidget, value) -> None:
    if isinstance(w, QtWidgets.QCheckBox):
        w.setChecked(bool(value))
    elif isinstance(w, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
        w.setValue(value)
    elif isinstance(w, QtWidgets.QComboBox):
        w.setCurrentText(str(value))
    else:
        w.setText(str(value))


class SchemaForm(QtWidgets.QWidget):
    changed = QtCore.Signal()

    def __init__(self, cls, columns: list[dict[str, str]], base_dir: pathlib.Path | None = None) -> None:
        super().__init__()
        self.cls = cls
        self.widgets: dict[str, QtWidgets.QWidget] = {}
        self.groups: dict[str, QtWidgets.QGroupBox] = {}
        self._rules: dict[str, list] = {}
        outer = QtWidgets.QVBoxLayout(self)
        row = QtWidgets.QHBoxLayout()
        outer.addLayout(row)
        self.bottom = QtWidgets.QVBoxLayout()
        outer.addLayout(self.bottom)
        outer.addStretch(1)
        for titles in columns:
            column = QtWidgets.QVBoxLayout()
            for section, title in titles.items():
                column.addWidget(self._group(section, title, base_dir))
            column.addStretch(1)
            row.addLayout(column, 1)

    def _group(self, section: str, title: str, base_dir: pathlib.Path | None) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox(title)
        form = QtWidgets.QFormLayout(box)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        for field in sections(self.cls).get(section, []):
            if field.default is dataclasses.MISSING:
                continue
            w = make_widget(field, base_dir)
            self.widgets[field.name] = w
            label = field.metadata.get("label") or field.name
            if isinstance(w, QtWidgets.QCheckBox):
                w.setText(label)
                form.addRow(w)
            else:
                form.addRow(label + ":", w)
            self._connect(w)
        self.groups[section] = box
        return box

    @staticmethod
    def signal(w: QtWidgets.QWidget):
        if isinstance(w, QtWidgets.QCheckBox):
            return w.toggled
        if isinstance(w, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
            return w.valueChanged
        if isinstance(w, QtWidgets.QComboBox):
            return w.currentTextChanged
        if isinstance(w, PathEdit):
            return w.changed
        return w.editingFinished

    def _connect(self, w: QtWidgets.QWidget) -> None:
        self.signal(w).connect(self.changed)

    def enable_when(self, source: str, fields: list[str], values) -> None:
        src = self.widgets[source]
        for name in fields:
            self._rules.setdefault(name, []).append(lambda: get_value(src) in values)
        self.signal(src).connect(self.apply_rules)

    def apply_rules(self) -> None:
        for name, predicates in self._rules.items():
            self.widgets[name].setEnabled(all(p() for p in predicates))

    def load(self, obj) -> None:
        self.blockSignals(True)
        for name, w in self.widgets.items():
            set_value(w, getattr(obj, name))
        self.blockSignals(False)
        self.apply_rules()

    def apply(self, obj):
        return dataclasses.replace(obj, **{name: get_value(w) for name, w in self.widgets.items()})

    def set_base_dir(self, base_dir: pathlib.Path) -> None:
        for w in self.widgets.values():
            if isinstance(w, PathEdit):
                w.base_dir = base_dir
