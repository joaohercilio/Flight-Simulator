from PySide6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QCheckBox, QGroupBox
import subprocess
import sys
import pathlib
import re


CONFIG_PATH = pathlib.Path("cases/mushu/flightgear_config.toml")

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Flight Simulator Launcher")

        layout = QVBoxLayout()

        
        group_box = QGroupBox("FlightGear Initialization")
        group_layout = QVBoxLayout()

        self.cb_in_air = QCheckBox("Start in Air")
        self.cb_trimmed = QCheckBox("Start Trimmed")
        self.cb_manual = QCheckBox("Manual Control")

        
        self.cb_in_air.toggled.connect(self.enforce_logic)
        self.cb_manual.toggled.connect(self.enforce_logic)

        group_layout.addWidget(self.cb_in_air)
        group_layout.addWidget(self.cb_trimmed)
        group_layout.addWidget(self.cb_manual)
        group_box.setLayout(group_layout)

        
        run_sim = QPushButton("Run Simulation")
        run_sim.clicked.connect(self.run_simulation)
        
        run_flightgear = QPushButton("Run FlightGear")
        run_flightgear.clicked.connect(self.run_flightgear)

        # Build Main Layout
        layout.addWidget(group_box)
        layout.addWidget(run_sim)
        layout.addWidget(run_flightgear)
        self.setLayout(layout)

        
        self.load_config()
        self.enforce_logic()

    def enforce_logic(self):
        """Disables and unchecks Start Trimmed if conditions aren't met."""
        if not self.cb_in_air.isChecked() or self.cb_manual.isChecked():
            self.cb_trimmed.setChecked(False)
            self.cb_trimmed.setEnabled(False)
        else:
            self.cb_trimmed.setEnabled(True)

    def load_config(self):
        """Reads the current config file to set the checkboxes."""
        if not CONFIG_PATH.exists():
            print(f"Warning: Config not found at {CONFIG_PATH}")
            return
        
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            content = f.read()

        if re.search(r'start_in_air\s*=\s*true', content, re.IGNORECASE):
            self.cb_in_air.setChecked(True)
        if re.search(r'start_trimmed\s*=\s*true', content, re.IGNORECASE):
            self.cb_trimmed.setChecked(True)
        if re.search(r'manual_control\s*=\s*true', content, re.IGNORECASE):
            self.cb_manual.setChecked(True)

    def save_config(self):
        """Updates the boolean toggles in the TOML without overwriting hidden keys."""
        if not CONFIG_PATH.exists():
            return

        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            content = f.read()

        def update_toml_bool(key, is_checked):
            val_str = "true" if is_checked else "false"
            
            pattern = rf"({key}\s*=\s*)(true|false)"
            return re.sub(pattern, rf"\g<1>{val_str}", content, flags=re.IGNORECASE)

        content = update_toml_bool("start_in_air", self.cb_in_air.isChecked())
        content = update_toml_bool("start_trimmed", self.cb_trimmed.isChecked())
        content = update_toml_bool("manual_control", self.cb_manual.isChecked())

        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            f.write(content)

    def run_simulation(self):
        subprocess.Popen([sys.executable, "main.py"])

    def run_flightgear(self):
        
        self.save_config()
        subprocess.Popen([sys.executable, "flightgear.py"])


app = QApplication([])
window = MainWindow()
window.show()
app.exec()