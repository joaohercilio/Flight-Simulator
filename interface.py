from PySide6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout
import subprocess
import sys


class MainWindow(QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Flight Simulator Launcher")

        layout = QVBoxLayout()

        run_sim = QPushButton("Run Simulation")
        run_sim.clicked.connect(self.run_simulation)
        run_flightgear = QPushButton("Run FlightGear")
        run_flightgear.clicked.connect(self.run_flightgear)


        layout.addWidget(run_sim)
        layout.addWidget(run_flightgear)

        self.setLayout(layout)

    def run_simulation(self):
        subprocess.Popen(
            [sys.executable, "main.py"]
        )

    def run_flightgear(self):
        subprocess.Popen(
            [sys.executable, "flightgear.py"]
        )

app = QApplication([])

window = MainWindow()
window.show()

app.exec()