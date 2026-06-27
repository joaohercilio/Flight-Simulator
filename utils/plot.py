# utils/plot.py
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6 import QtWidgets
import numpy as np
from flightsim.core.state import StateIndex


CHANNELS = {
    "x [m]":     StateIndex.X_E,
    "y [m]":     StateIndex.Y_E,
    "height [m]":     StateIndex.Z_E,
    "phi [deg]": StateIndex.PHI,
    "theta [deg]": StateIndex.THETA,
    "psi [deg]": StateIndex.PSI,
    "u [m/s]":   StateIndex.U,
    "v [m/s]":   StateIndex.V,
    "w [m/s]":   StateIndex.W,
    "p [deg/s]": StateIndex.P,
    "q [deg/s]": StateIndex.Q,
    "r [deg/s]": StateIndex.R,
}


def embed_canvas(widget: QtWidgets.QWidget) -> FigureCanvasQTAgg:
    fig = Figure(tight_layout=True)
    canvas = FigureCanvasQTAgg(fig)
    layout = QtWidgets.QVBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(canvas)
    return canvas


def plot_channel(canvas: FigureCanvasQTAgg, t, x, channel_name: str) -> None:
    idx = CHANNELS[channel_name]
    canvas.figure.clear()
    ax = canvas.figure.add_subplot(111)
    ax.plot(t, x[idx, :])
    ax.set_xlabel("t [s]")
    ax.set_ylabel(channel_name)
    ax.grid(True)
    canvas.draw()
