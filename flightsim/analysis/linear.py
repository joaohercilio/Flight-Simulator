from __future__ import annotations

import dataclasses

import numpy as np
from numpy.typing import NDArray

from flightsim.control.source import ControlInput, LiveControl
from flightsim.core.dynamics import Dynamics
from flightsim.core.state import StateIndex as I

LONGITUDINAL = [I.U, I.W, I.Q, I.THETA]
LATERAL = [I.V, I.P, I.R, I.PHI]
CONTROLS = ["elevator", "aileron", "rudder", "throttle"]


@dataclasses.dataclass(frozen=True)
class LinearModel:
    A: NDArray
    B: NDArray
    x0: NDArray
    u0: ControlInput

    def submatrix(self, states: list[int]) -> NDArray:
        return self.A[np.ix_(states, states)]


def linearize(dynamics: Dynamics, x0: NDArray, u0: ControlInput, eps: float = 1e-6) -> LinearModel:
    live = LiveControl(u0)
    previous, dynamics.controls = dynamics.controls, live

    def f(x, u):
        live.set(u)
        return dynamics(x, 0.0)

    try:
        with dynamics.env.still_air():
            n = len(x0)
            A = np.zeros((n, n))
            for j in range(n):
                d = np.zeros(n)
                d[j] = eps * max(1.0, abs(x0[j]))
                A[:, j] = (f(x0 + d, u0) - f(x0 - d, u0)) / (2 * d[j])
            B = np.zeros((n, len(CONTROLS)))
            for j, name in enumerate(CONTROLS):
                h = 1e-4
                up = dataclasses.replace(u0, **{name: getattr(u0, name) + h})
                dn = dataclasses.replace(u0, **{name: getattr(u0, name) - h})
                B[:, j] = (f(x0, up) - f(x0, dn)) / (2 * h)
    finally:
        dynamics.controls = previous
    return LinearModel(A, B, x0, u0)


def describe_modes(A: NDArray, labels: list[str]) -> str:
    lines = []
    seen = set()
    for lam, vec in zip(*np.linalg.eig(A)):
        key = (round(lam.real, 6), round(abs(lam.imag), 6))
        if key in seen:
            continue
        seen.add(key)
        dominant = labels[int(np.argmax(np.abs(vec)))]
        if abs(lam.imag) > 1e-9:
            wn = abs(lam)
            zeta = -lam.real / wn
            period = 2 * np.pi / abs(lam.imag)
            lines.append(f"  λ = {lam.real:+.4f} ± {abs(lam.imag):.4f}j   ωn {wn:.3f} rad/s   ζ {zeta:+.3f}   "
                         f"T {period:.2f} s   (dominant: {dominant})")
        else:
            tau = np.inf if abs(lam.real) < 1e-12 else 1.0 / abs(lam.real)
            kind = "τ" if lam.real < 0 else "t₂ (unstable)"
            value = tau * (1.0 if lam.real < 0 else np.log(2))
            lines.append(f"  λ = {lam.real:+.4f}            {kind} {value:.2f} s   (dominant: {dominant})")
    return "\n".join(lines)


def modes_report(model: LinearModel) -> str:
    names = list(I.NAMES)
    return ("Longitudinal modes (u, w, q, theta):\n" + describe_modes(model.submatrix(LONGITUDINAL), [names[i] for i in LONGITUDINAL])
            + "\n\nLateral-directional modes (v, p, r, phi):\n" + describe_modes(model.submatrix(LATERAL), [names[i] for i in LATERAL]))
