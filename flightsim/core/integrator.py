from __future__ import annotations

from typing import Callable

from numpy.typing import NDArray


def rk4_step(f: Callable[[NDArray, float], NDArray], x: NDArray, dx: NDArray, t: float, dt: float) -> None:
    k1 = f(x, t)
    k2 = f(x + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = f(x + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = f(x + dt * k3, t + dt)
    x += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    dx[:] = k1
