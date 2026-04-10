# flightsim/core/integrator.py
"""Numerical integration methods for ODEs of the form dx/dt = f(x, t)."""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray


def rk4(
    f: Callable[[NDArray, float], NDArray],
    x: NDArray,
    dx: NDArray,
    t: NDArray,
    dt: float,
) -> None:
    """Integrates dx/dt = f(x, t) over all time steps using RK4.

    Fills x and dx in-place. x[:, 0] must be set by the caller
    as the initial condition before calling this function.

    Args:
        f: RHS of the ODE. Callable with signature f(x, t) -> dx,
            where x and dx are NDArray of shape (n,).
        x: State storage array, shape (n, N). Modified in-place.
        dx: Derivative storage array, shape (n, N). Modified in-place.
        t: Time vector, shape (N,).
        dt: Time step (s).
    """
    for i in range(1, len(t)):
        xi = x[:, i - 1]    # state at current time step
        ti = t[i - 1]       # current time

        f1 = f(xi, ti)
        k1 = dt * f1

        f2 = f(xi + 0.5 * k1, ti + 0.5 * dt)
        k2 = dt * f2

        f3 = f(xi + 0.5 * k2, ti + 0.5 * dt)
        k3 = dt * f3

        f4 = f(xi + k3, ti + dt)
        k4 = dt * f4

        # --- weighted average (Simpson's rule weights: 1/6, 2/6, 2/6, 1/6) ---
        x[:, i]  = xi + (k1 + 2*k2 + 2*k3 + k4) / 6
        dx[:, i] = f1   # store derivative at the start of the step


def rk4_step(
    f: Callable[[NDArray, float], NDArray],
    x: NDArray,
    dx: NDArray,
    dt: float,
) -> None:
    """Advances the state by one RK4 step in-place.

    Intended for real-time simulation where the caller manages
    the time loop externally.

    Args:
        f: RHS of the ODE. Callable with signature f(x, t) -> dx.
        x: Current state, shape (n,). Modified in-place.
        dx: Current derivative, shape (n,). Modified in-place.
        dt: Time step (s).
    """
    f1 = f(x, 0.0)
    k1 = dt * f1

    f2 = f(x + 0.5 * k1, 0.0)
    k2 = dt * f2

    f3 = f(x + 0.5 * k2, 0.0)
    k3 = dt * f3

    f4 = f(x + k3, 0.0)
    k4 = dt * f4

    x[:]  = x + (k1 + 2*k2 + 2*k3 + k4) / 6
    dx[:] = f1
