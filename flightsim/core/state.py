# flightsim/core/state.py
"""Named view over the 12-element state vector."""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


class StateIndex:
    """Canonical indices for the state vector.

    State layout (12 elements):
        [0]  x_e   - NED position x (m)
        [1]  y_e   - NED position y (m)
        [2]  z_e   - NED position z (m)  (negative = altitude)
        [3]  phi   - roll angle (rad)
        [4]  theta - pitch angle (rad)
        [5]  psi   - yaw angle (rad)
        [6]  u     - body-axis velocity x (m/s)
        [7]  v     - body-axis velocity y (m/s)
        [8]  w     - body-axis velocity z (m/s)
        [9]  p     - roll rate (rad/s)
        [10] q     - pitch rate (rad/s)
        [11] r     - yaw rate (rad/s)
    """
    X_E   = 0
    Y_E   = 1
    Z_E   = 2
    PHI   = 3
    THETA = 4
    PSI   = 5
    U     = 6
    V     = 7
    W     = 8
    P     = 9
    Q     = 10
    R     = 11

    SIZE  = 12


class StateVector:
    """Named read/write view over a 12-element NDArray.

    The underlying array is never copied — this is a zero-overhead
    wrapper intended for use at the boundary (I/O, plotting, state_eq).
    The integrator (rk4, euler) operates on the raw array directly.

    Example:
        >>> arr = np.zeros(12)
        >>> s = StateVector(arr)
        >>> s.u = 50.0      # writes directly into arr[6]
        >>> s.theta         # reads arr[4]
    """

    __slots__ = ("_x",)  # Prevents the creation of new attributes

    def __init__(self, array: NDArray) -> None:
        if array.shape != (StateIndex.SIZE,):
            raise ValueError(
                f"Expected shape ({StateIndex.SIZE},), got {array.shape}"
            )
        object.__setattr__(self, "_x", array)

    @property
    def x_e(self) -> float: return self._x[StateIndex.X_E]
    @x_e.setter
    def x_e(self, v: float) -> None: self._x[StateIndex.X_E] = v

    @property
    def y_e(self) -> float: return self._x[StateIndex.Y_E]
    @y_e.setter
    def y_e(self, v: float) -> None: self._x[StateIndex.Y_E] = v

    @property
    def z_e(self) -> float: return self._x[StateIndex.Z_E]
    @z_e.setter
    def z_e(self, v: float) -> None: self._x[StateIndex.Z_E] = v

    @property
    def phi(self) -> float: return self._x[StateIndex.PHI]
    @phi.setter
    def phi(self, v: float) -> None: self._x[StateIndex.PHI] = v

    @property
    def theta(self) -> float: return self._x[StateIndex.THETA]
    @theta.setter
    def theta(self, v: float) -> None: self._x[StateIndex.THETA] = v

    @property
    def psi(self) -> float: return self._x[StateIndex.PSI]
    @psi.setter
    def psi(self, v: float) -> None: self._x[StateIndex.PSI] = v

    @property
    def u(self) -> float: return self._x[StateIndex.U]
    @u.setter
    def u(self, v: float) -> None: self._x[StateIndex.U] = v

    @property
    def v(self) -> float: return self._x[StateIndex.V]
    @v.setter
    def v(self, v: float) -> None: self._x[StateIndex.V] = v

    @property
    def w(self) -> float: return self._x[StateIndex.W]
    @w.setter
    def w(self, v: float) -> None: self._x[StateIndex.W] = v

    @property
    def p(self) -> float: return self._x[StateIndex.P]
    @p.setter
    def p(self, v: float) -> None: self._x[StateIndex.P] = v

    @property
    def q(self) -> float: return self._x[StateIndex.Q]
    @q.setter
    def q(self, v: float) -> None: self._x[StateIndex.Q] = v

    @property
    def r(self) -> float: return self._x[StateIndex.R]
    @r.setter
    def r(self, v: float) -> None: self._x[StateIndex.R] = v

    @property
    def altitude(self) -> float:
        """Altitude (m), positive upward. Derived from z_e."""
        return -self._x[StateIndex.Z_E]

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> StateVector:
        """Builds a StateVector from a named dict (e.g. from TOML).

        Args:
            d: Dict with keys matching state variable names.

        Returns:
            New StateVector backed by a fresh array.

        Raises:
            KeyError: If any required key is missing from d.
        """
        arr = np.zeros(StateIndex.SIZE)
        s = cls(arr)
        s.x_e  = d["x_e"]
        s.y_e  = d["y_e"]
        s.z_e  = d["z_e"]
        s.phi  = d["phi"]
        s.theta = d["theta"]
        s.psi  = d["psi"]
        s.u    = d["u"]
        s.v    = d["v"]
        s.w    = d["w"]
        s.p    = d["p"]
        s.q    = d["q"]
        s.r    = d["r"]
        return s

    def to_array(self) -> NDArray:
        """Returns the underlying array (no copy)."""
        return self._x

    def __repr__(self) -> str:
        return (
            f"StateVector(pos=({self.x_e:.1f}, {self.y_e:.1f}, {self.z_e:.1f}), "
            f"att=({self.phi:.3f}, {self.theta:.3f}, {self.psi:.3f}), "
            f"vel=({self.u:.2f}, {self.v:.2f}, {self.w:.2f}), "
            f"rate=({self.p:.3f}, {self.q:.3f}, {self.r:.3f}))"
        )
