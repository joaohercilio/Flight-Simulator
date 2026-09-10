from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class StateIndex:
    X_E, Y_E, Z_E, PHI, THETA, PSI, U, V, W, P, Q, R = range(12)
    SIZE = 12
    NAMES = ("x_e", "y_e", "z_e", "phi", "theta", "psi", "u", "v", "w", "p", "q", "r")


class StateVector:
    __slots__ = ("_x",)

    def __init__(self, array: NDArray) -> None:
        if array.shape != (StateIndex.SIZE,):
            raise ValueError(f"Expected shape ({StateIndex.SIZE},), got {array.shape}")
        object.__setattr__(self, "_x", array)

    @property
    def altitude(self) -> float:
        return -self._x[StateIndex.Z_E]

    @property
    def airspeed(self) -> float:
        return float(np.sqrt(self.u**2 + self.v**2 + self.w**2))

    def to_array(self) -> NDArray:
        return self._x

    def __repr__(self) -> str:
        return "StateVector(" + ", ".join(f"{n}={v:.4g}" for n, v in zip(StateIndex.NAMES, self._x)) + ")"


def _accessor(index: int):
    return property(lambda self: self._x[index], lambda self, value: self._x.__setitem__(index, value))


for _i, _name in enumerate(StateIndex.NAMES):
    setattr(StateVector, _name, _accessor(_i))
