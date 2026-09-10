from __future__ import annotations

import abc
import contextlib
import dataclasses

import numpy as np
from numpy.typing import NDArray


class DensityModel(abc.ABC):
    @abc.abstractmethod
    def get_density(self, altitude_m: float) -> float: ...

    @staticmethod
    def build(name: str, density: float) -> DensityModel:
        if name == "constant":
            return ConstantDensity(density)
        if name == "isa":
            return ISADensity()
        raise ValueError(f"Unknown density model '{name}' (available: constant, isa)")


class ConstantDensity(DensityModel):
    def __init__(self, density: float) -> None:
        self._density = density

    def get_density(self, altitude_m: float) -> float:
        return self._density


class ISADensity(DensityModel):
    def get_density(self, altitude_m: float) -> float:
        h = min(max(altitude_m, -1000.0), 11000.0)
        return 1.225 * (1.0 - 2.25577e-5 * h) ** 4.2559


class Wind:
    def __init__(self, steady_ned: tuple[float, float, float] = (0.0, 0.0, 0.0), gust_amplitude: float = 0.0,
                 duration: tuple[float, float] = (1.0, 5.0), interval: tuple[float, float] = (2.0, 4.0),
                 seed: int | None = None) -> None:
        self.steady = np.array(steady_ned, dtype=float)
        self.gust_amplitude = gust_amplitude
        self.duration = duration
        self.interval = interval
        self._rng = np.random.default_rng(seed)
        self._gust = np.zeros(3)
        self._start = 0.0
        self._length = 1.0
        self._next = self._rng.uniform(*interval) if gust_amplitude > 0 else np.inf

    def update(self, t: float) -> None:
        if t < self._next:
            return
        amp = self._rng.uniform(0.0, self.gust_amplitude)
        az = self._rng.uniform(0.0, 2 * np.pi)
        el = self._rng.uniform(-np.pi / 2, np.pi / 2)
        self._gust = amp * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        self._length = self._rng.uniform(*self.duration)
        self._start = t
        self._next = t + self._rng.uniform(*self.interval)

    def ned(self, t: float) -> NDArray:
        tau = t - self._start
        env = np.sin(np.pi * tau / self._length) ** 2 if 0.0 <= tau <= self._length else 0.0
        return self.steady + env * self._gust


@dataclasses.dataclass
class Environment:
    density: DensityModel = dataclasses.field(default_factory=lambda: ConstantDensity(1.225))
    gravity: float = 9.81
    wind: Wind = dataclasses.field(default_factory=Wind)
    ground_elevation: float = 0.0
    ground_contact: bool = True

    @property
    def ground_z(self) -> float:
        return -self.ground_elevation

    @contextlib.contextmanager
    def still_air(self):
        wind, self.wind = self.wind, Wind()
        try:
            yield
        finally:
            self.wind = wind

    @classmethod
    def from_case(cls, case) -> Environment:
        gust = case.gust_amplitude if case.gust_enable else 0.0
        wind = Wind((case.wind_north, case.wind_east, case.wind_down), gust,
                    (case.gust_duration_min, case.gust_duration_max),
                    (case.gust_interval_min, case.gust_interval_max),
                    case.gust_seed or None)
        return cls(DensityModel.build(case.density_model, case.density), case.gravity, wind,
                   case.ground_elevation, case.ground_contact)
