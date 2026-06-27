# flightsim/environment/density.py
"""Air density models."""
from __future__ import annotations
import abc

class DensityModel(abc.ABC):
    """Abstract base class for air density models."""

    @staticmethod
    def build(model_name: str, density: float) -> DensityModel:
        if model_name == "Constant density":
            return ConstantDensity(density=density)
        if model_name == "ISA Standart":
            return ISADensity()

    @abc.abstractmethod
    def get_density(self, altitude_m: float) -> float: ...


class ConstantDensity(DensityModel):

    def __init__(self, density: float) -> None:
        self._density = density

    def get_density(self, altitude_m: float) -> float:
        return self._density


class ISADensity(DensityModel):

    def get_density(self, altitude_m: float) -> float:
        H = 3.28084 * altitude_m  # meters to feet
        return ((1 - 6.875e-6 * H) ** 5.2561) * 1.225
