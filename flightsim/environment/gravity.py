# flightsim/environment/gravity.py
"""Gravity models."""
from __future__ import annotations
import abc

class GravityModel(abc.ABC):
    """Abstract base class for gravity models."""

    @staticmethod
    def build(model_name: str, gravity: float) -> GravityModel:
        if model_name == "Constant gravity":
            return ConstantGravity(gravity=gravity)

    @abc.abstractmethod
    def get_gravity(self, altitude_m: float) -> float: ...


class ConstantGravity(GravityModel):

    def __init__(self, gravity: float) -> None:
        self._gravity = gravity

    def get_gravity(self, altitude_m: float) -> float:
        return self._gravity

