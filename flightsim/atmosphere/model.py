# flightsim/atmosphere/model.py
"""Atmosphere and gravity models."""

from __future__ import annotations

import abc


class AtmosphereModel(abc.ABC):
    """Abstract base class for atmosphere and gravity models.

    Subclasses must implement get_density and get_gravity.
    The integrator depends only on this interface — swapping
    models requires no changes outside this module.
    """

    @classmethod
    def from_dict(cls, data: dict) -> AtmosphereModel:
        """Builds an AtmosphereModel from a config dict.

        Args:
            data: Parsed [atmosphere] section. Empty dict uses defaults.

        Returns:
            Configured AtmosphereModel instance.

        Raises:
            ValueError: If the model name is not recognised.
        """
        model_name = data.get("model", "constant")

        if model_name == "constant":
            return ConstantAtmosphere(
                density=data.get("density", ConstantAtmosphere._DEFAULT_DENSITY),
                gravity=data.get("gravity", ConstantAtmosphere._DEFAULT_GRAVITY),
            )

        raise ValueError(
            f"Unknown atmosphere model: '{model_name}'. "
            f"Available: 'constant'."
        )

    @abc.abstractmethod
    def get_density(self, altitude_m: float) -> float:
        """Returns air density at the given altitude.

        Args:
            altitude_m: Altitude above sea level (m), positive upward.

        Returns:
            Air density (kg/m³).
        """

    @abc.abstractmethod
    def get_gravity(self, altitude_m: float) -> float:
        """Returns gravitational acceleration at the given altitude.

        Args:
            altitude_m: Altitude above sea level (m), positive upward.

        Returns:
            Gravitational acceleration (m/s²), positive downward.
        """


class ConstantAtmosphere(AtmosphereModel):
    """Atmosphere model with constant density and gravity.

    Suitable for low-altitude, short-duration simulations where
    variations with altitude are negligible.

    Args:
        density: Air density (kg/m³). Defaults to sea-level ISA value.
        gravity: Gravitational acceleration (m/s²). Defaults to standard gravity.

    Example:
        >>> atm = ConstantAtmosphere(density=1.225, gravity=9.81)
        >>> atm.get_density(1000.0)
        1.225
    """

    _DEFAULT_DENSITY = 1.225   # kg/m³  — ISA sea level
    _DEFAULT_GRAVITY = 9.81    # m/s²   — standard gravity

    def __init__(
        self,
        density: float = _DEFAULT_DENSITY,
        gravity: float = _DEFAULT_GRAVITY,
    ) -> None:
        if density <= 0:
            raise ValueError(f"Density must be positive, got {density}")
        if gravity <= 0:
            raise ValueError(f"Gravity must be positive, got {gravity}")

        self._density = density
        self._gravity = gravity

    def get_density(self, altitude_m: float) -> float:
        return self._density

    def get_gravity(self, altitude_m: float) -> float:
        return self._gravity

    def __repr__(self) -> str:
        return (
            f"ConstantAtmosphere("
            f"density={self._density} kg/m³, "
            f"gravity={self._gravity} m/s²)"
        )
