# flightsim/aero/database.py
"""Aerodynamic coefficient database with interpolation."""

from __future__ import annotations

import pathlib

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator


class AeroDatabase:
    """Loads 2D aero coefficient tables and provides interpolated access.

    Each table is a .dat file where:
        - Row 0, Col 1..N : beta grid values (degrees)
        - Row 1..M, Col 0 : alpha grid values (degrees)
        - Row 1..M, Col 1..N : coefficient values

    Args:
        tables_dir: Path to the directory containing .dat files.

    Raises:
        FileNotFoundError: If tables_dir does not exist.
        ValueError: If any table has an invalid format.
    """

    def __init__(self, tables_dir: pathlib.Path) -> None:
        if not tables_dir.is_dir():
            raise FileNotFoundError(f"Aero tables directory not found: {tables_dir}")

        self._tables: dict[str, dict] = {}
        self._interpolators: dict[str, RegularGridInterpolator] = {}
        self._load_tables(tables_dir)

    def _load_tables(self, tables_dir: pathlib.Path) -> None:
        """Loads all .dat files from tables_dir.

        Args:
            tables_dir: Directory to scan for .dat files.
        """
        for path in sorted(tables_dir.glob("*.dat")):
            name = path.stem
            data = np.loadtxt(path)

            if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] < 2:
                raise ValueError(f"Invalid table format in file: {path.name}")

            alpha_grid = np.radians(data[1:, 0])
            beta_grid  = np.radians(data[0, 1:])
            table      = data[1:, 1:]

            self._tables[name] = {
                "alpha": alpha_grid,
                "beta": beta_grid,
                "table": table,
            }
            self._interpolators[name] = RegularGridInterpolator(
                (alpha_grid, beta_grid),
                table,
                method="linear",
                bounds_error=False,
                fill_value=None,
            )

    def get_coeff(self, name: str, alpha: float, beta: float, fallback: float = 0.0) -> float:
        """Returns an interpolated aerodynamic coefficient.

        Args:
            name: Coefficient name matching the .dat filename (e.g. "CL0").
            alpha: Angle of attack (rad).
            beta: Sideslip angle (rad).
            fallback: Value returned if the table is not present.

        Returns:
            Interpolated coefficient value.
        """
        if name not in self._interpolators:
            return fallback

        entry = self._tables[name]
        alpha_clipped = np.clip(alpha, entry["alpha"][0], entry["alpha"][-1])
        beta_clipped  = np.clip(beta,  entry["beta"][0],  entry["beta"][-1])

        return float(self._interpolators[name]([[alpha_clipped, beta_clipped]]))
