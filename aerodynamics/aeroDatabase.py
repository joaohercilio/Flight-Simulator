import os
import numpy as np
from scipy.interpolate import RegularGridInterpolator

class AeroDatabase:
    """
    Loads aerodynamic coefficient tables and provides fast interpolated access.
    """

    def __init__(self, data_dir=None):
        base_dir = os.path.dirname(__file__)
        self.data_dir = data_dir or os.path.join(base_dir, "data")

        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        self.tables = {}
        self._interpolators = {}

        self._load_tables()

    def _load_tables(self):
        for file in os.listdir(self.data_dir):
            if not file.endswith(".dat"):
                continue

            name = file[:-4]
            path = os.path.join(self.data_dir, file)
            data = np.loadtxt(path)

            if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] < 2:
                raise ValueError(f"Invalid table format: {file}")

            alpha_grid = np.radians(data[1:, 0])    
            beta_grid  = np.radians(data[0, 1:])       
            table      = data[1:, 1:]

            self.tables[name] = {
                "alpha": alpha_grid,
                "beta": beta_grid,
                "table": table
            }

            self._interpolators[name] = RegularGridInterpolator(
                (alpha_grid, beta_grid),
                table,
                method="linear",
                bounds_error=False,
                fill_value=None
            )

    def getCoeff(self, name, alpha, beta, fallback=0.0):
        """
        Returns interpolated coefficient value in degrees.
        """

        if name not in self._interpolators:
            return fallback

        entry = self.tables[name]

        alpha_deg = np.clip(
            alpha,
            entry["alpha"][0],
            entry["alpha"][-1]
        )
        beta_deg = np.clip(
            beta,
            entry["beta"][0],
            entry["beta"][-1]
        )

        return float(self._interpolators[name]([[alpha_deg, beta_deg]]))

