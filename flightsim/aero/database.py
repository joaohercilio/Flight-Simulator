from __future__ import annotations

import pathlib

import numpy as np
from numpy.typing import NDArray


class AeroDatabase:
    def __init__(self, tables_dir: pathlib.Path) -> None:
        tables_dir = pathlib.Path(tables_dir)
        if not tables_dir.is_dir():
            raise FileNotFoundError(f"Aero tables directory not found: {tables_dir}")
        self.tables_dir = tables_dir
        self._grids: list[tuple[NDArray, NDArray, NDArray, list[str]]] = []
        self._load(tables_dir)
        self.names = sorted(n for g in self._grids for n in g[3])
        if not self.names:
            raise ValueError(f"No .dat tables found in {tables_dir}")

    def _load(self, tables_dir: pathlib.Path) -> None:
        groups: dict[tuple, list[tuple[str, NDArray]]] = {}
        for path in sorted(tables_dir.glob("*.dat")):
            data = np.loadtxt(path)
            if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] < 2:
                raise ValueError(f"Invalid table format: {path.name}")
            key = (tuple(data[1:, 0]), tuple(data[0, 1:]))
            groups.setdefault(key, []).append((path.stem, data[1:, 1:]))
        for (alpha, beta), tables in groups.items():
            stack = np.stack([t for _, t in tables], axis=-1)
            self._grids.append((np.radians(alpha), np.radians(beta), stack, [n for n, _ in tables]))

    @staticmethod
    def _bilinear(alpha_grid: NDArray, beta_grid: NDArray, stack: NDArray, alpha: float, beta: float) -> NDArray:
        a = min(max(alpha, alpha_grid[0]), alpha_grid[-1])
        b = min(max(beta, beta_grid[0]), beta_grid[-1])
        i = min(int(np.searchsorted(alpha_grid, a, side="right")) - 1, len(alpha_grid) - 2)
        j = min(int(np.searchsorted(beta_grid, b, side="right")) - 1, len(beta_grid) - 2)
        i, j = max(i, 0), max(j, 0)
        ta = (a - alpha_grid[i]) / (alpha_grid[i + 1] - alpha_grid[i])
        tb = (b - beta_grid[j]) / (beta_grid[j + 1] - beta_grid[j])
        return ((1 - ta) * (1 - tb) * stack[i, j] + ta * (1 - tb) * stack[i + 1, j]
                + (1 - ta) * tb * stack[i, j + 1] + ta * tb * stack[i + 1, j + 1])

    def coefficients(self, alpha: float, beta: float) -> dict[str, float]:
        out: dict[str, float] = {}
        for alpha_grid, beta_grid, stack, names in self._grids:
            values = self._bilinear(alpha_grid, beta_grid, stack, alpha, beta)
            out.update(zip(names, values.tolist()))
        return out

    def get_coeff(self, name: str, alpha: float, beta: float, fallback: float = 0.0) -> float:
        return self.coefficients(alpha, beta).get(name, fallback)

    @property
    def alpha_range(self) -> tuple[float, float]:
        lo = min(g[0][0] for g in self._grids)
        hi = max(g[0][-1] for g in self._grids)
        return float(np.degrees(lo)), float(np.degrees(hi))
