# config/settings.py
"""Simulation configuration dataclass."""

from __future__ import annotations

import dataclasses
import pathlib
import tomllib

from numpy.typing import NDArray

from flightsim.core.state import StateVector

from flightsim.atmosphere.model import AtmosphereModel, ConstantAtmosphere


@dataclasses.dataclass
class SimConfig:
    """Holds all simulation parameters.

    Attributes:
        model_file: Path to the aircraft model file.
        t_start: Simulation start time (s).
        t_end: Simulation end time (s).
        dt: Integration time step (s).
        x0: Initial state vector, shape (12,).
        save_figures: Whether to save plots to disk.
        output_dir: Directory for saved plots.
        show_gui: Whether to open plot windows interactively.
    """

    model_file: pathlib.Path
    t_start: float
    t_end: float
    dt: float
    x0: NDArray
    atmosphere: AtmosphereModel
    save_figures: bool
    plot_config: pathlib.Path
    output_dir: pathlib.Path
    show_gui: bool

    @classmethod
    def from_toml(cls, data: dict, config_dir: pathlib.Path) -> SimConfig:
        """Constructs SimConfig from a parsed TOML dict.

        Args:
            data: Dict loaded from tomllib.

        Returns:
            Populated SimConfig instance.

        Raises:
            KeyError: If any required key is missing from the TOML.
            ValueError: If the initial condition vector is malformed.
        """
        sim = data["simulation"]
        plots = data.get("plots", {})

        model_file = config_dir / data["model"]["file"]
        plot_config = config_dir / plots.get("config_file", "C:/Users/enzo_/Documents/GitHub/Flight-Simulator/cases/mushu/plots.toml")

        x0 = StateVector.from_dict(data["initial_condition"]).to_array()

        return cls(
            model_file=model_file,
            t_start=sim["t_start"],
            t_end=sim["t_end"],
            dt=sim["dt"],
            x0=x0,
            atmosphere=AtmosphereModel.from_dict(data.get("atmosphere", {})),
            plot_config=plot_config,
            save_figures=plots.get("save_figures", False),
            output_dir=config_dir / plots.get("output_dir", "results/"),
            show_gui=plots.get("show_gui", True),
        )

    @classmethod
    def from_toml_file(cls, path: pathlib.Path) -> SimConfig:
        """Loads and parses a TOML file, returning a SimConfig.

        Args:
            path: Path to the .toml config file.

        Returns:
            Populated SimConfig instance.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "rb") as f:
            data = tomllib.load(f)

        return cls.from_toml(data, config_dir=path.parent)
