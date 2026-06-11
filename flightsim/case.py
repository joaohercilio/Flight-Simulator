# flightsim/case.py
"""A simulation case: one folder holding everything for a run.

A case directory contains:
    sim_config.toml      simulation / atmosphere / trim parameters
    aircraft_model.toml  inertia, geometry, control limits, aero path
    plots.toml           which figures to draw
    aero_tables/         the *.dat coefficient tables

``Case.load`` is the single entry point for "which aircraft / which run".
Nothing else in the codebase should hard-code a path under cases/.
"""

from __future__ import annotations

import dataclasses
import pathlib

from config.settings import SimConfig
from flightsim.aero.database import AeroDatabase
from flightsim.aircraft import AircraftModel
from utils.io import load_model

CONFIG_NAME = "sim_config.toml"
MODEL_NAME = "aircraft_model.toml"


@dataclasses.dataclass(frozen=True)
class Case:
    """A loaded case: resolved config, model and aero database.

    Attributes:
        case_dir: Path to the case directory.
        config: Parsed simulation configuration.
        model: Aircraft model.
        aero_db: Aerodynamic database (None if loaded with load_aero=False).
    """

    case_dir: pathlib.Path
    config: SimConfig
    model: AircraftModel
    aero_db: AeroDatabase | None

    @classmethod
    def load(cls, case_dir: pathlib.Path | str, load_aero: bool = True) -> Case:
        """Loads a case from its directory.

        The aircraft (aircraft_model.toml + aero tables) is required. The run
        configuration (sim_config.toml) is *optional*: if present it is loaded
        as a starting preset, otherwise defaults are used. The GUI edits the
        config from there — the file no longer needs to exist.

        Args:
            case_dir: Path to the case directory.
            load_aero: Whether to load the aero tables (slow). Set False when
                you only need config/model (e.g. inspecting parameters).

        Returns:
            Populated Case instance.

        Raises:
            NotADirectoryError: If case_dir is not a directory.
            FileNotFoundError: If the aircraft model file is missing.
        """
        case_dir = pathlib.Path(case_dir)
        if not case_dir.is_dir():
            raise NotADirectoryError(f"Case directory not found: {case_dir}")

        model = load_model(case_dir / MODEL_NAME)
        aero_db = AeroDatabase(model.aero_tables_dir) if load_aero else None

        config_path = case_dir / CONFIG_NAME
        if config_path.exists():
            config = SimConfig.from_toml_file(config_path)
        else:
            config = SimConfig()
        if config.plot_config is None:
            config.plot_config = case_dir / "plots.toml"

        return cls(case_dir=case_dir, config=config, model=model, aero_db=aero_db)
