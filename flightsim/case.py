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

@dataclasses.dataclass(frozen=True)
class Case:
    """Initial conditions and settings for the simulation

    Attributes:
        case_dir: Path to the case directory.
        name: case name
        model: Aircraft model.
    """

    total_time:       float
    time_step:        float
    gravity_model:    str
    g:                float
    atmosphere_model: str
    density:          float
    enable_trim:      bool
    trim_name:        str
    target_speed:     float
    trim_alt:         float
    trim_gamma:       float
    trim_radius:      float
    u:                float
    v:                float
    w:                float
    x:                float
    y:                float
    height:           float
    p:                float
    q:                float
    r:                float
    name:             str = "case"

