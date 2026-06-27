# flightsim/case.py
"""A simulation case: one folder holding everything for a run.

"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
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
    gravity:          float
    density_model:    str
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
    phi:              float
    theta:            float
    psi:              float
    name:             str = "case"

