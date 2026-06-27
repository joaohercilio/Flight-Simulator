# flightsim/aircraft.py
"""Aircraft domain model.
"""

from __future__ import annotations

import dataclasses
import pathlib


@dataclasses.dataclass(frozen=True)
class AircraftModel:
    """Everything that defines an aircraft for the simulation.

    Attributes:
        mass: Total mass (kg).
        ix, iy, iz: Moments of inertia about x/y/z (kg m^2).
        ixz: Product of inertia xz (kg m^2).
        x_cg, y_cg, z_cg: Centre-of-gravity position (m).
        s, b, c: Wing reference area (m^2), span (m), mean chord (m).
        arm_z_engine: Vertical distance from CG to thrust line (m).
        elevator_max, aileron_max, rudder_max: Control deflection limits (deg).
        elevator_min, aileron_min, rudder_min: Lower deflection limits (deg).
        aero_tables_dir: Folder with the .dat aero coefficient tables.
        ground_altitude: Ground elevation (m). Scenario data; default 0.
        thrust_a, thrust_b, thrust_c, thrust_d: Coefficients of
            T = av^3 + bv^2 + cv + d.
        main_gear_x: Main-gear x station (POS_MG) (m).
        main_gear_y: Main-gear half-track (Y_MG) (m).
        wheelbase: Nose-to-main wheelbase (m).
        gear_height: Gear height below CG (Z_GEAR) (m).
        design_deflection: Static strut deflection used to size stiffness (m).
        damping_ratio: Gear damping ratio (–).
        ground_effect: Ground-effect onset distance (m).
        name: Aircraft name.
    """

    mass:            float
    ix:              float
    iy:              float
    iz:              float
    ixz:             float
    x_cg:            float
    y_cg:            float
    z_cg:            float
    s:               float
    b:               float
    c:               float
    arm_z_engine:    float
    elevator_max:    float
    aileron_max:     float
    rudder_max:      float
    aero_tables_dir: pathlib.Path
    ground_altitude: float = 0
    # --- new: were hard-coded in Dynamics ---
    elevator_min:      float = -25.0
    aileron_min:       float = -20.0
    rudder_min:        float = -30.0
    thrust_a:          float = 0.001274
    thrust_b:          float = -0.07204
    thrust_c:          float = -0.5428
    thrust_d:          float = 40.89
    main_gear_x:       float = 0.44     # POS_MG
    main_gear_y:       float = 0.2      # Y_MG
    wheelbase:         float = 0.306    # WHEELBASE
    gear_height:       float = 0.15     # Z_GEAR
    design_deflection: float = 0.03
    damping_ratio:     float = 1.0
    ground_effect:     float = 0.0
    name:              str = "aircraft"
