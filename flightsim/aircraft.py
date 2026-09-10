from __future__ import annotations

import dataclasses
import pathlib

from flightsim.config import load_toml, save_toml, spec


@dataclasses.dataclass
class AircraftModel:
    name: str = spec("aircraft", "aircraft", "Name")
    mass: float = spec("inertia", 10.0, "Mass", "kg", decimals=3)
    ix: float = spec("inertia", 0.2, "Ix", "kg·m²", key="Ix", decimals=6)
    iy: float = spec("inertia", 0.4, "Iy", "kg·m²", key="Iy", decimals=6)
    iz: float = spec("inertia", 0.5, "Iz", "kg·m²", key="Iz", decimals=6)
    ixz: float = spec("inertia", 0.0, "Ixz", "kg·m²", key="Ixz", decimals=6, min=-1e6)
    x_cg: float = spec("inertia", 0.0, "CG x (from nose)", "m", decimals=4)
    y_cg: float = spec("inertia", 0.0, "CG y", "m", decimals=4, min=-1e3)
    z_cg: float = spec("inertia", 0.0, "CG height above ground (ground start)", "m", decimals=4)
    s: float = spec("geometry", 1.0, "Wing area S", "m²", key="S", decimals=4)
    b: float = spec("geometry", 2.0, "Wingspan b", "m", decimals=4)
    c: float = spec("geometry", 0.5, "Mean chord c", "m", decimals=4)
    elevator_max: float = spec("control_limits", 25.0, "Elevator limit", "deg")
    aileron_max: float = spec("control_limits", 20.0, "Aileron limit", "deg")
    rudder_max: float = spec("control_limits", 30.0, "Rudder limit", "deg")
    arm_z_engine: float = spec("propulsion", 0.0, "Thrust line z-offset", "m", decimals=4, min=-10.0)
    thrust_a: float = spec("propulsion", 0.001274, "a (V³)", "N·s³/m³", key="a", decimals=6, min=-1e3)
    thrust_b: float = spec("propulsion", -0.07204, "b (V²)", "N·s²/m²", key="b", decimals=6, min=-1e3)
    thrust_c: float = spec("propulsion", -0.5428, "c (V)", "N·s/m", key="c", decimals=6, min=-1e3)
    thrust_d: float = spec("propulsion", 40.89, "d (static)", "N", key="d", decimals=4, min=-1e3)
    main_gear_x: float = spec("landing_gear", 0.44, "Main gear x (from nose)", "m", decimals=4)
    main_gear_y: float = spec("landing_gear", 0.2, "Main gear half-track", "m", decimals=4)
    wheelbase: float = spec("landing_gear", 0.306, "Wheelbase", "m", decimals=4)
    gear_height: float = spec("landing_gear", 0.15, "Gear height below CG", "m", decimals=4)
    gear_stiffness: float = spec("landing_gear", 30.7e3, "Strut stiffness", "N/m", decimals=1, max=1e9)
    damping_ratio: float = spec("landing_gear", 1.0, "Strut damping ratio", "", decimals=3)
    rolling_friction: float = spec("landing_gear", 0.04, "Rolling friction", "", decimals=3)
    brake_friction: float = spec("landing_gear", 0.4, "Brake friction", "", decimals=3)
    tables_dir: str = spec("aero", "aero_tables", "Tables directory", path=True)
    stall_alpha: float = spec("aero", 20.0, "Stall angle (CL = 0 above)", "deg")

    @property
    def thrust_coeffs(self) -> tuple[float, float, float, float]:
        return self.thrust_a, self.thrust_b, self.thrust_c, self.thrust_d

    def tables_path(self, base_dir: pathlib.Path) -> pathlib.Path:
        return (pathlib.Path(base_dir) / self.tables_dir).resolve()

    def report(self) -> str:
        return (
            f"Aircraft: {self.name}\n"
            f"  mass {self.mass} kg   Ix {self.ix}  Iy {self.iy}  Iz {self.iz}  Ixz {self.ixz} kg·m²\n"
            f"  S {self.s} m²  b {self.b} m  c {self.c} m\n"
            f"  limits: elevator ±{self.elevator_max}°  aileron ±{self.aileron_max}°  rudder ±{self.rudder_max}°\n"
            f"  tables: {self.tables_dir}"
        )

    @classmethod
    def load(cls, path: pathlib.Path) -> AircraftModel:
        return load_toml(cls, path)

    def save(self, path: pathlib.Path) -> None:
        save_toml(self, path)
