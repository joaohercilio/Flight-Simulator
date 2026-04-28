# main.py
"""Entry point for the 6DOF flight simulator."""

import pathlib
from config.settings import SimConfig
from utils.io import load_model, generate_plots
from flightsim.core.simulation import run_simulation, sample_and_save_loads
from flightsim.aero.database import AeroDatabase

CASE_DIR = pathlib.Path("cases/mushu")

    


def main() -> None:
    """Loads config, runs simulation, and generates plots."""
    cfg = SimConfig.from_toml_file(CASE_DIR / "sim_config.toml")

    model = load_model(CASE_DIR / "aircraft_model.toml")
    model.report()

    t, x, dx = run_simulation(
        model, cfg.x0,
        cfg.t_start, cfg.t_end, cfg.dt,
        atmosphere=cfg.atmosphere
    )
    sample_and_save_loads(t=t, x=x, dx=dx, t_start=5, t_end=5.8, 
                       filename=r"C:\Users\enzo_\Documents\sim_results\TESTVc.csv")
    generate_plots(
        t, x, dx,
        plot_config=cfg.plot_config,
        output_dir=cfg.output_dir,
        save_figures=cfg.save_figures,
        show_gui=cfg.show_gui,
    )


if __name__ == "__main__":
    main()
