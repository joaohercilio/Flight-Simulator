# main.py
"""Entry point for the 6DOF flight simulator."""

import pathlib
from config.settings import SimConfig
from utils.io import load_model, generate_plots
from flightsim.core.simulation import run_simulation, sample_and_save_loads
from flightsim.aero.database import AeroDatabase
from flightsim.core.controlsys import trim_opt

CASE_DIR = pathlib.Path("cases/mushu")

    


def main() -> None:
    """Loads config, runs simulation, and generates plots."""
    cfg = SimConfig.from_toml_file(CASE_DIR / "sim_config.toml")

    model = load_model(CASE_DIR / "aircraft_model.toml")
    model.report()
    if cfg.trim:
        print(f"Performing trim optimization ({cfg.trim_condition})...")
        
        x0, trim_controls = trim_opt(
            cfg.v_des, 
            cfg.h_des, 
            cfg.gamma_des,
            cfg.radiusdes, 
            cfg.atmosphere, 
            model, 
            condition=cfg.trim_condition
        )
        print("Trim optimization complete. Initial conditions and control inputs:")
        #print(f"  Initial state: {x0}")
        print(f"  Elevator: {trim_controls['elevator']:.4f}, Aileron: {trim_controls['aileron']:.4f}, Rudder: {trim_controls['rudder']:.4f}, Throttle: {trim_controls['throttle']:.4f}, Brake: {trim_controls['brake']:.4f}")
    else:
        print("Bypassing trim optimization. Using manual initial conditions.")
        x0 = cfg.x0
        
        trim_controls = {
            "elevator": 0.0,
            "aileron": 0.0,
            "rudder": 0.0,
            "throttle": 0.5,
            "brake": 0.0
        }

    t, x, dx = run_simulation(
        model, x0, trim_controls,
        cfg.t_start, cfg.t_end, cfg.dt,
        atmosphere=cfg.atmosphere
    )

    #sample_and_save_loads(t=t, x=x, dx=dx, t_start=5, t_end=5.5, 
    #                      filename=r"C:\Users\enzo_\Documents\sim_results\aileronVd.csv")

    generate_plots(
        t, x, dx,
        plot_config=cfg.plot_config,
        output_dir=cfg.output_dir,
        save_figures=cfg.save_figures,
        show_gui=cfg.show_gui,
    )


if __name__ == "__main__":
    main()
