# main.py
"""Entry point for the 6DOF flight simulator."""

from config.cli import parse_args
from utils.io import load_model, generate_plots
from flightsim.core.simulation import run_simulation


def main() -> None:
    """Loads config, runs simulation, and generates plots."""
    cfg = parse_args()

    model = load_model(cfg.model_file)
    model.report()

    t, x, dx = run_simulation(model, cfg.x0, cfg.t_start, cfg.t_end, cfg.dt)

    generate_plots(
        t, x, dx,
        plot_config=cfg.plot_config,
        output_dir=cfg.output_dir,
        save_figures=cfg.save_figures,
        show_gui=cfg.show_gui,
    )


if __name__ == "__main__":
    main()
