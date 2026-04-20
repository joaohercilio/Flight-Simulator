# main.py
"""Entry point for the 6DOF flight simulator."""

import pathlib
import os
from config.settings import SimConfig
from utils.io import load_model, generate_plots
from flightsim.core.simulation import run_simulation
from flightsim.aero.database import AeroDatabase
import optvl

CASE_DIR = pathlib.Path("cases/mushu")
def levelflight(mass, rho, V, ixx, iyy, izz, izx):
    #os.chdir(r"C:\Users\enzo_\Desktop\AVL\Design Files-20260310T235542Z-3-001\Design Files\466 - 2C\Avl_00001")
    ovl = optvl.OVLSolver(geo_file="Setup_cruzeiro _subido_ev015_cortado.avl")
    #DADOS DE REFERÊNCIA
    ref_data = ovl.get_reference_data()

    Sref = ref_data['Sref']
    Bref = ref_data['Bref']
    Cref = ref_data['Cref']
    XYZref = ref_data['XYZref']
    xcg = XYZref[0]
    zcg = XYZref[2]
    CD0  = 0.02567
    g=9.81
    W = mass*g
    q=0.5*rho*V**2
    CL_required = W / (q * Sref)
    ovl.set_constraint("alpha", "CL", CL_required)
    ovl.set_constraint("D2", "Cm", 0.0)
    ovl.set_parameter("velocity",   V)
    ovl.set_parameter("mass",       mass)
    ovl.set_parameter("Ixx",        ixx)
    ovl.set_parameter("Iyy",        iyy)
    ovl.set_parameter("Izz",        izz)
    ovl.set_parameter("Izx",        izx)
    ovl.set_parameter("grav.acc.",  g)
    ovl.set_parameter("density",    rho)
    ovl.set_parameter("CD0",        CD0)
    ovl.set_parameter("X cg",       xcg)
    ovl.set_parameter("Z cg",       zcg)

    ovl.execute_run()

    alpha_trim=ovl.get_variable("alpha")
    elevator_trim = ovl.get_control_deflection("profundor")
    return alpha_trim, elevator_trim

    


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

    generate_plots(
        t, x, dx,
        plot_config=cfg.plot_config,
        output_dir=cfg.output_dir,
        save_figures=cfg.save_figures,
        show_gui=cfg.show_gui,
    )


if __name__ == "__main__":
    main()
