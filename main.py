from __future__ import annotations

import argparse
import pathlib
import sys

DEFAULT_CASE = pathlib.Path(__file__).resolve().parent / "cases" / "default_plane"


def cmd_gui(args) -> None:
    from gui.app import run
    run(args.case)


def cmd_run(args) -> None:
    from flightsim.analysis.export import export_csv
    from flightsim.analysis.plots import generate_figures
    from flightsim.session import Session
    s = Session(args.case)
    res = s.run()
    if args.csv:
        print(f"Exported {export_csv(res, args.csv)} samples to {args.csv}")
    generate_figures(res, s.case.figures, s.output_dir, save=s.case.save_figures or args.save, show=not args.no_show)


def cmd_trim(args) -> None:
    from flightsim.session import Session
    print(Session(args.case).trim().summary())


def cmd_modes(args) -> None:
    from flightsim.analysis.linear import linearize, modes_report
    from flightsim.control.source import ScriptedControl
    from flightsim.session import Session
    s = Session(args.case)
    dyn = s.dynamics(ScriptedControl(s.case.baseline_controls()))
    x0, u0 = s.initial_conditions(print, dyn)
    print(modes_report(linearize(dyn, x0, u0)))


def cmd_ceiling(args) -> None:
    from flightsim.analysis.performance import ceiling_sweep
    from flightsim.control.source import ScriptedControl
    from flightsim.environment import Environment, ISADensity
    from flightsim.session import Session
    s = Session(args.case)
    env = Environment.from_case(s.case)
    env.density = ISADensity()
    dyn = s.dynamics(ScriptedControl(s.case.baseline_controls()), env)
    print(ceiling_sweep(dyn, args.speed or s.case.trim_airspeed, args.throttle, args.h_max, args.step, print).summary())


def cmd_flightgear(args) -> None:
    from flightsim.flightgear.bridge import FlightGearBridge, fgfs_command, join_command
    from flightsim.session import Session
    s = Session(args.case)
    if args.command:
        print(join_command(fgfs_command(s.case)))
        return
    FlightGearBridge(s, control_kind=args.control).run()


def cmd_joysticks(args) -> None:
    from flightsim.control.devices import list_joysticks
    names = list_joysticks()
    print("\n".join(f"[{i}] {n}" for i, n in enumerate(names)) if names else "No joystick detected.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="6DOF flight simulator")
    sub = parser.add_subparsers(dest="command")

    def add(name, func, help_text):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("case", nargs="?", default=DEFAULT_CASE, type=pathlib.Path, help="case directory")
        p.set_defaults(func=func)
        return p

    add("gui", cmd_gui, "open the graphical interface (default)")
    p = add("run", cmd_run, "run the offline simulation and plot results")
    p.add_argument("--csv", type=pathlib.Path, help="export full time history to CSV")
    p.add_argument("--save", action="store_true", help="save figures to the case output directory")
    p.add_argument("--no-show", action="store_true", help="do not open plot windows")
    add("trim", cmd_trim, "solve the trim condition defined in the case")
    add("modes", cmd_modes, "linearize about the initial condition and report dynamic modes")
    p = add("ceiling", cmd_ceiling, "sweep altitude (ISA density) to estimate the ceiling")
    p.add_argument("--throttle", type=float, default=1.0)
    p.add_argument("--speed", type=float, default=None, help="airspeed (default: trim airspeed)")
    p.add_argument("--h-max", type=float, default=6000.0)
    p.add_argument("--step", type=float, default=25.0)
    p = add("flightgear", cmd_flightgear, "run the FlightGear bridge")
    p.add_argument("--control", choices=["joystick", "keyboard", "scripted"], help="override pilot input source")
    p.add_argument("--command", action="store_true", help="only print the fgfs launch command")
    p = sub.add_parser("joysticks", help="list connected joysticks")
    p.set_defaults(func=cmd_joysticks)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else (sys.argv[1:] or ["gui"]))
    args.func(args)


if __name__ == "__main__":
    main()
