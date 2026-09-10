# Flight-Simulator

6DOF flight simulator in Python with trim, linear-mode and performance analysis, scripted or piloted flight, and a real-time bridge to FlightGear.

## Install

```
pip install -r requirements.txt
```

Python ≥ 3.11 (uses `tomllib`). FlightGear itself is only needed for the visualization bridge.
Runs on Linux and Windows; on Windows point *fgfs executable* (FlightGear tab) to `C:\Program Files\FlightGear ...\bin\fgfs.exe`.

## Run

```
python main.py                       # GUI (default case: cases/mushu)
python main.py run  cases/mushu      # offline simulation + plots   (--csv out.csv, --save, --no-show)
python main.py trim cases/mushu      # trim solution for the case's trim condition
python main.py modes cases/mushu     # linearized longitudinal / lateral modes
python main.py ceiling cases/mushu   # ceiling sweep with ISA density (--throttle, --speed, --h-max)
python main.py flightgear cases/mushu   # real-time bridge (--control joystick|keyboard|scripted, --command)
python main.py joysticks             # list joysticks
```

## Case folder

```
cases/<name>/
├── case.toml        simulation, environment, wind, initial condition, trim, maneuvers, FlightGear, joystick, plots
├── aircraft.toml    mass/inertia, geometry, control limits, propulsion, landing gear, aero tables
└── aero_tables/     <coef>.dat tables: first row = beta grid (deg), first column = alpha grid (deg)
```

Both files are edited from the GUI (Aircraft / Simulation / FlightGear tabs); *New case…* copies the current folder.

## GUI

- **Aircraft** – every aircraft parameter, with the aero tables checked as you edit.
- **Simulation** – time, environment (constant/ISA density, ground elevation, landing gear), wind and random gusts, trim condition (level, climb, coordinated turn, glide) or manual initial condition, and a table of scripted maneuvers (time-windowed deflections added to the trim).
- **Analysis** – run the simulation in the background, plots per channel group (position, Euler angles, rates, aerodynamics, body velocities/accelerations, controls, 3D trajectory), trim only, linear modes (eigenvalues, ωn, ζ), ceiling sweep, CSV export of the full history or a time window (loads analysis), PNG export.
- **FlightGear** – connection ports and rates, start mode (trimmed / initial condition / on ground), pilot input (joystick with axis mapping and a live axis monitor, keyboard, scripted), generated `fgfs` command, launch FlightGear and start/stop the bridge with live telemetry.

## FlightGear bridge

1. Configure and save the case (FlightGear tab).
2. Launch FlightGear with the generated command (`--fdm=external --native-fdm=socket,...`).
3. Start the bridge. The model integrates at `fdm_hz`, paced by FlightGear packets at `packet_hz`, and streams position/attitude/control surfaces back.

Keyboard input opens a small window that must stay focused: arrows = elevator/aileron, A/D = rudder, W/S = throttle, Q/E = elevator trim, B = brake, Space = center.

## Package layout

```
flightsim/
├── aircraft.py, case.py, config.py    TOML-backed dataclasses (schema drives the GUI forms)
├── environment.py                     density models, wind/gusts, ground
├── aero/                              table database (bilinear) and force/moment build-up
├── control/                           control sources: constant, scripted maneuvers, live, joystick, keyboard
├── core/                              state, equations, RK4, landing gear, Dynamics, TrimSolver, Simulator
├── analysis/                          plot channels/figures, CSV export, linearization/modes, ceiling sweep
├── flightgear/                        FDM bridge and fgfs command builder
└── session.py                         application layer shared by CLI, GUI and bridge
gui/                                   PySide6 interface
```

## Authors

- João Hercílio Zucchi
- Enzo Pasa Dias Barboza
