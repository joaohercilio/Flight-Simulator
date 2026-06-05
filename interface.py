from __future__ import annotations
from PySide6.QtWidgets import (QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout,
                               QGroupBox, QFormLayout, QComboBox, QDoubleSpinBox, 
                               QLineEdit, QCheckBox, QTabWidget, QScrollArea, QLabel) 
import sys
import pathlib
import re
import subprocess
import tomllib
import tomli_w


def process_toml_file(file_path, ui_map, mode="load"):
    """Reads/Writes multi-section TOML files securely via regex sub-scoping."""
    if mode == "load":
        if not file_path.exists():
            return
        with open(file_path, "rb") as f:
            data = tomllib.load(f)

        for composite_key, (widget, val_type) in ui_map.items():
                section, key = composite_key.split('.')
                value = data.get(section, {}).get(key)
                if value is None:
                    continue
                if val_type == "num":
                    widget.setValue(float(value))
                elif val_type == "str":
                    if isinstance(widget, QComboBox): widget.setCurrentText(value)
                    else: widget.setText(value)
                elif val_type == "bool":
                    widget.setChecked(bool(value))
        
    elif mode == "save":
        data = {}
        if file_path.exists():
            with open(file_path, "rb") as f:
                data = tomllib.load(f)
        for composite_key, (widget, val_type) in ui_map.items():
            section, key = composite_key.split('.')
            data.setdefault(section, {})
            if val_type == "num":
                data[section][key] = widget.value()
            elif val_type == "str":
                data[section][key] = widget.currentText() if isinstance(widget, QComboBox) else widget.text()
            elif val_type == "bool":
                data[section][key] = widget.isChecked()


    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb") as f:
        tomli_w.dump(data, f)



class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Flight Simulator - Model Configurator")
        self.setMinimumSize(550, 700)

        main_layout = QVBoxLayout(self)
        
        case_group = QGroupBox("Active Simulation Target")
        case_layout = QFormLayout()
        self.line_case_name = QLineEdit("mushu")
        case_layout.addRow("Case Folder Name:", self.line_case_name)
        case_group.setLayout(case_layout)
        main_layout.addWidget(case_group)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        self.init_sim_env_tab()
        self.init_initial_conditions_tab()
        self.init_aircraft_tab()
        
        btn_next = QPushButton("Save All Configurations & Proceed ➡")
        btn_next.setStyleSheet("font-weight: bold; padding: 10px; background-color: #0d47a1; color: white;")
        btn_next.clicked.connect(self.proceed_to_execution)
        main_layout.addWidget(btn_next)

        self.line_case_name.editingFinished.connect(self.load_all_configs)
        self.load_all_configs()

    def init_sim_env_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        time_group = QGroupBox("Simulation Time Horizon")
        t_layout = QFormLayout()
        self.spin_t_start = QDoubleSpinBox(); self.spin_t_start.setRange(0, 100000); self.spin_t_start.setSuffix(" s")
        self.spin_t_end = QDoubleSpinBox(); self.spin_t_end.setRange(0, 100000); self.spin_t_end.setSuffix(" s")
        self.spin_dt = QDoubleSpinBox(); self.spin_dt.setRange(0.0001, 1.0); self.spin_dt.setDecimals(4); self.spin_dt.setSuffix(" s")
        t_layout.addRow("Start Time:", self.spin_t_start)
        t_layout.addRow("End Time:", self.spin_t_end)
        t_layout.addRow("Time Step (dt):", self.spin_dt)
        time_group.setLayout(t_layout)
        layout.addWidget(time_group)

        model_group = QGroupBox("Target Binary Definitions")
        m_layout = QFormLayout()
        self.line_sim_model_file = QLineEdit()
        m_layout.addRow("Aircraft Definition Data File:", self.line_sim_model_file)
        model_group.setLayout(m_layout)
        layout.addWidget(model_group)

        atm_group = QGroupBox("Atmospheric Models")
        atm_layout = QFormLayout()
        self.line_atm_model = QLineEdit()
        self.spin_atm_density = QDoubleSpinBox(); self.spin_atm_density.setRange(0.0, 5.0); self.spin_atm_density.setDecimals(4); self.spin_atm_density.setSuffix(" kg/m³")
        self.spin_atm_gravity = QDoubleSpinBox(); self.spin_atm_gravity.setRange(0.0, 30.0); self.spin_atm_gravity.setDecimals(3); self.spin_atm_gravity.setSuffix(" m/s²")
        atm_layout.addRow("Atmospheric Engine Model:", self.line_atm_model)
        atm_layout.addRow("Static Air Density:", self.spin_atm_density)
        atm_layout.addRow("Gravitational Constant:", self.spin_atm_gravity)
        atm_group.setLayout(atm_layout)
        layout.addWidget(atm_group)

        self.tabs.addTab(tab, "Simulation Basics")

        self.sim_env_map = {
            "simulation.t_start": (self.spin_t_start, "num"),
            "simulation.t_end":   (self.spin_t_end, "num"),
            "simulation.dt":      (self.spin_dt, "num"),
            "model.file":         (self.line_sim_model_file, "str"),
            "atmosphere.model":   (self.line_atm_model, "str"),
            "atmosphere.density": (self.spin_atm_density, "num"),
            "atmosphere.gravity": (self.spin_atm_gravity, "num")
        }

    def init_initial_conditions_tab(self):
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        content_widget = QWidget(); layout = QVBoxLayout(content_widget)

        trim_group = QGroupBox("State Optimization Constraints")
        trim_layout = QFormLayout()
        self.cb_trim_flight = QCheckBox("Enable System Trim Initialization Run")
        self.combo_trim_mode = QComboBox(); self.combo_trim_mode.addItems(["steady_level_flight", "coordinated_turn", "steady_climb", "glide", "turn"])
        self.spin_v_des = QDoubleSpinBox(); self.spin_v_des.setRange(0, 1000); self.spin_v_des.setSuffix(" m/s")
        self.spin_h_des = QDoubleSpinBox(); self.spin_h_des.setRange(-1000, 100000); self.spin_h_des.setSuffix(" m")
        self.spin_gamma_des = QDoubleSpinBox(); self.spin_gamma_des.setRange(-90, 90); self.spin_gamma_des.setSuffix(" °")
        self.spin_radius_des = QDoubleSpinBox(); self.spin_radius_des.setRange(0, 100000); self.spin_radius_des.setSuffix(" m")
        trim_layout.addRow(self.cb_trim_flight)
        trim_layout.addRow("Optimization Mode:", self.combo_trim_mode)
        trim_layout.addRow("Target Airspeed:", self.spin_v_des)
        trim_layout.addRow("Target Altitude:", self.spin_h_des)
        trim_layout.addRow("Flight Path Angle:", self.spin_gamma_des)
        trim_layout.addRow("Coordinated Turn Radius:", self.spin_radius_des)
        trim_group.setLayout(trim_layout)
        layout.addWidget(trim_group)

        ic_group = QGroupBox("Kinematic State Initial Vectors")
        ic_layout = QFormLayout()
        self.spin_u = QDoubleSpinBox(); self.spin_u.setRange(-2000, 2000); self.spin_u.setDecimals(3)
        self.spin_v = QDoubleSpinBox(); self.spin_v.setRange(-2000, 2000); self.spin_v.setDecimals(3)
        self.spin_w = QDoubleSpinBox(); self.spin_w.setRange(-2000, 2000); self.spin_w.setDecimals(3)
        self.spin_phi = QDoubleSpinBox(); self.spin_phi.setRange(-7, 7); self.spin_phi.setDecimals(4)
        self.spin_theta = QDoubleSpinBox(); self.spin_theta.setRange(-7, 7); self.spin_theta.setDecimals(4)
        self.spin_psi = QDoubleSpinBox(); self.spin_psi.setRange(-7, 7); self.spin_psi.setDecimals(4)
        self.spin_xe = QDoubleSpinBox(); self.spin_xe.setRange(-1000000, 1000000)
        self.spin_ye = QDoubleSpinBox(); self.spin_ye.setRange(-1000000, 1000000)
        self.spin_ze = QDoubleSpinBox(); self.spin_ze.setRange(-100000, 100000)
        self.spin_p = QDoubleSpinBox(); self.spin_p.setRange(-50, 50); self.spin_p.setDecimals(4)
        self.spin_q = QDoubleSpinBox(); self.spin_q.setRange(-50, 50); self.spin_q.setDecimals(4)
        self.spin_r = QDoubleSpinBox(); self.spin_r.setRange(-50, 50); self.spin_r.setDecimals(4)

        ic_layout.addRow("Velocity u:", self.spin_u); ic_layout.addRow("Velocity v:", self.spin_v); ic_layout.addRow("Velocity w:", self.spin_w)
        ic_layout.addRow("Roll phi (rad):", self.spin_phi); ic_layout.addRow("Pitch theta (rad):", self.spin_theta); ic_layout.addRow("Yaw psi (rad):", self.spin_psi)
        ic_layout.addRow("NED Position x_e:", self.spin_xe); ic_layout.addRow("NED Position y_e:", self.spin_ye); ic_layout.addRow("NED Position z_e:", self.spin_ze)
        ic_layout.addRow("Roll Rate p (rad/s):", self.spin_p); ic_layout.addRow("Pitch Rate q (rad/s):", self.spin_q); ic_layout.addRow("Yaw Rate r (rad/s):", self.spin_r)
        ic_group.setLayout(ic_layout)
        layout.addWidget(ic_group)

        scroll.setWidget(content_widget)
        self.tabs.addTab(scroll, "Initial Conditions & Trim")

        self.sim_ic_map = {
            "trim.trim_flight":             (self.cb_trim_flight, "bool"),
            "trimconditions.mode":          (self.combo_trim_mode, "str"),
            "trimconditions.v_des":         (self.spin_v_des, "num"),
            "trimconditions.h_des":         (self.spin_h_des, "num"),
            "trimconditions.gamma_des":     (self.spin_gamma_des, "num"),
            "trimconditions.radiusdes":      (self.spin_radius_des, "num"),
            "initial_condition.u":          (self.spin_u, "num"),
            "initial_condition.v":          (self.spin_v, "num"),
            "initial_condition.w":          (self.spin_w, "num"),
            "initial_condition.phi":        (self.spin_phi, "num"),
            "initial_condition.theta":      (self.spin_theta, "num"),
            "initial_condition.psi":        (self.spin_psi, "num"),
            "initial_condition.x_e":        (self.spin_xe, "num"),
            "initial_condition.y_e":        (self.spin_ye, "num"),
            "initial_condition.z_e":        (self.spin_ze, "num"),
            "initial_condition.p":          (self.spin_p, "num"),
            "initial_condition.q":          (self.spin_q, "num"),
            "initial_condition.r":          (self.spin_r, "num"),
        }

    def init_aircraft_tab(self):
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        content_widget = QWidget(); layout = QVBoxLayout(content_widget)

        inertia_group = QGroupBox("Mass Properties & Moments of Inertia")
        i_layout = QFormLayout()
        self.spin_mass = QDoubleSpinBox(); self.spin_mass.setRange(0, 100000); self.spin_mass.setDecimals(4)
        self.spin_ix = QDoubleSpinBox(); self.spin_ix.setRange(0, 100000); self.spin_ix.setDecimals(8)
        self.spin_iy = QDoubleSpinBox(); self.spin_iy.setRange(0, 100000); self.spin_iy.setDecimals(8)
        self.spin_iz = QDoubleSpinBox(); self.spin_iz.setRange(0, 100000); self.spin_iz.setDecimals(8)
        self.spin_ixz = QDoubleSpinBox(); self.spin_ixz.setRange(-50000, 50000); self.spin_ixz.setDecimals(8)
        self.spin_xcg = QDoubleSpinBox(); self.spin_xcg.setRange(-50, 50); self.spin_xcg.setDecimals(4)
        self.spin_ycg = QDoubleSpinBox(); self.spin_ycg.setRange(-50, 50); self.spin_ycg.setDecimals(4)
        self.spin_zcg = QDoubleSpinBox(); self.spin_zcg.setRange(-50, 50); self.spin_zcg.setDecimals(4)
        i_layout.addRow("Total Mass (kg):", self.spin_mass); i_layout.addRow("Moment Ix:", self.spin_ix); i_layout.addRow("Moment Iy:", self.spin_iy); i_layout.addRow("Moment Iz:", self.spin_iz)
        i_layout.addRow("Product Ixz:", self.spin_ixz); i_layout.addRow("CG x_cg:", self.spin_xcg); i_layout.addRow("CG y_cg:", self.spin_ycg); i_layout.addRow("CG z_cg:", self.spin_zcg)
        inertia_group.setLayout(i_layout); layout.addWidget(inertia_group)

        geo_group = QGroupBox("Aerodynamic Geometry")
        g_layout = QFormLayout()
        self.spin_geo_s = QDoubleSpinBox(); self.spin_geo_s.setRange(0, 5000); self.spin_geo_s.setDecimals(4)
        self.spin_geo_b = QDoubleSpinBox(); self.spin_geo_b.setRange(0, 200); self.spin_geo_b.setDecimals(4)
        self.spin_geo_c = QDoubleSpinBox(); self.spin_geo_c.setRange(0, 50); self.spin_geo_c.setDecimals(4)
        g_layout.addRow("Reference Area S:", self.spin_geo_s); g_layout.addRow("Wingspan b:", self.spin_geo_b); g_layout.addRow("Chord c:", self.spin_geo_c)
        geo_group.setLayout(g_layout); layout.addWidget(geo_group)

        limits_group = QGroupBox("Propulsion & Control Limits")
        l_layout = QFormLayout()
        self.spin_prop_armz = QDoubleSpinBox(); self.spin_prop_armz.setRange(-20, 20); self.spin_prop_armz.setDecimals(4)
        self.spin_lim_elv = QDoubleSpinBox(); self.spin_lim_elv.setRange(0, 90)
        self.spin_lim_ail = QDoubleSpinBox(); self.spin_lim_ail.setRange(0, 90)
        self.spin_lim_rud = QDoubleSpinBox(); self.spin_lim_rud.setRange(0, 90)
        self.spin_lim_brk = QDoubleSpinBox(); self.spin_lim_brk.setRange(0, 100000)
        l_layout.addRow("Engine Arm z Offset:", self.spin_prop_armz); l_layout.addRow("Elevator Max (deg):", self.spin_lim_elv)
        l_layout.addRow("Aileron Max (deg):", self.spin_lim_ail); l_layout.addRow("Rudder Max (deg):", self.spin_lim_rud); l_layout.addRow("Brake Max (N):", self.spin_lim_brk)
        limits_group.setLayout(l_layout); layout.addWidget(limits_group)

        aero_group = QGroupBox("Data Repository & Environment")
        a_layout = QFormLayout()
        self.line_aero_dir = QLineEdit()
        self.spin_gnd_alt = QDoubleSpinBox(); self.spin_gnd_alt.setRange(-2000, 10000); self.spin_gnd_alt.setDecimals(2)
        a_layout.addRow("Aero Tables Path:", self.line_aero_dir); a_layout.addRow("Ground Altitude:", self.spin_gnd_alt)
        aero_group.setLayout(a_layout); layout.addWidget(aero_group)

        scroll.setWidget(content_widget)
        self.tabs.addTab(scroll, "Aircraft Profile")

        self.aircraft_map = {
            "inertia.mass":                 (self.spin_mass, "num"),
            "inertia.Ix":                   (self.spin_ix, "num"),
            "inertia.Iy":                   (self.spin_iy, "num"),
            "inertia.Iz":                   (self.spin_iz, "num"),
            "inertia.Ixz":                  (self.spin_ixz, "num"),
            "inertia.x_cg":                 (self.spin_xcg, "num"),
            "inertia.y_cg":                 (self.spin_ycg, "num"),
            "inertia.z_cg":                 (self.spin_zcg, "num"),
            "geometry.S":                   (self.spin_geo_s, "num"),
            "geometry.b":                   (self.spin_geo_b, "num"),
            "geometry.c":                   (self.spin_geo_c, "num"),
            "propulsion.arm_z_engine":       (self.spin_prop_armz, "num"),
            "control_limits.elevator_max":  (self.spin_lim_elv, "num"),
            "control_limits.aileron_max":   (self.spin_lim_ail, "num"),
            "control_limits.rudder_max":    (self.spin_lim_rud, "num"),
            "control_limits.brake_max":     (self.spin_lim_brk, "num"),
            "aero.tables_dir":              (self.line_aero_dir, "str"),
            "ground_altitude.alt":          (self.spin_gnd_alt, "num")
        }

    def get_case_dir(self):
        case_name = self.line_case_name.text().strip() or "default_case"
        return pathlib.Path("cases") / case_name

    def load_all_configs(self):
        base_path = self.get_case_dir()
        process_toml_file(base_path / "sim_config.toml", self.sim_env_map, mode="load")
        process_toml_file(base_path / "sim_config.toml", self.sim_ic_map, mode="load")
        process_toml_file(base_path / "aircraft_model.toml", self.aircraft_map, mode="load")

    def proceed_to_execution(self):
        base_path = self.get_case_dir()
        (base_path / "aero_tables").mkdir(parents=True, exist_ok=True)
        (base_path / "results").mkdir(parents=True, exist_ok=True)
        
        process_toml_file(base_path / "sim_config.toml", self.sim_env_map, mode="save")
        process_toml_file(base_path / "sim_config.toml", self.sim_ic_map, mode="save")
        process_toml_file(base_path / "aircraft_model.toml", self.aircraft_map, mode="save")

        self.hide()
        self.run_window = RunWindow(self)
        self.run_window.show()


class RunWindow(QWidget):
    def __init__(self, creator_window):
        super().__init__()
        self.creator = creator_window
        self.setWindowTitle("Flight Simulator - Execution Control Center")
        self.setMinimumSize(500, 600)

        main_layout = QVBoxLayout(self)

        self.lbl_status = QLabel(f"<b>Active Target Case Directory:</b> cases/{self.creator.line_case_name.text()}")
        main_layout.addWidget(self.lbl_status)

        
        fg_group = QGroupBox("FlightGear Runtime Options (flightgear_config.toml)")
        fg_layout = QFormLayout()
        self.cb_find_ceiling = QCheckBox("Find Cruise Ceiling (Performance Sweep Mode)")
        self.cb_start_air = QCheckBox("Start Airborne Flight Profile")
        self.cb_start_trimmed = QCheckBox("Start Automatically Pre-Trimmed")
        self.cb_manual_control = QCheckBox("Enable Joystick Manual RC Overrides")
        self.spin_throttle_ceil = QDoubleSpinBox(); self.spin_throttle_ceil.setRange(0.0, 1.0); self.spin_throttle_ceil.setSingleStep(0.05)
        
        fg_layout.addRow(self.cb_find_ceiling)
        fg_layout.addRow(self.cb_start_air)
        fg_layout.addRow(self.cb_start_trimmed)
        fg_layout.addRow(self.cb_manual_control)
        fg_layout.addRow("Throttle Ceiling Limit Ratio:", self.spin_throttle_ceil)
        fg_group.setLayout(fg_layout)
        main_layout.addWidget(fg_group)

       
        plots_group = QGroupBox("Active Figure Array Subplots (plots.toml)")
        plots_layout = QVBoxLayout()
        self.cb_fig1 = QCheckBox("Figure 1: Position & Velocity NED")
        self.cb_fig2 = QCheckBox("Figure 2: Euler Angles, Euler Rates & Angular Velocity")
        self.cb_fig3 = QCheckBox("Figure 3: Aerodynamics & Body Velocity")
        self.cb_fig4 = QCheckBox("Figure 4: Trajectory 3D")
        
        plots_layout.addWidget(self.cb_fig1)
        plots_layout.addWidget(self.cb_fig2)
        plots_layout.addWidget(self.cb_fig3)
        plots_layout.addWidget(self.cb_fig4)
        plots_group.setLayout(plots_layout)
        main_layout.addWidget(plots_group)

        
        self.fg_map = {
            "flightgear.find_cruise_ceiling": (self.cb_find_ceiling, "bool"),
            "flightgear.start_in_air":        (self.cb_start_air, "bool"),
            "flightgear.start_trimmed":       (self.cb_start_trimmed, "bool"),
            "flightgear.manual_control":      (self.cb_manual_control, "bool"),
            "flightgear.throttleceiling":     (self.spin_throttle_ceil, "num")
        }

        self.load_phase2_configs()

        
        nav_layout = QHBoxLayout()
        btn_back = QPushButton("Back to Design Matrix")
        btn_back.setStyleSheet("padding: 8px; background-color: #616161; color: white; font-weight: bold;")
        btn_back.clicked.connect(self.go_back)
        nav_layout.addWidget(btn_back)
        main_layout.addLayout(nav_layout)

        run_layout = QHBoxLayout()
        btn_run_fg = QPushButton("Run Flightgear Bridge")
        btn_run_fg.setStyleSheet("font-weight: bold; padding: 12px; background-color: #c62828; color: white;")
        btn_run_fg.clicked.connect(lambda: self.execute_runner("flightgear.py"))
        
        btn_run_main = QPushButton("Generate plots")
        btn_run_main.setStyleSheet("font-weight: bold; padding: 12px; background-color: #ef6c00; color: white;")
        btn_run_main.clicked.connect(lambda: self.execute_runner("main.py"))
        
        run_layout.addWidget(btn_run_fg)
        run_layout.addWidget(btn_run_main)
        main_layout.addLayout(run_layout)

    def load_phase2_configs(self):
        base_path = self.creator.get_case_dir()
   
        process_toml_file(base_path / "flightgear_config.toml", self.fg_map, mode="load")
        
    
        plots_file = base_path / "plots.toml"
        if plots_file.exists():
            content = plots_file.read_text(encoding="utf-8")
            self.cb_fig1.setChecked('["Position", "Velocity NED"]' in content)
            self.cb_fig2.setChecked('["Euler angles", "Euler rates", "Angular velocity"]' in content)
            self.cb_fig3.setChecked('["Aerodynamics", "Body velocity"]' in content)
            self.cb_fig4.setChecked('["Trajectory 3D"]' in content)
        else:
           
            self.cb_fig1.setChecked(True)
            self.cb_fig2.setChecked(True)
            self.cb_fig3.setChecked(True)
            self.cb_fig4.setChecked(True)

    def save_phase2_configs(self):
        base_path = self.creator.get_case_dir()
        process_toml_file(base_path / "flightgear_config.toml", self.fg_map, mode="save")
        
        
        plots_buffer = "# Automatically generated plots layout structure\n"
        if self.cb_fig1.isChecked():
            plots_buffer += '\n[[figure]]\ngroups = ["Position", "Velocity NED"]\n'
        if self.cb_fig2.isChecked():
            plots_buffer += '\n[[figure]]\ngroups = ["Euler angles", "Euler rates", "Angular velocity"]\n'
        if self.cb_fig3.isChecked():
            plots_buffer += '\n[[figure]]\ngroups = ["Aerodynamics", "Body velocity"]\n'
        if self.cb_fig4.isChecked():
            plots_buffer += '\n[[figure]]\ngroups = ["Trajectory 3D"]\n'
            
        (base_path / "plots.toml").write_text(plots_buffer, encoding="utf-8")

    def go_back(self):
        self.save_phase2_configs()
        self.close()
        self.creator.load_all_configs()
        self.creator.show()

    def execute_runner(self, target_script):
        self.save_phase2_configs()
        print(f"Launching script process target: {target_script}")
        
        
        project_root = str(pathlib.Path(__file__).parent.absolute())
        
        subprocess.Popen(
            [sys.executable, target_script], 
            cwd=project_root
        )

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())