# flightsim/core/trim.py
"""Trim optimisation: solves for steady flight conditions.

"""

from __future__ import annotations

import dataclasses

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from flightsim.case import Case
from flightsim.aircraft import AircraftModel

from flightsim.core.state import StateIndex


@dataclasses.dataclass(frozen=True)
class TrimResult:
    """Outcome of a trim optimisation.

    Attributes:
        condition: The trim condition that was solved.
        x0: Trimmed initial state vector, shape (12,).
        controls: Trimmed control settings.
    """

    condition: str
    x0: NDArray
    controls: ControlInput

    def summary(self) -> str:
        c = self.controls
        return (
            "Trim optimization complete.\n"
            f"  Condition   : {self.condition}\n"
            f"  Elevator: {c.elevator:.4f}, Aileron: {c.aileron:.4f}, "
            f"Rudder: {c.rudder:.4f}, Throttle: {c.throttle:.4f}, Brake: {c.brake:.4f}"
        )


class TrimSolver:
    """Finds trimmed states and control settings for steady conditions.

    Args:
        model: Aircraft model dataclass.
    """

    def __init__(self, model: AircraftModel, case: Case) -> None:
        self.model = model
        self.case  = case


        # One control source + one dynamics, reused by every cost eval.
        self._controls = LiveControl()
        self._dynamics = Dynamics(model, self.aero_db, self._controls, atmosphere)

    def solve(self, case: Case) -> TrimResult:
        """Dispatches to the requested trim condition.

        Returns:
            A TrimResult.

        Raises:
            ValueError: If the condition is unknown or a required arg is missing.
            RuntimeError: If the optimiser fails to converge.
            NotImplementedError: For conditions not yet implemented.
        """
        name = case.trim_name
        v_des = case.target_speed
        h_des = case.trim_alt
        gamma_des = case.trim_gamma
        radius_des = case.trim_radius

        if name == "steady_level_flight":
            return self._steady_level_flight(v_des, h_des, gamma_des)
        if name == "coordinated_turn":
            return self._coordinated_turn(v_des, h_des, radius_des)
        if name == "steady_climb":
            return self._steady_climb(v_des, h_des, gamma_des)
        if name in ("glide", "turn"):
            raise NotImplementedError(f"Trim condition '{condition}' not implemented yet.")
        raise ValueError(f"Unknown trim condition: '{condition}'.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _residual(self, x: NDArray, controls: ControlInput) -> float:
        """Sum of squared body accelerations — zero at trim."""
        self._controls.set(controls)
        dx = self._dynamics(x, 0.0)
        return (
            dx[StateIndex.U]**2 + dx[StateIndex.V]**2 + dx[StateIndex.W]**2
            + dx[StateIndex.P]**2 + dx[StateIndex.Q]**2 + dx[StateIndex.R]**2
        )

    def _steady_level_flight(
        self, Vdes: float, hdes: float, gammades: float,) -> TrimResult:
        def slfres(guess):
            alpha, delta_e, throttle = guess

            u = Vdes*np.cos(alpha)
            w = Vdes*np.sin(alpha)
            v = 0.0
            p, q, r = 0.0, 0.0, 0.0
            phi, theta, psi = 0.0, alpha+gammades, 0.0
            x = np.zeros(12)
            x[StateIndex.U] = u
            x[StateIndex.V] = v
            x[StateIndex.W] = w
            x[StateIndex.PHI] = phi
            x[StateIndex.THETA] = theta
            x[StateIndex.PSI] = psi
            x[StateIndex.P] = p
            x[StateIndex.Q] = q
            x[StateIndex.R] = r
            x[StateIndex.Z_E] = -hdes
            #a, b, c, d = 0.001274, -0.07204, -0.5428, 40.89
            #throttle = thrust/((a * Vdes**3 + b * Vdes**2 + c * Vdes + d)*(rho / 1.225))

            return self._residual(x, ControlInput(elevator=delta_e, throttle=throttle))

        guess = np.array([0.1, 0.0, 0.5])  # alpha, delta_e, throttle
        bounds = [(np.deg2rad(-10.0), np.deg2rad(20.0)),
                  (-25, 25),
                  (0.0, 1.0)]
        res = minimize(slfres, guess, method='SLSQP', bounds=bounds, tol=1e-12)

        if not res.success:
            raise RuntimeError(f"Trim optimization failed to converge: {res.message}")

        alpha_trim, delta_e_trim, throttle_trim = res.x

        initcond = np.zeros(12)
        initcond[StateIndex.U] = Vdes * np.cos(alpha_trim)
        initcond[StateIndex.W] = Vdes * np.sin(alpha_trim)
        initcond[StateIndex.THETA] = alpha_trim + gammades
        initcond[StateIndex.Z_E] = -hdes

        controls = ControlInput(elevator=delta_e_trim, throttle=throttle_trim)
        return TrimResult("steady_level_flight", initcond, controls, float(res.fun))

    def _coordinated_turn(
        self, Vdes: float, hdes: float, radiusdes: float | None,) -> TrimResult:
        if radiusdes is None:
            raise ValueError("Radius must be specified for coordinated turn trim condition.")
        psi_dot = Vdes/radiusdes

        def cturn(guess):
            alpha, phi, delta_e, delta_a, delta_r, throttle = guess
            theta = np.arctan((np.tan(alpha))*(np.cos(phi)))  #ze = -usin(theta) + w*cos(theta)*cos(phi) -> ze = 0
            psi = 0.0
            p = -psi_dot*np.sin(theta)
            q = psi_dot*np.sin(phi)*np.cos(theta)
            r = psi_dot*np.cos(phi)*np.cos(theta)

            u = Vdes*np.cos(alpha)
            w = Vdes*np.sin(alpha)
            v = 0.0

            x = np.zeros(12)
            x[StateIndex.U] = u
            x[StateIndex.V] = v
            x[StateIndex.W] = w
            x[StateIndex.PHI] = phi
            x[StateIndex.THETA] = theta
            x[StateIndex.PSI] = psi
            x[StateIndex.P] = p
            x[StateIndex.Q] = q
            x[StateIndex.R] = r
            x[StateIndex.Z_E] = -hdes

            return self._residual(
                x, ControlInput(elevator=delta_e, aileron=delta_a,
                                rudder=delta_r, throttle=throttle))

        phi_guess = np.arctan((Vdes**2) / (9.81 * radiusdes))
        guess = np.array([np.deg2rad(2.0), phi_guess, 0.0, 0.0, 0.0, 0.5])

        bounds = [
            (np.deg2rad(-5.0), np.deg2rad(15.0)),   # alpha
            (np.deg2rad(-60.0), np.deg2rad(60.0)),  # phi (max 60 deg bank)
            (-25.0, 25.0),                          # elevator
            (-20.0, 20.0),                          # aileron
            (-30.0, 30.0),                          # rudder
            (0.0, 1.0)                              # throttle
        ]

        res = minimize(cturn, guess, method='SLSQP', bounds=bounds, tol=1e-8)

        if not res.success:
            raise RuntimeError(f"Coordinated turn trim failed: {res.message}")

        alpha_tr, phi_tr, de_tr, da_tr, dr_tr, thr_tr = res.x
        theta_tr = np.arctan(np.tan(alpha_tr) * np.cos(phi_tr))

        initcond = np.zeros(12)
        initcond[StateIndex.U] = Vdes * np.cos(alpha_tr)
        initcond[StateIndex.W] = Vdes * np.sin(alpha_tr)
        initcond[StateIndex.PHI] = phi_tr
        initcond[StateIndex.THETA] = theta_tr
        initcond[StateIndex.P] = -psi_dot * np.sin(theta_tr)
        initcond[StateIndex.Q] =  psi_dot * np.sin(phi_tr) * np.cos(theta_tr)
        initcond[StateIndex.R] =  psi_dot * np.cos(phi_tr) * np.cos(theta_tr)
        initcond[StateIndex.Z_E] = -hdes

        controls = ControlInput(elevator=de_tr, aileron=da_tr,
                                rudder=dr_tr, throttle=thr_tr)
        return TrimResult("coordinated_turn", initcond, controls, float(res.fun))

    def _steady_climb(
        self, Vdes: float, hdes: float, gammades: float,) -> TrimResult:
        def sclimb(guess):
            alpha, delta_e, throttle = guess
            theta = alpha + np.deg2rad(gammades)
            u = Vdes*np.cos(alpha)
            w = Vdes*np.sin(alpha)
            v = 0.0
            p, q, r = 0.0, 0.0, 0.0
            x = np.zeros(12)
            x[StateIndex.U] = u
            x[StateIndex.V] = v
            x[StateIndex.W] = w
            x[StateIndex.PHI] = 0.0
            x[StateIndex.THETA] = theta
            x[StateIndex.PSI] = 0.0
            x[StateIndex.P] = p
            x[StateIndex.Q] = q
            x[StateIndex.R] = r
            x[StateIndex.Z_E] = -hdes
            return self._residual(x, ControlInput(elevator=delta_e, throttle=throttle))

        guess = np.array([0.1, 0.0, 0.5])  # alpha, delta_e, throttle
        bounds = [(np.deg2rad(-10.0), np.deg2rad(20.0)),
                  (-25, 25),
                  (0.0, 1.0)]
        res = minimize(sclimb, guess, method='SLSQP', bounds=bounds, tol=1e-12)
        if not res.success:
            raise RuntimeError(f"Steady climb trim failed: {res.message}")

        alpha_trim, delta_e_trim, throttle_trim = res.x
        initcond = np.zeros(12)
        initcond[StateIndex.U] = Vdes * np.cos(alpha_trim)
        initcond[StateIndex.W] = Vdes * np.sin(alpha_trim)
        initcond[StateIndex.THETA] = alpha_trim + np.deg2rad(gammades)
        initcond[StateIndex.Z_E] = -hdes

        controls = ControlInput(elevator=delta_e_trim, throttle=throttle_trim)
        return TrimResult("steady_climb", initcond, controls, float(res.fun))
