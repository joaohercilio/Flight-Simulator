# flightsim/control/source.py
"""Control input sources for the simulator.

A ``ControlSource`` decouples *where* control commands come from
(trim hold, a scripted maneuver, a joystick/keyboard, or — later —
FlightGear) from the dynamics. This is the seam a Qt GUI plugs into:
the GUI installs a ``LiveControl`` and writes to it from input events,
while the engine reads from it once per integration step.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ControlInput:
    """Immutable set of control commands.

    Attributes:
        elevator: Elevator deflection (deg).
        aileron: Aileron deflection (deg).
        rudder: Rudder deflection (deg).
        throttle: Throttle setting (0..1).
        brake: Brake command (N).
    """

    elevator: float = 0.0
    aileron: float = 0.0
    rudder: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0

    def as_tuple(self) -> tuple[float, float, float, float, float]:
        """Returns (elevator, aileron, rudder, throttle, brake)."""
        return (self.elevator, self.aileron, self.rudder,
                self.throttle, self.brake)


class ControlSource(abc.ABC):
    """Abstract source of control commands, queried by simulation time.

    Subclasses must implement :meth:`get`. The dynamics depend only on
    this interface, so swapping the input (trim, script, live) requires
    no change to the engine.
    """

    @abc.abstractmethod
    def get(self, t: float) -> ControlInput:
        """Returns the control command at simulation time ``t`` (s)."""


class ConstantControl(ControlSource):
    """Holds a single fixed command, e.g. the trim solution.

    Args:
        command: The constant control input to return.
    """

    def __init__(self, command: ControlInput) -> None:
        self._command = command

    def get(self, t: float) -> ControlInput:
        return self._command


@dataclass(frozen=True)
class Doublet:
    """A timed deflection added on top of a baseline command.

    Args:
        surface: One of 'elevator', 'aileron', 'rudder', 'throttle', 'brake'.
        start: Window start time (s, inclusive).
        end: Window end time (s, inclusive).
        deflection: Amount added to the baseline within the window.
    """

    surface: str
    start: float
    end: float
    deflection: float


class ScriptedControl(ControlSource):
    """Baseline command (usually trim) plus timed deflection windows.

    Replaces the hard-coded ``timed_control`` closure. Build doublets to
    excite a mode, e.g. ``Doublet("aileron", 5.0, 6.0, 2.0)``.

    Args:
        base: Baseline command held outside any doublet window.
        doublets: Timed perturbations applied additively.
    """

    def __init__(self, base: ControlInput,
                 doublets: list[Doublet] | None = None) -> None:
        self._base = base
        self._doublets = doublets or []

    def get(self, t: float) -> ControlInput:
        cmd = self._base
        for d in self._doublets:
            if d.start <= t <= d.end:
                cmd = replace(cmd, **{d.surface: getattr(cmd, d.surface)
                                      + d.deflection})
        return cmd


class LiveControl(ControlSource):
    """Mutable command written by an external producer (GUI / joystick).

    The GUI computes a full :class:`ControlInput` from its widgets or
    input events and calls :meth:`set`; the engine reads it each step via
    :meth:`get`. Same-thread use (QTimer driving the step loop) needs no
    locking.

    Args:
        command: Initial command.
    """

    def __init__(self, command: ControlInput | None = None) -> None:
        self._command = command or ControlInput()

    def set(self, command: ControlInput) -> None:
        """Replaces the current command."""
        self._command = command

    def get(self, t: float) -> ControlInput:
        return self._command
