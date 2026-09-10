from __future__ import annotations

import abc
import dataclasses

SURFACES = ("elevator", "aileron", "rudder", "throttle", "brake")


@dataclasses.dataclass(frozen=True)
class ControlInput:
    elevator: float = 0.0
    aileron: float = 0.0
    rudder: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0

    def as_tuple(self) -> tuple[float, float, float, float, float]:
        return self.elevator, self.aileron, self.rudder, self.throttle, self.brake

    def summary(self) -> str:
        return (f"elevator {self.elevator:+.3f}°  aileron {self.aileron:+.3f}°  rudder {self.rudder:+.3f}°  "
                f"throttle {self.throttle:.3f}  brake {self.brake:.2f}")


@dataclasses.dataclass(frozen=True)
class Maneuver:
    surface: str = "elevator"
    start: float = 5.0
    end: float = 6.0
    deflection: float = 0.0


class ControlSource(abc.ABC):
    @abc.abstractmethod
    def get(self, t: float) -> ControlInput: ...

    def poll(self) -> None:
        pass

    def close(self) -> None:
        pass


class ConstantControl(ControlSource):
    def __init__(self, command: ControlInput) -> None:
        self._command = command

    def get(self, t: float) -> ControlInput:
        return self._command


class ScriptedControl(ControlSource):
    def __init__(self, base: ControlInput, maneuvers: list[Maneuver] | None = None) -> None:
        self._base = base
        self._maneuvers = maneuvers or []

    def get(self, t: float) -> ControlInput:
        cmd = self._base
        for m in self._maneuvers:
            if m.start <= t <= m.end:
                cmd = dataclasses.replace(cmd, **{m.surface: getattr(cmd, m.surface) + m.deflection})
        return cmd


class LiveControl(ControlSource):
    def __init__(self, command: ControlInput | None = None) -> None:
        self._command = command or ControlInput()

    def set(self, command: ControlInput) -> None:
        self._command = command

    def get(self, t: float) -> ControlInput:
        return self._command
