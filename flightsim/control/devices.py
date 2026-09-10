from __future__ import annotations

import dataclasses
import os

from flightsim.control.source import ControlInput, ControlSource

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")


def _pygame(headless: bool):
    if headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame
    pygame.display.init()
    pygame.joystick.init()
    return pygame


def stick_to_surfaces(pitch_up: float, roll_right: float, yaw_right: float, throttle: float, brake: float,
                      limits: tuple[float, float, float]) -> ControlInput:
    e_max, a_max, r_max = limits
    return ControlInput(-e_max * pitch_up, -a_max * roll_right, -r_max * yaw_right, throttle, brake)


def list_joysticks() -> list[str]:
    pg = _pygame(True)
    return [pg.joystick.Joystick(i).get_name() for i in range(pg.joystick.get_count())]


@dataclasses.dataclass(frozen=True)
class AxisMap:
    index: int = 0
    aileron: int = 0
    elevator: int = 1
    rudder: int = 3
    throttle: int = 2
    brake: int = -1
    invert_aileron: bool = False
    invert_elevator: bool = True
    invert_rudder: bool = False
    invert_throttle: bool = False
    deadband: float = 0.05


class Joystick:
    def __init__(self, axis_map: AxisMap, headless: bool = True) -> None:
        self.map = axis_map
        self._pg = _pygame(headless)
        if self._pg.joystick.get_count() <= axis_map.index:
            raise RuntimeError(f"Joystick {axis_map.index} not found ({self._pg.joystick.get_count()} connected)")
        self._js = self._pg.joystick.Joystick(axis_map.index)
        self._js.init()

    @property
    def name(self) -> str:
        return self._js.get_name()

    def axes(self) -> list[float]:
        self._pg.event.pump()
        return [self._js.get_axis(i) for i in range(self._js.get_numaxes())]

    def _axis(self, values: list[float], idx: int, invert: bool) -> float:
        if idx < 0 or idx >= len(values):
            return 0.0
        v = -values[idx] if invert else values[idx]
        return 0.0 if abs(v) < self.map.deadband else v

    def normalized(self) -> tuple[float, float, float, float, float]:
        values = self.axes()
        m = self.map
        ele = self._axis(values, m.elevator, m.invert_elevator)
        ail = self._axis(values, m.aileron, m.invert_aileron)
        rud = self._axis(values, m.rudder, m.invert_rudder)
        thr = (self._axis(values, m.throttle, m.invert_throttle) + 1.0) / 2.0
        brk = (self._axis(values, m.brake, False) + 1.0) / 2.0 if m.brake >= 0 else 0.0
        return ele, ail, rud, thr, brk

    def close(self) -> None:
        self._pg.joystick.quit()


class JoystickControl(ControlSource):
    def __init__(self, axis_map: AxisMap, limits: tuple[float, float, float]) -> None:
        self._js = Joystick(axis_map)
        self._limits = limits
        self._command = ControlInput()
        self.poll()

    @property
    def name(self) -> str:
        return self._js.name

    def poll(self) -> None:
        pitch_up, roll_right, yaw_right, thr, brk = self._js.normalized()
        self._command = stick_to_surfaces(pitch_up, roll_right, yaw_right, thr, brk, self._limits)

    def get(self, t: float) -> ControlInput:
        return self._command

    def close(self) -> None:
        self._js.close()


class KeyboardControl(ControlSource):
    KEYS = ("Up/Down: nose up/down   Left/Right: roll   A/D: yaw   W/S: throttle   B: brake   Space: center   "
            "Q/E: pitch trim")

    def __init__(self, limits: tuple[float, float, float], rate: float = 1.5) -> None:
        self._pg = _pygame(False)
        self._screen = self._pg.display.set_mode((640, 120))
        self._pg.display.set_caption("Flight Simulator — keyboard input (keep this window focused)")
        try:
            self._pg.font.init()
            self._font = self._pg.font.SysFont(None, 22)
        except Exception:
            self._font = None
        self._limits = limits
        self._rate = rate
        self._axes = [0.0, 0.0, 0.0, 0.5, 0.0]
        self._trim = 0.0
        self._clock = self._pg.time.get_ticks()
        self._drawn = 0
        self._command = ControlInput()

    def poll(self) -> None:
        pg = self._pg
        pg.event.pump()
        now = pg.time.get_ticks()
        step = self._rate * (now - self._clock) / 1000.0
        self._clock = now
        k = pg.key.get_pressed()
        ele, ail, rud, thr, brk = self._axes

        def move(value, up, down, hold=True):
            if up:
                return min(value + step, 1.0)
            if down:
                return max(value - step, -1.0)
            return value if hold else value * max(0.0, 1.0 - 4.0 * step)

        ele = move(ele, k[pg.K_UP], k[pg.K_DOWN], hold=False)
        ail = move(ail, k[pg.K_RIGHT], k[pg.K_LEFT], hold=False)
        rud = move(rud, k[pg.K_d], k[pg.K_a], hold=False)
        thr = min(max(move(thr, k[pg.K_w], k[pg.K_s]), 0.0), 1.0)
        brk = 1.0 if k[pg.K_b] else 0.0
        if k[pg.K_q]:
            self._trim = min(self._trim + 0.3 * step, 1.0)
        if k[pg.K_e]:
            self._trim = max(self._trim - 0.3 * step, -1.0)
        if k[pg.K_SPACE]:
            ele = ail = rud = self._trim = 0.0
        self._axes = [ele, ail, rud, thr, brk]
        self._command = stick_to_surfaces(min(max(ele + self._trim, -1.0), 1.0), ail, rud, thr, brk, self._limits)
        if now - self._drawn > 50:
            self._drawn = now
            self._draw()

    def _draw(self) -> None:
        pg = self._pg
        self._screen.fill((20, 24, 32))
        c = self._command
        lines = [self.KEYS, f"elevator {c.elevator:+6.1f}°   aileron {c.aileron:+6.1f}°   rudder {c.rudder:+6.1f}°   "
                            f"throttle {c.throttle:4.2f}   brake {c.brake:3.1f}"]
        for i, line in enumerate(lines if self._font else []):
            self._screen.blit(self._font.render(line, True, (220, 220, 220)), (12, 20 + 40 * i))
        pg.display.flip()

    def get(self, t: float) -> ControlInput:
        return self._command

    def close(self) -> None:
        self._pg.display.quit()
