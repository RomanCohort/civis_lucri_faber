"""Simulation Clock — tracks simulation time and circadian rhythm.

Provides:
  - tick(): advance simulation time by one step
  - circadian phase computation
  - time_dict(): export current time state as a dictionary
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class ClockState:
    step: int = 0
    hour: float = 0.0       # circadian hour [0, 24)
    day: int = 0


class SimulationClock:
    """Lightweight simulation clock with circadian tracking.

    Usage:
        clock = SimulationClock(step_duration=0.1)  # each step = 0.1 hours
        clock.tick()
        clock.tick()
        print(clock.state.hour)       # 0.2
        print(clock.time_dict())      # {'step': 2, 'hour': 0.2, 'day': 0, ...}
    """

    def __init__(self, step_duration: float = 0.1, start_hour: float = 8.0):
        """Args:
            step_duration: how many circadian hours each step advances
            start_hour: initial circadian hour (e.g., 8.0 = 8 AM)
        """
        self.step_duration = step_duration
        self.state = ClockState(hour=start_hour)

    def tick(self) -> dict:
        """Advance one step and return the current time state."""
        self.state.step += 1
        self.state.hour += self.step_duration
        # Wrap around at 24h
        while self.state.hour >= 24.0:
            self.state.hour -= 24.0
            self.state.day += 1
        return self.time_dict()

    def circadian_phase(self) -> float:
        """Return the circadian phase as sin(hour * pi / 12).

        - phase > 0: daytime (alertness higher)
        - phase < 0: nighttime (sleep pressure higher)
        """
        return float(np.sin(self.state.hour * np.pi / 12.0))

    def is_daytime(self) -> bool:
        """Return True if current hour is between 6 and 22."""
        return 6.0 <= self.state.hour < 22.0

    def time_dict(self) -> dict:
        """Export current clock state as a dictionary."""
        return {
            "step": self.state.step,
            "hour": self.state.hour,
            "day": self.state.day,
            "circadian_phase": self.circadian_phase(),
            "is_daytime": self.is_daytime(),
        }
