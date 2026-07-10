from typing import Callable, Optional

from .phase_transition import AlertLevel


class AlertCallback:
    """Configurable alert callback system for hallucination detection events.

    Supports registering callbacks for different alert levels,
    with optional cooldown to prevent callback flooding.
    """

    def __init__(self, cooldown_steps: int = 0):
        self._callbacks: dict[AlertLevel, list[Callable]] = {
            AlertLevel.WARNING: [],
            AlertLevel.CRITICAL: [],
        }
        self.cooldown_steps = cooldown_steps
        self._last_triggered_step: dict[AlertLevel, int] = {}
        self._current_step = 0

    def register(self, level: AlertLevel, callback: Callable):
        if level not in self._callbacks:
            self._callbacks[level] = []
        self._callbacks[level].append(callback)

    def on_warning(self, callback: Callable):
        self.register(AlertLevel.WARNING, callback)
        return callback

    def on_critical(self, callback: Callable):
        self.register(AlertLevel.CRITICAL, callback)
        return callback

    def notify(self, alert_level: AlertLevel, result: dict):
        self._current_step = result.get("step", self._current_step + 1)

        if alert_level not in self._callbacks:
            return

        if self.cooldown_steps > 0:
            last = self._last_triggered_step.get(alert_level, -self.cooldown_steps - 1)
            if self._current_step - last < self.cooldown_steps:
                return
            self._last_triggered_step[alert_level] = self._current_step

        for cb in self._callbacks[alert_level]:
            cb(result)