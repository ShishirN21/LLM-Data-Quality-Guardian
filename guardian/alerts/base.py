"""Alert manager and abstract handler."""

from __future__ import annotations

from abc import ABC, abstractmethod

from guardian.models import Alert, Severity


class AlertHandler(ABC):
    """Abstract base for alert delivery mechanisms."""

    @abstractmethod
    def send(self, alert: Alert) -> None:
        """Deliver an alert."""


class AlertManager:
    """Routes alerts to configured handlers based on severity."""

    _SEVERITY_ORDER = {
        Severity.DEBUG: 0,
        Severity.INFO: 1,
        Severity.WARNING: 2,
        Severity.CRITICAL: 3,
    }

    def __init__(self, config: dict) -> None:
        self._handlers: list[AlertHandler] = []
        self._min_severity = Severity(config.get("severity_threshold", "WARNING"))

    def register_handler(self, handler: AlertHandler) -> None:
        self._handlers.append(handler)

    def send(self, alert: Alert) -> None:
        """Send alert to all handlers if severity meets threshold."""
        if self._SEVERITY_ORDER[alert.severity] < self._SEVERITY_ORDER[self._min_severity]:
            return
        for handler in self._handlers:
            handler.send(alert)

    def create_and_send(
        self,
        severity: Severity,
        source: str,
        title: str,
        message: str,
        **context: object,
    ) -> Alert:
        """Factory method to create and send an alert."""
        alert = Alert(
            severity=severity,
            source_module=source,
            title=title,
            message=message,
            context=context,
        )
        self.send(alert)
        return alert
