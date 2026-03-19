"""Console alert handler using rich for colored output."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel

from guardian.alerts.base import AlertHandler
from guardian.models import Alert, Severity

_SEVERITY_STYLES = {
    Severity.DEBUG: ("dim", "white"),
    Severity.INFO: ("bold blue", "blue"),
    Severity.WARNING: ("bold yellow", "yellow"),
    Severity.CRITICAL: ("bold red", "red"),
}


class ConsoleAlertHandler(AlertHandler):
    """Renders alerts to the console using rich formatting."""

    def __init__(self) -> None:
        self._console = Console()

    def send(self, alert: Alert) -> None:
        style, border = _SEVERITY_STYLES.get(
            alert.severity, ("white", "white")
        )
        icon = {
            Severity.DEBUG: "[dim]>>>[/]",
            Severity.INFO: "[blue]INFO[/]",
            Severity.WARNING: "[yellow]WARN[/]",
            Severity.CRITICAL: "[red bold]CRIT[/]",
        }.get(alert.severity, ">>>")

        self._console.print(
            Panel(
                f"{alert.message}\n[dim]Source: {alert.source_module}[/]",
                title=f"{icon} {alert.title}",
                border_style=border,
                padding=(0, 1),
            )
        )
