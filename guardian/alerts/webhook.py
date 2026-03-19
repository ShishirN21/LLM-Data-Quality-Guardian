"""Webhook alert handler — sends alerts via HTTP POST."""

from __future__ import annotations

import json
import logging
import urllib.request

from guardian.alerts.base import AlertHandler
from guardian.models import Alert

logger = logging.getLogger(__name__)


class WebhookAlertHandler(AlertHandler):
    """Sends alerts as JSON via HTTP POST to a configured URL."""

    def __init__(self, webhook_url: str) -> None:
        self._url = webhook_url

    def send(self, alert: Alert) -> None:
        payload = {
            "severity": alert.severity.value,
            "source": alert.source_module,
            "title": alert.title,
            "message": alert.message,
            "context": alert.context,
            "timestamp": alert.created_at.isoformat(),
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                logger.debug("Webhook sent, status=%d", resp.status)
        except Exception:
            logger.exception("Failed to send webhook alert to %s", self._url)
