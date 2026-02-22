"""
CES Monitoring — Health context, rotating logger, and HTTP dashboard.

Three monitoring layers:

1. ``health_context()`` — Natural language string describing system state
   for prompt injection (e.g. "NeuroGraph: 1,234 nodes, 89% accuracy").
2. ``CESLogger`` — Rotating file logger to ``~/.neurograph/logs/ces.log``.
3. ``MonitoringDashboard`` — HTTP server (port 8847) with JSON endpoints
   for external monitoring tools.

Usage::

    from ces_monitoring import CESMonitor
    monitor = CESMonitor(ng_memory, ces_config)
    monitor.start()
    print(monitor.health_context())
    monitor.stop()

# ---- Changelog ----
# [2026-02-22] Claude (Opus 4.6) — Initial implementation.
#   What: CESMonitor with health_context, CESLogger with rotating file
#         handler, MonitoringDashboard HTTP server on port 8847.
#   Why:  Provides observability into CES subsystems for both AI
#         (natural language context) and human operators (HTTP + logs).
# -------------------
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

from ces_config import CESConfig

logger = logging.getLogger("neurograph.ces.monitoring")


# ── Health context (Layer 1) ───────────────────────────────────────────


def health_context(ng_memory: Any) -> str:
    """Generate a natural language health summary for prompt injection.

    Args:
        ng_memory: ``NeuroGraphMemory`` instance.

    Returns:
        Human-readable status string.
    """
    try:
        stats = ng_memory.stats()
        parts = [
            f"NeuroGraph: {stats.get('nodes', 0):,} nodes",
            f"{stats.get('synapses', 0):,} synapses",
        ]

        accuracy = stats.get("prediction_accuracy", 0)
        if accuracy > 0:
            parts.append(f"{accuracy:.0%} prediction accuracy")

        # CES subsystem status
        ces = stats.get("ces", {})
        sp = ces.get("stream_parser")
        if sp:
            if sp.get("is_running"):
                parts.append("stream parser active")
            else:
                parts.append("stream parser paused")

        surf = ces.get("surfacing")
        if surf:
            depth = surf.get("queue_depth", 0)
            if depth > 0:
                parts.append(f"{depth} concepts surfaced")

        return ", ".join(parts)
    except Exception as exc:
        return f"NeuroGraph: status unavailable ({exc})"


# ── Rotating logger (Layer 2) ─────────────────────────────────────────


class CESLogger:
    """Rotating file logger for CES events.

    Writes structured JSON-line events to ``ces.log`` with automatic
    rotation based on file size.

    Args:
        ces_config: ``CESConfig`` with monitoring parameters.
    """

    def __init__(self, ces_config: CESConfig) -> None:
        self._cfg = ces_config.monitoring
        self._logger = logging.getLogger("neurograph.ces.events")
        self._setup_handler()

    def _setup_handler(self) -> None:
        """Configure rotating file handler."""
        log_dir = Path(self._cfg.log_dir).expanduser()
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "ces.log"

        handler = logging.handlers.RotatingFileHandler(
            str(log_path),
            maxBytes=self._cfg.max_log_size_mb * 1024 * 1024,
            backupCount=self._cfg.backup_count,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        self._logger.addHandler(handler)
        self._logger.setLevel(logging.INFO)

    def log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Write a structured event to the CES log."""
        event = {
            "timestamp": time.time(),
            "event": event_type,
            "data": data,
        }
        self._logger.info(json.dumps(event, default=str))


# ── HTTP dashboard (Layer 3) ──────────────────────────────────────────


class _DashboardHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the monitoring dashboard."""

    # Set by MonitoringDashboard before server starts
    ng_memory: Any = None
    ces_monitor: Any = None

    def do_GET(self) -> None:
        """Handle GET requests."""
        if self.path == "/health":
            self._json_response(self._health_data())
        elif self.path == "/stats":
            self._json_response(self._stats_data())
        elif self.path == "/surfaced":
            self._json_response(self._surfaced_data())
        else:
            self.send_error(404, "Not Found")

    def _health_data(self) -> Dict[str, Any]:
        """Minimal health check response."""
        return {
            "status": "ok",
            "timestamp": time.time(),
            "context": health_context(self.ng_memory) if self.ng_memory else "unavailable",
        }

    def _stats_data(self) -> Dict[str, Any]:
        """Full telemetry response."""
        if self.ng_memory is not None:
            try:
                return self.ng_memory.stats()
            except Exception as exc:
                return {"error": str(exc)}
        return {"error": "not initialized"}

    def _surfaced_data(self) -> Dict[str, Any]:
        """Current surfaced concepts."""
        if self.ces_monitor is not None and self.ces_monitor._surfacing_monitor is not None:
            try:
                items = self.ces_monitor._surfacing_monitor.get_surfaced()
                return {"surfaced": items, "count": len(items)}
            except Exception as exc:
                return {"error": str(exc)}
        return {"surfaced": [], "count": 0}

    def _json_response(self, data: Dict[str, Any]) -> None:
        """Send a JSON response."""
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default stderr logging."""
        pass


class MonitoringDashboard:
    """HTTP monitoring server running in a daemon thread.

    Args:
        ces_config: ``CESConfig`` with monitoring parameters.
        ng_memory: ``NeuroGraphMemory`` instance for stats.
        ces_monitor: ``CESMonitor`` parent for surfacing access.
    """

    def __init__(
        self,
        ces_config: CESConfig,
        ng_memory: Any = None,
        ces_monitor: Any = None,
    ) -> None:
        self._cfg = ces_config.monitoring
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

        # Share refs with the handler class
        _DashboardHandler.ng_memory = ng_memory
        _DashboardHandler.ces_monitor = ces_monitor

    def start(self) -> None:
        """Start the HTTP server in a daemon thread."""
        if not self._cfg.http_enabled:
            logger.info("HTTP dashboard disabled by config")
            return

        try:
            self._server = HTTPServer(
                ("0.0.0.0", self._cfg.http_port), _DashboardHandler
            )
            self._thread = threading.Thread(
                target=self._server.serve_forever,
                daemon=True,
                name="ces-dashboard",
            )
            self._thread.start()
            logger.info("CES dashboard started on port %d", self._cfg.http_port)
        except Exception as exc:
            logger.warning("Failed to start CES dashboard: %s", exc)
            self._server = None

    def stop(self) -> None:
        """Shut down the HTTP server."""
        if self._server is not None:
            self._server.shutdown()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    @property
    def is_running(self) -> bool:
        """True if the server thread is alive."""
        return self._thread is not None and self._thread.is_alive()


# ── CESMonitor coordinator ────────────────────────────────────────────


class CESMonitor:
    """Coordinator for all CES monitoring infrastructure.

    Args:
        ng_memory: ``NeuroGraphMemory`` instance.
        ces_config: ``CESConfig`` with monitoring parameters.
    """

    def __init__(self, ng_memory: Any, ces_config: CESConfig) -> None:
        self._ng_memory = ng_memory
        self._cfg = ces_config
        self._ces_logger = CESLogger(ces_config)
        self._dashboard = MonitoringDashboard(
            ces_config, ng_memory=ng_memory, ces_monitor=self
        )

        # Access to surfacing monitor (set by openclaw_hook after init)
        self._surfacing_monitor: Any = None

        # Health check timer
        self._health_timer: Optional[threading.Timer] = None

    def start(self) -> None:
        """Start the dashboard and periodic health logging."""
        self._dashboard.start()
        self._schedule_health_check()

    def stop(self) -> None:
        """Stop all monitoring."""
        self._dashboard.stop()
        if self._health_timer is not None:
            self._health_timer.cancel()
            self._health_timer = None

    def get_health(self) -> Dict[str, Any]:
        """Return comprehensive health status."""
        result: Dict[str, Any] = {
            "timestamp": time.time(),
            "dashboard_running": self._dashboard.is_running,
            "dashboard_port": self._cfg.monitoring.http_port,
        }

        try:
            stats = self._ng_memory.stats()
            result["nodes"] = stats.get("nodes", 0)
            result["synapses"] = stats.get("synapses", 0)
            result["prediction_accuracy"] = stats.get("prediction_accuracy", 0)
            result["vector_db_count"] = stats.get("vector_db_count", 0)
        except Exception:
            result["status"] = "stats_unavailable"

        return result

    def health_context(self) -> str:
        """Generate a natural language health string."""
        return health_context(self._ng_memory)

    def log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Write an event to the CES rotating log."""
        self._ces_logger.log_event(event_type, data)

    # ── Internal ───────────────────────────────────────────────────────

    def _schedule_health_check(self) -> None:
        """Schedule periodic health check logging."""
        self._health_timer = threading.Timer(
            self._cfg.monitoring.health_interval,
            self._health_check_tick,
        )
        self._health_timer.daemon = True
        self._health_timer.start()

    def _health_check_tick(self) -> None:
        """Periodic health check callback."""
        try:
            health = self.get_health()
            self._ces_logger.log_event("health_check", health)
        except Exception as exc:
            logger.debug("Health check failed: %s", exc)
        self._schedule_health_check()
