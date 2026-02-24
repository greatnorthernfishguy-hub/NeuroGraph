"""
CES Configuration — Centralized configuration for the Cognitive Enhancement Suite.

Provides a single ``CESConfig`` dataclass that holds all tuneable parameters for
the three CES modules (StreamParser, ActivationPersistence, SurfacingMonitor)
plus the monitoring infrastructure.  Configuration can be loaded from a dict of
overrides, a JSON file, or left at sensible defaults.

Usage::

    from ces_config import CESConfig, load_ces_config

    # Defaults
    cfg = load_ces_config()

    # With overrides
    cfg = load_ces_config({"streaming": {"ollama_model": "mxbai-embed-large"}})

    # From JSON file
    cfg = load_ces_config(config_path="~/.neurograph/ces.json")

# ---- Changelog ----
# [2026-02-22] Claude (Opus 4.6) — Initial implementation.
#   What: CESConfig dataclass with four sections (streaming, surfacing,
#         persistence, monitoring) and load_ces_config() factory.
#   Why:  Centralise all CES tunables so the three modules + monitoring
#         share one source of truth, user-overridable from a dict or file.
# -------------------
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("neurograph.ces")


# ── Section dataclasses ────────────────────────────────────────────────


@dataclass
class StreamingConfig:
    """Configuration for the StreamParser module."""

    ollama_model: str = "nomic-embed-text"
    ollama_url: str = "http://localhost:11434"
    chunk_size: int = 50
    overlap: int = 10
    nudge_strength: float = 0.15
    similarity_threshold: float = 0.6
    max_queue: int = 1000
    ollama_check_interval: float = 60.0


@dataclass
class SurfacingConfig:
    """Configuration for the SurfacingMonitor module."""

    voltage_threshold: float = 0.6
    min_confidence: float = 0.3
    max_surfaced: int = 5
    decay_rate: float = 0.95
    format: str = "context_block"
    include_metadata: bool = True
    queue_capacity: int = 50


@dataclass
class PersistenceConfig:
    """Configuration for the ActivationPersistence module."""

    sidecar_suffix: str = ".activations.json"
    decay_per_hour: float = 0.1
    min_activation: float = 0.05
    max_entries: int = 10000
    auto_save_interval: float = 300.0


@dataclass
class MonitoringConfig:
    """Configuration for the CES monitoring infrastructure."""

    log_dir: str = "~/.neurograph/logs/"
    max_log_size_mb: int = 10
    backup_count: int = 5
    http_port: int = 8847
    health_interval: float = 30.0
    http_enabled: bool = True


# ── Top-level config ───────────────────────────────────────────────────


@dataclass
class CESConfig:
    """Top-level Cognitive Enhancement Suite configuration.

    Groups all tunables into four sections.  Use ``load_ces_config()``
    to create an instance with user overrides applied.
    """

    streaming: StreamingConfig = field(default_factory=StreamingConfig)
    surfacing: SurfacingConfig = field(default_factory=SurfacingConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)


# ── Factory ────────────────────────────────────────────────────────────


def _apply_overrides(obj: Any, overrides: Dict[str, Any]) -> None:
    """Apply a dict of overrides to a dataclass instance (in-place)."""
    for key, value in overrides.items():
        if hasattr(obj, key):
            setattr(obj, key, value)


def load_ces_config(
    overrides: Optional[Dict[str, Any]] = None,
    config_path: Optional[str] = None,
) -> CESConfig:
    """Create a ``CESConfig`` with defaults, optionally overridden.

    Override precedence (highest wins):
        1. ``overrides`` dict argument
        2. ``config_path`` JSON file
        3. Built-in defaults

    Args:
        overrides: Dict keyed by section name (``streaming``, ``surfacing``,
            ``persistence``, ``monitoring``) whose values are dicts of
            field→value pairs.
        config_path: Path to a JSON file with the same structure as
            ``overrides``.

    Returns:
        Fully populated ``CESConfig``.
    """
    cfg = CESConfig()

    # Layer 1: JSON file
    if config_path is not None:
        p = Path(config_path).expanduser()
        if p.exists():
            try:
                with open(p) as f:
                    file_data = json.load(f)
                for section in ("streaming", "surfacing", "persistence", "monitoring"):
                    if section in file_data:
                        _apply_overrides(getattr(cfg, section), file_data[section])
            except Exception as exc:
                logger.warning("Failed to load CES config from %s: %s", p, exc)

    # Layer 2: dict overrides (win over file)
    if overrides is not None:
        for section in ("streaming", "surfacing", "persistence", "monitoring"):
            if section in overrides:
                _apply_overrides(getattr(cfg, section), overrides[section])

    return cfg
