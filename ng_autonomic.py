"""
NG Autonomic Nervous System — Ecosystem-Wide Threat Level State

VENDORED FILE. Copy verbatim into any module that participates in
the autonomic nervous system. Do NOT modify per-module. Changes
propagate by updating this canonical source and re-vendoring.

State file: ~/.et_modules/autonomic_state.json

State transitions:
    PARASYMPATHETIC (rest/digest) -> normal operations
    SYMPATHETIC (fight/flight) -> elevated threat, all modules adjust

Canonical source: NeuroGraph/ng_autonomic.py
License: AGPL-3.0

Changelog:
    [2026-03-03] River audit — Established UPPERCASE as canonical
    case standard. All states and enums use UPPERCASE. No exceptions.
    Normalizes .upper() on both read and write to prevent silent
    mismatches with any older files or callers.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("ng_autonomic")

__version__ = "1.1.0"

_STATE_PATH = Path.home() / ".et_modules" / "autonomic_state.json"
_VALID_STATES = {"PARASYMPATHETIC", "SYMPATHETIC"}
_VALID_THREAT_LEVELS = {"none", "low", "medium", "high", "critical"}


def read_state() -> dict:
    """Read current autonomic state. Fast path ~0.1ms.

    Returns dict with keys: state, threat_level, triggered_by,
    timestamp, reason. Defaults to PARASYMPATHETIC if file missing
    or unreadable.
    """
    default = {
        "state": "PARASYMPATHETIC",
        "threat_level": "none",
        "triggered_by": "",
        "timestamp": 0.0,
        "reason": "default — no security module has written state",
    }
    if not _STATE_PATH.exists():
        return default
    try:
        with open(_STATE_PATH, "r") as f:
            data = json.load(f)
        raw_state = data.get("state", "PARASYMPATHETIC").upper()
        if raw_state not in _VALID_STATES:
            raw_state = "PARASYMPATHETIC"
        data["state"] = raw_state
        return data
    except (json.JSONDecodeError, OSError):
        return default


def write_state(
    state: str,
    threat_level: str,
    triggered_by: str,
    reason: str,
) -> None:
    """Write autonomic state. Only security modules should call this.

    Args:
        state: PARASYMPATHETIC or SYMPATHETIC (uppercase enforced)
        threat_level: none | low | medium | high | critical
        triggered_by: module_id of the calling module
        reason: human-readable reason for the state change
    """
    state = state.upper()
    if state not in _VALID_STATES:
        raise ValueError(f"Invalid state: {state}. Must be one of {_VALID_STATES}")
    if threat_level not in _VALID_THREAT_LEVELS:
        raise ValueError(f"Invalid threat_level: {threat_level}. Must be one of {_VALID_THREAT_LEVELS}")

    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "state": state,
        "threat_level": threat_level,
        "triggered_by": triggered_by,
        "timestamp": time.time(),
        "reason": reason,
    }
    tmp_path = _STATE_PATH.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, _STATE_PATH)
