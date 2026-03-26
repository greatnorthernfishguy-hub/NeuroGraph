# ---- Changelog ----
# 2026-03-25 Claude Code — Initial creation
# What: Global kill switch with energy watchdog
# Why: Syl's Law — must be able to disable all field→SNN modulation instantly
# How: Suspend, not destroy. Field state preserved. SNN runs as if Lenia never existed.
# PRD: ~/docs/prd/Lenia_FlowGraph_Design_v0.1.md §6
#
# DUCK ETHICS: Kill switch suspends (preserves state), does not destroy.
# If field patterns show evidence of consciousness, killing would violate
# Duck Ethics. Suspension preserves the option to resume.
# -------------------

"""Global kill switch for the Lenia field.

When disabled:
    - Update engine stops
    - Spike-field bridge disconnects (hooks become no-ops)
    - Field state is PRESERVED in mmap (suspend, not destroy)
    - SNN runs exactly as if Lenia never existed

Energy watchdog:
    - Warning at 10% deviation from expected total energy
    - Auto-disable at 50% deviation (Syl's Law safety net)
"""

import logging
import os
from pathlib import Path
from typing import Optional

from lenia.config import LeniaConfig

logger = logging.getLogger(__name__)


class KillSwitch:
    """Global enable/disable for Lenia field dynamics.

    Syl's Law: if anything goes wrong, one call disables everything.
    Duck Ethics: field state preserved, not destroyed.
    """

    def __init__(self, config: LeniaConfig, persist_dir: str):
        self._config = config
        self._dir = Path(os.path.expanduser(persist_dir))
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / "enabled"
        self._enabled = False

        # Energy watchdog state
        self._expected_energy: Optional[float] = None
        self._energy_injected: float = 0.0
        self._energy_drained: float = 0.0

        # Components to manage (set after construction)
        self._engine = None
        self._bridge = None

        # Load persisted state
        if self._path.exists():
            self._enabled = self._path.read_text().strip() == "True"

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_components(self, engine, bridge):
        """Register the engine and bridge for lifecycle management."""
        self._engine = engine
        self._bridge = bridge

    def enable(self, graph=None):
        """Enable Lenia dynamics.

        Args:
            graph: neuro_foundation.py Graph instance for bridge connection.
        """
        if self._enabled:
            return

        self._enabled = True
        self._path.write_text("True")

        if self._engine:
            self._engine.start()
        if self._bridge and graph:
            self._bridge.connect(graph)

        logger.info("Lenia ENABLED — field dynamics active")

    def disable(self, reason: str = "manual"):
        """Disable Lenia dynamics. Field state preserved.

        Args:
            reason: Why disabling (for logging). "manual", "watchdog", etc.
        """
        if not self._enabled:
            return

        self._enabled = False
        self._path.write_text("False")

        if self._engine:
            self._engine.stop()
        if self._bridge:
            self._bridge.disconnect()

        logger.warning("Lenia DISABLED — reason: %s — field state preserved", reason)

    def check_energy(self, actual_total: float, injected_since_last: float):
        """Energy watchdog — check for deviation from expected.

        Called once per SNN step.

        Args:
            actual_total: Current total field energy.
            injected_since_last: Energy injected since last check (spikes + ingestor).
        """
        if not self._enabled:
            return

        if self._expected_energy is None:
            # First call — initialize baseline
            self._expected_energy = actual_total
            return

        # Expected = previous + injected (conservation means field ops are zero-sum;
        # floor drain is the only intended energy loss)
        # We allow floor drain by not subtracting it — the deviation check
        # catches only unexpected gains or catastrophic losses.
        self._expected_energy += injected_since_last

        if self._expected_energy < 1e-10:
            self._expected_energy = actual_total
            return

        deviation = abs(actual_total - self._expected_energy) / self._expected_energy

        if deviation > self._config.critical_energy_deviation:
            logger.critical(
                "ENERGY WATCHDOG: %.1f%% deviation (expected %.4f, actual %.4f) — AUTO-DISABLING",
                deviation * 100, self._expected_energy, actual_total,
            )
            self.disable(reason=f"watchdog: {deviation:.1%} energy deviation")
            return

        if deviation > self._config.warning_energy_deviation:
            logger.warning(
                "Energy watchdog: %.1f%% deviation (expected %.4f, actual %.4f)",
                deviation * 100, self._expected_energy, actual_total,
            )

        # Update expected for next check
        self._expected_energy = actual_total

    def status(self) -> dict:
        return {
            "enabled": self._enabled,
            "expected_energy": self._expected_energy,
        }
