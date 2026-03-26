# ---- Changelog ----
# 2026-03-25 Claude Code — Initial creation
# What: Lenia update engine — the 5-phase tick loop
# Why: Continuous field dynamics independent of SNN timestep
# How: Daemon thread running read→growth→conservation→apply→bookkeeping
# PRD: ~/docs/prd/Lenia_FlowGraph_Design_v0.1.md §3, §4
# -------------------

"""Lenia update engine.

Runs as a daemon thread. Each tick:
    1. Read: gather kernel-weighted influence per entity per channel
    2. Growth: apply growth function → raw deltas
    3. Conservation: normalize deltas so sum is zero per channel
    4. Apply: write current + deltas to write buffer, floor clamp, swap
    5. Bookkeeping: tick counter, energy totals, observers

The engine has a per-tick time budget. If exceeded, the tick is skipped
and logged. The field degrades gracefully rather than starving the SNN.
"""

import logging
import threading
import time
from typing import Callable, List, Optional

import numpy as np

from lenia.channels import ChannelRegistry
from lenia.config import LeniaConfig
from lenia.field import FieldStore
from lenia.functions import evaluate
from lenia.kernel import KernelComputer

logger = logging.getLogger(__name__)


class UpdateEngine:
    """The Lenia update loop.

    Not started automatically — the kill switch controls lifecycle.
    """

    def __init__(
        self,
        config: LeniaConfig,
        field_store: FieldStore,
        kernel_computer: KernelComputer,
        channel_registry: ChannelRegistry,
    ):
        self._config = config
        self._field = field_store
        self._kernel = kernel_computer
        self._registry = channel_registry

        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._tick_interval = 0.0  # set when started, derived from SNN step rate

        # Observers called after each tick
        self._post_tick_observers: List[Callable[[np.ndarray], None]] = []

        # Stats
        self._ticks_completed = 0
        self._ticks_skipped = 0
        self._last_tick_ms = 0.0

    @property
    def running(self) -> bool:
        return self._running

    @property
    def ticks_completed(self) -> int:
        return self._ticks_completed

    @property
    def ticks_skipped(self) -> int:
        return self._ticks_skipped

    def register_post_tick(self, callback: Callable[[np.ndarray], None]):
        """Register observer called after each tick with the new field state."""
        self._post_tick_observers.append(callback)

    def start(self, snn_step_interval_s: float = 1.0):
        """Start the update loop in a daemon thread.

        Args:
            snn_step_interval_s: Estimated seconds between SNN steps.
                Field tick interval = snn_step_interval_s / field_ticks_per_snn_step.
        """
        if self._running:
            return

        self._tick_interval = (
            snn_step_interval_s / self._config.field_ticks_per_snn_step
        )
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="lenia-engine"
        )
        self._thread.start()
        logger.info(
            "Lenia engine started: %.1fms tick interval, %d ticks/SNN step",
            self._tick_interval * 1000,
            self._config.field_ticks_per_snn_step,
        )

    def stop(self, timeout: float = 5.0):
        """Stop the update loop gracefully."""
        if not self._running:
            return
        self._running = False
        if self._thread:
            self._thread.join(timeout=timeout)
            self._thread = None
        logger.info(
            "Lenia engine stopped: %d ticks completed, %d skipped",
            self._ticks_completed, self._ticks_skipped,
        )

    def _loop(self):
        """Main loop — runs in daemon thread."""
        while self._running:
            t0 = time.monotonic()
            try:
                self._tick()
                self._ticks_completed += 1
            except Exception:
                logger.exception("Lenia tick failed")
                self._ticks_skipped += 1

            elapsed_ms = (time.monotonic() - t0) * 1000
            self._last_tick_ms = elapsed_ms

            if elapsed_ms > self._config.tick_time_budget_ms:
                logger.warning(
                    "Lenia tick exceeded budget: %.1fms > %.1fms",
                    elapsed_ms, self._config.tick_time_budget_ms,
                )
                self._ticks_skipped += 1
                # Don't sleep — we're already behind
                continue

            sleep_time = self._tick_interval - (elapsed_ms / 1000)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _tick(self):
        """Execute one Lenia update tick (5 phases)."""
        n_entities = self._field.entity_count
        n_channels = self._field.channel_count
        channel_ids = self._registry.channel_ids

        if n_entities == 0 or n_channels == 0:
            return

        # Phase 1 — Read: compute kernel-weighted influence
        current = self._field.read_buffer()
        influences = self._kernel.compute_all_channels(current)

        # Phase 2 — Growth: apply growth functions → raw deltas
        deltas = np.zeros((n_entities, n_channels), dtype=np.float64)

        for cid in channel_ids:
            c_idx = self._registry.channel_index(cid)
            growth_spec = self._registry.growth_function(cid)
            influence_arr = influences[cid]

            # Vectorized: apply growth function to the whole influence array
            raw_deltas = evaluate(growth_spec, influence_arr)

            # Cross-channel modulation: scale by total local field energy
            # (nodes with more total energy respond more strongly)
            local_energy = current.sum(axis=1)
            max_energy = local_energy.max() if local_energy.max() > 0 else 1.0
            energy_factor = 0.5 + 0.5 * (local_energy / max_energy)
            raw_deltas = raw_deltas * energy_factor

            deltas[:, c_idx] = raw_deltas

        # Phase 3 — Conservation: normalize deltas so sum is zero per channel
        for c_idx in range(n_channels):
            col_sum = deltas[:, c_idx].sum()
            if abs(col_sum) > 1e-15:
                deltas[:, c_idx] -= col_sum / n_entities

        # Phase 4 — Apply: write to write buffer, floor clamp, swap
        target = self._field.write_buffer()
        np.add(current, deltas, out=target)
        np.clip(target, self._config.field_floor, None, out=target)
        self._field.swap()

        # Phase 5 — Bookkeeping
        self._field.tick()

        # Update channel energy totals
        new_state = self._field.read_buffer()
        for cid in channel_ids:
            c_idx = self._registry.channel_index(cid)
            self._registry.update_energy(cid, float(new_state[:, c_idx].sum()))

        # Notify observers
        for observer in self._post_tick_observers:
            try:
                observer(new_state)
            except Exception:
                logger.exception("Post-tick observer failed")

    def run_ticks(self, count: int):
        """Run N ticks synchronously (for testing / single-threaded use)."""
        for _ in range(count):
            self._tick()
            self._ticks_completed += 1

    def status(self) -> dict:
        """Return engine status dict."""
        return {
            "running": self._running,
            "ticks_completed": self._ticks_completed,
            "ticks_skipped": self._ticks_skipped,
            "last_tick_ms": round(self._last_tick_ms, 2),
            "tick_interval_ms": round(self._tick_interval * 1000, 2),
            "field_energy": self._field.total_energy(),
        }
