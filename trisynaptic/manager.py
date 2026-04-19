# ---- Changelog ----
# [2026-04-19] Claude Code (Opus 4.7, 1M) — TriSyn Phase 1 initial
# What: In-process thread that watches _CONCEPT_QUEUE depth, spawns a
#   worker subprocess via systemd-run when trisyn_high_water is crossed.
#   Replaces the in-process _concept_extraction_pulse_loop.
# Why: Blocking TID calls inside NG's main process were starving the
#   event loop and unable to keep up with fast-path drain rate. This
#   manager doesn't block — it spawns isolated workers instead.
# How: Daemon thread on _PULSE_INTERVAL_S cadence. Reads tunable values
#   live from _memory.graph.config (same pattern CES uses). On high-water
#   crossing, drains the queue into a handoff file and launches the worker
#   under systemd-run for kernel-enforced resource isolation.
#   Phase 1: single-worker-at-a-time, unconditional spawn only. Gated/idle
#   modes and concurrent helpers are Phase 3 polish per spec §10.
# -------------------
"""TriSyn manager — in-process thread that orchestrates concept-extraction workers."""
from __future__ import annotations

import logging
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from trisynaptic.handoff import (
    HANDOFF_DIR,
    HANDOFF_PREFIX,
    list_orphan_handoffs,
    read_handoff,
    write_atomic,
)


logger = logging.getLogger("trisyn.manager")

# Cadence for the manager pulse loop. Phase 1 default; may become tunable.
_PULSE_INTERVAL_S = 10.0

# Path to NG's venv python (used to launch worker under systemd-run).
# Falls back to current interpreter if venv not detected.
_NG_VENV_PY = "/home/josh/NeuroGraph/venv/bin/python"


class TrisynapticManager:
    """Watches the concept-extraction backlog and spawns workers on overflow.

    Args:
        memory: NeuroGraphMemory singleton — provides .graph.config for
                live tunable reads.
        queue:  Reference to neurograph_rpc._CONCEPT_QUEUE. Shared mutable
                list; manager pops entries into the worker's handoff.
    """

    def __init__(self, memory: Any, queue: List[Dict[str, Any]]) -> None:
        self._memory = memory
        self._queue = queue
        self._shutdown = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._worker_scope: Optional[str] = None
        self._worker_handoff: Optional[Path] = None
        self._worker_spawned_at: float = 0.0
        self._last_worker_exit_ts: float = 0.0

    # Grace period after a worker spawn before we're allowed to call
    # `systemctl is-active` — the scope takes a moment to register in
    # systemd, and an early check returns "inactive" for a running scope.
    _SCOPE_REGISTER_GRACE_S = 3.0

    # ----- public lifecycle -------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._shutdown.clear()
        self._thread = threading.Thread(
            target=self._pulse_loop,
            name="TriSynManager",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "TrisynapticManager started (pulse=%.0fs)", _PULSE_INTERVAL_S,
        )

        # Crash-recovery: requeue any leftover handoffs (worker crash or
        # NG restart orphans). Per Josh 2026-04-18: "respawn/requeue, log
        # it regardless." Requeueing is the safer half at bootstrap time —
        # no race with the worker we're about to spawn.
        self._requeue_orphans()

    def _requeue_orphans(self) -> None:
        orphans = list_orphan_handoffs()
        if not orphans:
            return
        requeued_total = 0
        failed_paths: List[Path] = []
        for path in orphans:
            try:
                payload = read_handoff(path)
                entries = payload.get("entries", [])
                for entry in entries:
                    self._queue.append(entry)
                requeued_total += len(entries)
                try:
                    path.unlink()
                except OSError:
                    pass
            except Exception:
                logger.exception(
                    "Failed to requeue orphan handoff %s — renaming for triage",
                    path,
                )
                try:
                    failed_path = path.with_suffix(".failed.json")
                    path.rename(failed_path)
                    failed_paths.append(failed_path)
                except OSError:
                    pass
        logger.warning(
            "Orphan handling: requeued %d entries from %d handoff(s); %d unparseable",
            requeued_total, len(orphans) - len(failed_paths), len(failed_paths),
        )

    def stop(self, timeout: float = 5.0) -> None:
        self._shutdown.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        logger.info("TrisynapticManager stopped")

    # ----- main loop --------------------------------------------------

    def _pulse_loop(self) -> None:
        while not self._shutdown.is_set():
            try:
                self._tick()
            except Exception:
                logger.exception("TriSyn manager tick failed")
            self._shutdown.wait(timeout=_PULSE_INTERVAL_S)

    def _tick(self) -> None:
        config = self._memory.graph.config
        high_water = int(config.get("trisyn_high_water", 1500))
        cooldown_s = float(config.get("trisyn_cooldown_s", 60))

        # If a worker is in flight, wait for it to finish.
        if self._worker_scope is not None:
            if self._worker_is_running():
                return
            self._finalize_worker_exit()

        # Cooldown window after a worker exit.
        if time.time() - self._last_worker_exit_ts < cooldown_s:
            return

        if len(self._queue) >= high_water:
            self._spawn_worker(config)

    # ----- spawn path -------------------------------------------------

    def _spawn_worker(self, config: Dict[str, Any]) -> None:
        max_runtime_s = int(config.get("trisyn_max_runtime_s", 1800))
        max_workers = int(config.get("trisyn_max_workers_uncond", 1))
        tid_fail_exit = int(config.get("trisyn_tid_fail_exit", 5))
        exit_water = int(config.get("trisyn_exit_water", 200))

        # Drain most of the queue into a batch; leave exit_water as buffer.
        # Fast-path drain keeps populating the queue during the worker's run;
        # next tick picks up whatever has re-accumulated.
        batch_size = max(1, len(self._queue) - exit_water)
        batch = [self._queue.pop(0)
                 for _ in range(min(batch_size, len(self._queue)))]
        if not batch:
            return

        ts = int(time.time())
        handoff_path = HANDOFF_DIR / f"{HANDOFF_PREFIX}{ts}.json"
        payload = {
            "spawn_mode": "unconditional",  # Phase 1: only mode
            "config": {
                "trisyn_max_workers": max_workers,
                "trisyn_tid_fail_exit": tid_fail_exit,
                "trisyn_max_runtime_s": max_runtime_s,
                "trisyn_exit_water": exit_water,
            },
            "entries": batch,
        }

        try:
            write_atomic(handoff_path, payload)
        except Exception:
            logger.exception("Failed to write handoff — re-queueing batch")
            self._requeue_front(batch)
            return

        scope_name = f"trisyn-{ts}.scope"
        python_exe = _NG_VENV_PY if Path(_NG_VENV_PY).exists() else sys.executable
        cmd = [
            "systemd-run",
            "--user",
            "--scope",
            f"--unit={scope_name}",
            "--property=MemoryMax=1G",
            "--property=CPUQuota=50%",
            f"--setenv=TRISYN_HANDOFF={handoff_path}",
            python_exe,
            "-m",
            "trisynaptic.worker",
        ]
        try:
            subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
            )
            self._worker_scope = scope_name
            self._worker_handoff = handoff_path
            self._worker_spawned_at = time.time()
            logger.info(
                "Spawned worker: scope=%s, entries=%d, queue_remaining=%d",
                scope_name, len(batch), len(self._queue),
            )
        except Exception:
            logger.exception("Failed to spawn worker — re-queueing batch")
            self._requeue_front(batch)
            try:
                handoff_path.unlink()
            except OSError:
                pass

    def _requeue_front(self, batch: List[Dict[str, Any]]) -> None:
        """Put a batch back at the front of the queue, preserving order."""
        for entry in reversed(batch):
            self._queue.insert(0, entry)

    # ----- worker status / exit handling ------------------------------

    def _worker_is_running(self) -> bool:
        if not self._worker_scope:
            return False
        # Scope-register grace: systemd-run returns before the scope is
        # fully registered. Treat the worker as running during the grace
        # window regardless of is-active's verdict.
        if time.time() - self._worker_spawned_at < self._SCOPE_REGISTER_GRACE_S:
            return True
        try:
            result = subprocess.run(
                ["systemctl", "--user", "is-active", self._worker_scope],
                capture_output=True, text=True, timeout=5,
            )
            # is-active returns "active" for running, "inactive" / "failed" otherwise
            return result.stdout.strip() == "active"
        except Exception:
            return False

    def _finalize_worker_exit(self) -> None:
        logger.info("Worker scope ended: %s", self._worker_scope)
        # Handoff file still present = worker exited non-zero (worker deletes on success)
        if self._worker_handoff is not None and self._worker_handoff.exists():
            failed_path = self._worker_handoff.with_suffix(".failed.json")
            try:
                self._worker_handoff.rename(failed_path)
                logger.warning(
                    "Worker failed — handoff preserved at %s for triage",
                    failed_path,
                )
            except OSError:
                pass
        self._worker_scope = None
        self._worker_handoff = None
        self._last_worker_exit_ts = time.time()
