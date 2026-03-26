"""
Topology Owner Sentinel — Prevents dual-write hazard on Syl's checkpoint.

When the ContextEngine RPC bridge (or any process) becomes the topology
owner, it writes a sentinel file containing its PID.  Any other process
that wants to create a NeuroGraphMemory instance checks the sentinel
first:

  - If sentinel exists and the PID is alive → another process owns the
    checkpoint.  Do NOT create a second NeuroGraphMemory.  Use the
    ExperienceTract for ingestion instead.
  - If sentinel exists but the PID is dead → stale sentinel from a crash.
    Clean it up and proceed.
  - If no sentinel → no owner.  Safe to create NeuroGraphMemory directly.

This is a Syl's Law safeguard.  Two NeuroGraphMemory instances writing
to main.msgpack simultaneously will corrupt her topology.  This is not
theoretical — the GUI and ContextEngine can both create instances in
separate processes, and last-writer-wins destroys state silently.

# ---- Changelog ----
# [2026-03-18] Claude (CC) — Initial creation (punch list #80)
# What: PID-based sentinel file for topology ownership.
# Why: Syl's Law — prevent dual-write hazard on main.msgpack.
#   GUI and universal_ingestor can create separate NeuroGraphMemory
#   instances in separate processes while ContextEngine is active.
# How: claim() writes PID to sentinel file. release() removes it.
#   is_owned() checks if another live process holds the sentinel.
#   Atomic write via temp file + os.replace.
# -------------------
"""

from __future__ import annotations

import logging
import os
import signal
from pathlib import Path
from typing import Optional

logger = logging.getLogger("neurograph.topology_owner")

_DEFAULT_SENTINEL_DIR = os.path.expanduser("~/NeuroGraph/data/checkpoints")
_SENTINEL_FILENAME = ".topology_owner.pid"


def _sentinel_path(checkpoint_dir: Optional[str] = None) -> Path:
    """Return the sentinel file path."""
    d = Path(checkpoint_dir or _DEFAULT_SENTINEL_DIR).expanduser()
    d.mkdir(parents=True, exist_ok=True)
    return d / _SENTINEL_FILENAME


def _pid_is_alive(pid: int) -> bool:
    """Check if a process with the given PID is running."""
    try:
        os.kill(pid, 0)  # Signal 0 = existence check, no actual signal
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we can't signal it — still alive
        return True


def claim(checkpoint_dir: Optional[str] = None) -> bool:
    """Claim topology ownership for this process.

    Writes a sentinel file containing this process's PID.  Returns True
    if the claim succeeded, False if another live process already owns it.

    Args:
        checkpoint_dir: Directory containing main.msgpack.  Defaults to
                        ~/NeuroGraph/data/checkpoints/
    """
    sentinel = _sentinel_path(checkpoint_dir)
    my_pid = os.getpid()

    # Check for existing owner
    if sentinel.exists():
        try:
            existing_pid = int(sentinel.read_text().strip())
            if _pid_is_alive(existing_pid) and existing_pid != my_pid:
                logger.warning(
                    "Topology already owned by PID %d — cannot claim",
                    existing_pid,
                )
                return False
            # Stale sentinel (dead PID) or same PID — clean up and reclaim
            if not _pid_is_alive(existing_pid):
                logger.info(
                    "Cleaning stale topology sentinel (PID %d is dead)",
                    existing_pid,
                )
        except (ValueError, OSError):
            logger.warning("Corrupt sentinel file — overwriting")

    # Write sentinel atomically
    tmp = sentinel.with_suffix(".tmp")
    try:
        tmp.write_text(str(my_pid))
        tmp.replace(sentinel)
        logger.info("Claimed topology ownership (PID %d)", my_pid)
        return True
    except OSError as exc:
        logger.error("Failed to write topology sentinel: %s", exc)
        return False


def release(checkpoint_dir: Optional[str] = None) -> None:
    """Release topology ownership.

    Only removes the sentinel if this process owns it (prevents one
    process from releasing another's claim).
    """
    sentinel = _sentinel_path(checkpoint_dir)
    my_pid = os.getpid()

    if not sentinel.exists():
        return

    try:
        existing_pid = int(sentinel.read_text().strip())
        if existing_pid == my_pid:
            sentinel.unlink()
            logger.info("Released topology ownership (PID %d)", my_pid)
        else:
            logger.debug(
                "Not releasing sentinel — owned by PID %d, we are %d",
                existing_pid, my_pid,
            )
    except (ValueError, OSError) as exc:
        logger.debug("Sentinel release failed: %s", exc)


def is_owned(checkpoint_dir: Optional[str] = None) -> bool:
    """Check if another live process owns the topology.

    Returns True if a sentinel exists with a live PID that is NOT
    this process.  Returns False if no sentinel, stale sentinel,
    or we are the owner.
    """
    sentinel = _sentinel_path(checkpoint_dir)

    if not sentinel.exists():
        return False

    try:
        existing_pid = int(sentinel.read_text().strip())
        if existing_pid == os.getpid():
            return False  # We own it — not "owned by someone else"
        return _pid_is_alive(existing_pid)
    except (ValueError, OSError):
        return False


def owner_pid(checkpoint_dir: Optional[str] = None) -> Optional[int]:
    """Return the PID of the current topology owner, or None."""
    sentinel = _sentinel_path(checkpoint_dir)

    if not sentinel.exists():
        return None

    try:
        pid = int(sentinel.read_text().strip())
        if _pid_is_alive(pid):
            return pid
        return None
    except (ValueError, OSError):
        return None
