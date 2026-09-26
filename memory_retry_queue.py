# ---- Changelog ----
# [2026-09-25] Claude Code (kimi-k2.7-code) — Packet 214 Q-1 corrective build.
# What: (1) drain() rotates survivors to the back of the queue instead of the
#   front (Q-1 starvation fix); (2) backlog() exposes pending_count and the
#   age of the oldest queued item so starved front-of-queue items are visible
#   even when attempts has not incremented; (3) the long-held warning is
#   rate-limited using the same pattern as NGEmbed._extraction_warn_due().
# Why:  Q-1: survivors at the front starved every later item, and the always-
#   succeeding item could be stuck behind a wall of failures. Backlog metric:
#   long_held_count gated on attempts cannot see items stuck behind others that
#   never get attempted. Warning rate-limit: per-drain warnings flood the log
#   on every turn when a long outage keeps items past max_attempts.
# How:  self._items = rest + survivors. enqueue() stamps created_at for age.
#   backlog() computes oldest_age from min(created_at). _long_held_warn_due()
#   tracks cumulative long-held items and the last warning timestamp (default
#   CC_EXTRACT_WARN_INTERVAL_S=60s). Cites Packet 214, chief-p226-ruling-001.
# -------------------
# [2026-06-05] CC (Sonnet 4.6) — #297 review fixes: atomic _save(), drain limit param, import os
# What: _save() now writes to .tmp then os.replace() — atomic, no zero-byte corruption on kill.
#       drain() gains optional limit= param — pulse processes only oldest N per pass.
#       import os added at top of file.
# Why: Kill mid-write previously left a zero-byte file; _load() silently returns [] losing all retries.
#      Long outages could queue many items; unbounded drain stalls the single-threaded sidecar.
# How: os.replace() is atomic on same filesystem (Linux). drain slices _items[:limit], retains rest.
# [2026-06-05] CC (Sonnet 4.6) — #297: bounded non-cyclic retry-queue for failed pass-2 extractions
# What: RetryQueue class — msgpack-backed, dedup by target_id, bounded by max_attempts.
# Why: Failed _conversational_dual_pass calls silently dropped turn memories forever.
#      This queue retries them on the autonomic pulse, bounded so failed items are
#      dropped (not cycled forever), and non-cyclic by construction — the drain uses
#      the CORE extraction function that never re-enqueues (spec §6.1).
# How: enqueue() deduplicates by target_id; drain() does one pass per call, incrementing
#      attempts, keeping survivors below max_attempts, dropping the rest. Persisted via
#      msgpack to survive process restarts.
# -------------------
"""Cyclic retry-queue for failed pass-2 concept extractions (#297).

Non-cyclic by construction: a drain pass NEVER re-enters extraction during the
same pass. Items are retained until success; no item is ever dropped for
hitting max_attempts. The explicit guard against the wire->absorb->extract OOM
recursion class (spec §6.1)."""
import os, msgpack, logging, time
from typing import Callable, Dict, Any, List
logger = logging.getLogger(__name__)

class RetryQueue:
    def __init__(self, path: str, max_attempts: int = 3):
        self.path = path
        self.max_attempts = max_attempts
        self._items: List[Dict[str, Any]] = self._load()
        # Rate-limit the long-held warning. These counters cover every item that
        # crosses max_attempts, regardless of whether it is currently still queued.
        self._last_long_held_warn: float = 0.0
        self._long_held_at_last_warn: int = 0
        self._long_held_total: int = 0

    def _load(self) -> List[Dict[str, Any]]:
        try:
            with open(self.path, "rb") as f:
                items = msgpack.unpackb(f.read(), raw=False) or []
                # Backfill created_at for items written by older builds;
                # missing values report age 0 (non-fatal).
                for it in items:
                    if "created_at" not in it:
                        it["created_at"] = time.time()
                return items
        except FileNotFoundError:
            return []
        except Exception as e:
            logger.warning("retry-queue load failed: %s", e); return []

    def _save(self):
        try:
            tmp = self.path + ".tmp"
            with open(tmp, "wb") as f:
                f.write(msgpack.packb(self._items))
            os.replace(tmp, self.path)  # atomic on same filesystem (Linux)
        except Exception as e:
            logger.warning("retry-queue save failed: %s", e)

    def pending_count(self) -> int:
        return len(self._items)

    def long_held_count(self, threshold: int = None) -> int:
        """Count items held longer than `threshold` attempts (default: max_attempts)."""
        threshold = threshold if threshold is not None else self.max_attempts
        return sum(1 for item in self._items if item.get("attempts", 0) > threshold)

    def oldest_age(self) -> float:
        """Age in seconds of the oldest still-queued item, or 0.0 if empty."""
        if not self._items:
            return 0.0
        now = time.time()
        oldest = min(
            item.get("created_at", now) for item in self._items
        )
        return max(0.0, now - oldest)

    def backlog(self) -> Dict[str, Any]:
        """Surface total pending and starved-front queue state.

        long_held_count alone is blind to items whose attempts field never
        increments while stuck behind other items. pending_count and
        oldest_age make those starved items visible.
        """
        return {
            "pending_count": self.pending_count(),
            "oldest_age": self.oldest_age(),
            "long_held_count": self.long_held_count(),
        }

    def _long_held_warn_due(self) -> int:
        """Rate-limit the long-held warning. Returns the number of long-held
        item-observations since the last emitted warning when a warning is due
        (>=1, truthy), else 0. Interval via CC_EXTRACT_WARN_INTERVAL_S
        (default 60s)."""
        interval = float(os.environ.get("CC_EXTRACT_WARN_INTERVAL_S", "60"))
        now = time.monotonic()
        last = getattr(self, "_last_long_held_warn", 0.0)
        if now - last >= interval:
            since = self._long_held_total - self._long_held_at_last_warn
            self._last_long_held_warn = now
            self._long_held_at_last_warn = self._long_held_total
            return max(1, since)
        return 0

    def enqueue(self, target_id: str, content: str):
        if any(i["target_id"] == target_id for i in self._items):
            return  # dedup
        self._items.append({
            "target_id": target_id,
            "content": content,
            "attempts": 0,
            "created_at": time.time(),
        })
        self._save()

    def drain(self, attempt: Callable[[Dict[str, Any]], bool], limit=None) -> int:
        """One bounded pass over up to `limit` items (oldest first; None = all).
        attempt(item)->bool. Succeeded items removed; survivors are retained
        and rotated to the back of the queue until success. Never re-queues
        within the pass. Returns #succeeded."""
        to_process = self._items if limit is None else self._items[:limit]
        rest = [] if limit is None else self._items[limit:]
        survivors: List[Dict[str, Any]] = []
        succeeded = 0
        for item in to_process:
            item["attempts"] += 1
            ok = False
            try:
                ok = bool(attempt(item))
            except Exception as e:
                logger.debug("retry attempt raised (non-fatal): %s", e)
            if ok:
                succeeded += 1
            else:
                # Retain until success (Q-1 / #297). Never drop for max_attempts.
                survivors.append(item)
                if item["attempts"] > self.max_attempts:
                    self._long_held_total += 1
        # Rotate survivors to the back so later items are not starved.
        self._items = rest + survivors
        self._save()
        _since = self._long_held_warn_due()
        if _since:
            logger.warning(
                "retry-queue: %d item(s) held past max_attempts without success "
                "(+%d since last warn); oldest queued age=%.0fs",
                self.long_held_count(), _since, self.oldest_age(),
            )
        return succeeded
