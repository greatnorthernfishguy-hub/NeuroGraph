# ---- Changelog ----
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
"""Bounded, non-cyclic retry-queue for failed pass-2 concept extractions (#297).

Non-cyclic by construction: a drain pass NEVER re-enters extraction during the
same pass, and any item reaching max_attempts is DROPPED (logged), never re-queued.
The explicit guard against the wire->absorb->extract OOM recursion class (spec §6.1)."""
import os, msgpack, logging
from typing import Callable, Dict, Any, List
logger = logging.getLogger(__name__)

class RetryQueue:
    def __init__(self, path: str, max_attempts: int = 3):
        self.path = path
        self.max_attempts = max_attempts
        self._items: List[Dict[str, Any]] = self._load()

    def _load(self) -> List[Dict[str, Any]]:
        try:
            with open(self.path, "rb") as f:
                return msgpack.unpackb(f.read(), raw=False) or []
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

    def enqueue(self, target_id: str, content: str):
        if any(i["target_id"] == target_id for i in self._items):
            return  # dedup
        self._items.append({"target_id": target_id, "content": content, "attempts": 0})
        self._save()

    def drain(self, attempt: Callable[[Dict[str, Any]], bool], limit=None) -> int:
        """One bounded pass over up to `limit` items (oldest first; None = all).
        attempt(item)->bool. Succeeded items removed; items at max_attempts dropped.
        Never re-queues within the pass. Returns #succeeded."""
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
            elif item["attempts"] < self.max_attempts:
                survivors.append(item)
            else:
                logger.info("retry-queue dropping %s after %d attempts",
                            item["target_id"], item["attempts"])
        self._items = survivors + rest   # unprocessed items retained for next pulse
        self._save()
        return succeeded
