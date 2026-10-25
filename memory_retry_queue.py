# ---- Changelog ----
# [2026-10-25] Claude Code (Opus 5.5; LANE B syl-stage1, #920 adapted) — an item at max_attempts is QUARANTINED, not lost
# What: drain() appends each item it would drop to "<path>.quarantine.msgpack" (append-only, length-prefixed msgpack
#       frames: target_id, content, attempts, dropped_at, last_error) before dropping it from the queue; the INFO log
#       names the file. A failed quarantine append keeps the item in the queue (never lost) and logs a WARNING.
# Why:  laptop #920 (proven live 2026-10-23): bounded retry -> durable quarantine, never silent loss. Here the queue
#       held 2,377 of Syl's turn memories (attempts 0) on 2026-10-24 while ng_embed's import was broken; three failed
#       drains would have discarded them.
# How:  the bounded, non-cyclic behaviour is unchanged (an item is still never retried past max_attempts).
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
import os, msgpack, logging, struct, time
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

    @property
    def quarantine_path(self) -> str:
        return self.path + ".quarantine.msgpack"

    def _quarantine(self, item: Dict[str, Any], last_error) -> bool:
        """Append one length-prefixed msgpack frame; fsync. True on success. Never raises."""
        try:
            frame = msgpack.packb({"target_id": item.get("target_id"), "content": item.get("content"),
                                   "attempts": item.get("attempts"), "dropped_at": time.time(),
                                   "last_error": last_error}, use_bin_type=True)
            with open(self.quarantine_path, "ab") as f:
                f.write(struct.pack(">I", len(frame)) + frame)
                f.flush()
                os.fsync(f.fileno())
            return True
        except Exception as e:
            logger.warning("retry-queue quarantine append failed for %s (kept in queue): %s",
                           item.get("target_id"), e)
            return False

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
            last_error = None
            try:
                ok = bool(attempt(item))
            except Exception as e:
                last_error = repr(e)[:500]
                logger.debug("retry attempt raised (non-fatal): %s", e)
            if ok:
                succeeded += 1
            elif item["attempts"] < self.max_attempts:
                survivors.append(item)
            elif self._quarantine(item, last_error):
                logger.info("retry-queue quarantined %s after %d attempts (%s)",
                            item["target_id"], item["attempts"], self.quarantine_path)
            else:
                survivors.append(item)  # could not quarantine: keep it rather than lose it
        self._items = survivors + rest   # unprocessed items retained for next pulse
        self._save()
        return succeeded
