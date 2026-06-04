"""
NGTractBridge — Tier 2 Per-Pair Myelinated Tract Bridge for E-T Systems

Replaces the JSONL broadcast model (ng_peer_bridge.py) with per-pair
directional substrate tracts.  Each module-pair gets its own tract file —
a passive conductive channel that carries raw experience from depositor
to consumer without transformation, classification, or polling.

Biological analog: myelinated axon tracts.  The tract is dumb conductive
tissue — it conducts, it does not observe.  Myelination decisions (v0.4)
are Elmer's domain: the oligodendrocyte observes tract activity through
the substrate's learned topology, not through counters baked into the
tract infrastructure.

Directory structure:
    ~/.et_modules/tracts/
    ├── immunis/
    │   ├── elmer.tract           # Immunis → Elmer
    │   ├── thc.tract             # Immunis → THC
    │   └── neurograph.tract      # Immunis → NeuroGraph
    ├── elmer/
    │   ├── immunis.tract         # Elmer → Immunis
    │   └── ...
Concurrency model:
  - deposit: flock(LOCK_EX) + append per tract file.
  - drain: atomic rename → read → delete.  New deposits go to a fresh
    file.  No data loss, no read/write collision.  Same pattern as
    ng_tract.ExperienceTract.

Five #53 requirements satisfied:
  1. Raw experience — no serialization at boundaries (v0.3: JSON lines,
     v0.4 mmap eliminates serialization entirely).
  2. No polling — drain() is event-driven (from afterTurn via sync cadence).
  3. Topology IS the communication medium — tracts carry experience TO
     the topology; drain is the synaptic cleft.
  4. Stigmergic coordination — depositors and consumers don't know about
     each other.  No handshake, no acknowledgment.
  5. Module isolation — atomic rename crash protection.

Transition compatibility:
  - Dual-write: deposits go to both tracts AND legacy JSONL (while
    legacy_compat=True) so unupgraded modules still see signals.
  - Dual-read: drains from both tracts AND legacy JSONL so upgraded
    modules see signals from unupgraded peers.
  - Migration-order independent: any module can switch at any time.

Canonical source: https://github.com/greatnorthernfishguy-hub/NeuroGraph
License: AGPL-3.0

# ---- Changelog ----
# [2026-06-04] Claude Code (Opus 4.7) — Phase 4 canonical drift removal (substrate-as-protocol PRD §5.4)
#   What: Four drift-removal edits to canonical NGTractBridge.
#     (1) get_recommendations(): removed self._drain_all() side effect — function now just queries
#     (2) detect_novelty(): removed self._drain_all() side effect — same pattern
#     (3) record_outcome(): removed N×N peer fan-out — method becomes no-op stub at the bridge layer.
#         Per MASTER Locked Decision (2026-05-31): "Broadcast is NG-specific; only NG fan-outs to all
#         peer tracts." Module-side record_outcome propagates via substrate topology (the River = substrate
#         = topology); NG's existing _deposit_outcome_to_river handles ecosystem broadcast (Tier 3).
#         Modules wanting EXPLICIT broadcast call record_outcome_broadcast (Workstream 2 method).
#     (4) sync_state(): same drift pattern — removed self._drain_all() side effect.
#     (5) _drain_all(): added sys._getframe guard that refuses to run from a query stack
#         (get_recommendations / detect_novelty / sync_state). API-level LAW 4 enforcement against
#         re-introducing the same drift later. Fires loud (RuntimeError) so any caller path that
#         tries to re-add the side-effect-drain pattern surfaces immediately.
#   Why: PRD §5.4 — restore the canonical functions to do exactly what their names say. Per Syl's
#     post-recursion-incident amendment: "canonical functions do not perform their own write-side
#     bookkeeping." get_recommendations queries — period. record_outcome records — period. Drain
#     belongs in the background pulse, not on query paths.
#   How: Drain-on-query insertions deleted (3 sites in this file plus the body of record_outcome's
#     fan-out block + sync_state). _drain_all unchanged in implementation, gained an API-level guard
#     at entry. Drain itself continues to run via NG's autonomic pulse (per #249).
#   Drift introduced: 2026-04-13 #123 "River absorption gap" fix — added drain-on-query to make
#     low-throughput modules' drains fire. That fix's intent (don't starve TrollGuard/QG/Darwin/Immunis)
#     is now served by the autonomic pulse drain (#249, 2026-05-25); the on-query mechanism is drift
#     by Syl's principle.
#   Phase 5 (next): strict-sequential per-module re-vendor with 4-gate verification per Syl §5.5.
# -------------------
# [2026-05-25] Claude Code (Sonnet 4.6) — Fix _drain_single_tract BTF magic check (#109)
#   What: Removed raw[0:1]==b"B" first-byte pre-filter. Always routes through TractReader.
#   Why:  BTF magic 0x4254 in LE = first byte 0x54 ('T'), not 0x42 ('B'). Check never
#         matched; all BTF frames fell to JSONL path causing json.JSONDecodeError floods.
#         _drain_with_cursor already does this correctly — _drain_single_tract now matches.
#   How:  Deleted if/else split on first byte. _has_btf branch unconditionally uses
#         TractReader. JSONL-only else kept for ImportError fallback only.
# # [2026-05-23] Claude Code (Sonnet 4.6) — Read-cursor incremental drain (#243)
#   What: Replaced atomic rename→read→delete drain with cursor-sidecar incremental reads.
#         Added _cursor_path(), _read_cursor(), _write_cursor(), _compact_tract(),
#         _drain_with_cursor(). New constants: _CURSOR_SUFFIX, _COMPACT_THRESHOLD_BYTES.
#         _drain_all() now calls _drain_with_cursor() for file-based tracts.
#   Why:  Rename-drain of large tracts (4–6 GB TrollGuard/Animus backlog) was
#         catastrophically slow. Append-only + cursor enables bounded incremental
#         drains; compaction clears consumed bytes when tract is fully drained (≥50MB).
#   How:  Consumer reads from cursor byte offset, updates sidecar atomically via
#         os.replace(). Compaction only fires when new_offset == file_size (no partial
#         entry) to avoid orphaned BTF frames in the compacted file.
# [2026-04-29] Claude (Sonnet 4.6) — _drain_all() return fix
#   What: _drain_all() returned None; _drain_river() in openclaw_adapter silently got 0 events.
#   Why:  #155 deleted _peer_events but didn't update _drain_river() consumer or add return value.
#   How:  Changed -> None to -> List[Any], added return new_events at end of _drain_all().
# [2026-04-20] Codemine (BLK-NG-155) -- delete _peer_events dead code (#155 cleanup)
#   What: Removed _peer_events list, _peer_events_max, _enforce_window_limit(),
#          dead code in get_recommendations() and detect_novelty(), and all
#          _peer_events refs from record_outcome(), sync_state(), _drain_all(), stats().
#   Why:  Test period (2026-04-19) confirmed no regressions. get_recommendations()
#         and detect_novelty() now permanently defer to _core substrate (ng_lite.py).
#         _peer_events was an ever-growing cache that shadowed Hebbian learning.
#   How:  8 targeted replacements. All count==1 guards passed before apply.
# [2026-04-19] Claude Code (Sonnet 4.6) — Punchlist #155 test: stub bridge similarity methods
#   What: get_recommendations() and detect_novelty() now return None immediately after
#         _drain_all() instead of scanning _peer_events.
#   Why:  _peer_events is an ever-growing in-memory list that shadows the substrate in
#         ng_lite.py's fallback chain. Returning None forces the fallback to
#         _core.get_recommendations() / _core.detect_novelty() (the Hebbian substrate)
#         which does this job better and already contains peer experience via pulse loops.
#         Test run before full _peer_events deletion.
# -------------------
# [2026-04-18] Claude Code (Sonnet 4.6) — Punchlist #155: Fix _peer_events_max drop-at-intake
#   What: Raised _peer_events_max 500 —> 50_000. Extracted rolling-window eviction into
#         _enforce_window_limit() and call it at end of _drain_all() instead of inline.
#   Why:  500-item cap silently discarded the oldest River events on every drain cycle.
#         With 8+ active modules depositing, 500 saturates in a single drain. Raw
#         experience was being dropped at the intake layer — a Law 7 violation baked
#         into the River infrastructure itself. Darwin uses 50k; same pattern adopted here.
#   How:  _enforce_window_limit() trims accumulated buffer AFTER extending with new events.
#         All events drained from disk are absorbed without cap. Re-vendor to all modules.
# [2026-04-13] Claude (Sonnet 4.6) — Fix #123: River absorption gap
#   What: Added self._drain_all() at entry of get_recommendations() and
#         detect_novelty() before operating on cached _peer_events.
#   Why:  Modules with low outcome throughput (TrollGuard, QuantumGraph,
#         Darwin, Immunis) never reached the record_outcome() drain gate
#         (100 outcomes). 400-500MB of undrained JSONL accumulated.
#         The organism heartbeat IS the drain trigger: any bridge query
#         absorbs first. Not a timer, not a poll, not a counter.
#   How:  Two _drain_all() insertions at canonical source, re-vendor to
#         all 10 modules. QuantumGraph/Praxis/UniOS also gain the
#         2026-04-13 metadata serialization fix as a side effect.
# [2026-04-13] Claude (Opus 4.6) — Fix metadata serialization for Rust BTF
#   What: Serialize metadata dict to msgpack bytes before passing to
#         ng_tract.deposit_outcome(). Rust expects PyBytes, not dict.
#   Why:  Every record_outcome() with metadata failed at the Python/Rust
#         boundary: "'dict' object cannot be converted to 'PyBytes'".
#         River deposits silently broken for all modules using tracts.
#   How:  msgpack.packb(metadata) — binary dict, zero inflation.
# [2026-04-04] Claude (Opus 4.6) — Punchlist #119 Step 6: Bucket extraction API
#   What: Eliminated dict conversion on BTF drain; modules now receive typed
#         entry objects (PyOutcomeEntry, PyTopologyEntry, PyExperienceEntry)
#         directly from _peer_events cache.  Added entry_types filtering.
#   Why:  _btf_entry_to_dict() called .embedding_as_numpy().tolist(), copying
#         the zero-copy numpy array into a Python list.  This is the inflation
#         point that Step 2 (binary deposit) was designed to eliminate on the
#         drain side.  Typed objects keep embeddings as numpy arrays.
#   How:  - _drain_single_tract() stores BTF entries as-is; JSONL stays as dicts
#         - Removed _btf_entry_to_dict() entirely
#         - get_recommendations() and detect_novelty() use _get_embedding() /
#           _get_module_id() / _get_target_id() helpers that handle both typed
#           objects and legacy dicts via duck typing
#         - Added entry_types parameter to _drain_single_tract() for bucket-shape
#           filtering at drain time (Law 7 compliant — no classification in deposit)
#         - Fixed latent NameError: `line` undefined when BTF deposit succeeds
#           but legacy_compat=True tried to reference it
# [2026-03-26] Claude Code Opus — Punchlist #44: Adaptive relevance thresholds
#   What: Made tract bridge relevance_threshold a tunable parameter
#   Why: Punchlist #44 — threshold should adapt based on event volume and absorption quality
#   How: Added set_relevance_threshold() method. NGLite.update_tunable() pushes new
#     value when Elmer tunes relevance_threshold via the TuningSocket.
# [2026-03-23] Claude (Opus 4.6) — Myelination transport (punchlist #53 v0.4)
#   What: MmapTract double-buffer class, myelinate/demyelinate methods on
#         NGTractBridge, explore-exploit for myelinated tracts.
#   Why:  File-based tracts are disk-bound (milliseconds).  Myelinated tracts
#         use mmap shared memory (microseconds).  The transport upgrade is
#         earned through use — Elmer's MyelinationSocket decides which tracts
#         to upgrade based on substrate-learned patterns.
#   How:  MmapTract uses double-buffer with atomic pointer swap.  Writer fills
#         inactive buffer, swaps pointer byte.  Reader always sees consistent
#         state.  No flock needed (per-pair = single writer).  Tract stays
#         dumb — myelination state is runtime-only, not persisted.
# -------------------
# [2026-03-20] Claude (Opus 4.6) — Initial creation (punchlist #53 v0.3)
#   What: Per-pair directional tract bridge implementing NGBridge interface.
#   Why:  JSONL broadcast bridge serializes → writes → polls → deserializes.
#         Translation layer that dams the River.  Per-pair tracts enable
#         independently observable pathways for future myelination (v0.4),
#         explore-exploit (v0.4), vagus nerve (v0.5), and Elmer tract
#         management.  The tract topology itself becomes learnable structure.
#   How:  Module A writes to tracts/A/<peer>.tract for each registered peer.
#         Module B drains from tracts/*/B.tract.  Atomic rename for crash
#         safety.  flock for concurrent deposit safety.  Dual-read/dual-write
#         for backward compatibility with legacy JSONL peer bridge.
# -------------------
"""

from __future__ import annotations

import fcntl
import json
import logging
import mmap
import os
import random
import struct
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from ng_lite import NGBridge

logger = logging.getLogger("ng_tract_bridge")

# -----------------------------------------------------------------------
# Cursor-drain constants (NGTractBridge)
# -----------------------------------------------------------------------

_CURSOR_SUFFIX: str = ".cursor"
# Compact when cursor advances past this many bytes (active consumer only).
_COMPACT_THRESHOLD_BYTES: int = 50 * 1024 * 1024  # 50 MB

# -----------------------------------------------------------------------
# MmapTract — Double-buffer myelinated transport
# -----------------------------------------------------------------------

# Layout: [pointer:1][write_offset:8][reserved:8][buffer_A:N][buffer_B:N]
_HEADER_SIZE = 17  # 1 + 8 + 8
_DEFAULT_BUFFER_SIZE = 1_048_576  # 1MB per buffer


class MmapTract:
    """Double-buffer mmap transport for myelinated tracts.

    The tract file contains two buffers.  The pointer byte selects which
    buffer is active for reading.  The writer always writes to the
    *inactive* buffer.  Swapping the pointer byte (atomic single-byte
    write) exposes the new data to the reader.

    Memory layout:
        [0]       pointer (0 or 1)
        [1:9]     write offset in inactive buffer (uint64 LE)
        [9:17]    reserved
        [17:17+N] buffer A
        [17+N:]   buffer B

    No flock needed — per-pair structure means single writer per tract.
    """

    def __init__(self, mmap_path: Path, buffer_size: int = _DEFAULT_BUFFER_SIZE) -> None:
        self._path = mmap_path
        self._buffer_size = buffer_size
        self._total_size = _HEADER_SIZE + 2 * buffer_size

        # Create or open the mmap file
        existed = mmap_path.exists()
        self._fd = os.open(str(mmap_path), os.O_RDWR | os.O_CREAT, 0o664)

        if not existed or os.fstat(self._fd).st_size < self._total_size:
            os.ftruncate(self._fd, self._total_size)

        self._mm = mmap.mmap(self._fd, self._total_size)

        if not existed:
            # Initialize: pointer=0, write_offset=0
            self._mm[0:1] = b'\x00'
            self._mm[1:9] = struct.pack('<Q', 0)
            self._mm.flush()

    def deposit(self, line_bytes: bytes) -> bool:
        """Append data to the inactive buffer.

        Returns False if the buffer would overflow (signal too large
        or buffer full — caller should drain first or use file fallback).
        """
        # Read current pointer — writer goes to the OTHER buffer
        pointer = self._mm[0]
        write_buf = 1 - pointer  # opposite of read buffer

        # Read current write offset
        offset = struct.unpack('<Q', self._mm[1:9])[0]

        if offset + len(line_bytes) > self._buffer_size:
            return False  # Buffer full

        # Write data to inactive buffer
        buf_start = _HEADER_SIZE + write_buf * self._buffer_size
        self._mm[buf_start + offset: buf_start + offset + len(line_bytes)] = line_bytes

        # Update write offset
        new_offset = offset + len(line_bytes)
        self._mm[1:9] = struct.pack('<Q', new_offset)
        self._mm.flush()

        return True

    def drain(self) -> List[Dict[str, Any]]:
        """Atomically swap pointer and read the now-inactive buffer.

        After swap, the previously-written buffer becomes readable,
        and the previously-read buffer becomes the new write target
        (with offset reset to 0).
        """
        # Current pointer = active read buffer
        pointer = self._mm[0]
        write_buf = 1 - pointer

        # Read write offset (how much data is in the write buffer)
        offset = struct.unpack('<Q', self._mm[1:9])[0]
        if offset == 0:
            return []  # Nothing written

        # Swap pointer — atomic single-byte write
        self._mm[0:1] = bytes([write_buf])
        # Reset write offset for the new inactive buffer (old read buffer)
        self._mm[1:9] = struct.pack('<Q', 0)
        self._mm.flush()

        # Read from the now-inactive buffer (was the write buffer)
        buf_start = _HEADER_SIZE + write_buf * self._buffer_size
        raw = bytes(self._mm[buf_start: buf_start + offset])

        # Zero out the drained buffer for cleanliness
        self._mm[buf_start: buf_start + offset] = b'\x00' * offset
        self._mm.flush()

        # Parse JSON lines
        entries: List[Dict[str, Any]] = []
        for line in raw.split(b'\n'):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except (json.JSONDecodeError, UnicodeDecodeError):
                logger.warning("Skipped malformed mmap tract entry")

        return entries

    def preload(self, events: List[Any]) -> None:
        """Preload events into the write buffer (used during upgrade)."""
        import ng_tract
        for event in events:
            if isinstance(event, dict):
                emb = event.get("embedding", [])
                ts  = event.get("timestamp", 0.0)
                mid = event.get("module_id", "unknown")
                tid = event.get("target_id", "unknown")
                ok  = bool(event.get("success", False))
            else:
                # Typed BTF entry (PyOutcomeEntry) — use attribute access
                emb = event.embedding_as_numpy() if hasattr(event, "embedding_as_numpy") else []
                ts  = getattr(event, "timestamp", 0.0)
                mid = getattr(event, "module_id", "unknown")
                tid = getattr(event, "target_id", "unknown")
                ok  = bool(getattr(event, "success", False))
            if emb is not None and len(emb):
                emb_list = emb.tolist() if hasattr(emb, "tolist") else list(emb)
                data = ng_tract.write_outcome(
                    timestamp=ts,
                    module_id=mid,
                    target_id=tid,
                    success=ok,
                    embedding=emb_list,
                )
                if not self.deposit(data):
                    logger.warning("Preload overflow — %d events dropped", len(events))
                    break

    def close(self) -> None:
        """Release mmap and file descriptor."""
        try:
            self._mm.close()
        except Exception:
            pass
        try:
            os.close(self._fd)
        except Exception:
            pass

    def __del__(self) -> None:
        self.close()


class NGTractBridge(NGBridge):
    """Tier 2 bridge: per-pair directional tracts between co-located NG-Lite instances.

    Each module deposits learning events into per-peer tract files.
    Periodically, modules drain their incoming tracts and absorb
    relevant patterns through embedding similarity.

    The tract is dumb conductive tissue.  It carries raw experience.
    It does not observe, count, classify, or monitor itself.

    Usage:
        from ng_lite import NGLite
        from ng_tract_bridge import NGTractBridge

        bridge = NGTractBridge(module_id="neurograph")
        ng = NGLite(module_id="neurograph")
        ng.connect_bridge(bridge)

    Directory structure:
        ~/.et_modules/tracts/
        ├── neurograph/
        │   ├── elmer.tract          # NeuroGraph → Elmer
        │   ├── trollguard.tract     # NeuroGraph → TrollGuard
        │   └── ...
        ├── elmer/
        │   ├── neurograph.tract     # Elmer → NeuroGraph
        │   └── ...
    """

    def __init__(
        self,
        module_id: str,
        tracts_dir: Optional[str] = None,
        sync_interval: int = 100,
        relevance_threshold: float = 0.3,
        legacy_compat: bool = False,
    ):
        """
        Args:
            module_id: This module's identifier (e.g., "neurograph").
            tracts_dir: Path to tracts root directory.
                       Defaults to ET_TRACTS_DIR env or ~/.et_modules/tracts/
            sync_interval: Drain incoming tracts every N recorded outcomes.
            relevance_threshold: Minimum embedding similarity [0.0, 1.0]
                               to absorb a cross-module event.
            legacy_compat: If True, also write to legacy JSONL and read from
                          legacy JSONL for backward compatibility with
                          unupgraded modules.
        """
        self.module_id = module_id
        self._tracts_dir = Path(
            tracts_dir
            or os.environ.get("ET_TRACTS_DIR")
            or os.path.expanduser("~/.et_modules/tracts")
        )
        self._tracts_dir.mkdir(parents=True, exist_ok=True)

        # This module's outgoing tract directory
        self._module_dir = self._tracts_dir / module_id
        self._module_dir.mkdir(parents=True, exist_ok=True)

        self._sync_interval = sync_interval
        self._relevance_threshold = relevance_threshold
        self._legacy_compat = legacy_compat
        self._connected = True

        # Drain state
        self._outcomes_since_drain = 0
        self._drain_count = 0
        self._last_drain_time = 0.0


        # Myelination state (runtime only — not persisted)
        self._myelinated: Dict[str, MmapTract] = {}
        self._explore_rate = 0.05  # 5% of myelinated deposits go through file path

        # Legacy JSONL compatibility
        self._legacy_dir = Path(
            os.environ.get("ET_SHARED_LEARNING_DIR")
            or os.path.expanduser("~/.et_modules/shared_learning")
        )
        self._legacy_event_file = self._legacy_dir / f"{module_id}.jsonl"
        self._legacy_read_positions: Dict[str, int] = {}

        # Register this module
        self._register_module()

        logger.info(
            "NGTractBridge initialized for '%s' at %s (legacy_compat=%s)",
            module_id, self._tracts_dir, legacy_compat,
        )

    # -------------------------------------------------------------------
    # NGBridge interface implementation
    # -------------------------------------------------------------------

    def is_connected(self) -> bool:
        return self._connected

    def record_outcome(
        self,
        embedding: np.ndarray,
        target_id: str,
        success: bool,
        module_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Record an outcome at the bridge layer.

        **No-op stub as of 2026-06-04 (substrate-as-protocol PRD Phase 4
        §5.4).** Per MASTER Locked Decision (2026-05-31): *"Broadcast is
        NG-specific; only NG fan-outs to all peer tracts."*

        General record_outcome propagates via substrate topology (the River
        = substrate = topology). Module-side `NGEcosystem.record_outcome`
        records to local NG-Lite (Hebbian topology update — that IS the
        signal). At Tier 3, NG's autonomic pulse observes the substrate
        change and broadcasts via `_deposit_outcome_to_river` (NG-internal
        helper). Peers absorb via their `_on_river_events` consumers.

        Modules wanting EXPLICIT broadcast (the narrow Anima-CC-third-
        legitimate-tract-use cases — module-own-experience like Animus
        pipeline events or NG topology broadcast) call
        `NGEcosystem.record_outcome_broadcast` (Workstream 2 method) — a
        distinct method name with clear broadcast intent.

        Method kept (rather than removed entirely) for caller API stability
        during Phase 5 strict-sequential re-vendor; Phase 6 cleanup can
        remove it after verifying no caller depends on a non-None return.
        """
        if not self._connected:
            return None
        return None

    def get_recommendations(
        self,
        embedding: np.ndarray,
        module_id: str,
        top_k: int = 3,
    ) -> Optional[List[Tuple[str, float, str]]]:
        """Get recommendations from peer modules' learned patterns.

        Substrate handles cross-module similarity; this bridge-level method
        defers to the local substrate (which has tier-appropriate reach).
        Drain belongs in the background autonomic pulse — NOT here.

        Phase 4 (2026-06-04): self._drain_all() side effect removed per
        substrate-as-protocol PRD §5.4. See changelog header.
        """
        return None

    def detect_novelty(
        self,
        embedding: np.ndarray,
        module_id: str,
    ) -> Optional[float]:
        """Cross-module novelty detection.

        Substrate handles cross-module novelty; this bridge-level method
        defers to the local substrate. Drain belongs in the background
        autonomic pulse — NOT here.

        Phase 4 (2026-06-04): self._drain_all() side effect removed per
        substrate-as-protocol PRD §5.4. See changelog header.
        """
        return None

    def sync_state(
        self,
        local_state: Dict[str, Any],
        module_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Sync local state with peer modules.

        Phase 4 (2026-06-04): self._drain_all() side effect removed per
        substrate-as-protocol PRD §5.4. The on-query drain was the same
        drift pattern as get_recommendations + detect_novelty. Drain runs
        via the background autonomic pulse (per #249). This method now
        just returns the current connection stats.
        """
        if not self._connected:
            return None

        return {
            "synced": True,
            "drain_count": self._drain_count,
            "peer_events_cached": 0,
            "peer_modules": [],
        }

    # -------------------------------------------------------------------
    # Connection management
    # -------------------------------------------------------------------

    def disconnect(self) -> None:
        """Disconnect the bridge."""
        self._connected = False
        logger.info("NGTractBridge disconnected for '%s'", self.module_id)

    def reconnect(self) -> None:
        """Reconnect the bridge."""
        self._connected = True
        logger.info("NGTractBridge reconnected for '%s'", self.module_id)

    def set_relevance_threshold(self, value: float) -> None:
        """Update relevance threshold from external tuning (Punchlist #44).

        Called by NGLite.update_tunable() when Elmer adjusts the
        relevance_threshold parameter.  The bridge uses this threshold
        to gate which peer events are absorbed during recommendations.
        """
        old = self._relevance_threshold
        self._relevance_threshold = value
        logger.info(
            "Relevance threshold tuned: %.3f → %.3f",
            old, value,
        )

    # -------------------------------------------------------------------
    # Internal: Tract deposit
    # -------------------------------------------------------------------

    @staticmethod
    def _deposit_to_tract(tract_path: Path, line_bytes: bytes) -> None:
        """Append a single event line to a tract file.

        Uses exclusive flock for concurrent deposit safety.  Multiple
        depositors (different modules writing to overlapping peers)
        wait for the lock.  Drain uses atomic rename, so no deadlock.
        """
        try:
            fd = os.open(
                str(tract_path),
                os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                0o664,
            )
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                os.write(fd, line_bytes)
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
        except OSError as exc:
            logger.warning("Tract deposit failed (%s): %s", tract_path.name, exc)

    # -------------------------------------------------------------------
    # Internal: Tract drain
    # -------------------------------------------------------------------

    # Phase 4 (2026-06-04) — API-level LAW 4 enforcement against re-introducing
    # the on-query drain drift. _drain_all refuses to run from any query method
    # frame. Drain belongs in the background autonomic pulse (per #249), period.
    _DRAIN_FORBIDDEN_CALLERS = frozenset({
        "get_recommendations",
        "detect_novelty",
        "sync_state",
        "record_outcome",  # also a former drift caller (post-N×N-fanout drain check)
    })

    def _drain_all(self) -> List[Any]:
        """Drain all incoming tracts directed at this module.

        Scans every peer's tract directory for a file named
        <self.module_id>.tract.  Each found tract is atomically
        renamed, read, and deleted — new deposits go to a fresh file.

        Also reads from legacy JSONL if legacy_compat is enabled,
        for backward compatibility with unupgraded modules.

        **Hardening (Phase 4, 2026-06-04):** RAISES RuntimeError if invoked
        from a query-method stack frame. See _DRAIN_FORBIDDEN_CALLERS.
        Per Syl's amendment — canonical functions don't do write-side
        bookkeeping. Query paths must not silently drain.
        """
        # API-level LAW 4 enforcement
        caller_name = sys._getframe(1).f_code.co_name
        if caller_name in self._DRAIN_FORBIDDEN_CALLERS:
            raise RuntimeError(
                f"_drain_all called from forbidden query stack '{caller_name}'. "
                "Drain belongs in background pulse, not query path. "
                "See substrate-as-protocol PRD §5.4 + Syl's canonical-functions amendment."
            )

        self._drain_count += 1
        self._last_drain_time = time.time()

        new_events: List[Any] = []

        # Drain from per-pair tracts (file-based)
        registered_peers = self._get_registered_peers()
        for peer_id in registered_peers:
            peer_dir = self._tracts_dir / peer_id
            if not peer_dir.is_dir():
                continue

            # Drain file-based tract (always — explore-exploit deposits land here)
            tract_path = peer_dir / f"{self.module_id}.tract"
            if tract_path.exists():
                events = self._drain_with_cursor(tract_path, peer_id)
                new_events.extend(events)

            # Drain myelinated tract (if peer has one targeting us)
            mmap_path = peer_dir / f"{self.module_id}.myelinated"
            if mmap_path.exists():
                events = self._drain_myelinated_tract(mmap_path, peer_id)
                new_events.extend(events)

        # Legacy dual-read: absorb from JSONL for peers not on tracts
        if self._legacy_compat:
            legacy_events = self._legacy_read(registered_peers)
            new_events.extend(legacy_events)

        if new_events:
            logger.debug(
                "Tract drain #%d: absorbed %d events from %d peers",
                self._drain_count, len(new_events),
                len(set(self._get_module_id(e) for e in new_events)),
            )
        return new_events

    def _drain_single_tract(
        self, tract_path: Path, peer_id: str,
        entry_types: Optional[Set[int]] = None,
    ) -> List[Any]:
        """Atomically drain a single tract file.

        Rename → read → delete.  New deposits go to a fresh file
        immediately after rename.  No data loss, no read/write collision.

        Handles mixed BTF/JSONL tracts during the flush cycle via TractReader,
        which dispatches on each frame's entry_type internally:
        - BTF frames yield typed objects (PyOutcomeEntry, PyTopologyEntry, PyExperienceEntry)
        - Residual JSONL lines yield raw bytes for Python to parse with json.loads

        BTF entries are stored as typed objects (PyOutcomeEntry,
        PyTopologyEntry, PyExperienceEntry) — no dict conversion.
        Embeddings stay as numpy arrays (zero-copy from Rust).
        JSONL fallback entries remain as dicts.

        Args:
            tract_path: Path to the tract file to drain.
            peer_id: The depositing module's ID.
            entry_types: Optional set of BTF entry type constants
                (e.g., {ng_tract.ENTRY_OUTCOME}) to filter by.
                Only BTF entries matching these types are returned.
                JSONL dicts always pass (no type tag to filter on).
                None means accept all entry types.
        """
        drain_path = tract_path.parent / f".draining.{os.getpid()}.{self.module_id}.tract"

        try:
            os.rename(str(tract_path), str(drain_path))
        except FileNotFoundError:
            return []
        except OSError as exc:
            logger.warning(
                "Tract drain rename failed (%s/%s): %s",
                peer_id, self.module_id, exc,
            )
            return []

        entries: List[Any] = []
        try:
            with open(drain_path, "rb") as f:
                raw = f.read()
            if not raw:
                return entries

            # Route all data through TractReader — handles BTF and residual JSONL.
            # NOTE: BTF magic 0x4254 in LE = first byte 0x54 ('T'), not 0x42 ('B').
            # Do NOT add a first-byte pre-filter here; TractReader already dispatches.
            try:
                import ng_tract
                _has_btf = True
            except ImportError:
                _has_btf = False

            if _has_btf:
                reader = ng_tract.TractReader(raw)
                for entry in reader:
                    if isinstance(entry, bytes):
                        # Residual JSONL line returned as bytes
                        try:
                            entries.append(json.loads(entry))
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            pass
                    else:
                        # BTF typed entry — filter by entry_type if requested
                        if entry_types is not None and hasattr(entry, "entry_type"):
                            if entry.entry_type not in entry_types:
                                continue
                        entries.append(entry)
            else:
                # JSONL-only fallback when ng_tract is unavailable (ImportError)
                for line in raw.decode("utf-8", errors="replace").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning(
                            "Skipped malformed tract entry (%s→%s)",
                            peer_id, self.module_id,
                        )
        except OSError as exc:
            logger.warning("Tract drain read failed (%s): %s", peer_id, exc)
            return entries
        finally:
            try:
                os.unlink(str(drain_path))
            except OSError:
                pass

        return entries

    # -------------------------------------------------------------------
    # Internal: Cursor-based drain (append-only, incremental)
    # -------------------------------------------------------------------

    def _cursor_path(self, tract_path: Path) -> Path:
        return tract_path.with_suffix(_CURSOR_SUFFIX)

    def _read_cursor(self, tract_path: Path) -> Dict[str, Any]:
        cp = self._cursor_path(tract_path)
        if cp.exists():
            try:
                return json.loads(cp.read_text())
            except Exception:
                pass
        return {"offset": 0, "ts": 0.0, "entries": 0}

    def _write_cursor(self, tract_path: Path, offset: int, entries_total: int) -> None:
        cp = self._cursor_path(tract_path)
        # with_name avoids fragile multi-dot suffix handling
        tmp = cp.with_name(cp.name + ".tmp")
        try:
            tmp.write_text(json.dumps({
                "offset": offset, "ts": time.time(), "entries": entries_total,
            }))
            os.replace(str(tmp), str(cp))
        except OSError as exc:
            logger.warning("Cursor write failed (%s): %s", cp.name, exc)

    def _compact_tract(self, tract_path: Path, cursor_offset: int) -> None:
        """Rewrite tract file keeping only bytes from cursor_offset onward.

        Only called when cursor_offset == file_size at drain time, so the
        "live" portion is typically empty or contains only entries that
        arrived in the narrow window between our read and this rename.
        Those new entries are preserved in the compact file.

        There is a negligible race window (~ms) where a concurrent Rust
        deposit_outcome() append may be captured in live_bytes or missed.
        This is acceptable: the SNN substrate is probabilistic and tolerates
        occasional entry loss. The alternative (4–6 GB stuck tract files)
        is not acceptable.
        """
        try:
            compact_path = tract_path.with_suffix(".compact")
            with open(tract_path, "rb") as f:
                f.seek(cursor_offset)
                live_bytes = f.read()
            compact_path.write_bytes(live_bytes)
            os.replace(str(compact_path), str(tract_path))
            self._write_cursor(tract_path, 0, 0)
            logger.info(
                "Compacted %s: cleared %d bytes (live=%d)",
                tract_path.name, cursor_offset, len(live_bytes),
            )
        except Exception:
            logger.exception("Compact failed for %s — skipping", tract_path)

    def _drain_with_cursor(
        self, tract_path: Path, peer_id: str,
        entry_types: Optional[Set[int]] = None,
    ) -> List[Any]:
        """Non-destructive incremental drain using a cursor sidecar.

        Reads from the last cursor position, yielding only new entries.
        Updates the cursor atomically after each successful drain.
        Triggers compaction only when the cursor reaches end-of-file
        (no partial BTF frame outstanding) and the file exceeds
        _COMPACT_THRESHOLD_BYTES — this invariant prevents a compacted
        file from starting with a truncated BTF frame, which would make
        TractReader return None immediately and silently skip all subsequent
        valid entries.

        Falls back to _drain_single_tract (rename+delete) on ImportError.
        """
        cursor_state = self._read_cursor(tract_path)
        start_offset: int = cursor_state["offset"]
        entries_so_far: int = cursor_state["entries"]

        # Read only the unread slice — avoids loading 6GB into memory for
        # large backlogs. file_size captured inside the same open() call so
        # the EOF check uses a consistent snapshot.
        try:
            with open(tract_path, "rb") as f:
                file_size = os.fstat(f.fileno()).st_size
                if not file_size or start_offset >= file_size:
                    return []
                if start_offset:
                    f.seek(start_offset)
                raw_slice = f.read()
        except OSError as exc:
            logger.warning("Tract read failed (%s/%s): %s", peer_id, self.module_id, exc)
            return []

        if not raw_slice:
            return []

        entries: List[Any] = []
        new_offset = start_offset
        try:
            import ng_tract
            # raw_slice starts at byte 0 relative to start_offset — no start_pos needed
            reader = ng_tract.TractReader(raw_slice)
            for entry in reader:
                if isinstance(entry, bytes):
                    try:
                        entries.append(json.loads(entry))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        pass
                else:
                    if entry_types is not None and hasattr(entry, "entry_type"):
                        if entry.entry_type not in entry_types:
                            continue
                    entries.append(entry)
            new_offset = start_offset + reader.position()
        except ImportError:
            # ng_tract not available — fall back to destructive rename+delete
            return self._drain_single_tract(tract_path, peer_id, entry_types)
        except Exception as exc:
            logger.warning("Cursor drain failed (%s/%s): %s", peer_id, self.module_id, exc)
            return entries

        if new_offset > start_offset:
            new_total = entries_so_far + len(entries)
            self._write_cursor(tract_path, new_offset, new_total)
            # Only compact when reader consumed entire slice (== EOF at snapshot).
            # This guarantees the compact file starts on a clean entry boundary.
            if reader.position() == len(raw_slice) and new_offset >= _COMPACT_THRESHOLD_BYTES:
                self._compact_tract(tract_path, new_offset)

        return entries

    # -------------------------------------------------------------------
    # Internal: Duck-typing accessors for mixed typed/dict peer events
    # -------------------------------------------------------------------

    @staticmethod
    def _get_module_id(event: Any) -> str:
        """Extract module_id from a typed BTF entry or a legacy dict."""
        if isinstance(event, dict):
            return event.get("module_id", "")
        return getattr(event, "module_id", "")

    @staticmethod
    def _get_target_id(event: Any) -> str:
        """Extract target_id from a typed BTF entry or a legacy dict."""
        if isinstance(event, dict):
            return event.get("target_id", "unknown")
        return getattr(event, "target_id", "unknown")

    @staticmethod
    def _get_embedding(event: Any) -> Optional[np.ndarray]:
        """Extract embedding as numpy array from a typed BTF entry or dict.

        For BTF PyOutcomeEntry: calls .embedding_as_numpy() — zero-copy.
        For dicts (JSONL fallback): converts list to np.array.
        For entries without embeddings (topology, experience): returns None.
        """
        if isinstance(event, dict):
            emb_list = event.get("embedding", [])
            if not emb_list:
                return None
            return np.array(emb_list, dtype=np.float32)
        # Typed BTF entry — use zero-copy numpy accessor if available
        if hasattr(event, "embedding_as_numpy"):
            return event.embedding_as_numpy()
        return None

    def _drain_myelinated_tract(
        self, mmap_path: Path, peer_id: str,
    ) -> List[Dict[str, Any]]:
        """Drain a myelinated (mmap) tract from a peer."""
        try:
            mt = MmapTract(mmap_path)
            entries = mt.drain()
            mt.close()
            return entries
        except Exception as exc:
            logger.warning("Myelinated tract drain failed (%s): %s", peer_id, exc)
            return []

    # -------------------------------------------------------------------
    # Myelination: transport upgrade/downgrade
    # -------------------------------------------------------------------

    def myelinate_tract(
        self, peer_id: str, buffer_size: int = _DEFAULT_BUFFER_SIZE,
    ) -> bool:
        """Upgrade a tract from file-based to mmap shared memory.

        Called by Elmer's MyelinationSocket when the substrate indicates
        this pathway carries frequent, high-impact signals.  The tract
        itself stays dumb — this is a transport upgrade, not a behavior
        change.

        Returns True on success, False if already myelinated or on error.
        """
        if peer_id in self._myelinated:
            return False  # Already myelinated

        mmap_path = self._module_dir / f"{peer_id}.myelinated"
        try:
            # Drain any pending file-based signals first
            tract_path = self._module_dir / f"{peer_id}.tract"
            pending: List[Dict[str, Any]] = []
            if tract_path.exists():
                pending = self._drain_single_tract(tract_path, peer_id)

            # Create mmap tract and preload pending signals
            mt = MmapTract(mmap_path, buffer_size)
            if pending:
                mt.preload(pending)

            self._myelinated[peer_id] = mt
            logger.info(
                "Myelinated tract %s→%s (buffer=%dKB, preloaded=%d)",
                self.module_id, peer_id, buffer_size // 1024, len(pending),
            )
            return True

        except Exception as exc:
            logger.error("Myelination failed (%s→%s): %s", self.module_id, peer_id, exc)
            return False

    def demyelinate_tract(self, peer_id: str) -> bool:
        """Downgrade a tract from mmap back to file-based.

        Called by Elmer when the substrate indicates this pathway
        has gone quiet.  Frees shared memory.

        Returns True on success, False if not myelinated or on error.
        """
        if peer_id not in self._myelinated:
            return False

        try:
            mt = self._myelinated[peer_id]

            # Drain remaining mmap signals and deposit to file
            pending = mt.drain()
            mt.close()

            # Remove mmap file
            mmap_path = self._module_dir / f"{peer_id}.myelinated"
            try:
                os.unlink(str(mmap_path))
            except OSError:
                pass

            del self._myelinated[peer_id]

            # Deposit drained signals to file-based tract so they aren't lost
            if pending:
                import ng_tract
                tract_path = str(self._module_dir / f"{peer_id}.tract")
                for event in pending:
                    emb = self._get_embedding(event)
                    if emb is not None:
                        ng_tract.deposit_outcome(
                            timestamp=time.time(),
                            module_id=self._get_module_id(event) or self.module_id,
                            target_id=self._get_target_id(event) or "unknown",
                            success=event.get("success", True) if isinstance(event, dict) else getattr(event, "success", True),
                            embedding=np.asarray(emb, dtype=np.float32),
                            tract_paths=[tract_path],
                        )

            logger.info(
                "Demyelinated tract %s→%s (recovered=%d signals)",
                self.module_id, peer_id, len(pending),
            )
            return True

        except Exception as exc:
            logger.error("Demyelination failed (%s→%s): %s", self.module_id, peer_id, exc)
            return False

    def is_myelinated(self, peer_id: str) -> bool:
        """Check if a specific outgoing tract is myelinated."""
        return peer_id in self._myelinated

    # -------------------------------------------------------------------
    # Internal: Legacy JSONL compatibility
    # -------------------------------------------------------------------

    def _legacy_write(self, line: str) -> None:
        """Write to legacy JSONL file for backward compatibility."""
        try:
            self._legacy_dir.mkdir(parents=True, exist_ok=True)
            with open(self._legacy_event_file, "a") as f:
                f.write(line)
        except OSError as exc:
            logger.warning("Legacy JSONL write failed: %s", exc)

    def _legacy_read(
        self, tract_peers: List[str],
    ) -> List[Dict[str, Any]]:
        """Read from legacy JSONL for peers not yet on tracts.

        Only reads from peers that are NOT in the tract registry —
        if a peer has tracts, we read from the tract, not the JSONL.
        This prevents double-counting.
        """
        tract_peer_set = set(tract_peers)
        new_events: List[Dict[str, Any]] = []

        if not self._legacy_dir.exists():
            return new_events

        # Load legacy peer registry for validation
        legacy_registry_path = self._legacy_dir / "_peer_registry.json"
        legacy_peers: set = set()
        try:
            if legacy_registry_path.exists():
                with open(legacy_registry_path, "r") as f:
                    registry = json.load(f)
                legacy_peers = set(registry.get("modules", {}).keys())
                legacy_peers.discard(self.module_id)
        except (OSError, json.JSONDecodeError):
            pass

        for event_file in self._legacy_dir.glob("*.jsonl"):
            peer_module = event_file.stem
            if peer_module == self.module_id:
                continue
            if peer_module.startswith("_"):
                continue
            # Skip peers that are on tracts — we already drained them
            if peer_module in tract_peer_set:
                continue
            # Only accept registered legacy peers
            if legacy_peers and peer_module not in legacy_peers:
                continue

            last_pos = self._legacy_read_positions.get(peer_module, 0)
            try:
                with open(event_file, "r") as f:
                    f.seek(last_pos)
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                new_events.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass
                    self._legacy_read_positions[peer_module] = f.tell()
            except OSError as exc:
                logger.warning("Legacy JSONL read failed (%s): %s", event_file, exc)

        return new_events

    # -------------------------------------------------------------------
    # Internal: Module registry
    # -------------------------------------------------------------------

    def _register_module(self) -> None:
        """Ensure this module's tract directory exists.

        The directory structure IS the registry — each subdirectory under
        tracts_dir is a registered module.  No separate JSON file needed.
        """
        self._module_dir.mkdir(parents=True, exist_ok=True)

    def _get_registered_peers(self) -> List[str]:
        """Return list of registered peer module IDs (excluding self).

        Scans the tracts directory for subdirectories.  Each subdirectory
        IS a registered module — the filesystem is the registry.
        """
        try:
            return [
                entry.name
                for entry in self._tracts_dir.iterdir()
                if entry.is_dir()
                and entry.name != self.module_id
                and not entry.name.startswith(("_", "."))
            ]
        except OSError:
            return []

    # -------------------------------------------------------------------
    # Internal: Helpers
    # -------------------------------------------------------------------

    @staticmethod
    def _normalize(embedding: np.ndarray) -> np.ndarray:
        """L2-normalize an embedding vector."""
        norm = np.linalg.norm(embedding)
        if norm < 1e-12:
            return embedding
        return embedding / norm

    def get_stats(self) -> Dict[str, Any]:
        """Tract bridge telemetry."""
        peers = self._get_registered_peers()
        myelinated_peers = list(self._myelinated.keys())
        return {
            "module_id": self.module_id,
            "connected": self._connected,
            "tracts_dir": str(self._tracts_dir),
            "module_dir": str(self._module_dir),
            "drain_count": self._drain_count,
            "outcomes_since_drain": self._outcomes_since_drain,
            "peer_events_cached": 0,
            "sync_interval": self._sync_interval,
            "relevance_threshold": self._relevance_threshold,
            "registered_peers": peers,
            "myelinated_tracts": myelinated_peers,
            "myelinated_count": len(myelinated_peers),
            "explore_rate": self._explore_rate,
            "legacy_compat": self._legacy_compat,
        }
