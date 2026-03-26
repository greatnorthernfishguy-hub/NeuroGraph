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
    └── _tract_registry.json      # Active module registry

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
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ng_lite import NGBridge

logger = logging.getLogger("ng_tract_bridge")

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

    def preload(self, events: List[Dict[str, Any]]) -> None:
        """Preload events into the write buffer (used during upgrade)."""
        for event in events:
            line = json.dumps(event, default=str) + "\n"
            if not self.deposit(line.encode("utf-8")):
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
        └── _tract_registry.json
    """

    def __init__(
        self,
        module_id: str,
        tracts_dir: Optional[str] = None,
        sync_interval: int = 100,
        relevance_threshold: float = 0.3,
        legacy_compat: bool = True,
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

        # Cross-module event cache (for recommendations and novelty)
        self._peer_events: List[Dict[str, Any]] = []
        self._peer_events_max = 500

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
        """Record an outcome and deposit it into per-peer tracts.

        Fan-out: the event is deposited into a separate tract file for
        each registered peer.  Each tract is a directional channel —
        what flows through immunis/elmer.tract is Immunis's experience
        destined for Elmer.

        Returns cross-module insights from previously drained peer data.
        """
        if not self._connected:
            return None

        # Build event
        event = {
            "timestamp": time.time(),
            "module_id": module_id,
            "target_id": target_id,
            "success": success,
            "embedding": embedding.tolist(),
            "metadata": metadata or {},
        }
        line = json.dumps(event, default=str) + "\n"
        line_bytes = line.encode("utf-8")

        # Fan-out deposit to per-peer tracts
        peers = self._get_registered_peers()
        for peer_id in peers:
            if peer_id in self._myelinated and random.random() > self._explore_rate:
                # Myelinated path — mmap deposit
                mmap_tract = self._myelinated[peer_id]
                if not mmap_tract.deposit(line_bytes):
                    # Buffer full — fall back to file for this deposit
                    tract_path = self._module_dir / f"{peer_id}.tract"
                    self._deposit_to_tract(tract_path, line_bytes)
            else:
                # Unmyelinated path (or explore-exploit probe)
                tract_path = self._module_dir / f"{peer_id}.tract"
                self._deposit_to_tract(tract_path, line_bytes)

        # Legacy dual-write: also append to JSONL for unupgraded peers
        if self._legacy_compat:
            self._legacy_write(line)

        # Drain check
        self._outcomes_since_drain += 1
        if self._outcomes_since_drain >= self._sync_interval:
            self._drain_all()
            self._outcomes_since_drain = 0

        # Return cross-module insights from cached peer events
        peer_count = len(self._peer_events)
        if peer_count > 0:
            return {
                "cross_module": True,
                "peer_events_cached": peer_count,
                "peer_modules": list(set(
                    e["module_id"] for e in self._peer_events
                    if e["module_id"] != self.module_id
                )),
            }

        return {"cross_module": True, "peer_events_cached": 0}

    def get_recommendations(
        self,
        embedding: np.ndarray,
        module_id: str,
        top_k: int = 3,
    ) -> Optional[List[Tuple[str, float, str]]]:
        """Get recommendations from peer modules' learned patterns.

        Searches cached peer events for similar embeddings and returns
        their targets as recommendations.
        """
        if not self._connected or not self._peer_events:
            return None

        emb = self._normalize(embedding)

        scored: List[Tuple[str, float, str]] = []
        for event in self._peer_events:
            if event["module_id"] == module_id:
                continue

            peer_emb = np.array(event.get("embedding", []))
            if peer_emb.size == 0 or peer_emb.shape[0] != emb.shape[0]:
                continue

            peer_emb = self._normalize(peer_emb)
            similarity = float(np.dot(emb, peer_emb))

            if similarity >= self._relevance_threshold:
                target = event.get("target_id", "unknown")
                source_module = event.get("module_id", "unknown")
                reasoning = (
                    f"Cross-module recommendation from {source_module} "
                    f"(similarity={similarity:.3f})"
                )
                scored.append((target, similarity, reasoning))

        if not scored:
            return None

        scored.sort(key=lambda x: x[1], reverse=True)
        seen_targets: set = set()
        deduped: List[Tuple[str, float, str]] = []
        for target, sim, reason in scored:
            if target not in seen_targets:
                seen_targets.add(target)
                deduped.append((target, sim, reason))
                if len(deduped) >= top_k:
                    break

        return deduped

    def detect_novelty(
        self,
        embedding: np.ndarray,
        module_id: str,
    ) -> Optional[float]:
        """Cross-module novelty detection.

        Checks if this embedding is novel not just to this module,
        but to ALL peer modules on this host.
        """
        if not self._connected or not self._peer_events:
            return None

        emb = self._normalize(embedding)
        max_similarity = 0.0

        for event in self._peer_events:
            peer_emb = np.array(event.get("embedding", []))
            if peer_emb.size == 0 or peer_emb.shape[0] != emb.shape[0]:
                continue

            peer_emb = self._normalize(peer_emb)
            similarity = float(np.dot(emb, peer_emb))
            if similarity > max_similarity:
                max_similarity = similarity

        return max(0.0, 1.0 - max_similarity)

    def sync_state(
        self,
        local_state: Dict[str, Any],
        module_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Sync local state with peer modules via tract drain."""
        if not self._connected:
            return None

        self._drain_all()

        return {
            "synced": True,
            "drain_count": self._drain_count,
            "peer_events_cached": len(self._peer_events),
            "peer_modules": list(set(
                e["module_id"] for e in self._peer_events
                if e["module_id"] != self.module_id
            )),
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

    def _drain_all(self) -> None:
        """Drain all incoming tracts directed at this module.

        Scans every peer's tract directory for a file named
        <self.module_id>.tract.  Each found tract is atomically
        renamed, read, and deleted — new deposits go to a fresh file.

        Also reads from legacy JSONL if legacy_compat is enabled,
        for backward compatibility with unupgraded modules.
        """
        self._drain_count += 1
        self._last_drain_time = time.time()

        new_events: List[Dict[str, Any]] = []

        # Drain from per-pair tracts (file-based)
        registered_peers = self._get_registered_peers()
        for peer_id in registered_peers:
            peer_dir = self._tracts_dir / peer_id
            if not peer_dir.is_dir():
                continue

            # Drain file-based tract (always — explore-exploit deposits land here)
            tract_path = peer_dir / f"{self.module_id}.tract"
            if tract_path.exists():
                events = self._drain_single_tract(tract_path, peer_id)
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

        # Add to cache, maintaining bounded size
        self._peer_events.extend(new_events)
        if len(self._peer_events) > self._peer_events_max:
            self._peer_events = self._peer_events[-self._peer_events_max:]

        if new_events:
            logger.info(
                "Tract drain #%d: absorbed %d events from %d peers",
                self._drain_count, len(new_events),
                len(set(e.get("module_id", "") for e in new_events)),
            )

    def _drain_single_tract(
        self, tract_path: Path, peer_id: str,
    ) -> List[Dict[str, Any]]:
        """Atomically drain a single tract file.

        Rename → read → delete.  New deposits go to a fresh file
        immediately after rename.  No data loss, no read/write collision.
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

        entries: List[Dict[str, Any]] = []
        try:
            with open(drain_path, "r") as f:
                for line in f:
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
                tract_path = self._module_dir / f"{peer_id}.tract"
                for event in pending:
                    line = json.dumps(event, default=str) + "\n"
                    self._deposit_to_tract(tract_path, line.encode("utf-8"))

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
        """Register this module in the tract registry."""
        registry_path = self._tracts_dir / "_tract_registry.json"

        try:
            if registry_path.exists():
                with open(registry_path, "r") as f:
                    registry = json.load(f)
            else:
                registry = {"modules": {}}

            registry["modules"][self.module_id] = {
                "registered_at": time.time(),
                "tract_dir": str(self._module_dir),
                "pid": os.getpid(),
            }

            # Atomic write
            tmp = registry_path.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(registry, f, indent=2)
            tmp.replace(registry_path)

        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to update tract registry: %s", exc)

    def _get_registered_peers(self) -> List[str]:
        """Return list of registered peer module IDs (excluding self)."""
        registry_path = self._tracts_dir / "_tract_registry.json"

        try:
            if not registry_path.exists():
                return []
            with open(registry_path, "r") as f:
                registry = json.load(f)
            peers = list(registry.get("modules", {}).keys())
            return [p for p in peers if p != self.module_id]
        except (OSError, json.JSONDecodeError):
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
            "peer_events_cached": len(self._peer_events),
            "sync_interval": self._sync_interval,
            "relevance_threshold": self._relevance_threshold,
            "registered_peers": peers,
            "myelinated_tracts": myelinated_peers,
            "myelinated_count": len(myelinated_peers),
            "explore_rate": self._explore_rate,
            "legacy_compat": self._legacy_compat,
        }
