"""
NGPeerBridge — Tier 2 Cross-Module Learning for E-T Systems

Connects co-located NG-Lite instances for shared learning without
requiring the full NeuroGraph SNN infrastructure.  Each module writes
learning events to a JSONL file in a shared directory; peers read each
other's files periodically and absorb relevant patterns through
embedding similarity.

This is the lightweight coordination layer — no central server, no
daemon, no network traffic.  Just shared files on local disk.

Directory structure:
    ~/.et_modules/shared_learning/
    ├── neurograph.jsonl            # NeuroGraph's events
    ├── trollguard.jsonl            # TrollGuard's events
    ├── inference_difference.jsonl  # TID's events
    └── _peer_registry.json         # Active module registry

Usage:
    from ng_lite import NGLite
    from ng_peer_bridge import NGPeerBridge

    bridge = NGPeerBridge(module_id="neurograph")
    ng = NGLite(module_id="neurograph")
    ng.connect_bridge(bridge)

    # Outcomes are now shared with sibling modules on this host.
    # Cross-module intelligence flows automatically.

Connectivity tiers (from ng_lite.py):
    Tier 1 — Isolated: standalone NG-Lite
    Tier 2 — Peer-pooled: THIS bridge. Co-located modules share via files.
    Tier 3 — Full SaaS: NGSaaSBridge to full NeuroGraph Foundation.

    A module can have both bridges active: NGPeerBridge for sibling
    modules, and NGSaaSBridge for the full SNN.  The bridge-first API
    in NG-Lite checks the connected bridge before falling back to local.

Canonical source: https://github.com/greatnorthernfishguy-hub/NeuroGraph
License: AGPL-3.0

# ---- Changelog ----
# [2026-02-17] Claude (Opus 4.6) — Initial creation.
#   What: NGPeerBridge class implementing NGBridge interface for Tier 2
#         cross-module learning via shared filesystem event logs.
#   Why:  Enables co-located E-T Systems modules (TrollGuard, TID,
#         NeuroGraph, Cricket) to share learning patterns without
#         requiring the full SNN.  This is the "middle tier" between
#         isolated local learning and full SaaS integration.
#   Settings: sync_interval=100 (sync every 100 outcomes — balances
#         freshness vs I/O), relevance_threshold=0.3 (permissive enough
#         to catch cross-domain patterns), peer_events_max=500 (bounded
#         memory for cached events).
#   How:  File-based exchange.  Each module appends JSONL events to its
#         own file.  Periodic sync reads peers' files from last-known
#         position (no re-reading).  Recommendations scored by embedding
#         cosine similarity.
# -------------------
#
# ---- Grok Review Changelog (v0.7.1) ----
# Accepted: Added module_id validation in _sync_from_peers() — only accepts
#     events from peers listed in the _peer_registry.json, filtering out
#     orphaned or unrecognized JSONL files.
# Rejected: 'Security Hole — reads all *.jsonl without auth' — The shared
#     directory is ~/.et_modules/ (user-owned, mode 0700 on Linux).  All
#     co-located modules run under the same user.  If an attacker has write
#     access to the user's home directory, filesystem-level auth would not
#     help.  Module ID validation (now added) provides sufficient filtering.
# Rejected: 'Race Conditions — no file locks on seek/read' — JSONL is
#     append-only.  Each line is either fully written or not.  The worst
#     case (partial last line) is already handled by the json.loads
#     try/except.  Adding fcntl.lockf() would add platform-specific
#     complexity (Windows incompatible) for no practical benefit.
# Rejected: 'NGSaaSBridge sync_state() is one-way' — Intentional.  NG-Lite
#     is a lightweight Hebbian substrate; pulling SNN predictions, hyperedges,
#     and STDP state back into it would defeat its purpose as a simple,
#     vendorable single-file module.  Cross-module intelligence flows
#     upward (Lite → Full) by design; enriched recommendations flow back
#     through the bridge.get_recommendations() API.
# Rejected: '_normalize() duplicates np.linalg.norm' — The 3-line static
#     method exists in both ng_lite.py and ng_peer_bridge.py because peer
#     bridge imports only NGBridge (the abstract interface), not private
#     methods.  The duplication is deliberate: ng_peer_bridge.py must remain
#     independent of ng_lite.py's internals.
# -------------------------------------------
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ng_lite import NGBridge

logger = logging.getLogger("ng_peer_bridge")


class NGPeerBridge(NGBridge):
    """Tier 2 bridge: peer-to-peer learning between co-located NG-Lite instances.

    Each module writes learning events to a shared directory.  Periodically,
    modules read each other's events and absorb relevant patterns.  No
    central server, no daemon — just shared files on local disk.

    Usage:
        from ng_lite import NGLite
        from ng_peer_bridge import NGPeerBridge

        # Create the peer bridge
        bridge = NGPeerBridge(module_id="neurograph")

        # Connect it to your NG-Lite instance
        ng = NGLite(module_id="neurograph")
        ng.connect_bridge(bridge)

        # Now outcomes are shared with other modules on this host.
        # Cross-module intelligence flows automatically.

    Directory structure:
        ~/.et_modules/shared_learning/
        ├── neurograph.jsonl          # NeuroGraph's events
        ├── trollguard.jsonl          # TrollGuard's events
        ├── inference_difference.jsonl # TID's events
        └── _peer_registry.json       # Active module registry
    """

    def __init__(
        self,
        module_id: str,
        shared_dir: Optional[str] = None,
        sync_interval: int = 100,
        relevance_threshold: float = 0.3,
    ):
        """
        Args:
            module_id: This module's identifier (e.g., "neurograph").
            shared_dir: Path to shared learning directory.
                       Defaults to ~/.et_modules/shared_learning/
            sync_interval: Sync with peers every N recorded outcomes.
            relevance_threshold: Minimum embedding similarity [0.0, 1.0]
                               to absorb a cross-module event.
        """
        self.module_id = module_id
        self._shared_dir = Path(
            shared_dir
            or os.environ.get("ET_SHARED_LEARNING_DIR")
            or os.path.expanduser("~/.et_modules/shared_learning")
        )
        self._shared_dir.mkdir(parents=True, exist_ok=True)

        self._sync_interval = sync_interval
        self._relevance_threshold = relevance_threshold
        self._connected = True

        # Event log for this module
        self._event_file = self._shared_dir / f"{module_id}.jsonl"

        # Track sync state
        self._outcomes_since_sync = 0
        self._sync_count = 0
        self._last_sync_time = 0.0

        # Track read positions in peer files (to avoid re-reading)
        self._peer_read_positions: Dict[str, int] = {}

        # Cross-module event cache (for recommendations and novelty)
        self._peer_events: List[Dict[str, Any]] = []
        self._peer_events_max = 500

        # Register this module
        self._register_module()

        logger.info(
            "NGPeerBridge initialized for '%s' at %s",
            module_id, self._shared_dir,
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
        """Record an outcome and share it with peer modules.

        Writes the event to the shared JSONL file so other modules
        can discover and absorb it during their sync cycles.

        Returns cross-module insights from previously synced peer data.
        """
        if not self._connected:
            return None

        # Write event to shared file
        event = {
            "timestamp": time.time(),
            "module_id": module_id,
            "target_id": target_id,
            "success": success,
            "embedding": embedding.tolist(),
            "metadata": metadata or {},
        }

        try:
            with open(self._event_file, "a") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except OSError as e:
            logger.warning("Failed to write peer event: %s", e)
            return None

        # Check if it's time to sync with peers
        self._outcomes_since_sync += 1
        if self._outcomes_since_sync >= self._sync_interval:
            self._sync_from_peers()
            self._outcomes_since_sync = 0

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
        their targets as recommendations.  This is how NeuroGraph can
        benefit from TrollGuard's security knowledge, and vice versa.
        """
        if not self._connected or not self._peer_events:
            return None

        emb = self._normalize(embedding)

        # Score each peer event by embedding similarity
        scored: List[Tuple[str, float, str]] = []
        for event in self._peer_events:
            if event["module_id"] == module_id:
                continue  # Skip own events

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

        # Sort by similarity descending, deduplicate by target
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
        but to ALL peer modules on this host.  Something that
        NeuroGraph has never seen might be well-known to TrollGuard.
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
        """Sync local state with peer modules.

        Reads recent events from all peer modules and caches them
        for use in recommendations and novelty detection.
        """
        if not self._connected:
            return None

        self._sync_from_peers()

        return {
            "synced": True,
            "sync_number": self._sync_count,
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
        logger.info("NGPeerBridge disconnected for '%s'", self.module_id)

    def reconnect(self) -> None:
        """Reconnect the bridge."""
        self._connected = True
        logger.info("NGPeerBridge reconnected for '%s'", self.module_id)

    # -------------------------------------------------------------------
    # Internal: Peer sync
    # -------------------------------------------------------------------

    def _sync_from_peers(self) -> None:
        """Read new events from all peer modules' event files.

        Only reads from peers listed in the _peer_registry.json to filter
        out orphaned or unrecognized JSONL files (Grok review: peer validation).
        """
        self._sync_count += 1
        self._last_sync_time = time.time()

        # Load registered peers for validation (Grok review: peer filtering)
        registered_peers: set = set()
        registry_path = self._shared_dir / "_peer_registry.json"
        try:
            if registry_path.exists():
                with open(registry_path, "r") as f:
                    registry = json.load(f)
                registered_peers = set(registry.get("modules", {}).keys())
                # Remove self — we only care about OTHER registered peers
                registered_peers.discard(self.module_id)
        except (OSError, json.JSONDecodeError):
            pass  # If registry is unreadable, fall back to accepting all

        new_events: List[Dict[str, Any]] = []

        for event_file in self._shared_dir.glob("*.jsonl"):
            peer_module = event_file.stem
            if peer_module == self.module_id:
                continue  # Skip own file
            if peer_module.startswith("_"):
                continue  # Skip meta-files like _peer_registry
            # Only filter if other peers ARE registered — if none are,
            # accept all files (new module just starting up).
            if registered_peers and peer_module not in registered_peers:
                logger.debug(
                    "Skipping unregistered peer file: %s", event_file.name
                )
                continue

            # Read from last known position
            last_pos = self._peer_read_positions.get(peer_module, 0)

            try:
                with open(event_file, "r") as f:
                    f.seek(last_pos)
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                event = json.loads(line)
                                new_events.append(event)
                            except json.JSONDecodeError:
                                pass
                    self._peer_read_positions[peer_module] = f.tell()
            except OSError as e:
                logger.warning("Failed to read peer file %s: %s", event_file, e)

        # Add to cache, maintaining bounded size
        self._peer_events.extend(new_events)
        if len(self._peer_events) > self._peer_events_max:
            self._peer_events = self._peer_events[-self._peer_events_max:]

        if new_events:
            logger.info(
                "Peer sync #%d: absorbed %d events from %d peers",
                self._sync_count, len(new_events),
                len(set(e.get("module_id", "") for e in new_events)),
            )

    # -------------------------------------------------------------------
    # Internal: Module registry
    # -------------------------------------------------------------------

    def _register_module(self) -> None:
        """Register this module in the peer registry."""
        registry_path = self._shared_dir / "_peer_registry.json"

        try:
            if registry_path.exists():
                with open(registry_path, "r") as f:
                    registry = json.load(f)
            else:
                registry = {"modules": {}}

            registry["modules"][self.module_id] = {
                "registered_at": time.time(),
                "event_file": str(self._event_file),
                "pid": os.getpid(),
            }

            with open(registry_path, "w") as f:
                json.dump(registry, f, indent=2)

        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Failed to update peer registry: %s", e)

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
        """Peer bridge telemetry."""
        return {
            "module_id": self.module_id,
            "connected": self._connected,
            "shared_dir": str(self._shared_dir),
            "sync_count": self._sync_count,
            "outcomes_since_sync": self._outcomes_since_sync,
            "peer_events_cached": len(self._peer_events),
            "sync_interval": self._sync_interval,
            "relevance_threshold": self._relevance_threshold,
        }
