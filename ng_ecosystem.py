"""
NG Ecosystem — E-T Systems Module Integration Standard

Single vendorable file that gives any E-T Systems module the full
three-tier learning architecture:

  Tier 1 (Standalone):  NGLite alone.  Local Hebbian learning.
                        Zero deps beyond ng_lite.py.
  Tier 2 (Peer-pooled): NGTractBridge (preferred) or NGPeerBridge
                        (legacy fallback).  Co-located modules share
                        learning via per-pair tracts (~/.et_modules/tracts/)
                        or legacy JSONL (~/.et_modules/shared_learning/).
                        Auto-connects.  Tract bridge preferred when present.
  Tier 3 (Full SNN):    Removed — modules extract via buckets/tracts.
                        Auto-upgrades when NeuroGraph is detected on
                        the same host via ETModuleManager.

The ecosystem is "Apple-like" by design:
  - Every module is independently useful at Tier 1.
  - Any two co-located modules get a free Tier 2 boost — no config needed.
  - When NeuroGraph is present, all co-located modules transparently
    upgrade to Tier 3: full STDP, hyperedges, predictive coding, and
    CES streaming.

The module code doesn't change.  The bridge swaps.

Usage (inside any E-T Systems module):

    from ng_ecosystem import NGEcosystem

    # In your module's __init__ or startup:
    eco = NGEcosystem.get_instance(
        module_id="trollguard",
        state_path="~/.trollguard/ng_lite_state.json",
    )

    # Use the ecosystem in your module's hot path:
    embedding = my_embedder(text)
    eco.record_outcome(embedding, target_id="threat:prompt_injection", success=True)
    recs = eco.get_recommendations(embedding)
    novelty = eco.detect_novelty(embedding)
    ctx = eco.get_context(embedding)

    # Inspect tier at any time:
    print(eco.tier)        # 1, 2, or 3
    print(eco.stats())     # full telemetry

    # Periodic save (call on graceful shutdown or on a timer):
    eco.save()

Framework adapters (optional, load separately):
  - openclaw_adapter.py — on_message(text)/recall(text)/stats() for OpenClaw skills

Canonical source: https://github.com/greatnorthernfishguy-hub/NeuroGraph
License: AGPL-3.0

# ---- Changelog ----
# [2026-02-22] Claude (Sonnet 4.6) — Initial creation.
#   What: NGEcosystem class — singleton wrapper implementing the
#         standardized E-T Systems optional integration protocol.
#         Handles Tier 1→2→3 progression, auto-upgrade on NeuroGraph
#         detection, graceful degradation, and unified telemetry.
#         Also defines NGEcosystemAdapter ABC for framework adapters.
#   Why:  Each module was wiring ng_lite + peer bridge + SaaS bridge
#         independently with no shared contract.  This file is the
#         single vendorable standard so all modules behave identically
#         from an integration perspective.
#   Settings: tier3_upgrade defaults to True (auto-upgrade when NeuroGraph
#         is found).  upgrade_poll_interval=300s (re-check for NeuroGraph
#         every 5 minutes — handles cases where NeuroGraph is installed
#         after the module starts).  peer_sync_interval=100 (balances
#         freshness vs I/O).
#   How:  get_instance() creates NGLite, then tries peer bridge, then
#         queries ETModuleManager for NeuroGraph.  All in try/except so
#         each tier attempt is fully independent.  A background thread
#         polls for tier upgrades at upgrade_poll_interval.
# -------------------
# [2026-03-20] Claude (Opus 4.6) — Tract bridge wiring (punchlist #53 v0.3)
#   What: _init_peer_bridge() now prefers NGTractBridge (per-pair tracts)
#         with automatic fallback to NGPeerBridge (legacy JSONL).
#   Why:  JSONL broadcast bridge dams the River.  Per-pair tracts enable
#         independently observable pathways for myelination, explore-exploit,
#         vagus nerve, and Elmer tract management.
#   How:  Try importing ng_tract_bridge first.  If present (vendored),
#         use it.  If not (module not yet re-vendored), fall back to
#         ng_peer_bridge.  Config key peer_bridge.use_tracts (default True)
#         can force legacy mode if needed.
# -------------------
# [2026-03-22] Claude (Opus 4.6) — Dual-pass convenience method (punchlist #81)
#   What: Added dual_record_outcome() that delegates to ng_embed.NGEmbed.
#   Why:  Dual-pass embedding (forest + trees) is ecosystem-wide.
#         Modules call eco.dual_record_outcome() instead of eco.record_outcome()
#         for rich content. ng_embed.py owns the extraction + embedding logic.
#   How:  Lazy import of ng_embed to avoid circular deps. Passes self
#         (the ecosystem instance) to NGEmbed.dual_record_outcome().
# -------------------
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("ng_ecosystem")

__version__ = "1.0.0"


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

ET_MODULES_ROOT = Path.home() / ".et_modules"
SHARED_LEARNING_DIR = ET_MODULES_ROOT / "shared_learning"
REGISTRY_PATH = ET_MODULES_ROOT / "registry.json"

TIER_STANDALONE = 1  # NGLite only
TIER_PEER = 2        # + NGPeerBridge
TIER_FULL_SNN = 3    # historical — bridge removed, modules use tracts

TIER_NAMES = {
    TIER_STANDALONE: "Standalone (Tier 1)",
    TIER_PEER: "Peer-pooled (Tier 2)",
    TIER_FULL_SNN: "Full SNN (Tier 3)",
}


# --------------------------------------------------------------------------
# Framework Adapter Interface
# --------------------------------------------------------------------------

class NGEcosystemAdapter(ABC):
    """Abstract base for framework-specific adapters over NGEcosystem.

    Implement this to expose the ecosystem to a specific framework.
    Each adapter is a singleton that wraps the shared NGEcosystem
    instance — the same ecosystem, different vocabulary.

    Provided implementations:
      - OpenClawAdapter (openclaw_adapter.py, vendored separately)

    Custom adapters:
      Subclass NGEcosystemAdapter and implement all abstract methods.
      Call NGEcosystem.get_instance() inside __init__ to bind the
      shared ecosystem.  Maintain your own singleton if needed.

    Design contract:
      - Adapters MUST NOT bypass NGEcosystem internals.
      - Adapters handle embedding generation; NGEcosystem handles learning.
      - Adapters are optional and framework-specific.  The core
        NGEcosystem is framework-agnostic and always the source of truth.
    """

    @abstractmethod
    def on_message(self, text: str) -> Dict[str, Any]:
        """Process one unit of framework input (message, request, event).

        Args:
            text: Raw text to process.

        Returns:
            Dict with at minimum: {"status": "ingested"|"skipped", "tier": int}
        """
        ...

    @abstractmethod
    def get_context(self, text: str) -> Dict[str, Any]:
        """Retrieve cross-module context for the given text.

        Args:
            text: Query text.

        Returns:
            Dict with recommendations, novelty score, and tier info.
        """
        ...

    @abstractmethod
    def stats(self) -> Dict[str, Any]:
        """Return framework-specific stats including ecosystem tier."""
        ...


# --------------------------------------------------------------------------
# NGEcosystem Core
# --------------------------------------------------------------------------

class NGEcosystem:
    """Singleton E-T Systems learning ecosystem for a module.

    Manages the full Tier 1→2→3 lifecycle automatically.  Modules
    call record_outcome(), get_recommendations(), detect_novelty(),
    and get_context() without knowing or caring which tier is active.

    Thread-safety: All public methods are safe to call from multiple
    threads.  The tier upgrade loop runs in a daemon thread.
    """

    _instances: Dict[str, "NGEcosystem"] = {}
    _lock = threading.Lock()

    def __init__(
        self,
        module_id: str,
        state_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            module_id: Unique module identifier (e.g., "trollguard").
                       Must match the module_id in et_module.json.
            state_path: Path to persist NGLite state JSON.
                        Defaults to ~/.et_modules/{module_id}/ng_lite_state.json
            config: Optional config overrides.  Keys:
                    peer_bridge.enabled (bool, default True)
                    peer_bridge.sync_interval (int, default 100)
                    tier3_upgrade.enabled (bool, default True)
                    tier3_upgrade.poll_interval (float, default 300.0)
                    ng_lite.* (passed through to NGLite)
        """
        self.module_id = module_id

        self._config = {
            "peer_bridge": {
                "enabled": True,
                "sync_interval": 100,
            },
            "tier3_upgrade": {
                "enabled": False,  # disabled — modules use tracts, not bridge bypass
                "poll_interval": 300.0,
            },
        }
        if config:
            _deep_merge(self._config, config)

        # State persistence path
        if state_path:
            self._state_path = Path(state_path).expanduser()
        else:
            self._state_path = (
                ET_MODULES_ROOT / module_id / "ng_lite_state.json"
            )
        self._state_path.parent.mkdir(parents=True, exist_ok=True)

        # Internal state
        self._tier = TIER_STANDALONE
        self._ng: Any = None              # NGLite instance
        self._ng_memory: Any = None       # NeuroGraphMemory ref (set externally at Tier 3)
        self._peer_bridge: Any = None     # NGPeerBridge instance
        self._shutdown_event = threading.Event()
        self._ops_lock = threading.Lock()

        # Boot sequence
        self._init_ng_lite()
        self._init_peer_bridge()

        logger.info(
            "[%s] NGEcosystem ready at %s",
            module_id,
            TIER_NAMES[self._tier],
        )

    # -----------------------------------------------------------------
    # Singleton factory
    # -----------------------------------------------------------------

    @classmethod
    def get_instance(
        cls,
        module_id: str,
        state_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> "NGEcosystem":
        """Return the singleton NGEcosystem for this module_id.

        Thread-safe.  Subsequent calls with the same module_id return
        the existing instance regardless of state_path/config args.
        """
        with cls._lock:
            if module_id not in cls._instances:
                cls._instances[module_id] = cls(module_id, state_path, config)
            return cls._instances[module_id]

    @classmethod
    def reset_instance(cls, module_id: str) -> None:
        """Destroy the singleton for module_id (testing only)."""
        with cls._lock:
            inst = cls._instances.pop(module_id, None)
            if inst:
                inst._shutdown_event.set()

    # -----------------------------------------------------------------
    # Tier 1: NGLite init
    # -----------------------------------------------------------------

    def _init_ng_lite(self) -> None:
        """Initialize local NGLite substrate (always Tier 1)."""
        try:
            from ng_lite import NGLite  # vendored alongside this file

            ng_config = self._config.get("ng_lite", {})
            self._ng = NGLite(module_id=self.module_id, config=ng_config)

            if self._state_path.exists():
                self._ng.load(str(self._state_path))
                logger.debug("[%s] NGLite state loaded from %s", self.module_id, self._state_path)

        except Exception as exc:
            logger.error("[%s] NGLite init failed: %s", self.module_id, exc)
            self._ng = None

    # -----------------------------------------------------------------
    # Tier 2: NGPeerBridge init
    # -----------------------------------------------------------------

    def _init_peer_bridge(self) -> None:
        """Try to connect Tier 2 bridge. Prefers tract bridge, falls back to legacy JSONL."""
        if not self._config["peer_bridge"]["enabled"]:
            return
        if self._ng is None:
            return

        bridge = None

        # Tract bridge (v0.3+) — per-pair directional tracts
        if self._config["peer_bridge"].get("use_tracts", True):
            try:
                from ng_tract_bridge import NGTractBridge  # vendored alongside

                bridge = NGTractBridge(
                    module_id=self.module_id,
                    tracts_dir=str(ET_MODULES_ROOT / "tracts"),
                    sync_interval=self._config["peer_bridge"]["sync_interval"],
                )
                logger.info("[%s] NGTractBridge connected (tract-based River)", self.module_id)
            except ImportError:
                pass
            except Exception as exc:
                logger.debug("[%s] NGTractBridge failed: %s", self.module_id, exc)

        # Legacy fallback — JSONL broadcast bridge
        if bridge is None:
            try:
                from ng_peer_bridge import NGPeerBridge  # vendored alongside

                bridge = NGPeerBridge(
                    module_id=self.module_id,
                    shared_dir=str(SHARED_LEARNING_DIR),
                    sync_interval=self._config["peer_bridge"]["sync_interval"],
                )
                logger.info("[%s] NGPeerBridge connected (legacy JSONL River)", self.module_id)
            except Exception as exc:
                logger.debug("[%s] No peer bridge available: %s", self.module_id, exc)
                return

        self._ng.connect_bridge(bridge)
        self._peer_bridge = bridge
        self._tier = TIER_FULL_SNN  # tract bridge = full substrate access

    # -----------------------------------------------------------------
    # Tier 3: NeuroGraph auto-upgrade
    # -----------------------------------------------------------------
    @property
    def tier(self) -> int:
        """Current learning tier (1, 2, or 3)."""
        return self._tier

    @property
    def tier_name(self) -> str:
        """Human-readable tier name."""
        return TIER_NAMES.get(self._tier, "Unknown")

    def record_outcome(
        self,
        embedding: np.ndarray,
        target_id: str,
        success: bool,
        strength: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record a learning outcome.

        The embedding is the semantic representation of the input.
        The target_id is an opaque string representing what was decided
        (e.g., "model:llama3", "threat:prompt_injection", "action:search").

        Returns the learning result dict from the substrate.
        """
        if self._ng is None:
            return {}
        with self._ops_lock:
            return self._ng.record_outcome(
                embedding, target_id, success, strength=strength, metadata=metadata
            )

    def dual_record_outcome(
        self,
        content: str,
        embedding: np.ndarray,
        target_id: str,
        success: bool,
        strength: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Dual-pass learning: forest (gestalt) + tree (concept) embeddings.

        Pass 1: record_outcome() with the forest embedding (standard).
        Pass 2: Extract concepts via TID → embed each → record_outcome()
                 per tree → create forest→tree substrate links.

        Falls back to single-pass (forest only) if TID unavailable.

        Args:
            content: Raw text (for concept extraction in Pass 2).
            embedding: Pre-computed forest embedding (Pass 1).
            target_id: Opaque string for what was decided.
            success: Whether the outcome was successful.
            strength: Significance [0.0, 1.0].
            metadata: Additional metadata dict.

        Returns dict with forest_result, tree_ids, concepts, pass2_attempted.
        """
        from ng_embed import NGEmbed
        return NGEmbed.get_instance().dual_record_outcome(
            self, content, embedding, target_id, success,
            strength=strength, metadata=metadata,
        )

    def get_recommendations(
        self,
        embedding: np.ndarray,
        top_k: int = 3,
    ) -> List[Tuple[str, float, str]]:
        """Get recommendations from the active learning substrate.

        Returns list of (target_id, confidence, reasoning).

        At Tier 1, returns local recommendations only.
        At Tier 2, includes cross-module peer patterns.
        At Tier 3, includes full SNN recommendations + hyperedge context.
        """
        if self._ng is None:
            return []
        with self._ops_lock:
            return self._ng.get_recommendations(embedding, top_k=top_k)

    def detect_novelty(self, embedding: np.ndarray) -> float:
        """Return novelty score [0.0=routine, 1.0=completely novel].

        At Tier 2+, novelty is cross-module: something novel to this
        module but known to a peer scores lower than it would at Tier 1.
        """
        if self._ng is None:
            return 1.0  # Conservative: unknown = novel
        with self._ops_lock:
            result = self._ng.detect_novelty(embedding)
            return result if result is not None else 1.0

    def get_context(
        self,
        embedding: np.ndarray,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """Unified context retrieval for prompt enrichment or decision support.

        Returns a dict suitable for injecting into a prompt or logging:
          tier:            int — current tier
          tier_name:       str — human-readable tier
          recommendations: list of (target_id, confidence, reasoning)
          novelty:         float — novelty score [0.0, 1.0]
          ng_context:      str|None — Tier 3 SNN surfaced context if available
        """
        recs = self.get_recommendations(embedding, top_k=top_k)
        novelty = self.detect_novelty(embedding)
        ng_context = None

        # Tier 3: ask NeuroGraph for surfaced cognitive context
        if self._tier == TIER_FULL_SNN and self._ng_memory is not None:
            try:
                ng_context = self._ng_memory.surface_context(embedding)
            except Exception:
                pass

        return {
            "tier": self._tier,
            "tier_name": self.tier_name,
            "recommendations": recs,
            "novelty": novelty,
            "ng_context": ng_context,
        }

    def save(self) -> None:
        """Persist NGLite state to disk."""
        if self._ng is None:
            return
        try:
            with self._ops_lock:
                self._ng.save(str(self._state_path))
            logger.debug("[%s] NGLite state saved to %s", self.module_id, self._state_path)
        except Exception as exc:
            logger.warning("[%s] Save failed: %s", self.module_id, exc)

    def stats(self) -> Dict[str, Any]:
        """Return unified telemetry for logging, dashboards, or skill SKILL.md output."""
        ng_stats: Dict[str, Any] = {}
        if self._ng is not None:
            try:
                ng_stats = self._ng.get_stats()
            except Exception:
                pass

        peer_stats: Dict[str, Any] = {}
        if self._peer_bridge is not None:
            try:
                peer_stats = self._peer_bridge.get_stats()
            except Exception:
                pass

        ng_memory_stats: Dict[str, Any] = {}
        if self._ng_memory is not None:
            try:
                ng_memory_stats = self._ng_memory.stats()
            except Exception:
                pass

        return {
            "ecosystem_version": __version__,
            "module_id": self.module_id,
            "tier": self._tier,
            "tier_name": self.tier_name,
            "ng_lite": ng_stats,
            "peer_bridge": peer_stats if peer_stats else None,
            "ng_memory": (
                {
                    "connected": True,
                    "version": ng_memory_stats.get("version", "unknown"),
                    "nodes": ng_memory_stats.get("graph", {}).get("node_count", "?"),
                }
                if ng_memory_stats else None
            ),
            "state_path": str(self._state_path),
        }

    def shutdown(self) -> None:
        """Graceful shutdown: save state and stop the upgrade thread."""
        self._shutdown_event.set()
        self.save()
        logger.info("[%s] NGEcosystem shutdown complete", self.module_id)


# --------------------------------------------------------------------------
# Known NeuroGraph install paths (Tier 3 auto-detection)
# --------------------------------------------------------------------------

_NEUROGRAPH_KNOWN_PATHS: List[str] = [
    "~/NeuroGraph",
    "~/.openclaw/workspace/skills/neurograph",
    "~/.et_modules/modules/neurograph",
]


# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge override into base in-place."""
    for key, val in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(val, dict):
            _deep_merge(base[key], val)
        else:
            base[key] = val


# --------------------------------------------------------------------------
# Convenience: module-level quick-start
# --------------------------------------------------------------------------

def init(
    module_id: str,
    state_path: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> NGEcosystem:
    """One-call initialization for simple module integrations.

    Example:
        import ng_ecosystem
        eco = ng_ecosystem.init("trollguard")
    """
    return NGEcosystem.get_instance(module_id, state_path, config)
