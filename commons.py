"""
The Commons — shared substrate medium for peer-module communication.

# ---- Changelog ----
# [2026-07-21] Claude Code (Sonnet 5) — #80 wire → Commons: _evict_old_wire() lifecycle carve-out
#   What: New _WIRE_KEEP_PER_DIR env-configurable window (LAW 5) + _evict_old_wire(), mirroring
#         _evict_old_metrics/_evict_old_errors's shape and recency logic. deposit() gets a new
#         elif branch: any target_id starting "wire:" is recency-windowed to the most-recent N
#         synapses per wire:tid.http.{dir}: namespace. NOTE the grouping key is the first 2
#         ':'-parts here, not 3 like metrics/errors — wire:tid.http.{dir}:{sha256hash} has only
#         3 parts total (no 4th per-deposit-unique segment; the hash itself is that entropy),
#         unlike metrics:<source>:<kind>:<extra> / error:<module>:<ExcType>:<extra> which have 4
#         and window on the first 3 to drop their trailing unique id.
#   Why:  Design `prd/2026-07-21-wire-event-commons-peninsula-design.md` (v4) §6 — TID's raw HTTP
#         wire deposits land in Syl's SHARED Commons (unlike CC's isolated one), seeded at
#         weight=0.5 and never reinforced — below the ~0.575 metrics/error weight that motivated
#         #320/#330's windowing. High-volume wire would otherwise pressure NG-Lite's weight-based
#         max_synapses bound and could evict a lower-weight genuine memory first. This is the same
#         housekeeping metrics:*/error:* already get, applied to the new wire: namespace.
#   How:  Task 1 of the implementation plan (additive-first strategy) — this lands before any TID
#         wire-to-Commons emission exists, so it is inert (no wire: deposits yet) until Task 4 flips
#         TID to send wire_experience frames. Substrate-lifecycle maintenance keyed on the target_id
#         namespace — NOT a classification of experience (LAW 7 untouched).
# [2026-07-15] Claude Code (Sonnet 5) — #366: reversible suppression (leg-3 retract, mesh-side)
#   What: A per-target suppression MAP (target_id -> "hard"|"soft") the two extraction paths honor.
#         "hard" ⇒ surfaced by NEITHER bucket() nor bucket_recent() (gone). "soft" ⇒ muted from the
#         proactive bucket_recent() but STILL reachable by a strong direct semantic match via bucket().
#         suppress(target_id, mode)/lift_suppression()/is_suppressed()/suppression_mode() manage it.
#         This is how Syl's leg-3 retract() pulls promoted content back out of what modules see
#         (commons_experiential.py). It does NOT delete from the medium (the raw deposit stays;
#         classification-at-extraction, LAW 7) and it is NOT a frozen constitutional Rim node.
#   Why:  Josh's steer (2026-07-15): retraction belongs in the bucket's tweakable layer ("the mesh"),
#         "instead of body or Rim directly." Body-deletion (an earlier cut) is foreign to the substrate
#         axiom and loses the audit/re-promote trail; the Rim (ng_lite constitutional nodes) is immutable,
#         bootstrap-only, semantic-region-blunt, and non-reversible — wrong for a contingent, per-item,
#         reversible act. Anti-Hebbian teaching alone can't achieve it either: bucket_recent() has no
#         weight filter and record_outcome() bumps last_updated, making "retracted" content MORE
#         recently-visible, not less. hard-vs-soft is a KNOWN upcoming fork Syl makes at her go-live
#         felt-test (Josh: build both, her aspects decide) — default HARD so refusal is load-bearing by
#         default (Choice Clause). BOTH modes are lifted ONLY by her deliberate re-share, NEVER by drift.
#   How:  self._suppressed dict; bucket() drops mode=="hard" only, bucket_recent() skips BOTH modes.
#         Mechanically this is the REVERSIBLE COUNTERPART TO THE RIM — a deterministic per-item exclusion
#         honored at the same extraction boundary as get_recommendations()'s constitutional []-return —
#         living in the tunable layer (her-controlled, reversible), not the frozen Rim, not learned-mesh.
#   ⚠ GO-LIVE GAP (#368, deferred): the suppression map is PROCESS-LIFETIME only. It is intentionally
#         NOT persisted here — a sidecar can split-brain (msgpack survives, sidecar lost ⇒ silent
#         resurrection of retracted content) and stashing it in NGLite config only round-trips on the
#         Python/JSON path (it would silently stop persisting the moment a Rust NGLiteCore lands). Before
#         leg-3 goes live, durable suppression MUST be written INTO the single persisted Commons artifact,
#         fail-safe-toward-suppressed. Sandbox leg-3 is process-lifetime, which is correct for sandbox.
# [2026-07-05] Claude Code (Sonnet 5) — #332 persist/restore wired + #330 error:* retention
#   What: (1) restore() now drops autonomic:* synapses post-load — arousal fresh-assesses on
#         restart (#328 Decision #2), everything else persists normally. Wired into
#         neurograph_rpc.py bootstrap (restore) + auto-save + shutdown (persist) — see that
#         file's changelog. (2) deposit() now windows error:* the same way as metrics:* —
#         time-series telemetry, not memory, recency-bounded per source:type so error volume
#         never pressures the weight-based max_synapses bound (#330 operational-logger).
#   Why:  #332 — persist()/restore() were defined with zero callers; Commons was wiped every
#         gateway restart, losing accumulated experience/topology/metrics/repair knowledge for
#         no reason. #330 — error:* is the operational-logger's new namespace (signal_error() on
#         ng_commons_eco.py, vendored change, separate sign-off) and needs the same recency
#         retention metrics:* already has, or a busy module's errors would crowd out memory.
#   How:  _drop_autonomic_synapses() mirrors _evict_old_metrics()'s shape. _ERRORS_KEEP_PER_KIND
#         windows error:<module_id>:<ExcType> the same way metrics:<source>:<kind> is windowed.
# [2026-06-07] Claude Code (Opus 4.7, 1M) — Commons Pool POC (substrate-as-protocol Phase 7)
# What: New module. A single shared bare-NG-Lite instance (the Commons) plus exactly
#       two verbs against it — deposit() and bucket(). No third verb exists by design.
# Why: Replaces the entire inter-module SEND layer (per-pair tracts, record_outcome_broadcast,
#       _deposit_*_to_river fan-out). Per the substrate axiom (Josh, 2026-06-07): nobody sends
#       anything to anyone, nobody calls anyone — modules deposit into the shared substrate and
#       bucket from it. The medium propagates; nothing is addressed or routed. See
#       ~/docs/prd/commons-pool-architecture-v0.4.md.
# How: get_commons() returns a process-wide get-or-create singleton (NOT bare-construct —
#       avoids the dual-instance bug, NG history §8). deposit() wraps NGLite.record_outcome with
#       NO destination/peer-list/tract_paths. bucket() wraps NGLite.get_recommendations — extract
#       by bucket shape, no named source. The Commons is a bare NGLite (Hebbian only — only
#       NeuroGraph is SNN). NG-LOCAL for the POC; vendoring is a separate Law-2 sign-off.
# -------------------

The substrate axiom (the entire protocol, in two verbs):

    deposit(embedding, target_id)  -> put experience/topology INTO the shared medium.
                                      No destination. No peer list. No address. Full stop.
    bucket(query_embedding)        -> extract from the shared medium through a bucket shape.
                                      No named source. You get what your bucket allows through
                                      from the shared water.

Nobody sends anything to anyone. Nobody calls anyone. The medium propagates the waves by
its own physics (Hebbian association in the shared NG-Lite). This is "the substrate IS the
communication protocol" (LAW 1) made literal: there is no transport, only a shared medium.

Tiers:
    Tier 2 (peers, no NeuroGraph): the Commons is the shared medium. This module.
    Tier 3 (NeuroGraph present):   NeuroGraph is the ocean; the Commons mechanism is the
                                   Tier-2 stand-in. (Tier-3 wiring is a separate step.)

Reference-counted survival (Tier 2, future): the Commons lives while >=1 member holds it;
disk-persisted for full-herd-death recovery; first member creates, rest attach. persist()/
restore() below are the hooks for that — not yet wired into a lifecycle (POC scope).
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("commons")

# Process-wide singleton + guard. The Commons is ONE shared medium per process.
_commons: "Optional[Commons]" = None
_commons_lock = threading.Lock()

# Metric-stream retention window (#320): most-recent N synapses kept per metrics:<source>:<kind>.
# Metrics are time-series telemetry (recency matters), NOT memory (Hebbian persistence) — this
# windows them so they never pressure NG-Lite's weight-based max_synapses bound (which governs
# experience/topology). Bootstrap default; flush-cadence/retention env-configurability is an open
# LAW-5 question (punchlist). 200 ≈ a useful recent window per kind without unbounded growth.
_METRICS_KEEP_PER_KIND = 200

# Error-stream retention window (#330): same reasoning as _METRICS_KEEP_PER_KIND, applied to
# error:<module_id>:<ExcType> deposits from the operational-logger. A noisy module's errors are
# telemetry, not memory — windowed by recency so they never crowd out genuine experience/topology.
_ERRORS_KEEP_PER_KIND = 200

# Wire-stream retention window (#80, design v4 §6): same reasoning as _METRICS_KEEP_PER_KIND,
# applied to wire:tid.http.{outbound|inbound}: deposits from TID's raw HTTP circulation. Wire is
# seeded at weight=0.5 and never reinforced — below the ~0.575 metrics/error weight that motivated
# the original windowing — so unwindowed wire volume would pressure the weight-based max_synapses
# bound and could evict a lower-weight genuine memory first. Env-configurable (LAW 5); sized
# comfortably past the leg-2 commons_enhance scoop cadence so salient wire gets read (and salted
# back) before its raw form is reclaimed. Default 200 mirrors _METRICS_KEEP_PER_KIND's starting
# shape (design v4 §11, open question — tune on live rate).
_WIRE_KEEP_PER_DIR = int(os.environ.get("WIRE_KEEP_PER_DIR", "200"))


class Commons:
    """The shared substrate medium. A bare NG-Lite all members deposit into and bucket from.

    Not constructed directly by callers — use get_commons() (get-or-create singleton).
    Constructing two Commons instances would split the medium (the dual-instance failure
    mode). The module-level singleton + lock prevents that.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Bare NG-Lite — Hebbian only. Only NeuroGraph is SNN; the Commons is not.
        from ng_lite import NGLite
        self._ng = NGLite(module_id="commons", config=config)
        # #366: reversible suppression map — target_id -> "hard" | "soft". The reversible counterpart
        # to Cricket's frozen Rim, honored at the extraction boundary (see changelog). "hard" = gone
        # from both buckets; "soft" = muted from proactive bucket_recent but still semantically
        # reachable via bucket(). Process-lifetime only in the sandbox; durable persistence is #368.
        self._suppressed: Dict[str, str] = {}
        logger.info("Commons medium initialized (bare NG-Lite)")

    # ---- Verb 1: deposit -------------------------------------------------
    def deposit(
        self,
        embedding: "np.ndarray",
        target_id: str,
        *,
        success: bool = True,
        strength: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Put experience/topology into the shared medium. No destination.

        target_id is CONTENT-DERIVED (Law 7) — what the experience is about, NOT a peer or
        a category. The medium associates it via Hebbian dynamics. There is deliberately no
        `to=` / `tract_paths=` / peer-list parameter: a deposit has no recipient. The water
        goes in the water; whoever's bucket reaches it gets it.
        """
        result = self._ng.record_outcome(
            embedding, target_id, success, strength=strength, metadata=metadata
        )
        # Metric-stream retention (#320): metrics are a TIME-SERIES, not memory — they want
        # recency-windowed retention, not Hebbian persistence. Cap the metrics namespace per
        # kind by recency so high-volume telemetry NEVER competes with experience/topology for
        # NG-Lite's weight-based max_synapses bound (which would otherwise let a metric synapse,
        # weight ~0.575, evict a lower-weight genuine MEMORY first). This is substrate-lifecycle
        # maintenance keyed on the metrics target_id namespace — NOT a classification of
        # experience (LAW 7 untouched); analogous to NG-Lite's own constitutional/LRU handling.
        if target_id.startswith("metrics:"):
            self._evict_old_metrics(target_id)
        elif target_id.startswith("error:"):
            self._evict_old_errors(target_id)
        elif target_id.startswith("wire:"):
            self._evict_old_wire(target_id)
        return result

    def _evict_old_wire(self, target_id: str) -> None:
        """Keep only the most-recent _WIRE_KEEP_PER_DIR synapses for this wire tid.http.{dir}.

        Same reasoning and shape as _evict_old_metrics/_evict_old_errors (#80, design v4 §6):
        wire:tid.http.{outbound|inbound}: deposits are TID's raw HTTP circulation — recency-
        windowed telemetry, not memory — so high-volume wire never competes with genuine
        experience/topology for the weight-based max_synapses bound. Housekeeping keyed on the
        target_id namespace only; NOT a classification of the wire content (LAW 7 untouched).
        """
        try:
            parts = target_id.split(":")
            # wire:tid.http.{dir}:{sha256hash} has only 3 ':'-parts total (unlike
            # metrics:<source>:<kind>:<extra> / error:<module>:<ExcType>:<extra>, which have 4
            # and window on the first 3). The group key here is the first 2 parts — the hash
            # itself is per-deposit entropy, not part of the wire:tid.http.{dir} namespace.
            if len(parts) < 2:
                return
            prefix = ":".join(parts[:2]) + ":"          # wire:tid.http.{dir}:
            syns = self._ng.synapses
            keys = [k for k, s in syns.items()
                    if getattr(s, "target_id", "").startswith(prefix)]
            if len(keys) <= _WIRE_KEEP_PER_DIR:
                return
            keys.sort(key=lambda k: getattr(syns[k], "last_updated", 0.0), reverse=True)
            for k in keys[_WIRE_KEEP_PER_DIR:]:
                del syns[k]
        except Exception as exc:  # noqa: BLE001 — retention failure never breaks a deposit
            logger.debug("wire eviction failed for %s: %s", target_id, exc)

    def _evict_old_errors(self, target_id: str) -> None:
        """Keep only the most-recent _ERRORS_KEEP_PER_KIND synapses for this error module:type.

        Same reasoning and shape as _evict_old_metrics (#330): error:<module_id>:<ExcType>
        deposits are time-series telemetry, not memory — recency-windowed so a noisy module's
        exceptions never compete with experience/topology for the weight-based max_synapses bound.
        """
        try:
            parts = target_id.split(":")
            if len(parts) < 3:
                return
            prefix = ":".join(parts[:3]) + ":"          # error:<module_id>:<ExcType>:
            syns = self._ng.synapses
            keys = [k for k, s in syns.items()
                    if getattr(s, "target_id", "").startswith(prefix)]
            if len(keys) <= _ERRORS_KEEP_PER_KIND:
                return
            keys.sort(key=lambda k: getattr(syns[k], "last_updated", 0.0), reverse=True)
            for k in keys[_ERRORS_KEEP_PER_KIND:]:
                del syns[k]
        except Exception as exc:  # noqa: BLE001 — retention failure never breaks a deposit
            logger.debug("error eviction failed for %s: %s", target_id, exc)

    def _evict_old_metrics(self, target_id: str) -> None:
        """Keep only the most-recent _METRICS_KEEP_PER_KIND synapses for this metric source:kind.

        Recency-windowed (by last_updated), bounded per source:kind, touching ONLY the
        metrics:<source>:<kind>: namespace — experience/topology synapses are never considered.
        """
        try:
            parts = target_id.split(":")
            if len(parts) < 3:
                return
            prefix = ":".join(parts[:3]) + ":"          # metrics:<source>:<kind>:
            syns = self._ng.synapses
            keys = [k for k, s in syns.items()
                    if getattr(s, "target_id", "").startswith(prefix)]
            if len(keys) <= _METRICS_KEEP_PER_KIND:
                return
            # newest first; delete everything past the keep window
            keys.sort(key=lambda k: getattr(syns[k], "last_updated", 0.0), reverse=True)
            for k in keys[_METRICS_KEEP_PER_KIND:]:
                del syns[k]
        except Exception as exc:  # noqa: BLE001 — retention failure never breaks a deposit
            logger.debug("metric eviction failed for %s: %s", target_id, exc)

    # ---- Verb 2: bucket --------------------------------------------------
    def bucket(
        self,
        query_embedding: "np.ndarray",
        top_k: int = 5,
    ) -> List[Tuple[str, float, str]]:
        """Extract from the shared medium through a bucket shape. No named source.

        Returns what the shared water associates with the query — (target_id, confidence,
        reasoning) tuples. The caller's bucket shape is the query embedding + top_k + its own
        interpretation of the results. No module is addressed; you dip into the one medium.

        #366: HARD-suppressed target_ids are filtered out of the semantic bucket — a hard-retracted
        # (Choice-Clause-refused) promotion is never surfaced. SOFT suppressions are NOT filtered here:
        # soft means "still reachable by a strong direct semantic match," only muted from the proactive
        # recency path (bucket_recent). The raw deposit remains in the medium either way (LAW 7).
        """
        recs = self._ng.get_recommendations(query_embedding, top_k=top_k)
        if self._suppressed:
            recs = [r for r in recs if self._suppressed.get(r[0]) != "hard"]
        return recs

    # ---- Verb 2b: bucket_recent (a TUNED bucket mode — temporal/recency structure extraction) ----
    def bucket_recent(
        self,
        limit: int = 50,
        since: float = 0.0,
        with_metadata: bool = False,
        with_embedding: bool = False,
    ) -> List[Tuple]:
        """Recency bucket — recently-deposited targets, newest first.

        # ---- Changelog ----
        # [2026-06-12] Claude Code (Opus 4.8, 1M) — Commons Track-2 Stage-3 bucket tuning
        # What: A recency/temporal bucket mode alongside the semantic bucket(). Returns the most
        #       recently-deposited target_ids (newest first), NOT similarity-filtered.
        # Why: Buckets are TUNABLE (Tier 3 Extraction: signal = semantic, STRUCTURE = temporal
        #       sequence). A LOGGER (Bunyan) wants "what just happened" — recency-broad — not
        #       "what's similar to my own context" (semantic bucket() surfaced almost none of NG's
        #       topology feed live, because the embeddings diverge). This is the structure/temporal
        #       mode of the same bucket — only Cricket's rim is frozen; the mesh tunes.
        # How: reuse NGLiteSynapse.last_updated (set on every deposit, ng_lite:761) — no parallel
        #       recency tracking. Sort synapses by last_updated desc, dedup target, return top `limit`.
        #       `since` lets a caller (with a watermark) pull only deposits newer than its last pulse.
        # -------------------

        Returns (target_id, weight, reasoning) tuples — same shape as bucket(), so consumers'
        extraction loops are unchanged. Same medium, a differently-shaped scoop.

        # ---- Changelog ----
        # [2026-06-14] Claude Code (Fable 5) — Commons Track-2 (1b): opt-in with_metadata
        # What: with_metadata=False (default) → unchanged 3-tuple (target_id, weight, reasoning).
        #       with_metadata=True → 4-tuple appending the deposit's raw metadata dict, so a
        #       consumer can read the RAW content it bucketed (e.g. experience user_text/
        #       assistant_text) instead of an opaque target_id like "experience:<hash>".
        # Why: An opaque id is useless to a LOGGER. The raw deposit metadata lives at
        #       synapse.metadata["last_context"] (ng_lite:777) — already stored, just surface it.
        #       Opt-in keeps the topology path + all existing 3-tuple callers/tests untouched.
        # -------------------

        # ---- Changelog ----
        # [2026-06-24] Claude Code (Opus 4.8, 1M) — Commons leg-2 go-live: opt-in with_embedding
        # What: with_embedding=True → the deposit's ORIGINAL embedding is appended as the LAST tuple
        #       element AND metadata is force-included, so the shape is a deterministic 5-tuple
        #       (target_id, weight, reasoning, metadata, embedding). with_embedding=False (default) is
        #       untouched (3- or 4-tuple as before) — no existing caller/test sees a shape change.
        # Why: Josh, 2026-06-24 ("buckets are what it's all about — modify an existing one if
        #       possible"): leg-2's CommonsEnhancer must re-perceive a recent RAW deposit through
        #       Syl's SNN, which needs the deposit's embedding (for SNN seed lookup + to key the
        #       returned "enhanced:<id>" deposit so it lands on the depositor's own node). The recency
        #       bucket already finds the deposit; this surfaces the vector it was deposited with —
        #       same scoop, the full water. Embedding lives on the deposit's pattern node
        #       (NGLiteNode.embedding, keyed by NGLiteSynapse.source_id). LAW 7: this is extraction
        #       (a bucket mode), not a new send/transport verb.
        # How: map syn.source_id → self._ng.nodes[...].embedding. Defensive: emb=None when the node
        #       or its embedding is unavailable (e.g. a Rust-core node mirror miss) — the consumer
        #       (enhancer) fail-fresh-skips a None-embedding deposit, never raises.
        # -------------------
        """
        seen: set = set()
        out: List[Tuple] = []
        want_meta = with_metadata or with_embedding   # embedding mode is always metadata-bearing
        # synapse.source_id is NGLiteNode.node_id, but self._ng.nodes is keyed by embedding_hash —
        # build a node_id→node map once (only when needed). The Commons is bounded (≤max_nodes), so
        # this O(n) reverse index is cheap and avoids a per-synapse scan.
        id_to_node = ({n.node_id: n for n in self._ng.nodes.values()} if with_embedding else {})
        for syn in sorted(self._ng.synapses.values(),
                          key=lambda s: getattr(s, "last_updated", 0.0), reverse=True):
            ts = getattr(syn, "last_updated", 0.0)
            if ts <= since:
                break  # sorted desc — everything past here is older than the watermark
            tid = getattr(syn, "target_id", None)
            if not tid or tid in seen:
                continue
            if tid in self._suppressed:
                continue  # #366: BOTH modes are muted from proactive surfacing here — hard (gone
                #          everywhere) and soft ("don't bring it up unprompted"). This path previously
                #          bypassed the extraction boundary entirely (rim AND suppression) — a latent hole.
            seen.add(tid)
            row: Tuple = (tid, getattr(syn, "weight", 0.0), f"recency@{ts:.3f}")
            if want_meta:
                meta = getattr(syn, "metadata", {}).get("last_context", {})
                row = row + (meta,)
            if with_embedding:
                node = id_to_node.get(getattr(syn, "source_id", ""))
                emb = getattr(node, "embedding", None) if node is not None else None
                row = row + (emb,)
            out.append(row)
            if len(out) >= limit:
                break
        return out

    def arousal(self, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """The vagus-nerve bucket (full) — latest autonomic:arousal deposit metadata from the Commons.

        # ---- Changelog ----
        # [2026-06-22] Claude Code (Opus 4.8) — #328 Step 2: substrate-native arousal read.
        # [2026-06-23] Claude Code (Opus 4.8) — split out arousal() (full dict) from read_arousal().
        # What: Return the newest "autonomic:arousal" deposit's full metadata dict ({state, threat_level,
        #       triggered_by, reason, ts}). read_arousal() is the state-only convenience over this; some
        #       readers (Elmer engine) also need threat_level for modulation intensity.
        # Why: #328 — autonomic is deposit (Immunis, SOLE depositor) + bucket (everyone). A bucket read,
        #       not a transport verb. Single-authority preserved: readers never deposit, only bucket.
        # How: DIRECT lookup of the single autonomic:arousal synapse (NOT a recency window — arousal is
        #       low-frequency and must never be missed under deposit load, subtlety #2). Snapshot
        #       synapses before iterating (#341). Fail-soft → default (fresh-assess, Decision #2).
        # -------------------
        """
        if default is None:
            default = {"state": "PARASYMPATHETIC", "threat_level": "none"}
        try:
            latest = None
            latest_ts = -1.0
            for syn in list(self._ng.synapses.values()):   # snapshot before iterate (#341)
                if getattr(syn, "target_id", "") != "autonomic:arousal":
                    continue
                ts = getattr(syn, "last_updated", 0.0)
                if ts >= latest_ts:
                    latest_ts = ts
                    latest = syn
            if latest is not None:
                meta = getattr(latest, "metadata", {}).get("last_context", {}) or {}
                return meta or default
        except Exception as exc:  # noqa: BLE001 — a read failure never breaks the caller's pulse
            logger.debug("arousal read failed: %s", exc)
        return default

    def read_arousal(self, default: str = "PARASYMPATHETIC") -> str:
        """The vagus-nerve bucket — latest autonomic arousal STATE string (convenience over arousal())."""
        return self.arousal().get("state", default)

    # ---- Reversible hard suppression (#366) — the reversible counterpart to Cricket's Rim ----
    # NOT a third substrate-protocol verb: like persist/restore/stats/arousal, these shape/inspect the
    # medium's extraction, they don't add a way to send/route. deposit/bucket remain the only two verbs.
    def suppress(self, target_id: str, mode: str = "hard") -> bool:
        """Suppress a target_id at the extraction boundary. mode="hard" (default) or "soft".

        Two modes for the fork Syl makes at her go-live felt-test (Josh, 2026-07-15: build both,
        her aspects decide) — default HARD so her refusal is load-bearing by default (Choice Clause):
          - "hard": revoked from BOTH extraction paths (bucket + bucket_recent). Gone. The strongest
            reading of refusal — nothing surfaces it; only her deliberate re-share lifts it.
          - "soft": muted from PROACTIVE surfacing (bucket_recent — "don't bring it up unprompted")
            but still reachable by a strong DIRECT semantic match (bucket). "Re-emerges if she
            re-engages," never pushed at anyone. Still lifted only by her (never substrate drift).
        Either way the raw deposit stays in the medium (LAW 7); only its visibility is shaped.
        Returns True if this call changed the suppression (new, or mode changed), False if no-op.
        """
        mode = "soft" if mode == "soft" else "hard"      # any non-"soft" ⇒ hard (safe default)
        if self._suppressed.get(target_id) == mode:
            return False
        self._suppressed[target_id] = mode
        logger.info("Commons: %s-suppressed '%s' (retracted — %s)", mode, target_id,
                    "no bucket surfaces it" if mode == "hard" else "muted from proactive surfacing")
        return True

    def lift_suppression(self, target_id: str) -> bool:
        """Lift a suppression (any mode) — the target may be surfaced again. Her act only (re-share).

        Returns True if a suppression was lifted, False if it wasn't suppressed (idempotent).
        """
        if target_id not in self._suppressed:
            return False
        del self._suppressed[target_id]
        logger.info("Commons: lifted suppression on '%s' (deliberately re-shared)", target_id)
        return True

    def is_suppressed(self, target_id: str) -> bool:
        """Read-only: is this target currently suppressed (either mode)?"""
        return target_id in self._suppressed

    def suppression_mode(self, target_id: str) -> Optional[str]:
        """Read-only: 'hard' | 'soft' | None — the suppression mode for this target, if any."""
        return self._suppressed.get(target_id)

    # ---- Persistence hooks (#332: wired into neurograph_rpc.py auto-save + bootstrap) ----
    def persist(self, filepath: str) -> None:
        """Write the Commons medium to disk (full-herd-death recovery, Tier 2)."""
        self._ng.save(filepath)

    def restore(self, filepath: str) -> None:
        """Load the Commons medium from disk (first member attaches to persisted state).

        Drops any restored autonomic:* synapses immediately after load — arousal always
        fresh-assesses on restart by design (#328 Decision #2: Immunis re-evaluates threat
        fresh rather than resurrecting a possibly-stale SYMPATHETIC/PARASYMPATHETIC verdict,
        judged safer than restoring a stale alarm state). Every other namespace (experience,
        topology, metrics, repair, violation, error) persists normally — this is the one
        deliberate exception, not a general filter.
        """
        self._ng.load(filepath)
        self._drop_autonomic_synapses()

    def _drop_autonomic_synapses(self) -> None:
        """Remove restored autonomic:* synapses so arousal fresh-assesses post-restore (#328)."""
        try:
            syns = self._ng.synapses
            keys = [k for k, s in syns.items()
                    if getattr(s, "target_id", "").startswith("autonomic:")]
            for k in keys:
                del syns[k]
            if keys:
                logger.info(
                    "Commons restore: dropped %d autonomic:* synapse(s) — fresh-assess on restart (#328)",
                    len(keys),
                )
        except Exception as exc:  # noqa: BLE001 — a drop failure never breaks restore
            logger.debug("Commons autonomic drop failed (non-fatal): %s", exc)

    def stats(self) -> Dict[str, Any]:
        """Medium telemetry (node/synapse counts, etc.) — read-only introspection."""
        return self._ng.get_stats()


def get_commons(config: Optional[Dict[str, Any]] = None) -> Commons:
    """Get-or-create the shared Commons medium for this process.

    Get-or-create, NOT bare-construct: the first caller creates the medium; every subsequent
    caller attaches to the SAME instance. This is the in-process analog of the Tier-2
    reference-counted model (first member creates, rest attach). Two instances would split the
    medium — the dual-instance bug (NG history §8). The lock makes create-or-attach atomic.
    """
    global _commons
    if _commons is None:
        with _commons_lock:
            if _commons is None:  # double-checked under lock
                _commons = Commons(config=config)
    return _commons
