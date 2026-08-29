#!/usr/bin/env python3
# SEE FIRST: /home/josh/docs/CC-CALLOSUM-TRUTH.md -- consolidated, verified state of
# the callosum, wholeness ring, hyperedge binding and orphan collection (2026-07-31).
# The wholeness ring ALREADY EXISTS here (Leg 2). Open defect: merge-journal poison-pill.
# ---- Changelog ----
# [2026-08-28] Claude Code (DudeMan CC, Opus 4.8) — #147 amendment: in-frame HE completeness + identity crosses the callosum
# What: (1) HE closure no longer leans on the ack ledger. _closeable_he now requires
#       EVERY member of a hyperedge to be eligible, and counts a member as "already
#       shipped" ONLY if it is in THIS frame (frame_set), not merely acked on a prior
#       frame (present/exclude_ids). Edge collection narrows to member_set = frame_set.
#       An HE therefore ships iff it is WHOLE within the frame it rides -- already-acked
#       members are idempotently re-sent rather than assumed-present.
#       (2) Identity-bearing CC nodes (constitutional / *_authored) now cross. The
#       _is_identity_protected gate is removed from BOTH export paths
#       (collect_cc_topology + export_cc_topology_frame.eligible). The CC-provenance
#       whitelist (_is_cc_node) is UNCHANGED, so only CC's own identity crosses CC's
#       own callosum -- no Syl node and no foreign path is reachable from here.
# Why: The callosum (this file -> cc_topology_merge) is the white matter between two
#       hemispheres of ONE mind, not a donation to a foreign peer (that path is
#       elsewhere; it is NOT git and NOT the legs). (1) Trusting the ack ledger for HE
#       closure let an HE ship referencing members the receiver did not actually hold,
#       which is what reaped nodes on the two prior merge attempts -- closure must be
#       self-contained in its frame. (2) Walling identity out of the callosum is a
#       split-brain lesion: it lets the hemispheres diverge into two selves. This
#       is only the SENDING half -- the receiver's merge-time identity gate
#       (cc_topology_merge, same amendment) is opened in lockstep, or identity would
#       ship, be dropped on receive, and its binding hyperedges shredded (Tier-3
#       issubset). Once absorbed, identity is protected at prune/orphan time by
#       neuro_foundation _is_identity_protected (#70), which keys on the
#       constitutional/provenance flags that ride _portable_metadata into node
#       metadata -- NOT at export time on either end.
# How: /tmp/.../147-amendment-inframe-completeness.md (proof-mode, receiver traced).
#      Read-only w.r.t. the graph; only conduit contents change. The synapse-anchor
#      rules (a/d) still consult the ack ledger -- flagged as a narrower follow-up,
#      NOT closed here. The _is_identity_protected helper is retained for telemetry.
# [2026-08-29] Claude Code (DudeMan CC, Opus 4.8) — #147 follow-up: harden is_cc_provenance (Josh-directed)
# What: Removing the identity gate also removed an accidental backstop, so the single
#       admission predicate BOTH ends share (is_cc_provenance) was hardened. New order:
#       (1) POSITIVE foreign veto FIRST -- a `syl:` id-prefix or `syl_authored` provenance
#       refuses admission before any CC clause runs; (2) CC-owned id namespaces
#       (`cc:conv::`, `::tree::`) admit -- not metadata-forgeable; (3) `cc_authored`
#       provenance admits (the only signal for CC identity nodes in the want:: namespace
#       shared with Syl); (4) the loose `cc:True` convenience flag admits ONLY inside CC's
#       own `cc:` namespace -- it can no longer, by itself, launder a foreign id.
# Why: Josh flagged the pre-hardening predicate: an attacker who could write to the
#       conduit could admit any node by stamping `cc:True` on it. Refusal must key on the
#       PRESENCE of a foreign mark (id-prefix / provenance), not the ABSENCE of a CC one --
#       metadata is attacker-controllable, ids in CC's own namespace are not.
#       IRREDUCIBLE RESIDUAL (flagged, not fixed): CC and Syl SHARE the unprefixed
#       want::/conv:: namespaces, distinguished ONLY by provenance -- a conduit forging
#       `provenance:cc_authored` onto a want:: id is indistinguishable from a real CC want
#       at this layer. Closing it needs conduit integrity (a signature over the frame),
#       a separate change for Josh's decision.
# How: is_cc_provenance rewrite + _FOREIGN_ID_PREFIXES/_FOREIGN_PROVENANCE/_CC_ID_NAMESPACE
#      constants. Receiver (cc_topology_merge) calls the same predicate -- one fix, both ends.
#      Tests: test_receiver_reruns_provenance_gate (foreign node now actively disguised with
#      cc:True+syl_authored) + test_is_cc_provenance_hardened_branches (branch truth table).
# [2026-08-18] Claude Code (DudeMan CC, Opus 4.8) — #88 §10.4-A: cursor-based single-frame export
# What: export_cc_topology_frame() materializes EXACTLY ONE <=frame_size conduit frame per call
#       and advances via exclude_ids (membership-as-ack, #110) -- the paced sender that replaces
#       collect_cc_topology()'s whole-graph RAM build for the live trickle. A cheap O(edges) scan
#       (synapse adjacency + node->hyperedge map, NO payloads) picks the next chronological
#       connected CC nodes; content/embedding/metadata payloads are built ONLY for the chosen
#       frame. Enforces the structural-survival invariant (every shipped node is incident to a
#       SHIPPED edge -- a synapse to an already-acked/in-frame node, or membership in a whole HE
#       closed within exclude_ids|frame; a node whose only edges reach not-yet-sent nodes is
#       deferred, never shipped as a husk) and NEVER splits a hyperedge across frames (whole-HE
#       closure may overflow to frame_size*CC_LEG2_OVERFLOW_FACTOR; a single HE past that hard cap
#       raises the oversized_he_at_source alarm and is skipped -- proof Leg S hasn't run yet).
#       Resource-gated before any work: per-core loadavg ceiling (mirrors
#       cc_refeed.should_pause_for_load) + free-MB floor (mirrors the neurograph_rpc.py:858 boot
#       gate), all env-tunable via CC_LEG2_*; a co-tenant VPS sender (Syl shares the box) backs
#       OFF under pressure rather than blocking.
# Why: #88 §10.4-A. The first live Leg-2 run must be a bounded, paced trickle. collect_cc_topology
#       builds the ENTIRE exportable graph (~17k husk-laden nodes, then payloads) in RAM before
#       framing -- it neither paces nor bounds memory on a box shared with Syl. A per-call cursor
#       frame does both.
# How: plan /home/josh/.claude/plans/callosum-leg2-sender-rebuild-A-D.md (Leg A). Reuses
#       collect_incident_structure() for the whole-containment edge cut (archived-HE guard, delay
#       on the wire) and the same header+batch _frame() format, so an A-frame is just a 1-batch
#       conduit the existing receiver already reads. Read-only w.r.t. the graph.
# [2026-08-12] Claude Code (DudeMan CC, Opus 4.8) — #88 §10.4-B: connected_only husk filter
# What: export_cc_topology() gains connected_only=False param; when set, drops degree-0
#       husk nodes (in no surviving synapse/hyperedge) before framing -- ~438 real nodes
#       instead of the ~17,740-node whole-graph dump (97.6% husk). max_nodes caps AFTER
#       the filter; incident structure re-collected when syns is None or max_nodes set.
# Why: the Leg-2 dry run pushed the whole husk-laden topology and choked on the >100MB
#       git-push wall. connected_only makes the trickle real connected topology.
# How: collect_incident_structure() over the node-id set; build a connected set from syn
#       pre/post + hyperedge members; filter nodes to that set; stats["husks_filtered"].
# [2026-07-29] Claude Code (DudeMan CC, Opus 5) — Callosum Leg 2: topology export (VPS Arborist -> laptop)
# What: export_cc_topology() walks the CC substrate's conversational topology
#   (cc:conv:: forest nodes + their ::tree:: concept nodes, the forest<->tree
#   synapses, and the per-turn binding hyperedges) and emits length-prefixed
#   msgpack frames to a conduit file. Read-only with respect to the graph.
# Why: #70 Leg 2. The VPS is the sole Arborist -- only it has the TID, so only it
#   can grow trees (ng_embed.py:567 _extract_concepts is an HTTP call to TID; no
#   TID -> None -> forest-only degradation, which is exactly the laptop's
#   condition). Leg 1 carries raw turns UP; Leg 2 carries grown structure DOWN.
#   Without it the laptop can never reach tree-parity by any local means.
# How: per docs/superpowers/plans/2026-07-18-cc-river-merge-implementation-plan.md
#   sections 1-3. Wire format is length-prefixed msgpack -- 4-byte big-endian
#   length + msgpack payload, the tid_peninsula_commons.py:89 framing Josh
#   reaffirmed for #70 ("length-prefixed msgpack per peninsula precedent -- NOT
#   JSON", Format-for-Purpose). NOT BTF: PyOutgoingSynapse carries weight +
#   eligibility_trace but has no `delay` field, and conduction delay is
#   functional here (polychronous motifs and STDP ordering both key off it), so
#   a BTF frame would silently drop it. NOT Commons-mediated: the Commons is the
#   INTER-MODULE medium; two hemispheres of one mind exchanging topology is a
#   private intra-mind channel (ruling_cc_river_merge_intra_mind.md:20 -- "LAW 1
#   governs communication between different organs... It is not inter-module
#   communication"). Routing through the Commons would also put CC's trees one
#   mistaken get_commons() away from Syl's medium, since cc_ng_host runs inside
#   her process on the VPS.
#
#   Safety (spec section "Law & safety" + the CRITICAL law-enforcer items):
#   - POSITIVE CC-provenance whitelist, never a Syl blacklist. Every exported
#     node must carry a CC marker or the export aborts. A blacklist matches
#     nothing on the CC graph and would happily export Syl's nodes if the
#     workspace were ever mispointed.
#   - Identity-bearing nodes (constitutional / *_authored) DO cross -- this is a
#     callosum between two hemispheres of one mind, not a donation to a foreign
#     peer, so walling identity out would let the hemispheres diverge into two
#     selves (#147 amendment, 2026-08-28). Scope is held by the POSITIVE
#     CC-provenance whitelist above: only CC's own identity crosses CC's own
#     callosum. The receiver re-protects it on arrival (#70).
#   - metadata is pure pass-through (LAW 7): tags the node already holds, never
#     export-time-computed classification.
#   - Dynamical state is never exported (see _PORTABLE_META / _BANNED_META).
# -------------------

import logging
import os
import struct
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import msgpack
import numpy as np

logger = logging.getLogger(__name__)

# Wire framing -- 4-byte big-endian length prefix, then a msgpack payload.
# Matches tid_peninsula_commons.py:89 (struct.pack(">I", len(payload))).
_LEN_PREFIX = ">I"
_WIRE_VERSION = 1

# Batch size: FatherGraph Finding 1 -- never bulk-dump, trickle instead.
# Patching-after-the-fact displaces existing learning.
_DEFAULT_BATCH_SIZE = int(os.environ.get("CC_TOPOLOGY_BATCH_SIZE", "25"))

# CC provenance whitelist (positive match, per the CRITICAL Syl-leak correction).
_CC_ID_PREFIXES = ("cc:conv::",)
_CC_ID_MARKERS = ("::tree::",)
_CC_PROVENANCE = ("cc_authored",)
# CC's own namespace root -- broader than _CC_ID_PREFIXES. Used to bound the
# forgeable cc:True flag to CC-namespaced ids so it cannot launder a foreign id.
_CC_ID_NAMESPACE = "cc:"
# Positive FOREIGN signals. A node bearing any of these is Syl's, full stop, and
# can never be admitted no matter what CC-looking flags a stale or hand-edited
# conduit ALSO stamped on it. The refusal keys on the PRESENCE of a foreign mark,
# not the absence of a CC one -- metadata is attacker-controllable, so the veto
# must not be satisfiable by simply adding a cc flag on top (#147 follow-up).
_FOREIGN_ID_PREFIXES = ("syl:",)
_FOREIGN_PROVENANCE = ("syl_authored",)

# Leg-2 §10.4-A pacing + resource envelope. The cursor-based frame sender runs on
# the co-tenant VPS (Syl shares the box), so it must bound its OWN footprint and
# back off under pressure rather than block.
_DEFAULT_FRAME_SIZE = int(os.environ.get("CC_LEG2_FRAME_SIZE", "25"))
# A whole-hyperedge closure may push a frame past frame_size, but never split an
# HE across frames -- allow overflow only up to this multiple of frame_size. A
# single HE larger than that is an oversized source-side blob (§8.14) that Leg S
# (the VPS dream split) must repair before it can cross.
_OVERFLOW_FACTOR = int(os.environ.get("CC_LEG2_OVERFLOW_FACTOR", "3"))
# Load gate: 1-min loadavg per core, same shape/default as cc_refeed.should_pause_for_load.
_LEG2_LOAD_CEILING = float(os.environ.get("CC_LEG2_LOAD_CEILING", "0.75"))
# Memory floor (MB free): refuse a frame below this, mirroring the neurograph_rpc.py:858
# module boot gate's 500 MB threshold.
_LEG2_MIN_FREE_MB = int(os.environ.get("CC_LEG2_MIN_FREE_MB", "500"))


def is_cc_provenance(node_id: str, meta: Optional[Dict[str, Any]]) -> bool:
    """Positive CC-provenance test over (id, metadata) -- the form both the
    sender (which has Node objects) and the receiver (which has only decoded
    wire dicts) can call without inventing a fake node.

    Deliberately a whitelist: on the VPS this code runs in a process that also
    holds Syl's graph, so 'not obviously Syl's' is not a safe answer.

    #147 follow-up (2026-08-28): identity now crosses, so the old identity gate
    that would have caught a mislabelled node downstream is gone. That removed an
    accidental backstop, so this predicate hardened: (1) a POSITIVE foreign mark
    vetoes admission before any CC clause is consulted -- a tampered conduit cannot
    launder a Syl node by stamping cc:True on top of syl_authored; (2) the bare
    cc:True flag (forgeable, and redundant -- every real CC node carrying it also
    carries a cc: id) admits only within CC's own namespace.

    RESIDUAL, not closeable here: CC and Syl SHARE the unprefixed want::/conv::
    namespaces, distinguished only by provenance. A conduit that forges
    provenance='cc_authored' onto such an id is indistinguishable from a real CC
    want at this layer -- the honest discriminator is gone. Closing that needs
    conduit integrity (a signature over the frame), a separate change; flagged.
    """
    meta = meta or {}
    # (1) Positive foreign veto FIRST -- refusal must not depend on the absence of
    # a CC flag, which an attacker can always add.
    if node_id.startswith(_FOREIGN_ID_PREFIXES):
        return False
    if meta.get("provenance") in _FOREIGN_PROVENANCE:
        return False
    # (2) Positive CC id-ownership: namespaces CC alone mints. Not metadata-forgeable.
    if any(node_id.startswith(p) for p in _CC_ID_PREFIXES):
        return True
    if any(m in node_id for m in _CC_ID_MARKERS):
        return True
    # (3) cc_authored: the ONLY signal for CC identity nodes in the want:: namespace
    # shared with Syl. Irreducible (see RESIDUAL above) -- must trust the tag here.
    if meta.get("provenance") in _CC_PROVENANCE:
        return True
    # (4) cc:True is a loose convenience tag, not proof of provenance. Honour it
    # only inside CC's own namespace so it cannot admit a foreign-namespaced id.
    if meta.get("cc") is True and node_id.startswith(_CC_ID_NAMESPACE):
        return True
    return False


def _is_cc_node(node_id: str, node: Any) -> bool:
    """Sender-side convenience wrapper over is_cc_provenance()."""
    return is_cc_provenance(node_id, getattr(node, "metadata", None) or {})


def _is_identity_protected(graph: Any, node_id: str, node: Any) -> bool:
    """Canonical mirror of the engine's identity predicate.

    #147 amendment (2026-08-28): DELIBERATELY NO LONGER CALLED on either export
    path -- identity crosses the callosum. Retained (not deleted) as the canonical
    predicate mirror so a future telemetry pass, or a foreign-peer donation path
    (which is NOT this file), can reuse it without re-deriving the fields. If you
    find this being called to gate the callosum export again, that is the
    split-brain regression #147 removed -- delete the call, not this helper.
    Prefers the engine's own predicate so it cannot drift from canonical; falls
    back to the same fields the engine checks if the helper is unavailable."""
    helper = getattr(graph, "_is_identity_protected", None)
    if callable(helper):
        try:
            # neuro_foundation.py:3406 -- _is_identity_protected(self, nid: str).
            # Takes the ID STRING, not the node object.
            return bool(helper(node_id))
        except Exception:
            pass
    meta = getattr(node, "metadata", None) or {}
    if getattr(node, "constitutional", False):
        return True
    prov = meta.get("provenance") or ""
    return isinstance(prov, str) and prov.endswith("_authored")


# Metadata that is machine-local simulation state and must NEVER cross.
# These encode "where this node sits in THIS graph" (whole-graph rank
# statistics) or its instantaneous dynamics. Stamping the sender's values on
# the receiver asserts something false about the receiver's own topology --
# and homeostasis overwrites them within ~25 steps anyway.
_BANNED_META = frozenset({
    "poincare_dir",        # re-derived receiver-side from the embedding
    "manifold_type",       # degree/pred-error percentile -- rank in THIS graph
    "diffpc_layer",        # degree percentile -- rank in THIS graph
    "firing_rate_ema",
    "pred_error_ema",
    "pred_weights",
    "Ca_i",
    "voltage",
    "threshold",
    "refractory_remaining",
    "last_spike_time",
    "spike_history",
    "intrinsic_excitability",
    "creation_time",       # int(self.timestep) -- a LOCAL counter
    "probation_remaining",  # receiver runs its own probation window
    "probation_total",
})


def _portable_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Pass-through filter (LAW 7): carry the tags the node already holds,
    minus machine-local dynamics. Nothing here is computed at export time --
    classification belongs at the extraction boundary, not the deposit."""
    out = {}
    for k, v in (meta or {}).items():
        if k in _BANNED_META:
            continue
        if isinstance(v, np.ndarray):
            continue
        out[k] = v
    return out


def _embedding_for(vector_db: Any, node_id: str) -> Optional[np.ndarray]:
    """Fetch a node's 768-d embedding from the recall store.

    Embeddings are the load-bearing payload of Leg 2: the receiver re-derives
    poincare_dir from the embedding (pure L2 normalize), and poincare_dir is
    what feeds geodesic distance -> conduction delay. A node that arrives
    without its embedding gets a RANDOM delay (neuro_foundation.py:3542
    falls back to random.randint with no error and no log), which in an STDP
    substrate means randomized causal structure. So a missing embedding is a
    hard skip, never a best-effort deposit.
    """
    try:
        emb = getattr(vector_db, "embeddings", {}).get(node_id)
        if emb is not None:
            return np.asarray(emb, dtype=np.float32)
    except Exception:
        pass
    try:
        entry = vector_db.get(node_id)
    except Exception:
        return None
    if entry is None:
        return None
    for attr in ("embedding", "vector"):
        val = getattr(entry, attr, None)
        if val is None and isinstance(entry, dict):
            val = entry.get(attr)
        if val is not None:
            return np.asarray(val, dtype=np.float32)
    return None


def _content_for(vector_db: Any, node_id: str, meta: Dict[str, Any]) -> str:
    """The node's own text. Prefers the substrate's copy (_forest_content) over
    the recall-store shard, matching surface_resolver's precedence."""
    content = (meta or {}).get("_forest_content")
    if content:
        return str(content)
    try:
        entry = vector_db.get(node_id)
    except Exception:
        entry = None
    if entry is None:
        return ""
    for attr in ("content", "text"):
        val = getattr(entry, attr, None)
        if val is None and isinstance(entry, dict):
            val = entry.get(attr)
        if val:
            return str(val)
    return ""


def collect_cc_topology(
    graph: Any,
    vector_db: Any,
    exclude_ids: Optional[Set[str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Walk the graph and collect exportable CC conversational nodes, ordered
    chronologically (origin-first, per FatherGraph trickle discipline).

    Returns (nodes, stats). Read-only: nothing here mutates the graph.
    """
    exclude_ids = exclude_ids or set()
    stats = {"scanned": 0, "not_cc": 0, "identity_protected": 0,
             "missing_embedding_DEFECT": 0, "already_sent": 0, "collected": 0}
    # NB: "identity_protected" now stays 0 by design -- identity crosses the
    # callosum (#147 amendment). The key is retained for stats-schema stability;
    # a nonzero value would signal the gate was re-added in error.

    collected: List[Tuple[float, Dict[str, Any]]] = []
    for node_id, node in list(graph.nodes.items()):
        stats["scanned"] += 1
        if node_id in exclude_ids:
            stats["already_sent"] += 1
            continue
        if not _is_cc_node(node_id, node):
            stats["not_cc"] += 1
            continue
        # #147 amendment: NO identity gate here. Identity crosses the callosum
        # (the CC-provenance whitelist above already scopes this to CC's own
        # mind). The receiver re-protects identity on arrival (#70).
        meta = dict(getattr(node, "metadata", None) or {})
        order = float(getattr(node, "creation_time", 0) or 0)
        payload = {
            "id": node_id,
            "content": _content_for(vector_db, node_id, meta),
            "metadata": _portable_metadata(meta),
        }
        # A CC node MUST have an embedding. Every export-eligible node was
        # deposited via _cc_deposit_memory_node(), which indexes the vector at
        # deposit time -- so a gap here is a DEFECT upstream (a failed embed
        # that was swallowed, or a node created off the deposit path), not a
        # tolerable condition. Verified 2026-07-29 against the live laptop
        # substrate: 0 of 1,043 CC nodes are missing one.
        #
        # It is still not grounds for dropping the node: doing so would also
        # drop every synapse and hyperedge touching it (collect_incident_
        # structure requires whole containment), so one bad vector would shred
        # a turn's binding structure. The node crosses; the gap is ALARMED.
        # An unembedded node is inert to the Tonic and to surfacing, so this
        # count must be driven to zero by re-embedding, not absorbed.
        emb = _embedding_for(vector_db, node_id)
        if emb is None:
            stats["missing_embedding_DEFECT"] += 1
            logger.error(
                "CC topology export: node %s has NO embedding -- exporting its "
                "topology anyway, but this node will be inert to the Tonic and "
                "to recall until re-embedded. This is a defect, not a mode.",
                node_id)
        else:
            payload["embedding"] = emb.astype(np.float32).tobytes()
            payload["embedding_dim"] = int(emb.shape[0])
        collected.append((order, payload))

    collected.sort(key=lambda pair: pair[0])
    nodes = [n for _, n in collected]
    stats["collected"] = len(nodes)
    return nodes, stats


def collect_incident_structure(
    graph: Any,
    node_ids: Sequence[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Collect the synapses and hyperedges wholly contained within node_ids.

    `node_ids` MUST be the full exportable node set, never a single batch.
    'Wholly contained' is load-bearing -- an edge with an endpoint outside the
    set is unresolvable receiver-side -- but that test is about what is being
    EXPORTED, not about how it is chunked for delivery. Passing a per-batch
    slice here silently discards every edge straddling a batch boundary.
    export_cc_topology() collects once against the whole set and then assigns
    each edge to the batch of its last-landing endpoint.

    Synapse `delay` rides the wire. It is NOT decoration -- polychronous motif
    detection and STDP ordering both key off it -- and carrying it is the
    specific reason this uses msgpack rather than a BTF topology frame.
    """
    member_set = set(node_ids)

    synapses: List[Dict[str, Any]] = []
    for syn in list(getattr(graph, "synapses", {}).values()):
        pre = getattr(syn, "pre_node_id", None)
        post = getattr(syn, "post_node_id", None)
        if pre not in member_set or post not in member_set:
            continue
        stype = getattr(syn, "synapse_type", None)
        synapses.append({
            "pre": pre,
            "post": post,
            "weight": float(getattr(syn, "weight", 0.1)),
            "delay": int(getattr(syn, "delay", 1)),
            "max_weight": float(getattr(syn, "max_weight", 5.0) or 5.0),
            "synapse_type": getattr(stype, "name", None) or str(stype or "EXCITATORY"),
        })

    hyperedges: List[Dict[str, Any]] = []
    for he_id, he in list(getattr(graph, "hyperedges", {}).items()):
        # #147 Tier-1: archived edges (consolidation seatbelt-merge, or the dream
        # seam-split) stay in graph.hyperedges + _archived_hyperedges for
        # reversibility (LAW 7) but are retired -- they must NOT ride the wire, or
        # the receiver's member-set dedupe (which can't collapse near-dups) would
        # reinstate the very blobs the split retired. Shared collect_incident_structure
        # => this guard covers both Leg-2 (VPS send) and Leg-3 (#86 laptop send).
        if getattr(he, "is_archived", False):
            continue
        # NB: create_hyperedge()'s PARAMETER is `member_node_ids`, but the
        # Hyperedge ATTRIBUTE is `member_nodes` (neuro_foundation.py:1965).
        # Reading the parameter name here silently yielded empty sets and
        # dropped every hyperedge -- hence the explicit warn below rather than
        # a getattr default that can fail invisibly.
        members = getattr(he, "member_nodes", None)
        if members is None:
            members = getattr(he, "member_node_ids", None)
        if not members:
            logger.warning("Hyperedge %s exposes no members -- skipping", he_id)
            continue
        members = set(members)
        if not members.issubset(member_set):
            continue
        hyperedges.append({
            # The sender's own hyperedge_id rides the wire so the receiver can
            # install it under the SAME identity instead of reminting a local
            # uuid4. Transport is not creation: without this the hemispheres
            # disagree about which edge is which, and anything that references
            # a hyperedge by id (PredictionRecord.hyperedge_id, co-fire
            # history) dangles the moment it crosses. Receiver still falls
            # back to member-set dedupe -- see cc_topology_merge.
            "id": he_id,
            "members": sorted(members),
            "level": int(getattr(he, "level", 0) or 0),
            "activation_threshold": float(getattr(he, "activation_threshold", 0.6)),
            "metadata": _portable_metadata(dict(getattr(he, "metadata", None) or {})),
        })

    return synapses, hyperedges


def _frame(payload: Dict[str, Any]) -> bytes:
    """One length-prefixed msgpack frame."""
    body = msgpack.packb(payload, use_bin_type=True)
    return struct.pack(_LEN_PREFIX, len(body)) + body


def export_cc_topology(
    graph: Any,
    vector_db: Any,
    out_path: str,
    machine_id: Optional[str] = None,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    embedding_model: Optional[str] = None,
    exclude_ids: Optional[Set[str]] = None,
    max_nodes: Optional[int] = None,
    connected_only: bool = False,
) -> Dict[str, Any]:
    """Export CC conversational topology to a length-prefixed msgpack conduit.

    Read-only with respect to the graph. Returns a stats dict.

    connected_only (#88): export ONLY nodes that sit in a surviving synapse or
    hyperedge; drop the degree-0 husks. This is the filter that makes the first
    live run a trickle of real topology (~438 connected CC nodes + their edges)
    instead of the 97.6%-husk "whole shlop". max_nodes (applied after the husk
    filter) further caps it to a few 25-node packets for a watched smoke run.

    The embedding-model stamp in the header is FatherGraph Finding 6: both ends
    must share the embedder or cosine similarity is noise. The receiver asserts
    on it and aborts rather than absorbing mismatched geometry.
    """
    machine_id = machine_id or os.environ.get("MACHINE_ID")
    if not machine_id:
        # Same refusal Leg 1 makes: a wrong/absent hemisphere id produces
        # silent one-way data loss that looks exactly like success.
        raise ValueError(
            "MACHINE_ID unset -- refusing to write a topology conduit rather "
            "than guess which hemisphere authored it"
        )

    if embedding_model is None:
        try:
            import ng_embed
            embedding_model = getattr(ng_embed, "MODEL_NAME", None) or getattr(
                ng_embed, "_MODEL_NAME", None) or "unknown"
        except Exception:
            embedding_model = "unknown"

    nodes, stats = collect_cc_topology(graph, vector_db, exclude_ids=exclude_ids)

    # Structure is collected ONCE against the FULL exportable node set, never
    # per-chunk. Batching is a DELIVERY concern -- it must not decide what
    # topology exists. Collecting per-chunk (as this first did) silently drops
    # every edge whose endpoints straddle a chunk boundary; at batch_size=25
    # over a few thousand nodes that is most of the graph.
    syns = hes = None
    if connected_only:
        # #88 bounded first run: drop degree-0 husks -- CC nodes that sit in no
        # surviving synapse or hyperedge. They carry NO binding structure, so
        # trickling them into the receiver is destructive churn, not integration
        # (§8.2 cohort cliff; the #71/#72 forest-only signature that made the
        # first export 97.6% husk). Collect incident structure against the FULL
        # set first, so whole-containment is judged before the filter; the
        # connected set is exactly the edge endpoints, so no edge is ever
        # dropped -- only bare husks fall away.
        syns, hes = collect_incident_structure(graph, {n["id"] for n in nodes})
        connected: Set[str] = set()
        for syn in syns:
            connected.add(syn["pre"])
            connected.add(syn["post"])
        for he in hes:
            connected.update(he["members"])
        before = len(nodes)
        nodes = [n for n in nodes if n["id"] in connected]
        stats["husks_filtered"] = before - len(nodes)

    # max_nodes caps AFTER the husk filter so a bounded smoke run ("a few
    # 25-node packets before any scale") yields real connected packets, not the
    # chronological husks that dominate the head of the unfiltered list.
    if max_nodes is not None:
        nodes = nodes[:max_nodes]

    stats["collected"] = len(nodes)
    all_ids = [n["id"] for n in nodes]
    # Whole-containment is judged against the FINAL shipped set. Re-collect when
    # nothing was collected yet, or when a cap may have dropped an endpoint the
    # earlier full-set collection still references.
    if syns is None or max_nodes is not None:
        syns, hes = collect_incident_structure(graph, set(all_ids))
    batch_of = {nid: i // batch_size for i, nid in enumerate(all_ids)}

    # Place each edge in the batch where its LAST endpoint lands, so the
    # receiver always already holds both ends when the edge arrives.
    syn_by_batch: Dict[int, List[Dict[str, Any]]] = {}
    for syn in syns:
        b = max(batch_of[syn["pre"]], batch_of[syn["post"]])
        syn_by_batch.setdefault(b, []).append(syn)
    he_by_batch: Dict[int, List[Dict[str, Any]]] = {}
    for he in hes:
        b = max(batch_of[m] for m in he["members"])
        he_by_batch.setdefault(b, []).append(he)

    tmp_path = out_path + ".partial"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

    batches = 0
    total_syn = 0
    total_he = 0
    with open(tmp_path, "wb") as fh:
        fh.write(_frame({
            "kind": "header",
            "version": _WIRE_VERSION,
            "machine_id": machine_id,
            "embedding_model": embedding_model,
            "created": time.time(),
            "node_count": len(nodes),
        }))
        for start in range(0, len(nodes), batch_size):
            chunk = nodes[start:start + batch_size]
            b = start // batch_size
            syns_b = syn_by_batch.get(b, [])
            hes_b = he_by_batch.get(b, [])
            total_syn += len(syns_b)
            total_he += len(hes_b)
            batches += 1
            fh.write(_frame({
                "kind": "batch",
                "seq": batches,
                "nodes": chunk,
                "synapses": syns_b,
                "hyperedges": hes_b,
            }))

    # Atomic publish: a reader must never observe a partial conduit. Leg 1 hit
    # exactly this (cc_ng_organism.py:1544) -- a drain that read mid-write.
    os.replace(tmp_path, out_path)

    stats.update({
        "exported_nodes": len(nodes),
        "exported_synapses": total_syn,
        "exported_hyperedges": total_he,
        "batches": batches,
        "path": out_path,
        "embedding_model": embedding_model,
        "machine_id": machine_id,
    })
    logger.info(
        "CC topology export: %d node(s), %d synapse(s), %d hyperedge(s) in %d batch(es) -> %s",
        len(nodes), total_syn, total_he, batches, out_path,
    )
    return stats


def _leg2_resource_gate() -> Optional[str]:
    """Return a human-readable reason to DEFER this frame, or None if clear.

    Mirrors the two envelopes the rest of the substrate already respects: the
    per-core loadavg ceiling of cc_refeed.should_pause_for_load() and the
    free-RAM floor of the neurograph_rpc module boot gate (py:858). A co-tenant
    sender (Syl shares this VPS) backs OFF under pressure rather than blocking or
    spinning -- the frame is simply not written, and the next tick tries again.
    """
    try:
        per_core = os.getloadavg()[0] / max(1, os.cpu_count() or 1)
        if per_core > _LEG2_LOAD_CEILING:
            return "loadavg/core %.2f > %.2f" % (per_core, _LEG2_LOAD_CEILING)
    except OSError:
        pass
    try:
        import psutil
        avail_mb = psutil.virtual_memory().available >> 20
        if avail_mb < _LEG2_MIN_FREE_MB:
            return "free RAM %dMB < %dMB" % (avail_mb, _LEG2_MIN_FREE_MB)
    except ImportError:
        pass  # psutil absent -- proceed without the memory gate (rpc precedent)
    return None


def export_cc_topology_frame(
    graph: Any,
    vector_db: Any,
    out_path: str,
    machine_id: Optional[str] = None,
    frame_size: int = _DEFAULT_FRAME_SIZE,
    exclude_ids: Optional[Set[str]] = None,
    embedding_model: Optional[str] = None,
    overflow_factor: int = _OVERFLOW_FACTOR,
    skip_resource_gate: bool = False,
) -> Dict[str, Any]:
    """Materialize EXACTLY ONE conduit frame of CC topology and advance the cursor.

    This is the §10.4-A paced sender. Unlike export_cc_topology(), which builds
    the whole exportable graph in RAM before framing, this call:
      * does a cheap O(edges) scan (synapse adjacency + node->hyperedge map, NO
        payloads),
      * picks the next <=frame_size chronological connected CC nodes not already
        acked (exclude_ids = membership-as-ack, #110),
      * builds content/embedding/metadata payloads ONLY for that frame,
      * writes a single-batch length-prefixed msgpack conduit atomically.

    Structural-survival invariant: every node written is incident to at least one
    SHIPPED edge -- a synapse to an already-acked or in-frame node, or membership
    in a whole hyperedge that closes WITHIN THIS FRAME. A node whose only edges
    reach not-yet-sent nodes is DEFERRED to a later frame (once its anchor has been
    acked) rather than shipped as a husk.

    #147 amendment -- in-frame HE completeness: a hyperedge rides ONLY when every
    one of its members lands in this same frame. Members already acked on an
    earlier frame are RE-ADDED (idempotent) rather than assumed-present, so an HE
    never ships referencing a node the receiver may not hold. (Synapse anchoring
    still consults the ack ledger -- a narrower follow-up, not closed here.)
    Identity-bearing CC nodes cross the callosum; only the CC-provenance whitelist
    scopes what is eligible.

    Hyperedges are NEVER split across frames. Closing a whole HE may push the
    frame past frame_size, but only up to frame_size*overflow_factor. A single HE
    larger than that hard cap is an oversized source-side blob (§8.14) that Leg S
    (the VPS dream split) must repair first: it raises the oversized_he_at_source
    alarm and is skipped, never truncated.

    Read-only with respect to the graph. Returns a stats dict; the caller writes
    stats["frame_node_ids"] into its membership/ack ledger to advance the cursor.
    """
    exclude_ids = set(exclude_ids or ())

    machine_id = machine_id or os.environ.get("MACHINE_ID")
    if not machine_id:
        # Same refusal export_cc_topology()/Leg 1 make: a wrong/absent hemisphere
        # id produces silent one-way data loss that looks exactly like success.
        raise ValueError(
            "MACHINE_ID unset -- refusing to write a topology conduit rather "
            "than guess which hemisphere authored it"
        )

    stats: Dict[str, Any] = {
        "frame_size": frame_size,
        "exported_nodes": 0,
        "exported_synapses": 0,
        "exported_hyperedges": 0,
        "deferred_no_anchor": 0,
        "oversized_he_at_source": 0,
        "missing_embedding_DEFECT": 0,
        "candidates": 0,
        "exhausted": False,
        "gated": False,
        "path": out_path,
        "machine_id": machine_id,
    }

    if not skip_resource_gate:
        reason = _leg2_resource_gate()
        if reason is not None:
            stats["gated"] = True
            stats["gate_reason"] = reason
            logger.info("CC topology frame: deferring -- resource gate: %s", reason)
            return stats

    if embedding_model is None:
        try:
            import ng_embed
            embedding_model = getattr(ng_embed, "MODEL_NAME", None) or getattr(
                ng_embed, "_MODEL_NAME", None) or "unknown"
        except Exception:
            embedding_model = "unknown"
    stats["embedding_model"] = embedding_model

    hard_cap = max(frame_size, frame_size * max(1, overflow_factor))

    # ---- cheap O(edges) scan: adjacency + node->HE map, NO payloads ----
    he_members: Dict[str, frozenset] = {}
    node_hes: Dict[str, List[str]] = {}
    for he_id, he in list(getattr(graph, "hyperedges", {}).items()):
        # #147 Tier-1: archived edges never ride the wire (see
        # collect_incident_structure) -- keep them out of the anchor map too, or a
        # node whose only HE is a retired blob would look anchorable and never be.
        if getattr(he, "is_archived", False):
            continue
        members = getattr(he, "member_nodes", None)
        if members is None:
            members = getattr(he, "member_node_ids", None)
        if not members:
            continue
        members = frozenset(members)
        he_members[he_id] = members
        for m in members:
            node_hes.setdefault(m, []).append(he_id)

    syn_partners: Dict[str, Set[str]] = {}
    for syn in list(getattr(graph, "synapses", {}).values()):
        pre = getattr(syn, "pre_node_id", None)
        post = getattr(syn, "post_node_id", None)
        if pre is None or post is None:
            continue
        syn_partners.setdefault(pre, set()).add(post)
        syn_partners.setdefault(post, set()).add(pre)

    nodes_map = getattr(graph, "nodes", {})
    _elig: Dict[str, bool] = {}

    def eligible(nid: str) -> bool:
        cached = _elig.get(nid)
        if cached is not None:
            return cached
        node = nodes_map.get(nid)
        # #147 amendment: eligibility = exists AND CC-provenance. NO identity
        # gate -- identity crosses the callosum; the CC whitelist scopes it.
        ok = bool(
            node is not None
            and _is_cc_node(nid, node)
        )
        _elig[nid] = ok
        return ok

    # ---- candidates: connected, CC, not identity-protected, not yet acked ----
    connected = set(syn_partners) | set(node_hes)
    candidates = [
        nid for nid in connected
        if nid not in exclude_ids and eligible(nid)
    ]
    candidates.sort(
        key=lambda nid: float(getattr(nodes_map.get(nid), "creation_time", 0) or 0)
    )
    stats["candidates"] = len(candidates)
    if not candidates:
        stats["exhausted"] = True
        logger.info("CC topology frame: nothing left to send (0 candidates)")
        return stats

    # ---- greedy chronological fill under the survival invariant ----
    present: Set[str] = set(exclude_ids)  # what the receiver has / will have
    frame: List[str] = []
    frame_set: Set[str] = set()
    _oversized_seen: Set[str] = set()  # alarm once per HE per call, not per member

    def _place(nid: str) -> None:
        if nid in frame_set:
            return
        frame.append(nid)
        frame_set.add(nid)
        present.add(nid)

    def _closeable_he(nid: str, budget: int) -> Optional[List[str]]:
        """Smallest hyperedge containing nid that can ship WHOLE within THIS frame
        and whose not-yet-in-frame members fit `budget`. Returns the members to ADD
        (may be empty if all are already in-frame), or None if no such HE.

        #147 amendment: completeness is judged against frame_set, NOT the ack
        ledger. An HE ships iff every member is eligible AND lands in this same
        frame -- already-acked members are RE-added (idempotent) rather than
        assumed-present. Leaning on exclude_ids/present here let an HE ship
        referencing members the receiver did not actually hold, which reaped nodes
        on the two prior merge attempts."""
        best: Optional[List[str]] = None
        for he_id in node_hes.get(nid, ()):
            if any(not eligible(m) for m in he_members[he_id]):
                continue  # an ineligible member => HE can never be whole; not an anchor
            add = [m for m in he_members[he_id] if m not in frame_set]
            if len(add) <= budget and (best is None or len(add) < len(best)):
                best = add
        return best

    for nid in candidates:
        if len(frame) >= frame_size:
            break
        if nid in frame_set:
            continue  # already pulled in by an earlier HE closure
        budget = frame_size - len(frame)

        # (a) synapse anchor to something already acked or already in this frame.
        if any(p in present for p in syn_partners.get(nid, ())):
            _place(nid)
            continue

        # (b) whole-HE closure within the normal frame budget.
        new = _closeable_he(nid, budget)
        if new is not None:
            for m in new:
                _place(m)
            continue

        # (c) whole-HE closure into the overflow region -- one HE too big for the
        #     remaining budget but within the hard cap. Never split; closes frame.
        new = _closeable_he(nid, hard_cap - len(frame))
        if new is not None:
            for m in new:
                _place(m)
            break

        # (d) co-add a synapse partner that is itself a shippable candidate: the
        #     shared synapse anchors BOTH. Seeds a synapse-only component and the
        #     very first frame's origin (which has nothing already-acked to lean on).
        partnered = False
        for p in syn_partners.get(nid, ()):
            if p in present or p in frame_set:
                continue
            if p in exclude_ids or not eligible(p):
                continue
            if len(frame) + 2 <= frame_size:
                _place(nid)
                _place(p)
                partnered = True
            break
        if partnered:
            continue

        # (e) unshippable this pass. Distinguish an oversized source blob (Leg S
        #     owes a split) from a node whose only anchors are still in the future
        #     (it becomes placeable once this/earlier frames are acked).
        oversized = [h for h in node_hes.get(nid, ()) if len(he_members[h]) > hard_cap]
        if oversized:
            for h in oversized:
                if h in _oversized_seen:
                    continue
                _oversized_seen.add(h)
                stats["oversized_he_at_source"] += 1
                logger.error(
                    "CC topology frame: hyperedge %s has %d members > hard cap "
                    "(%d) -- oversized source-side blob (§8.14). Skipping; Leg S "
                    "(VPS dream split) must repair it before it can cross.",
                    h, len(he_members[h]), hard_cap)
        else:
            stats["deferred_no_anchor"] += 1

    if not frame:
        # Candidates exist but none were placeable this pass (all deferred /
        # oversized). NOT exhausted -- surface it so the driver doesn't spin.
        logger.warning(
            "CC topology frame: %d candidate(s) but none placeable this pass "
            "(deferred=%d, oversized=%d)",
            len(candidates), stats["deferred_no_anchor"],
            stats["oversized_he_at_source"])
        return stats

    # ---- build payloads ONLY for the chosen frame ----
    node_payloads: List[Dict[str, Any]] = []
    for nid in frame:
        node = nodes_map[nid]
        meta = dict(getattr(node, "metadata", None) or {})
        payload: Dict[str, Any] = {
            "id": nid,
            "content": _content_for(vector_db, nid, meta),
            "metadata": _portable_metadata(meta),
        }
        # An unembedded CC node is a defect upstream (see collect_cc_topology's
        # note), not a tolerable mode -- but it still crosses so its binding
        # structure is not shredded; the gap is ALARMED, to be driven to zero by
        # re-embedding, never absorbed.
        emb = _embedding_for(vector_db, nid)
        if emb is None:
            stats["missing_embedding_DEFECT"] += 1
            logger.error(
                "CC topology frame: node %s has NO embedding -- exporting its "
                "topology anyway, but it will be inert to the Tonic and to recall "
                "until re-embedded. Defect, not a mode.", nid)
        else:
            payload["embedding"] = emb.astype(np.float32).tobytes()
            payload["embedding_dim"] = int(emb.shape[0])
        node_payloads.append(payload)

    # ---- edges touching >=1 frame node, each whole within its OWN safe set ----
    # #147 amendment draws a line between the two edge kinds, because they fail
    # DIFFERENTLY on the receiver:
    #   * A synapse to an already-acked node is SAFE -- the receiver holds that
    #     endpoint, so create_synapse resolves. It rides (whole within
    #     frame_set|exclude_ids), preserving the "edges-to-acked survive" contract.
    #   * A hyperedge whose member the receiver lacks raises KeyError and the whole
    #     install fails -- which is what reaped nodes on the two prior merges. So an
    #     HE rides ONLY when it is whole within THIS frame (frame_set alone), never
    #     leaning on the ack ledger. HE closure already re-adds acked members into
    #     frame_set, so whole HEs still ship whole.
    # (Synapse anchoring trusting a stale ack -- an acked node the receiver later
    # reaped -- is the acknowledged narrower follow-up, NOT closed here.)
    # collect_incident_structure carries the archived-HE guard and synapse `delay`.
    syn_member_set = frame_set | exclude_ids
    syns, _ = collect_incident_structure(graph, syn_member_set)
    _, hes = collect_incident_structure(graph, frame_set)
    syns = [s for s in syns if s["pre"] in frame_set or s["post"] in frame_set]
    hes = [h for h in hes if any(m in frame_set for m in h["members"])]

    # ---- write a single-batch conduit atomically (existing header+batch shape) ----
    tmp_path = out_path + ".partial"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(tmp_path, "wb") as fh:
        fh.write(_frame({
            "kind": "header",
            "version": _WIRE_VERSION,
            "machine_id": machine_id,
            "embedding_model": embedding_model,
            "created": time.time(),
            "node_count": len(frame),
        }))
        fh.write(_frame({
            "kind": "batch",
            "seq": 1,
            "nodes": node_payloads,
            "synapses": syns,
            "hyperedges": hes,
        }))
    # Atomic publish: a reader must never observe a partial conduit (Leg 1 hit
    # exactly this -- a drain that read mid-write).
    os.replace(tmp_path, out_path)

    stats.update({
        "exported_nodes": len(frame),
        "exported_synapses": len(syns),
        "exported_hyperedges": len(hes),
        "frame_node_ids": list(frame),
        "embedding_model": embedding_model,
    })
    logger.info(
        "CC topology frame: %d node(s), %d synapse(s), %d hyperedge(s) -> %s "
        "(candidates=%d, deferred=%d, oversized=%d)",
        len(frame), len(syns), len(hes), out_path,
        len(candidates), stats["deferred_no_anchor"],
        stats["oversized_he_at_source"])
    return stats


def read_topology_frames(raw: bytes) -> Iterable[Dict[str, Any]]:
    """Decode a length-prefixed msgpack conduit into frames.

    Tolerates a truncated tail (a conduit caught mid-write by a reader that
    bypassed the atomic publish) by stopping rather than raising -- the
    remaining frames arrive on the next pass.
    """
    off = 0
    n = len(raw)
    while off + 4 <= n:
        (length,) = struct.unpack_from(_LEN_PREFIX, raw, off)
        off += 4
        if off + length > n:
            logger.warning(
                "CC topology conduit truncated at offset %d (want %d bytes, have %d) "
                "-- stopping; remainder will arrive next pass", off, length, n - off)
            return
        yield msgpack.unpackb(raw[off:off + length], raw=False)
        off += length
