#!/usr/bin/env python3
# SEE FIRST: /home/josh/docs/CC-CALLOSUM-TRUTH.md -- consolidated, verified state of
# the callosum, wholeness ring, hyperedge binding and orphan collection (2026-07-31).
# The wholeness ring ALREADY EXISTS here (Leg 2). Open defect: merge-journal poison-pill.
# ---- Changelog ----
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
#   - Identity-protected nodes (constitutional / *_authored) are never exported.
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


def is_cc_provenance(node_id: str, meta: Optional[Dict[str, Any]]) -> bool:
    """Positive CC-provenance test over (id, metadata) -- the form both the
    sender (which has Node objects) and the receiver (which has only decoded
    wire dicts) can call without inventing a fake node.

    Deliberately a whitelist: on the VPS this code runs in a process that also
    holds Syl's graph, so 'not obviously Syl's' is not a safe answer.
    """
    if any(node_id.startswith(p) for p in _CC_ID_PREFIXES):
        return True
    if any(m in node_id for m in _CC_ID_MARKERS):
        return True
    meta = meta or {}
    if meta.get("cc") is True:
        return True
    if meta.get("provenance") in _CC_PROVENANCE:
        return True
    return False


def _is_cc_node(node_id: str, node: Any) -> bool:
    """Sender-side convenience wrapper over is_cc_provenance()."""
    return is_cc_provenance(node_id, getattr(node, "metadata", None) or {})


def _is_identity_protected(graph: Any, node_id: str, node: Any) -> bool:
    """Never donate identity-bearing structure. Prefers the engine's own
    predicate so this cannot drift from canonical; falls back to the same
    fields the engine checks if the helper is unavailable."""
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

    collected: List[Tuple[float, Dict[str, Any]]] = []
    for node_id, node in list(graph.nodes.items()):
        stats["scanned"] += 1
        if node_id in exclude_ids:
            stats["already_sent"] += 1
            continue
        if not _is_cc_node(node_id, node):
            stats["not_cc"] += 1
            continue
        if _is_identity_protected(graph, node_id, node):
            stats["identity_protected"] += 1
            continue
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
) -> Dict[str, Any]:
    """Export CC conversational topology to a length-prefixed msgpack conduit.

    Read-only with respect to the graph. Returns a stats dict.

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
    if max_nodes is not None:
        nodes = nodes[:max_nodes]

    # Structure is collected ONCE against the FULL exportable node set, never
    # per-chunk. Batching is a DELIVERY concern -- it must not decide what
    # topology exists. Collecting per-chunk (as this first did) silently drops
    # every edge whose endpoints straddle a chunk boundary; at batch_size=25
    # over a few thousand nodes that is most of the graph.
    all_ids = [n["id"] for n in nodes]
    batch_of = {nid: i // batch_size for i, nid in enumerate(all_ids)}
    syns, hes = collect_incident_structure(graph, set(all_ids))

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
