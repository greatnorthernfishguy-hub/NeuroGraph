#!/usr/bin/env python3
# SEE FIRST: /home/josh/docs/CC-CALLOSUM-TRUTH.md -- consolidated, verified state of
# the callosum, wholeness ring, hyperedge binding and orphan collection (2026-07-31).
# The wholeness ring ALREADY EXISTS here (Leg 2). Open defect: merge-journal poison-pill.
# ---- Changelog ----
# [2026-07-29] Claude Code (DudeMan CC, Opus 5) — Callosum Leg 2: topology merge (laptop side)
# What: merge_cc_topology() absorbs the length-prefixed msgpack conduit written by
#   cc_topology_export.py, depositing the VPS Arborist's grown ::tree:: structure
#   into the laptop's CC substrate — nodes, then synapses, then hyperedges.
# Why: #70 Leg 2 receive half. The laptop has no TID (ng_embed.py:567
#   _extract_concepts is an HTTP call; no TID -> None -> forest-only), so it can
#   never grow trees locally by any means. This is the only path to tree-parity.
# How: per docs/superpowers/plans/2026-07-18-cc-river-merge-implementation-plan.md.
#   Ordering is forced, not stylistic: neuro_foundation.py:1951 create_hyperedge
#   raises KeyError on an absent member, and a synapse needs both endpoints. So
#   nodes -> synapses -> hyperedges, and a batch that fails partway leaves the
#   graph consistent because each tier only references tiers already landed.
#
#   Two aborts, both deliberately hard failures rather than degradations:
#   - EMBEDDING MODEL MISMATCH (FatherGraph Finding 6). If the hemispheres embed
#     with different models, cosine similarity between their vectors is noise.
#     The failure is silent and unrecoverable-in-place: bad geometry gets
#     consolidated into weights, and by the time recall is visibly wrong the
#     substrate has already learned on it. Abort beats absorb.
#   - SELF-ABSORPTION. If header machine_id == local, we are reading our own
#     export. Absorbing it would re-deposit our own nodes as if foreign,
#     double-counting structure and corrupting the trickle accounting.
#
#   Idempotency is `node_id in graph.nodes` and nothing else. The journal is
#   DELIVERY BOOKKEEPING, not a receive-side guard: it records what landed so
#   the SENDER can pass exclude_ids and stop re-transmitting. It has no say in
#   what the receiver admits -- see the 2026-07-31 entry below.
#
# [2026-07-31] Claude Code (Opus 5) — #106: remove the merge-journal poison-pill
# What: deleted the receive-side `if nid in journal: continue` veto in Tier 1.
#   Presence in the graph is now the sole admission guard. A journaled-but-absent
#   node is re-absorbed and counted as `journal_stale_readmitted`.
# Why: the journal is append-only with no invalidation path, and the graph guard
#   above it already catches every node that is actually present. So the journal
#   branch could only ever fire on the set (journal - graph.nodes) -- precisely
#   the nodes that were absorbed once and then destroyed locally (#104 cull,
#   orphan sweep, checkpoint rolled back behind the merge). That is exactly the
#   set that needs re-delivery, and the veto made it permanently undeliverable.
#   The damage compounds past the node: Tier 2 gates on both endpoints being in
#   graph.nodes and Tier 3 needs every member present, so one permanently-vetoed
#   node silently shredded every synapse and hyperedge incident to it, on every
#   future pass, forever. A conduit is the laptop's ONLY route to tree structure
#   (no TID here) -- a permanent veto is a permanent hole in the topology.
# How: the branch stays, minus the `continue` -- it now logs and counts instead
#   of dropping, so the stale-journal condition is observable rather than silent.
#   Ruled at .claude/agent-memory/neurograph-law-enforcer/
#   ruling_half_brain_wholeness_ring.md ("Q2 ... Delete the journal skip from the
#   RECEIVE path ... zero new state"); additive-only governs graph CONTENT, not
#   delivery bookkeeping. Re-absorb churn is bounded on both ends: the sender's
#   exclude_ids stops re-sending journaled nodes, and a node re-absorbed here
#   arrives with its own synapses and hyperedge in the same batch, so it does not
#   land orphaned into the next sweep.
# -------------------

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Set

import numpy as np

from cc_topology_export import read_topology_frames, is_cc_provenance

logger = logging.getLogger(__name__)

# FatherGraph Finding 1: trickle, never bulk-dump. Patching a large block of
# structure in after the fact displaces existing learning rather than
# integrating with it.
_DEFAULT_MAX_NODES_PER_CALL = int(os.environ.get("CC_TOPOLOGY_MERGE_MAX_NODES", "50"))


class TopologyMergeAbort(RuntimeError):
    """Raised when the conduit must not be absorbed at all (model mismatch,
    self-absorption, malformed header). Distinct from per-item skips, which are
    counted and logged but never abort the pass."""


def _load_journal(journal_path: Optional[str]) -> Set[str]:
    """Node IDs already delivered to us, for the SENDER's exclude_ids.

    This is delivery bookkeeping, NOT an admission guard -- `nid in graph.nodes`
    decides what the receiver takes (#106). Append-only with no invalidation
    path, so an entry outlives the node it names; treating it as authority is
    what made journaled-but-culled nodes permanently undeliverable. Missing or
    corrupt journal is harmless: it costs the sender a re-scan, nothing more.
    """
    if not journal_path or not os.path.exists(journal_path):
        return set()
    seen: Set[str] = set()
    try:
        with open(journal_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    seen.add(line)
    except Exception as exc:
        logger.warning("CC topology journal unreadable (%s): %s -- continuing "
                       "with graph-membership guard only", journal_path, exc)
    return seen


def _append_journal(journal_path: Optional[str], node_ids: List[str]) -> None:
    if not journal_path or not node_ids:
        return
    try:
        os.makedirs(os.path.dirname(os.path.abspath(journal_path)) or ".", exist_ok=True)
        with open(journal_path, "a", encoding="utf-8") as fh:
            for nid in node_ids:
                fh.write(nid + "\n")
            fh.flush()
            os.fsync(fh.fileno())
    except Exception as exc:
        logger.warning("CC topology journal append failed (non-fatal): %s", exc)


def _local_embedding_model() -> str:
    try:
        import ng_embed
        return (getattr(ng_embed, "MODEL_NAME", None)
                or getattr(ng_embed, "_MODEL_NAME", None) or "unknown")
    except Exception:
        return "unknown"


def _synapse_type(name: Optional[str]):
    from neuro_foundation import SynapseType
    if not name:
        return SynapseType.EXCITATORY
    try:
        return SynapseType[str(name).upper()]
    except KeyError:
        logger.debug("Unknown synapse_type %r -- defaulting EXCITATORY", name)
        return SynapseType.EXCITATORY


def merge_cc_topology(
    graph: Any,
    vector_db: Any,
    conduit_path: str,
    local_machine_id: Optional[str] = None,
    journal_path: Optional[str] = None,
    max_nodes_per_call: int = _DEFAULT_MAX_NODES_PER_CALL,
    expected_embedding_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Absorb a CC topology conduit into the local substrate.

    Returns a stats dict. Raises TopologyMergeAbort when the conduit must not
    be absorbed at all.
    """
    from cc_ng_organism import _cc_deposit_memory_node

    local_machine_id = local_machine_id or os.environ.get("MACHINE_ID")
    if not local_machine_id:
        raise TopologyMergeAbort(
            "MACHINE_ID unset -- cannot verify this conduit is not our own export")

    if not os.path.exists(conduit_path):
        return {"status": "no_conduit", "path": conduit_path, "absorbed_nodes": 0}

    with open(conduit_path, "rb") as fh:
        raw = fh.read()

    frames = list(read_topology_frames(raw))
    if not frames:
        return {"status": "empty", "path": conduit_path, "absorbed_nodes": 0}

    header = frames[0]
    if header.get("kind") != "header":
        raise TopologyMergeAbort(
            f"conduit {conduit_path} does not begin with a header frame "
            f"(got kind={header.get('kind')!r})")

    sender = header.get("machine_id")
    if sender == local_machine_id:
        raise TopologyMergeAbort(
            f"conduit was authored by this machine ({sender}) -- refusing to "
            "re-absorb our own export")

    wire_model = header.get("embedding_model")
    local_model = expected_embedding_model or _local_embedding_model()
    if wire_model != local_model and "unknown" not in (wire_model, local_model):
        # FatherGraph Finding 6. Not degradable -- see module header.
        raise TopologyMergeAbort(
            f"embedding model mismatch: conduit={wire_model!r} local={local_model!r}. "
            "Cosine similarity between differently-embedded vectors is noise; "
            "refusing to absorb rather than silently corrupt recall geometry")

    journal = _load_journal(journal_path)
    stats = {
        "status": "ok", "path": conduit_path, "sender": sender,
        "embedding_model": wire_model,
        "absorbed_nodes": 0, "absorbed_synapses": 0, "absorbed_hyperedges": 0,
        "skipped_present": 0, "journal_stale_readmitted": 0, "skipped_not_cc": 0,
        "skipped_identity": 0, "bad_embedding": 0,
        "absorbed_without_embedding_DEFECT": 0,
        "skipped_synapses": 0, "skipped_hyperedges": 0,
        "hyperedge_id_reminted": 0,
        "batches_read": 0, "deferred_by_budget": 0,
    }

    budget = max_nodes_per_call
    landed_ids: List[str] = []

    for frame in frames[1:]:
        if frame.get("kind") != "batch":
            continue
        stats["batches_read"] += 1

        if budget <= 0:
            # Trickle discipline: stop cleanly, resume next pass. The journal
            # + graph guard make the resume a no-op for what already landed.
            stats["deferred_by_budget"] += len(frame.get("nodes") or ())
            continue

        # --- Tier 1: nodes -------------------------------------------------
        batch_landed: Set[str] = set()
        for rec in frame.get("nodes") or ():
            if budget <= 0:
                stats["deferred_by_budget"] += 1
                continue
            nid = rec.get("id")
            if not nid:
                continue
            if nid in graph.nodes:
                stats["skipped_present"] += 1
                batch_landed.add(nid)   # present == usable as an endpoint
                continue
            if nid in journal:
                # NO `continue` HERE -- #106. Falling through is the fix.
                #
                # The graph check above already caught every node that is
                # actually present, so this branch can only be reached by a node
                # in (journal - graph.nodes): one that landed once and was then
                # destroyed locally. Vetoing on that record made re-delivery
                # impossible forever, and the loss did not stop at the node --
                # Tier 2 requires both endpoints in graph.nodes and Tier 3 every
                # member, so a single permanently-vetoed node shredded all of its
                # incident structure on every subsequent pass. The journal is the
                # sender's bookkeeping (exclude_ids); presence in the graph is
                # the receiver's authority. A journaled node that arrived anyway
                # means the sender chose to re-send it -- honour the delivery.
                # Counted at the absorption site below, not here -- this point
                # is only "detected", and the node can still be rejected by the
                # provenance gates or the deposit try/except before it lands.
                logger.info(
                    "CC topology: %s is journaled as landed but is absent from "
                    "the graph -- re-absorbing (stale journal entry, #106)", nid)

            meta = dict(rec.get("metadata") or {})
            # Defense in depth: re-run both provenance gates on receive. The
            # sender is trusted but not authoritative -- a conduit is a file,
            # and a file can be stale, hand-edited, or from a mispointed
            # workspace. Cheap check, catastrophic miss.
            if not is_cc_provenance(nid, meta):
                stats["skipped_not_cc"] += 1
                continue
            if meta.get("constitutional") or str(meta.get("provenance") or "").endswith("_authored"):
                stats["skipped_identity"] += 1
                continue

            # An embedding is an attribute of the node, not a precondition for
            # it. Absent or corrupt, the node still installs with its metadata
            # and stays a full participant in synapses and hyperedges -- which
            # are the payload. What is lost is recall-store indexing and the
            # poincare_dir stamp, both recoverable later by re-embedding.
            # Dropping the node instead would also drop every edge touching it.
            emb = None
            dim = int(rec.get("embedding_dim") or 0)
            blob = rec.get("embedding")
            if blob and dim > 0:
                candidate = np.frombuffer(blob, dtype=np.float32)
                if candidate.shape[0] == dim and np.all(np.isfinite(candidate)):
                    emb = candidate
                else:
                    # Corrupt numbers are worse than none: a malformed vector
                    # would be indexed for recall and stamped as a position.
                    stats["bad_embedding"] += 1
                    logger.warning(
                        "CC topology: discarding malformed embedding for %s "
                        "(installing node structurally)", nid)

            try:
                if emb is not None:
                    # Deposits into graph + recall AND re-derives poincare_dir
                    # locally from the embedding (cc_ng_organism.py:1219),
                    # which is why the wire never carries it.
                    _cc_deposit_memory_node(graph, vector_db, nid, emb,
                                            rec.get("content") or "", meta)
                else:
                    # Structural install. Deliberately NOT stamping a zero
                    # poincare_dir -- absent is honest, whereas zeros asserts a
                    # false position at the origin that delay derivation would
                    # then treat as real.
                    #
                    # Reaching here means the sender exported a node with no
                    # usable vector. That is a DEFECT on the far side (see the
                    # matching alarm in cc_topology_export.collect_cc_nodes),
                    # not a supported wire mode -- the node lands structurally
                    # so its synapses and hyperedges survive, but it is inert
                    # to recall and to the Tonic until re-embedded. Drive this
                    # count to zero; do not learn to live with it.
                    graph.create_node(node_id=nid, metadata=dict(meta))
                    stats["absorbed_without_embedding_DEFECT"] += 1
                    logger.error(
                        "CC topology merge: node %s arrived with NO embedding -- "
                        "installed structurally, but it is inert to recall and "
                        "the Tonic. Upstream export defect.", nid)
            except Exception as exc:
                logger.warning("CC topology deposit failed for %s: %s", nid, exc)
                continue

            stats["absorbed_nodes"] += 1
            if nid in journal:
                # Counted here, after the node has actually landed, so the stat
                # reports re-admissions that happened rather than ones merely
                # attempted. `journal` is not mutated inside this loop, so this
                # re-test is the same predicate evaluated at the Tier-1 branch.
                stats["journal_stale_readmitted"] += 1
            landed_ids.append(nid)
            batch_landed.add(nid)
            budget -= 1

        # --- Tier 2: synapses (both endpoints must exist) -------------------
        for syn in frame.get("synapses") or ():
            pre, post = syn.get("pre"), syn.get("post")
            if pre not in graph.nodes or post not in graph.nodes:
                stats["skipped_synapses"] += 1
                continue
            if _synapse_exists(graph, pre, post):
                stats["skipped_synapses"] += 1
                continue
            try:
                graph.create_synapse(
                    pre_node_id=pre,
                    post_node_id=post,
                    weight=float(syn.get("weight", 0.1)),
                    # The reason this is msgpack and not a BTF frame: delay is
                    # functional (polychronous motifs, STDP ordering), and BTF
                    # has no field for it.
                    delay=int(syn.get("delay", 1)),
                    synapse_type=_synapse_type(syn.get("synapse_type")),
                    max_weight=float(syn.get("max_weight", 5.0)),
                )
                stats["absorbed_synapses"] += 1
            except Exception as exc:
                logger.debug("CC topology synapse %s->%s failed: %s", pre, post, exc)
                stats["skipped_synapses"] += 1

        # --- Tier 3: hyperedges (all members must exist) --------------------
        for he in frame.get("hyperedges") or ():
            members = set(he.get("members") or ())
            if not members or not members.issubset(graph.nodes):
                # create_hyperedge raises KeyError here (neuro_foundation.py:1951);
                # skipping keeps the pass alive for the rest of the batch.
                stats["skipped_hyperedges"] += 1
                continue
            if _hyperedge_exists(graph, members):
                # Re-merge must not stack a second binding edge over the same
                # members -- that double-counts the turn's activation. This
                # member-set check is kept as the PRIMARY guard even though the
                # sender's id now rides the wire, because it also catches the
                # case id-matching cannot: an edge the two hemispheres grew
                # INDEPENDENTLY over the same members, which has two legitimate
                # but different ids. Id-preservation and member-set dedupe
                # cover different halves of convergence; we want both.
                stats["skipped_hyperedges"] += 1
                continue
            try:
                # Preserve the sender's identity when it supplied one. Frames
                # written before the id was added to the wire simply omit it,
                # and .get() -> None restores the old mint-locally behaviour --
                # so an in-flight older frame still merges cleanly.
                wire_id = he.get("id") or None
                if wire_id is not None and wire_id in getattr(graph, "hyperedges", {}):
                    # Same id, different member set (member-set dedupe above
                    # already cleared identical ones). create_hyperedge would
                    # raise ValueError; mint locally instead of losing the edge.
                    logger.warning(
                        "CC topology: hyperedge id %s already present with "
                        "different members -- installing under a fresh local id",
                        wire_id)
                    wire_id = None
                edge = graph.create_hyperedge(
                    member_node_ids=members,
                    activation_threshold=float(he.get("activation_threshold", 0.6)),
                    metadata=dict(he.get("metadata") or {}),
                    hyperedge_id=wire_id,
                )
                lvl = he.get("level")
                if lvl is not None and hasattr(edge, "level"):
                    # No `level` param on create_hyperedge -- set post-hoc.
                    edge.level = int(lvl)
                stats["absorbed_hyperedges"] += 1
                if wire_id is None and he.get("id"):
                    stats["hyperedge_id_reminted"] += 1
            except Exception as exc:
                logger.debug("CC topology hyperedge failed: %s", exc)
                stats["skipped_hyperedges"] += 1

    # Re-admitted nodes (#106) are already journaled; appending them again would
    # grow the file by one duplicate line per node per pass without changing the
    # loaded set. Only record genuinely new deliveries.
    _append_journal(journal_path, [n for n in landed_ids if n not in journal])

    stats["completed"] = stats["deferred_by_budget"] == 0
    logger.info(
        "CC topology merge from %s: +%d node(s), +%d synapse(s), +%d hyperedge(s); "
        "%d deferred to next pass",
        sender, stats["absorbed_nodes"], stats["absorbed_synapses"],
        stats["absorbed_hyperedges"], stats["deferred_by_budget"],
    )
    return stats


def _hyperedge_exists(graph: Any, members: Set[str]) -> bool:
    """Idempotency guard for hyperedges -- the counterpart to _synapse_exists.

    Without this, a re-merge stacks a second binding hyperedge over the same
    member set, and the turn it binds gets its activation counted twice on
    every pass. Uses graph._node_hyperedges (neuro_foundation.py:1532,
    node_id -> set of hyperedge_ids) so only edges touching one member are
    examined rather than the whole hyperedge table.
    """
    if not members:
        return False
    try:
        probe = next(iter(members))
        candidate_ids = getattr(graph, "_node_hyperedges", {}).get(probe) or ()
        table = getattr(graph, "hyperedges", {})
        for hid in candidate_ids:
            he = table.get(hid)
            if he is None:
                continue
            existing = getattr(he, "member_nodes", None)
            if existing is not None and set(existing) == members:
                return True
        return False
    except Exception:
        try:
            for he in getattr(graph, "hyperedges", {}).values():
                existing = getattr(he, "member_nodes", None)
                if existing is not None and set(existing) == members:
                    return True
        except Exception:
            pass
        return False


def _synapse_exists(graph: Any, pre: str, post: str) -> bool:
    """Idempotency guard for synapses. Re-absorbing must not stack duplicate
    edges between the same pair -- that would silently multiply effective
    conductance on every pass.

    Uses the graph's own sparse adjacency index (neuro_foundation.py:1529,
    _outgoing: node_id -> set of synapse_ids), so this is O(out-degree) rather
    than a full scan of graph.synapses on every candidate edge.
    """
    try:
        syn_ids = getattr(graph, "_outgoing", {}).get(pre) or ()
        synapses = getattr(graph, "synapses", {})
        for sid in syn_ids:
            syn = synapses.get(sid)
            if syn is not None and getattr(syn, "post_node_id", None) == post:
                return True
        return False
    except Exception:
        # Index unavailable/shaped differently -> fall back to a full scan
        # rather than reporting "no edge" and creating a duplicate.
        try:
            for syn in getattr(graph, "synapses", {}).values():
                if (getattr(syn, "pre_node_id", None) == pre
                        and getattr(syn, "post_node_id", None) == post):
                    return True
        except Exception:
            pass
        return False
