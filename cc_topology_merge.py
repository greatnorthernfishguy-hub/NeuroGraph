#!/usr/bin/env python3
# SEE FIRST: /home/josh/docs/CC-CALLOSUM-TRUTH.md -- consolidated, verified state of
# the callosum, wholeness ring, hyperedge binding and orphan collection (2026-07-31).
# The wholeness ring ALREADY EXISTS here (Leg 2). Open defect: merge-journal poison-pill.
# ---- Changelog ----
# [2026-08-12] Claude Code (DudeMan CC, Opus 4.8) — #88 §10.4-C: receiver budget pinned 50->25
# What: _DEFAULT_MAX_NODES_PER_CALL default 50 -> 25 (env CC_TOPOLOGY_MERGE_MAX_NODES).
# Why: FatherGraph Finding 1 + Finding 3 (25/250). The driver ships BATCH_SIZE=25; a
#       50-node receiver budget would merge two sender frames before a single 250-step
#       consolidation pass -- a silent 2x bulk-dump against the exact Finding this guards.
# How: one-line default change + comment; no signature/behavior change beyond the floor.
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
#
# [2026-08-02] Claude Code (Opus 5) — #108 / ruling condition (c): consolidation
#   steps between merge batches, and the guard that makes them safe.
# What: after each batch's Tier 3, run `idle_steps` of pure graph.step() via
#   cc_ng_organism._cc_callosum_consolidate (LAW 3 — reuse, do not mint a second
#   stepping loop). Env-sourced from CC_NG_IDLE_STEPS (LAW 5), default 250.
#   Guarded: skip while ANY arrival this merge has landed is still unbound.
# Why: grace exists so STDP-via-spreading-activation and sprouting-via-co-firing
#   can wire an arriving node (neuro_foundation.py:3474) -- local, same-step
#   mechanisms. Merge ran zero steps, so arrivals got the grace window and no
#   wiring opportunity inside it. Measured consequence, CC-CALLOSUM-TRUTH.md
#   §8.6: cc_gateway nodes that fired are 98% wired; nodes that never fired are
#   0.4% wired. Firing is what wires; no steps means no firing.
# Guard scope: MERGE-scoped, not batch-scoped, and that distinction is the fix.
#   Binding structure splits across batches, so a batch-scoped predicate cannot
#   see the node it must protect -- batch 1 lands X unbound and skips, batch 2
#   lands whole, X is not in batch 2's set, the guard passes, and 250 steps age
#   X from 0 to 250 against a grace of 25. That is §8.2's cohort cliff authored
#   into the merge path in the name of fixing it. Caught in law-enforcer review
#   2026-08-02 before commit.
# Not #117: this advances the clock on a merge, which is part of absorbing the
#   merge. It is NOT the missing autonomic loop and must not be read as progress
#   on the LAW 8 violation (§0.2, §8.5). The clock is still conversation-gated.
#
# [2026-08-04] Claude Code (Opus 4.8) — #110: exclude_ids from live membership,
#   not the append-only journal (the §3 poison-pill, relocated to the send side).
# What: the receiver-written file the SENDER reads as exclude_ids is now a
#   MEMBERSHIP SNAPSHOT of current CC graph.nodes, overwritten each merge
#   (cc_current_membership + _write_membership), replacing the append-only
#   _append_journal. Param journal_path -> membership_path; stat
#   journal_stale_readmitted -> membership_stale_readmitted (nothing in-tree read
#   the old key; tests updated).
# Why: #106 made presence-in-graph authoritative on the RECEIVE side, but the
#   sender still built exclude_ids from an append-only record with no
#   invalidation path. A node culled locally (#104 sweep, orphan collection,
#   rolled-back checkpoint) stayed in that record forever, so the sender never
#   re-sent it and #106's re-admission had nothing to re-admit -- the identical
#   append-only-record-as-authority shape §3 warns about, moved one hop. A live
#   snapshot SHRINKS when a node is culled, closing the loop: culled -> drops out
#   of exclude_ids -> re-sent -> re-admitted (#106). Presence in the graph is now
#   the single authority on BOTH sides. Doc: CC-CALLOSUM-TRUTH.md §4 caveat, §10
#   Phase 3. Not yet load-bearing (merge_cc_topology has no production caller);
#   wired correct now so #88's first live run inherits it.
# -------------------

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Set

import numpy as np

from cc_topology_export import read_topology_frames, is_cc_provenance

logger = logging.getLogger(__name__)

# FatherGraph Finding 1 + Finding 3: trickle, never bulk-dump, and 25 nodes per
# batch with 250 idle consolidation steps BETWEEN batches (the measured 47%->74%
# recall gain -- "not optional"). Patching a large block of structure in after
# the fact displaces existing learning rather than integrating with it. Pinned to
# 25 (CC-CALLOSUM-TRUTH.md §10.4-C): the driver ships BATCH_SIZE=25, so a 50-node
# receiver budget would merge two sender frames before a single consolidation
# pass -- a silent 2x bulk-dump against the exact Finding this guards.
_DEFAULT_MAX_NODES_PER_CALL = int(os.environ.get("CC_TOPOLOGY_MERGE_MAX_NODES", "25"))


class TopologyMergeAbort(RuntimeError):
    """Raised when the conduit must not be absorbed at all (model mismatch,
    self-absorption, malformed header). Distinct from per-item skips, which are
    counted and logged but never abort the pass."""


def cc_current_membership(graph: Any) -> Set[str]:
    """The CC node IDs the receiver currently holds -- the authoritative source
    for the SENDER's exclude_ids (#110).

    It is `graph.nodes` intersected with CC provenance, read live. Because it is
    regenerated from the graph rather than appended to, a node culled locally
    (#104 sweep, orphan collection, a rolled-back checkpoint) DROPS OUT of it --
    so the exporter re-sends that node and #106's receive-side re-admission has
    something to re-admit. The append-only journal this replaces could only grow,
    so a culled id stayed excluded forever and the node was permanently
    un-resendable: the §3 poison-pill, merely relocated to the send side. After
    #110, presence in the graph is the authority on BOTH sides -- receive
    admission (#106) and send exclusion.

    Uses the same predicate the exporter classifies with (is_cc_provenance, via
    cc_topology_export._is_cc_node), so what the receiver advertises as held and
    what the sender considers CC-exportable cannot drift apart.
    """
    return {nid for nid, node in graph.nodes.items()
            if is_cc_provenance(nid, getattr(node, "metadata", None) or {})}


def _load_membership(membership_path: Optional[str]) -> Set[str]:
    """Load the receiver's last-written membership snapshot -- what the SENDER
    reads as exclude_ids.

    Delivery bookkeeping, NOT an admission guard: `nid in graph.nodes` alone
    decides what the receiver takes (#106). Missing or corrupt is harmless -- it
    costs the sender a re-scan (it re-sends and the receiver idempotently skips
    what it already has), nothing more.
    """
    if not membership_path or not os.path.exists(membership_path):
        return set()
    held: Set[str] = set()
    try:
        with open(membership_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    held.add(line)
    except Exception as exc:
        logger.warning("CC topology membership snapshot unreadable (%s): %s -- "
                       "continuing with graph-membership guard only",
                       membership_path, exc)
    return held


def _write_membership(membership_path: Optional[str], node_ids: Set[str]) -> None:
    """Overwrite the snapshot with the receiver's CURRENT CC membership (#110).

    Overwrite, never append: the file MUST be able to shrink when a node is
    culled, or it becomes the same append-only-record-as-authority the §3
    poison-pill was. Written via a temp file + os.replace so a crash mid-write
    cannot leave the sender reading a half-truncated snapshot (a truncated
    snapshot only ever over-excludes-less -> re-sends more, never corrupts, but
    the atomic swap keeps even that from happening). Sorted output is
    deterministic across passes.
    """
    if not membership_path:
        return
    try:
        os.makedirs(os.path.dirname(os.path.abspath(membership_path)) or ".", exist_ok=True)
        tmp = membership_path + ".partial"
        with open(tmp, "w", encoding="utf-8") as fh:
            for nid in sorted(node_ids):
                fh.write(nid + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, membership_path)
    except Exception as exc:
        logger.warning("CC topology membership snapshot write failed (non-fatal): %s", exc)


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
    membership_path: Optional[str] = None,
    max_nodes_per_call: int = _DEFAULT_MAX_NODES_PER_CALL,
    expected_embedding_model: Optional[str] = None,
    idle_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """Absorb a CC topology conduit into the local substrate.

    Returns a stats dict. Raises TopologyMergeAbort when the conduit must not
    be absorbed at all.

    CONSOLIDATION BETWEEN BATCHES (FatherGraph Finding 3, ruling condition (c),
    #108): after each batch's Tier 3 completes, `idle_steps` of pure graph.step()
    run with no new input so homeostatic regulation can absorb the new topology
    before the next batch arrives. The report calls this "not optional -- it's
    what makes merge work" (47%->74%). Reuses `_cc_callosum_consolidate`
    (cc_ng_organism.py:1856), which already slices the lock -- LAW 3, restore the
    existing mechanism rather than write a second one. Default 250 via
    CC_NG_IDLE_STEPS, the same env the nightly cron already exports.
    """
    from cc_ng_organism import _cc_deposit_memory_node, _cc_callosum_consolidate

    if idle_steps is None:
        # LAW 5, and deliberately the SAME env name the nightly cc-ng-sync cron
        # already exports on both halves (cc_ng_organism.py:1938). No new knob.
        idle_steps = max(0, int(os.environ.get("CC_NG_IDLE_STEPS", "250")))

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

    held = _load_membership(membership_path)
    stats = {
        "status": "ok", "path": conduit_path, "sender": sender,
        "embedding_model": wire_model,
        "absorbed_nodes": 0, "absorbed_synapses": 0, "absorbed_hyperedges": 0,
        "skipped_present": 0, "membership_stale_readmitted": 0, "skipped_not_cc": 0,
        "skipped_identity": 0, "bad_embedding": 0,
        "absorbed_without_embedding_DEFECT": 0,
        "skipped_synapses": 0, "skipped_hyperedges": 0,
        "hyperedge_id_reminted": 0,
        "batches_read": 0, "deferred_by_budget": 0,
        "consolidation_passes": 0, "consolidation_steps": 0,
        "consolidation_skipped_unbound_arrivals": 0,
    }

    budget = max_nodes_per_call
    landed_ids: List[str] = []
    # Merge-scoped arrival set for the consolidation guard below. Deliberately
    # NOT `batch_landed` and NOT `landed_ids`: the guard has to see every node
    # this merge has put in play, across all batches. See the guard at the
    # bottom of the loop for why batch scope is unsafe. `landed_ids` feeds the
    # membership snapshot and excludes already-present nodes (:249), which are
    # legitimate endpoints and can be left unbound by a split batch too.
    merge_landed: Set[str] = set()

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
            if nid in held:
                # NO `continue` HERE -- #106. Falling through is the fix.
                #
                # The graph check above already caught every node that is
                # actually present, so this branch can only be reached by a node
                # in (held - graph.nodes): one that was in the receiver's last
                # membership snapshot but has since been destroyed locally.
                # Vetoing on that record made re-delivery impossible forever, and
                # the loss did not stop at the node -- Tier 2 requires both
                # endpoints in graph.nodes and Tier 3 every member, so a single
                # permanently-vetoed node shredded all of its incident structure
                # on every subsequent pass. The snapshot is the sender's
                # bookkeeping (exclude_ids); presence in the graph is the
                # receiver's authority. A held-but-absent node that arrived anyway
                # means the sender chose to re-send it -- honour the delivery.
                # (After #110 the sender WILL re-send it: a culled node drops out
                # of cc_current_membership, so exclude_ids no longer names it.)
                # Counted at the absorption site below, not here -- this point
                # is only "detected", and the node can still be rejected by the
                # provenance gates or the deposit try/except before it lands.
                logger.info(
                    "CC topology: %s was in our last membership snapshot but is "
                    "absent from the graph -- re-absorbing (culled since, #106)", nid)

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
            if nid in held:
                # Counted here, after the node has actually landed, so the stat
                # reports re-admissions that happened rather than ones merely
                # attempted. `held` is not mutated inside this loop, so this
                # re-test is the same predicate evaluated at the Tier-1 branch.
                stats["membership_stale_readmitted"] += 1
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

        # --- Consolidation: sleep on this batch before taking the next ------
        # FatherGraph Finding 3 / ruling condition (c) / #108. Tier 3 has run,
        # so everything this batch could bind is bound; the steps let threshold
        # adaptation, synaptic scaling and excitability catch up before more
        # foreign topology lands.
        #
        # GUARDED, and the guard is the whole reason this is not a two-line
        # change. Consolidation advances graph.timestep, and
        # orphan_node_grace_period is denominated in exactly that clock (default
        # 25). idle_steps defaults to 250. So running the steps while any node
        # that landed in THIS merge is still unbound would march it straight
        # past grace and hand it to the orphan sweep -- authoring
        # CC-CALLOSUM-TRUTH.md §8.2's cohort cliff into the merge path, in the
        # name of a fix for it. A node is left unbound here when its binding
        # structure was split across batches or skipped (skipped_synapses /
        # skipped_hyperedges), which whole-containment permits.
        #
        # THE PREDICATE IS MERGE-SCOPED, NOT BATCH-SCOPED, and that is the
        # whole point. Binding splits ACROSS batches (the comment above), so a
        # batch-scoped check cannot see the node it needs to protect: batch 1
        # lands X unbound and correctly skips; batch 2 lands whole, X is not in
        # batch 2's set, the guard passes, and 250 steps age X from 0 to 250
        # against a grace of 25. The merge kills the arrival the guard exists
        # to protect, one batch later. `merge_landed` accumulates every arrival
        # this call has put in play so the guard stays honest for all of them.
        #
        # So: consolidate only when everything this merge has landed is bound.
        # Otherwise skip, count, and log loudly -- an unbound arrival is a real
        # defect worth seeing, and deferring its consolidation costs only
        # integration quality, whereas running it costs the node.
        merge_landed |= batch_landed
        if idle_steps > 0 and merge_landed:
            unbound = _unbound_nodes(graph, merge_landed)
            if unbound:
                stats["consolidation_skipped_unbound_arrivals"] += len(unbound)
                logger.error(
                    "CC topology: skipping %d consolidation step(s) after batch %d -- "
                    "%d of %d arrival(s) so far this merge are still unbound (no "
                    "synapse, no hyperedge) and grace is denominated in the clock "
                    "consolidation would advance. Sample: %s. This is a "
                    "whole-containment gap upstream, not a consolidation problem "
                    "(#108/#107).",
                    idle_steps, stats["batches_read"], len(unbound), len(merge_landed),
                    sorted(unbound)[:3])
            elif _cc_callosum_consolidate(graph, idle_steps):
                stats["consolidation_passes"] += 1
                stats["consolidation_steps"] += idle_steps

    # #110: overwrite the membership snapshot with the receiver's CURRENT CC
    # membership -- what the sender reads as exclude_ids. Full overwrite, not an
    # append of `landed_ids`: a node culled since the last pass must DROP OUT so
    # the sender re-sends it (the send-side counterpart of #106's receive-side
    # re-admission). Sourced from graph.nodes, so re-admitted nodes reappear and
    # culled ones vanish without any per-pass dedup bookkeeping.
    _write_membership(membership_path, cc_current_membership(graph))

    stats["completed"] = stats["deferred_by_budget"] == 0
    logger.info(
        "CC topology merge from %s: +%d node(s), +%d synapse(s), +%d hyperedge(s); "
        "%d deferred to next pass",
        sender, stats["absorbed_nodes"], stats["absorbed_synapses"],
        stats["absorbed_hyperedges"], stats["deferred_by_budget"],
    )
    return stats


def _unbound_nodes(graph: Any, node_ids: Set[str]) -> Set[str]:
    """Which of `node_ids` are anchored by nothing the orphan sweep respects.

    Mirrors the sweep's own predicate (neuro_foundation.py:3487): a node is
    reap-eligible only when it has no outgoing synapse, no incoming synapse AND
    no hyperedge membership. Hyperedge membership is an independent, equally
    sufficient anchor -- counting synapse degree alone reports catastrophic
    false positives (CC-CALLOSUM-TRUTH.md §1.1, and note synapses key on
    pre_node_id/post_node_id, not source_id/target_id).

    Identity protection is deliberately NOT consulted: this asks "would
    advancing the clock endanger this arrival", and a protected node is spared
    for reasons unrelated to whether the batch bound it. Treating protection as
    boundness would let a half-bound batch consolidate.
    """
    outgoing = getattr(graph, "_outgoing", {}) or {}
    incoming = getattr(graph, "_incoming", {}) or {}
    node_hyperedges = getattr(graph, "_node_hyperedges", {}) or {}
    return {
        nid for nid in node_ids
        if not outgoing.get(nid)
        and not incoming.get(nid)
        and not node_hyperedges.get(nid)
    }


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
