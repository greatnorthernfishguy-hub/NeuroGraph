#!/usr/bin/env python3
# SEE FIRST: /home/josh/docs/CC-CALLOSUM-TRUTH.md -- consolidated, verified state of
# the callosum, wholeness ring, hyperedge binding and orphan collection (2026-07-31).
# The wholeness ring ALREADY EXISTS here (Leg 2). Open defect: merge-journal poison-pill.
# ---- Changelog ----
# [2026-10-01] Claude Sonnet 5.5 (Z12 builder, lane emergent-want-bound-905, dispatch #13491) — #905 ROUND 2: COMMENTS AND DOCSTRINGS ONLY
# What: (E.2) the two stale "no production caller yet (Phase 3)" header comments below are corrected: the daemon's handle_merge_topology
#   (scripts/cc-ng-daemon.py, imports and calls merge_cc_topology) IS a production caller. (E.1) the "pre-existing" wording in the stats
#   comment, the merge_landed comment, the merge_cc_topology docstring and a NOTE above the ERROR now says what is true: "pre-existing" =
#   NOT in merge_landed (not delivered by this merge); a node the conduit re-sent that the receiver already held IS in merge_landed and counts
#   under "from this merge" / ..._arrivals. (P493 R1) the _unbound_nodes docstring and the comment at its predicate now state what "bound" means
#   for these guards (NOT sweep-eligible; NOT "a complete turn"; turn completeness is not a gate condition), next to the age exclusion and the
#   P492 wait-then-escalate sentence that were already there.
# Why: the #905 pair (checker-041 + le-054) found these two brief items not delivered by build-001 (ee94f7d2); Chief-003 ROUND 2 + AMENDMENTS 1/2.
# How: NO code change. Every non-comment, non-docstring token is identical to ee94f7d2 (tokenize + ast.dump proof in returns/build-002.md).
#   The :694 ERROR format string is BYTE-IDENTICAL; its loose parenthetical is REPORTED, not changed (AMENDMENT 2). No test file edited.
# [2026-10-01] Claude Sonnet 5.5 (Z12 builder, lane ack-bound-918, dispatch #14104) — #918 fold-up 2: D1 + Exec P534 wording (comments/docstrings ONLY)
# What: (D1) the `cc_current_membership` docstring is rewritten TRUE and non-contradictory: it is every CC node currently held;
#   the ACK / exclude_ids source is `cc_ack_membership` (its old opening sentence said the opposite of its own pointer paragraph).
#   (P534) the superseded P522 "H-1" clause (a blanket no-write-to-identity-nodes rule) is DROPPED everywhere it was live in this file
#   (the entry below, the `cc_ack_membership` docstring, the end-of-call comment) and replaced with: "Protected nodes are excluded
#   from re-offer because they are not sweep-eligible (they survive at any degree). Identity-touching binding structure still
#   crosses per #147 (identity crosses the callosum)."
# Why: le-056 D1; Exec P534 (via Chief-003): Q3 answered NOT a breach (the sender's synapse to an already-acked protected node is
#   by design, #147); H-1 as ruled gates deletion/re-tag/de-flag/edit/re-text/placement, not additive cross-hemisphere structure.
# How: NO code change: `ast.dump` of this module with docstrings stripped is IDENTICAL before and after (proved in build-003).
# [2026-10-01] Claude Sonnet 5.5 (Z12 builder, lane ack-bound-918, dispatch #13930) — #918 fold-up: Q1 (the NAME), Q2, N4
# What: (Q1, Exec P522, LAW 4) `cc_current_membership` is BACK to exactly its original meaning, byte-equivalent in behaviour to
#   base ee94f7d2 (the CC-provenance nodes currently held; its docstring was rewritten by the fold-up 2 entry above). The ack
#   the sender reads as exclude_ids is a NEW pure function `cc_ack_membership` = `cc_current_membership` minus `_unbound_nodes`
#   (the #905 predicate, called, never copied). The ack writer at the end of `merge_cc_topology` calls the NEW function.
#   (Q2, confirmed intended) protected unbound CC nodes stay in the ack and are never re-offered. Protected nodes are excluded
#   from re-offer because they are not sweep-eligible (they survive at any degree). Identity-touching binding structure still
#   crosses per #147 (identity crosses the callosum). (N4) the re-offer streak table is now ALSO reset when the merge raises
#   TopologyMergeAbort (MACHINE_ID unset, no header, own export, embedding-model mismatch): ONE helper
#   `_reset_reoffer_streaks`, used by the two early returns (no_conduit / empty) and the four abort sites; the two inline
#   copies the early returns used are gone.
# Why: le-056 (ROLE B, ETHOS DRIFT, no Law violation) N4; Exec P522 Q1/Q2. A canonical query named for what it returns must
#   return that (the narrowing belongs in a separate function); a streak carried across an aborted call could later yield a
#   spurious WARNING for a node that was not re-offered consecutively.
# How: see `cc_current_membership`, `cc_ack_membership`, `_reset_reoffer_streaks`. No sender change, no ack-content change
#   (the ack file is byte-identical to the 86cf5afd one), no new env variable.
# [2026-10-01] Claude Sonnet 5.5 (Z12 builder, lane ack-bound-918, dispatch #13352) — #918: the ack means "I hold it BOUND"
# (NAME SUPERSEDED by the entry above: the narrowed set is `cc_ack_membership`; `cc_current_membership` is the base function.)
# What: (1) `cc_current_membership` (the snapshot the sender reads as exclude_ids) is now the CC-provenance nodes MINUS
#   the #905 sweep-eligible-unbound set (`_unbound_nodes`, called, never copied). Same name, signature and return type;
#   still a PURE QUERY. (2) `merge_cc_topology` now counts, IN MEMORY ONLY, how many consecutive merge calls re-sent a
#   node the receiver already held while it STAYED sweep-eligible-unbound (`_track_reoffers`, a separate function: the
#   query stays pure) and logs ONE redacted WARNING per node when the streak reaches N
#   (`_REOFFER_WARN_STREAK_DEFAULT` = 5, env CC_TOPOLOGY_REOFFER_WARN_STREAK). New stats keys `reoffered_unbound`,
#   `reoffer_streak_warnings`, `reoffer_streak_nodes_at_or_over`. Nothing else in the merge changed (Tier 1/2/3,
#   budget, guards, `_load_membership`, `_write_membership`).
# Why: Exec Packet 496 via Chief-003 (RULED ADOPT, S4b-GATING). The #110 snapshot counted nodes the receiver holds
#   UNBOUND, so after the first merge+push none of the laptop's 147 unbound cohort nodes was ever re-offered
#   (exclude_ids): the #897/#905 clock hold forbids the cull and the #110 ack forbade the re-send -- a deadlock. A
#   held-UNBOUND node still needs its binding; counting it acked was the defect. No deletion, no absence window and no
#   per-node ack state: the ack is just recomputed from the same two live facts (CC provenance, bound-ness).
# How: see the docstrings of `cc_current_membership` and `_track_reoffers`. RECEIVER-SIDE ONLY: the sender
#   (cc_topology_export.py) is untouched. Cost: one whole-graph pass per merge call (the #897 guard measured ~9 ms at
#   16k nodes). The one new env variable is CC_TOPOLOGY_REOFFER_WARN_STREAK (LAW 5).
# [2026-10-01] Claude Sonnet 5.5 (Z12 builder, lane emergent-want-bound-905, dispatch #13138) — #905 parts B, C, D (merge side)
# What: (B) `_unbound_nodes` is now unbound AND sweep-eligible: a node the orphan sweep would never reap
#   (graph._is_identity_protected: constitutional / '*_authored') no longer blocks the clock. Its signature is
#   unchanged. The sweep's AGE term is left out ON PURPOSE (see the docstring). (C) `redact_node_id`, the ONE
#   redaction rule for any unbound-id sample: '<kind>:<12 hex of sha256(id)>', never the id; applied to the #897
#   per-batch ERROR sample (tree ids embed the user's concept text). (D) `whole_graph_guard(graph)`, the ONE
#   constructor of the per-slice guard both _cc_callosum_consolidate callers use; the merge now PASSES it plus a
#   `progress` dict, and reports a pass held mid-way as `consolidation_held_midpass` (not counted as consolidated).
#   #897's batch-end check stays as the OUTER check (does the pass start at all).
# Why: Exec Packets 489/490 (Chief-003), le-047 C3 / le-048. A protected unbound node blocked consolidation forever
#   although the sweep spares it; a raw id sample printed the user's words; a node that became unbound BETWEEN the
#   25-step slices aged against orphan grace 25 through the rest of the 250 (the guard was evaluated once).
# How: see the docstrings of `_unbound_nodes`, `redact_node_id`, `whole_graph_guard`. No env variable, no knob.
# [2026-09-30] Claude Sonnet 5.5 (Z12 builder, lane merge-guard-897, dispatch #12973) — #897: the consolidation guard covers the WHOLE graph
# What: the between-batch consolidation guard no longer asks `_unbound_nodes(graph, merge_landed)`
#   (this merge's arrivals only); it asks `_unbound_nodes(graph, set(graph.nodes))` -- the SAME
#   predicate the daemon's #896 rule 1 uses. If ANY unbound node exists (a pre-existing one OR an
#   arrival that did not bind) the batch's idle steps are skipped, counted and logged LOUD (one
#   ERROR per blocked batch); once none exists they run exactly as before. New stats keys:
#   consolidation_skipped_unbound_preexisting, consolidation_blocked_batches;
#   consolidation_skipped_unbound_arrivals keeps its meaning (this merge's arrivals still unbound
#   at a skip). `_unbound_nodes` itself, `_cc_callosum_consolidate`, Tier 1/2/3 and the membership
#   snapshot are untouched. No env variable, no knob, no default change.
# Why: S4 step-source run-down (Exec Packet 484, Finding 1, HIGH; Chief-003 ruled GO, Packet 485 /
#   #897). The merge-scoped guard never saw the laptop CC graph's 147 PRE-EXISTING unbound
#   conversational nodes (15 forests + 132 trees, source=cc_gateway). The first Leg 2 tick whose
#   arrivals were all bound would have run 250 steps against orphan_node_grace_period 25 and the
#   orphan sweep (neuro_foundation.py _collect_orphan_nodes) would have reaped every still-unbound
#   one of them in that single tick -- the defect Packet 482 (#896 rule 1) reversed on the daemon's
#   drain side (CC-CALLOSUM-TRUTH.md §2, §10.4-H: the clock stays gated until the backlog is wired;
#   do not 'fix' the cull by disabling, host-scoping or exempting). The merge's OWN copy of the
#   guard had never been widened. This SUPERSEDES the "Guard scope: MERGE-scoped" paragraph of the
#   2026-08-02 (#108) entry below: that scope was right about batch-vs-merge, and wrong about
#   merge-vs-graph.
# How: the predicate is evaluated where it was (inside graph._step_lock, only when idle_steps > 0
#   and the merge landed something); nothing is held across the steps. A node that cannot bind now
#   blocks consolidation loudly until it does -- deferring consolidation costs integration
#   quality; running it costs the node.
# [2026-09-11] Codex — #423 completed topology batches are capture boundaries
# What: serialize in-memory Tier 1/2/3 apply and membership reads on _step_lock.
# Why: graph/vector capture must not observe nodes before their binding arrives.
# How: release before consolidation (concurrent -> step order) and membership I/O;
#      retain the existing budget, grace, admission and 25/250 stepping behavior.
# [2026-08-28] Claude Code (DudeMan CC, Opus 4.8) — #147 amendment (receiving half): identity crosses the callosum
# What: removed the merge-time identity-rejection gate (was: skip any node with
#       metadata['constitutional'] or provenance ending '_authored'). Identity CC
#       nodes now absorb like any other CC node. Sender+receiver share ONE admission
#       predicate, is_cc_provenance; protection moves to the correct layer.
# Why: the sender half (#147 amendment in cc_topology_export) stopped withholding
#       identity because the callosum is white matter between two hemispheres of one
#       mind, not a foreign donation. Leaving this receiver gate in place would have
#       made the change not just inert but HARMFUL: identity nodes would ship, be
#       dropped here, and any hyperedge binding them would then fail the Tier-3
#       members.issubset(graph.nodes) check and be dropped whole -- silently
#       shredding identity-touching binding structure. Both ends must move together.
# How: identity, once absorbed, is protected at prune/orphan time by
#       neuro_foundation._is_identity_protected (3517), whose own docstring names
#       this case ("a want arrives synapse-poor via corpus-callosum consolidation,
#       #70"); the constitutional/provenance flags ride _portable_metadata and land
#       in node.metadata, which is exactly what that protector keys on. is_cc_provenance
#       at the top of the node loop still scopes admission to CC's own mind (no Syl
#       node, no foreign path). Josh-directed (identity crosses the callosum);
#       reviewed by neurograph-law-enforcer, which caught the missing receiver half.
#       [Corrected 2026-10-01, #905 round 2: the original sentence here -- "merge_cc_topology has no production caller yet (Phase 3), so
#       blast radius is 0" -- is STALE. merge_cc_topology HAS a production caller: the daemon's handle_merge_topology
#       (scripts/cc-ng-daemon.py, which imports it and calls it), so a change to this function is NOT blast-radius 0.]
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
#   Phase 3. [Corrected 2026-10-01, #905 round 2: the original "Not yet load-bearing (merge_cc_topology has no production caller)" is
#   STALE -- the daemon's handle_merge_topology (scripts/cc-ng-daemon.py) IS a production caller, so this is load-bearing.]
#   Wired correct so #88's first live run inherits it.
# -------------------

import hashlib
import json
import logging
import os
import re
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

# #918 receiver-side starvation alarm (see `_track_reoffers`). The sender re-offers a
# node the receiver holds UNBOUND, oldest-first by VPS creation_time; a node that can
# never bind could occupy frame budget every tick. The receiver cannot prevent that
# (it is sender-side) but it must SEE and SAY it: one WARNING per node when it has been
# re-offered `N` consecutive merge calls and is STILL sweep-eligible-unbound.
# `N` = _REOFFER_WARN_STREAK_DEFAULT unless the env variable below (LAW 5; the ONLY new
# one) overrides it. The table is IN MEMORY only (process lifetime): never persisted,
# keyed by the redacted 12-hex sha256 form (never the raw id), HARD-CAPPED at
# _REOFFER_TABLE_CAP entries and replaced -- never mutated -- on every call.
_REOFFER_WARN_STREAK_DEFAULT = 5
_REOFFER_WARN_STREAK_ENV = "CC_TOPOLOGY_REOFFER_WARN_STREAK"
_REOFFER_TABLE_CAP = 1024
_reoffer_streaks: Dict[str, int] = {}


class TopologyMergeAbort(RuntimeError):
    """Raised when the conduit must not be absorbed at all (model mismatch,
    self-absorption, malformed header). Distinct from per-item skips, which are
    counted and logged but never abort the pass."""


def cc_current_membership(graph: Any) -> Set[str]:
    """The CC node IDs the receiver currently HOLDS: every CC-provenance node in
    `graph.nodes`, bound or not, protected or not. It is "what is held" and
    nothing narrower.

    It is NOT the ack. The ACK the sender reads as exclude_ids (#110, narrowed by
    #918) is `cc_ack_membership` -- this set minus the sweep-eligible-unbound set;
    the ack writer calls that, not this.

    Read live from `graph.nodes` intersected with CC provenance. Because it is
    regenerated from the graph rather than appended to, a node culled locally
    (#104 sweep, orphan collection, a rolled-back checkpoint) DROPS OUT of it --
    and so out of the ack built from it -- so the exporter re-sends that node and
    #106's receive-side re-admission has something to re-admit. The append-only
    journal this replaces could only grow, so a culled id stayed excluded forever
    and the node was permanently un-resendable: the §3 poison-pill, merely
    relocated to the send side. After #110, presence in the graph is the
    authority on BOTH sides -- receive admission (#106) and send exclusion.

    Uses the same predicate the exporter classifies with (is_cc_provenance, via
    cc_topology_export._is_cc_node), so what the receiver advertises as held and
    what the sender considers CC-exportable cannot drift apart.
    """
    with graph._step_lock:
        return {nid for nid, node in graph.nodes.items()
                if is_cc_provenance(nid, getattr(node, "metadata", None) or {})}


def cc_ack_membership(graph: Any) -> Set[str]:
    """The ACK: the CC nodes the receiver holds BOUND -- what the SENDER reads as
    exclude_ids (#110, narrowed by #918 / Exec Packet 496; split out of
    `cc_current_membership` by the Exec's P522 ruling Q1, LAW 4).

    THE ACK MEANS "I HOLD IT BOUND (not sweep-eligible)". It is
    `cc_current_membership(graph)` MINUS `_unbound_nodes` of it -- the #905
    sweep-eligible-unbound set (no synapse, no hyperedge, not identity-protected),
    the SAME function the #897/#905 clock hold and the daemon's #896 rule 1 use
    (LAW 3: called, never copied). So a bound node is in; a PROTECTED unbound node
    (constitutional / '*_authored') is in too -- it is "held", exactly as the guard
    treats it; an unprotected unbound node is OUT. Protected nodes are excluded from
    re-offer because they are not sweep-eligible (they survive at any degree).
    Identity-touching binding structure still crosses per #147 (identity crosses the
    callosum).

    WHY THE UNBOUND ONES MUST NOT BE IN. The snapshot used to be every CC node
    held, bound or not. A node the receiver holds UNBOUND then counted as acked and
    was never a sender candidate again, while the clock hold (#897/#905) forbids the
    cull that would have dropped it out of the snapshot (#110's recovery) -- a
    deadlock: the cull-and-resend recovery and the hold forbid each other, and the
    147-node unbound cohort could never be re-offered with the hyperedges that bind
    it. A held-UNBOUND node still needs its binding; "I hold it bound" is the only
    reading of an ack under which the sender keeps offering it. No deletion, no
    absence window and no per-node ack state is involved: the ack is recomputed
    from the live graph on every call.

    What #110 established is unchanged: a node culled locally DROPS OUT of the ack
    (it is no longer held), so the exporter re-sends it and #106's receive-side
    re-admission has something to re-admit. Presence in the graph stays the
    authority on both sides -- receive admission (#106) and send exclusion -- and
    now BOUND presence for send.

    PURE QUERY (LAW 4): no counter, no log, no state, no write. The re-offer
    starvation counter lives in `_track_reoffers`, called from the merge, never
    here. Cost: one whole-graph pass per call (the `_unbound_nodes` walk; the #897
    guard measured ~9 ms at 16k nodes) -- once per merge call.
    """
    with graph._step_lock:
        held = cc_current_membership(graph)
        return held - _unbound_nodes(graph, held)


def _reset_reoffer_streaks() -> None:
    """The ONE reset of the #918 re-offer streak table (in-memory only). A merge
    call that observed no re-offer -- the early returns (no_conduit / empty) and a
    TopologyMergeAbort -- resets every streak, so a streak carried across such a
    call cannot later produce a spurious "N consecutive" WARNING. The table is
    replaced, never mutated (see `_track_reoffers`)."""
    global _reoffer_streaks
    _reoffer_streaks = {}


def _track_reoffers(graph: Any, reoffered: Set[str]) -> Dict[str, int]:
    """The #918 receiver-side re-offer counter (bounded and LOUD). Bookkeeping, so
    it lives HERE and not in `cc_ack_membership` (LAW 4: the query stays pure).

    `reoffered` = the ids this merge call's frames carried that the receiver
    already held. Of those, the ones STILL sweep-eligible-unbound after the call
    (`_unbound_nodes`, the one shared predicate) extend their streak by one; every
    other entry is dropped -- a node that became bound, was not re-sent in this call,
    or is protected resets to zero. When a node's streak reaches `N` consecutive
    merge calls (`_REOFFER_WARN_STREAK_DEFAULT`, overridable by
    CC_TOPOLOGY_REOFFER_WARN_STREAK) ONE WARNING is logged for it; it is not logged
    again unless the streak resets and climbs back to `N`.

    Privacy (LAW 7): the table key and the record carry ONLY the redacted
    `<kind>:<12 hex of sha256(id)>` form from `redact_node_id` -- never the id and
    never text; tree ids embed the user's own words. The table is in memory only (no
    persisted per-node state, nothing in the ack), hard-capped at
    `_REOFFER_TABLE_CAP`: when more nodes than that qualify in one call the longest
    streaks are kept (the ones nearest to / past the alarm) and the rest start over.
    The table is rebuilt and swapped as a whole, so a concurrent reader never sees a
    half-updated one.

    Returns {"still_unbound", "warnings", "at_or_over"} for the stats dict.
    """
    global _reoffer_streaks
    try:
        n = int(os.environ.get(_REOFFER_WARN_STREAK_ENV, _REOFFER_WARN_STREAK_DEFAULT))
    except (TypeError, ValueError):
        n = _REOFFER_WARN_STREAK_DEFAULT
    n = max(1, n)
    with graph._step_lock:
        still = _unbound_nodes(graph, set(reoffered))
    prior = _reoffer_streaks
    streaks = {}
    for nid in still:
        key = redact_node_id(nid)
        streaks[key] = prior.get(key, 0) + 1
    if len(streaks) > _REOFFER_TABLE_CAP:
        keep = sorted(streaks, key=lambda k: (-streaks[k], k))[:_REOFFER_TABLE_CAP]
        streaks = {k: streaks[k] for k in keep}
    _reoffer_streaks = streaks
    at_or_over = sum(1 for s in streaks.values() if s >= n)
    warnings = 0
    for key in sorted(streaks):
        if streaks[key] == n:
            warnings += 1
            logger.warning(
                "CC topology: node %s has been re-offered by the sender in %d consecutive "
                "merge call(s) and is STILL unbound and sweep-eligible (alarm threshold "
                "N=%d, %s; %d node(s) now at/over it). A node that cannot bind and is "
                "re-offered every tick can occupy the sender's frame budget (candidates "
                "are ordered oldest-first by creation_time): #918 receiver-side "
                "starvation alarm. The id is redacted (kind:sha256-prefix).",
                key, streaks[key], n, _REOFFER_WARN_STREAK_ENV, at_or_over)
    return {"still_unbound": len(still), "warnings": warnings, "at_or_over": at_or_over}


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

    THE GUARD COVERS THE WHOLE GRAPH (#897). The steps are skipped -- counted
    and logged at ERROR, once per blocked batch -- while ANY node in the graph
    is unbound AND sweep-eligible (`_unbound_nodes`: no synapse, no hyperedge,
    not identity-protected; #905): one of this merge's own arrivals that
    did not bind, OR one that was already there. A PROTECTED unbound node (the
    sweep spares it) does not block. The first guard (#108) asked
    only about this merge's arrivals; it never saw the pre-existing unbound
    conversational cohort, so the first merge whose arrivals were all bound
    would have run 250 steps against orphan grace 25 and handed the cohort to
    the orphan sweep. Leg 2 frames still merge and bind (that is how the cohort
    gets bound); only the clock is held. A node that cannot bind blocks
    consolidation, loudly, until it does. Stats: `consolidation_blocked_batches`
    (+1 per blocked batch), `consolidation_skipped_unbound_arrivals` (this
    merge's arrivals still unbound at a skip) and
    `consolidation_skipped_unbound_preexisting` (the rest, summed per blocked
    batch). Same predicate as the daemon's #896 rule 1. Wording note (#905 round 2):
    "pre-existing" means NOT in `merge_landed` (not delivered by this merge). A node
    the conduit re-sent that the receiver already held IS in `merge_landed`, so it is
    counted under "from this merge" / `consolidation_skipped_unbound_arrivals`.

    THE GUARD IS ALSO PER SLICE (#905 part D). The check above decides whether a
    pass STARTS. The pass itself (`_cc_callosum_consolidate`) re-checks the same
    predicate (`whole_graph_guard`) before EACH lock slice, so a node that becomes
    unbound between slices stops the pass at the slice boundary instead of aging
    through the rest of the 250 steps. A pass held that way is counted in
    `consolidation_held_midpass`, never in `consolidation_passes` /
    `consolidation_steps`.

    THE ACK MEANS "I HOLD IT BOUND" (#918). The membership snapshot written at the
    end of every call (the sender's exclude_ids) is `cc_ack_membership`: the CC
    nodes currently held (`cc_current_membership`) MINUS the sweep-eligible-unbound
    set, so a node this call left unbound is NOT acked and the sender re-offers it. A re-offered node the
    receiver already holds is `skipped_present` in Tier 1 and is a usable
    endpoint, so Tier 2 lands the frame's synapses onto it and Tier 3 its
    hyperedge (a bound node then enters the next ack). A node that stays
    unbound across `N` consecutive calls is reported, once, by `_track_reoffers`
    (stats `reoffered_unbound`, `reoffer_streak_warnings`,
    `reoffer_streak_nodes_at_or_over`).
    """
    from cc_ng_organism import _cc_deposit_memory_node, _cc_callosum_consolidate

    if idle_steps is None:
        # LAW 5, and deliberately the SAME env name the nightly cc-ng-sync cron
        # already exports on both halves (cc_ng_organism.py:1938). No new knob.
        idle_steps = max(0, int(os.environ.get("CC_NG_IDLE_STEPS", "250")))

    local_machine_id = local_machine_id or os.environ.get("MACHINE_ID")
    if not local_machine_id:
        _reset_reoffer_streaks()   # #918/N4: an aborted call resets the streaks too
        raise TopologyMergeAbort(
            "MACHINE_ID unset -- cannot verify this conduit is not our own export")

    if not os.path.exists(conduit_path):
        _reset_reoffer_streaks()   # #918: a call that re-sent nothing resets every streak
        return {"status": "no_conduit", "path": conduit_path, "absorbed_nodes": 0}

    with open(conduit_path, "rb") as fh:
        raw = fh.read()

    frames = list(read_topology_frames(raw))
    if not frames:
        _reset_reoffer_streaks()   # #918: same -- nothing was re-sent in this call
        return {"status": "empty", "path": conduit_path, "absorbed_nodes": 0}

    header = frames[0]
    if header.get("kind") != "header":
        _reset_reoffer_streaks()   # #918/N4
        raise TopologyMergeAbort(
            f"conduit {conduit_path} does not begin with a header frame "
            f"(got kind={header.get('kind')!r})")

    sender = header.get("machine_id")
    if sender == local_machine_id:
        _reset_reoffer_streaks()   # #918/N4
        raise TopologyMergeAbort(
            f"conduit was authored by this machine ({sender}) -- refusing to "
            "re-absorb our own export")

    wire_model = header.get("embedding_model")
    local_model = expected_embedding_model or _local_embedding_model()
    if wire_model != local_model and "unknown" not in (wire_model, local_model):
        # FatherGraph Finding 6. Not degradable -- see module header.
        _reset_reoffer_streaks()   # #918/N4
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
        # #897: the whole-graph guard. `..._arrivals` counts the unbound nodes that
        # are in `merge_landed`: every node this conduit delivered, INCLUDING a node it
        # re-sent that the receiver ALREADY held (the skipped_present branch adds it to
        # `batch_landed` too: "present == usable as an endpoint"); that is its original
        # meaning. `..._preexisting` counts the unbound nodes NOT in `merge_landed`,
        # summed per blocked batch. "Pre-existing" is a loose label: it means "not
        # delivered by this merge", NOT "was in the graph before it" -- a re-sent
        # already-present unbound node lands in the OTHER bucket (arrivals).
        # `consolidation_blocked_batches` is +1 per blocked batch. The decision
        # (whole graph, sweep-eligible) is unaffected by the split.
        # (#905 round 2: wording only; keys and counting unchanged.)
        "consolidation_skipped_unbound_preexisting": 0,
        "consolidation_blocked_batches": 0,
        # #905 part D: passes the batch-end guard let START but the per-slice guard
        # stopped part-way (a node became unbound between slices). NOT counted in
        # consolidation_passes / consolidation_steps: held, not done.
        "consolidation_held_midpass": 0,
        # #918 receiver-side starvation alarm (`_track_reoffers`): nodes the frames
        # re-sent that the receiver already held and that are STILL sweep-eligible-
        # unbound after this call; the WARNINGs logged this call (a node's streak just
        # reached N); and the nodes at/over N in the in-memory table.
        "reoffered_unbound": 0,
        "reoffer_streak_warnings": 0,
        "reoffer_streak_nodes_at_or_over": 0,
    }

    budget = max_nodes_per_call
    landed_ids: List[str] = []
    # Merge-scoped arrival set. Deliberately NOT `batch_landed` and NOT
    # `landed_ids`: it is every node this merge has put in play, across all
    # batches. Since #897 the consolidation guard asks about the WHOLE graph, so
    # this set no longer bounds the guard; it (a) keeps the guard's trigger
    # (the merge landed something) and (b) splits a skip's unbound nodes into
    # "from this merge" vs "pre-existing" for the stats and the ERROR record.
    # `landed_ids` feeds the membership snapshot and excludes already-present
    # nodes (:249), which are legitimate endpoints and can be left unbound by a
    # split batch too -- `merge_landed` includes them. So "pre-existing" in the
    # stats and the ERROR means "NOT in `merge_landed`" (not delivered by this
    # merge), NOT "was already in the graph": a node the conduit re-sent that the
    # receiver already held is counted under "from this merge" /
    # `consolidation_skipped_unbound_arrivals`. (#905 round 2: wording only.)
    merge_landed: Set[str] = set()
    # #918: ids the frames carried that the receiver ALREADY held (Tier 1's
    # `skipped_present`), across all batches of this call. Fed to `_track_reoffers`.
    reoffered_present: Set[str] = set()

    for frame in frames[1:]:
        if frame.get("kind") != "batch":
            continue
        stats["batches_read"] += 1

        if budget <= 0:
            # Trickle discipline: stop cleanly, resume next pass. The journal
            # + graph guard make the resume a no-op for what already landed.
            stats["deferred_by_budget"] += len(frame.get("nodes") or ())
            continue

        # Existing Graph RLock: one bounded topology batch is indivisible to
        # checkpoint capture. Conduit read/msgpack decode already completed.
        # Never hold it through consolidation, which acquires _concurrent_lock.
        with graph._step_lock:
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
                    reoffered_present.add(nid)   # #918: counted by `_track_reoffers`
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
                    # of the ack (cc_ack_membership), so exclude_ids no longer names it.)
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
                # #147 amendment (2026-08-28): identity CROSSES the callosum. The gate
                # that used to reject constitutional / *_authored nodes here has been
                # removed -- walling identity out of the callosum is a split-brain
                # lesion (the sender stopped withholding it in cc_topology_export; this
                # is its receiving half). Sender and receiver now share ONE admission
                # predicate: is_cc_provenance above. Protection is applied at the
                # CORRECT layer -- prune/orphan time, by neuro_foundation
                # _is_identity_protected (3517), which keys on the very metadata flags
                # (constitutional / provenance) that ride the wire via _portable_metadata
                # and land in node.metadata below. That protector's own docstring names
                # this exact case: "a want arrives synapse-poor via corpus-callosum
                # consolidation (#70)". The stats["skipped_identity"] counter (init at
                # the top of this fn) now stays 0 by design -- a nonzero value would
                # mean the split-brain lesion was re-introduced.

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

            merge_landed |= batch_landed
            # #897: the WHOLE graph, not `merge_landed` -- the same call the daemon's
            # #896 rule 1 makes. Evaluated here, under the lock and only when the
            # guard will be consulted, exactly where the merge-scoped one was.
            unbound = (_unbound_nodes(graph, set(graph.nodes))
                       if idle_steps > 0 and merge_landed else set())

        # --- Consolidation: sleep on this batch before taking the next ------
        # FatherGraph Finding 3 / ruling condition (c) / #108. Tier 3 has run,
        # so everything this batch could bind is bound; the steps let threshold
        # adaptation, synaptic scaling and excitability catch up before more
        # foreign topology lands.
        #
        # GUARDED, and the guard is the whole reason this is not a two-line
        # change. Consolidation advances graph.timestep, and
        # orphan_node_grace_period is denominated in exactly that clock (default
        # 25). idle_steps defaults to 250. So running the steps while ANY node
        # in the graph is still unbound would march it straight past grace and
        # hand it to the orphan sweep (neuro_foundation._collect_orphan_nodes) --
        # authoring CC-CALLOSUM-TRUTH.md §8.2's cohort cliff into the merge path,
        # in the name of a fix for it.
        #
        # THE PREDICATE IS WHOLE-GRAPH (#897). What the first guard did (#108):
        # it asked only about THIS merge's arrivals (`merge_landed`, accumulated
        # across batches because binding splits across them, so a batch-scoped
        # check would let a later whole batch age an earlier arrival past grace).
        # That was right about batch-vs-merge and wrong about merge-vs-graph:
        # the laptop CC graph holds 147 PRE-EXISTING unbound conversational nodes
        # (15 forests + 132 trees, source=cc_gateway) that are in nobody's
        # `merge_landed`, so the first Leg 2 tick whose arrivals were all bound
        # passed the guard, ran 250 steps against grace 25, and the sweep reaped
        # every still-unbound one of them in that one tick. The pre-existing
        # cohort binds only OVER Leg 2 (CALLOSUM-TRUTH §2, §10.4-H: the clock
        # stays gated until the backlog is wired -- do not 'fix' the cull by
        # disabling, host-scoping or exempting), so the clock must not move
        # until it has. Same predicate as the daemon's #896 rule 1.
        #
        # So: consolidate only when NO node in the graph is unbound AND sweep-eligible
        # (#905: `_unbound_nodes` leaves out identity-protected nodes, which the sweep
        # never reaps, and deliberately has no age term). Otherwise
        # skip, count, and log loudly (one ERROR per blocked batch, never
        # rate-limited). Frames still merge and bind -- Tier 3 is untouched; only
        # the clock is held. An unbound node that cannot bind blocks
        # consolidation, loudly, until it does: deferring consolidation costs
        # only integration quality, whereas running it costs the node.
        if idle_steps > 0 and merge_landed:
            if unbound:
                from_merge = unbound & merge_landed
                preexisting = unbound - merge_landed
                stats["consolidation_skipped_unbound_arrivals"] += len(from_merge)
                stats["consolidation_skipped_unbound_preexisting"] += len(preexisting)
                stats["consolidation_blocked_batches"] += 1
                # NOTE (#905 round 2; a comment ONLY -- the format string below is
                # BYTE-IDENTICAL, a log string whose change is its own look): its
                # parenthetical "pre-existing (not landed by this merge)" is loose.
                # `preexisting` is `unbound - merge_landed`, i.e. NOT delivered by this
                # merge; a node the conduit re-sent that the receiver already held is in
                # `merge_landed` and is counted in the "from this merge" figure. The
                # accurate wording is "not delivered by this merge" (stats comment above).
                logger.error(
                    "CC topology: skipping %d consolidation step(s) after batch %d -- "
                    "%d node(s) in the graph are still unbound (no synapse, no "
                    "hyperedge): %d pre-existing (not landed by this merge), %d from "
                    "this merge (%d landed so far). Sample: %s. The clock is held "
                    "because orphan grace is denominated in the clock consolidation "
                    "would advance, and the pre-existing cohort binds only over Leg 2 "
                    "(CC-CALLOSUM-TRUTH §2/§10.4-H). Consolidation stays blocked "
                    "until every one of them is bound (#897). Sample ids are redacted "
                    "(kind:sha256-prefix, #905): tree ids embed the user's own words.",
                    idle_steps, stats["batches_read"], len(unbound),
                    len(preexisting), len(from_merge), len(merge_landed),
                    [redact_node_id(n) for n in sorted(unbound)[:3]])
            else:
                # #905 part D: the per-slice guard re-checks the whole graph before
                # EACH 25-step slice, inside the shared _cc_callosum_consolidate
                # (the daemon's drain builds the SAME guard from
                # whole_graph_guard). The check above is the OUTER one: it decides
                # whether the pass starts; the per-slice one decides whether it
                # continues. A pass held part-way is reported, not counted as
                # consolidated; the function already logged its own loud record.
                progress: Dict[str, Any] = {}
                if _cc_callosum_consolidate(
                        graph, idle_steps,
                        guard=whole_graph_guard(graph), progress=progress):
                    stats["consolidation_passes"] += 1
                    stats["consolidation_steps"] += idle_steps
                elif progress.get("held"):
                    stats["consolidation_held_midpass"] += 1

    # #918: the receiver-side starvation alarm. Evaluated AFTER every batch (and its
    # Tier 2/3) has run, so "still unbound" means still unbound once this call's
    # frames have bound what they could. Bookkeeping in its own function (the
    # membership query below stays pure). One WARNING per node when its streak of
    # consecutive re-offers reaches N; in-memory only; never in the ack.
    track = _track_reoffers(graph, reoffered_present)
    stats["reoffered_unbound"] = track["still_unbound"]
    stats["reoffer_streak_warnings"] = track["warnings"]
    stats["reoffer_streak_nodes_at_or_over"] = track["at_or_over"]

    # #110: overwrite the membership snapshot with the receiver's CURRENT CC
    # membership -- what the sender reads as exclude_ids. Full overwrite, not an
    # append of `landed_ids`: a node culled since the last pass must DROP OUT so
    # the sender re-sends it (the send-side counterpart of #106's receive-side
    # re-admission). Sourced from graph.nodes, so re-admitted nodes reappear and
    # culled ones vanish without any per-pass dedup bookkeeping.
    #
    # #918: THE ACK MEANS "I HOLD IT BOUND". The ack is `cc_ack_membership` (NOT
    # `cc_current_membership`, which is just what is held): the CC nodes held MINUS
    # the #905 sweep-eligible-unbound set, so a node this call left unbound is NOT
    # acked and the sender keeps re-offering it (with the hyperedge that binds it).
    # Counting a held-unbound node as acked deadlocked against the #897/#905 clock
    # hold: the hold forbids the cull, the ack forbade the re-send. Protected nodes
    # are excluded from re-offer because they are not sweep-eligible (they survive at
    # any degree). Identity-touching binding structure still crosses per #147
    # (identity crosses the callosum).
    _write_membership(membership_path, cc_ack_membership(graph))

    stats["completed"] = stats["deferred_by_budget"] == 0
    logger.info(
        "CC topology merge from %s: +%d node(s), +%d synapse(s), +%d hyperedge(s); "
        "%d deferred to next pass",
        sender, stats["absorbed_nodes"], stats["absorbed_synapses"],
        stats["absorbed_hyperedges"], stats["deferred_by_budget"],
    )
    return stats


def _unbound_nodes(graph: Any, node_ids: Set[str]) -> Set[str]:
    """Which of `node_ids` would the orphan sweep reap once the clock moves.

    #905 / Exec Packet 489 (class ruling, both guards): the guard asks "would
    advancing the clock make the sweep reap this node", so a node counts when it
    has NO outgoing synapse, NO incoming synapse, NO hyperedge membership AND is
    NOT identity-protected. That is the sweep's own test
    (neuro_foundation._collect_orphan_nodes, ~:3595-3603) MINUS ONE TERM -- see
    the age exclusion below. The protection leg is `graph._is_identity_protected`
    itself (constitutional / '*_authored'), CALLED, never copied: one definition
    (LAW 3), so the guard and the sweep cannot drift apart. A protected unbound
    node is spared by the sweep for reasons unrelated to binding, so it must not
    hold the clock forever; an unprotected one (e.g. a '*_emergent' want, a
    conversational forest/tree) is exactly what the clock would reap, so it does.
    This is NOT an exemption FROM the sweep (the sweep is unchanged); it is the
    guard reading its rule by what it protects. Hyperedge membership stays an
    independent, equally sufficient anchor -- counting synapse degree alone
    reports catastrophic false positives (CC-CALLOSUM-TRUTH.md §1.1, and note
    synapses key on pre_node_id/post_node_id, not source_id/target_id).

    WHAT "BOUND" MEANS FOR THESE GUARDS (Exec P493 ruling 1): hyperedge-bound =
    BOUND. "Bound" = NOT SWEEP-ELIGIBLE, i.e. the sweep's own predicate
    (`_collect_orphan_nodes`: a node with an outgoing synapse, an incoming
    synapse, OR hyperedge membership via `_node_hyperedges` is not reaped;
    identity-protected nodes are not reaped either), and it is NOT "a complete
    turn". Turn completeness (a tree with its forest, a forest with its trees) is
    NOT a gate condition -- not an S4b gate condition -- and no guard here tests it.

    THE AGE TERM IS EXCLUDED ON PURPOSE -- do not "fix" it back in. The sweep
    also requires `timestep - creation_time > orphan_node_grace_period`. This
    guard exists to stop the CLOCK from aging a node PAST that grace. At the
    moment the guard decides, an unbound unprotected node is typically age 0
    (just minted or just landed), so an age term would say "not reap-eligible
    yet" and let the clock run -- defeating the guard on the very case it exists
    for. Intended consequence: an unprotected node that cannot bind HOLDS THE
    CLOCK until it is ruled on by class. The "bounded observation window" the S4
    plan gives for nodes that CAN bind is a WAIT-THEN-ESCALATE window: its expiry
    puts those nodes on the #909 list to Josh for a decision per class. It NEVER
    releases the clock hold, and nothing in this code has an expiry or an early
    release.

    A graph object without `_is_identity_protected` (an incomplete test double)
    is NOT exempted: its unbound nodes still count -- fail toward holding the
    clock. Pure query: no write, no lock (callers hold what they need).
    Signature is pinned by the daemon (arity) -- do not change it.
    """
    outgoing = getattr(graph, "_outgoing", {}) or {}
    incoming = getattr(graph, "_incoming", {}) or {}
    node_hyperedges = getattr(graph, "_node_hyperedges", {}) or {}
    is_protected = getattr(graph, "_is_identity_protected", None)
    return {
        nid for nid in node_ids
        if not outgoing.get(nid)
        and not incoming.get(nid)
        and not node_hyperedges.get(nid)
        # NO AGE TERM, ON PURPOSE (Exec P490; see the docstring): the sweep's
        # `age > orphan_node_grace_period` is the one term left out. This guard
        # stops the clock from aging a node past grace, and a node unbound at the
        # moment of the decision is typically age 0 -- an age term would let the
        # clock run on the very case the guard exists for. Do not add it back.
        # Also (Exec P493 R1): hyperedge-bound = BOUND; "bound" = NOT sweep-eligible
        # (the sweep's own predicate: an outgoing synapse, an incoming synapse OR
        # hyperedge membership is not reaped; protected nodes are not reaped), and NOT
        # "a complete turn" -- turn completeness is NOT a gate condition and no guard
        # here tests it. And (Exec P492): the "bounded observation window" for nodes
        # that CAN bind is WAIT-THEN-ESCALATE (expiry -> the #909 list to Josh); it
        # NEVER releases the clock hold, and nothing here has an expiry or early release.
        # Evaluated last: only structurally-unbound candidates reach the lookup.
        and not (is_protected is not None and is_protected(nid))
    }


def redact_node_id(nid: Any) -> str:
    """The ONE redaction rule for any unbound-node-id sample that reaches a log
    (#905 part C / Exec Packet 489 / #907): `<kind>:<first 12 hex of sha256(id)>`.

    Never the id and never any substring of it: tree ids are
    f"{target_id}::tree::{concept}" (ng_embed.py ~:1062), so the id embeds the
    user's own concept words (LAW 7 / privacy). `<kind>` comes ONLY from
    structural markers, never from free text: `tree` if the id contains
    '::tree::', `window` if it contains '::window::', `want` if it starts with
    'cc:want::' or 'want::', `forest` if it is exactly 'cc:conv::<40 hex>',
    otherwise `node`. Pure, deterministic (the same id always gives the same
    string) and cheap: only stdlib hashlib/re, already imported at module load.

    Callers that cannot import this (the D24 daemon imports it lazily) must print
    NO ids at all rather than fall back to the raw id.
    """
    s = nid if isinstance(nid, str) else str(nid)
    if "::tree::" in s:
        kind = "tree"
    elif "::window::" in s:
        kind = "window"
    elif s.startswith("cc:want::") or s.startswith("want::"):
        kind = "want"
    elif re.fullmatch(r"cc:conv::[0-9a-f]{40}", s):
        kind = "forest"
    else:
        kind = "node"
    # surrogatepass: a lone surrogate must not raise inside a logging call.
    return "%s:%s" % (kind, hashlib.sha256(s.encode("utf-8", "surrogatepass")).hexdigest()[:12])


def whole_graph_guard(graph: Any):
    """The ONE constructor of the per-slice consolidation guard (#905 part D;
    LAW 3 / LAW 4: no per-caller copy of the predicate).

    Returns a zero-argument callable that evaluates
    `_unbound_nodes(graph, set(graph.nodes))` -- the same sweep-eligible,
    whole-graph predicate as the merge's batch-end check and the daemon's
    #896 rule 1 -- under `graph._step_lock` (an RLock), taken ONLY for the read.
    The result is a set of currently blocking node ids (empty = clear). The ids
    are for COUNTING ONLY by the consumer, `cc_ng_organism._cc_callosum_consolidate`,
    which never logs them. The callable holds no lock between calls.
    """
    def _guard() -> Set[str]:
        with graph._step_lock:
            return _unbound_nodes(graph, set(graph.nodes))
    return _guard


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
