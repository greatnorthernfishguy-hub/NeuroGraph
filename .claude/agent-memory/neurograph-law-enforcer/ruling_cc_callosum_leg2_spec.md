---
name: ruling-cc-callosum-leg2-spec
description: #70 Leg 2 spec review (2026-07-28) — LAW 1/5 clean and Finding-7 resolved, but §2's two "BTF can't carry this" claims are BOTH false; hyperedges ARE transportable and re-running the binder on receive is a 3-way defect
metadata:
  type: project
---

> ⚠️ **SUPERSEDED IN PART — READ [`docs/CC-CALLOSUM-TRUTH.md`](/home/josh/docs/CC-CALLOSUM-TRUTH.md) FIRST.**
> The consolidated, measurement-verified state of the callosum, the wholeness ring,
> hyperedge binding and orphan collection lives there (2026-07-31). The wholeness ring
> **already exists** (Leg 2); the real open defect is the merge-journal poison-pill.
> Do not re-derive any of it from this file.


Constitutional review of `docs/superpowers/plans/2026-07-28-cc-corpus-callosum-leg2-spec.md`, pre-build. Verdict: **VIOLATIONS FOUND (design-level), LAW 1/2/3/5 structurally clean.**

**The headline — spec §2 is empirically wrong on both prongs, and it set its own falsification test ("if either depends on state absent from the frame, the design fails and must be re-reviewed rather than patched"). Both fail it.**

**(a) "The binding hyperedge is not representable in a topology frame" — FALSE.** Verified live: `ng_tract.write_topology(ts, step, nodes, fired_hyperedges=[{hyperedge_id,label,member_node_ids,output_target_ids,activation_count}])` accepts them; `PyTopologyEntry.fired_hyperedges()` returns them; `member_node_ids` round-trips byte-exact (160→208 bytes with one hyperedge). Ship the hyperedge in the frame. Delete the re-derivation mechanism.

**(b) "delay is a pure function of geometry, so the receiver computes the SAME delay" — FALSE.** The GSG formula (`neuro_foundation.py:3543-3572`) reads three inputs: `poincare_dir` (embedding-derived ✓ via `_cc_embed_to_poincare_dir`, cc_ng_organism.py:1123), `manifold_type` (assigned from **pred_error_ema percentile + neighbor co-confirmation over the whole receiving graph**, `:1193-1221`), `diffpc_layer` (assigned by **degree percentile p33/p67**, `:1191`, and it scales the Poincaré vectors via `_GSG_LAYER_NORMS_NF`). The last two are global-graph-state, not embedding. Cross-manifold mismatch skips the formula entirely → `random.randint` fallback. **The conclusion survives on better grounds:** the forest↔tree synapses Leg 2 ships are created by `_cc_bind_conversational_topology` with *no delay arg* → `create_synapse` default `delay=1`. They were never geometry-informed; the GSG formula lives only in `_sprout_synapses` (spontaneous plasticity), not the payload. Nothing to lose.

**The load-bearing defect — re-running `_cc_bind_conversational_topology()` on the receive side (spec §3) has three failure modes**, because it does far more than build a hyperedge:
1. **Duplicate synapses.** It calls `create_synapse(forest,tid,0.2)` / `(tid,forest,0.15)` unconditionally, and `create_synapse` never dedupes (see [[ruling-novelty-gated-binding-weight]]). §3 merges synapses *then* re-runs the binder → every forest↔tree edge doubled → degree inflation → which feeds `diffpc_layer` degree percentiles → corrupts the receiver's own geometry.
2. **Fabricated cross-machine temporal edge.** The binder wires `state["last_forest_id"] → forest_id` at `randint(2, _CC_CONV_SYNAPSE_DELAY_MAX)`. On the laptop `last_forest_id` is the laptop's last LOCAL turn — forging "these two turns were adjacent" between an unrelated laptop conversation and a VPS one. Invented experience; LAW-7 adjacent.
3. **Recall contamination.** It tail-calls `cc_anticipate(graph, [forest_id]+tree_ids, state)`, priming `state["primed_nodes"]` that the daemons' `_recall()` reads. A 25-item absorb batch primes the laptop's next recall with VPS nodes.

**Silent metadata loss → probation bypass (HIGH).** `PyFiredNode` carries exactly `node_id`, `label`, `embedding_dim`, `embedding_as_numpy`, `outgoing_synapses` — **no metadata**. So the frame drops `probation_remaining`/`probation_total`/`novelty_dampening`/`threshold`/`intrinsic_excitability`/`poincare_dir`/provenance. Raw `create_node` on absorb ⇒ VPS trees land at FULL excitability, and `cc_update_probation` only touches nodes that already have `probation_remaining`, so they skip probation permanently. Remedy is LAW-3 reuse: absorb through `_cc_deposit_memory_node` (cc_ng_organism.py:1133), which re-stamps probation + poincare_dir + vdb index in one call.

**`cc_gsg_backfill` stamps the wrong value — do not lean on it.** `cc_ng_organism.py:2396` sets `poincare_dir = emb.tolist()` (RAW embedding); `_cc_deposit_memory_node:1151` sets `_cc_embed_to_poincare_dir(embedding).tolist()` (unit-normalized). GSG math assumes unit-normalized (`neuro_foundation.py:214`). Pre-existing defect, fix at source (LAW 4).

**What is genuinely clean and should not be re-litigated:** uuid5 id-aliasing is *necessary* and validated (`write_topology` hard-rejects non-UUID `node_id`: "invalid character"); Finding 7 prong (b) is now satisfied with canonical BTF and no bespoke format — this **retires the standing HIGH** in [[ruling-cc-river-merge-intra-mind]]; the `get_cc_commons` vs `get_commons()` co-residency boundary is real and correctly treated as non-negotiable (honors [[ruling-coresident-cc-export-scoping]]), and asserting `get_commons()` is never called is the right test; one-directional + receiver-plasticity-ON + batch-25/250-idle honors the FatherGraph merge contract and my standing condition; absorb kept out of the 60s pulse is correct.

**Both spec blockers CONFIRMED real.** B1: `_autosave_loop` (cc_ng_host.py:848-880) never calls `commons.persist()`; the "matching Syl's own Commons today" docstring (cc_ng_organism.py:495) IS stale — Syl has `_COMMONS_CHECKPOINT_PATH` (neurograph_rpc.py:732), restore `:1957`, persist `:3486`/`:5621`. B2: `modules/immunis/commons.py` is 453 lines vs canonical 515, missing the whole #80 wire carve-out (`_WIRE_KEEP_PER_DIR`, `_evict_old_wire`, the `wire:` deposit branch) — live LAW-2 drift; re-vendor, never hand-edit.

**§7's flat "never bidirectional" prohibition is WITHDRAWN 2026-07-28** — replaced by the two-clause additive-only test in [[ruling-cc-river-merge-intra-mind]]. Leg 2 stays one-directional as specced, but §7 should carry the test, not the prohibition. Clause 2 (echo/provenance) imposes a NEW requirement on THIS build: absorb must re-stamp origin provenance, because `PyFiredNode` carries no metadata and `cc:conv::`+sha1 ids are host-neutral.

**How to apply:** at build-diff review, check (1) hyperedge rides in the frame, (2) the binder is NOT re-run wholesale on receive — only an extracted hyperedge-only helper, (3) absorb goes through `_cc_deposit_memory_node`, (4) no duplicate forest↔tree edges. Related: [[finding-cc-authored-export-protection-collision]] (provenance loss interacts with the export whitelist), [[ruling-conduit-snapshot-vs-destructive-drain]] (Leg 1 sibling).
