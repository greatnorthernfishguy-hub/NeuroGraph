```
---- Changelog ----
[2026-09-30] Claude Code (claude-sonnet-5-5, Z12 BUILD worker seat, dispatch #11805, DELTA BUILD) - build-tool-007b.
             SECTION 1 (item 8, the P437 finding) is written and pushed BEFORE any code, as ordered. The build sections
             are appended below as they are completed.
-------------------
```

# build-tool-007b - DELTA BUILD: streamed content-subset vectors read + (item 8) Graph-free Phase-1 analysis

Lane `z12-s3-restore-bundle-20260929` - Z12 - dispatch #11805 - tool branch `cc-laptop-want-repair-tool-20260930`; base = `09a032c3426baf8307cb968496c65f779894304b` (build-tool-007). Pin worktree `ae798b94cb14740d200fc3f4fd8d36eef8b86c6a` (read-only throughout), pin/stack head `c7921b8436fb174c3f70fcf02827f16bb16deff0`.
Related: [[NeuroGraph]] - [[The Laws]]

## 1. ITEM 8 (Exec P437) - does Phase 1 need a LIVE `Graph`? FINDING, written before any code

**Answer: the classify / id-mapping / report side does NOT need a live `Graph`. Every value `analyze()` takes from the Graph is a DIRECT stored field of `main.msgpack` or a pure function of stored fields. One Phase-1 step DOES still need the canonical `Graph.restore`: V11, the verifier's restore of the OUTPUT, which the plan puts in Phase 1.**

### 1.1 Every value `analyze()` (tool `:1595-1640`) and its callees take FROM the Graph

The canonical restore is `Graph.restore` (pin `neuro_foundation.py:5039`; it already streams the outer map, slicing `synapses` as raw bytes and skipping `he_prediction_window_fired`) then `Graph._deserialize` (pin `:5368`).

| Value the tool takes | Tool file:line | Where it comes from in `Graph._deserialize` (pin) | Stored or derived? | Streamable? |
|---|---|---|---|---|
| `nodes_meta = {nid: n.metadata}` | `:1601` | `Node(metadata=nd.get("metadata", {}))` `:5403`, `self.nodes[nid] = node` `:5412` (key = the `nodes` map key) | **stored** (`nodes[nid].metadata`, default `{}` applied at `:5403`) | yes: decode one node entry at a time, keep `.get("metadata", {})` |
| `existing_ids = set(g.nodes)` | `:1602` (used by `apply_collision_rule` `:1111`) | the `nodes` map keys `:5387-5412` | **stored** (map keys) | yes |
| `counts` (nodes / wants / protected) | `:1622-1625` | `len(g.nodes)`; `kind == "want"` and `is_protected(md)` over `nodes_meta` | **stored** (function of metadata) | yes |
| scope, `derive_scope` | `:999`, `:1607-1609` | function of `nodes_meta` only | **derived from stored metadata** | yes |
| `incident_figures(g, ids)` = (`len(_outgoing[i])`, `len(_incoming[i])`, `len(_node_hyperedges[i])`) | `:1557-1560`, `:1620` | `_outgoing`/`_incoming` rebuilt from the synapse store `:5454-5457` (`pre_node_id`, `post_node_id` of each synapse, keyed by synapse id, `bulk_load_msgpack` `:5451`); `_node_hyperedges` rebuilt from `hyperedges[*].member_nodes` (`set(hd["member_nodes"])` `:5463`) at `:5487` - NOT from `archived_hyperedges` (`:5495-5517` never touches `_node_hyperedges`) | **derived indexes** (plan-004 section 3.2, line 189: "Derived, never written") but each is a count of **stored** fields: synapse entries with `pre_node_id == i`, with `post_node_id == i`, and non-archived hyperedges listing `i` in `member_nodes` | yes: one streamed pass over `synapses` and `hyperedges` counting for a small id set |
| `render_len_before = len(org.render_wants(g).encode())` | `:1621` (`render_wants` pin `cc_ng_organism.py:2285`) | reads only `graph.nodes` items: `node.metadata` and `node.creation_time` (`:2303-2312`); `creation_time=nd.get("creation_time", 0)` `:5410` | **stored** | yes: a graph-like object holding only the want nodes (metadata + stored `creation_time`), in `nodes` map order (the sort is stable, so equal timestamps keep the same order) |
| `synapse_stats(raw_main, ...)` | `:1563`, `:1627` | already a streaming pass over the raw bytes, never the Graph | - | already Graph-free |

**Derived state the tool does NOT need:** restored defaults for the ~15 node fields other than `metadata` / `creation_time` / key (`voltage`, `threshold`, `spike_history` and the rest), the spike deques, the delay buffer, prediction validation (`:5368+`), the native synapse store itself, the `config` merge. Nothing in `analyze()` reads any of them. `is_protected` (tool `:565`) is a mirror of a metadata predicate.

**The one place a "restore semantic" could differ from a raw stream** is the native synapse store's (`ng_tract.SynapseStore`, a Rust extension not in the pin tree) treatment of the synapse map key versus the inner fields. I cannot read that source, so the claim rests on a TEST: the equivalence test builds the OLD result from the canonical `Graph.restore` (real native store) INSIDE the test and asserts the streamed figures equal it on a synthetic graph with several synapses, including a synapse whose endpoints are not nodes (the `setdefault` case `:5456-5457`) and archived/non-archived hyperedges. If the real store normalised something the synthetic world does not exercise, only the real-file run would show it (that is on the "not verified" list).

### 1.2 Which step still needs the canonical restore, and whose phase it is

`build_outputs` (tool `:2545-2546`) runs `pinned.nf.Graph().restore(out_main)` on the OUTPUT - "the canonical restore IS the verifier (V11)" - and V11 (verifier `:1926-1950`) compares per-id `_outgoing/_incoming/_node_hyperedges` figures, dangling ids and the `render_wants` byte length before/after. **By the plan's own text this is a Phase 1 step:** plan-004 (worktree `z12-want-text-repair-20260930` at `d18323e`, `returns/plan-004.md` line 286) section 6.6 "Phase 1 (COPY ...): ... `systemd-run ... MemoryMax=3G` for classify/rewrite **and 6G for V11 (measured restore ~= 3.6 GiB)** ... rewrite to a tmp file; V1-V19", and plan section 7: "V11 canonical restore (per-id `_outgoing/_incoming/_node_hyperedges` figures equal; no dangling id; the rendered wants block's byte length before/after reported)". So V11 is Phase 1 (the rewrite step of TURN B), it is deliberately canonical (plan 7: the restore is the verifier), and I do **not** change it: replacing V11's canonical restore with a stream would remove the very check it exists for. `Graph().restore` is therefore removed from **`analyze()`** (both Phase-1 steps call it; the Phase-2 `--apply` path calls it too on the live bytes), and stays in `build_outputs` (V11), where the `Graph` is the only thing resident besides the small content subset.

### 1.3 CONSEQUENCE, stated plainly (nothing lowered)

- With `analyze()` Graph-free and the vectors streamed, the **classify step** (`--step classify`) no longer constructs a `Graph` or loads the vectors: its memory is the copy/hash streaming, the node-metadata dicts, a streamed content subset, and the existing `synapse_stats` pass (which reads `main.msgpack` into `bytes`, ~230 MB, plus the Unpacker's copy). **[unmeasured - a real peak is a TURN B / probe fact, not claimed here.]**
- The **rewrite step** is still heavy: it holds the raw main bytes plus the rewritten output (`build_outputs` `:2538-2542`, verifier lockstep `:1763`) **and then V11's canonical `Graph.restore` of the output (~3.6 GiB, plan-004's cited figure, not re-measured)**. The heavy pass is therefore reduced to **V11 alone** (the input Graph and the 1 GB vectors load are gone from it), but it is **not** moved out of Phase 1.
- So the dry run does NOT become light end to end. The classify step may become light; the rewrite step keeps one canonical restore. **No pre-flight is lowered:** the 8 GiB heavy floor stands until a MEASURED peak re-derives it (and this turn measures nothing real); the 6 GB cap and `MemorySwapMax=0` are unchanged. If the Chief wants the dry run to be light, the lever left is to split TURN B into the light classify step and a separately-scheduled V11 step; that is a scheduling decision, not a code change I am making.

### 1.4 What this changes for the probes (for the re-dispatch, `probe.py` NOT edited here)

- **P1** (vectors keys + field byte sizes) and **P2** (streamed content subset): still the right probes; they mirror the new vectors reader. Keep.
- **P3** (graph-metadata stream of `main.msgpack`): now the **relevant** probe for the classify step. It should additionally do what the new pass does: one decode per node entry retaining `metadata` (plus `creation_time` for want nodes), then a second streamed pass over `synapses` / `hyperedges` counting incident figures for a small id set. The current P3 already decodes each entry transiently and retains node metadata; it does not do the incident-figure pass, and it does not read the `synapses` value as the tool's `synapse_stats` does (whole `bytes`).
- **P4** (canonical `Graph().restore(main)` under 1G): **no longer a Phase-1 classify probe**, but still the right measurement of **V11** (the output restore is the same call on a same-sized file). Keep it, relabelled "V11 restore", expected OOM under 1G as the finding.
- A probe of the tool's own `synapse_stats` whole-bytes read (~230 MB + copy) would be worth adding to P3 because it is now the largest allocation left in the classify step.

