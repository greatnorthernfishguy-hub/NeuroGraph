<!--
# ---- Changelog ----
# [2026-09-22] Cursor — operator record for landed Anima/rpc intra-turn window chains
# What: new docs/ANIMA_WINDOW_CHAINS_OPERATOR.md (docs only)
# Why: c53b903 + tests/test_conversational_window_chains.py landed on main
#   (tip 71f5d537) but are unmentioned in docs/; draft #49
#   (docs/NG_EMBED_OPERATOR.md) explicitly deferred this to a later PR
# How: verified against neurograph_rpc.py (_run_conversational_dual_pass /
#   _bind_conversational_topology / _deposit_memory_node),
#   tests/test_conversational_window_chains.py, and commit c53b903
# -------------------
-->

# Anima / rpc intra-turn window chains

Intra-turn conversational window chains already landed on `main` in commit
[`c53b903`](https://github.com/greatnorthernfishguy-hub/NeuroGraph/commit/c53b903)
(2026-09-20). Current tip when this record was written: `71f5d537`. Missing
mentions in CHANGELOG / USER_GUIDE / PUNCHLIST / README are a docs gap, not a
missing feature. Do not reopen that commit as a code change.

Canonical wiring: `neurograph_rpc.py` —
`_run_conversational_dual_pass`, `_bind_conversational_topology`,
`_deposit_memory_node`. Tests: `tests/test_conversational_window_chains.py`.

Draft GitHub PR #49 (`docs/NG_EMBED_OPERATOR.md`) covers the **embed-side**
fail-closed / HF remote / overlapping CLS/SEP window stack and **deferred**
these rpc/Anima graph topology chains to a later docs PR. This file is that
record. Do not fold embed operator knobs into this file; do not rewrite #49.

Anima (Native Gateway) talks to NeuroGraph over HTTP on `127.0.0.1:8850`
(`neurograph_rpc.py`). Per turn Anima calls `POST /assemble` and
`POST /afterTurn`. Conversational dual-pass (and therefore window chains)
runs on the after-turn filing path
(`_file_conversational_experience` → `_conversational_dual_pass` →
`_run_conversational_dual_pass`).

---

## What this is

On a long conversational turn, after a successful dual-pass forest (and tree)
deposit, rpc also materializes **ordered window nodes in the SNN only**,
delay-chains them with the existing #257 polychrony sampler, and links each
window to the forest both ways. Window order is substrate structure. Mid-turn
fragments must not compete in cosine recall, so they stay out of the recall
store (`index_in_recall=False`).

Josh's rationale in the landing commit: *"We have polychrony, we use
polychrony."*

---

## Path on a conversational turn

```
afterTurn / experience drain
  → _file_conversational_experience(text, …)
    → _conversational_dual_pass(text, embedding)
      → _run_conversational_dual_pass(text, embedding)
          1. embedder.embed_windows(text).windows   # () on short turns
          2. dual_record_outcome(…, embedding=caller embedding, …)
          3. for each window: _deposit_memory_node(…, index_in_recall=False)
          4. _bind_conversational_topology(forest, result, embedding, window_ids)
```

Forest id is always `conv::` + SHA1 hex of the full turn text.

---

## Long turns — graph-only window nodes

When `embed_windows(text).windows` is non-empty:

| Piece | Exact shape / rule |
|---|---|
| Forest id | `conv::` + SHA1 hex of the turn text |
| Window id | `{forest_id}::window::{i}` (0-based) |
| Recall | `index_in_recall=False` — written into the SNN graph only; **not** inserted into `vector_db` |
| Forest vector | Caller-supplied `embedding` remains the forest vector (unchanged by windowing) |
| Window vector | Each window's own embedding from `embed_windows` |
| Window content | Per-window text from `embed_windows` |
| Metadata | `_window=True`, `_window_index=i`, `_forest_id=<forest_id>`, plus the conversational meta (`source`, `creation_mode`, `_forest_content`) |

Windows also receive the usual conversational node stamps from
`_deposit_memory_node` (threshold boost, novelty dampening, probation,
`poincare_dir`). They simply skip the recall insert.

---

## Delay-chain ordering (#257)

When two or more window nodes exist, `_bind_conversational_topology` chains
them in order: window `i` → window `i+1`, weight `0.2`, delay sampled with
the same #257 sampler used for the prev→current forest link:

```text
random.randint(2, max(2, ANIMA_CONV_SYNAPSE_DELAY_MAX))
```

Default max delay is **5** (`ANIMA_CONV_SYNAPSE_DELAY_MAX`, read into
`_CONV_SYNAPSE_DELAY_MAX`). That is the only env knob this record names for
window chaining — it is what the delay sampler reads. Same sampler as the
delayed prev-forest → current-forest synapse.

---

## Forest ↔ window synapses and binding hyperedge

Each window also gets bidirectional forest links (same weights as
forest↔tree):

- forest → window: weight `0.2`
- window → forest: weight `0.15`

If any trees or windows exist, they join the conversational binding
hyperedge with the forest (`creation_mode: conversational`, `syl: True`).

Synapse / hyperedge creation failures are non-fatal (same pattern as
forest↔tree — node may already be wired).

---

## Short turns keep prior topology

Short turns: `embed_windows(...).windows == ()`.

- No `::window::N` nodes
- No window delay-chain
- No forest↔window synapses
- Forest dual-pass deposit and prior forest↔tree / prev-forest topology
  behavior unchanged (this layer is a no-op)

Verified by `test_short_turn_creates_no_window_nodes`.

---

## Caller-supplied embedding stays the forest vector

`_run_conversational_dual_pass` passes the caller's `embedding` straight into
`dual_record_outcome`. Window embeds are used only for the graph-only window
nodes. The pooled / caller forest vector is not replaced by window pooling
on this path. The long-turn test asserts the recall store entry for the
forest matches the (normalized) caller embedding, not the window pooled
vector.

---

## Window embed failure is fail-soft

`embed_windows` is wrapped alone inside `_run_conversational_dual_pass`:

- On failure: debug log
  `embed_windows failed (non-fatal, no window nodes): …`
- Continues with `windows = ()` — no window nodes, no window synapses
- Forest dual-pass deposit and the rest of topology binding still run

A full conversational dual-pass exception remains non-fatal at the core
(`_run_conversational_dual_pass` returns `False`); that path is the existing
#296a / #297 retry contract, not window-specific.

---

## How this differs from ng_embed overlapping CLS/SEP windows

Two different “window” concepts. Do not conflate them.

| | **Embed-side** (`ng_embed`) | **This file — rpc / Anima topology** |
|---|---|---|
| Where | `ng_embed.py` `embed_windows` / `_window_token_ids` | `neurograph_rpc.py` after dual-pass |
| Purpose | Tokenize long text into overlapping interior slices, wrap `[CLS]+slice+[SEP]`, pool into one vector | Deposit ordered mid-turn fragments as SNN nodes and delay-chain them |
| Short path | ≤512 tokens → one embed, `windows=()` | Same `windows=()` → no topology nodes |
| Long path | Overlap 64 / interior 510; pooled vector for `embed()` | Each returned window becomes `{forest}::window::N` in the graph |
| Recall | N/A (vectors only) | Windows: `index_in_recall=False`; forest stays in recall |
| Operator doc | Draft #49 — `docs/NG_EMBED_OPERATOR.md` | This file |

`NGEmbed.dual_record_outcome` may echo optional `windows=` on its result dict
and does **not** deposit them. Rpc is what turns those window structs into
graph topology.

For fail-closed behavior, `NG_EMBED_REMOTE=hf`, token hygiene, CLS/SEP wrap,
keepalive, and dual-pass atomicity, read draft #49 /
`docs/NG_EMBED_OPERATOR.md`. Keep this file about rpc/Anima graph topology
only.

---

## Out of scope

- Rewriting CLAUDE.md, ARCHITECTURE.md, PUNCHLIST.md, README, USER_GUIDE,
  CHANGELOG, or sibling operator docs
- Editing open PR areas (#31–#35, #37, #40, drafts #47–#50) or rewriting #49
- Runtime / test / hook / script changes
- Fixing unrelated leftovers
