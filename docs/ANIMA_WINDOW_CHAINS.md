<!--
# ---- Changelog ----
# [2026-09-22] Cursor — operator record for landed Anima intra-turn window chains
# What: new docs/ANIMA_WINDOW_CHAINS.md (docs only)
# Why: c53b903 + tests/test_conversational_window_chains.py landed on main
#   (tip 71f5d537; embed parent 1d6c659) but are unmentioned in docs/; draft
#   #49 (NG_EMBED_OPERATOR.md) explicitly deferred this to a later PR
# How: verified against neurograph_rpc.py (_run_conversational_dual_pass /
#   _bind_conversational_topology), tests/test_conversational_window_chains.py,
#   and commit c53b903
# -------------------
-->

# Anima intra-turn window chains

Intra-turn window chains already landed on `main` in commit
[`c53b903`](https://github.com/greatnorthernfishguy-hub/NeuroGraph/commit/c53b903)
(2026-09-20/21 changelog date on the RPC header). Current tip:
`71f5d537`. Missing mentions in CHANGELOG / USER_GUIDE / PUNCHLIST / README
are a docs gap, not a missing feature. Do not reopen that commit as a code
change.

Canonical wiring: `neurograph_rpc.py`
(`_run_conversational_dual_pass`, `_bind_conversational_topology`). Tests:
`tests/test_conversational_window_chains.py`.

Draft GitHub PR #49 (`docs/NG_EMBED_OPERATOR.md`) covers the ng_embed
fail-closed / HF remote stack and **deferred** these chains here. Drafts
#47 / #48 / #50 do not cover this. Do not fold embed operator knobs into
this file.

Anima talks to NeuroGraph over HTTP (`neurograph_rpc.py` on
`127.0.0.1:8850`). Anima still depends on OpenClaw gateway infrastructure;
this record is about conversational deposit topology only.

---

## What lands on a long turn

After a successful conversational `dual_record_outcome`, long turns also
call `NGEmbed.embed_windows(text)` and deposit each window as an SNN node:

| Piece | Exact shape / rule |
|---|---|
| Forest id | `conv::` + SHA1 hex of the turn text |
| Window id | `{forest_id}::window::{i}` (0-based) |
| Recall | `index_in_recall=False` — graph-only; **not** inserted into `vector_db` |
| Forest vector | Caller-supplied embedding stays the forest vector (unchanged) |
| Metadata | `_window=True`, `_window_index=i`, `_forest_id=<forest_id>` |

Window order is substrate structure (polychrony). Mid-turn fragments must
not compete in cosine recall, so they stay out of the recall store.

---

## Delay-chain (#257)

When two or more window nodes exist, `_bind_conversational_topology` chains
them in order: window `i` → window `i+1`, weight `0.2`, delay sampled with
the same #257 sampler used elsewhere in conversational topology:

`random.randint(2, max(2, ANIMA_CONV_SYNAPSE_DELAY_MAX))`

Default max delay is **5** (`ANIMA_CONV_SYNAPSE_DELAY_MAX`). That is the
only env knob this record names for window chaining — it is what the delay
sampler reads.

---

## Forest ↔ window synapses; short turns

Each window also gets bidirectional forest links (same weights as
forest↔tree): forest→window `0.2`, window→forest `0.15`. Windows join the
conversational binding hyperedge with the forest (and any trees).

Short turns: `embed_windows(...).windows == ()` — no window nodes, no
window synapses, forest dual-pass topology unchanged (no-op for this
layer).

---

## Fail-soft

`embed_windows` is wrapped alone. On failure the RPC logs at debug
(`embed_windows failed (non-fatal, no window nodes)`) and continues with
an empty window list. Forest dual-pass deposit and the rest of topology
binding still run. Synapse / hyperedge creation failures for windows are
also non-fatal (same pattern as forest↔tree).

A full conversational dual-pass exception remains non-fatal at the core
(`_run_conversational_dual_pass` returns `False`); that path is the
existing #296a / #297 retry contract, not window-specific.

---

## Relation to dual-pass conversational deposit

Window chains are an **extra topology pass** after dual-pass forest (and
tree) deposit. They do not replace dual-pass, do not change the forest
embedding, and do not index windows into recall. For embed primitives,
fail-closed / remote / pooling behavior, see draft #49
(`docs/NG_EMBED_OPERATOR.md`) — not this file.
