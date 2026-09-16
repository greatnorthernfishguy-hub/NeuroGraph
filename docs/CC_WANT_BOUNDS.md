<!--
# ---- Changelog ----
# [2026-09-16] Cursor Grok 4.6 — operator record for landed CC want bounds
# What: new docs/CC_WANT_BOUNDS.md (docs only)
# Why: WANT_MAX_CHARS, WANT_RENDER_LIMIT, surface_wants, render_wants, and the
#   2026-09-16 context bomb are already on main (d75efeb) but unmentioned in
#   docs/, CHANGELOG.md, PUNCHLIST.md, README, USER_GUIDE, ARCHITECTURE, and
#   CLAUDE.md, so agents keep treating the bounds as missing.
# How: verified against main cc_ng_organism.py, cc_ng_host.py,
#   neurograph_rpc.py, tests/test_cc_want_bounds.py, and commit d75efeb
# -------------------
-->

# CC want bounds

Want extraction and `## What I Want` rendering for hosted Claude Code already
have hard caps. They landed on `main` in commit
[`d75efeb`](https://github.com/greatnorthernfishguy-hub/NeuroGraph/commit/d75efeb)
(2026-09-16). Do not treat a missing mention in `ARCHITECTURE.md`,
`PUNCHLIST.md`, or `CHANGELOG.md` as a missing feature. Do not reopen that
commit as a code change.

This file is the operator record for that contract. Hosted Pith
(`docs/PITH_HOST_CONTRACT.md`, merged GitHub #42) is a different boundary.
Draft GitHub PR #47 records Tonic stage timing; do not duplicate it.

## Why it exists

On 2026-09-16, `## What I Want` was **2,267,508 of 2,269,232 chars**
(~567k tokens) injected into CC on **every** `UserPromptSubmit`,
query-independent.

182 want-nodes, all `cc_authored` (`surface_wants` path;
`generate_emergent_want` is not implicated). 118 over 600 chars, largest
136,449. Only about five were genuine wants.

Cause: `_WANT_RE = \[WANT\](.*?)\[/WANT\]` with `DOTALL` is non-greedy but
**unbounded**. Prose that merely *mentions* the marker — documenting the
syntax, quoting a transcript, a code span like `` `[WANT]` `` — matched as
an opening tag, and the span ran to the next `[/WANT]` tens of thousands of
characters later. The swallowed text became one "want". 83 of the 118
oversized nodes begin with the backtick that closed such a code span.
Want-nodes are prune-protected, so nothing culled them.

## Live contract (CC bounded path)

Constants in `cc_ng_organism.py`:

| Name | Value | Role |
|---|---|---|
| `WANT_MAX_CHARS` | **600** | Max captured `[WANT]…[/WANT]` span; also the per-line clamp in `render_wants` |
| `WANT_RENDER_LIMIT` | **40** | Max open wants rendered into `## What I Want` |

A want is an utterance, not a document — a sentence or three, no more.

### `surface_wants(graph, vector_db, provenance="cc_authored")`

Called from `cc_ng_host._autosave_loop` (not from the per-prompt deposit).

- Regex is `_WANT_RE`: `\[WANT\](.{1,600}?)\[/WANT\]` with `DOTALL`.
- Skips a `[WANT]` immediately preceded by a backtick (documentation of the
  marker, not a want).
- Rejects any match whose inner text still contains `[WANT]` or `[/WANT]`
  (mismatched / nested markers).
- Stamps first-class want nodes as `cc:want::<sha1[:16]>` with
  `provenance` default `cc_authored`, `kind=want`, `want_state=open`.
- Idempotent: existing id is skipped.

### `render_wants(graph, provenance=("cc_authored", "cc_emergent"))`

Called from `cc_ng_host._handle_user_prompt_submit` so the block is injected
every turn, live, not a snapshot.

- Newest 40 open wants (by `creation_time`).
- Each body clamped to 600 chars.
- If more than 40 exist: one elision line
  `- ... and N older open wants`.
- Returns `""` if none.

This render cap is independently sufficient. A poisoned graph cannot dump
megabytes into `## What I Want` on the CC host path, even if oversized
want-nodes already exist.

## Leftovers (report only — LAW 4 / Josh-gated)

Do not "fix" these in this PR or as a drive-by. Canonical Syl files and
identity-adjacent want paths need Josh.

### 1. Syl path in `neurograph_rpc.py` is still unbounded

`_surface_wants()` at line 4902 still uses `re.finditer(r'\[WANT\](.*?)\[/WANT\]',
content, re.DOTALL)` and stamps `want::` ids with `provenance=syl_authored`.
That is the identical pre-`d75efeb` pattern. Canonical file; needs Josh.

`_render_self_and_wants()` is uncapped: it joins **all** `syl_authored`
want texts into `## What I Want` with no `WANT_RENDER_LIMIT` and no
per-body clamp. Anima `handle_assemble` prepends that block every turn.

`_strip_structural_markers()` still strips with unbounded
`re.sub(r'\[WANT\].*?\[/WANT\]', '', text, flags=re.DOTALL)`.

### 2. `surface_wants_for_graph()` is still unbounded — and it is live

`cc_ng_organism.surface_wants_for_graph()` still uses the same unbounded
`(.*?)` `DOTALL` regex. It stamps `want::` ids (not `cc:want::`) with
`provenance=cc_authored`.

`cc_ng_host._deposit()` calls it on **every** `UserPromptSubmit` (background
thread after recall). That is the live per-prompt Cricket bucket, not a
dead helper.

The `d75efeb` changelog line that "only `neurograph_rpc.py:4902` remains
unbounded" is **wrong**. Two unbounded extractors remain: Syl's
`_surface_wants` and CC's `surface_wants_for_graph`. Bounded
`surface_wants` + bounded `render_wants` closed the **injection** bomb on
the CC host path. They did not close CC want *creation* on `_deposit`.

`render_wants` still lists any open `cc_authored` want-node, including
`want::` nodes this leftover can still create. The 40×600 clamp is what
stops those from becoming another megabyte injection.

## Tests

`tests/test_cc_want_bounds.py` covers the bounded `surface_wants` +
`render_wants` path only: real want extraction, backtick skip, span-over-cap
reject, span-at-cap accept, nested-marker reject, two real wants, render
count/length cap + elision, render under-limit with no elision.

It does **not** cover `surface_wants_for_graph`, `_surface_wants`,
`_render_self_and_wants`, or `_strip_structural_markers`.

## What this is not

- Not a code fix. Bounds are already on `main`.
- Not a rewrite of `CLAUDE.md`, `ARCHITECTURE.md`, `README`, `ECOSYSTEM`,
  `PUNCHLIST.md`, `docs/PITH_HOST_CONTRACT.md`, or
  `docs/CC_TONIC_STAGE_TIMING.md`.
- Not GitHub PRs #31–#35, #37, #40, or #47.
- Anima HTTP (`neurograph_rpc.py` on `127.0.0.1:8850`) is the current
  gateway path for Syl. Anima still depends on OpenClaw gateway
  infrastructure.

Live sources: `cc_ng_organism.py` (`WANT_MAX_CHARS`, `WANT_RENDER_LIMIT`,
`surface_wants`, `render_wants`, leftover `surface_wants_for_graph`),
`cc_ng_host.py` (`_handle_user_prompt_submit`, `_deposit`, `_autosave_loop`),
`neurograph_rpc.py` (leftover `_surface_wants`, `_render_self_and_wants`,
`_strip_structural_markers`). Tests: `tests/test_cc_want_bounds.py`.
