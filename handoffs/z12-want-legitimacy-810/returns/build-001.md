# #810 build-001 — WANT parser: structural LEGITIMACY, no length limit

Lane `want-parser-legitimacy-810` · Zone manager Z12 · dispatch #10715 · worker turn 1 of 2
Branch `cc-laptop-want-legitimacy-810-20260930` · base `e4ebf982b1989fd9066d610b94853bc68bf70d37`
Related: [[NeuroGraph]] · [[The Laws]] (LAW 3/4/7) · [[NeuroGraph Is a Mind, Not a Database]]

> **Status: RETURNED (commit 3 of 3).** Sections 0-7 are the PLAN exactly as committed first
> (`2ab4a85`, before any code). Section A onward is what was built, the evidence, the deviations
> from the plan, the read-only render-retirement inventory (section 8) and the flags (section 9).
> Code + tests: `c9fe56d`. Nothing merged, wired or restarted; not self-accepted.
> **Turn 2 (`build-002.md`) superseded parts of this return:** le-014's C1-C4 corrections, the #815 delta and several
> statements marked `[corrected turn 2]` below. Read `build-002.md` for the current state of the parser.

## 0. Scope, and the two amendments that changed it

| Source | Ruling | Effect on this turn |
|---|---|---|
| Exec P406 (Josh) | "WANTs just need to have the WANT brackets on either side. A parser just needs to make certain that it's legit, and not just us talking about WANTs." | The rule: a WANT is the text between a real `[WANT]` and its paired `[/WANT]`. **No length limit.** |
| Exec P407 | render half HELD pending #811 | **Superseded** by P408. |
| Exec P408 (Josh resolved #811) | The standing `## What I Want` block is to be **RETIRED** (wants surface associatively, in full). It is *not* a size bound. | **`render_wants` is NOT touched this turn** (no clamp change, no 40-cap change). A second turn carries the retirement. This turn adds a read-only inventory (section 8) so that turn can be sized. |
| Sequencing ruling | #810 must land (fixed, reviewed/paired, and the code the laptop daemon runs) BEFORE the #801 repair is applied live; both before S4. | Stated in section 7. |

In scope this turn (parser half only): the legitimacy test as ONE pure function, `_WANT_RE` no
longer length-limited, a flood-safe INFO skip log, tests, a golden test against base, the
function's use as the #801 separation rule, and the render-retirement inventory.

Out of scope / not touched: `render_wants`, `WANT_RENDER_LIMIT`, `surface_wants_for_graph`
(the host twin, parked in #755), `cc_ng_host.py`, `neurograph_rpc.py`, every protected and
vendored file, `~/.claude/plugins/neurograph`, `~/NeuroGraph/data/checkpoints`, the live tract,
any primary checkout. Nothing merged, wired or restarted.

**`WANT_MAX_CHARS` stays DEFINED** (value 600): `render_wants` still reads it at
`cc_ng_organism.py:1604` and that function is not touched this turn. It leaves the *pattern*
(and every parser use). Turn 2 deletes it together with the renderer it serves.

## 1. Callers and consumers (confirmed, file:line at base `e4ebf982`)

NeuroGraph repo (this worktree):

| Symbol | Where | Notes |
|---|---|---|
| `WANT_MAX_CHARS` def | `cc_ng_organism.py:1512` | used at `:1514` (pattern) and `:1604` (render clamp) only |
| `_WANT_RE` def | `cc_ng_organism.py:1514` | used at `:1545` only (`surface_wants`). **Replaced** by a marker tokenizer; nothing else imports it. |
| `surface_wants` def | `cc_ng_organism.py:1517-1572` | the ONE parser consumer this lane changes |
| `surface_wants` call | `cc_ng_host.py:1519,1531` | VPS/host autosave pulse (under `_STATE.cc_ng.graph`), fail-soft `try`, `logger.debug` on exception |
| `render_wants` def / calls | `cc_ng_organism.py:1575-1610`; `cc_ng_host.py:1141-1149` | NOT touched (section 8) |
| `surface_wants_for_graph` def | `cc_ng_organism.py:1128-1195`; call `cc_ng_host.py:698-700` | **the host twin, PARKED (#755).** Its own unbounded `re.finditer(r'\[WANT\](.*?)\[/WANT\]')` and `want::` (no `cc:`) ids are NOT touched. Flagged, see section 9. |
| tests | `tests/test_cc_want_bounds.py` (encodes the OLD extraction rule AND the render clamp), `tests/test_cc_host_stop_door.py`, `tests/test_cc_deposit_step.py:239,290,400,559` (patch the functions), `tests/test_self_render.py`, `tests/test_conversational_recall.py:355-406` (Syl's `_surface_wants`, not this function) | only `test_cc_want_bounds.py` changes |
| Syl twin | `neurograph_rpc.py:4902/4914` (`_surface_wants`, identical unbounded regex) | canonical protected-adjacent file; NOT touched, LAW 4 propagation question stays open |

Laptop daemon (docs repo, read-only): the brief cites `scripts/cc-ng-daemon.py:1929`. That file is
in the **docs** repo (`~/docs/scripts/cc-ng-daemon.py`), not in NeuroGraph. On `origin/main`
of the docs repo `surface_wants(STATE.ng.graph, STATE.ng.vector_db)` is at `:1929` (confirmed by
`git show origin/main:scripts/cc-ng-daemon.py`); on the docs branch
`cc-laptop-daemon-recall-756-20260930` HEAD `155343e4` the same call is at `:2116`. It imports
`from cc_ng_organism import surface_wants` from the NeuroGraph checkout the unit runs
(`cc-ng-daemon.service`: `ExecStart=%h/NeuroGraph/.venv/bin/python3 %h/docs/scripts/cc-ng-service.py run`),
so this branch changes nothing live until it is merged and the daemon restarted (P329: merge = deploy).

## 2. The one function

```python
WANT_OPEN, WANT_CLOSE = "[WANT]", "[/WANT]"

@dataclass(frozen=True)
class WantSpan:        # one legitimate want
    text: str          # inner.strip() -- exactly what surface_wants stores and hashes
    want_id: str       # want_id_for_text(text)
    open_start: int    # char offset of the real [WANT] in `content`
    close_end: int     # char offset just past the paired [/WANT]

@dataclass(frozen=True)
class SkippedMarker:   # one marker that is discussion, not a want
    marker: str        # "[WANT]" or "[/WANT]"
    start: int         # char offset in `content`
    reason: str        # one of WANT_SKIP_REASONS

@dataclass(frozen=True)
class WantParse:
    wants: Tuple[WantSpan, ...]
    skipped: Tuple[SkippedMarker, ...]

def parse_wants(content: str) -> WantParse: ...          # PURE: no I/O, no logging, no graph
def want_id_for_text(text: str) -> str: ...              # "cc:want::" + sha1(text.encode("utf-8")).hexdigest()[:16]
```

`surface_wants` calls `parse_wants` and uses `want_id_for_text`; the id derivation moves out of the
loop into the one named function so the #801 repair tool mints byte-identical ids by calling the
same two names (LAW 3/4: one implementation, fixed at the source). The text is **not normalised**
(CRLF, unicode and interior whitespace are kept as written; only `str.strip()` at the ends, as
base does), so a hash over it equals the base hash.

## 3. The algorithm

Offsets are character indexes into the raw `content` string. Steps, in order:

1. **Tokenise markers.** `_WANT_MARKER_RE = \[(/?)WANT\]`, case-sensitive, exactly as base. Every
   match is a candidate marker (opener or closer). No marker, or no `WANT]` substring (cheap
   pre-filter) → empty result.
2. **Mask fenced code blocks** (line-based, CommonMark): a line with 0-3 spaces of indent then a
   run of ≥3 backticks (whose info string contains no backtick) or ≥3 tildes opens a fence; it is
   closed by a later line of the same character, at least as long, and nothing but whitespace after;
   CRLF is handled (`\r?\n`). **An unclosed fence runs to the end of the content** (CommonMark).
   Chosen on purpose: minting a bogus want is permanent (wants are prune-protected); skipping a real
   one is logged and recoverable (the source node still holds the text; a later re-parse mints it).
3. **Mask inline code spans** in the text outside fences, per paragraph (paragraphs split on blank
   lines; a span never crosses a blank line or a fence). CommonMark rule: a backtick run of length n
   opens a span closed by the next run of exactly n backticks; a run with no matching closer in its
   paragraph is literal (masks nothing). O(n) via a precomputed next-run-of-same-length table.
4. **Classify each marker; a failing one goes to `skipped` with a reason and never takes part in
   pairing:**
   - `in_fence` — inside a masked fenced block;
   - `in_code_span` — inside a masked inline span;
   - `code_adjacent` — the character immediately before the marker is a backtick (this keeps the old
     `content[m.start()-1] == "`"` guard as a fallback for backtick runs that step 3 cannot pair
     (unbalanced / cross-paragraph)). The mirror case (a backtick right AFTER the opener, none
     before) is deliberately NOT rejected: a real want may begin with inline code;
   - `escaped` — an odd number of backslashes immediately before the `[` (`\[WANT]`);
   - `quoted` — the marker token itself is wrapped in a matching quote pair: `"[WANT]"`, `'[WANT]'`,
     `“[WANT]”`, `‘[WANT]’`, `«[WANT]»` (and the same for `[/WANT]`). Both the opening and the
     closing quote must be immediately adjacent to the token, so `He wrote "[WANT]fix x[/WANT]"`
     (closing quote not adjacent to the opener) is still a real want.
5. **Pair the live markers, nearest opener wins.** Walk the remaining markers in order with a single
   "pending opener":
   - live opener, none pending → pending = this opener;
   - live opener, one already pending → the pending one had no closer before a nearer opener: move it
     to `skipped` (`opener_unclosed`), pending = this (nearer) opener;
   - live closer with a pending opener → a pair: text = content between them, `.strip()`; if empty →
     `skipped` both as `empty_pair`, else a `WantSpan`; clear pending;
   - live closer with none pending → `skipped` (`closer_without_opener`; this covers a `[/WANT]`
     before any opener);
   - end of content with a pending opener → `skipped` (`opener_unclosed`).
   By construction a returned want contains **no live marker**. A marker that is masked inside the
   pair (a backticked `[WANT]` *inside* a real want) is part of the text, not a marker.
6. **Order** of `wants` is source order; `skipped` is ordered by offset.

What is and is not recognised (explicit, so reviewers can test the edges):

| Recognised as a mention | NOT recognised (stays a real marker) |
|---|---|
| fenced block (``` / ~~~), incl. unclosed-to-EOF | indented (4-space) code blocks — would mask wants inside nested lists |
| inline code span, CommonMark pairing, per paragraph | backslash-escaped backtick (`\``) — treated as a delimiter |
| backtick immediately before the marker | blockquote (`> `) lines — pasted quotes still pair normally |
| `\[WANT]` (odd backslashes) | bold/emphasis around the token (`**[WANT]**`), `<code>`, HTML comments |
| token-level quote wrap (`"[WANT]"` …) | a longer quoted span that merely *contains* a marker |
| a stray closer / an opener with no closer / `[WANT][/WANT]` | bare prose that mentions both markers with no quoting (see failure modes) |

### Failure modes (stated up front)
1. **Bare, unquoted prose mention that pairs cleanly** ("use [WANT] to mark one, and [/WANT] closes
   it") is structurally identical to a real want and WILL be minted. This is the price of "no length
   limit" (the old 600 cap hid it by dropping the span). It is the residual the reviewers should
   weigh; there is no structural signal that does not reintroduce a length heuristic.
2. **Unclosed fence swallows later wants** in the same node (skipped, logged `in_fence`, recoverable).
3. **Nearest-opener pairing is a behaviour change vs base** for `[WANT] a [WANT] b [/WANT]`: base
   drops both; new mints `b` and skips the first opener (`opener_unclosed`). Deliberate (it is what
   P406's "closer pairs with the NEAREST opener" says and it recovers a real want that follows an
   unclosed mention); the old `test_nested_marker_is_rejected` expectation is updated with this reason.
4. **A masked marker inside a real pair** is now captured inside the text where base skipped the
   whole pair. Deliberate (legitimacy is about the opener/closer, not about words in the body).
5. Pathological input is linear-time (single pass per step); no backtracking regex on the body.

## 4. Flood-safe INFO skip log

`surface_wants` runs on every autosave pulse (~60 s) over every conversational node, and the skipped
markers live in historical nodes forever. An unfiltered per-marker line would repeat every pulse.
`parse_wants` stays pure; `surface_wants` collects the skips and logs AFTER it releases the graph
lock (no I/O under `_step_lock`). Bounded form, module constants:

- **Summary** (`logger.info`): one line, `surface_wants: skipped N marker(s) in M node(s) (reason=count, ...)`
  — emitted when the per-reason counts differ from the last summary emitted, or as an hourly
  heartbeat (`WANT_SKIP_SUMMARY_INTERVAL_S = 3600`, `time.monotonic`). A static corpus therefore
  costs ~1 line/hour, not 1/minute. Never silent: any change is reported immediately.
- **Per-marker detail** (`logger.info`): `node=<id> offset=<n> marker=<[WANT]|[/WANT]> reason=<r>`,
  the FIRST time a `(node_id, offset, reason)` key is seen in-process. Seen-set is a bounded
  FIFO (`WANT_SKIP_SEEN_MAX = 4096`); at most `WANT_SKIP_DETAIL_PER_CALL_MAX = 50` detail lines per
  call, the rest deferred (not marked seen) to the next pulses, the summary says how many.
- **No marker TEXT at all** *[corrected turn 2: the detail line carries the literal marker token `[WANT]`/`[/WANT]` as its kind; there is no want body and no surrounding text]* in the log — node id, offset, marker kind, reason only. This is stricter
  than the brief's "short length-bounded prefix": a prefix of surrounding prose could carry a pasted
  secret, and offsets are enough to find the span in the source node. Never a secret by construction.
- **Expected volume:** first pulse after a daemon start = 1 summary + up to 50 detail lines, then
  the remaining distinct skips drain 50 per pulse (K skips ≈ K/50 minutes, once per process), then
  ~1 line/hour while the corpus is unchanged; +1 summary and +1 detail line per NEW mention node.

## 5. What this changes for an already-surfaced want

- **Id derivation is UNCHANGED:** `cc:want::` + sha1(inner.strip() utf-8)[:16]. Same node shape and
  metadata (`kind`, `want_text`, `want_state`, `provenance`, `source_node`, `creation_mode`) and the
  same `graph.create_synapse(nid, want_id, weight=0.3)`.
- **Nothing existing is deleted or altered.** The parser only adds nodes; the 182 laptop want nodes
  keep their ids until the separate #801 repair.
- **Derived data for a re-parse of the 09-16 sources** (`analysis-scratch/want-rows-laptop.json`, 182
  rows, all `cc_authored`/open, 60 distinct source nodes; `probe-laptop.json` keeps only ≤100-char
  heads of source content, so the exact re-parse output cannot be computed from derived data alone):
  - 123 rows hold a marker inside their stored text (mis-parse: a mention swallowed a far closer);
    the new rule never mints such a span as one want; 91 rows start with the backtick that closed a
    code span (83 of them >600): the opener sits inside an inline code span → `in_code_span`, not minted.
  - 59 rows are marker-free; 16 of those are >600 chars (11 at 1,280-1,648, 5 at 6,312-32,387). These
    are the only rows where the no-limit parser can re-derive the **same** id (→ `want_id in
    graph.nodes` → no duplicate) if the span is structurally legitimate — the old 600-limited parser
    silently dropped them on a re-parse.
  - A genuine want that an unbounded 09-16 parse *swallowed* inside a mention-to-closer span was
    never minted under its own text; the new parser mints it once (a recovered want, not a duplicate).
  - Exact per-node counts are computed by the #801 dry-run on a COPY using this function.

## 6. Build, tests, validation (what commits 2-3 will deliver)

Commit 2 — `cc_ng_organism.py`: add `parse_wants`/`want_id_for_text` and dataclasses, replace
`_WANT_RE` with `_WANT_MARKER_RE`, make `surface_wants` call them and log, constants above, changelog
header entry, comment at `:1506-1511` corrected (the "BOUNDED" rationale is superseded). `WANT_MAX_CHARS`
and `render_wants` untouched. `tests/test_cc_want_legitimacy_810.py` (fake in-memory graph/vdb only):
a real 2,000-char want captured whole; the 09-16 shapes (backticked mention to a far closer,
backtick-led, fenced, quoted, escaped) rejected; nested rejected with nearest-opener pairing shown;
backticks inside a real pair captured; idempotence; INFO skip line with reason and bounded volume and
no marker text; `render_wants` unchanged (asserts the held clamp behaviour is byte-for-byte as base);
golden comparison vs `git show e4ebf982:cc_ng_organism.py` on the same inputs; P379 preamble at
session start (resolved module paths, NG modules in `sys.modules`, FAIL if `cc_ng_organism` is not
the worktree copy). `tests/test_cc_want_bounds.py` updated: the two extraction tests that encode the
old 600 rule (`test_span_longer_than_cap_creates_no_want`, `test_span_at_cap_is_accepted`) and the
nested test are replaced with the reason; the render tests stay (renderer untouched).
Validation: the new file once + the updated want-bounds file, `NG_EMBED_*` unset; not the full suite.

Commit 3 — this file completed: what was built, evidence (commands + output), hashes,
the render-retirement inventory (section 8), sequencing (section 7), flags (section 9).

## 7. Sequencing (for the zone manager)

**#810 must land — fixed, reviewed/paired, and the code the laptop daemon runs — BEFORE the #801
repair is applied live, and both before S4.** The repaired `cc:want::`+sha1(text)[:16] ids must equal
what the running parser mints on the next re-parse. The offline dry-run of the repair on a COPY may
import `parse_wants` / `want_id_for_text` from this branch.

## A. What was built (commit `c9fe56d`)

Files: `cc_ng_organism.py`, `tests/test_cc_want_legitimacy_810.py` (new), `tests/test_cc_want_bounds.py`.
`git diff e4ebf982 --name-only` = those three + this doc. No protected, vendored, `cc_ng_host.py`
or `neurograph_rpc.py` file is in the diff (checked).

**The ONE function and its signature (for the #801 repair tool to import):**
```python
from cc_ng_organism import parse_wants, want_id_for_text
parse_wants(content: str) -> WantParse          # cc_ng_organism.py:1682  (PURE)
#   WantParse(wants: Tuple[WantSpan, ...], skipped: Tuple[SkippedMarker, ...])
#   WantSpan(text, want_id, open_start, close_end)      text == inner.strip(), exactly what is stored and hashed
#   SkippedMarker(marker, start, reason)                reason in WANT_SKIP_REASONS
want_id_for_text(text: str) -> str              # cc_ng_organism.py:1587  "cc:want::" + sha1(text utf-8)[:16]
```
`surface_wants` (`:1791`) is now: for each conversational node containing `WANT]`, `parse_wants(content)`,
create a node per `WantSpan` (same metadata / synapse `weight=0.3` / id as base), collect
`parsed.skipped`, and call `_log_want_skips` AFTER the graph lock is released (`:1745`).
Removed: `_WANT_RE` and the inline hashlib derivation. Helpers: `_want_fence_spans` `:1595`,
`_want_code_span_ranges` `:1616`, `_want_marker_mention_reason` `:1658`. Constants `:1541-1556`.
`WANT_MAX_CHARS` (`:1541`) and `WANT_RENDER_LIMIT` (`:1542`) and `render_wants` (`:1844`) are
untouched; `WANT_MAX_CHARS` is referenced by no parser path (asserted by a test).

**Repair-tool notes (#801):**
- Only `provenance == "cc_authored"` want nodes are parser outputs. `cc_emergent` wants
  (`generate_emergent_want`, `cc_ng_organism.py:~1916`) also use `cc:want::`+sha1(want_text)[:16], but
  their text is a synthesised `"tonic-triggered: ..."` string, never parsed from a conversation node —
  the separation rule must not be applied to them.
- The host twin mints `want::`+sha1 (no `cc:` prefix, see section 9) — a different id scheme.
- Offsets in `WantSpan`/`SkippedMarker` are character indexes into the exact `content` string given.

**Flood-safe INFO log, as built** (constants `WANT_SKIP_SUMMARY_INTERVAL_S=3600`,
`WANT_SKIP_SEEN_MAX=4096`, `WANT_SKIP_DETAIL_PER_CALL_MAX=50`):
`surface_wants: skipped 2 marker(s) in 1 node(s) as mentions, not wants (closer_without_opener=1, in_code_span=1)`
and `surface_wants: skipped [/WANT] node=<node id> offset=<n> reason=<reason>`. No want body and no
surrounding text, ever (a sentinel test proves prose around a masked marker never reaches a record) *[corrected turn 2: the
detail line does carry the literal marker token as its kind, so "no marker text" overstated it]*.
Expected volume (logic fixed by the tests with a fake clock; asserted: the first call over 120 distinct skips emits 51
lines, the 5th pulse 0, every marker detailed exactly once, 30 pulses over an unchanged corpus = 0 new lines, one heartbeat
summary after an hour, a new mention node = one summary + its own detail line). The 50/20 middle calls follow from the cap
and are not separately asserted.
Steady state ≈ 1 line/hour, vs 1/minute unfiltered. *[corrected turn 2 (le-014 C4): true only while the corpus holds
<= `WANT_SKIP_SEEN_MAX` distinct (node, offset, reason) skips; beyond that the FIFO evicts entries that re-qualify, so the steady
state degrades to at most `WANT_SKIP_DETAIL_PER_CALL_MAX` (50) detail lines per pulse — bounded, not ~1/hour.]*

## B. Evidence

Environment: worktree root, `env -u NG_EMBED_REMOTE` (`NG_EMBED_REMOTE` was the only `NG_EMBED_*`
name set in the shell; the preamble prints `NG_EMBED_* names set in env: none`), python 3.12.3, pytest 9.0.2.
Command: `env -u NG_EMBED_REMOTE PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_cc_want_legitimacy_810.py tests/test_cc_want_bounds.py -s -q -p no:cacheprovider`

- **Run 1:** `64 passed, 1 failed`. The failure was a bug in MY test harness, not the parser:
  `inspect.getsource` cannot read a function compiled under the synthetic base filename, so the
  "render_wants source identical to base" test raised `OSError`. Fixed (compare `ast` source segments of
  the two files). The brief asked for the new file to be run once; it was run twice for this reason.
- **Run 2 (final):** `65 passed in 1.61s` — 56 in `test_cc_want_legitimacy_810.py` (incl. the preamble and the golden
  / delta tests) and 8 in `test_cc_want_bounds.py` (its 2 render tests are unchanged and passing).
- **P379 preamble (run 2):** `cc_ng_organism in sys.modules /home/josh/NeuroGraph-worktrees/z12-want-legitimacy-810-20260930/cc_ng_organism.py`;
  `neurograph_rpc`, `neuro_foundation`, `ng_lite`, `ng_embed`, `ng_ecosystem`, `ng_tract_bridge`,
  `ng_autonomic`, `openclaw_adapter`, `surface_resolver`, `surfacing`, `cc_ng_host` all `not loaded`
  (the code under test imports none of them, so none is in `sys.modules`; the test FAILS the whole file
  if `cc_ng_organism`, or any of those that IS loaded, resolves outside the worktree).
- **Golden vs base `e4ebf982b1989fd9066d610b94853bc68bf70d37`** (`git show` of base `cc_ng_organism.py`,
  loaded under a private module name, never skipped): 11 well-formed corpora + preexisting-want-nodes +
  provenance-argument + `graph=None` all give an IDENTICAL `(returned wants, node ids+metadata, synapses)`.
- **Documented deltas vs base** (asserted both ways): >600-char want (base `[]`, new captured); nested
  opener (base `[]`, new `["inner"]`); fenced mention (base minted `documented`, new `[]`); quote-wrapped
  mention (base minted `" tag and "`, new `[]`); backticked marker inside a real pair (base `[]`, new captured whole).
- **`render_wants`:** source segment byte-identical to base and output equal to base on 4 graphs,
  including the held 600-clamp (`"w"*1500` still renders as 600 chars): it is untouched.
- Raw transcripts: `/tmp/z12-810-pytest-run1.txt`, `/tmp/z12-810-pytest-run2.txt` (local to this machine).
- Not run, by instruction: the full suite. Not verified: behaviour under the live daemon (nothing restarted).

## C. Deviations from the plan
1. The plan said the two old extraction tests "and the nested test" are replaced; in fact only the
   over-cap test and the nested test flipped. `test_span_at_cap_is_accepted` (a 600-char want is captured)
   is still true and was kept.
2. "render_wants source identical" compares `ast` source segments, not `inspect.getsource` (see Run 1).
3. The cheap pre-filter is `"WANT]" not in content` (covers a stray `[/WANT]`-only node, which base ignored
   silently and is now logged as `closer_without_opener`); base used `"[WANT]" not in content`.
4. `surface_wants` logs after leaving the `with` block; `return open_wants` moved to the end (same value).
Everything else is as planned.

## D. What the re-parse of the 09-16 sources now produces (derived data only)
As in section 5. Additionally, the derived probe holds no source content beyond ≤100-char heads, so no
per-node re-parse count is claimed here; the #801 dry-run computes it on a COPY by importing the function
above. The category statement stands: marker-bearing (123) and backtick-led (91) stored wants are
mention-shaped and are not re-minted; the 16 marker-free >600 rows are the only ones the no-limit parser
can re-derive under the same id (no duplicate), *if* their source opener is structurally real.
For the record (derived), what the UNTOUCHED renderer produces over those 182 open `cc_authored` rows
today: 18,577 chars (40-cap + 600-clamp); 486,803 with the clamp removed (40-cap only); 78,635 with the
clamp and no cap; 2,267,508 with neither — the 09-16 figure. This is context for why P408 retires the
block rather than clamping it; no decision is made here.

## 8. Render-retirement inventory (read-only) — sizing the second turn

Nothing below was edited. Daemon lines are given for docs `origin/main` (the lines the brief cites; the
unit runs `~/docs/scripts/...`) AND for the docs branch `cc-laptop-daemon-recall-756-20260930` HEAD
`155343e4`, because that branch restructured the same function (#779/#756) and the numbers differ.

### 8.1 (1) Remove the standing wants block from the per-prompt context; fix the SessionStart hint
| What | `origin/main` | branch `155343e4` |
|---|---|---|
| SessionStart hint text promising `'## What I Want'` | `scripts/cc-ng-daemon.py:1067-1073` (in `handle_session_start` `:1058`) | `:1125-1131` (`handle_session_start` `:1115`) |
| per-prompt import + `render_wants(STATE.ng.graph)` + append loop, one `try` with the "Who I Am" block | `:1113-1120` (`handle_user_prompt_submit` `:1092`) | `_append_identity_blocks` `:1951-1985`; import `:1965`; `render_wants` call `:1977`; caller `:1177` in `handle_user_prompt_submit` `:1152` |
| wants-specific failure codes/fields | (none on main) | `IDENTITY_ERR_WANTS` `:776`, `IDENTITY_ERR_CORE_AND_WANTS` `:777`, `_identity_fields` `:1937-1949` (the `core_and_wants_render_failed` collapse `:1946-1947`), `_report_recall(..., IDENTITY_ERR_WANTS ...)` `:1979`, module header note `:13-14` |
| docs-repo tests that encode the block | — | `scripts/tests/test_cc_ng_daemon_recall_status.py`: `:137` (fake `render_wants`), `:283-301`, `:317-319` (`drop=('render_constitutional_core','render_wants')`), `:329-334`, `:389-392`, `:439-447`, `:458` |
| NeuroGraph host (VPS half) | `cc_ng_host.py:1141-1149` — imports `render_constitutional_core, render_wants`, appends `wants_block` | same (this repo; NOT touched this turn; the brief says `cc_ng_host.py` is out of this lane — turn 2 must be told whether the VPS host block is in scope) |
| the renderer | `cc_ng_organism.py:1844` `render_wants`; constants `:1541` `WANT_MAX_CHARS`, `:1542` `WANT_RENDER_LIMIT` (both die with it; `WANT_MAX_CHARS` has no other user) | |
| NeuroGraph tests touching it | `tests/test_cc_want_bounds.py:114-135` (render count-cap + clamp + no-elision: obsolete on retirement); `tests/test_cc_want_legitimacy_810.py:459-511` (this turn's "render_wants unchanged vs base" — must be deleted or inverted in turn 2); `tests/test_cc_deposit_step.py:54,290,400` (monkeypatch `render_wants` to a no-op: harmless but stale). `tests/test_self_render.py` is **Syl's** `_render_self_and_wants` (`syl_authored`, `neurograph_rpc`) — NOT affected ("Syl's block is untouched"). |
| "Who I Am" | `render_constitutional_core` `cc_ng_organism.py:1882` — unchanged; it shares the `try`/append loop with the wants block on main (`:1113-1120`), so the edit must keep it |
| docs / vault references to the string `## What I Want` | `constitutional spine` and dev-log pages, punchlist `punchlist/open/neurograph.md`, `punchlist/open/tid.md`, `prd/2026-06-21-reach-teaching-plan.md`, `handoffs/z12-daemon-recall-swallow-756/returns/{plan,build}-001.md` (history; update only if they describe current behaviour) |

### 8.2 (2) Make want nodes eligible in the EXISTING recall/surfacing path, rendered in FULL
The existing path (one shared pipeline for both hemispheres): `cc_assemble_recall`
`cc_ng_organism.py:5595` = `SurfacingMonitor` block (`surfacing.py`) + Active Recall block
(`cc_pattern_completion_recall` `:3117`, formatted by `_format_cc_recall_block` `:3301`), optionally Pith.

**How a want node is shaped today:** created by `surface_wants` with metadata
`{kind:"want", want_text, want_state:"open", provenance, source_node, creation_mode:"conversational"}`
plus a `poincare_dir` stamped by `cc_stamp_missing_geometry` (`:3657`, which reads `want_text`), and ONE synapse
`source conversational node -> want` at weight 0.3. It has NO `_forest_content`, NO `_label`, and NO vector_db entry.

**Why it is not eligible today (read from the code; not exercised against a live graph):** the want node carries a
synapse and a `poincare_dir`, so spreading activation can plausibly fire it, but every consumer then resolves display text with `surface_resolver.resolve_surface_content`
(`surface_resolver.py:54-115`), which reads only `_forest_content`, then the vdb entry, then `_label` — never
`want_text` — so the want resolves to `None` and is dropped:
- Active Recall: `cc_ng_organism.py:3252` (`text = resolve_surface_content(node, r, allow_ingested=True, max_chars=300)`; `if not text: continue` `:3253-3254`);
- SurfacingMonitor: `surfacing.py:192-193` (`resolve_surface_item` -> `resolve_surface_content`), dropped at `surfacing.py:207` (`if not content and not image_ref: continue`).
(The Pith text resolver `_pith_node_raw_text` `cc_ng_organism.py:4909` ALREADY reads `want_text` — it is only reached for items that survive the two filters above.)

**Smallest change that makes a want eligible:** teach the resolver (or a CC-only wrapper at `:3252`) that a
`kind == "want"` node with `want_state == "open"` resolves to its `want_text`. Retrieval needs no new node/embedding
because the substrate path (synapse + `poincare_dir`) is what fires it.

**Four things the turn-2 plan must decide (surfaced, not decided):**
1. **`surface_resolver.py` is shared with Syl.** `resolve_surface_content` is also called by `neurograph_rpc.py:3276,3458,3520`
   (Syl's `/assemble`) and `tonic_thread.py:588`. Editing it would make Syl's own want nodes (also `kind:"want"`) eligible in
   her recall and Tonic surfacing too — "Syl's block is untouched" does not cover that. Either gate it (CC-only wrapper
   at `cc_ng_organism.py:3252` + a `surfacing.py` hook) or get Josh's ruling that Syl's wants surfacing is wanted.
2. **"In FULL, never cut" vs the path's existing bounds.** Every stage of the existing path clips: `resolve_surface_content`
   `max_chars` (240 default; `300` at `cc_ng_organism.py:3252`; `surface_resolver.py:112-114`), `SurfacingMonitor.format_context`
   hard-trims to 200 chars (`surfacing.py:293-295`), and the Pith provider clips per node to `_CC_PITH_PROVIDER_NODE_CHARS`
   (default 700, `cc_ng_organism.py:3781`; `_pith_node_text` `:4924`) inside an L1 budget (`cc_l1_budget` `:4504`).
   Making a want render whole needs a per-kind exemption at each of those stages, or a separate want lane beside them.
3. **Existing `want_state`/`provenance` semantics:** `render_wants` filtered `provenance in {cc_authored, cc_emergent}` and
   `want_state == "open"`; the eligibility rule must reproduce that filter (closed wants must not surface).
4. **The 182 existing nodes** include ~123 mis-parse wants (section D). Making wants eligible BEFORE the #801 repair
   would surface up to 136k-char mention-spans through the recall path, unbounded. Sequencing: #810 lands -> #801 repair
   applied -> THEN eligibility (this is the same ordering as the sequencing ruling, section 7, plus this constraint).

### 8.3 (3) Spontaneous surfacing — plan-only for S4 (what already fires)
Existing independent clocks a plan could reuse, none changed: the autosave pulse that already calls
`surface_wants` + `generate_emergent_want` (`cc_ng_host.py:1519-1531`; docs `scripts/cc-ng-daemon.py:1929` on main /
`:2116` on the branch, ~60 s), and the Tonic (`_tonic_thread`, reported in `handle_status`; the laptop has none until S4,
same Door-B-style gate as the brief says). `generate_emergent_want` already writes `cc_emergent` want nodes from the
substrate's own predictions. No timer or quota is proposed.

### 8.4 Sizing read (for the zone manager)
Turn 2 = one docs-repo daemon edit (two hunks on main, four on the #756 branch) + its test file, one NG edit
(`cc_ng_organism.py` renderer + constants removal, resolver eligibility) + NG tests (delete/invert ~6), and — only if
Josh wants them — the `cc_ng_host.py` VPS block and the `surface_resolver` Syl question. The decisions in 8.2 (1), (2)
and (4) gate the size; everything else is mechanical.

## 9. Flags raised (not acted on)
1. **Host twin `surface_wants_for_graph`** (`cc_ng_organism.py:1151-1218`, called from `cc_ng_host.py:698-700`): still the
   UNBOUNDED, legitimacy-free `re.finditer(r'\[WANT\](.*?)\[/WANT\]')` — the exact 09-16 mis-parse bug class — and it mints
   `want::`+sha1 ids (no `cc:` prefix), a DIFFERENT scheme from `surface_wants`, so the same want text can exist under two
   ids if both run. Parked in #755 per the brief; flagged for the punchlist (LAW 4: fix at the source, one implementation).
2. **`neurograph_rpc.py:4902/4914`** (Syl's `_surface_wants`): identical unbounded regex on the `syl_authored` path; needs Josh's approval.
3. **Residual failure mode:** a bare, unquoted prose mention that pairs cleanly ("use [WANT] to mark one and [/WANT] closes
   it") is structurally identical to a real want and WILL be minted. There is no structural signal for it that is not a length
   heuristic. Reviewers should weigh it; the INFO log does not cover it (it is not skipped).
4. **Nearest-opener pairing changes base behaviour** for `[WANT] a [WANT] b [/WANT]` (base dropped both; new mints `b`) and a
   masked marker inside a real pair is now kept in the text (base dropped the whole pair). Both deliberate (P406) and asserted;
   listed so the review pair can reject them.
5. **`code_adjacent` keeps the pre-#810 guard:** a real want typed directly after a closing backtick (`foo`[WANT]...`) is
   still skipped (logged). Same as base. *[corrected turn 2 (le-014 C1): "same as base" held for OPENERS only. The
   build also applied `code_adjacent` to CLOSERS, which base never did, so a real want ENDING in inline code
   (`[WANT]check `foo()`[/WANT]`) was dropped, opener included. Fixed in `6e3376e`; see `build-002.md`.]*
6. **Same-function conflict risk for turn 2:** the docs branch `cc-laptop-daemon-recall-756-20260930` restructured
   `_append_identity_blocks` (#779/#756); a retirement edit on `origin/main` lines will conflict with it. Decide the base branch.
7. **Return-doc location:** written into THIS (NeuroGraph) branch under `handoffs/z12-want-legitimacy-810/returns/`, because the
   NeuroGraph branch is the only one I was told to push. Copy it into the docs repo's `handoffs/` if the vault layout needs it there.
   Vault Context-Map/wikilink updates were NOT done (the docs branch is not mine to edit this turn).
8. **P379 nuance:** in the test process the NG modules other than `cc_ng_organism` are not imported at all, so the "resolved
   path" check for them is vacuous; the load-bearing check is `cc_ng_organism` resolving to the worktree.
9. A stray `find / ...` I started early (before scoping it) was killed and its partial output discarded; nothing was read
   outside the named locations except the listed read-only greps of the docs repo daemon/tests and git objects.
