# #810 build-001 — WANT parser: structural LEGITIMACY, no length limit

Lane `want-parser-legitimacy-810` · Zone manager Z12 · dispatch #10715 · worker turn 1 of 2
Branch `cc-laptop-want-legitimacy-810-20260930` · base `e4ebf982b1989fd9066d610b94853bc68bf70d37`
Related: [[NeuroGraph]] · [[The Laws]] (LAW 3/4/7) · [[NeuroGraph Is a Mind, Not a Database]]

> **Status: PLAN (commit 1 of 3).** Nothing below is built yet. Sections 6-8 (what was built,
> evidence, render-retirement inventory) are appended by the later commits of this same file.

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
- **No marker TEXT at all** in the log — node id, offset, marker kind, reason only. This is stricter
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

## 8. Render-retirement inventory (read-only) — appended in commit 3

## 9. Flags raised — appended in commit 3
