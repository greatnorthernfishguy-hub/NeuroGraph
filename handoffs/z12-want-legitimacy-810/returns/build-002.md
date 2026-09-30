# #810 build-002 — turn 2: le-014 corrections (C1-C4 + LAW 5) and the #815 delta

Lane `want-parser-legitimacy-810` · Zone manager Z12 · dispatch #10831 · worker turn 2 of the parser lane
Branch `cc-laptop-want-legitimacy-810-20260930` · base `e4ebf982b1989fd9066d610b94853bc68bf70d37`
Reviewed-and-pinned inputs: turn-1 build `e462c17a` (code `c9fe56d`), branch head at turn start `16f02f9f`
Related: [[NeuroGraph]] · [[The Laws]] (LAW 3/4/5/7) · [[NeuroGraph Is a Mind, Not a Database]] · turn 1: `build-001.md`

> **Status: RETURNED.** Not self-accepted; nothing merged, wired or restarted. `render_wants` is UNTOUCHED
> (its source segment is byte-identical to base; a test in the run below asserts it). No protected, vendored,
> `cc_ng_host.py` or `neurograph_rpc.py` file is in the diff.

## 0. Commits (each pushed by name; hashes from `git rev-parse HEAD`)

| Step | What | Commit |
|---|---|---|
| 1/3 | le-014 C1-C4 + LAW 5 + wording, golden cases asserted against BASE | `6e3376e29ebebd234055a5485d1f155269ba02cd` |
| 2/3 | #815 delta: `in_json_string`, `in_url`, `in_link_target` | `2a2703f565e1f3e9dd7b3504e60909c4e1cd4b89` |
| 3/3 | this return (+ `[corrected turn 2]` markers in `build-001.md`) | see the final report / `git log -1` |

Files changed across steps 1-2: `cc_ng_organism.py`, `tests/test_cc_want_legitimacy_810.py` only.

## 1. le-014's corrections — disposition

| le-014 item | Disposition | Where proved |
|---|---|---|
| **C1 (HIGH)** a real want that ends in inline code is dropped, opener included | **Fixed at the source.** `_want_marker_mention_reason` now takes `is_close`; the adjacency guesses (`code_adjacent`, `escaped`, `quoted`, and #815's link/URL/escaped-quote-hug) apply to OPENERS only. Fences, balanced code spans and (new) JSON string literals still mask ANY marker. | `test_le014_golden_against_base[...]` (11 cases, each asserted against `git show e4ebf982`: ends in code, begins with code, both, follow-on second want, closer after quote pair, ends in backslash, ends in URL, URL-ending want + adjacent want, want containing URL/quote/code, opener-after-backtick guard kept) |
| **C2 (MEDIUM)** `quoted` on closers swallows a real want | Fixed by the same `is_close` short-circuit (`x "[WANT]I want "x"[/WANT]" y` → `I want "x"`, same as base). | `test_le014_golden_against_base[C2 ...]`, `test_closer_adjacency_guesses_do_not_apply_to_closers` |
| **C3 (MEDIUM)** adversarial corpus + residuals stated | Corpus added (`_ADVERSARIAL_PARITY`, 16 shapes of a real want next to code / quotes / stray backticks / URL / link / JSON, asserted `== base`). The stray-backtick false negative is **pinned as a documented residual** (`_ADVERSARIAL_RESIDUAL`, asserts base mints and the build does not, and that the skip is reported). | `test_adversarial_real_want_near_code_quotes_backticks_equals_base`, `test_stray_backtick_residual_is_exactly_as_documented` |
| **C4 (LOW)** flood claim | **Claim corrected, FIFO design kept.** Stated exactly in the code comment and pinned by a test: while distinct skips ≤ `WANT_SKIP_SEEN_MAX` a static corpus costs ≈ 1 line per heartbeat interval (1/hour default); beyond it the FIFO evicts keys that re-qualify, degrading to at most `WANT_SKIP_DETAIL_PER_CALL_MAX` detail lines per pulse — never unbounded. | `test_flood_claim_beyond_seen_max_degrades_to_the_per_call_cap` |
| **LAW 5 (LOW)** env-source the bounds | `CC_WANT_SKIP_SUMMARY_INTERVAL_S` (3600), `CC_WANT_SKIP_SEEN_MAX` (4096), `CC_WANT_SKIP_DETAIL_PER_CALL_MAX` (50) via `_want_skip_env_int` (junk → default, below-minimum → minimum 1; never breaks import). | `test_skip_log_bounds_are_env_sourced_with_current_values_as_defaults` (subprocess import with the env set) |
| **Wording** "NEVER logs marker text" | Corrected in the code docstring, the changelog header and here: the detail line carries the **literal marker token** (`[WANT]` / `[/WANT]`) as its kind, plus node id, offset and reason — **no want body and no surrounding text**. | `test_log_docstring_no_longer_claims_no_marker_text` |
| **flag-5** "`code_adjacent` same as base" | Was true for OPENERS only; it was false for closers (that was C1). Corrected in `build-001.md` with a `[corrected turn 2]` marker. Now true for what it claims: `code_adjacent` fires on openers only, exactly as base. | above |
| **Vault sync (LOW)** | **Owed by the zone manager — not done by me.** `handoffs/` lives in the **NeuroGraph repo** on this branch (`handoffs/z12-want-legitimacy-810/{returns,reviews}/`), not in the docs repo; the docs-side Context-Map/wikilink update and the punchlist entries (#755: host twin `surface_wants_for_graph` unbounded + second id scheme; Syl's `_surface_wants`) are the zone manager's to file. | — |

### 1.1 Item 5 — "the SAME result as base for EVERY well-formed want shape": how the corpus proves it
`test_parity_with_base_on_every_well_formed_shape` builds `prefix × body × suffix` and asserts, for every one, that
`surface_wants` returns identical wants, ids, node metadata and synapses to BASE `e4ebf982` (loaded by `git show`,
private module name; a missing base FAILS, never skips).
- **18 prefixes:** empty, prose, newline / blank-line / CRLF before, inline code before, quoted word before, backslash
  path, list-ish, parens, emoji, and the #815 near-misses (URL + space, parenthesised URL glued, URL + sentence dot glued,
  escaped quotes in prose, a JSON object before, a markdown link before with and without a space).
- **23 bodies** (each ≤ 600 chars, since base drops longer): plain; begins with code; **ends with code**; both; code mid;
  quoted word; **ends with a quote**; begins with a quote; **ends with a backslash**; contains a URL; **ends with a URL**;
  ends with a URL carrying `[1]`; JSON object inside; multi-line; unicode; CRLF; brackets/parens; tab; ``double`` code;
  a stray `[y"` fragment; contains a markdown link; a JSON array; **ends with a markdown link**.
- **9 suffixes:** none, prose, newline prose, a **follow-on second want** (spaced and adjacent), code after, a fenced block
  after, a quoted word after, a lone backtick after.
- **Result: 18 × 23 × 9 = 3,726 shapes, 0 divergences** (checked in the targeted run, and independently by a scratch
  probe before the tests were written — same number, same result; it found 0 of 1,980 at the narrower turn-1/step-1 corpus).
- **What the corpus is and is not:** every shape in it has the author's real `[WANT]` in plain prose (no fence / span /
  quote / escape / URL / link / JSON string wrapping the OPENER), so by design any divergence is a regression. It is a
  constructive proof over a product grammar, not a proof over all strings; the deliberate deltas (over-600, nested,
  fenced/quoted/escaped mention, backticked marker inside a pair, and the #815 set) are asserted separately against base in
  `_DELTAS` / `_815_EXAMPLES`; the stray-backtick residual is pinned in `_ADVERSARIAL_RESIDUAL`. Anything outside
  these — e.g. exotic nesting of several features at once — is covered only by the unit cases, not by an exhaustive claim.

## 2. #815 — the recognition rules, exactly

The parser's mention test now has three tiers (`_want_marker_mention_reason`, docstring):
1. **Structure, ANY marker:** `in_json_string` (first) → `in_fence` → `in_code_span`.
2. A closer that survives tier 1 is **real** (base never guarded closers; a closer guard is the C1 class).
3. **Opener context only:** `in_link_target`, `in_url`, the JSON-escaped-quote hug (`in_json_string`), `code_adjacent`,
   `escaped`, `quoted`. A mentioned opener's orphan closer is skipped as `closer_without_opener`.

### `in_json_string`
- **Rule A (structural literal, any marker).** The marker lies inside a *structurally valid JSON string literal*: an
  unescaped `"` whose previous non-space character is `{` or `[`, or `:` preceded by a `"` (a `"key":` value), or `,`
  preceded by a finished JSON value (`"`, digit, `]`, `}`, `true`, `false`, `null`); closed by the next unescaped `"`
  **on the same line** (JSON strings carry no raw newline; `\n` escapes are fine); and the closing `"` followed by
  `,` `}` `]` `:` or end of content (whitespace allowed). Checked **before** fences and code spans, so
  `{"code": "```\n[WANT] documented [/WANT]\n```"}` is rejected *as JSON*, not by the coincidence that its two ``` runs pair
  as an inline span (a test proves the coincidence exists and that the reason is still `in_json_string`).
- **Rule B (escaped-quote hug, opener only).** The opener is immediately wrapped by JSON-escaped quotes: `\"[WANT]\"`.
- **Failure modes:** (1) a legitimate want typed inside a JSON-*looking* sentence (`Use {"note": "[WANT] revisit X [/WANT]"} for it`)
  is rejected — pinned as a KNOWN false negative; (2) a quoted list in prose (`"a", "b [WANT]x[/WANT]"`) that happens to
  match the opener/closer context is rejected; (3) quotes inside the want text never matter (only where the opener sits).
- **NOT recognised:** single-quoted JSON-ish; a "string" containing a raw newline (invalid JSON); a whole-content string
  (`"[WANT] x [/WANT]"` at content start with no structural context); YAML/TOML strings; `\"` hug on a *closer*.

### `in_url`
- **Rule (opener only).** The OPENER is glued — no whitespace, and no other WANT marker between — to a run that contains
  `scheme://` (`[A-Za-z][A-Za-z0-9+.-]*://`), looking back at most 2,048 characters. A character right before the opener
  that *ends* a URL rather than continuing it (`) ] } > " ' ` , ; ! .`) means it is NOT in the URL.
- `https://example.com/path/[WANT]secret-want[/WANT]/docs` → opener `in_url`, closer `closer_without_opener`.
- **Why opener-only:** `[WANT]read https://x.com/a[/WANT]` — a want that ENDS in a URL — must keep its closer
  (le-014 C1 class); `[WANT]read https://x.com/a[/WANT][WANT]second[/WANT]` must keep the second opener.
- **Failure modes:** a real opener typed directly against URL characters with no whitespace and no terminator
  (`https://x.org/a?[WANT]...`) is rejected; `www.` and scheme-less hosts are not URLs here.
- **NOT recognised:** scheme-less URLs, `mailto:`-style (no `//`), a URL whose opener is separated by whitespace
  (that is the ordinary "URL then a want in the same paragraph": captured).

### `in_link_target`
- **Rule (opener only).** The glued run before the opener contains `](` with no `)` after it — the opener sits inside
  the destination of a markdown link/image: `[the docs](https://example.com/[WANT]linked[/WANT])`. Checked before `in_url`.
- **Failure modes:** `[t](dest with no spaces/[WANT]...` only; a destination containing a space before the opener is not seen.
- **NOT recognised:** link TEXT containing a pair (`[[WANT]text[/WANT]](https://x.org)` → captured), scheme-less
  reference definitions (`[id]: ./dir/[WANT]x[/WANT]` → captured), autolinks, HTML `<a>`/`<code>`/comments, bold/emphasis wrappers.

### The boundary (tested): the reason applies to the OPENER's context, never to the want text
Captured whole, with real tags in the author's prose: a want that contains a URL; contains JSON; contains a markdown link;
contains a quoted word; ends in a URL; ends in a link; a URL then a want in the same paragraph (space, sentence dot,
or a closing paren glued to the opener); escaped quotes in prose; a prose quote followed by a comma. A genuine
**> 600-character want in plain prose placed beside each of checker-016's five examples (+ the backtick-fence variant),
before and after, same line and new paragraph (24 arrangements) is captured whole** and the example is still skipped with its reasons.

### Performance (found and fixed while testing)
The first implementation ran a regex per `"` and took **17.8 s on a 200,000-quote node** (this runs on every autosave
pulse over every node containing `WANT]`). Replaced by `_want_json_opens_literal` (constant work per quote). Now 0.5 s
worst case on the same node; `test_815_finders_are_linear_on_adversarial_input` bounds four adversarial shapes at 5 s.

## 3. Residuals and known limits (for the zone manager / reviewers)
- **Stray backtick (le-014 C3):** a lone backtick masks a real pair in the same paragraph when another backtick follows in
  that paragraph (CommonMark-faithful; base minted it). A skip line is the only trace; the pinned test documents both shapes.
- **JSON-escaped pasted tool output:** the escaped-quote hug (rule B) and JSON literals (rule A) are recognised; a fenced block
  whose newlines are JSON-escaped inside a *non-literal* (no structural context) is masked only if its backtick runs pair.
- **Bare prose mention** that pairs cleanly ("use [WANT] to mark one, and [/WANT] closes it") is still minted (stated in turn 1).
- **A legitimate want inside a JSON-looking sentence** is rejected (rule A failure mode 1) — a deliberate precision trade under P406.
- **Open from le-014, not addressed here:** `cc_stamp_missing_geometry` embeds the full `want_text`, so unbounded wants now reach
  the embedder whole (`ng_embed` context-limit behaviour not verified). Belongs to the geometry-backfill owner.
- **Host twin / Syl twin** (`surface_wants_for_graph`, `neurograph_rpc._surface_wants`): unchanged, parked (#755 / Josh).

## 4. Evidence

Command (worktree root, `NG_EMBED_*` unset — `NG_EMBED_REMOTE` was the only name set):
`env -u NG_EMBED_REMOTE PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_cc_want_legitimacy_810.py tests/test_cc_want_bounds.py -s -v -p no:cacheprovider`
- **Run (once): `156 passed in 4.10s`** — 148 in `test_cc_want_legitimacy_810.py`, 8 in `test_cc_want_bounds.py` (the earlier 56+8 are
  all still green; the file grew by 92 tests). Transcript: `/tmp/z12-810-t2/pytest-run.txt` (local to this machine).
- **P379 preamble (printed by the run):** `cc_ng_organism in sys.modules /home/josh/NeuroGraph-worktrees/z12-want-legitimacy-810-20260930/cc_ng_organism.py`;
  `neurograph_rpc`, `neuro_foundation`, `ng_lite`, `ng_embed`, `ng_ecosystem`, `ng_tract_bridge`, `ng_autonomic`, `openclaw_adapter`,
  `surface_resolver`, `surfacing`, `cc_ng_host` all `not loaded` (nothing under test imports them); `NG_EMBED_* names set in env: none`.
  The file FAILS itself if `cc_ng_organism` does not resolve to the worktree copy.
- **Disclosed non-pytest probes (my own, scratch, `/tmp/z12-810-t2/*.py`):** pure-string calls of `parse_wants` and of BASE's
  `surface_wants` on fake in-memory graphs, used to choose the rules and to size the performance fix *before* the tests were
  written (the brief allows the targeted test files to be run once). No graph load, no checkpoint, no live tract, no daemon,
  no embed. They are not evidence of the test results above.
- Golden vs base (asserted in the run): 11 turn-1 corpora + 11 le-014 cases + the 3,726-shape parity corpus + 16 adversarial
  cases are identical to `e4ebf982`; every deliberate delta is asserted both ways (`_DELTAS`, `_815_EXAMPLES`).
- Not run, by instruction: the full suite. Not verified: behaviour under the live daemon (nothing restarted).

## 5. The #801 repair plan's derived-data recount MUST be re-taken on the FINAL function

Pin the recount to the final branch tip (the commit that contains this file and its parents), never to `c9fe56d` / `e462c17`.
I did not read the repair plan `plan-003` (it lives on branch `cc-laptop-want-text-repair-20260930`); from the turn-3 request
I can name what depends on the parser, and all of it must be recomputed, not patched:
1. **Any count taken with the turn-1 function (`c9fe56d`/`e462c17`) is invalid for wants that end in inline code** (C1 dropped
   them, opener included): source nodes that looked like "no legitimate pair" may now have one, and their ids now equal base's.
2. **The A/B/C classes of plan-001 §3.4 and the G1-G4 separation rule / backtick proxy** — the separation rule IS `parse_wants`,
   so each node's class must be recomputed by calling it on the node's SOURCE content. The new reasons move nodes:
   sources that are JSON-escaped tool results (the common case) may now have their opener skipped as `in_json_string`;
   URL/link-carried tags as `in_url` / `in_link_target`. A want that was "genuine" under fence/code/quote-only rules can become
   "skipped mention" (and vice-versa a wrongly-skipped real want becomes genuine after C1/C2).
3. **The "marker-free >600" population (16 of 182, derived in build-001 §5) that I said the no-limit parser could re-derive under
   the same id** — that statement assumed fence/code/quote rules only; a source opener sitting in a JSON string, URL or link
   destination would now NOT re-mint. Re-derive per node; do not reuse the 16.
4. **The id-follows-text acceptance (P402) and the collision rule:** unchanged mechanics (`want_id_for_text`), but the
   *set* of texts the parser mints — hence which repaired ids have a re-parse twin — changes with both steps.
5. **The LEFT-UNCHANGED list** (cases where mention-vs-real cannot be told mechanically): its membership changes — some former
   members are now decided by `in_json_string` / `in_url` / `in_link_target`; the residuals in section 3 (stray backtick, bare
   prose mention, legitimate want inside a JSON-looking sentence) are exactly what a frozen-list human review still must catch.
6. **Recovered-want count** (genuine wants a mention-to-closer span swallowed, which the nearest-opener rule now mints once)
   depends on the new masks: recount.
7. The dry-run on a COPY imports `from cc_ng_organism import parse_wants, want_id_for_text` from this branch; importing runs the
   whole module but opens no file and loads no NG engine (checker-016 A5).
Sequencing (ruled, unchanged): #810 (this branch, fixed, pair-reviewed on the delta, and the code the laptop daemon runs)
BEFORE the #801 repair is applied live; both before S4. The delta review (ROLE A + ROLE B on `6e3376e..2a2703f`) is the gate
before the recount is trusted.

## 6. State for the next turn (not started)
`render_wants`, `WANT_MAX_CHARS`, `WANT_RENDER_LIMIT` and the host twin are untouched and still as inventoried in `build-001.md`
§8. The render-retirement turn is the zone manager's to dispatch; nothing here pre-empts its decisions.
