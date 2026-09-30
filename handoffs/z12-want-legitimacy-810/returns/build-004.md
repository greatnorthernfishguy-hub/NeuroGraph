# #810 build-004 — turn 4: the NEW final parse_wants (le-019 F1-F5)

Lane `want-parser-legitimacy-810` · Zone manager Z12 · dispatch #10988 · worker turn 4 of the parser lane (then STOP)
Branch `cc-laptop-want-legitimacy-810-20260930` · parser base `e4ebf982b1989fd9066d610b94853bc68bf70d37`
Input reviewed: turn-3 head `0748805a19ad6213d37bf45ad38b030db60a8242` (code-identical to `01e0065`), `le-019-810-turn3.md` (read in full), the brief's TURN 4 section.
Related: [[NeuroGraph]] · [[The Laws]] (LAW 3/4/7) · [[NeuroGraph Is a Mind, Not a Database]] · turns 1-3: `build-001.md`, `build-002.md`, `build-003.md`

> **Status: RETURNED.** Not self-accepted; nothing merged, wired or restarted. `render_wants` is UNTOUCHED (source segment byte-identical to base,
> asserted in the run). No protected, vendored, `cc_ng_host.py`, `neurograph_rpc.py`, `surface_resolver.py` or `surfacing.py` file is in the branch diff.
> The INFO skip log is unchanged and no content-aware exemption was added. **The turn-3 pin (`01e0065` / `42745f5d…`) is SUPERSEDED by section 0.**

## 0. THE NEW FINAL FUNCTION PIN — the #801 counts and the repair plan's pin are taken ONLY on this

| What | Value |
|---|---|
| Code commit (parser + tests, ONE diff) | `ae798b94cb14740d200fc3f4fd8d36eef8b86c6a` |
| Defining file | `cc_ng_organism.py` (`parse_wants` + `want_id_for_text` + helpers) |
| Blob id (`git rev-parse ae798b9:cc_ng_organism.py`) | `a3aa8a0ddb6a89fe468a9a20beadc5e381624cab` |
| **sha256 of the file** | `8ad0f69ed4a7849c98e97e0f3970bbe4a36c727ef880455e8ddecec14de5a8e2` |
| Test file sha256 (`tests/test_cc_want_legitimacy_810.py`) | `04b1a494e4a5762b1e3f97d07e12c0937a50bf5baac09e5208b2bd9f82a46c53` |
| Return commit (this file; `cc_ng_organism.py` unchanged by it) | the branch tip — `git rev-parse HEAD` after this commit |

Verify with `git show ae798b9:cc_ng_organism.py | sha256sum`. Any later change to `cc_ng_organism.py` re-opens the recount.

## 1. The fixes — each with its goldens (asserted against BASE `e4ebf982`; a missing base FAILS, never skips)

### F2 (MEDIUM) — a bare prose word/number + comma + quote is not a JSON literal
- **What was wrong (le-019, reproduced):** `_want_json_opens_literal` accepted ANY whole-token `true/false/null` or ANY digit run before `comma + quote`, in ordinary prose, so
  `In 2026, "[WANT]revisit the exit policy[/WANT]", I wrote.` and `It is true, "[WANT]…[/WANT]", she said.` dropped a well-formed want that base minted. Turn 3 mis-described this as
  "a real `true,` token = a JSON-looking sentence"; it was not limited to that, so it is **fixed, not named**.
- **"Actual JSON context", stated precisely (`_want_json_element_in_container`, memoised per comma):** a `"` opens a JSON literal only when, looking back over non-blank characters, it is
  (a) right after `{` or `[`; or (b) right after `"key":` (a quoted key, then colon — unchanged from turn 2); or (c) right after a **comma that continues a JSON array or object**: walking back
  over complete JSON VALUES — a string (same line, ≤ 512 characters, escaped quotes skipped), a number (`-1.5e3`, whole token), a **whole-token** `true`/`false`/`null`, or a balanced `[…]`/`{…}`
  (≤ 256 characters) — and `"key":` members, separated by commas, reaches an opening `{` or `[`. **A bare prose word, number or quoted phrase before the comma never does** (`it was 3,`,
  `the answer is true,`, `ok: true,` (a YAML-ish colon is not a quoted key), `She said "a",`, `3,"`). The literal must still close on the same line before `, } ] :` or end of node.
- **Goldens (15 mint-again, each `== base`):** true, false, null, year (le-019's shape), chapter number, number, `-true,`, `.null,`, `3,"`, a quoted phrase list in prose (le-016), YAML-ish `ok: true,`,
  `the answer is true,`, `it was 3,`, and two with a JSON-closer-looking tail (`"}` / `"]`). **A second regression net:** 8 prose prefixes × 8 bodies × 9 JSON-closer-looking suffixes = **576 shapes, 0 divergences from base.**
- **Rejected examples kept (18, each asserted `wants == ()` with `in_json_string`):** checker-016's object value, escaped quotes, one-line tilde fence and backtick fence in JSON, the JSON string list; `{"a": true, "…": 1}`,
  `[true, "…"]`, `{"a": 1, "…": 2}`, `[1, 2, "…"]`, `[-1.5e3, "…"]`, `[["a"], "…"]`, `[{"k": 1}, "…"]`, `[null, "…"]`, pretty-printed array and object, a many-member object, an object with an array member,
  and the known `Use {"note": "[WANT]…[/WANT]"} for it`. 18 unit cases pin the walk itself (`_T4_WALK_CASES`, plain and memoised forms).
- **Behaviour changes vs turn 3, both toward base and stated:** `She said "a", "b [WANT]x[/WANT]", and left.` and `3,"[WANT]x[/WANT]",done` (le-016's rows) now MINT; the turn-3 test that pinned `ok: true,` as a
  residual is replaced by `test_t3_real_json_tokens_inside_a_container_still_count` (real containers reject; the prose/YAML-ish form mints like base).

### F1 (LOW-MEDIUM) — cost: the `? # = &` branch is linear
- `_want_url_continues_after_pair` ran `content.find(WANT_CLOSE, opener_end)` per opener (O(openers × node) under `_cc_mutation_lock`). `parse_wants` now computes the sorted list of closer offsets **once per node** and the
  helper does a `bisect_left` (same semantics: nearest `[/WANT]` at or after the opener's end, then one character of lookahead).
- **Timings, same input on the turn-3 function (`01e0065`) and this one, this laptop, results identical** (`(https://x.org/a?[WANT]w ) × N + [/WANT]-x`, every opener reaches the `?` branch):

| N openers | node size | turn 3 (`01e0065`) | turn 4 |
|---:|---:|---:|---:|
| 5,000 | 117 KB | 0.37 s | 0.13 s |
| 20,000 | 469 KB | 4.19 s | 0.56 s |
| 40,000 | 938 KB | **18.50 s** | **1.33 s** |

  (le-019 measured 16.3 s for the turn-3 function at 40,000; same quadratic shape.) Other adversarial shapes on the new function: 40,000 × `"a", ` 0.53 s; 40,000 × `["a", ` 0.30 s; 100,000 × `1, ` 0.36 s;
  50,000 × `{}, ` 0.25 s; 200,000 quotes 0.49 s. Tests: `test_t4_f1_*` (timing bounds, the bisect unit cases, and exact behaviour at N = 5,000 / 20,000 / 40,000) and `test_t4_json_and_link_scanners_stay_bounded_*`.

### F3 (LOW-MEDIUM) — a nested mention pair stays inside the real want
- **What was wrong (pre-existing, le-019):** an opener masked by an OPENER-ONLY reason (`in_url`, `in_link_target`, `quoted`, `escaped`, `code_adjacent`, JSON-escaped hug) INSIDE a real want kept no counter, so that mention's own
  closer ended the real want: `[WANT]see https://x.org/[WANT]z[/WANT] ok[/WANT]` minted the truncated, marker-bearing `see https://x.org/[WANT]z` permanently (base mints nothing).
- **The rule now (`mention_stack` in `parse_wants`):** a mention opener that arrives while a real opener is pending is pushed; the next closer pairs with the **nearest** opener (the mention one) — the nearest-opener rule the
  function already used — and only then does a closer close the real want. The consumed closer is skipped with the mention opener's reason. A new live opener clears the stack.
- **What it produces, and why (as the brief asks to state):** a **mention pair** inside a real want → **the WHOLE want** is minted (the pair is text inside it), consistent with region mention pairs since turn 3; base minted nothing
  (its inner text holds a marker) — a deliberate P406 delta. An **unpaired** mention opener → its closer pairs with it, the real want is unclosed → **nothing minted, exactly like base** (and a visible `opener_unclosed` skip).
  Stated plainly: a want that contains ONE bare escaped/quoted marker mention and nothing else is not minted (same as base); previously turn 3 minted a truncated marker-bearing fragment of it.
- **Goldens:** the four shapes in the brief (URL-carried, link-carried, quoted, escaped mention pairs) and a region pair (contrast) all mint the whole want, `base == []`; two escaped mention pairs mint whole (base minted the garbage
  fragment `d\`); three unpaired cases (`use \[WANT] to start`, `stop writing "[WANT]" by hand`, unpaired + a second want → `["second"]`) `== base`; mention openers OUTSIDE a pending want unchanged.

### F4 — the complete named-residual list (section 3) and F5 — doc
- Residuals (a) previous body's region masks a later opener, (b) `word[docs](…)`, (c) `)/[WANT]` after a link, and "JSON container with an element longer than the window" are each **pinned by a test at CURRENT behaviour**.
- F5: the turn-2 (#815) changelog entry now says "SUPERSEDED BY TURN 3" beside the old terminator list; a turn-4 entry states What/Why/How (asserted by a test).

## 2. The recognition rules AFTER this change (complete, for the #801 plan and the final delta reviewer)

A want = the `.strip()`ped text between a **live opener** and its **paired closer**, no length limit, id `cc:want::`+sha1(text)[:16]. Markers: `\[(/?)WANT\]`, case-sensitive.
- **Regions (any marker):** *JSON string literal* (unescaped `"` in REAL JSON opener context per F2; same line; closed before `, } ] :` or end), *fenced block* (CommonMark; unclosed → to end of node), *inline code span* (CommonMark pairing per paragraph). JSON is tested first.
- **Openers** inside a region → `in_json_string` / `in_fence` / `in_code_span`. **Closers** in a region: mention iff they close a masked opener of the SAME region, or no live opener is pending; else the closer of the real want.
- **Opener-only guesses, in order:** `in_link_target` (glued to `](…` of a real link) → `in_url` (glued to `scheme://…` with `/`, or `? # = &` when URL characters continue after the paired closing tag) → JSON-escaped-quote hug (`\"[WANT]\"`, `in_json_string`)
  → `code_adjacent` → `escaped` → `quoted`. A mention opener arriving while a real opener is pending is **nested** (F3).
- **Pairing:** nearest opener wins — mention openers included (`mention_stack`) — with `opener_unclosed`, `closer_without_opener`, `empty_pair`. No closer is judged by a guess on its own surroundings.

## 3. Named residuals — COMPLETE and CURRENT (after this change)

**Still DROPS a well-formed want (every case logged at INFO with its reason; re-mintable while the source node lives):**
1. *F4a — a previous want's OWN body starts a region:* an unpaired backtick run, a fence-looking line (`~~~`, ```), or a JSON-literal opener in an earlier want's body masks the openers of LATER wants in the same paragraph/node
   (`in_code_span` / `in_fence` / `in_json_string`). The FIRST want never diverges from base (le-019: 0 of 5,828). Includes a want whose text ends with a fence-closing line glued to its closer (the fence never closes).
2. An opener inside a stray-backtick span in the same paragraph (le-014 C3 first shape; CommonMark-faithful).
3. A want typed inside REAL JSON (`{"note": "[WANT]…"}`, a container's string element) and the quote-colon-quote shape `the "task": "[WANT]…[/WANT]"` (the `"key":` branch is unchanged and has no container requirement; base mints it).
4. A JSON container whose element is longer than the 512-character string window (or nested container longer than 256) is not walked — fail-open, it mints.
5. An opener glued to a URL path `/` (incl. `See [t](https://x.org/a)/[WANT]…`, F4c), or inside a URL token via `? # = &`; `arr[0](https://…/[WANT]…` (`in_url`); an opener in a real link destination.
6. An opener right after a backtick / after an odd backslash / hugged by a quote pair (`"[WANT]"`); a later want in a node after an unclosed fence.
7. A want that contains exactly ONE unpaired escaped/quoted marker mention: nothing minted (= base; F3).
**Still MINTS a mention (not caught; frozen-list material for the #801 repair):** a bare unquoted prose pair; single-quoted JSON-ish; JSON with raw newlines; a whole-content string without JSON context; scheme-less reference definitions; link TEXT
containing a pair; `www.` / `mailto:` / `data:`; HTML comments and `<code>`; YAML/TOML strings; bold/emphasis around the tokens; blockquotes; 4-space/tab-indented code; `word[docs](./d/[WANT]…)` (F4b: a real link glued to a word fails the `[`-neighbour rule).
No semantic exemption exists or was added. **Carried debt (#755):** the host twin `surface_wants_for_graph` and Syl's `_surface_wants` still use the unbounded, legitimacy-free regex and a second id scheme (`want::`); punchlist filing and the vault
Context-Map/wikilink sync are the zone manager's (`handoffs/` rides this code-repo branch).

## 4. Evidence

Command (worktree root, `NG_EMBED_*` unset — `NG_EMBED_REMOTE` was the only name set): `env -u NG_EMBED_REMOTE PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_cc_want_legitimacy_810.py tests/test_cc_want_bounds.py -s -v -p no:cacheprovider`
- **Runs — two, not one (stated plainly):** run 1 = `287 passed, 1 failed` — the failure was MY test's expectation (`test_t4_f1_url_query_glue_scan…` asserted N skipped where the shape yields N−1, because with the closer at the end of the node the last pair mints; the parser was
  correct). I fixed the test (a URL character now follows the closer so every opener reaches `in_url`; exact counts asserted) and re-ran. **Run 2 (final) = `288 passed in 10.49 s`** — 280 in `test_cc_want_legitimacy_810.py` (was 209; +71), 8 in `test_cc_want_bounds.py`. No skip.
- **P379 preamble (printed by the run):** `cc_ng_organism in sys.modules /home/josh/NeuroGraph-worktrees/z12-want-legitimacy-810-20260930/cc_ng_organism.py`; `neurograph_rpc`, `neuro_foundation`, `ng_lite`, `ng_embed`, `ng_ecosystem`, `ng_tract_bridge`, `ng_autonomic`, `openclaw_adapter`, `surface_resolver`,
  `surfacing`, `cc_ng_host` all `not loaded`; `NG_EMBED_* names set in env: none`. The file fails itself if `cc_ng_organism` is not the worktree copy.
- **Parity corpora (asserted in the run, against base):** 8,400 shapes (25 prefixes × 28 bodies × 12 suffixes) and the new 576 prose-comma-quote shapes: **0 divergences** in both; le-014 goldens, the turn-3 closer/URL/token/link goldens and every earlier delta table still hold.
- **Disclosed non-pytest probes (mine, scratch `/tmp/z12-810-t2/check5-7.py`, `fuzz4.py`, `fuzz5.py`; pure strings and fake in-memory graphs; no graph load, checkpoint, tract, daemon or embed):** used to choose the rules, to verify every table's premise
  against base before each run, and to take the timings. Two generators, like le-019's method: a 60-token soup with structured JSON-ish tokens (seeds 2 and 3, 80,000 trials each) — every divergent class carries an opener-context or region reason
  (`in_fence`, `in_url`, `escaped`, `in_code_span`, `quoted`, `in_json_string` from the `\"[WANT]\"` hug, `in_link_target`) and I read the shortest example of each top class; and a plain-prose-opener generator (prefixes of words/years/`true,`/`null,`/numbers + comma, adversarial bodies and JSON-closer-looking suffixes; seeds 11 and 12,
  150,000 trials each) — **every divergence (1,577 and 1,521) carries the `quoted` reason**: the generator builds a `"[WANT]"` token hug whenever the body starts with `"`; no `in_json_string`-only class remains. These are probes, not proof over all strings,
  and not evidence of the pytest result.
- Not run (instruction): the full suite. Not verified: behaviour under the live daemon (nothing restarted); prune lifetime of conversational source nodes; `ng_embed` on very long want text.

## 5. For the final delta reviewer and the #801 owner
1. **Pin** the recount and the repair plan's function pin to section 0 (commit + blob + sha256). Take the per-reason histogram over the source corpus against the turn-1/turn-3 functions, and hand-review every node where `in_url` / `in_json_string` / `in_link_target` fired,
   plus (le-019's interim guard) every minted want whose text contains `[WANT]` or `[/WANT]` — after F3 that is exactly the set of nested mention pairs minted whole.
2. **Two design choices to check:** (i) F2 requires a REAL container, so a bare JSON fragment pasted WITHOUT its `[`/`{` (`"a", "[WANT]x[/WANT]"`) now mints (mention missed) — chosen because the opposite error silently drops prose wants (le-019's finding); (ii) F3 keeps a
   mention pair WHOLE inside a real want (rather than minting nothing), matching how region pairs already behave, and an unpaired mention yields nothing exactly as base.
3. Per the Chief's stopping rule: if the final delta finds only the NAMED residual classes in section 3 (nothing that drops a well-formed plain-prose want), this pin is frozen and the rest goes to the #801 frozen-list review.
