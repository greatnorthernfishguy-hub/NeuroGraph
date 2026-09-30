# #810 build-003 — turn 3: the FINAL parse_wants (le-016 MEDIUM #2/#3, checker-018 notes 1-2)

Lane `want-parser-legitimacy-810` · Zone manager Z12 · dispatch #10915 · worker turn 3 of the parser lane
Branch `cc-laptop-want-legitimacy-810-20260930` · parser base `e4ebf982b1989fd9066d610b94853bc68bf70d37`
Inputs reviewed: turn-2 head `545995a59c15993d9a57a3c97965b9cf5716a3ec` (code-identical to `ab2c29d1`/`8864c8c`), `le-016-810-delta.md`, `checker-018-810-delta.md` (both read in full)
Related: [[NeuroGraph]] · [[The Laws]] (LAW 3/4/7) · [[NeuroGraph Is a Mind, Not a Database]] · turns 1-2: `build-001.md`, `build-002.md`

> **Status: RETURNED.** Not self-accepted; nothing merged, wired or restarted. `render_wants` is UNTOUCHED (source segment
> byte-identical to base, asserted in the run). No protected, vendored, `cc_ng_host.py`, `neurograph_rpc.py`,
> `surface_resolver.py` or `surfacing.py` file is in the branch diff. The INFO skip log is unchanged and no content-aware
> exemption was added.

## 0. THE FINAL FUNCTION PIN — the #801 counts and the repair plan's pin are taken ONLY on this

| What | Value |
|---|---|
| Code commit (parser + tests, ONE diff) | `01e006588c0982726d6c8cd31e29ca0e1e1e57c5` |
| Defining file | `cc_ng_organism.py` (`parse_wants` + `want_id_for_text` + helpers) |
| Blob id (`git rev-parse 01e0065:cc_ng_organism.py`) | `66169c851accfff8c15bb9b00c9c6f8c87a4d4f1` |
| **sha256 of the file** | `42745f5dad8f346b2b759017cf3c08b6b967dc190166ba1c9b944859d2480781` |
| Test file sha256 (`tests/test_cc_want_legitimacy_810.py`) | `986bb514e0c7dcab58b724393b542411b2123941643bcda021bb8ef293f0ea60` |
| Return commit (this file; `cc_ng_organism.py` unchanged by it) | the branch tip — `git rev-parse HEAD` after this commit |

Any later change to `cc_ng_organism.py` re-opens the recount; the two hashes above let a reviewer or the repair tool verify it
is running exactly this function (`git show 01e0065:cc_ng_organism.py | sha256sum`).

## 1. The five fixes — each with its golden cases (asserted against BASE `e4ebf982`; a missing base FAILS, never skips)

### Fix 1 (MEDIUM #3) — the CLOSER rule, at the source (LAW 4)
- **What was wrong:** tier 1 applied `in_json_string` (and fence / code span) to ANY marker, so a closer inside a quote fragment
  that merely looked like a JSON literal was masked even when its opener was plain prose: `[WANT]rename "a", "b[/WANT]", next`
  (base mints `rename "a", "b`) lost the whole want, opener included — a closer judged by a lexical guess, the C1 class.
- **The rule now** (`parse_wants`, `_want_marker_region`): a JSON literal / fence / inline code span is a *region*. A live
  opener is never inside a region (masked openers are skipped first), so:
  - a masked **opener** in a region → mention; the region's count of unclosed masked openers goes up;
  - a **closer** in a region that closes a masked opener **of the same region** → mention (the pair is discussion);
  - a **closer** in a region with **no live opener pending** → mention (a stray closer, region reason);
  - a **closer** in a region with a **live opener pending** and no mention opener in that region → **the closer of the real want**,
    whatever surrounds it. No lexical guess on the closer's own text remains.
- **Audit (brief: every other place a closer is judged by a heuristic):** `quoted`, `escaped`, `code_adjacent`, `in_url`,
  `in_link_target` and the JSON-escaped hug were already OPENER-only (`_want_opener_mention_reason`); the region rule above was
  the only closer judgement left, and it now depends on the opener. `test_t3_no_closer_is_judged_by_a_guess_on_its_own_surroundings`
  proves each of the six guesses both ways (a closer in that exact context pairs; the same context on an opener is still a mention).
- **Golden cases (14, each `== base` and `== explicit text`):** le-016's `rename "a", "b`; fuzz-derived fragments (`set "x": "y`,
  `list 1, "b`, `use {"a": "b`, `pick ["a", "b`, `use "a","b`); a closer in a FENCE and in a CODE SPAN with the real opener outside;
  a closer in a stray-backtick span (le-014 C3 second shape — **this residual is now removed**); a closer wrapped in quotes; after an odd
  backslash; glued to a URL path; glued to a link destination; right after inline code.
- **A behaviour change you should know about (deliberate, base-parity):** a mention closer typed inside code within a real want,
  with no mention opener in that code, now ends the want there, exactly as base did: `[WANT] replace `[/WANT]` tokens [/WANT]` →
  `replace `` (+ a stray closer skip). Turns 1-2 kept it whole by masking the closer. A mention PAIR inside one region is still
  masked (a want may discuss markup): `[WANT]see `[WANT]z[/WANT]` ok[/WANT]`, the fenced and JSON equivalents → the whole want.
  Two turn-2 test expectations flipped for this reason (`test_structural_masks_*`, the `_815_BOUNDARY` closer-in-JSON entry) and are
  rewritten with the reason stated.

### Fix 2 (MEDIUM #2) — URL glue is an allowlist, not a guessed blocklist
- **What was wrong:** `_WANT_URL_TERMINATORS` listed punctuation that ENDS a URL; every character not on the list (`: — – * _ ~ ? #`,
  a literal backslash-n …) let a real opener glued after a URL be dropped as `in_url`, where base minted it.
- **The rule now:** `in_url` fires only when the opener is glued (no whitespace, no other WANT marker between) to a run containing
  `scheme://` **and the character right before the opener is URL-internal glue**:
  - `/` (`_WANT_URL_PATH_GLUE`) — **always** (the opener sits in a URL path; checker-016's shape `https://example.com/path/[WANT]secret-want[/WANT]/docs` stays rejected);
  - `? # = &` (`_WANT_URL_QUERY_GLUE`) — **only when the marker is INSIDE the URL token**, i.e. URL characters
    (`[A-Za-z0-9/_~%&=+#@-]`) **continue immediately after the paired closing tag** (`_want_url_continues_after_pair`);
  - **anything else mints** — `: - – — * _ ~ . , ) ] } > " ' ! ;`, a letter, a literal backslash-n. `_WANT_URL_TERMINATORS` is deleted.
- **The `?` / `#` decision (asserted both ways):** `https://x.org/a?[WANT]secret[/WANT]&p=1` (and `#…section`, `q=`, `&`) → the closer is followed by URL
  characters → the marker is inside the token → `in_url`. `https://x.org/a?[WANT]follow up[/WANT]` ending the node, followed by a
  space, or by a sentence dot → the want is typed after the URL → **mints like base**.
- **Golden cases (15 mint-again, each `== base`):** colon, em dash, en dash, bold URL, italic URL, tilde, literal backslash-n, sentence dot,
  closing paren, comma, hyphen, and `?`/`#` with the want AFTER the URL (end / space / dot). **6 reject cases** pinned as deliberate deltas
  (base minted them): slash (checker-016's example and at end of node), `?`/`#`/`=`/`&` inside the token.

### Fix 3 — token boundary for `true` / `false` / `null`
- `_want_json_opens_literal` now requires a whole token: the character before the word must not be a letter, digit or `_`.
  `untrue,` `nonnull,` `intrue,` `notfalse,` `my_true,` `2null,` no longer open a JSON literal; all six mint like base (goldens).
- **The stronger variant le-016 found, stated exactly:** with a REAL token — `ok: true, "[WANT]revisit this[/WANT]"` (also `false`, `null`) —
  the text genuinely has the shape *finished JSON value, comma, opening quote*, so it is classed `in_json_string` and the want is dropped
  (base minted it). That is the same accepted precision trade as a want typed inside `{"note": "..."}` (P414), **not** a bug the token check
  can fix. It is **pinned as a named residual** (`test_t3_a_real_json_token_still_counts_named_residual`), so any change is a visible decision.

### Fix 4 — `in_link_target` needs a real link (the `[`)
- The `](` must close the text of a real markdown link: `_want_link_text_opens_a_link` scans back (≤ 1,024 characters, not past a blank line, backslash-escaped
  brackets ignored) for the balanced `[`, and that `[` must not be glued to an identifier / `)` / `]` (so `arr[0](`, `f(x)[0](`, `a[b][c](` are index/call shapes).
- **Both ways, tested:** NOT a link target and mints like base — `arr[0](./dir/[WANT]x[/WANT])`, `f(x)[0](...)`, `foo_bar[i](...)`, `a[b][c](...)`, a `](` with no `[`
  in the paragraph, link text broken by a blank line. STILL `in_link_target` — a plain link, a link with spaces in its text, a bold link, an image, nested brackets in the
  text, a link after punctuation (all base-minted; deliberate deltas).
- **checker-018's exact example, stated precisely:** `arr[0](https://x.org/[WANT]x[/WANT])` is **not a link target any more** — but the opener is glued to a URL path
  (`https://x.org/`), so the URL rule still rejects it, now reported as `in_url`. The scheme-less sibling mints. Pinned in `test_t3_index_then_a_url_destination_...`.

### Fix 5 — the changelog sentence and the notes
- The `cc_ng_organism.py` header claim "the SAME result as base for every well-formed want shape" is qualified in place to *the tested grammar … NOT every string*, naming le-016's
  counterexamples; a new turn-3 entry states What/Why/How. (`test_t3_changelog_claim_is_qualified_to_the_tested_grammar`.)
- **Parity claim, exactly:** `test_parity_with_base_on_every_well_formed_shape` now covers **25 prefixes × 28 bodies × 12 suffixes = 8,400 shapes, 0 divergences** (widened with
  glue characters after URLs, index/call shapes, want text ending inside a quote fragment / JSON-looking literal / a properly closed fence, closers followed by `", next` `"}` `",`).
  It is a constructive proof over that grammar, not over all strings.
- **Adversarial parse cost (le-016 LOW):** I could **not reproduce 3.05 s**. On the final function (scratch, this laptop): 811 KB with 5,000 × (150-character URL run + glued opener) =
  **0.55 s** (rejected path) / 0.67 s (mint path); 3,000 real links 0.30 s; 3,000 index shapes 0.36 s; 1.2 MB of prose with 20,000 pairs 0.99 s; 50,000 `[` then a destination 0.01 s.
  No change made (none needed). The new link lookback is bounded (1,024 characters) and only runs when a glued run contains `](` with no `)` after it.
- **#755 debt carried (not this lane's to fix):** the host twin `surface_wants_for_graph` (`cc_ng_organism.py`, called at `cc_ng_host.py:698-700`) and Syl's `_surface_wants`
  (`neurograph_rpc.py`) still use the unbounded, legitimacy-free regex and a second id scheme (`want::`); the definition of "a want" in this file now diverges from the twin by
  eleven skip reasons (the whole classifier). It must be on the punchlist as a concrete statement; the punchlist and the vault Context-Map/wikilink sync are the zone manager's (handoffs/ lives in this repo's branch).

## 2. The recognition rules AFTER this change (complete, for the #801 plan and the delta reviewer)

A want = the `.strip()`ped text between a **live opener** and its **paired closer**, no length limit, id `cc:want::`+sha1(text)[:16]. Markers are found by `\[(/?)WANT\]` (case-sensitive).

**Regions (any marker):** *JSON string literal* (unescaped `"` after `{` `[` `"key":` or a finished value + `,`; same line; whole-token `true/false/null`; closed before `, } ] :` or end), *fenced block*
(CommonMark, unclosed → to end of node), *inline code span* (CommonMark pairing per paragraph). JSON is tested first.
**Openers** in a region → `in_json_string` / `in_fence` / `in_code_span`. **Closers** per the Fix-1 rule.
**Opener-only guesses, in order:** `in_link_target` (glued to `](…` of a real link) → `in_url` (glued to `scheme://…` with `/`, or `? # = &` inside the token) → JSON-escaped-quote hug (`\"[WANT]\"`, `in_json_string`)
→ `code_adjacent` (backtick right before) → `escaped` (odd backslashes) → `quoted` (quote pair hugs the token). **Pairing:** nearest live opener wins (`opener_unclosed`, `closer_without_opener`, `empty_pair`).

## 3. What still mints, and what still drops (named residuals)

**Still DROPS a real want (logged at INFO, re-mintable while the source node lives):** a want typed inside a JSON-looking sentence, a quoted list, or after a REAL `true,`/`false,`/`null,`/`3,` + quote
(Fix 3); an opener inside a stray-backtick span in the same paragraph (le-014 C3 first shape; CommonMark-faithful); an opener glued to a URL path `/` or inside a URL token via `? # = &`; an opener in a real
link destination; `arr[0](https://…/[WANT]…` (in_url); an opener right after a backtick / after an odd backslash / hugged by a quote pair; **a later want in a node after an unclosed fence** — including a want whose text
ends with a fence-closing line glued to its closer (pinned: the first want now mints, the second is masked `in_fence`).
**Still MINTS a mention:** a bare unquoted prose pair; single-quoted JSON-ish; JSON with raw newlines; a whole-content string without JSON context; scheme-less reference definitions; link TEXT containing a pair;
`www.` / `mailto:` / `data:`; HTML comments and `<code>`; YAML/TOML strings; bold/emphasis around the tokens; blockquotes; 4-space/tab-indented code. These are frozen-list material for the #801 repair; no semantic exemption exists or was added.

## 4. Evidence

Command (worktree root, `NG_EMBED_*` unset — `NG_EMBED_REMOTE` was the only name set): `env -u NG_EMBED_REMOTE PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_cc_want_legitimacy_810.py tests/test_cc_want_bounds.py -s -v -p no:cacheprovider`
- **Run (once): `217 passed in 8.09s`** — 209 in `test_cc_want_legitimacy_810.py` (was 148; +61), 8 in `test_cc_want_bounds.py`. No failure, no skip. Transcript `/tmp/z12-810-t2/pytest-run-t3.txt` (local).
- **P379 preamble (printed by the run):** `cc_ng_organism in sys.modules /home/josh/NeuroGraph-worktrees/z12-want-legitimacy-810-20260930/cc_ng_organism.py`; `neurograph_rpc`, `neuro_foundation`, `ng_lite`, `ng_embed`, `ng_ecosystem`,
  `ng_tract_bridge`, `ng_autonomic`, `openclaw_adapter`, `surface_resolver`, `surfacing`, `cc_ng_host` all `not loaded`; `NG_EMBED_* names set in env: none`. The file fails itself if `cc_ng_organism` is not the worktree copy.
- **Disclosed non-pytest probes (mine, `/tmp/z12-810-t2/check3.py`, `check4.py`, `parity.py`, pure strings, fake in-memory graphs; no graph load, checkpoint, tract, daemon or embed):** used to choose the rules, to verify every test
  table's premise against base before the single run, and to take the timings in section 1. Not evidence of the pytest result.
- Base assertion mechanism unchanged: `git show e4ebf982:cc_ng_organism.py` loaded under a private module name; a missing base FAILS.
- Not run (instruction): the full suite. Not verified: behaviour under the live daemon (nothing restarted); prune lifetime of conversational source nodes.

## 5. For the delta re-check and the #801 owner
1. **Pin** the recount and the repair plan's function pin to section 0 (commit + file sha256). Take le-016's prescribed **per-reason histogram** over the source corpus and compare it with the turn-1 function's so the movement from each reason is visible,
   and hand-review every node where `in_url` / `in_json_string` / `in_link_target` fired before that skip list is trusted (the closer rule and URL allowlist shift which nodes fire).
2. **One design choice for the reviewer to check:** the closer rule pairs masked markers *per region* (a mention pair inside one region stays masked) rather than the simpler "closer with a live opener pending is always real". The simpler rule would truncate a
   want that contains a fenced/JSON/code mention pair at that pair's closer; the per-region rule keeps it whole and still equals base on every shape where no mention opener precedes in the region.
3. `handoffs/` still rides this code-repo branch; docs-side Context-Map/wikilink sync and the #755 punchlist filing remain the zone manager's.
