# checker-018 ROLE A — #810 TURN-2 DELTA (cross-family)

STATUS: COMPLETE

Lane: `want-parser-legitimacy-810`
Dispatch: #10868
Reviewer: checker-018 (grok-4.6, T3 Code, report_only)
Role: ROLE A only (DELTA code review). ROLE B is a separate turn.
Scope: DELTA `e462c17a80a8c14797a61d496b32f241c83befb4` → `8864c8caf3979bfd95548d92f5d83555a637051c` (commits `6e3376e2` le-014 C1-C4+LAW 5, `2a2703f5` #815, return `8864c8c`). Exec P414: this pair is the delta only.
Worktree: `/home/josh/NeuroGraph-worktrees/z12-want-legitimacy-810-20260930`
Branch: `cc-laptop-want-legitimacy-810-20260930`
Packet: `docs/.claude/worktrees/daemon-recall-756-20260930/handoffs/z12-silent-failure-lanes-20260930/assignments/review-packet-810-delta.md`
This review stub: `f62bce6c2603b5d9d742679a9d05cc60735691b8`
Reviewed code+return tip: `8864c8caf3979bfd95548d92f5d83555a637051c`
Base for parity: `e4ebf982b1989fd9066d610b94853bc68bf70d37`

Hashes (`sha256sum` in the worktree, `git rev-parse` for commits):
- `cc_ng_organism.py` `5c55582f7422e27db50ea53ce7e1bb6df45937b065617c30d1b8f9c1bc0bdeb0`
- `tests/test_cc_want_legitimacy_810.py` `7dca3908791c4d87c4eb52517c160d7cb430ed07547e8831b399bc81bd548763`
- `tests/test_cc_want_bounds.py` `3b8305172ff7bfa26dac72c3a87abb88580742eb6e0ff2366dc69745f77f7289` (unchanged vs `e462c17`; same hash checker-016 recorded)
- `handoffs/z12-want-legitimacy-810/returns/build-002.md` `1c0688e53e770d5209b8ac606717e0077455965340b81a9265a449ed7db1c6fd`
- le-014 `56696110f75a2b9f1c5478f9f0b38ed645971dfaa2dd1a46f6f66ecd36def928`
- checker-016 `a6ad6c3a369001d546b9a978ff5e6907dc386fb190c27e04fdd87459d223246b`

Code files in the delta (`git diff --name-only e462c17 8864c8c` restricted to the packet's three paths): `cc_ng_organism.py`, `tests/test_cc_want_legitimacy_810.py`. `tests/test_cc_want_bounds.py` has an empty diff against `e462c17` and was still run as instructed.

Nothing merged, settled, dispatched, restarted, or loaded from a live graph/checkpoint/tract. Secrets by NAME only. Scratch probes under `/tmp/checker018-810-delta-probe.py` only.

## A1 C1/C2 at the source and parity vs base

**Verdict: PASS**

`_want_marker_mention_reason` (`cc_ng_organism.py:1817`) takes `is_close`. Tier 1 (JSON string literal / fence / balanced code span) still masks any marker. At `:1843-1844` a closer that survives tier 1 returns `None` and pairs. Tier 3 adjacency guesses run on openers only: `_want_opener_carried_reason` (`in_link_target` / `in_url`, `:1803`), JSON-escaped hug `\"[WANT]\"` (`:1848-1849`), `code_adjacent` (`:1850-1851`), odd-backslash `escaped` (`:1852-1856`), token-hugging `quoted` (`:1857-1860`). That is the le-014 C1/C2 fix at the source (LAW 4): base `e4ebf982` guarded only the opener (`content[m.start()-1] == "\`` on `_WANT_RE` match start).

Targeted tests run **once** from the worktree:

```
cd /home/josh/NeuroGraph-worktrees/z12-want-legitimacy-810-20260930
env -u NG_EMBED_REMOTE -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_cc_want_legitimacy_810.py tests/test_cc_want_bounds.py -s -q -p no:cacheprovider
```

Result: **156 passed in 4.99s** (python 3.12.3). Parent shell had `NG_EMBED_REMOTE=hf` and `PYTHONPATH=/home/josh/NeuroGraph:`; both stripped for the run. The file fails itself if `cc_ng_organism` is not the worktree copy.

P379 preamble (printed by the module-scoped autouse fixture):

```
P379 cc_ng_organism     in sys.modules /home/josh/NeuroGraph-worktrees/z12-want-legitimacy-810-20260930/cc_ng_organism.py
P379 neurograph_rpc     not loaded     -
P379 neuro_foundation   not loaded     -
P379 ng_lite            not loaded     -
P379 ng_embed           not loaded     -
P379 ng_ecosystem       not loaded     -
P379 ng_tract_bridge    not loaded     -
P379 ng_autonomic       not loaded     -
P379 openclaw_adapter   not loaded     -
P379 surface_resolver   not loaded     -
P379 surfacing          not loaded     -
P379 cc_ng_host         not loaded     -
P379 NG_EMBED_* names set in env: none
```

Pre-test (no NG import): python `/usr/bin/python3` 3.12.3; NG-related `sys.modules` was NONE; `NG_EMBED_REMOTE` was set in the parent and stripped.

Own parity probe (`/tmp/checker018-810-delta-probe.py`): `sys.path[0]` pinned to the worktree, `PYTHONPATH` and `NG_EMBED_*` unset, fail-closed if `cc_ng_organism.__file__` is `/home/josh/NeuroGraph/cc_ng_organism.py`. Base loaded via `git show e4ebf982:cc_ng_organism.py` under a private module name. Compared `surface_wants` on fake in-memory graphs (same shape the worker tests use) plus `parse_wants` on the new function. Shapes **outside** the worker's 18×23×9 product (different code tokens, host, unicode, wording):

| Input class | base | new / parse_wants |
|---|---|---|
| le-014 C1 `I noticed it. [WANT]check \`foo()\`[/WANT] done.` | `check \`foo()\`` | equal |
| le-014 C1 `[WANT]fix \`a\`[/WANT] and later [WANT]rest[/WANT]` | both | equal |
| le-014 C1 `[WANT]do \`x\`[/WANT] tail [WANT]second[/WANT]` | both | equal |
| ends in different inline code `inspect \`parse_wants()\`` | minted | equal |
| begins with `\`cc_ng_host.py\`` | minted | equal |
| both `\`open()\` then \`close()\`` | minted | equal |
| code-ending want + follow-on second want (blank line) | both | equal |
| C2 `x "[WANT]I want "x"[/WANT]" y` | `I want "x"` | equal |
| want ending in `https://neurograph.local/handbook#wants` | minted | equal |
| want containing `{"alpha": 1, "beta": [2, 3]}` | minted | equal |
| CRLF inner `line one\r\nline two` | kept | equal |
| tabs `\tkeep\t\tthis  spacing intact` | kept | equal |
| unicode `αβγ 日本語 — café` | minted | equal |
| code-ending want then URL-containing second want | both | equal |

0 divergences. Opener guards still fire: ``see `x`[WANT]real[/WANT]`` empty; `"[WANT]" a "[/WANT]"` empty; `\[WANT]a[/WANT]` empty. Structural masks still apply to closers (`[WANT] replace \`[/WANT]\` tokens [/WANT]` keeps the inner masked closer as text — worker test, read).

C1/C2 are closed at the source. The #801 id-equality property holds on every well-formed shape run here.

## A2 #815 recognition rules

**Verdict: PASS-WITH-NOTES**

Rules as built (`build-002.md` §2 and `cc_ng_organism.py:1709-1861`):

1. **`in_json_string` (any marker, first).** Marker inside a structurally valid JSON string literal (`_want_json_string_ranges` `:1751`): unescaped `"` whose previous non-space char is `{` / `[`, or `:` preceded by `"`, or `,` preceded by a finished JSON value (`"` / digit / `]` / `}` / suffix `true`/`false`/`null`); closed by the next unescaped `"` on the same line; closer context `,` `}` `]` `:` or end (`_WANT_JSON_AFTER_RE` `:1714`). Plus opener-only hug `\"[WANT]\"` (`:1848-1849`).
2. **`in_link_target` (opener only, before URL).** Glued run before the opener contains `](` with no `)` after it (`:1809-1811`).
3. **`in_url` (opener only).** Glued run (max 2048 chars, `:1716`) contains `scheme://` and the char immediately before the opener is outside `_WANT_URL_TERMINATORS` `).]}>\"'\`,;!.` (`:1718`, `:1812-1813`).

Checker-016's five examples plus the backtick-fence variant, own run of `parse_wants`:

| Input | opener reason | closer reason | wants |
|---|---|---|---|
| `{"cmd":"[WANT] not a want [/WANT]"}` | `in_json_string` | `in_json_string` | none |
| `\"[WANT]\" then later \"[/WANT]\"` | `in_json_string` | `closer_without_opener` | none |
| `{"code": "~~~\n[WANT] documented [/WANT]\n~~~"}` (literal `\n`) | `in_json_string` | `in_json_string` | none |
| `{"code": "```\n[WANT] documented [/WANT]\n```"}` (literal `\n`) | `in_json_string` | `in_json_string` | none |
| `https://example.com/path/[WANT]secret-want[/WANT]/docs` | `in_url` | `closer_without_opener` | none |
| `[the docs](https://example.com/[WANT]linked[/WANT])` | `in_link_target` | `closer_without_opener` | none |

Base `e4ebf982` minted a want from every one of those six. The backtick-fence coincidence checker-016 noted is real (`_want_code_span_ranges` pairs the ``` runs) and the skip reason is still `in_json_string` (JSON is checked first, `:1834-1836`).

**Boundary (opener context, want text ignored):** each of these minted whole, skip list empty unless noted: want text containing a URL; containing JSON; containing a markdown link; containing a quoted word; URL then a space then a want; URL then a sentence-dot glued to the opener; parenthesised URL glued; markdown list with a link before (space and newline); `[t](https://x.org/a)[WANT]follow up[/WANT]`; `(see [WANT]real want[/WANT])`; `He typed \"go\" and then [WANT]follow up[/WANT]`; Choice-Clause-shaped want after a URL with a space; same after `{"ok": true}` with a space. A genuine >600-char want in plain prose beside each of the six examples (before/after × space/new-paragraph = 24 arrangements) minted that long text whole with the example still skipped.

**False positives the new reasons introduce (genuine want skipped):**

| Input | Result | Class |
|---|---|---|
| `Use {"note": "[WANT] revisit X [/WANT]"} for it` | both markers `in_json_string` | documented known FN (rule A) |
| `"a", "b [WANT]x[/WANT]"` | both `in_json_string` | documented (quoted list matching opener/closer context) |
| `The claim is untrue, "[WANT]revisit this[/WANT]"` | both `in_json_string` | **extra FN** — `:1745-1747` `content.endswith("true", 0, j+1)` matches the suffix of `untrue` with no token boundary. Same for `nonnull, "[WANT]check it[/WANT]"` via `endswith("null", …)` |
| `https://x.org/a?[WANT]secret[/WANT]` | opener `in_url` | documented (`?` is outside the terminator set) |
| `https://x.org/a#[WANT]secret[/WANT]` | opener `in_url` | same family (`#` outside the terminator set) |
| `arr[0](https://x.org/[WANT]x[/WANT])` | opener `in_link_target` | **extra FN** — `:1809-1811` `rfind("](")` treats `0](` as a markdown destination |
| `[id]: https://x.org/dir/[WANT]x[/WANT]` | opener `in_url` | scheme-bearing reference definition; the pinned "not recognised" case is the scheme-less sibling |
| `<https://example.com/[WANT]secret[/WANT]>` | opener `in_url` | autolink caught by the URL rule (no dedicated autolink recogniser; scheme:// is enough) |
| `<a href="https://example.com/[WANT]secret[/WANT]">` | opener `in_url` | same |

`The claim is untrue "[WANT]…[/WANT]"` (comma absent) stays a real want — the suffix check only fires in comma / `{` / `[` / `"key":` opener context. Visible: each skip is `in_json_string` / `in_url` / `in_link_target` at INFO with node/offset/reason. Recoverable: the source conversational node is untouched.

**False negatives left (mentions that still mint):**

| Input | minted |
|---|---|
| `{'cmd': '[WANT] x [/WANT]'}` (single-quoted) | `x` |
| `{"cmd": "[WANT] x\n [/WANT]"}` (raw newline) | `x` |
| `"[WANT] x [/WANT]"` at content start (no JSON opener context) | `x` |
| `[id]: ./dir/[WANT]x[/WANT]` (scheme-less refdef) | `x` |
| `[[WANT]text[/WANT]](https://x.org)` (link TEXT) | `text` |
| `www.example.com/path/[WANT]secret[/WANT]/docs` | `secret` |
| `mailto:dev@[WANT]secret[/WANT].example` | `secret` |
| `data:text/plain,[WANT]secret[/WANT]` | `secret` |
| `use [WANT] to mark one, and [/WANT] closes it` | `to mark one, and` |
| `<!-- [WANT] not a want [/WANT] -->` | `not a want` |
| `<code>[WANT] not a want [/WANT]</code>` | `not a want` |
| YAML `cmd: '[WANT] x [/WANT]'` / TOML `cmd = "[WANT] x [/WANT]"` | `x` |

These match the worker's "NOT recognised" list plus the pre-existing HTML/bare-mention residuals from checker-016. Frozen-list material for the #801 repair; none reintroduces a length heuristic.

`code_adjacent` on openers is kept (P414). A closer immediately after a backtick is a real closer (A1).

## A3 Performance / robustness of the new scanners

**Verdict: PASS**

`_want_json_string_ranges` (`:1751`) walks `content.find('"')` with `pos = q+1` / `pos = i+1`; `_want_json_opens_literal` (`:1721`) looks at the previous one or two non-space characters (constant work per quote). `_want_glued_run_before` (`:1789`) looks back at most 2048 characters per marker and runs `_WANT_MARKER_RE` on that slice. No recursion. `_WANT_JSON_AFTER_RE` / `_WANT_URL_SCHEME_RE` are single-pass with no nested overlapping quantifiers.

Own timings on `parse_wants` (worktree copy, `/tmp` probe):

| Input | dt | result |
|---|---|---|
| 200,000 quotes then `[WANT] x [/WANT]` | 0.417 s | 1 want |
| 20,000 `{"a":"` prefixes then a pair | 0.251 s | 1 want |
| 100,000 `a` then a pair | 0.029 s | 1 want |
| 5,000 × `https://x.org/` then a pair | 0.008 s | opener `in_url` (2 skips) |
| ~1 MB prose, 2,000 real pairs | 0.163 s | 2000 wants |
| 200 kB JSON blob carrying a pair | 0.085 s | 2 skips (`in_json_string`) |
| 50,000 escaped quotes inside a JSON string, then a real pair | 0.023 s | 1 want |

All well under the worker's 5 s bound (`test_815_finders_are_linear_on_adversarial_input`). A 1 MB string of quotes **with no `WANT]`** returns in 0.0006 s because `parse_wants` short-circuits at `:1878`; that input does not exercise the JSON scanner. The 200 k-quote + pair case does.

Robustness (no exception): closer-first, 50 nested openers (nearest pairs, 49 `opener_unclosed`), empty pair, unclosed opener, NUL bytes around a real pair (minted `x`).

## A4 LAW 5 env knobs

**Verdict: PASS**

`_want_skip_env_int` (`:1598-1602`) sources `CC_WANT_SKIP_SUMMARY_INTERVAL_S` / `CC_WANT_SKIP_SEEN_MAX` / `CC_WANT_SKIP_DETAIL_PER_CALL_MAX` with defaults 3600 / 4096 / 50 (`:1605-1607`), `max(minimum=1, int(…))`, `ValueError`/`TypeError` → default. Own subprocess imports (sys.path[0] pinned to the worktree, `PYTHONPATH` unset, `__file__` printed):

| env | printed |
|---|---|
| unset | `3600 4096 50` |
| `120` / `7` / `3` | `120 7 3` |
| `CC_WANT_SKIP_SEEN_MAX=not-a-number` | `3600 4096 50` |
| empty string | default |
| `CC_WANT_SKIP_SEEN_MAX=-5` | `3600 1 50` (clamped to minimum 1) |
| `CC_WANT_SKIP_SUMMARY_INTERVAL_S=0` | `1 4096 50` |
| `CC_WANT_SKIP_DETAIL_PER_CALL_MAX=0` | `3600 4096 1` |
| `CC_WANT_SKIP_SEEN_MAX=3.14` | default (ValueError) |

Import never failed. In-process values with no knobs in the parent env: 3600 / 4096 / 50, equal to the old literals.

C4 flood claim as restated (`:1593-1597` and `_log_want_skips` docstring `:1933-1939`): while distinct `(node, offset, reason)` keys ≤ `WANT_SKIP_SEEN_MAX`, a static corpus costs about 1 summary line per heartbeat interval (default 1 hour) after details drain; beyond that the FIFO evicts keys that re-qualify, capped at `WANT_SKIP_DETAIL_PER_CALL_MAX` detail lines per pulse. Worker test `test_flood_claim_beyond_seen_max_degrades_to_the_per_call_cap` (monkeypatch 10 / 5, 30 stray closers) was in the 156-pass run: last pulse still 5 in the overflow regime, 0 after drain when under the cap. Wording "NEVER logs marker text" is gone from the docstring; the detail line carries the literal `[WANT]` / `[/WANT]` token as kind (`:1973-1974`).

## A5 Unchanged surfaces

**Verdict: PASS**

- `render_wants` ast source segment is byte-identical to base `e4ebf982` (own `ast.get_source_segment` comparison). `WANT_RENDER_LIMIT == 40` and `WANT_MAX_CHARS == 600` on both copies. Renderer still clamps `t[:WANT_MAX_CHARS]` (`:2061`).
- Packet code delta is `cc_ng_organism.py` + `tests/test_cc_want_legitimacy_810.py`. Empty diff against `e462c17` for `neuro_foundation.py`, the six vendored files, `ng_salience_gate.py`, `ng_updater.py`, `cc_ng_host.py`, `neurograph_rpc.py`, `surface_resolver.py`, `surfacing.py`, `tests/test_cc_want_bounds.py`.
- Pure-function contract for the #801 tool is unchanged: `parse_wants(content: str) -> WantParse`, `want_id_for_text(text: str) -> str` (`cc:want::` + sha1 utf-8 `[:16]`), frozen dataclasses `WantSpan` / `SkippedMarker` / `WantParse`. `parse_wants.__code__.co_names` contains no `open` / `print` / `logger`. Own import of the two names loaded only `cc_ng_organism` in `sys.modules`.
- `surface_wants` still calls `parse_wants` under `_cc_mutation_lock` and `_log_want_skips` after the lock (`:2011`, `:2028`).

**#801 recount must be re-taken on this tip (`8864c8c`), not on `c9fe56d` / `e462c17`.** build-002 §5 is right on every item: C1 restores wants that end in inline code (ids now equal base); the three new reasons move JSON-escaped / URL / link-carried tags from "genuine" to "skipped mention"; the "marker-free >600" set of 16 cannot be reused; id mechanics (`want_id_for_text`) are the same function over a different minted set; LEFT-UNCHANGED membership changes; recovered-want count depends on the new masks; the COPY dry-run imports these two names from this branch.

## A6 Verdict

**Overall: PASS-WITH-NOTES**

| Item | Verdict |
|---|---|
| A1 C1/C2 at source + parity vs `e4ebf982` | PASS |
| A2 #815 recognition rules | PASS-WITH-NOTES |
| A3 scanner performance / robustness | PASS |
| A4 LAW 5 env knobs + C4 flood claim | PASS |
| A5 unchanged surfaces + #801 contract | PASS |

C1 (HIGH) and C2 (MEDIUM) are fixed at the source and equal base on the le-014 repros and on well-formed shapes outside the worker matrix. #815 rejects checker-016's five examples plus the backtick-fence variant with the named reasons; the opener-context boundary holds; a long plain-prose want beside them is captured whole. LAW 5 knobs default to the old literals; junk / zero / negative / empty / float never break import. `render_wants` / `WANT_RENDER_LIMIT` are byte-identical to base.

### Numbered corrections

1. **Note — JSON `true`/`false`/`null` suffix has no token boundary.** `_want_json_opens_literal` `:1745-1747` uses `content.endswith("true"/"false"/"null", 0, j+1)` after a comma. `The claim is untrue, "[WANT]revisit this[/WANT]"` and `The pointer is nonnull, "[WANT]check it[/WANT]"` are skipped `in_json_string`. A one-character token check (previous char outside `[A-Za-z]`) would close it. Same family as the pinned JSON-looking-sentence FN; rare prose shape; visible and recoverable. Severity: note.
2. **Note — `in_link_target` fires on `name[i](scheme://…/[WANT]…)`.** `:1809-1811` `rfind("](")` has no requirement that the `[` start a markdown link. `arr[0](https://x.org/[WANT]x[/WANT])` is skipped `in_link_target`. Severity: note.
3. **Note — leftover mention mints** (single-quoted / raw-newline JSON, `www.` / `mailto:` / `data:`, HTML comment/`<code>`, YAML/TOML, bare prose pairing, link TEXT, scheme-less refdefs). Worker-stated; frozen-list for #801. Severity: note.
4. **Note — #801 derived-data recount is invalid until re-taken on `8864c8c`.** Agree with build-002 §5 in full. Severity: note (sequencing, not a parser defect).
5. **Note — `tests/test_cc_want_bounds.py` is outside the code delta** (empty diff vs `e462c17`; hash unchanged from checker-016). Ran it anyway: 8 tests in the 156. Informational.

No must-fix corrections for this delta pair. Notes 1–2 are extra false negatives of the new reasons, in the same precision-trade family P414 already accepted for JSON-looking sentences.

### Not-verified

- Live daemon / autosave pulse (nothing started or restarted; P329 merge = deploy).
- Full NeuroGraph test suite (packet: targeted files only).
- FIFO eviction at the production 4096 cap (code-read; test monkeypatches to 10).
- Two-thread interleaving of `_log_want_skips` INFO lines.
- URL lookback > 2048 characters empirically (code-read: scheme falling out of the 2048-char glued run would mint).
- `ng_embed.embed` on unbounded `want_text` (geometry-backfill owner; open from le-014).
- Exact per-node re-parse of the 09-16 182 rows (#801 COPY dry-run, after this pair).
- ROLE B (law enforcer) — separate turn; this file is ROLE A only.
- `cc_ng_host.py` / `neurograph_rpc.py` bodies (absent from the delta; not re-read).

STATUS: COMPLETE
