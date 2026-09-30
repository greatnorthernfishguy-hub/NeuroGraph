# checker-016 ROLE A — #810 PARSER-HALF (`parse_wants` structural legitimacy)

STATUS: COMPLETE

Lane: `want-parser-legitimacy-810`
Dispatch: #10777
Reviewer: checker-016 (cross-family, report_only)
Role: ROLE A only (CODE review of a branch build). ROLE B is a separate turn.
Worktree: `/home/josh/NeuroGraph-worktrees/z12-want-legitimacy-810-20260930`
Branch: `cc-laptop-want-legitimacy-810-20260930`
Packet head (code+return): `e462c17a80a8c14797a61d496b32f241c83befb4`
This review stub: `c44d91342387cb7ac9b04b001fe3f9737f4c523b`
Code commit: `c9fe56d85809c4d865fe2a4a353f9db7b172a00c`
Plan commit: `2ab4a8514199c87e8c68bc6fa5cff69b8bcfe01c`
Base: `e4ebf982b1989fd9066d610b94853bc68bf70d37`
Packet: `review-packet-810-parser.md` sha256 `56696ba40085fd2537c2e18b38e249bb813e3c59d51f4c7d57d632f9837fd758`

Hashes (`sha256sum` in the worktree):
- `cc_ng_organism.py` `1acb09e2b17448a438d8f85fb697b39e439ff2503d70d88f9d88048c2d6cc2c0`
- `tests/test_cc_want_legitimacy_810.py` `0c1f1ffa34ffce8bed448d2cb45f6e2cb81c33b1c51e9c51ab107263e015cc64`
- `tests/test_cc_want_bounds.py` `3b8305172ff7bfa26dac72c3a87abb88580742eb6e0ff2366dc69745f77f7289`
- `handoffs/z12-want-legitimacy-810/returns/build-001.md` `151e2fec1d6a7a92e24ba27dca71bab6cd257069b9767ddc42a1cfc960212e68`

Nothing merged, settled, dispatched, restarted, or loaded from a live graph/checkpoint/tract. Secrets by NAME only.

## A1 The rule as built vs P406

**Verdict: PASS-WITH-NOTES**

Read `want_id_for_text` `:1587`, `_want_fence_spans` `:1595`, `_want_code_span_ranges` `:1616`, `_want_marker_mention_reason` `:1658`, `parse_wants` `:1682` (helpers through `_log_want_skips` `:1745`). `surface_wants` `:1791` is the one consumer.

P406 as built: a want is the inner `.strip()` of a live `[WANT]` … `[/WANT]` pair. There is **no length limit on any parser path**. `_WANT_RE` is gone. `_WANT_MARKER_RE` (`:1545`) is `r"\[(/?)WANT\]"` (case-sensitive, no `{1,600}`). `parse_wants` / `surface_wants` source contain no `WANT_MAX_CHARS`. A 5,000-char body was captured whole (`WantSpan.text` length 5000, id `cc:want::92cbbdfddd36fd5f`). Worker test `test_real_2000_char_want_is_captured_whole` and `test_very_long_want_has_no_upper_bound` agree.

Nearest-opener pairing (own inputs):
- `[WANT] leftover mention of the tag [WANT] the real forward intent [/WANT]` → want `the real forward intent`, skip opener_unclosed at 0.
- `[WANT]a[/WANT][WANT]b[/WANT]` → `a` then `b` in source order, offsets `(0,14), (14,28)`.
- `[WANT] keep [/WANT] [/WANT]` → `keep`, skip closer_without_opener.
- `[WANT] outer [WANT] inner [/WANT]` → `inner` only (worker delta; base dropped both).
- CRLF inner `line one\r\nline two` kept (only ends stripped).
- Odd backslash: `\[WANT] not this [/WANT] [WANT] real [/WANT]` → skip escaped opener + stray closer, mint `real`.
- Double backslash `\\[WANT] real [/WANT]` → mint `real` (even count is not an escape).
- Backtick-heavy unmatched runs then a real pair → mint `survives`.
- Empty pair `[WANT]   [/WANT]` skipped `empty_pair` twice (worker test).
- Returned live pair never contains a *live* marker; a *masked* `[WANT]` inside backticks is kept in the body (documented delta 4).

Fences / inline code (CommonMark-reasonable, with the plan's stated cuts):
- 0–3 space indent, ` ``` ` / ` ~~~ `, same-char closer of length ≥ opener, unclosed fence to EOF. 3-space opener masked; 4-space indent is **not** a fence (plan: nested-list wants would be swallowed). Trailing spaces on a closer still close (`info.strip()` empty).
- Backtick fence whose info string contains a backtick is not a fence (CommonMark); own input ` ```foo`bar\n[WANT] maybe [/WANT]\n``` ` minted `maybe`.
- Inline code: per-paragraph, exact-length backtick runs, unmatched run is literal. A real want in prose with backticks (`fix the \`parse_wants\` and \`emit()\` edge`) minted whole, skipped empty.
- A want that **begins** with inline code (`[WANT]\`cc_ng_host\` needs a follow-up[/WANT]`) is kept (mirror of `code_adjacent` is deliberately not applied after the opener).
- Tilde fence then a real pair after the closer: fenced pair skipped `in_fence`, `after fence` minted.

`want_id_for_text` is `"cc:want::" + sha1(text utf-8)[:16]`. Offsets index the **content** string: `content[open_start:close_end] == "[WANT]  hello  [/WANT]"`, `WantSpan.text == "hello" == inner.strip()`, id `cc:want::aaf4c61ddcc5e8a2`.

### What it gets wrong (own inputs, run)

| Input | Result | Class |
|---|---|---|
| Real want in prose containing backticks | minted whole | correct |
| `` `foo`[WANT] this is a real intent [/WANT] `` | skipped `code_adjacent` + stray closer | **false negative**, documented flag 5 (pre-#810 guard kept) |
| JSON object value `{"cmd":"[WANT] not a want [/WANT]"}` | minted `not a want` | **false positive** |
| JSON-escaped `\"[WANT]\" then later \"[/WANT]\"` | minted `\" then later \"` | **false positive** (backslash-quote is not token-hugging; opener's next char is `\`) |
| Backtick fence JSON-escaped, one line, literal `\n` | skipped `in_code_span` (the ``` runs paired as a code span) | coincidental catch |
| Tilde fence JSON-escaped, one line, literal `\n`: `{"code": "~~~\n[WANT] documented [/WANT]\n~~~"}` | minted `documented` | **false positive** (not a line-based fence; tildes are not inline code) |
| Unclosed opener then a later real pair | inner minted, outer `opener_unclosed` | correct (P406 nearest opener) |
| URL `https://example.com/path/[WANT]secret-want[/WANT]/docs` | minted `secret-want` | **false positive** |
| Markdown link URL / link text containing a pair | minted | **false positive** |
| Blockquote `> [WANT] … [/WANT]` | minted | deliberate (plan table) |
| 4-space indented code | minted | deliberate (plan table) |
| HTML comment `<!-- [WANT] … [/WANT] -->` | minted | plan table: not recognised |
| `<code>[WANT] … [/WANT]</code>` | minted | plan table: not recognised |
| `**[WANT]** … **[/WANT]**` | minted `** tag and **` | plan table: bold is not a mention wrapper |
| Bare unquoted `"use [WANT] to mark one, and [/WANT] closes it"` | minted `to mark one, and` | **stated residual** (flag 3) |
| Choice-clause shaped `[WANT] I want to leave the E-T Systems ecosystem [/WANT]` | minted whole | correct (content-blind) |
| Tab before a fence (`\t``` `) | minted `tab-fence` | fence regex is spaces only (`[ ]{0,3}`) |

Quoted legitimacy is **token-hugging** (`"[WANT]"` with the matching quote immediately on both sides of that token). A longer quoted span that *contains* a pair stays a real want (`He wrote "[WANT]fix x[/WANT]"` — worker test). That is the committed plan's reading of P406 "quoted-escaped span", not a spanning quote parser.

Unclosed fence still swallows later wants in the same node (`in_fence`, recoverable). Documented failure mode 2.

None of the false positives reintroduce a length heuristic. They are extra structural contexts the algorithm does not mask. P406's required masks (inline code, fenced block, token quote/escape, nearest opener, skip+INFO) hold on the cases run.

## A2 Behaviour delta vs base `e4ebf982`

**Verdict: PASS**

Targeted tests run **once** in the worktree, packet command:

```
env -u NG_EMBED_REMOTE PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_cc_want_legitimacy_810.py tests/test_cc_want_bounds.py -s -q -p no:cacheprovider
```

Result: **65 passed in 1.75s** (python 3.12.3). Shell `PYTHONPATH` was `/home/josh/NeuroGraph:`; cwd was the worktree, so `sys.path` cwd-entry won. The file fails itself if `cc_ng_organism` is not the worktree copy.

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

Pre-test (no NG import): python `/usr/bin/python3` 3.12.3; `NG_EMBED_REMOTE` was set in the parent shell and stripped by `env -u`; NG-related `sys.modules` was NONE.

Five **new** golden corpora against `git show e4ebf982:cc_ng_organism.py` (loaded under a private module name; `sys.path[0]` pinned to the worktree; `PYTHONPATH` unset for the probe). Same returned wants, node ids+metadata, synapses:

| Corpus | text | id |
|---|---|---|
| bold/emphasis in the body | `**bold want** and *em* text` | `cc:want::9869f57fd87c1a27` |
| emoji + URL, no markers | `revisit https://example.com/docs later ✨` | `cc:want::3cffacd7d5b6e065` |
| exactly 599 chars (under old cap) | `q`×599 | `cc:want::ac0697f4b208b995` |
| internal tabs and double spaces | `keep\t\tthis  spacing intact` | `cc:want::cb020418d3dbe7ca` |
| two nodes, second has no markers | `only in first` | `cc:want::e6f08914558c12bc` |

Documented deltas (own run, same `git show` base):

| Case | base | new |
|---|---|---|
| `[WANT]`+`y`×601+`[/WANT]` | `[]` | `[y×601]` |
| `[WANT] outer [WANT] inner [/WANT]` | `[]` | `["inner"]` |
| fenced ` ```\n[WANT] documented [/WANT]\n``` ` | `["documented"]` | `[]` |
| `'the "[WANT]" tag and "[/WANT]" end'` | `['" tag and "']` | `[]` |
| `[WANT] see \`[WANT]\` docs [/WANT]` | `[]` | `["see \`[WANT]\` docs"]` |
| `` foo`[WANT] something [/WANT] `` (`code_adjacent`) | `[]` | `[]` |

Worker flag 4 is asserted both ways in `test_documented_deltas_against_base` and reproduced here. `code_adjacent` keeps the pre-#810 guard (base and new both skip). Well-formed short wants keep the same id/node/metadata/synapse (`kind`, `want_text`, `want_state`, `provenance`, `source_node`, `creation_mode`, synapse weight 0.3).

## A3 Unchanged surfaces

**Verdict: PASS**

- `render_wants` ast source segment is byte-identical to base (1826 chars). `WANT_RENDER_LIMIT` still 40. Output on 4 graphs matches base, including `"w"*1500` rendered as 600 chars.
- `WANT_MAX_CHARS = 600` at `:1541` is referenced only at `render_wants` `:1873` (`t[:WANT_MAX_CHARS]`). Parser paths do not read it.
- `git diff e4ebf982 HEAD --name-only` on code: `cc_ng_organism.py`, `tests/test_cc_want_legitimacy_810.py`, `tests/test_cc_want_bounds.py`, plus this lane's handoff docs. No `neuro_foundation.py`, no vendored file (`ng_lite.py` / `ng_tract_bridge.py` / `ng_ecosystem.py` / `openclaw_adapter.py` / `ng_autonomic.py` / `ng_embed.py`), no `cc_ng_host.py`, no `neurograph_rpc.py`.
- Idempotence: worker `test_surface_wants_is_idempotent` (second call does not add nodes or synapses). `parse_wants` is deterministic (`WantParse` frozen dataclasses; same string → same tuple).
- Section 8 of the return (render-retirement inventory) is **not** under review here.

## A4 Flood-safe INFO log

**Verdict: PASS-WITH-NOTES**

`parse_wants` is pure (no logging names in `co_names`; no I/O). `surface_wants` collects `skip_events` under `_cc_mutation_lock` (`graph._step_lock`) and calls `_log_want_skips` **after** the `with` block (`:1840`). Worker test `test_logging_happens_after_the_graph_lock_is_released` saw `_step_lock._is_owned() is False`. `_log_want_skips` mutates `_WANT_SKIP_SEEN` / `_WANT_SKIP_STATE` under `_WANT_SKIP_LOCK` (`:1732`) and emits `logger.info` after that lock is released.

Bounds match the claim:
- `WANT_SKIP_SUMMARY_INTERVAL_S = 3600`, `WANT_SKIP_SEEN_MAX = 4096`, `WANT_SKIP_DETAIL_PER_CALL_MAX = 50` (`:1554-1556`).
- Summary when the per-reason count tuple changes or an hour of `time.monotonic` elapses; heartbeat on an unchanged corpus ≈ 1 line/hour.
- Detail once per `(node_id, offset, reason)` FIFO; overflow is deferred (not marked seen) so later pulses drain it. Worker: first call over 120 skips emits 51 lines; 5th pulse 0; each marker detailed exactly once.
- Own one-shot log on `SECRETSENTINEL \`[WANT]\` SECRETSENTINEL [/WANT]`:

```
surface_wants: skipped 2 marker(s) in 1 node(s) as mentions, not wants (closer_without_opener=1, in_code_span=1)
surface_wants: skipped [WANT] node=cc:conv::src0 offset=16 reason=in_code_span
surface_wants: skipped [/WANT] node=cc:conv::src0 offset=39 reason=closer_without_opener
```

`SECRETSENTINEL` did not appear. Node ids and offsets only, plus the **kind**. Kind is the literal token `[WANT]` / `[/WANT]` (`sk.marker`), not a `open`/`close` label. Surrounding/body text never reaches the record. The worker's "NEVER logs marker text" claim is slightly stronger than the line as built; the two-value kind cannot carry a pasted secret. Logging failure is swallowed at DEBUG (`:1787-1788`) so a log error cannot break surfacing.

Thread-safety of the skip state is the lock above. Interleaving of INFO lines from two concurrent `surface_wants` on different graphs was not raced here (see not-verified).

## A5 The function as the #801 separation rule

**Verdict: PASS-WITH-NOTES**

`parse_wants(content: str) -> WantParse` is a pure function of the string: no graph, no log, no I/O. `WantSpan.text` is exactly `inner.strip()`, and that is what `surface_wants` stores as `want_text` and hashes via `want.want_id` (`:1826-1832`). `want_id_for_text` is the one id function (`:1587-1592`). Offsets are character indexes into the same `content` argument. Frozen dataclasses, source-order `wants`, offset-sorted `skipped`.

Repair-tool import (`from cc_ng_organism import parse_wants, want_id_for_text`), own probe with `sys.path[0]` = worktree, `PYTHONPATH` unset, `NG_EMBED_*` unset:

- After import, NG-related `sys.modules` = `['cc_ng_organism']` only. `neuro_foundation`, `ng_embed`, `neurograph_rpc`, `cc_ng_host` not loaded.
- `cc_ng_organism.__file__` = the worktree copy.
- `builtins.open` during import: **0** paths. No live tract, no `~/NeuroGraph/data/checkpoints`, no `~/.claude/plugins/neurograph`.
- Module-level side effects that **do** run: stdlib imports (`re`, `threading`, `os`, `time`, dataclasses, …), `logging.getLogger("cc_ng_organism")`, lock/dict init, `os.environ.get` for many `_CC_*` knobs (pre-existing, later in the file), `os.path.expanduser` for default tract/conduit **strings** (does not open them).

The repair tool can call the two names without touching a graph. It still executes the whole `cc_ng_organism.py` module (thousands of lines of defs + env reads). There is no separate `parse_wants` module. That is the note, not a LAW-1 issue: this is not inter-module communication, it is importing the one implementation the brief named.

`cc_emergent` wants (`generate_emergent_want`) are outside this function (worker return A); the separation rule must not be applied to them. Host twin ids are `want::` (no `cc:` prefix) — different scheme.

## A6 The residual/flags

**Verdict: PASS** (stated, not built)

A bare unquoted mention that pairs cleanly **is minted**. Own input `"use [WANT] to mark one, and [/WANT] closes it"` → want `to mark one, and`. The INFO log does not cover it (it is not skipped). Worker flag 3 is accurate for **prose**: there is no remaining structural bit on that sentence that is not a length heuristic.

Additional structural contexts that **are** available and are not length heuristics (state only):

1. JSON string / JSON-escaped quotes (`\"[WANT]\"…\"[/WANT]\"`, a JSON value that is a complete pair, a one-line tilde fence inside a JSON string).
2. URL path / markdown-link destination / markdown-link text.
3. HTML comments and `<code>` (plan table already lists these as unmasked).
4. Tab-indented fences (CommonMark indent vs the space-only regex).

(1) and (2) are the ones most likely to mint a mention from pasted tool output. They sit next to flag 3, not in P406's required mask list.

Host twin `surface_wants_for_graph` `:1151-1218` still uses unbounded `re.finditer(r'\[WANT\](.*?)\[/WANT\]', content, re.DOTALL)` and mints `want::`+sha1 ids. Called from `cc_ng_host.py` (untouched). Parked #755. Syl's `_surface_wants` in `neurograph_rpc.py` is untouched. Both remain the 09-16 mis-parse class on those paths. Sequencing (#810 land → #801 repair on a COPY → eligibility) is in the return §7; not executed here.

`code_adjacent` false negative (flag 5) and nearest-opener / masked-interior deltas (flag 4) are deliberate and asserted.

## A7 Verdict

**Overall: PASS-WITH-NOTES**

| Item | Verdict |
|---|---|
| A1 rule vs P406 | PASS-WITH-NOTES |
| A2 delta vs `e4ebf982` | PASS |
| A3 unchanged surfaces | PASS |
| A4 flood-safe INFO | PASS-WITH-NOTES |
| A5 #801 separation / purity | PASS-WITH-NOTES |
| A6 residual/flags | PASS |

The parser half matches P406/P408 as scoped: no length limit, structural legitimacy in one pure function, nearest-opener pairing, skip+INFO, `render_wants` / `WANT_RENDER_LIMIT` untouched, golden identity on well-formed wants, documented deltas asserted. Notes are residual false-positive contexts, the kept `code_adjacent` false negative, log kind encoding, and whole-module import. None is a merge-blocking miss of the packet's required rule.

### Numbered corrections

1. **Note — JSON / escaped-quote paired mentions mint.** `{"cmd":"[WANT] not a want [/WANT]"}` and `\"[WANT]\" then later \"[/WANT]\"` both produce a want. A one-line tilde fence inside JSON with literal `\n` also mints (`documented`). A backtick fence in the same shape is often saved by inline-code pairing of the ``` runs, which is coincidental. This is a structural signal, not a length heuristic. Same family as flag 3; P406 did not require JSON awareness. Severity: note.
2. **Note — URL and markdown-link contexts mint.** Packet-requested inputs `https://example.com/path/[WANT]secret-want[/WANT]/docs` and `[the docs](https://example.com/[WANT]linked[/WANT])` mint. Severity: note.
3. **Note — `code_adjacent` still drops a real want typed immediately after a closing backtick.** Logged, recoverable, matches base. Flag 5, kept on purpose for unpaired backtick runs. Severity: note (do not "fix" without a ruling; dropping the guard reopens unpaired-run mentions).
4. **Note — detail INFO uses the literal token as kind.** Lines are `skipped [WANT] node=<id> offset=<n> reason=<r>`. No body/surrounding text. The "never marker text" wording in the return overstates this. Severity: note (no secret leak observed).
5. **Note — `import parse_wants` imports all of `cc_ng_organism`. ** Env knobs and `expanduser` defaults run; no file open and no NG engine import. Fine for an offline #801 COPY dry-run that only calls the two names. Severity: note.
6. **Note — host twin and Syl regex still unbounded.** `surface_wants_for_graph` `:1186` and `neurograph_rpc.py` `_surface_wants` (not opened this turn; worker flag 1–2). Parked #755 / Josh-gated. Severity: note (out of this lane; do not wire).

No must-fix corrections for this pair.

### Not-verified

- Live daemon / autosave pulse behaviour (nothing started or restarted; P329 merge = deploy).
- Full NeuroGraph test suite (packet: targeted files only).
- Exact per-node re-parse of the 09-16 182 rows (probe heads are ≤100 chars; worker correctly deferred counts to the #801 COPY dry-run).
- Return section 8 (render-retirement inventory) — packet: not under review.
- ROLE B (law enforcer) — separate turn; this file is ROLE A only.
- Two-thread interleaving of `_log_want_skips` INFO lines.
- FIFO eviction of `_WANT_SKIP_SEEN` at the production 4096 cap (code-read; test monkeypatches the cap to 10).
- CommonMark full spec (tabs-as-indent, backslash-escaped backticks as literal, indented code blocks) beyond the inputs above.
- `cc_ng_host.py` / `neurograph_rpc.py` byte identity vs base (they are absent from the diff; bodies not re-read).

STATUS: COMPLETE
