<!--
# ---- Changelog ----
# [2026-09-30] Z12 worker (Claude Sonnet 5.5, Claude Code) — #813 audit + plan, committed BEFORE code
# What: character-cap audit of the whole Pith path (NG organism/host/surfacing, docs daemon/preflight,
#   Condensate miniTID read-only) + the exact change list for the clip removal.
# Why: Exec P411/P413 via Chief-003 (Josh: no truncation). Assignment build-813-pith-clip.md step 1.
# How: every line number below was read at NG base e4ebf982, docs base 933f7158, Condensate master 4086540
#   (`git show master:rust_core/src/minitid.rs`, read-only). Nothing was changed to produce this file.
# -------------------
-->

# #813 — Pith clip removal + character-cap audit: PLAN (pre-code)

Lane `pith-clip-removal-813` · dispatch #10841 · owner Z12 · worker seat · authority `worktree_write`.
Related: [[NeuroGraph]] · [[Pith]] · [[NeuroGraph Is a Mind, Not a Database]] · [[Format-for-Purpose Principle]]

Recentred first on the two concept pages the global rules name. The relevant consequence: a connected
assembly is the organism's relationship unit ("one fired root plus its learned companions"); the fix must
move **assemblies** whole in or out, never shorten a node inside one — that would be the database reflex
(snippet ranking) the concept page exists to prevent.

## 0. What Josh's ruling means, concretely (P411/P413)

1. Every node renders **WHOLE**.
2. The budget is met by **fewer whole assemblies**: the lowest-relevance whole items are dropped, and the
   drop is **visible** — one INFO line with the count and total size (chars), never silent.
3. A keyframe applies only if it carries its **delta**.

**What "a keyframe carries the delta" means for `pith_stage2_keyframe` (`cc_ng_organism.py:4101-4213`).**
The function already returns `(keyframe, delta)`; `delta` is exactly the elided segments joined by spaces
(`:4209`). So `keyframe` + `delta` is a reordering of the original — *lossless only if both travel*. Every
production caller today unpacks `text, _ =` / `_delta` and **throws the delta away** (`:2990`, `:4381`,
`:4520`, `:4666`, `:4988`); the result is a cut with a marker. Model-facing output has no channel for the
delta *inside the same budget* (keyframe + delta ≈ the whole item, which is precisely what a binding budget
refused). Therefore, concretely: **a keyframe can never satisfy a binding budget, so it is inapplicable to
budgeted output** — the item is whole or it is dropped. `pith_stage2_keyframe` stays as a pure function
(its unit tests are unchanged and valid); its docstring gets an explicit "applies only with the delta"
note; and no budgeted path may call it. A test enforces the latter by AST scan (no call inside the
provider path or Stage 3).

## 1. Audit table

Class: **CUT** (truncates content; must be fixed) · **REJECT-LOUDLY** (refuses the whole item with a
visible error; not truncation; may stay) · **DROP-SILENTLY** (item vanishes without a trace; must become
loud) · **DIAG** (diagnostic, never model-facing) · **ID** (identifier digest, not content) ·
**COUNT** (item-count bound, not a character cap; listed as an adjacent finding).
"This branch" = whether the NG/docs branch changes it. Every "no" carries its reason.

### 1A. NeuroGraph — `cc_ng_organism.py` (Pith provider + L1 path)

| # | Cap | file:line (base) | Default | When exceeded | Class | This branch |
|---|---|---|---|---|---|---|
| 1 | `CC_PITH_PROVIDER_NODE_CHARS` | `:3512`, used `:4664-4669` (`_pith_node_text`, called `:4825`) | 700, clamp [120,2000] | keyframe with `⋯[+N]` marker, delta discarded; else `_pith_cut_at_word_boundary` hard cut | **CUT** | **REMOVE** (const, config-key `:3816`, status dict `:3858`) |
| 2 | budget-time prose shortening `_pith_fit_statement` / `_pith_fit_connected_line` | `:4972-4992`, `:4995-5052` | envelope = budget | every member's prose water-filled down to a share; keyframe or cut + `" …"` | **CUT** (this is *how the budget is met today*) | **REPLACE** with whole-or-drop |
| 3 | `_pith_provider_admit` strict-prefix `break` | `:5055-5076` (`:5069-5070`) | — | first line that cannot fit ends admission; that line and **every lower-ranked line vanish with no log** (only `capacity_empty` warns, and only if nothing at all was kept) | **DROP-SILENTLY** | **MAKE LOUD** (INFO: count + total chars) |
| 4 | source-label clip `value.strip()[:80]` | `:4675-4678` (`_pith_node_sources`) | 80 | provenance label silently cut (no marker) | **CUT** | **REMOVE** the slice |
| 5 | `CC_PITH_PROVIDER_MAX_INSTRUCTION_CHARS` | `:3513`, used `:5149` | 8000, clamp [500,16000] | closed `unavailable`, warning `instruction_too_large`; miniTID shows `[Pith unavailable: …]` | **REJECT-LOUDLY** | no — stays (may stay per ruling). *Advisory:* the instruction is only an attention cue and is never echoed, so a pasted 9k-char human turn blanks Pith for that turn; lifting it is a separate call |
| 6 | `CC_PITH_PROVIDER_MAX_QUEST_CHARS` | `:3515`, used `:5153` | 8000, clamp [0,16000] | closed `unavailable`, warning `invalid_quest_focus` | **REJECT-LOUDLY** | no — stays. **NOT DEAD** (see §1E): Condensate master still calls `extract_quest_focus` (`minitid.rs:1366`, also `:1671`) and sends `quest_focus` on every provider request; both hosts forward it (`cc_ng_host.py:1027`, `cc-ng-daemon.py:1667`). The assignment's "likely DEAD" premise does not hold at the versions I read |
| 7 | recall snippet `resolve_surface_content(..., max_chars=300)` | `:2983` (impl `surface_resolver.py:112-114`) | 300 (fn default 240) | cut at word boundary + `"…"` | **CUT** | **PARTLY**: provider path only — new opt-in `whole_content=False` param on `cc_pattern_completion_recall`, `True` from `pith_provider_context`; the provider needs it because `_display_text` falls back to the 300-clipped `content` when a node has no text metadata (`:4640-4652`, `:4825`). Every other caller stays byte-identical (default off). **The gate-off L1/`## Active Recall` path is NOT changed**: with `CC_PITH_ENABLED` off there is no budget, so removing the cap would inject unbounded whole turns — see Decisions D1 |
| 8 | Stage 3 keyframe fallback | `:4520-4527` (`pith_stage3`) | budget = `CC_PITH_L1_BUDGET` 4000 (clamp [500,40000]) or breathing budget | an over-budget line is replaced by a keyframe (delta discarded) so it "fits" | **CUT** | **REMOVE** the branch → whole-or-drop |
| 9 | Stage 3 tail `break` | `:4529-4530` | — | remaining ranked lines dropped; counted in `ranked_dropped`, recaptured by the Stage-5 victim buffer, **not logged** | DROP (counted, not visible) | **ADD** INFO (count + chars). Kept: "first line always kept" (`:4515`) — never emit an empty L1 |
| 10 | `pith_compress_history` keyframe | `:4381`, budget `max(60,min(1000,base*mult))` `:4380`; `CC_PITH_KEYFRAME_CHARS` `:3520` (220, clamp [60,1000]) | 220 | old history turns compressed to keyframes, delta discarded (lossy *by design*) | **CUT** | no — **no live caller**: Condensate master has `compress_history` only in a header comment (`minitid.rs:12,15`, `git grep` confirms); handler `cc_ng_host._handle_compress_history` is live-callable but unfired. Fixing = retiring/redefining a documented socket event → Decisions D2 |
| 11 | Prefetch LOD keyframe | `:2986-2992`; `CC_PITH_PREFETCH_SUMMARY_CHARS` `:3138` (150, clamp [60,1000]), `CC_PITH_PREFETCH_LOD_DIST` `:3137` (1.5) | gated `CC_PITH_PREFETCH_ENABLED` default **off** | far predicted node staged as keyframe, delta discarded | **CUT** (latent: gate off) | **REMOVE** the staging (near/far both stay whole). The now-unused constants and the `_cc_node_query_distance` helper are left in place and flagged (dead-tunable removal touches the host allow-list, not editable this turn) |
| 12 | `pith_stage2_keyframe` internal single-segment cut | `:4186-4187` via `_pith_cut_at_word_boundary` `:4088-4098` | budget − 12 | a single segment longer than the budget is word-cut | CUT *inside the pure function* | no — unreachable from budgeted output after this change (only #10 remains a caller); documented |
| 13 | `_format_cc_recall_block`, `render_constitutional_core`, `_pith_render_connected_line`, `_pith_exact_anchors` | `:3032`, `:1613`, `:4958`, `:4616` | — | **no cap** (whole) | — (clean) | — |
| 14 | `render_wants` clip + `_WANT_RE` span | `:1512-1514`, `:1603-1606` | `WANT_MAX_CHARS`=600, `WANT_RENDER_LIMIT`=40 | render: `t[:600]` cut (no marker); **extraction: a `[WANT]…[/WANT]` span > 600 chars does not match at all → the want is never captured**; >40 wants → visible line `- ... and N older open wants` | render **CUT**; extraction **DROP-SILENTLY**; count **COUNT/loud** | no — outside the provider context (provider renders the constitutional core only, not wants). Listed for Josh → Decisions D3 |
| 15 | Commons deposit metadata `text[:2000]` | `:1122` | 2000 | the metadata copy of `text` is cut (the embedding is of the full text) | **CUT** (deposit path) | no — deposit path / LAW 7 territory → D4 |
| 16 | recall debug log | `:5287-5296` (`[:15]`, `[:48]`, `[:70]`, `[:120]`) | gated `CC_RECALL_DEBUG` off | previews cut | **DIAG** | no |
| 17 | id digests `hexdigest()[:16]`, `uuid.hex[:8]`, log `[:40]` | `:1167,1558,1749,1760,2523,1952` | — | identifier prefixes | **ID/DIAG** | no |
| 18 | count bounds | `CC_PITH_PROVIDER_ROOTS` 8 (≤24), `_MEMBERS` 6 (≤16), `_DEPTH` 2 (≤3) `:3509-3511`; basin overlap dedup `≥0.6` `:4949-4954`; `out[:k]` `:3015` | | neighbours beyond `MEMBERS` and overlapping basins are skipped **with no log** | **COUNT / DROP-SILENTLY (nodes, not chars)** | no — adjacent finding, not a character cap; flagged → D5 |

### 1B. NeuroGraph — `cc_ng_host.py` (VPS host — **not edited this turn**)

| # | Item | file:line | Class | Effect / note |
|---|---|---|---|---|
| 19 | `PITH_SNAPSHOT_GATE_KEYS` lists `CC_PITH_PROVIDER_NODE_CHARS` | `:509` (`:510-511` the other two) | telemetry allow-list | **Effect if the VPS host still exports the variable:** none functional — nothing reads it after this change; the raw env value is still copied into `gates` of `pith_metrics.jsonl` (harmless), while `config.resolved` (from `pith_effective_config()`) no longer contains the key. A launcher that still *requires* it (the docs preflight) is handled in §3. Leave the name in the allow-list until the host is next edited |
| 20 | tool-experience clips | `:1190,1191` `[:1000]`, `:1194` `[:2000]`, `:1197` `[:2000]`, `:1199` `[:1000]`, `:1201` `[:200]`+`[:1500]`, `:1203` `[:1000]` | **CUT — on the DEPOSIT path** (raw experience is clipped *before* it reaches the substrate; LAW 7) | not edited (host excluded; changes substrate ingestion volume) → D4. Same 7 clips in the laptop daemon (§1D #27) |
| 21 | `hexdigest()[:16]` target ids | `:688,729,520` | ID | — |

### 1C. NeuroGraph — CES `surfacing.py`, `surface_resolver.py`

| # | Item | file:line | Default | Class | This branch |
|---|---|---|---|---|---|
| 22 | `SurfacingMonitor.format_context` cut | `surfacing.py:293-295` | 200 chars, `"..."` | **CUT** | no — shared with Syl's live sidecar `/assemble` (P329: merge = deploy) and not on the Pith-on path (Pith renders `_format_cc_recall_block` from whole `item.content`; `monitor_ctx` is used only in the gate-off and Pith-failure fallbacks, `cc_ng_organism.py:5522-5524`) → D6 |
| 23 | `resolve_surface_content` default cut | `surface_resolver.py:112-114` | `max_chars=240` | CUT (default arg; per-caller) | no — see #7; callers keep their own bound |

### 1D. docs repo (`origin/main` 933f7158)

| # | Item | file:line | Class | This branch |
|---|---|---|---|---|
| 24 | preflight `required` includes `CC_PITH_PROVIDER_NODE_CHARS` | `scripts/cc-ng-service.py:109-113` | launch gate → REJECT-LOUDLY (`missing canonical export: …`) | **UPDATE** — drop the name (same change as the contract, so the launcher does not refuse a launch that no longer needs the variable) |
| 25 | preflight tests | `scripts/tests/test_cc_ng_service.py:26-34`, `:76-83` | — | **UPDATE** — six→five required names + a test that the removed name is *not* required and is tolerated if still exported |
| 26 | telemetry allow-list names the variable | `scripts/cc-ng-daemon.py:1990` | telemetry | no — live laptop daemon file, and another lane (daemon-recall-756) has a worktree on it; inert once unread |
| 27 | tool-experience clips | `scripts/cc-ng-daemon.py:1156,1157,1160,1163,1165,1167,1169` | **CUT — DEPOSIT path** | no → D4 |
| 28 | `PITH_HOST_CONTRACT.md` is in the **NG** repo, not docs | `docs/PITH_HOST_CONTRACT.md:114-125` (export list), `:170-174` (shortening prose), `:137-139` (`capacity_empty`) | contract | **UPDATE** in the NG branch (same change) |

### 1E. Condensate — `rust_core/src/minitid.rs` @ master 4086540 (**READ-ONLY; nothing edited**)

| # | Cap | file:line | Default | When exceeded | Class | Smallest fix (proposal only) |
|---|---|---|---|---|---|---|
| 29 | `MAX_PROVIDER_CONTEXT_CHARS` | `:508`; checks `:801` (`parse_provider_response`), `:1095` (`provider_context_is_usable`) | 40,000 chars | daemon reply → `parse_provider_response` returns `None` → `Err("daemon response was not a fresh provider_context envelope")` (`:761`) → `pith_failure_outcome`: provider-facing notice `[Pith unavailable: daemon provider_context — …]` **and** the raw text is deposited to the gateway tract (`deposit_text`). Post-strip check → `"empty, oversized, or carrying a live rail marker"` (`:1419`). **Never cut, never silent.** | **REJECT-LOUDLY** | none required. *Wording defect:* the first message does not say "oversized" (it says "not a fresh envelope"), so an operator cannot distinguish a size rejection from a schema rejection — split `context_chars > MAX` into its own `Err("provider context exceeded N chars")`. *Consistency:* the NG side already guarantees `len(context) ≤ budget ≤ 40000` (`budget_chars` validated 500–40000 `:5158-5161`; `cc_l1_budget` clamped `:3505`) and this branch removes the last in-flight shortening, so a compliant daemon never trips it. The two 40000s are independent constants that must stay equal — worth a shared comment |
| 30 | `MAX_PROVIDER_RESPONSE_BYTES` | `:509`; enforced `:747-751` | 256 KiB | `Err("daemon response exceeded N bytes")` → failure notice | **REJECT-LOUDLY** | none. (40,000 chars ≤ 160 KB UTF-8 + JSON overhead < 256 KiB — consistent) |
| 31 | `PITH_NOTICE_WHY_MAX` | `:1242`; cut `:1264-1265` | 200 chars | the provider-facing `why` is cut to 200 and `...` appended (whitespace-flattened first). The raw `why` is still deposited uncut (`deposit_text` `:1274`) | **CUT** | delete the `take(200)`+`"..."` branch, keep the whitespace flatten. Safe: every `why` is a closed static string or a one-line transport/OS error (`:1262`-`:1300`, `:1385`, `:1407-1424`), so the notice stays one bounded line by construction. Update the test at `:3499-3514` (asserts `ends_with("...]")` and `≤ MAX+60`) |
| 32 | tool-tail byte envelope `MINITID_PITH_TOOL_TAIL_BYTES` | `:490-512`, `bounded_current_episode` `:1000-1065` | 64 KiB, clamp [8 KiB, 1 MiB] | older *whole* tool messages/pairs are dropped from the live tail; the newest indivisible pair is kept exact even if it alone exceeds the bound. Aggregate `msgs_in`/`msgs_out` reported in the request counts line (`:1712-1731`) | whole-item DROP — visible in aggregate, **not itemised** | consider logging count+bytes of dropped messages (same INFO discipline as this lane) — proposal only |
| 33 | `MAX_BODY` | `:361`; `:1753`, `serialized_messages_bytes` fallback `:992` | 20 MiB | `to_bytes` error → HTTP 400 with the error text | **REJECT-LOUDLY** (request ceiling, outside the provider text path) | none |
| 34 | `MAX_SESSION_ID_BYTES`/`MAX_AGENT_ID_BYTES` | `:503-504`, `:528` | 512 / 128 | `NoSessionIdentity` → Pith failure notice | ID / REJECT-LOUDLY | none |
| 35 | KISS window/warmup/cadence, `MAX_SESSION_CAPACITY` | `:351-357`, `KISS_RECENT_WINDOW` | count bounds | not character caps (`:291`: "never truncates message content itself") | COUNT | none |

A grep of the whole file for `.chars().take`, `truncate`, `[..N]`, `...` found only `:1265` (#31) as a content cut; the other `[..x]` hits are wire-buffer indexing (`:747`), header parsing (`:846,:885`) and tool-pair scanning (`:1031`).

## 2. Exact change list

### NG branch `cc-laptop-pith-clip-813-20260930` (base e4ebf982)

`cc_ng_organism.py` (changelog header entry added):
1. `_pith_node_text` → return `_pith_node_raw_text(node, fallback)` whole. Delete `_CC_PITH_PROVIDER_NODE_CHARS` (`:3512`), its `_PITH_CONFIG_KEYS` entry (`:3816`) and its `pith_effective_config()` status entry (`:3858`). Rewrite the docstring (it says "bound").
2. Delete `_pith_fit_statement` and the water-filling in `_pith_fit_connected_line`; the latter becomes "whole render ≤ budget → copy; else `None`". (LAW 3: repaired in place — same function names, no parallel implementation.)
3. `_pith_provider_admit`: same strict rank-order prefix; a line that cannot fit **the remaining envelope** ends admission (rank is preserved: nothing lower-ranked jumps a dropped higher-ranked line). One deliberate refinement, flagged for the reviewer: a line that could not fit **even an empty envelope** (rendered cost > total learned budget) is skipped rather than ending admission — otherwise one giant assembly ranked first would blank every other assembly. After the loop: if anything was dropped, `logger.info("pith provider_context: budget %d chars met by dropping %d whole assemblies (%d chars rendered); kept %d (%d chars)")`. No log when nothing is dropped.
4. `_pith_node_sources`: remove `[:80]`.
5. `pith_stage3`: delete the keyframe fallback (`:4520-4527`); after the loop, if lines were dropped, one INFO line with count and total `len(content)`. The empty-L1 guard (`:4515`) stays.
6. `cc_pattern_completion_recall(..., whole_content: bool = False)`: `max_chars = sys.maxsize if whole_content else 300` at the single `resolve_surface_content` call; `pith_provider_context` passes `whole_content=True`. Remove the LOD keyframe staging (`:2964-2992`) and the now-pointless `query_dir` embed that only fed it.
7. Docstring note on `pith_stage2_keyframe`: applies only with its delta; no budgeted caller.

`docs/PITH_HOST_CONTRACT.md`: export list `:114-121` drops `CC_PITH_PROVIDER_NODE_CHARS` (six → five) with an explanatory line; `:170-174` rewritten (whole-or-drop, INFO log, no shortening); `capacity_empty` wording `:138-139` keeps meaning (no whole line fit); changelog entry.

`tests/test_cc_pith_clip_813.py` (new) + updates with reasons to `tests/test_pith_provider_context.py` (`test_total_context_bound_fits_oversized_connected_line_without_tearing` asserted the shortening), `tests/test_pith_stage2.py` (the Stage 3 keyframe cases `:141-187`), `tests/test_pith_stage4.py` (`test_promotion_lod_summarizes_far_content` `:217`).

Staged, **not applied**: `handoffs/z12-pith-clip-813/returns/bashrc-drop-node-chars.sh` (§4).

### docs branch `cc-laptop-pith-clip-813-20260930` (base 933f7158)
`scripts/cc-ng-service.py`: drop the name from `required` (`:109-113`). `scripts/tests/test_cc_ng_service.py`: update the six-name fixtures and add the not-required / tolerated-if-present test. Changelog headers. Vault pointer `handoffs/z12-pith-clip-813/plan-001.md` (wikilinked) and the return pointer.

### Condensate: nothing. Proposals only (#29 wording, #31, #32).

## 3. Contract in both places, ONE change (assignment ruling 3)
Contract (`docs/PITH_HOST_CONTRACT.md`, NG) and preflight (`scripts/cc-ng-service.py`, docs) are updated **together**, and are inert until deployed. Ordering constraint for S4 (P329, merge = deploy): leaving the export in `.bashrc` is **always safe** (new code ignores it; the new preflight tolerates it). *Removing* it is safe only once the new preflight is live — with the current preflight still in place a missing export is `missing canonical export` and the launch is refused. So the `.bashrc` edit runs strictly after the docs merge is deployed (§4).

## 4. `.bashrc` — STAGED, not applied
`~/.bashrc` is read for **names only** (never written this turn). Read-only fact recorded at plan time: exactly one line matches `^export CC_PITH_PROVIDER_NODE_CHARS=` (assignment says `:291`). The staged script (written in the build step) does: timestamped backup → removal by **line pattern** (`^export CC_PITH_PROVIDER_NODE_CHARS=`), not line number → name-only verify → reverse command; it refuses to run if the pattern matches ≠ 1 line. **Runs at S4 batch step: immediately AFTER the deploy of the NG merge (code no longer reads the variable) AND the docs merge (preflight no longer requires it), and BEFORE the service restart** — i.e. inside the same batched go, after the merges land, before the restart that re-reads the environment. Running it earlier (before both merges) makes the *current* preflight refuse launch.

## 5. Behaviour when assemblies exceed the L1 budget (the new rule)
`pith_provider_context` → `_pith_provider_admit(candidates, learned_budget)`: rank order, whole assemblies only; the first assembly that does not fit the remaining envelope ends admission; assemblies that follow are dropped whole. One INFO line reports how many and their total rendered chars. State is `ok` if anything was kept, else `empty` with warning `capacity_empty` (unchanged vocabulary — the Rust side validates only that warnings are strings, so the closed response schema is untouched). The response `assemblies` count already equals the kept count.

## 6. Callers and ripple
* `_CC_PITH_PROVIDER_NODE_CHARS`: read only by `_pith_node_text`; surfaced by `pith_effective_config()` (`:3858`) and `_PITH_CONFIG_KEYS` (`:3816`) — both edited here; host allow-list `cc_ng_host.py:509` and daemon allow-list `cc-ng-daemon.py:1990` inert (§1B/1D).
* `_pith_node_text`: one caller (`:4825`). `_pith_fit_statement`: one caller (`_pith_fit_connected_line`). `_pith_fit_connected_line`: one caller (`_pith_provider_admit`), plus `tests/test_pith_provider_context.py`.
* `pith_stage3` callers: `cc_assemble_recall` (`:5492`); tests stage1/2/3/5, l1_provenance, metrics_concurrency, cc_recall_*. `compressed_count`/`chars_saved` counters (PithMetrics, documented as best-effort in the contract `:321-322`) will stay at 0 from Stage 3 — the fields remain (no schema churn); `test_pith_stage2.py:141-187` asserts them and is updated with the reason.
* Effect on `pith_metrics.jsonl`: `config.resolved` loses `CC_PITH_PROVIDER_NODE_CHARS`; nothing consumes that key.
* miniTID: the closed response is unchanged; `len(context) ≤ budget ≤ 40000` still holds (the invariant is now trivially stronger).

## 7. Interaction with the #810 turn-3 CC-only surfacing wrapper
Independent. That wrapper concerns how `[NeuroGraph Surfaced Knowledge]` is stripped/placed by miniTID and the surfaced-marker rail; this lane touches neither `surfacing.py` (#22 is listed, not edited) nor the rail markers. **Shared code:** none is edited by both. The only overlap is read-only: `strip_rail_marked_assemblies` (miniTID) removes whole assemblies that echo a rail marker — consistent with whole-or-drop, and it strictly benefits (whole node text is now present, so a marker is never hidden by a keyframe cut). The live-rail placeholder (`_display_text`, `:4820-4825`) compares the *unclipped* `raw` with the rail text — unchanged.

## 8. Decisions I need from the manager / Josh (not made by me)
* **D1** — gate-off `## Active Recall` snippets (`:2983`, 300 chars): make whole only when Pith is on (so the budget can drop), or bound another way?
* **D2** — `compress_history` (#10): no live caller in Condensate master; retire the handler, or make it identity/lossless?
* **D3** — wants: `_WANT_RE` 600-char span silently un-captures long wants (#14).
* **D4** — deposit-path clips (#15, #20, #27): LAW 7 says the substrate receives raw experience; these cut before deposit.
* **D5** — silent member/overlap drops (#18), same INFO discipline?
* **D6** — `surfacing.py` 200-char cut (#22), shared with Syl's `/assemble`.
* **D7** — `MAX_QUEST_CHARS`: the "removed Quest lane" premise is contradicted by Condensate master (§1A #6). Confirm intent before anyone removes it.
