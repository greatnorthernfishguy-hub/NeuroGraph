# checker-017 ROLE A (cross-family) — 118-want text repair plan-003

STATUS: COMPLETE

- Seat: checker-017 (cross-family, grok-4.6, `report_only`). ROLE A only; ROLE B not written.
- Lane: `z12-s3-restore-bundle-20260929`. Dispatch #10817. Zone manager Z12 session `52d39aba-db92-4bf2-b3b1-0e4c13f77d8c`.
- Packet: `/home/josh/docs/.claude/worktrees/daemon-recall-756-20260930/handoffs/z12-silent-failure-lanes-20260930/assignments/review-packet-want-text-repair-118.md` (docs branch `cc-laptop-daemon-recall-756-20260930`, not edited).
- Plan: `handoffs/z12-want-text-repair/returns/plan-003.md` on `cc-laptop-want-text-repair-20260930` at `e4afc85b6f86315f936ce39522f77faaa69649ab` (202 lines).
- Plan sha256 (`sha256sum` of the blob at that pin): `7e76f52f6b598fdf290f00d2a934a1edbd2407c760a8bc7caf2603e86ae523dd` (matches packet).
- Predecessors read as evidence: plan-001 `92af58222f1898c617256fb429c0c4c29258d9e5`, plan-002 `55fdb2b15531741868d457408630c27412e43ae8`.
- Parser pin (read-only `git show`, 810 branch never checked out into this worktree): `07eeaaaa1356276f518dc2b423e895736ba46c77` (`parse_wants` `:1682`, `want_id_for_text` `:1587`). Defining-file blob `07eeaaaa:cc_ng_organism.py` sha256 `1acb09e2b17448a438d8f85fb697b39e439ff2503d70d88f9d88048c2d6cc2c0`. Code commit that introduced the function: `c9fe56d85809c4d865fe2a4a353f9db7b172a00c`.
- Code pin for cites: `e4ebf982b1989fd9066d610b94853bc68bf70d37` via `git show e4ebf982:<file>`. `neuro_foundation.py` was never edited. `git merge-base --is-ancestor e4ebf982 HEAD` holds; 810 is a sibling branch.
- Derived JSON only: `summary-laptop.json`, `probe-laptop.json`, `want-rows-laptop.json` under `/home/josh/backups/z12-s3-restore-bundle-20260929T222529Z/analysis-scratch/`. Stdlib `json` only. Want bodies / VDB prefixes not printed.
- Authority: report_only. No build, no PR, no merge, no settle, no dispatch. No graph, msgpack, checkpoint, or tract load. Primary `/home/josh/NeuroGraph` not edited. `NG_EMBED_*` unset.

## P379 session start

```
python: /usr/bin/python3
sys.path:
  
  /home/josh/NeuroGraph
  /home/josh
  /usr/lib/python312.zip
  /usr/lib/python3.12
  /usr/lib/python3.12/lib-dynload
  /home/josh/.local/lib/python3.12/site-packages
  /usr/local/lib/python3.12/dist-packages
  /usr/lib/python3/dist-packages
NG-related sys.modules: NONE
PYTHONPATH: /home/josh/NeuroGraph:
NG_EMBED_*: unset or none
cwd: /home/josh
```

`PYTHONPATH` would resolve every NG import to the **primary** checkout `/home/josh/NeuroGraph`. No NG module was imported this turn (`neuro_foundation`, `cc_ng_organism`, `checkpoint_guardian`, `openclaw_hook`, `neurograph_rpc`, `ng_embed` all absent from `sys.modules`). Number checks used stdlib `json` on the three named JSON files. `parse_wants` was read as text via `git show` and paper-traced; it was never called.

Worktree `/home/josh/NeuroGraph-worktrees/z12-want-text-repair-20260930` at plan pin `e4afc85` before the stub; `git pull --rebase origin cc-laptop-want-text-repair-20260930` was already up to date.

Stub first-write: commit `404a0b7907c084da0c3ff42671cf1873d86df440` (`STATUS: INCOMPLETE - review in progress`), pushed to `origin cc-laptop-want-text-repair-20260930`.

---

## Overall verdict

**PASS-WITH-NOTES**

plan-003's rewrite-and-verify design matches the code at `e4ebf982`: want ids are minted only at creation, the `:1559-1560` dedup is what makes re-keying necessary, there is no canonical `rename_node`, and `Graph.restore` rebuilds `_outgoing` / `_incoming` / `_node_hyperedges` without editing `neuro_foundation.py`. The separation rule's GENUINE / SEPARATE / ANOMALY / NONE table matches the shipped `WantParse` contract. Independent recount of the derived JSON reproduces A 67 / B 16 / C 35, repaired upper bound 102, no derived lower bound, GENUINE upper bound ≤ 118.

Three HIGH notes must be folded before a Phase-1 freeze: (C1) pin the shipped parser (`c9fe56d` / `07eeaaaa`) in place of the stale "no code yet" `2ab4a851` plan document; (C2) `code_adjacent` currently applies to closers and maps some genuine pairs onto NONE; (C3) Phase-1 import of `parse_wants` must pin `sys.path[0]` to the P1 file and unset `PYTHONPATH` (this host's `PYTHONPATH=/home/josh/NeuroGraph:` binds the primary checkout, which has no `parse_wants`).

The safety envelope (delete nothing, collisions listed, census STOP, `atomic_file_write`) holds as written.

---

## A1 The id crux against the code

**Verdict: PASS-WITH-NOTES**

### Id is minted only at creation

At base `e4ebf982`, `cc:want::` + `sha1(...)[:16]` is assigned only when a node is created:

| Site | Line | Expression | Population |
|---|---|---|---|
| `surface_wants` | `:1558` then dedup `:1559-1560` | `"cc:want::" + sha1(inner utf-8)[:16]` | text-marker, `cc_authored` — this repair's S |
| `generate_emergent_want` | `:1749` (resolved concept key) / `:1760` (unresolved `want_text`) | same prefix | `cc_emergent`, outside S |
| `surface_wants_for_graph` | `:1167` | `"want::" + sha1(inner)[:16]` | host twin, different prefix |

Packet lines `:2151` / `:4374` mint **`cc:conv::`** conversation nodes (`run_conversational_dual_pass`, `pith_compress_history`). They are creation-time ids of a different kind. All 182 derived want ids are 25-character `cc:want::` keys; zero `want::` prefixes in `want-rows-laptop.json`.

On the 810 pin the text-marker expression lives once, in `want_id_for_text` `:1587`; `surface_wants` `:1791` uses `want.want_id` and the same `if want_id in graph.nodes: continue`.

### Why option (ii) is necessary

`surface_wants` `:1559-1560` (810: the `want_id in graph.nodes` continue after `want.want_id`) skips minting when the **hash of the current inner text** already exists. Keeping the old id while replacing `want_text` with `X` leaves a node whose id is `sha1(T)` and lets the next pulse mint `sha1(X)` as a second node. Re-keying the existing node to `want_id_for_text(X)` is the path that lands on that continue. That is P402 as implemented.

### Persisted reference sites — completeness

Repo-wide `git grep` at `e4ebf982` found no `rename_node` / `rekey` / `replace_node`. `remove_node` `:1955` cascades synapses and hyperedge membership and **deletes**. `create_synapse` `:1989` (`_outgoing`/`_incoming` add at `:2032-2033`) resets weight/ages. Those are correctly rejected as the rewrite vehicle.

Plan S1–S13 vs the serializer/restore at `e4ebf982`:

| # | Structure | Code check | Volume check |
|---|---|---|---|
| S1 | `nodes` key + `node_id` | `_serialize_node` `:5090+`; `_serialize_full` nodes map `:5193`; restore loop `:5387-5416` | ≤118 by construction of S |
| S2 | `Node.pred_weights` keyed by post id | field `:673`; live update `:3463`/`:3473`; serialize `:5106` area; restore `:5407` | ≤1,735 **not recomputed** (compact probe has no `pred_weights`) |
| S3 | synapse `pre_node_id`/`post_node_id` | create `:2014-2033`; serialize native blob `:5194-5199`; restore bulk-load then index rebuild `:5454-5474` | **81,999** of 138,753 touch S; 13,203 S↔any-want; **131** rim↔S — matches |
| S4 | `synapse.metadata["expected_target"]` | written `:3308` | unknown until census (plan already says so); 5,172 surprise-driven synapses carry that key (`summary-laptop.json`) |
| S5 | hyperedge `member_nodes` / `member_weights` / `output_targets` | serialize `:5112-5136`; restore `:5460-5517` | **48** hyperedges, **145** memberships — matches; 50 HEs touch any of the 182 |
| S6–S12 | predictions, outcomes, HE predictions, `he_output_candidates`, `novel_sequence_log`, `delay_buffer`, `recent_spikes` | serialize `:5201-5278`; restore `:5427-5435`, `:5557-5647` | occurrences unknown until census. `prediction_outcomes` restore `:5588-5610` really does skip node-existence validation. `he_output_candidates` inner keys **are node ids** (`:2414-2429`). `novel_sequence_log` events carry `source` + `firing_nodes` (`:3347-3353`) |
| S13 | activation sidecar `entries` keyed by id | `capture_state` `:113-136`; `restore()` skips missing nodes at `:246-250` | sidecar has 3,985 entries (`summary`); **all-182-present not recomputed** (packet files have no sidecar key set) |

Derived indexes the plan correctly treats as non-persisted: `_outgoing`/`_incoming` rebuilt at restore `:5413-5414` and `:5472-5474`; `_node_hyperedges` `:5415` / `:5486-5487`; dirty sets cleared at restore `:5698-5700`. `synapse_confirmation_history` is keyed by synapse UUID.

Packet extras:

- **vdb keys:** 0 of 182 want ids appear as keys in `probe-laptop.json` `vdb` (18,502 UUID-shaped entries). Omitting vdb from the rewrite set is correct for this copy. Source conversation content still lives under those 49 source-node keys; the plan streams `C` and leaves vdb bytes unchanged.
- **`guardian_nodes` / manifest / guard state:** integer counts (`guardian_nodes` 7253 == `nodes` 7253). `SaveGate.permit` `:451` compares counts. Leaving those files byte-identical is consistent with V1's unchanged node count.
- **Commons:** `commons.msgpack` is one of the six hashed laptop-copy files. It is a separate NGLite medium (`cc_ng_organism.py:1036-1104`). It is outside S1–S13. The whole-string census over the six files is the fail-closed gate: an old id in Commons STOPs the run.
- **Tract:** not in the six-file census. `drain_ingest_tract` feeds conversational text into dual-pass; it does not persist want-node ids. Live tract was not opened (forbidden). Residual: unopened.
- **`cc_topology_export.collect_cc_topology` `:351-416`:** exports CC nodes whose id is absent from the peer membership ledger. New ids will re-send. Plan §6 / Q5 records this.
- **`cc_topology_merge`:** idempotent by `nid in graph.nodes` (loop around `:369-376`). Plan cite `:373-374` is the `if nid in held` fall-through, adjacent to that check.
- **Protected census / #799 export:** no `#799` / `protected_census` hit in the NeuroGraph tree at `e4ebf982`. Plan already marks `#799 export` **[unverified]**.
- **`render_wants` `:1575-1610`**, pith `_pith_node_raw_text` `:4640` / `_pith_node_text` `:4658`, `_CC_PITH_PROVIDER_NODE_CHARS` **`:3511`** (plan says `:3512`), `cc_stamp_missing_geometry` `:3388-3484`: consumers of `want_text` / ids via `graph.nodes`. They follow a re-key automatically. Stamp carries `poincare_dir` as planned.

### Canonical path without editing `neuro_foundation.py`

There is no re-key API. The available canonical pieces the tool can import read-only are: `msgpack.Packer(use_bin_type=True)` at `write_checkpoint` **`:5013-5021`** (cite matches), `checkpoint_guardian.atomic_file_write` `:343-364`, and `Graph.restore` as V11. A full `Graph.checkpoint` round-trip would re-normalize `prediction_outcomes` / delay-buffer / recent-spikes. The value-granular rewrite plus restore-as-verifier is the path that keeps non-want bytes stable without touching the protected file. Same-length 25-character id substitution is true of every derived want id.

**Notes for A1:** S13's file:line points at `capture_state`; the skip is `restore` `:246-250`. S2 volume and S13 "all 182 present" stay unverified from the three packet JSONs. Packet `:2151/:4374` are conversation ids.

---

## A2 The separation algorithm vs the actual `parse_wants`

**Verdict: PASS-WITH-NOTES**

### Contract vs shipped function (`git show 07eeaaaa:cc_ng_organism.py`)

Plan §0 / §8 still describe #810 as `2ab4a851`, "a plan document only; no code exists yet". The function **exists** at `c9fe56d` / review pin `07eeaaaa`. Against that shipped text, the plan's G-* contract holds:

| Guarantee | Shipped |
|---|---|
| G-pure | `parse_wants` `:1682-1734` is a function of the string: no I/O, no log, no graph. Logging is `_log_want_skips` after `surface_wants` releases the lock |
| G-text | `inner = content[pending[1]:m.start()].strip()`; `want_id_for_text(inner)`; no length cap. `_WANT_RE` is gone. `WANT_MAX_CHARS = 600` remains render-only (`:1541`, `render_wants` `:1873`) |
| G-offsets | `WantSpan(inner, want_id, pending[0], m.end())`. `WANT_OPEN` is 6 chars, `WANT_CLOSE` is 7. `content[open_start+6 : close_end-7].strip() == text` holds |
| G-reasons | `WANT_SKIP_REASONS` `:1547-1550` = `in_fence`, `in_code_span`, `code_adjacent`, `escaped`, `quoted`, `opener_unclosed`, `closer_without_opener`, `empty_pair` — the plan's closed set |
| G-pair | nearest live opener via `pending`; a nearer opener emits `opener_unclosed` for the earlier one |
| Fields | `WantSpan(text, want_id, open_start, close_end)`, `SkippedMarker(marker, start, reason)`, `WantParse(wants, skipped)` — exact |

Anchor arithmetic: `C.count(T)==1` at `p`; `C[:p].rstrip()` ends with `[WANT]` at `i`; after `T` plus whitespace, `[/WANT]` at `j`; `closer_end = j + 6 + 1` equals `j+7`. That is the closer length. It works when `j` is the start of `[/WANT]`.

### Decision table

Given `w` = the shipped want with `w.close_end == closer_end`:

- **GENUINE** (`w.open_start == i` and `w.text == T`): the unbounded pre-fix span is a legitimate pair. UNCHANGED. `want_id_for_text(T) == N.id` holds because the old mint was `sha1(strip(inner))` of the same slice.
- **SEPARATE** (`w.open_start > i`): nearest-opener pairing took an inner live opener and the **same** closer. `X = w.text` is the stripped inner of that inner pair, which is a suffix of the stripped outer inner. `T.endswith(X)` and `X in T` hold by construction after `.strip()`. `new_id = want_id_for_text(X)` equals what `surface_wants` will mint. A SEPARATE result is a suffix; it does not keep the old id.
- **ANOMALY** (`w.open_start < i`, or same opener with `w.text != T`): the live pair reaches back past the old span, or the function disagrees with the anchor. UNCHANGED, listed.
- **NONE** (no `w` closes at `j`): every marker in the old span was skipped. UNCHANGED, listed, with the skip reasons.

Class B (no nested marker in `T`) cannot be SEPARATE. Plan table is right.

Post-condition 6 refuses a write when `T.endswith(X)` fails. The algorithm does not name an outcome code for that assertion; a freeze should list it as `assert_failed` rather than crashing the tool.

Whitespace-only prefix (`open_start > i` but `X == T` after strip) is classified SEPARATE then dropped by the collision rule (`new_id` already exists as `N`). Harmless, noisy.

### Honest limits (§3.4) vs shipped behaviour

Paper traces plus checker-016's recorded residuals (JSON object values, JSON-escaped quotes, URLs, markdown links still mint; token-hugging quotes and odd-backslash escape skip) agree with §3.4-1..4. Derived heads: 4 of 118 contain a marker in the first 70 characters; 3 are token-wrapped `('[WANT]')`; 1 is the JSON-escaped opener at length 642. Plan already sends those through frozen-list review.

**HIGH — closer `code_adjacent`.** `_want_marker_mention_reason` `:1668-1669` returns `code_adjacent` whenever `content[start-1] == "`"`, for **every** marker. Base `surface_wants` applied that guard only to the opener (`:1549`). A real pair whose closer sits immediately after a closing backtick is skipped (`code_adjacent` on the closer, `opener_unclosed` on the opener). The repair then sees NONE and leaves the node listed. Conservative (nothing deleted), and it misses SEPARATE/GENUINE for that class. le-014 recorded the same defect as parser C1 on the 810 branch; this review reached it by reading `07eeaaaa` directly. Phase-1 must not freeze against a fingerprint that still applies `code_adjacent` to closers, unless Exec explicitly accepts those NONE rows.

### Five traces (synthetic; plan rule, shipped pairing)

1. **SEPARATE (nested live pair).** `C = "[WANT]swallowed prose [WANT]the real intent[/WANT]"`. Unbounded inner `T` ends with `the real intent`. `parse_wants` skips the outer opener as `opener_unclosed`, mints `X = "the real intent"` with `close_end` at the same closer. `w.open_start > i`. Repair candidate; `new_id = want_id_for_text(X)`.
2. **GENUINE (long legitimate pair, no inner live marker).** `C = "[WANT]" + (700 chars of prose) + "[/WANT]"`. One live pair, `open_start == i`, `text == T`. UNCHANGED. Id already equals `want_id_for_text(T)`.
3. **NONE (pair inside inline code).** `C = "see `[WANT]documented[/WANT]` in the spec"`. Both markers `in_code_span` (or `quoted`/`code_adjacent` depending on wrappers). No `w` closes at `j`. UNCHANGED, listed with those reasons. This is the expected dominant fate of backtick-led class A.
4. **NONE vs quoted (token-hugging).** `C = "use ('[WANT]') and ('[/WANT]') as markers"`. `quoted` fires. NONE. Matches the three derived heads of lengths 18067 / 18357 / 35579.
5. **Residual mint (JSON-escaped, §3.4-2).** `C = "(\"[WANT]I want to feel …[/WANT]\""`. Quote sits before the opener; the next char is `I`, so `quoted` does not fire; `escaped` wants an odd backslash run immediately before `[`. The pair is live → GENUINE or SEPARATE. Frozen-list review is the only mechanical backstop. Matches the one derived head of length 642.

---

## A3 The recount

**Verdict: PASS**

From `want-rows-laptop.json` (182 rows; fields `len`, `backtick_lead`, `has_marker` only; bodies unread beyond structural head flags):

| Claim | Independent figure |
|---|---|
| oversized `len > 600` | **118** (64 ≤ 600) |
| A = backtick-lead and marker | **67** |
| B = backtick-lead, no marker | **16** |
| C = no backtick-lead, has marker | **35** |
| D = neither | **0** |
| marker rows among 182 | **123** |
| backtick-lead among 182 | **91** |
| marker-free oversized | **16** |
| distinct `source_node` among the 118 | **49** (all present, conversational, non-want, non-constitutional) |
| repaired upper bound (A+C) | **102** |
| derived lower bound on any outcome | **none** (every class still admits GENUINE / NONE / ANOMALY; B excludes only SEPARATE) |
| GENUINE upper bound | **≤ 118** |
| newest 40 wants that are in the 118 | **28** |

`summary-laptop.json`: 182 `cc_authored`, 1 constitutional, protected 183, 138,753 synapses, rim degree 4,127. Probe: rim id `constitutional::rim::choice_clause` is outside the 182; rim↔S synapses **131**; synapses touching S **81,999**.

Cannot recompute from these files: the GENUINE / SEPARATE / NONE / ANOMALY split (needs `parse_wants(C)` on source content), collision count, skip-reason histogram, JSON-escaped fence behaviour inside full `C` (heads are 70 chars), S2 pred_weights node count, S13 sidecar key membership, S4/S6–S12 occurrence counts. Plan §8 already says the split is a Phase-1 product.

---

## A4 The execution procedure and verifier

**Verdict: PASS-WITH-NOTES**

Copy-first, hash-verified, one loader, `MemAvailable ≥ 8 GiB`, `systemd-run` 3G classify / 6G V11, `NG_EMBED_REMOTE` unset, `HF_HUB_OFFLINE=1`: executable as a procedure. Writer `atomic_file_write` `:343-364` is tmp+`os.replace` and leaves the final file untouched on `write_fn` failure. Sidecar `write_state` `:162-164` writes in place; routing it through `atomic_file_write` is the right wrap. Phase 2 re-runs on live bytes (daemon down, full backup, mapping must equal the freeze). Two-file `os.replace` (main then sidecar) with idempotent sidecar completion is stated. Rollback is restore-two-files to recorded sha256. SaveGate sees equal counts. V1–V14 plus T6 are mechanically specified. Unit-test matrix covers the outcome codes, collisions, rim synapses, 5,000-character `X`, P1–P5 STOP.

**P1** (record 810 commit + defining-file blob): executable. The blob to record today is `07eeaaaa:cc_ng_organism.py` sha256 `1acb09e2…`.

**P2** (daemon-deployed file sha256 equals the imported file): executable once Q9 names the unit and its WorkingDirectory / `PYTHONPATH`. `cc-ng-daemon.py` on docs `155343e4` really does `from cc_ng_organism import surface_wants` inside the same DEBUG-swallowing `try` as `drain_ingest_tract` (`:2088-2119`; host twin of that shape at `cc_ng_host.py:1518-1534`). P2 hashes the **organism file the unit imports**, which is the right object.

**P3** (golden battery through the imported function): executable as `tests/test_cc_want_legitimacy_810.py` once that file is the P1 pin. The battery today does not include closer-after-backtick (A2 C2). P3 as written would go green on a fingerprint that still maps those genuine pairs to NONE.

**P4 / P5:** pre-apply census that no `new_id` is already present, and freeze tied to P1–P3 hashes, are executable.

Failures if run as written, without the notes below:

1. **Import path (HIGH).** This host exports `PYTHONPATH=/home/josh/NeuroGraph:`. `import cc_ng_organism` binds the primary checkout. At `e4ebf982` that file has no `parse_wants`. Phase 1 "imports the #810-branch function" must pin `sys.path[0]` to the P1 tree, unset `PYTHONPATH` / `NG_EMBED_*`, and print `cc_ng_organism.__file__` plus its sha256 (P379). The plan currently says only "imports it under the memory cap".
2. **Stale function pin.** §3 is "as planned at `2ab4a851`". A freeze that records that commit is the wrong blob.
3. **Q9 still open.** Unit name / slot / retention are Executive questions; P2 cannot be pointed at a file until they are answered. The check itself is well-defined.
4. **V11 restore ≈ 3.6 GiB under 6G** is a prior measurement, not re-run here (plan review; nothing loads).

V9 correctly drops the `<= WANT_MAX_CHARS` assertion. V11 correctly allows the rendered block to grow (`render_wants` still clamps at `:1604` until P408).

---

## A5 Ripple and references outside the graph

**Verdict: PASS-WITH-NOTES**

§6 table is complete for the consumers that actually key or display these nodes:

| Consumer | Plan claim | Check |
|---|---|---|
| `render_wants` `:1575-1610` (host `:1141-1144`) | clamps to 600 / shows 40; 28 of newest 40 are in the 118 | clamp `:1604`, limit `:1603`; **28/40** recomputed |
| pith `:4640-4669`, provider bound default 700 | flagged, not decided | `_CC_PITH_PROVIDER_NODE_CHARS` is **`:3511`** |
| `cc_stamp_missing_geometry` `:3388-3484` | `poincare_dir` carried | stamp reads `want_text` only when geometry is missing; carried bytes skip the embed |
| activation sidecar | re-keyed (S13) | restore skips unknown ids — re-key is required or those 118 go cold |
| guardian / manifest / #743 | counts unchanged | count-based gate |
| protected census | flag-based, 183 | `_is_identity_protected` `:3551-3572` keys on `provenance` suffix `_authored` and `constitutional`; provenance is carried |
| want-hub / analyses by old id | mapping | off-path `id-map-<UTC>.json` |
| vault / Quest | none of the 118 cited; other boards unverified | not re-scanned here; plan already **[unverified]** |
| callosum export/merge | new ids re-sent; peer admits beside old copies | `collect_cc_topology` `:351-416`; merge idempotent by id. Q5 remains Exec |
| legacy twin | parked #755; laptop daemon calls `surface_wants` only | host `:698-700` still calls `surface_wants_for_graph`; daemon `155343e4:2116` calls `surface_wants`. Correction in §1 is right |
| Syl `_surface_wants` | `neurograph_rpc.py:4902` | **def is `:4914`**; unbounded `want::` mint at `:4949`. Punchlist item (1) still valid |

Punchlist items (1)–(5) in §8 are real and correctly not filed by this lane:

1. Unbounded host twin + Syl twin (cites above).
2. Daemon/host DEBUG `try` around `drain_ingest_tract` + `surface_wants` — a raise in the drain skips surfacing.
3. `quoted`/`escaped` miss JSON-escaped `(\"[WANT]…` (one derived head).
4. `ActivationPersistence.write_state` in-place `:162-164`.
5. Dedupe-by-id-only for text-hashed ids (the same continue that forces option (ii)).

Constitutional rim node is outside S. One of the 64 short wants has a head matching leave-ecosystem phrasing; zero of the 118 oversized heads mention choice. Full Choice-Clause identity is ROLE B.

---

## Numbered corrections

Fold these into the plan (or the Phase-1 tool brief) before a freeze. None require editing `neuro_foundation.py`.

1. **HIGH — pin the shipped parser.** Replace §0/§8 "no code yet / `2ab4a851`" with `c9fe56d` (function) / `07eeaaaa` (checker-016 pin) and blob sha256 `1acb09e2b17448a438d8f85fb697b39e439ff2503d70d88f9d88048c2d6cc2c0`. G-* stays valid against that text. A later parser commit invalidates any freeze (already P5).
2. **HIGH — closer `code_adjacent`.** Restrict the fallback to openers in #810 before this repair freezes, **or** name `closer_code_adjacent` as an accepted NONE reason and keep those nodes listed. P3's golden battery must include ``[WANT]check `foo()`[/WANT]``. Sequencing already deploys #810 first; deploying the current closer behaviour would also drop those wants on the live pulse.
3. **HIGH — Phase-1 import isolation.** Tool startup: `env -u PYTHONPATH -u NG_EMBED_REMOTE …`, `sys.path[0]` = directory of the P1 file, print `sys.executable`, `sys.path`, `cc_ng_organism.__file__`, sha256 of that file, and NG-related `sys.modules`. Fail if `__file__` is `/home/josh/NeuroGraph/cc_ng_organism.py`.
4. **MED — assertion-failure outcome.** If `T.endswith(X)` / G-offset equality fails, emit `assert_failed`, leave the node unchanged, list it; do not let a bare `assert` abort a 118-node run after partial writes.
5. **MED — line-cite tidy (non-blocking):** pith bound `:3511`; Syl twin `:4914`/`:4949`; sidecar skip `activation_persistence.py:246-250`; merge idempotency is `nid in graph.nodes` just above `:373`.
6. **LOW — `closer_end = j+7`** in the tool, with a comment that 7 is `len("[/WANT]")`.
7. **LOW — P2** waits on Q9 (unit / slot / retention) before it can be pointed at a path; the hash comparison itself is the right check.

---

## Not verified

- Exact GENUINE / SEPARATE / NONE / ANOMALY split, collisions, skip-reason histogram (needs Phase-1 on a COPY with the pinned function and full `C`).
- S2 pred_weights node count (1,735); S4 `expected_target` hits; S6–S12 occurrence counts; S13 "all 182 sidecar keys present".
- Whether live laptop checkpoint still equals the 09-23 analysis copy (Phase 2 re-derives).
- Daemon unit name, WorkingDirectory, autosave interval (Q9).
- `#799 export` location; vault/Quest board records other than this repo's grep.
- Tract contents (forbidden to open). Commons contents (census is the gate; msgpack not loaded).
- `parse_wants` runtime behaviour (paper trace of `07eeaaaa` only; no import, no pytest).
- V11 RSS ≈ 3.6 GiB (prior measurement).

No graph, checkpoint, or live tract was opened. Primary checkout untouched. Only this reviews file is in scope to commit.
