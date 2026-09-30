STATUS: INCOMPLETE - review in progress

# le-035 — ROLE B delta review: want-repair delta stream reader (build-tool-007b)

Reviewer: le-035 (fresh LAW ENFORCER, report_only). Zone Z12, lane z12-s3-restore-bundle-20260929, dispatch #11860.
Agent definition: /home/josh/.claude/agents/neurograph-law-enforcer.md, sha256 verified = 6daf1621b844b9b72d567b329f2c9f40ca0b4516744608c125147e601c4acf23 (matches the required value).
Delta: git diff 09a032c3426baf8307cb968496c65f779894304b 3cd271b2a79657afe5a6feeee7454f04b4e479d3 (tool + test files). Tool worktree head at start: 3cd271b2a79657afe5a6feeee7454f04b4e479d3.

Independence: checker-028 not opened before this draft is committed (see Independence log at the end).

## COMPLIANCE VERDICT
(INTERIM - code-read findings committed; own-world run, fail-closed probes and the test run still pending)

## ROLE B checks (ADDENDUM 5, checks 1-10 with the LAWS)
Headings from ADDENDUM 5 of review-packet-want-text-repair-118.md. Filled in as the review proceeds.

### Check 1 — Byte-identity (faithful OLD copy? own world)
(pending)
### Check 2 — `incident_figures` equivalence (Graph._deserialize reading; V11 'before' figures)
FIRST FINDINGS (code-read; own-world run follows). **PASS (code-read).**
- `stream_incident_figures` (tool :1637-1663) vs pin `Graph._deserialize`: `_outgoing/_incoming` come from `for sid in self.synapses.keys(): ref=self.synapses[sid]; _outgoing.setdefault(ref.pre_node_id,set()).add(sid)` (pin neuro_foundation.py:5454-5457); the tool keys each endpoint by the synapse MAP KEY (`eid`) into a per-id set, so counts are distinct synapse ids - same. `_node_hyperedges` is built at :5487 from `data["hyperedges"]` ONLY (`setdefault(nid,set()).add(hid)`), and `archived_hyperedges` (:5495-5517) never touches it; the tool descends `hyperedges`, skips `archived_hyperedges`, and does NOT filter on the stored `is_archived` flag - identical (the canonical also indexes a `hyperedges` entry whose stored flag says archived). `set(val["member_nodes"])` = `set(hd["member_nodes"])` (:5463); a duplicate member id collapses in both. A non-node endpoint / member is ignored by the tool (`& want`, `want = ids & existing_ids`) and by the old `incident_figures` (`if i in g.nodes`) - same.
- Nothing after :5487 in `_deserialize` (tail :5520+, read) deletes or rewrites synapses or hyperedges or the three indexes (no orphan sweep): the figures are exactly counts of stored fields. `Graph.restore` (:5039-5089) only slices `synapses` and skips `he_prediction_window_fired`; it does not alter either section.
- V11 reads `A["before_figures"]` in exactly ONE place (tool :2040, `for o, n in m.items()`); `before_fig` is otherwise only assigned into `A` (:1739). The mapping `m` is a subset of the candidates, which are a subset of S, so every key V11 reads is present with the same value; the extra keys (S minus candidates, and the three watch ids) are never consumed (the watch figures were unread before too). Never serialised into any report. Computing them BEFORE classification is independent of `cand_old` - correct.
- Residual (not a finding against the code, see Check 9): the native `SynapseStore.bulk_load_msgpack` (Rust, not in the pin tree) is assumed to key the store by the map key and expose `pre_node_id/post_node_id` unchanged; only a real-file run can close that.
### Check 3 — Fail-closed / LAW 7 'raw means complete' (+ disclosed behaviour differences a-c)
(pending)
### Check 4 — Old path gone, no shrapnel (LAW 3)
FIRST FINDINGS. **PASS.**
- `grep -n "SimpleVectorDB\|load_pair\|vdb\.load\|vdb\.content\|incident_figures(g"` on the tool: hits only on `#` comment lines (changelog header 5, 11, 14, 29); **0 non-comment hits**.
- `grep -n "\.restore("` in code: exactly ONE, `:2651 g2.restore(out_main)` (V11, `build_outputs`).
- `git diff -U0 09a032c 3cd271b` on the tool: hunks only in the header and `:1559-1740` (the old loader/figures region and `analyze`); `load_pair` and `incident_figures` are DELETED, not left beside the new readers; no second parser; the tool stays under `oneshot-tool/`; the branch touches only the tool, its test file and the return (3 files: `git diff --stat 09a032c 3cd271b`). LAW 3 holds: modified in place, no shrapnel.
- LAW 3 note (not a violation): the three new readers are one-shot-tool-local, ~100 lines, with no canonical streaming API in the pin to reuse (I re-read `universal_ingestor.py:548-605`: `msgpack.unpack(f, raw=False)` + per-entry `np.frombuffer`, no iterator; `Graph.restore` :5039 streams the outer map but always builds the Graph) - the builder's "NONE EXISTS" answer to brief item 1 is accurate.
### Check 5 — V11 untouched and still a GATE
FIRST FINDINGS. **PASS.**
- The diff hunks (`git diff -U0`) end at tool line ~1740; V11's verifier block (`# V11 canonical restore`, tool :2033-2055), `build_outputs` (:2545-2651 incl. `g2 = pinned.nf.Graph(); g2.restore(out_main)` :2650-2651) and the verifier list are NOT in the diff: unchanged in behaviour (byte-for-byte untouched).
- V11 still consumes the canonical `g2` (`_outgoing/_incoming/_node_hyperedges`, dangling sweep, `render_wants(g2)`), compares against `A["before_figures"]`, and is `ok11 = not bad_fig and dangling == 0 and len(g2.nodes) == W["nodes_b"]` - still a GATE. Nothing stubs, weakens or skips it.
- Consequence worth stating (LAW 7 / gate logic): Phase 1 `analyze()` no longer runs the canonical restore on the INPUT, so an input that canonical `Graph.restore` would reject (e.g. a node missing `voltage`/`threshold`, a bad `activation_mode`) is not detected at analyze time; it is detected at V11 only transitively (the OUTPUT restore of a rewrite of that input fails), and in Phase 2 by the live restore. This is acceptable because V11 remains a blocking gate, but it is a real weakening of early detection - see Check 9 item (ii).
### Check 6 — LAW 4 / LAW 2 (canonical untouched, pin clean, no vendored/protected, no pre-flight change)
FIRST FINDINGS. **PASS.**
- LAW 4: canonical `SimpleVectorDB.load` (pin `universal_ingestor.py:548-605`) and `Graph.restore/_deserialize` (pin `neuro_foundation.py:5039-5089, 5368-5517`) are read-only reference; the tool neither edits nor wraps them (the new readers are independent tool-local code that replaces the tool's call; nothing monkeypatches canonical code).
- Pin worktree: `git rev-parse HEAD` = ae798b94cb14740d200fc3f4fd8d36eef8b86c6a; `git status --porcelain --ignored` = 0 lines; `sha256sum cc_ng_organism.py` = 8ad0f69ed4a7849c98e97e0f3970bbe4a36c727ef880455e8ddecec14de5a8e2 (matches). I never wrote there.
- LAW 2 / protected: `git diff --name-only 09a032c 3cd271b` = the tool, its test and the return only; none of ng_lite, ng_tract_bridge, ng_ecosystem, openclaw_adapter, ng_autonomic, ng_embed, ng_salience_gate, ng_updater, neuro_foundation, openclaw_hook, stream_parser, activation_persistence, universal_ingestor is touched; no new vendored file.
- Pre-flight: `grep -ciE "MemAvailable|loadavg|preflight|MemoryMax|MemorySwap"` on the tool = 0 (the pre-flight is an operator step, not tool code), and the delta has no such hunk.
- LAW 1/5/8: no inter-module call, no new config mechanism or hardcoded config beyond the existing tool constants (`MAIN_NAME`, `VECTORS_NAME`), no autonomic path. LAW 6: a bespoke reader of a canonical format is the price of the ruling (Chief-003), not a normalisation.
### Check 7 — Memory evidence honesty (shape check, nothing lowered)
(pending)
### Check 8 — Mutation / discrimination
(pending)
### Check 9 — Worker's disclosures and what it did NOT verify
(pending)
### Check 10 — Independence
(pending)

## Law Violations
(pending)
## Ethos Drift
(pending)
## Correct Implementations
(pending)
## Remediation Priority
(pending)
## Recommended Next Steps
(pending)
## Independence log
- Draft not yet committed. checker-028 not opened.
