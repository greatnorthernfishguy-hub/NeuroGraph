# ---- Changelog ----
# [2026-09-30] Claude Sonnet 5.5 (Z12 builder, dispatch #12361) — PG-1 artifact generator
# What: reads the six harness records (4 x part1, 2 x part2) + the gate readings + the source/copy hash records from the z12-pg1-<UTC>
#   scratch directory, copies them into this directory's records/, and writes pg1-artifact.md and pg1-compare.json.
# Why: the artifact's numbers must be copied by code, not transcribed. The per-copy verdict is a CLAIM computed from recorded booleans;
#   it is NOT an acceptance — a separate acceptor verifies it (plan-005 4A.5 "who accepts it").
# How: pure reads of JSON + a write under this directory. No graph is loaded. ids / hashes / counts / booleans only.
# -------------------
import json
import os
import shutil
import subprocess
import sys

SCR = sys.argv[1]
OUT = os.path.dirname(os.path.abspath(__file__))
REC = os.path.join(SCR, "records")
os.makedirs(os.path.join(OUT, "records"), exist_ok=True)
for n in sorted(os.listdir(REC)):
    if n.startswith("smoke-"):
        continue                      # synthetic harness-plumbing smoke records stay in scratch
    shutil.copyfile(os.path.join(REC, n), os.path.join(OUT, "records", n))

J = lambda n: json.load(open(os.path.join(REC, n)))
P1 = {k: J("part1-%s.json" % k) for k in ("a-base", "a-fold", "b-base", "b-fold")}
P2 = {k: J("part2-%s.json" % k) for k in ("first", "fold")}
GATES = {n[len("gate-"):-5]: J(n) for n in sorted(os.listdir(REC)) if n.startswith("gate-")}
G = lambda x: "%.4f" % x
RULED_P1, RULED_P2 = 3.0, 6.0

COMPARED = ["return", "removed_count", "removed_ids_sha256_in_order", "removed_ids_sha256_sorted", "pruned_events", "synapses_before",
            "synapses_after", "pre_state_digest", "state_digest_before_checkpoint", "state_digest", "checkpoint_sha256", "checkpoint_size",
            "counts_after_restore", "counts_final"]


def compare(copy):
    b, f = P1[copy + "-base"], P1[copy + "-fold"]
    eq = {k: b[k] == f[k] for k in COMPARED}
    facts = {
        "base_path_is_own_checkout": (not b["header"]["void"]) and b["header"]["neuro_foundation_file"].startswith("/home/josh/NeuroGraph-worktrees/z12-want-hub-base-e4ebf982/"),
        "fold_path_is_own_checkout": (not f["header"]["void"]) and f["header"]["neuro_foundation_file"].startswith("/home/josh/NeuroGraph-worktrees/z12-pg1-engine-29f47f65/"),
        "different_code_under_test (blob differs, new_api False vs True)": b["header"]["neuro_foundation_blob"] != f["header"]["neuro_foundation_blob"]
        and b["header"]["new_api_present"] is False and f["header"]["new_api_present"] is True,
        "same_ng_tract_wheel": (b["header"]["ng_tract_file"], b["header"]["ng_tract_version"]) == (f["header"]["ng_tract_file"], f["header"]["ng_tract_version"]),
        "same_pythonhashseed_pinned": b["header"]["pythonhashseed"] == f["header"]["pythonhashseed"] == "0",
        "source_main_sha256_equal_before_after (base, fold)": b["source"]["main_equal_before_after"] and f["source"]["main_equal_before_after"],
        "all_sidecars_equal_before_after (base, fold)": b["source"]["all_sidecars_equal_before_after"] and f["source"]["all_sidecars_equal_before_after"],
        "only_write_mode_open_was_the_scratch_checkpoint (base, fold)": b["audit"]["write_mode_opens"] == [b["checkpoint_path"]] and f["audit"]["write_mode_opens"] == [f["checkpoint_path"]]
        and not b["audit"]["mutating_calls"] and not f["audit"]["mutating_calls"],
        "no_oom (cgroup oom_kill 0, base and fold)": "oom_kill 0" in b["memory"]["cgroup_memory_events"] and "oom_kill 0" in f["memory"]["cgroup_memory_events"],
        "removal_set_non_empty (the removal comparison is NOT vacuous)": b["removed_count"] > 0,
        "removal_order_differs_from_id_sorted_order (an unconditional sort-by-id would be visible)": b["removed_ids_sha256_in_order"] != b["removed_ids_sha256_sorted"],
    }
    allfields = all(eq.values())
    must = ["base_path_is_own_checkout", "fold_path_is_own_checkout", "different_code_under_test (blob differs, new_api False vs True)", "same_ng_tract_wheel",
            "same_pythonhashseed_pinned", "source_main_sha256_equal_before_after (base, fold)", "all_sidecars_equal_before_after (base, fold)",
            "only_write_mode_open_was_the_scratch_checkpoint (base, fold)", "no_oom (cgroup oom_kill 0, base and fold)"]
    return eq, facts, allfields and all(facts[m] for m in must)


CMP = {c: compare(c) for c in ("a", "b")}
json.dump({c: {"field_equal": CMP[c][0], "facts": CMP[c][1], "all_compared_fields_identical_and_preconditions_hold": CMP[c][2]} for c in CMP},
          open(os.path.join(OUT, "pg1-compare.json"), "w"), indent=1, sort_keys=True)


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


W = "/home/josh/NeuroGraph-worktrees/"
harness_blob = sh(["git", "hash-object", os.path.join(OUT, "pg1_harness.py")])
L = []
w = L.append
w("# PG-1 artifact — the PRE-MERGE real-graph golden for the (d) engine FOLD, plus the `_step_lock` HOLD (dispatch #12361)")
w("")
w("Lane `want-hub-engine-d-build-20260930` · builder: a FRESH thread independent of the engine author · tests branch `cc-laptop-want-hub-build-20260930` · spec of record plan-005 §4A.5 \"PG-1\" + brief `pg1-want-hub-engine-fold.md`.")
w("**Everything below is a MEASUREMENT by the builder. The per-copy VERDICT lines are CLAIMS for the separate acceptor to verify; this builder does not accept its own artifact.** ids / hashes / counts / booleans only — no want text, no node metadata.")
w("")
w("## 0. What was run")
w("- Checkouts (read-only, detached; `git hash-object` of each `neuro_foundation.py` re-checked after all runs, `git status --porcelain --ignored` empty): "
  "BASE `%s` (`%s`, blob `%s`); FOLD `%s` (`%s`, blob `%s`); FIRST engine commit `%s` (`%s`, blob `%s`)." % (
      W + "z12-want-hub-base-e4ebf982", P1["a-base"]["header"]["git_rev"], P1["a-base"]["header"]["neuro_foundation_blob"],
      W + "z12-pg1-engine-29f47f65", P1["a-fold"]["header"]["git_rev"], P1["a-fold"]["header"]["neuro_foundation_blob"],
      W + "z12-pg1-engine-first-8e578532", P2["first"]["header"]["git_rev"], P2["first"]["header"]["neuro_foundation_blob"]))
w("- Harness: `pg1_harness.py` (git blob `%s`) in this directory; it imports `tests/want_hub_golden_driver.py` BY FILE PATH and reuses its `state_digest`, `instrument`, `ng_tract_info`, `git_rev`, `new_api_present` — so the digest compared is the SAME one Test G makes (`state_digest(g, events)` computed AFTER `Graph.checkpoint()`, as the driver does; the pre-checkpoint digest is recorded as an extra). Only deviation: the checkpoint FILE's sha256 is computed in 8 MiB chunks (a sha256 of the same bytes is the same value; verified independently with `sha256sum`, §2)." % harness_blob)
w("- Every load: its OWN fresh process inside `systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0` (the process read back `memory.max` = 6442450944 and `memory.swap.max` = 0 from its own cgroup), under `env -u NG_EMBED_REMOTE -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 PYTHONHASHSEED=0`; `sys.path[0]` pinned to its own checkout before import; a printed path outside its checkout would be VOID (none was). Canonical `Graph().restore(<copy>/main.msgpack)`; `_prune_synapses()` at ALL new parameters at defaults; `Graph.checkpoint()` to a scratch temp `.msgpack` under `/home/josh/backups/z12-pg1-*/scratch/` (never `save()`, never a live path). A Python audit hook recorded every write-mode `open`/mutating call in each process.")
w("- Copy (a) = the CEREMONY BACKUP `/home/josh/backups/z12-s3-restore-bundle-20260929T222529Z/pre-placement-laptop-cc/` used READ-ONLY. Copy (b) = `vps-pull-staged/` copied to the NEW scratch dir (`%s`), sha256 before/after equal." % os.path.dirname(P1["b-base"]["source"]["copy_dir"]))
w("- The random-uuid-shaped-id class (M07a, le-036/checker-029 C1) is exactly what REAL graph ids exercise: on copy (b) the removal ORDER is not id-sorted order (recorded boolean in §3) and is identical base vs fold, so an unconditional sort-by-id on the default path would have shown. I did not separately verify the uuid4 SHAPE of the real ids.")
w("")
w("## 1. Gate readings immediately before each load (ruled: Part 1 MemAvailable ≥ 3.0 GiB, Part 2 ≥ 6 GiB; load < 6; `cc-ng-daemon.service` inactive and no daemon process)")
w("| load | UTC | MemAvailable GiB | load 1/5/15 | daemon unit | daemon procs | gate applied (GiB) | gate met |")
w("|---|---|---|---|---|---|---|---|")
for k, lab in (("a-base", "(a) BASE"), ("a-fold", "(a) FOLD"), ("b-base", "(b) BASE"), ("b-fold", "(b) FOLD"), ("p2-first", "Part 2 FIRST 8e578532"), ("p2-fold", "Part 2 FOLD 29f47f65")):
    g = GATES[k]
    w("| %s | %s | %s | %s | %s | %d | %s | %s |" % (lab, g["utc"], g["mem_available_gib"], "/".join(g["load_1_5_15"]), g["daemon_unit_state"], g["daemon_processes_found"], g["min_mem_gib"], g["gate_met"]))
w("")
w("Gate at load 1 (a-base) was the ruled 3.0; its first-run heap re-derived the gate to 3.0734 → **applied 3.074 (a-fold) and 3.08 (b loads)** per \"if that exceeds the ruled gate it rises\". Interpretation stated: I read \"if a gate is off STOP\" as \"if MemAvailable is below the (re-derived) gate\"; every reading was ≥ 9.1 GiB, so no stop condition arose.")
w("")
w("## 2. Memory — the org rule is HEAP (`ru_maxrss` / sampled anon), NOT the cache-inclusive cgroup `memory.peak`; both reported")
w("| load | restore-only ru_maxrss GiB | whole-procedure ru_maxrss GiB | sampled RssAnon peak GiB | cgroup memory.peak GiB (cache-incl.) | heap used for gate (max of ru_maxrss, anon) | re-derived gate = heap + 2.0 | ruled gate | effective gate | oom_kill |")
w("|---|---|---|---|---|---|---|---|---|---|")
for k, ruled in (("a-base", RULED_P1), ("a-fold", RULED_P1), ("b-base", RULED_P1), ("b-fold", RULED_P1)):
    m = P1[k]["memory"]
    d = m["heap_for_gate_gib"] + 2.0
    w("| %s | %s | %s | %s | %s | %s | %.4f | %.1f | %.4f | %s |" % (k, G(m["stages"][2]["ru_maxrss_gib"]), G(m["ru_maxrss_gib"]), G(m["sampled_anon_peak_gib"]), G(m["cgroup_memory_peak_gib"]),
                                                                  G(m["heap_for_gate_gib"]), d, ruled, max(ruled, d), m["cgroup_memory_events"].split("|")[-2].strip()))
for k, lab in (("first", "Part 2 first 8e578532"), ("fold", "Part 2 fold 29f47f65")):
    m = P2[k]["memory"]
    d = m["heap_for_gate_gib"] + 2.0
    w("| %s | %s | %s | %s | %s | %s | %.4f | %.1f | %.4f | %s |" % (lab, G(m["stages"][0]["ru_maxrss_gib"]), G(m["ru_maxrss_gib"]), G(m["sampled_anon_peak_gib"]), G(m["cgroup_memory_peak_gib"]),
                                                                  G(m["heap_for_gate_gib"]), d, RULED_P2, max(RULED_P2, d), m["cgroup_memory_events"].split("|")[-2].strip()))
w("")
w("The restore-only column reproduces build-tool-007c's 0.992 GiB (0.995 here); the whole-procedure heap on copy (a) is ~1.075 GiB because PG-1 also builds two state digests and serializes the checkpoint. No load approached the 6 GiB cap. Independent cross-check: `sha256sum` of the four scratch checkpoints equals the in-process values (a-base = a-fold = `%s`; b-base = b-fold = `%s`)." % (P1["a-base"]["checkpoint_sha256"], P1["b-base"]["checkpoint_sha256"]))
w("")
w("## 3. Part 1 — one record per copy × checkout")
for copy, lab in (("a", "(a) laptop CEREMONY-BACKUP copy"), ("b", "(b) staged VPS bundle copy")):
    for co in ("base", "fold"):
        r = P1[copy + "-" + co]
        h = r["header"]
        w("")
        w("### %s × %s" % (lab, co.upper()))
        w("- printed `neuro_foundation.__file__`: `%s`; git rev: `%s`; blob: `%s`; `sys.path[0]`: `%s`; VOID: %s" % (h["neuro_foundation_file"], h["git_rev"], h["neuro_foundation_blob"], h["sys_path_0"], h["void"]))
        w("- `ng_tract` file: `%s`, version `%s` (an installed wheel, not a file of either checkout); `Graph.synapses` type: `%s`; `_prune_synapses` new keyword-only API present: %s" % (h["ng_tract_file"], h["ng_tract_version"], r["counts_after_restore"]["synapses_type"], h["new_api_present"]))
        w("- PYTHONHASHSEED=`%s`; env: NG_EMBED_REMOTE=%s PYTHONPATH=%s PYTHONDONTWRITEBYTECODE=%s HF_HUB_OFFLINE=%s; python %s; pid %d; cgroup `%s`" % (
            h["pythonhashseed"], h["env"]["NG_EMBED_REMOTE"], h["env"]["PYTHONPATH"], h["env"]["PYTHONDONTWRITEBYTECODE"], h["env"]["HF_HUB_OFFLINE"], h["python"], h["pid"], r["cgroup"]["path"]))
        w("- source `main.msgpack`: sha256 before `%s`, after `%s` (equal: %s); all sibling files equal before/after: %s; load start `%s` MemAvailable %s GiB, load %s" % (
            r["source"]["main_sha256_before"], r["source"]["main_sha256_after"], r["source"]["main_equal_before_after"], r["source"]["all_sidecars_equal_before_after"],
            r["gate_reading_at_load_start"]["utc"], r["gate_reading_at_load_start"]["mem_available_gib"], r["gate_reading_at_load_start"]["load_1_5_15"][0]))
        c = r["counts_after_restore"]
        w("- after restore: nodes %d, synapses %d, hyperedges %d, timestep %d; after prune/checkpoint: synapses %d" % (c["nodes"], c["synapses"], c["hyperedges"], c["timestep"], r["counts_final"]["synapses"]))
        w("- `_prune_synapses()` return: **%d**; removed ids: %d; removed-id sha256 in removal order `%s`; sorted `%s` (equal: %s); `pruned` events `%s`; default-path wall %.2f s" % (
            r["return"], r["removed_count"], r["removed_ids_sha256_in_order"], r["removed_ids_sha256_sorted"], r["removed_ids_sha256_in_order"] == r["removed_ids_sha256_sorted"], json.dumps(r["pruned_events"]), r["prune_default_path_wall_s"]))
        w("- pre-prune state digest `%s`; post-state digest before checkpoint `%s`; **full state digest (Test G order) `%s`**" % (r["pre_state_digest"], r["state_digest_before_checkpoint"], r["state_digest"]))
        w("- serialized `Graph.checkpoint()` → scratch temp `.msgpack`: %d bytes, sha256 `%s`" % (r["checkpoint_size"], r["checkpoint_sha256"]))
        w("- audit: write-mode opens `%s`; mutating os/shutil calls `%s`; programs spawned `%s`" % (r["audit"]["write_mode_opens"], r["audit"]["mutating_calls"], r["audit"]["spawned_programs"]))
w("")
w("## 4. Part 1 comparison and per-copy VERDICT (a CLAIM — for the acceptor to verify against `records/` and `pg1-compare.json`)")
for copy, lab in (("a", "(a) laptop ceremony-backup copy"), ("b", "(b) staged VPS bundle copy")):
    eq, facts, ok = CMP[copy]
    w("")
    w("**%s** — field-by-field BASE vs FOLD equal: %s" % (lab, ", ".join("`%s`=%s" % (k, v) for k, v in eq.items())))
    w("Preconditions: " + "; ".join("%s=%s" % (k, v) for k, v in facts.items()))
    if copy == "a":
        w("Non-vacuity note: on this copy the default path removes **0** synapses (as the plan predicts: 0 eligible), so the removed-id and removal-order comparisons here are comparisons of two EMPTY sets; what this copy proves is that every counter the function advances (`low_weight_steps` on all %d evaluated non-protected synapses) and the full state and serialized bytes are identical. The load-bearing removal comparison is copy (b)." % (P1["a-base"]["counts_after_restore"]["synapses"]))
    else:
        w("Non-vacuity note: this copy removes **%d** synapses on BOTH checkouts with identical removal order, identical `pruned` event and identical post-state and serialized bytes (the plan's expected count is 10,433)." % P1["b-base"]["removed_count"])
    w("**VERDICT (claim, builder): %s** — %s" % ("PASS" if ok else "FAIL", "all %d compared fields identical between BASE `e4ebf982` and FOLD `29f47f65` and every precondition above holds." % len(eq) if ok else "see the False entries above."))
w("")
w("## 5. Part 2 — the `_step_lock` HOLD on the laptop copy (K = 50, B = 5000; in-memory scratch call, nothing saved)")
w("Each variant in its own fresh process/scope after ONE canonical restore of copy (a) (`main.msgpack` sha256 equal before/after in both). `compete_protected_links` holds `graph._step_lock` for its whole body, so its wall time IS the hold seen by `step()` / Door B / StreamParser / Commons. Calls 2–3 run on the already-pruned scratch graph (each call removed 5,000 and advanced `low_weight_steps` — nothing is persisted). `perf_counter` = wall, `process_time` = CPU. One process per variant, host load ~3–4 with other threads active: wall and CPU are both given because CPU is less sensitive to contention.")
w("| variant | call | wall s | cpu s | inside `_prune_synapses` wall s | outside it (orchestrator) wall s | eligible | removed | conducting removed | held-back last-link | floors_ok | synapses after |")
w("|---|---|---|---|---|---|---|---|---|---|---|---|")
for k, lab in (("first", "FIRST 8e578532"), ("fold", "FOLD 29f47f65")):
    for c in P2[k]["calls"]:
        w("| %s | %d | %s | %s | %s | %s | %d | %d | %d | %d | %s | %d |" % (lab, c["call"], c["wall_s"], c["cpu_s"], c["inner_prune_wall_s"], c["outside_prune_wall_s"], c["eligible"], c["removed"], c["conducting_links_removed"], c["held_back_last_link"], c["floors_ok"], c["synapses_after"]))
w("")
w("Fold-minus-first (same call index): " + "; ".join(
    "call %d: wall %+.3f s, cpu %+.3f s, inside-prune wall %+.3f s" % (i + 1, P2["fold"]["calls"][i]["wall_s"] - P2["first"]["calls"][i]["wall_s"],
                                                                      P2["fold"]["calls"][i]["cpu_s"] - P2["first"]["calls"][i]["cpu_s"],
                                                                      P2["fold"]["calls"][i]["inner_prune_wall_s"] - P2["first"]["calls"][i]["inner_prune_wall_s"]) for i in range(3)) + ".")
w("")
w("**Orchestrator before the prune call** (two `_plan()` builds + floor check + capture map, measured by aborting at the `_prune_synapses` call WITHOUT mutating anything; identical code in both variants): FIRST wall %s s / cpu %s s; FOLD wall %s s / cpu %s s." % (
    P2["first"]["orchestrator_pre_prune_wall_s"], P2["first"]["orchestrator_pre_prune_cpu_s"], P2["fold"]["orchestrator_pre_prune_wall_s"], P2["fold"]["orchestrator_pre_prune_cpu_s"]))
w("")
w("**The validation pass in isolation** (`_prune_synapses` called with the orchestrator's own captured `competing_ids` ∪ one absent id that sorts last, so it raises at the END of its pre-loop validation having validated every real id; state fingerprint unchanged — a refusal mutates nothing; harness-side only, no engine edit; ×3 each):")
w("| variant | run | wall s | cpu s | raised at the absent id | state fingerprint unchanged |")
w("|---|---|---|---|---|---|")
for k, lab in (("first", "FIRST 8e578532"), ("fold", "FOLD 29f47f65")):
    for i, v in enumerate(P2[k]["validation_isolated"]["runs"]):
        w("| %s | %d | %s | %s | %s | %s |" % (lab, i + 1, v["wall_s"], v["cpu_s"], v["raised_at_the_absent_id"], P2[k]["validation_isolated"]["state_fingerprint_unchanged"]))
fv = sum(v["cpu_s"] for v in P2["fold"]["validation_isolated"]["runs"]) / 3.0
iv = sum(v["cpu_s"] for v in P2["first"]["validation_isolated"]["runs"]) / 3.0
w("")
w("Mean validation CPU: FIRST %.3f s, FOLD %.3f s → fold-minus-first **%+.3f s** (the fold adds per-id tuple/kind checks to the pre-loop validation; the first commit checks only membership in `order_key`).  Counts (identical for both variants): `competing_ids` = %d, `excluded_ids` = %d, `order_key` entries = %d, `max_removals` passed = %d; call-1 `eligible` = %d (⇒ ceil(eligible/B) = %d cycles); `F_links` = %d; `protected_nodes` = %d; `held_back_last_link` = %d." % (
    iv, fv, fv - iv, P2["fold"]["competing_ids"], P2["fold"]["excluded_ids"], P2["fold"]["order_key_entries"], P2["fold"]["max_removals_passed"],
    P2["fold"]["calls"][0]["eligible"], -(-P2["fold"]["calls"][0]["eligible"] // 5000), P2["fold"]["calls"][0]["F_links"], P2["fold"]["calls"][0]["protected_nodes"], P2["fold"]["calls"][0]["held_back_last_link"]))
w("")
w("**What the hold means (no judgement on acceptability — the Executive decides, C8):** while `compete_protected_links` runs, every other holder of `graph._step_lock` waits — `step()`, Door B, StreamParser's `_nudge_nodes`/`_trigger_completions`, the Commons leg-2 read. On the laptop copy a call holds the lock for roughly 12–15 s (table above), of which about 10 s is the orchestrator's own set-building before `_prune_synapses` is reached. The pass is idle-gated (≥ 1,800 s idle), so a turn arriving mid-pass waits up to that long. The fold's marginal cost is the validation delta above plus sort/inside-prune differences; the fold-minus-first wall difference is of the SAME SIZE as the call-to-call spread within one process (call 3 of the FIRST variant was slower than call 3 of the FOLD), so only the CPU-time validation delta is steady.")
w("")
w("## 6. Observations to flag (not rulings)")
w("- **Plan figure vs engine count:** plan-005 §4A.3 says 16 partners are held back at K = 50 (competing 106,841 → 106,825). Both engine variants report `held_back_last_link` = %d and `competing_ids` = %d (equal to the plan's 106,825) — i.e. one fewer held-back link and, implicitly, one fewer link in the pre-hold set than the plan's probe figures. Not explained here; PG-1 (default path) is unaffected. For the plan owner / dry run." % (P2["fold"]["calls"][0]["held_back_last_link"], P2["fold"]["competing_ids"]))
w("- The re-serialized checkpoint of copy (a) has the SAME byte size as the source `main.msgpack` (230,539,966) but a different sha256 (`%s` vs source `%s`): expected for a re-save, noted only because the size coincidence is exact." % (P1["a-base"]["checkpoint_sha256"][:16] + "…", P1["a-base"]["source"]["main_sha256_before"][:16] + "…"))
w("")
w("## 7. NOT verified here")
w("Real arming behaviour; the daemon slice and its dream-loop wiring; #825 (pruned links staying pruned across save AND restore — PG-1 never calls `save()` and does not restore the checkpoint it writes); concurrency (no other thread contends for `_step_lock` in these runs, so the table is an uncontended hold); the post-merge dry run (items 1–12); a repeat of Part 2 for run-to-run variance (the brief asked for one process per variant); the uuid4 SHAPE of real ids; that Syl's own graph is never loaded (copy (b) is a staged VPS-bundle copy named by the plan; nothing from `~/NeuroGraph/data/checkpoints` was opened).")
open(os.path.join(OUT, "pg1-artifact.md"), "w").write("\n".join(L) + "\n")
print("wrote", os.path.join(OUT, "pg1-artifact.md"), {c: CMP[c][2] for c in CMP})
