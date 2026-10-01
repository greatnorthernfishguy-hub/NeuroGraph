# ---- Changelog ----
# [2026-09-30] Claude Sonnet 5.5 (Z12 builder, dispatch #12361, lane want-hub-engine-d-build-20260930) — PG-1 real-graph harness
# What: NEW script (NOT a test module). Sub-commands: gate (load gate reading), hash (sha256/stat of a copy), copy (stable-source copy
#   with sha256 before/after), part1 (ONE real-graph default-path load: restore -> pre digest -> _prune_synapses() at all-defaults ->
#   Graph.checkpoint() to a scratch temp .msgpack -> post digest), part2 (the _step_lock HOLD timing of compete_protected_links(50, 5000)).
# Why: plan-005 sec 4A.5 "PG-1" + the pg1-want-hub-engine-fold brief: the default path of the edited _prune_synapses must be identical to
#   base e4ebf982 on REAL graphs, and the fold's hold time on the laptop copy must be measured (le-036 C8). Hashing / serialization is the
#   driver's (tests/want_hub_golden_driver.py: state_digest, instrument, ng_tract_info, git_rev, new_api_present), imported BY FILE PATH
#   so the comparison is the SAME one Test G makes. The only deviation: the sha256 of the checkpoint FILE is computed in 8 MiB chunks
#   (the driver's sha256_file reads the whole file into memory); a sha256 of the same bytes is the same value.
# How: sys.path[0] is pinned to --checkout BEFORE neuro_foundation is imported (the driver file is loaded with importlib so the tests
#   worktree, which carries its own neuro_foundation.py, never gets on sys.path); a run is VOID (exit 3) if the imported module is not
#   inside --checkout. A Python audit hook records every write-mode open / mutating os call / subprocess so the artifact can show the
#   only path written is the scratch checkpoint. Output: ids / hashes / counts / booleans only — never node metadata, never want text.
#   Nothing here calls save(); the only checkpoint() target is a scratch temp .msgpack the caller names under the z12-pg1-<UTC> dir.
# -------------------
import argparse
import hashlib
import importlib.util
import json
import os
import resource
import shutil
import subprocess
import sys
import threading
import time

DRIVER = "/home/josh/NeuroGraph-worktrees/z12-want-hub-build-20260930/tests/want_hub_golden_driver.py"
GIB = 1024.0 ** 3
_O_WRITE_BITS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_chunked(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(8 * 1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def stat_row(path):
    st = os.stat(path)
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "inode": st.st_ino}


def meminfo():
    out = {}
    with open("/proc/meminfo") as f:
        for ln in f:
            k, v = ln.split(":", 1)
            out[k] = int(v.split()[0])
    return out


def gate_reading():
    mi = meminfo()
    load = open("/proc/loadavg").read().split()[:3]
    d = subprocess.run(["systemctl", "--user", "is-active", "cc-ng-daemon.service"], capture_output=True, text=True)
    p = subprocess.run(["pgrep", "-af", "cc-ng-daemon|neurograph_rpc"], capture_output=True, text=True)
    procs = [ln for ln in p.stdout.splitlines() if "pgrep" not in ln]
    return {"utc": utc(), "mem_available_kb": mi["MemAvailable"], "mem_available_gib": round(mi["MemAvailable"] / 1048576.0, 3),
            "load_1_5_15": load, "daemon_unit_state": d.stdout.strip(), "daemon_processes_found": len(procs)}


def cgroup_paths():
    cg = open("/proc/self/cgroup").read().strip().split("::")[1]
    base = "/sys/fs/cgroup" + cg
    return cg, base


def cgroup_read(base, name):
    try:
        return open(base + "/" + name).read().strip().replace("\n", " | ")
    except Exception as exc:
        return "ERR:%s" % type(exc).__name__


def proc_status():
    out = {}
    with open("/proc/self/status") as f:
        for ln in f:
            if ln.startswith(("VmRSS", "RssAnon", "RssFile", "VmHWM")):
                k, v = ln.split(":", 1)
                out[k] = int(v.split()[0])
    return out


class Mem(object):
    """Heap measurement for the org rule: ru_maxrss + a sampled RssAnon peak (NOT the cache-inclusive cgroup memory.peak)."""

    def __init__(self):
        self.peak_anon_kb = 0
        self.peak_rss_kb = 0
        self.stages = []
        self._stop = False
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        while not self._stop:
            s = proc_status()
            self.peak_anon_kb = max(self.peak_anon_kb, s.get("RssAnon", 0))
            self.peak_rss_kb = max(self.peak_rss_kb, s.get("VmRSS", 0))
            time.sleep(0.05)

    def stage(self, name):
        s = proc_status()
        self.peak_anon_kb = max(self.peak_anon_kb, s.get("RssAnon", 0))
        self.peak_rss_kb = max(self.peak_rss_kb, s.get("VmRSS", 0))
        self.stages.append({"stage": name, "utc": utc(), "ru_maxrss_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576.0, 4),
                            "rss_anon_now_gib": round(s.get("RssAnon", 0) / 1048576.0, 4), "rss_now_gib": round(s.get("VmRSS", 0) / 1048576.0, 4),
                            "sampled_anon_peak_gib": round(self.peak_anon_kb / 1048576.0, 4)})

    def final(self, base):
        self._stop = True
        self.stage("final")
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return {"ru_maxrss_gib": round(ru / 1048576.0, 4), "ru_maxrss_kb": ru, "sampled_anon_peak_gib": round(self.peak_anon_kb / 1048576.0, 4),
                "sampled_rss_peak_gib": round(self.peak_rss_kb / 1048576.0, 4),
                "heap_for_gate_gib": round(max(ru, self.peak_anon_kb) / 1048576.0, 4),
                "cgroup_memory_peak_gib": _gib(cgroup_read(base, "memory.peak")), "cgroup_memory_peak_bytes": cgroup_read(base, "memory.peak"),
                "cgroup_memory_events": cgroup_read(base, "memory.events"), "stages": self.stages}


def _gib(s):
    try:
        return round(int(s) / GIB, 4)
    except Exception:
        return None


class Audit(object):
    """Records write-mode opens, mutating os/shutil calls and spawned programs (names only) — the artifact's evidence that nothing
    but the scratch checkpoint was written."""

    def __init__(self):
        self.write_opens = []
        self.mutations = []
        self.spawned = []
        self.read_opens = set()
        sys.addaudithook(self._hook)

    def _hook(self, event, args):
        try:
            if event == "open":
                path, mode, flags = args
                p = os.fsdecode(path) if not isinstance(path, int) else "fd:%d" % path
                w = bool(flags & _O_WRITE_BITS) if isinstance(flags, int) else False
                if not w and isinstance(mode, str) and any(c in mode for c in "wax+"):
                    w = True
                if w:
                    self.write_opens.append(p)
                else:
                    self.read_opens.add(p)
            elif event in ("os.remove", "os.rename", "os.mkdir", "os.rmdir", "os.truncate", "os.chmod", "os.symlink", "os.link",
                           "shutil.copyfile", "shutil.copytree", "shutil.rmtree", "shutil.move"):
                self.mutations.append([event] + [os.fsdecode(a) if isinstance(a, (str, bytes)) else repr(a) for a in args[:2]])
            elif event == "subprocess.Popen":
                self.spawned.append(os.fsdecode(args[0]) if isinstance(args[0], (str, bytes)) else repr(args[0]))
        except Exception:
            pass

    def summary(self, checkout):
        pref = (sys.prefix, "/usr/lib/python3", "/usr/lib/x86_64-linux-gnu", "/home/josh/.local/lib/python3.12", checkout, "/proc/", "/sys/",
                "/dev/", "/etc/", "/lib/")
        data_reads = sorted(p for p in self.read_opens if not p.startswith(pref) and not p.endswith((".py", ".pyc", ".so", ".pth")))
        return {"write_mode_opens": sorted(set(self.write_opens)), "mutating_calls": self.mutations, "spawned_programs": sorted(set(self.spawned)),
                "non_library_read_opens": data_reads}


def load_driver():
    spec = importlib.util.spec_from_file_location("want_hub_golden_driver", DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pin_and_import(checkout):
    """Pin sys.path[0] to the checkout, import neuro_foundation, and return (nf, header). Exit 3 (VOID) when the path is not its own."""
    checkout = os.path.realpath(checkout)
    sys.path.insert(0, checkout)
    assert sys.path[0] == checkout
    import random
    random.seed(0)
    try:
        import numpy
        numpy.random.seed(0)
    except Exception:
        pass
    import neuro_foundation as nf
    nf_file = os.path.realpath(nf.__file__)
    hdr = {"checkout": checkout, "neuro_foundation_file": nf_file, "sys_path_0": sys.path[0], "sys_path_1": sys.path[1] if len(sys.path) > 1 else None}
    hdr["void"] = os.path.dirname(nf_file) != checkout
    return nf, hdr


def common_header(drv, checkout, nf, hdr):
    tf, tv = drv.ng_tract_info()
    hdr.update({
        "git_rev": drv.git_rev(checkout),
        "neuro_foundation_blob": subprocess.run(["git", "-C", checkout, "hash-object", hdr["neuro_foundation_file"]], capture_output=True, text=True).stdout.strip(),
        "ng_tract_file": tf, "ng_tract_version": tv,
        "new_api_present": drv.new_api_present(nf),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"), "python": sys.version.split()[0],
        "env": {"NG_EMBED_REMOTE": os.environ.get("NG_EMBED_REMOTE"), "PYTHONPATH": os.environ.get("PYTHONPATH"),
                "PYTHONDONTWRITEBYTECODE": os.environ.get("PYTHONDONTWRITEBYTECODE"), "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE")},
        "pid": os.getpid(),
    })
    return hdr


def counts(g):
    return {"nodes": len(g.nodes), "synapses": len(g.synapses), "hyperedges": len(g.hyperedges), "timestep": g.timestep,
            "synapses_type": "%s.%s" % (type(g.synapses).__module__, type(g.synapses).__qualname__)}


def src_sidecars(d):
    """sha256 of every small sibling file + stat of every file (the big vectors file is hashed by the `hash` sub-command, not per load)."""
    rows = {}
    for n in sorted(os.listdir(d)):
        p = os.path.join(d, n)
        if os.path.isfile(p):
            rows[n] = dict(stat_row(p))
            if n != "vectors.msgpack":
                rows[n]["sha256"] = sha256_chunked(p)
    return rows


# --------------------------------------------------------------------------------------------------------------------
def cmd_gate(a):
    r = gate_reading()
    ok = (r["mem_available_gib"] >= a.min_gib and float(r["load_1_5_15"][0]) < a.max_load
          and r["daemon_unit_state"] == "inactive" and r["daemon_processes_found"] == 0)
    r.update({"gate_name": a.name, "min_mem_gib": a.min_gib, "max_load": a.max_load, "gate_met": ok})
    if a.out:
        json.dump(r, open(a.out, "w"), indent=1, sort_keys=True)
    print(json.dumps(r, sort_keys=True))
    return 0 if ok else 4


def cmd_hash(a):
    rows = {}
    for n in sorted(os.listdir(a.dir)):
        p = os.path.join(a.dir, n)
        if os.path.isfile(p):
            rows[n] = dict(stat_row(p), sha256=sha256_chunked(p))
    r = {"utc": utc(), "dir": a.dir, "files": rows}
    if a.out:
        json.dump(r, open(a.out, "w"), indent=1, sort_keys=True)
    print(json.dumps(r, sort_keys=True))
    return 0


def cmd_copy(a):
    """Copy a source dir (files only; sub-directories are left behind and named) to a NEW dst dir, sha256 before/after, equal."""
    src, dst = a.src, a.dst
    assert not os.path.exists(dst), "dst must be new: %s" % dst
    names = sorted(n for n in os.listdir(src) if os.path.isfile(os.path.join(src, n)))
    skipped_dirs = sorted(n for n in os.listdir(src) if os.path.isdir(os.path.join(src, n)))
    stab1 = {n: stat_row(os.path.join(src, n)) for n in names}
    time.sleep(3)
    stab2 = {n: stat_row(os.path.join(src, n)) for n in names}
    newest_age = time.time() - max(v["mtime_ns"] for v in stab2.values()) / 1e9
    src_before = {n: sha256_chunked(os.path.join(src, n)) for n in names}
    os.makedirs(dst)
    for n in names:
        shutil.copyfile(os.path.join(src, n), os.path.join(dst, n))
    dst_after = {n: sha256_chunked(os.path.join(dst, n)) for n in names}
    src_after = {n: sha256_chunked(os.path.join(src, n)) for n in names}
    r = {"utc": utc(), "src": src, "dst": dst, "files": {n: {"size": stab2[n]["size"], "src_sha256_before": src_before[n], "dst_sha256": dst_after[n],
                                                          "src_sha256_after": src_after[n]} for n in names},
         "source_stable_across_3s": stab1 == stab2, "newest_source_mtime_age_s": round(newest_age),
         "all_equal": all(src_before[n] == dst_after[n] == src_after[n] for n in names), "subdirs_not_copied": skipped_dirs}
    json.dump(r, open(a.out, "w"), indent=1, sort_keys=True)
    print(json.dumps(r, sort_keys=True))
    return 0 if r["all_equal"] and r["source_stable_across_3s"] else 5


def cmd_part1(a):
    audit = Audit()
    drv = load_driver()
    mem = Mem()
    cg, base = cgroup_paths()
    rec = {"mode": "part1", "label": a.label, "start_utc": utc(), "gate_reading_at_load_start": gate_reading(),
           "cgroup": {"path": cg, "memory.max": cgroup_read(base, "memory.max"), "memory.swap.max": cgroup_read(base, "memory.swap.max")}}
    nf, hdr = pin_and_import(a.checkout)
    rec["header"] = common_header(drv, hdr["checkout"], nf, hdr)
    mem.stage("after_import")
    if hdr["void"]:
        rec["VOID"] = True
        print(json.dumps(rec, sort_keys=True))
        return 3
    main = os.path.join(a.copy_dir, "main.msgpack")
    rec["source"] = {"copy_dir": a.copy_dir, "main_msgpack": main}
    rec["source"]["sidecars_before"] = src_sidecars(a.copy_dir)
    rec["source"]["main_sha256_before"] = rec["source"]["sidecars_before"]["main.msgpack"]["sha256"]
    g = nf.Graph()
    g.restore(main)
    mem.stage("after_restore")
    rec["counts_after_restore"] = counts(g)
    rec["pre_state_digest"] = drv.state_digest(g, None)
    mem.stage("after_pre_digest")
    order, events = drv.instrument(g)
    before_ids = len(g.synapses)
    t0 = time.perf_counter()
    ret = g._prune_synapses()
    rec["prune_default_path_wall_s"] = round(time.perf_counter() - t0, 4)
    mem.stage("after_prune")
    del g._remove_synapse_internal  # drop the recording wrapper before serializing (as the driver does)
    rec["synapses_before"] = before_ids
    rec["synapses_after"] = len(g.synapses)
    rec["return"] = ret
    rec["removed_count"] = len(order)
    rec["removed_ids_sha256_in_order"] = hashlib.sha256("\n".join(order).encode()).hexdigest()
    rec["removed_ids_sha256_sorted"] = hashlib.sha256("\n".join(sorted(order)).encode()).hexdigest()
    rec["pruned_events"] = events
    rec["state_digest_before_checkpoint"] = drv.state_digest(g, events)
    ck = a.ckpt
    assert ck.endswith(".msgpack") and "/z12-pg1-" in ck and "/data/checkpoints" not in ck, "checkpoint target must be a scratch temp .msgpack"
    g.checkpoint(ck)
    mem.stage("after_checkpoint")
    rec["checkpoint_path"] = ck
    rec["checkpoint_size"] = os.path.getsize(ck)
    rec["checkpoint_sha256"] = sha256_chunked(ck)
    rec["state_digest"] = drv.state_digest(g, events)          # the Test G order: checkpoint FIRST, then the digest
    rec["counts_final"] = counts(g)
    rec["source"]["sidecars_after"] = src_sidecars(a.copy_dir)
    rec["source"]["main_sha256_after"] = rec["source"]["sidecars_after"]["main.msgpack"]["sha256"]
    rec["source"]["main_equal_before_after"] = rec["source"]["main_sha256_before"] == rec["source"]["main_sha256_after"]
    rec["source"]["all_sidecars_equal_before_after"] = rec["source"]["sidecars_before"] == rec["source"]["sidecars_after"]
    rec["memory"] = mem.final(base)
    rec["audit"] = audit.summary(hdr["checkout"])
    rec["end_utc"] = utc()
    json.dump(rec, open(a.out, "w"), indent=1, sort_keys=True)
    print(json.dumps({k: rec[k] for k in ("label", "return", "removed_count", "state_digest", "checkpoint_sha256", "pre_state_digest")}, sort_keys=True))
    print(json.dumps({"heap_for_gate_gib": rec["memory"]["heap_for_gate_gib"], "memory_peak_gib": rec["memory"]["cgroup_memory_peak_gib"]}))
    return 0


def cmd_part2(a):
    audit = Audit()
    drv = load_driver()
    mem = Mem()
    cg, base = cgroup_paths()
    rec = {"mode": "part2", "label": a.label, "start_utc": utc(), "gate_reading_at_load_start": gate_reading(),
           "cgroup": {"path": cg, "memory.max": cgroup_read(base, "memory.max"), "memory.swap.max": cgroup_read(base, "memory.swap.max")}}
    nf, hdr = pin_and_import(a.checkout)
    rec["header"] = common_header(drv, hdr["checkout"], nf, hdr)
    if hdr["void"]:
        rec["VOID"] = True
        print(json.dumps(rec, sort_keys=True))
        return 3
    main = os.path.join(a.copy_dir, "main.msgpack")
    rec["source"] = {"copy_dir": a.copy_dir, "main_sha256_before": sha256_chunked(main), "stat_before": stat_row(main)}
    g = nf.Graph()
    g.restore(main)
    mem.stage("after_restore")
    rec["counts_after_restore"] = counts(g)
    K, B = 50, 5000

    class _Abort(Exception):
        pass

    real_prune = g._prune_synapses            # bound method of the engine as imported (never edited)
    captured = {}

    def _capture(**kw):
        captured.update(kw)
        raise _Abort()

    # (1) the orchestrator's OWN sets, captured WITHOUT mutating anything: abort at the _prune_synapses call. The elapsed time is
    #     everything the orchestrator does under _step_lock before it calls _prune_synapses (two _plan()s + floor check + capture map).
    g._prune_synapses = _capture
    t0, c0 = time.perf_counter(), time.process_time()
    try:
        g.compete_protected_links(K, B)
        raise SystemExit("orchestrator did not call _prune_synapses")
    except _Abort:
        pass
    rec["orchestrator_pre_prune_wall_s"] = round(time.perf_counter() - t0, 4)
    rec["orchestrator_pre_prune_cpu_s"] = round(time.process_time() - c0, 4)
    mem.stage("after_plan_capture")
    rec["competing_ids"] = len(captured["competing_ids"])
    rec["excluded_ids"] = len(captured["excluded_ids"])
    rec["order_key_entries"] = len(captured["order_key"])
    rec["max_removals_passed"] = captured["max_removals"]

    # (2) the VALIDATION pass in isolation. An absent id that sorts LAST ('~' > hex) makes _prune_synapses raise at the end of its pre-loop
    #     validation, so every real competing id has been validated and nothing was mutated (a refusal mutates nothing — plan 4.2(d)).
    probe = set(captured["competing_ids"])
    probe.add("~~pg1-absent-synapse")
    fp0 = (len(g.synapses), sum(g.synapses[s].low_weight_steps for s in captured["competing_ids"]))
    vals = []
    for _ in range(3):
        t0, c0 = time.perf_counter(), time.process_time()
        msg = None
        try:
            real_prune(competing_ids=probe, excluded_ids=captured["excluded_ids"], max_removals=B, order_key=captured["order_key"], report={})
        except ValueError as exc:
            msg = str(exc)
        vals.append({"wall_s": round(time.perf_counter() - t0, 4), "cpu_s": round(time.process_time() - c0, 4),
                     "raised_at_the_absent_id": bool(msg and "~~pg1-absent-synapse" in msg and "does not exist" in msg)})
    fp1 = (len(g.synapses), sum(g.synapses[s].low_weight_steps for s in captured["competing_ids"]))
    rec["validation_isolated"] = {"runs": vals, "state_fingerprint_unchanged": fp0 == fp1,
                                  "includes": "sorted(set(competing_ids)) + the per-id checks of this checkout's pre-loop validation, up to the last id"}
    mem.stage("after_validation_isolated")
    del probe

    # (3) the real calls, three times in this process; _prune_synapses is wrapped ONLY to time its inner span (pass-through).
    inner = []

    def _timed(**kw):
        t = time.perf_counter()
        try:
            return real_prune(**kw)
        finally:
            inner.append(time.perf_counter() - t)

    g._prune_synapses = _timed
    calls = []
    for i in (1, 2, 3):
        t0, c0 = time.perf_counter(), time.process_time()
        r = g.compete_protected_links(K, B)
        wall, cpu = time.perf_counter() - t0, time.process_time() - c0
        calls.append({"call": i, "wall_s": round(wall, 4), "cpu_s": round(cpu, 4), "inner_prune_wall_s": round(inner[-1], 4),
                      "outside_prune_wall_s": round(wall - inner[-1], 4), "eligible": r["eligible"], "removed": r["removed"],
                      "conducting_links_removed": r["conducting_links_removed"], "held_back_last_link": r["held_back_last_link"],
                      "floors_ok": r["floors_ok"], "F_links": r["F_links"], "protected_nodes": r["protected_nodes"],
                      "wants_with_removals": r["wants_with_removals"], "wants_zero": r["wants_zero"], "synapses_after": len(g.synapses),
                      "ru_maxrss_gib_after": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576.0, 4),
                      "load_1_after": open("/proc/loadavg").read().split()[0]})
        mem.stage("after_call_%d" % i)
    del g._prune_synapses
    rec["calls"] = calls
    rec["counts_final"] = counts(g)
    rec["source"]["main_sha256_after"] = sha256_chunked(main)
    rec["source"]["main_equal_before_after"] = rec["source"]["main_sha256_before"] == rec["source"]["main_sha256_after"]
    rec["memory"] = mem.final(base)
    rec["audit"] = audit.summary(hdr["checkout"])
    rec["end_utc"] = utc()
    json.dump(rec, open(a.out, "w"), indent=1, sort_keys=True)
    print(json.dumps({"label": a.label, "calls": [(c["wall_s"], c["removed"]) for c in calls],
                      "validation": [v["wall_s"] for v in vals], "heap_for_gate_gib": rec["memory"]["heap_for_gate_gib"],
                      "memory_peak_gib": rec["memory"]["cgroup_memory_peak_gib"]}))
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gate")
    g.add_argument("--name", required=True)
    g.add_argument("--min-gib", type=float, required=True)
    g.add_argument("--max-load", type=float, default=6.0)
    g.add_argument("--out")
    h = sub.add_parser("hash")
    h.add_argument("--dir", required=True)
    h.add_argument("--out")
    c = sub.add_parser("copy")
    c.add_argument("--src", required=True)
    c.add_argument("--dst", required=True)
    c.add_argument("--out", required=True)
    for nm in ("part1", "part2"):
        p = sub.add_parser(nm)
        p.add_argument("--label", required=True)
        p.add_argument("--checkout", required=True)
        p.add_argument("--copy-dir", required=True)
        p.add_argument("--out", required=True)
        if nm == "part1":
            p.add_argument("--ckpt", required=True, help="scratch temp .msgpack under z12-pg1-<UTC>")
    a = ap.parse_args()
    return {"gate": cmd_gate, "hash": cmd_hash, "copy": cmd_copy, "part1": cmd_part1, "part2": cmd_part2}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
