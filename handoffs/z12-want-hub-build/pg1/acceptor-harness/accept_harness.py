# ---- Changelog ----
# [2026-10-01] le-039 (PG-1 ACCEPTOR, dispatch #12386) — INDEPENDENT acceptor harness (NOT the builder's pg1_harness.py)
# What: gate | part1 sub-commands. part1 = ONE real-graph load: pin sys.path[0] to --checkout, import neuro_foundation (VOID if not inside
#   the checkout), Graph().restore(<copy>/main.msgpack), _prune_synapses() at all-defaults, Graph.checkpoint() to a scratch temp .msgpack,
#   record the 14 compared fields + extras.
# Why: plan-005 4A.5 PG-1; an acceptor must re-run the comparison itself, with its own code where it can.
# How: own audit hook, own removal-order wrapper, own full-field synapse digest (every dataclass field, items() order) IN ADDITION to the
#   driver's state_digest (imported by FILE PATH from tests/want_hub_golden_driver.py, UNMODIFIED, so the field is comparable to the
#   builder's records). Output: ids / hashes / counts / booleans only. Never save(); the only checkpoint() target is a scratch temp
#   .msgpack under z12-pg1-accept-*; nothing is written to any source or live path.
# -------------------
import argparse, dataclasses, hashlib, importlib.util, json, os, random, resource, subprocess, sys, time

DRIVER = "/home/josh/NeuroGraph-worktrees/z12-want-hub-build-20260930/tests/want_hub_golden_driver.py"
WBITS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 23), b""):
            h.update(b)
    return h.hexdigest()


def gate(min_gib, max_load):
    mi = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo")}
    la = open("/proc/loadavg").read().split()[:3]
    unit = subprocess.run(["systemctl", "--user", "is-active", "cc-ng-daemon.service"], capture_output=True, text=True).stdout.strip()
    procs = [l for l in subprocess.run(["pgrep", "-af", "cc-ng-daemon|neurograph_rpc"], capture_output=True, text=True).stdout.splitlines() if "pgrep" not in l]
    gib = mi["MemAvailable"] / 1048576.0
    ok = gib >= min_gib and float(la[0]) < max_load and unit == "inactive" and not procs
    return {"utc": utc(), "mem_available_kb": mi["MemAvailable"], "mem_available_gib": round(gib, 3), "load_1_5_15": la, "daemon_unit": unit,
            "daemon_procs": len(procs), "min_gib": min_gib, "max_load": max_load, "gate_met": ok}


class Audit:
    def __init__(self):
        self.w, self.mut, self.sp, self.r = [], [], [], set()
        sys.addaudithook(self.h)

    def h(self, ev, a):
        try:
            if ev == "open":
                p, mode, fl = a
                p = os.fsdecode(p) if not isinstance(p, int) else "fd:%d" % p
                w = bool(fl & WBITS) if isinstance(fl, int) else False
                if not w and isinstance(mode, str) and any(c in mode for c in "wax+"):
                    w = True
                (self.w.append(p) if w else self.r.add(p))
            elif ev in ("os.remove", "os.rename", "os.mkdir", "os.rmdir", "os.truncate", "os.chmod", "os.symlink", "os.link", "os.utime",
                        "shutil.copyfile", "shutil.copytree", "shutil.rmtree", "shutil.move", "os.chown"):
                self.mut.append([ev] + [os.fsdecode(x) if isinstance(x, (str, bytes)) else repr(x) for x in a[:2]])
            elif ev == "subprocess.Popen":
                self.sp.append(os.fsdecode(a[0]) if isinstance(a[0], (str, bytes)) else repr(a[0]))
        except Exception:
            pass


def own_digest(nf, g):
    """Independent digest: EVERY field of every Synapse (dataclass fields, incl. metadata), in store items() order; dirty set; node ids;
    timestep; hyperedge count. Hash only — nothing printed."""
    names = [f.name for f in dataclasses.fields(nf.Synapse)]
    h = hashlib.sha256()
    n = 0
    for sid, s in g.synapses.items():
        h.update(repr((sid,) + tuple(repr(getattr(s, k)) for k in names)).encode())
        h.update(b"\n")
        n += 1
    h.update(json.dumps([sorted(g._dirty_synapses), sorted(g.nodes.keys()), g.timestep, len(g.hyperedges)]).encode())
    return {"sha256": h.hexdigest(), "n_synapses_hashed": n, "fields": names}


def counts(g):
    return {"nodes": len(g.nodes), "synapses": len(g.synapses), "hyperedges": len(g.hyperedges), "timestep": g.timestep,
            "synapses_type": "%s.%s" % (type(g.synapses).__module__, type(g.synapses).__qualname__)}


def cg():
    p = open("/proc/self/cgroup").read().strip().split("::")[1]
    b = "/sys/fs/cgroup" + p
    rd = lambda n: open(b + "/" + n).read().strip().replace("\n", " | ")
    return p, rd


def part1(a):
    audit = Audit()
    spec = importlib.util.spec_from_file_location("want_hub_golden_driver", DRIVER)
    drv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(drv)
    cgp, rd = cg()
    rec = {"label": a.label, "start_utc": utc(), "gate_at_start": gate(0, 99), "cgroup": {"path": cgp, "memory.max": rd("memory.max"), "memory.swap.max": rd("memory.swap.max")}}
    co = os.path.realpath(a.checkout)
    sys.path.insert(0, co)
    random.seed(0)
    try:
        import numpy
        numpy.random.seed(0)
    except Exception:
        pass
    import neuro_foundation as nf
    f = os.path.realpath(nf.__file__)
    tf, tv = drv.ng_tract_info()
    hdr = {"checkout": co, "neuro_foundation_file": f, "sys_path_0": sys.path[0],
           "git_rev": subprocess.run(["git", "-C", co, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
           "blob": subprocess.run(["git", "-C", co, "hash-object", f], capture_output=True, text=True).stdout.strip(),
           "ng_tract_file": tf, "ng_tract_version": tv, "new_api_present": drv.new_api_present(nf), "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
           "python": sys.version.split()[0], "pid": os.getpid(),
           "env": {k: os.environ.get(k) for k in ("NG_EMBED_REMOTE", "PYTHONPATH", "PYTHONDONTWRITEBYTECODE", "HF_HUB_OFFLINE")},
           "void": os.path.dirname(f) != co}
    rec["header"] = hdr
    print(json.dumps({"label": a.label, "neuro_foundation__file__": f, "git_rev": hdr["git_rev"], "blob": hdr["blob"], "ng_tract": [tf, tv],
                      "new_api_present": hdr["new_api_present"], "void": hdr["void"], "pythonhashseed": hdr["pythonhashseed"]}, sort_keys=True), flush=True)
    if hdr["void"]:
        rec["VOID"] = True
        json.dump(rec, open(a.out, "w"), indent=1, sort_keys=True)
        return 3
    main = os.path.join(a.copy_dir, "main.msgpack")
    rec["main_sha256_before"] = sha_file(main)
    rec["main_stat_before"] = [os.stat(main).st_size, os.stat(main).st_mtime_ns, os.stat(main).st_ino]
    g = nf.Graph()
    g.restore(main)
    rec["ru_maxrss_after_restore_gib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576.0, 4)
    rec["counts_after_restore"] = counts(g)
    rec["pre_state_digest"] = drv.state_digest(g, None)
    rec["own_digest_pre"] = own_digest(nf, g)
    # own removal-order wrapper + pruned-event handler
    order, events = [], []
    orig = g._remove_synapse_internal

    def wrap(sid):
        order.append(sid)
        return orig(sid)

    g._remove_synapse_internal = wrap
    g.register_event_handler("pruned", lambda **kw: events.append([kw.get("count"), kw.get("timestep")]))
    rec["synapses_before"] = len(g.synapses)
    t0 = time.perf_counter()
    ret = g._prune_synapses()
    rec["prune_wall_s"] = round(time.perf_counter() - t0, 4)
    del g._remove_synapse_internal
    rec["return"] = ret
    rec["synapses_after"] = len(g.synapses)
    rec["removed_count"] = len(order)
    rec["removed_ids_sha256_in_order"] = hashlib.sha256("\n".join(order).encode()).hexdigest()
    rec["removed_ids_sha256_sorted"] = hashlib.sha256("\n".join(sorted(order)).encode()).hexdigest()
    rec["removed_is_sorted_order"] = order == sorted(order)
    rec["removed_ids_unique"] = len(set(order)) == len(order)
    rec["pruned_events"] = events
    rec["state_digest_before_checkpoint"] = drv.state_digest(g, events)
    rec["own_digest_post"] = own_digest(nf, g)
    ck = a.ckpt
    assert ck.endswith(".msgpack") and "/z12-pg1-accept-" in ck and "/data/checkpoints" not in ck
    g.checkpoint(ck)
    rec["checkpoint_path"] = ck
    rec["checkpoint_size"] = os.path.getsize(ck)
    rec["checkpoint_sha256"] = sha_file(ck)
    rec["state_digest"] = drv.state_digest(g, events)
    rec["counts_final"] = counts(g)
    rec["main_sha256_after"] = sha_file(main)
    rec["main_equal_before_after"] = rec["main_sha256_before"] == rec["main_sha256_after"]
    rec["main_stat_equal"] = rec["main_stat_before"] == [os.stat(main).st_size, os.stat(main).st_mtime_ns, os.stat(main).st_ino]
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rec["memory"] = {"ru_maxrss_gib": round(ru / 1048576.0, 4), "cgroup_memory_peak_gib": round(int(rd("memory.peak")) / 1024.0 ** 3, 4), "cgroup_memory_events": rd("memory.events")}
    pref = (sys.prefix, "/usr/lib/python3", "/usr/lib/x86_64-linux-gnu", "/home/josh/.local/lib/python3.12", co, "/proc/", "/sys/", "/dev/", "/etc/", "/lib/")
    rec["audit"] = {"write_mode_opens": sorted(set(audit.w)), "mutating_calls": audit.mut, "spawned_programs": sorted(set(audit.sp)),
                    "data_read_opens": sorted(p for p in audit.r if not p.startswith(pref) and not p.endswith((".py", ".pyc", ".so", ".pth")))}
    rec["end_utc"] = utc()
    json.dump(rec, open(a.out, "w"), indent=1, sort_keys=True)
    print(json.dumps({k: rec[k] for k in ("label", "return", "removed_count", "state_digest", "checkpoint_sha256")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    g = sp.add_parser("gate"); g.add_argument("--min-gib", type=float, required=True); g.add_argument("--max-load", type=float, default=6.0); g.add_argument("--out")
    p = sp.add_parser("part1")
    for n in ("label", "checkout", "copy_dir", "out", "ckpt"):
        p.add_argument("--" + n.replace("_", "-"), required=True)
    a = ap.parse_args()
    if a.cmd == "gate":
        r = gate(a.min_gib, a.max_load)
        if a.out:
            json.dump(r, open(a.out, "w"), indent=1, sort_keys=True)
        print(json.dumps(r, sort_keys=True))
        sys.exit(0 if r["gate_met"] else 4)
    sys.exit(part1(a))
