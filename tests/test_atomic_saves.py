# ---- Changelog ----
# [2026-10-14] Claude Code (Opus 5.5, lane atomic-saves; Josh 2026-10-14 "anything that can't risk a failed save, be
#   atomic") — created
# What: (1) checkpoint_guardian.atomic_file_write: a failing writer leaves the previous file and no tmp; the tmp is
#           fsynced BEFORE os.replace and the directory after; stale tmps of dead writers are reported, never deleted.
#       (2) SIGKILL proofs: a forked child runs a REAL writer in a loop and the parent SIGKILLs it at random points,
#           repeatedly; after every kill the file loads with its real reader: NeuroGraphMemory.save() (main.msgpack,
#           vectors.msgpack, manifest, activation sidecar — a fresh NeuroGraphMemory restores the workspace),
#           ChannelRegistry._save, KillSwitch enable/disable,
#           ETModuleManager._save_registry, DistanceCache.save. (ActivationPersistence.write_state: its own file,
#           test_atomic_saves_activation.py, with the protected commit.) A control child doing an in-place write tears under
#           the same kills, so the harness can see a torn file.
# Why:  #1095 (Elmer's torn proto_weights.pt on a daemon restart) — every save that can't risk a failed write.
# How:  os.fork from the test process (imports already loaded), os.kill(SIGKILL); small payloads.
# -------------------
"""Atomic-save proofs for NeuroGraph's persistent-state writers."""

import json
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import checkpoint_guardian as cg  # noqa: E402

msgpack = pytest.importorskip("msgpack")


# ---- (1) the helper ----

def test_failed_write_leaves_previous_file_and_no_tmp(tmp_path):
    p = tmp_path / "main.msgpack"
    cg.atomic_write_bytes(p, b"good")

    def boom(tmp):
        Path(tmp).write_bytes(b"half")
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError):
        cg.atomic_file_write(str(p), boom)
    assert p.read_bytes() == b"good"
    assert [n for n in os.listdir(tmp_path) if ".tmp-" in n] == []


def test_tmp_fsynced_before_replace_and_dir_after(tmp_path, monkeypatch):
    events = []
    real_fsync, real_replace = os.fsync, os.replace

    def fsync(fd):
        events.append(("fsync", "dir" if os.path.isdir(f"/proc/self/fd/{fd}") else "file"))
        return real_fsync(fd)

    def replace(a, b):
        events.append(("replace", None))
        return real_replace(a, b)

    monkeypatch.setattr(cg.os, "fsync", fsync)
    monkeypatch.setattr(cg.os, "replace", replace)
    cg.atomic_write_text(tmp_path / "x.json", "{}")
    assert events == [("fsync", "file"), ("replace", None), ("fsync", "dir")]


def test_stale_tmps_reported_not_deleted(tmp_path, caplog):
    final = tmp_path / "vectors.msgpack"
    cg.atomic_write_bytes(final, b"v")
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    stale = tmp_path / f"vectors.tmp-{dead.pid}.msgpack"
    stale.write_bytes(b"torn")
    mine = tmp_path / f"vectors.tmp-{os.getpid()}.msgpack"
    assert cg.find_stale_temps(final) == [str(stale)]
    with caplog.at_level("WARNING", logger="checkpoint_guardian"):
        cg.atomic_write_bytes(final, b"v2")
        cg.atomic_write_bytes(final, b"v3")
    warns = [r for r in caplog.records if "stale temp" in r.getMessage()]
    assert len(warns) == 1                      # once per distinct stale set, not every autosave
    assert stale.exists() and not mine.exists()
    assert final.read_bytes() == b"v3"


# ---- (2) SIGKILL harness ----

def _kill_rounds(loop_fn, verify_fn, rounds, seed, lo=0.02, hi=0.4):
    """Fork a child running loop_fn() forever; SIGKILL it after a random delay; verify. Returns verdict list."""
    rng = random.Random(seed)
    verdicts = []
    for r in range(rounds):
        rfd, wfd = os.pipe()
        pid = os.fork()
        if pid == 0:
            try:
                os.close(rfd)
                os.write(wfd, b"r")
                loop_fn(r)
            finally:
                os._exit(3)
        os.close(wfd)
        assert os.read(rfd, 1) == b"r"
        os.close(rfd)
        time.sleep(rng.uniform(lo, hi))
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
        try:
            verify_fn()
            verdicts.append(True)
        except FileNotFoundError:
            verdicts.append(None)
        except Exception:
            verdicts.append(False)
    return verdicts


def _clean_tmps(d):
    for n in os.listdir(d):
        if ".tmp" in n:
            try:
                os.remove(os.path.join(d, n))
            except OSError:
                pass


def test_control_inplace_sidecar_tears(tmp_path):
    ckpt = str(tmp_path / "main.msgpack.activations.json")
    big = {f"n{i}": {"voltage": i * 0.001, "last_fired": i, "excitability": 1.0} for i in range(40000)}

    def loop(r):
        while True:
            with open(ckpt, "w") as f:
                json.dump({"entries": big}, f)

    def verify():
        assert len(json.load(open(ckpt))["entries"]) == 40000

    v = _kill_rounds(loop, verify, rounds=20, seed=1)
    assert False in v, v                        # the harness does catch a torn write


def test_sigkill_channel_registry_kill_switch_and_et_registry(tmp_path):
    from lenia.channels import ChannelRegistry
    from lenia.config import LeniaConfig
    from lenia.kill_switch import KillSwitch
    from et_modules.manager import ETModuleManager, ModuleManifest

    cfg = LeniaConfig()
    reg = ChannelRegistry(cfg, str(tmp_path / "lenia"))
    reg._save()
    ks = KillSwitch(cfg, str(tmp_path / "lenia"))
    mgr = ETModuleManager(root_dir=str(tmp_path / "et"))
    for i in range(400):
        mgr._registry[f"mod{i}"] = ModuleManifest(module_id=f"mod{i}", display_name="x" * 200)
    mgr._save_registry()

    def loop(r):
        while True:
            reg._save()
            ks._enabled = False
            ks.enable()
            ks._enabled = True
            ks.disable()
            mgr._save_registry()

    def verify():
        json.load(open(tmp_path / "lenia" / "channels.json"))
        assert (tmp_path / "lenia" / "enabled").read_text() in ("True", "False")
        assert len(json.load(open(tmp_path / "et" / "registry.json"))["modules"]) == 400
        ChannelRegistry(cfg, str(tmp_path / "lenia"))
        _clean_tmps(tmp_path / "lenia")
        _clean_tmps(tmp_path / "et")

    v = _kill_rounds(loop, verify, rounds=20, seed=2)
    assert False not in v and v.count(True) >= 15, v


def test_sigkill_distance_cache_never_tears(tmp_path):
    np = pytest.importorskip("numpy")
    sparse = pytest.importorskip("scipy.sparse")
    from lenia import kernel

    n = 3000
    rng = np.random.default_rng(0)

    class Stub:
        _populated = True
        _n = n
        _entity_ids = [f"e{i}" for i in range(n)]
        _watermark = None
        _components_csr = [sparse.random(n, n, density=0.01, format="csr", random_state=c, dtype=np.float32)
                           for c in range(kernel.NUM_DIST_COMPONENTS)]

        def _ensure_csr_current(self):
            pass

    stub = Stub()
    path = str(tmp_path / "distance_cache.npz")
    kernel.DistanceCache.save(stub, path)

    def loop(r):
        while True:
            stub._watermark = (int(rng.integers(1 << 30)),)
            kernel.DistanceCache.save(stub, path)

    def verify():
        with np.load(path, allow_pickle=False) as a:
            for c in range(kernel.NUM_DIST_COMPONENTS):
                assert a[f"c{c}_data"].shape == stub._components_csr[c].data.shape
        _clean_tmps(tmp_path)

    v = _kill_rounds(loop, verify, rounds=20, seed=3)
    assert False not in v and v.count(True) >= 15, v


def _ngmemory_kill_proof(tmp_path, check_sidecar):
    """The real checkpoint path: NeuroGraphMemory.save() -> guardian gate -> atomic graph / vectors / manifest
    (+ the activation sidecar when check_sidecar: that writer is the protected activation_persistence.py)."""
    from openclaw_hook import NeuroGraphMemory
    cfg = {"tonic": {"enabled": False}, "peer_bridge": {"enabled": False}}
    ws = str(tmp_path / "ws")
    ng = NeuroGraphMemory(workspace_dir=ws, config=cfg)
    for i in range(400):
        ng.graph.create_node(node_id=f"atomic-test-node-{i}")
    ng.save()
    ckpt_dir = os.path.join(ws, "checkpoints")

    def loop(r):
        k = 0
        while True:
            k += 1
            ng.graph.create_node(node_id=f"atomic-test-r{r}-{k}")
            ng.save()

    def verify():
        with open(os.path.join(ckpt_dir, "main.msgpack"), "rb") as f:
            state = msgpack.unpack(f, raw=False, strict_map_key=False)
        assert isinstance(state, dict) and state
        with open(os.path.join(ckpt_dir, "vectors.msgpack"), "rb") as f:
            msgpack.unpack(f, raw=False, strict_map_key=False)
        json.load(open(os.path.join(ckpt_dir, "main.msgpack.manifest.json")))
        side = os.path.join(ckpt_dir, "main.msgpack.activations.json")
        if check_sidecar:
            json.load(open(side))
        _clean_tmps(ckpt_dir)

    v = _kill_rounds(loop, verify, rounds=15, seed=4, lo=0.05, hi=0.6)
    assert False not in v and v.count(True) >= 12, v
    ng2 = NeuroGraphMemory(workspace_dir=ws, config=cfg)   # the restart: a fresh instance restores the workspace
    assert len(ng2.graph.nodes) >= 400


def test_sigkill_neurographmemory_save_never_tears(tmp_path):
    _ngmemory_kill_proof(tmp_path, check_sidecar=False)
