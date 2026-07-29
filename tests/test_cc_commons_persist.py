"""
CC Commons persistence (#84) — the medium survives a process restart, and stays
off Syl's.

# ---- Changelog ----
# [2026-07-28] Claude Code (DudeMan CC, Opus 5) — #84 CC Commons persist/restore
# What: Covers cc_commons_checkpoint_path()/persist_cc_commons() and the restore that
#       now happens inside get_cc_commons(). Four properties: (1) the path is under CC's
#       OWN workspace and is not Syl's, (2) persist is a no-op returning False when the
#       medium was never constructed, (3) deposit -> persist -> fresh-process -> restore
#       round-trips, and (4) building CC's medium never touches the canonical
#       commons._commons singleton.
# Why: CC's Commons was in-memory only -- every daemon/gateway restart wiped the
#       deposited topology, so nothing bucketed across process lifetimes. The restore
#       lives inside get_cc_commons rather than in each host, which makes property (4)
#       the one that actually needs a guard: on the VPS this code runs INSIDE Syl's
#       process, so a regression to canonical get_commons() would silently hand CC
#       Syl's medium and start dual-writing it. That is the failure this file exists
#       to catch, not the round-trip.
# How: reset cc_ng_organism._cc_commons around each test (it is a process-wide
#       singleton, exactly like the thing it stands in for) and point workspace_dir at
#       tmp_path. Simulating "restart" is just clearing that global -- get_cc_commons
#       then re-runs its create-and-restore path against the same on-disk file.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import commons as commons_mod
import cc_ng_organism


def _emb(seed, dim=768):
    r = np.random.RandomState(seed)
    v = r.randn(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def _count(commons, prefix):
    return sum(1 for s in commons._ng.synapses.values()
               if getattr(s, "target_id", "").startswith(prefix))


@pytest.fixture(autouse=True)
def _isolate_singletons():
    """Both media are process-wide singletons; hand every test a cold one and put
    whatever was there back afterward so test order can't matter."""
    cc_prev = cc_ng_organism._cc_commons
    syl_prev = commons_mod._commons
    cc_ng_organism._cc_commons = None
    commons_mod._commons = None
    yield
    cc_ng_organism._cc_commons = cc_prev
    commons_mod._commons = syl_prev


def test_checkpoint_path_is_under_cc_workspace_not_syls(tmp_path):
    """The medium's file lives under the workspace it was asked for, and nowhere near
    Syl's ~/.claude/plugins/neurograph or the canonical ~/NeuroGraph/data."""
    path = cc_ng_organism.cc_commons_checkpoint_path(str(tmp_path))

    assert path == os.path.join(str(tmp_path), "checkpoints", "commons.msgpack")
    # Distinct workspaces must yield distinct files -- this is the whole isolation
    # story at the filesystem layer.
    other = cc_ng_organism.cc_commons_checkpoint_path("/some/other/workspace")
    assert other != path

    # ~ expansion happens here, not at the call sites, so the hosts can pass the
    # tilde'd workspace strings they already hold.
    expanded = cc_ng_organism.cc_commons_checkpoint_path("~/somewhere")
    assert "~" not in expanded
    assert expanded.startswith(os.path.expanduser("~"))


def test_persist_is_a_noop_when_medium_was_never_built(tmp_path):
    """A host that never constructed the medium (gate off, early crash) must not
    write an empty file over a good one -- persist returns False and writes nothing."""
    assert cc_ng_organism._cc_commons is None
    assert cc_ng_organism.persist_cc_commons(str(tmp_path)) is False
    assert not os.path.exists(cc_ng_organism.cc_commons_checkpoint_path(str(tmp_path)))


def test_deposits_survive_a_simulated_restart(tmp_path):
    """deposit -> persist -> (process dies) -> get_cc_commons restores from disk."""
    ws = str(tmp_path)

    c1 = cc_ng_organism.get_cc_commons(ws)
    c1.deposit(_emb(1), "experience:cc-turn", metadata={"kind": "experience"})
    c1.deposit(_emb(2), "topology:cc-n1", metadata={"kind": "topology_delta"})

    assert cc_ng_organism.persist_cc_commons(ws) is True
    assert os.path.exists(cc_ng_organism.cc_commons_checkpoint_path(ws))

    # The restart: the singleton is gone, the file is not.
    cc_ng_organism._cc_commons = None
    c2 = cc_ng_organism.get_cc_commons(ws)

    assert c2 is not c1, "post-restart medium is a genuinely new object"
    assert _count(c2, "experience:") == 1
    assert _count(c2, "topology:") == 1


def test_missing_checkpoint_starts_fresh_rather_than_raising(tmp_path):
    """First-ever boot has no file. That is the normal path, not an error path."""
    c = cc_ng_organism.get_cc_commons(str(tmp_path))
    assert c is not None
    assert _count(c, "experience:") == 0


def test_corrupt_checkpoint_is_non_fatal(tmp_path):
    """A truncated/garbage checkpoint must degrade to a fresh medium, not take down
    the host it is booting inside (on the VPS, that host is Syl's process)."""
    path = cc_ng_organism.cc_commons_checkpoint_path(str(tmp_path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"this is not msgpack")

    c = cc_ng_organism.get_cc_commons(str(tmp_path))
    assert c is not None
    assert _count(c, "experience:") == 0


def test_building_cc_medium_never_touches_syls_singleton(tmp_path):
    """The #84 restore lives inside get_cc_commons, so this is where a regression to
    canonical get_commons() would land. On the VPS that would hand CC Syl's medium
    and dual-write it -- the exact thing the direct-constructor design prevents."""
    ws = str(tmp_path)

    cc = cc_ng_organism.get_cc_commons(ws)
    assert commons_mod._commons is None, \
        "get_cc_commons must not populate the canonical process-wide singleton"

    syl = commons_mod.get_commons()
    assert cc is not syl, "CC's medium and Syl's are separate objects"
    assert commons_mod._commons is syl

    # And the separation holds through a deposit: CC's deposit must not appear in
    # Syl's medium, nor Syl's in CC's.
    cc.deposit(_emb(3), "experience:cc-only", metadata={"kind": "experience"})
    syl.deposit(_emb(4), "experience:syl-only", metadata={"kind": "experience"})

    assert _count(cc, "experience:cc-only") == 1
    assert _count(cc, "experience:syl-only") == 0
    assert _count(syl, "experience:syl-only") == 1
    assert _count(syl, "experience:cc-only") == 0


def test_persist_writes_only_ccs_medium(tmp_path):
    """persist_cc_commons writes CC's file and leaves Syl's untouched, even when both
    media are live in the same process (the VPS co-tenancy case)."""
    ws = str(tmp_path)
    syl_path = tmp_path / "syls_commons.msgpack"

    syl = commons_mod.get_commons()
    syl.deposit(_emb(5), "experience:syl-only", metadata={"kind": "experience"})
    syl.persist(str(syl_path))
    syl_bytes_before = syl_path.read_bytes()

    cc = cc_ng_organism.get_cc_commons(ws)
    cc.deposit(_emb(6), "experience:cc-only", metadata={"kind": "experience"})
    assert cc_ng_organism.persist_cc_commons(ws) is True

    assert syl_path.read_bytes() == syl_bytes_before, \
        "CC's persist must not rewrite Syl's checkpoint"

    # And what CC wrote contains CC's deposit, not Syl's.
    fresh = commons_mod.Commons()
    fresh.restore(cc_ng_organism.cc_commons_checkpoint_path(ws))
    assert _count(fresh, "experience:cc-only") == 1
    assert _count(fresh, "experience:syl-only") == 0
