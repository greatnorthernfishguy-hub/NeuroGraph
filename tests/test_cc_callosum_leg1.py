# tests/test_cc_callosum_leg1.py
#
# ---- Changelog ----
# [2026-07-27] Claude Code (Sonnet 5) — CC Corpus Callosum Leg 1 (#70) tests
# What: Coverage for cc_ng_organism.trickle_gateway_conduit() (laptop-side
#   per-batch conduit write) and drain_gateway_conduit() (VPS-side drain +
#   delete of every conduit file), per docs/superpowers/plans/2026-07-27-cc-
#   corpus-callosum-leg1-spec.md §3: append-correctness (here: per-batch-file
#   correctness, since the built design uses one immutable file per trickle
#   rather than a shared append target -- see spec §2b race note), collision-
#   free filenames under rapid calls, VPS drain-and-delete, gate-off no-op on
#   both sides, and fail-soft on a missing/corrupt conduit dir.
# How: Real ng_tract.deposit_experience()/TractReader round-trips and a real
#   NeuroGraphMemory instance (same cc_ng fixture pattern as
#   test_cc_dual_pass.py/test_cc_refeed.py) for the end-to-end drain proof;
#   the gate (_CC_CALLOSUM_LEG1_ENABLED, computed once at import like
#   _CC_PITH_ENABLED) is toggled via monkeypatch.setattr on the module
#   object, the established pattern in this suite (see test_cc_recall_dedup.py,
#   test_cc_recall_unification.py) since it's read at import time from env.
# -------------------
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob
import threading
import tempfile
import shutil

import pytest

import cc_ng_organism
from cc_ng_organism import (
    trickle_gateway_conduit,
    drain_gateway_conduit,
    drain_ingest_tract,
    cc_gateway_conduit_dir,
)


# NOTE on idle_steps=0 in the file-lifecycle tests below: consolidation is
# REAL homeostatic regulation (graph.step() x N), and on a 2-node toy graph
# 250 steps trips the zero-fire breaker and culls the very nodes the test
# just absorbed -- an artifact of the toy scale, not of the drain. Those
# tests pin idle_steps=0 to isolate what they actually assert (per-file
# delete/quarantine lifecycle). The consolidation discipline itself is
# tested separately, with a counting fake graph, further down.


@pytest.fixture
def leg1_enabled(monkeypatch):
    """Flip the Leg 1 gate on for the duration of a test -- mirrors the
    monkeypatch.setattr(cc_ng_organism, '_CC_PITH_ENABLED', ...) pattern
    already used for CC_PITH_ENABLED elsewhere in this suite."""
    monkeypatch.setattr(cc_ng_organism, "_CC_CALLOSUM_LEG1_ENABLED", True)
    # Hemisphere identity is DECLARED, never guessed -- the producer refuses to
    # write a conduit file without it (a wrong guess would disarm the drain's
    # self-consumption guard). Production sets this in .bashrc on both halves.
    monkeypatch.setenv("MACHINE_ID", "laptop")


@pytest.fixture
def draining_hemisphere(monkeypatch, leg1_enabled):
    """Run the test AS the receiving half. The conduit files under test are all
    laptop-produced, so the drain's self-consumption guard -- which now defaults
    from MACHINE_ID rather than trusting the caller to pass exclude_prefix -- must
    see a DIFFERENT identity or it will (correctly) skip every one of them.
    A drain test declaring MACHINE_ID=laptop while draining laptop_* files was
    describing a hemisphere eating its own outgoing turns."""
    monkeypatch.setenv("MACHINE_ID", "vps")


@pytest.fixture
def cc_ng():
    from openclaw_hook import NeuroGraphMemory
    workspace = tempfile.mkdtemp(prefix='cc_callosum_leg1_test_')
    ng = NeuroGraphMemory(workspace_dir=workspace,
                          config={"tonic": {"enabled": False}, "peer_bridge": {"enabled": False}})
    yield ng
    shutil.rmtree(workspace, ignore_errors=True)


# =============================================================================
# Gate-off default: inert on both sides
# =============================================================================

def test_trickle_gateway_conduit_is_noop_when_gate_off(tmp_path, monkeypatch):
    monkeypatch.setattr(cc_ng_organism, "_CC_CALLOSUM_LEG1_ENABLED", False)
    conduit_dir = str(tmp_path / "conduit")
    result = trickle_gateway_conduit(b"some raw tract bytes", conduit_dir=conduit_dir)
    assert result is None
    assert not os.path.exists(conduit_dir)


def test_drain_gateway_conduit_is_noop_when_gate_off(tmp_path, monkeypatch, cc_ng):
    monkeypatch.setattr(cc_ng_organism, "_CC_CALLOSUM_LEG1_ENABLED", False)
    conduit_dir = str(tmp_path / "conduit")
    os.makedirs(conduit_dir)
    # A leftover conduit file must survive untouched -- gate-off means
    # drain_gateway_conduit doesn't even list the directory.
    leftover = os.path.join(conduit_dir, "laptop_cc_gateway.123_deadbeef.tract")
    with open(leftover, "wb") as f:
        f.write(b"untouched")

    state = {"last_forest_id": None}
    absorbed = drain_gateway_conduit(cc_ng.graph, cc_ng.vector_db, state, conduit_dir=conduit_dir,
                                    idle_steps=0, load_ceiling=999.0)
    assert absorbed == 0
    assert os.path.exists(leftover)
    with open(leftover, "rb") as f:
        assert f.read() == b"untouched"


# =============================================================================
# Laptop side: trickle_gateway_conduit
# =============================================================================

def test_trickle_gateway_conduit_writes_byte_identical_snapshot(tmp_path, leg1_enabled):
    conduit_dir = str(tmp_path / "conduit")
    data = b"raw BTF bytes exactly as read from the local cc_gateway tract"
    dest = trickle_gateway_conduit(data, conduit_dir=conduit_dir)

    assert dest is not None
    assert os.path.dirname(dest) == conduit_dir
    assert os.path.basename(dest).startswith("laptop_cc_gateway.")
    assert os.path.basename(dest).endswith(".tract")
    with open(dest, "rb") as f:
        assert f.read() == data


def test_trickle_gateway_conduit_empty_data_is_noop(tmp_path, leg1_enabled):
    conduit_dir = str(tmp_path / "conduit")
    assert trickle_gateway_conduit(b"", conduit_dir=conduit_dir) is None
    assert trickle_gateway_conduit(None, conduit_dir=conduit_dir) is None
    assert not os.path.exists(conduit_dir)


def test_trickle_gateway_conduit_collision_free_under_rapid_calls(tmp_path, leg1_enabled):
    """Per-batch filenames must never collide even when generated back-to-back
    in the same pulse (spec §2b: per-batch filenames sidestep the binary-
    merge scenario a shared append target would hit under repo-sync.sh)."""
    conduit_dir = str(tmp_path / "conduit")
    dests = [trickle_gateway_conduit(f"batch {i}".encode(), conduit_dir=conduit_dir)
             for i in range(50)]

    assert all(d is not None for d in dests)
    assert len(set(dests)) == 50, "every rapid-fire call must land a distinct file"

    on_disk = sorted(glob.glob(os.path.join(conduit_dir, "laptop_cc_gateway.*.tract")))
    assert len(on_disk) == 50
    # No leftover .tmp files -- write-tmp-then-rename must always complete.
    assert not glob.glob(os.path.join(conduit_dir, "*.tmp"))


def test_trickle_gateway_conduit_fails_soft_on_unwritable_dir(leg1_enabled):
    """A conduit dir that can't be created/written to must never raise --
    it must fail soft and return None (the daemon's autosave pulse must
    never break because of this)."""
    # A path that can't possibly be created as a directory: a regular file
    # sitting where a directory component is expected.
    with tempfile.NamedTemporaryFile() as blocker:
        bogus_dir = os.path.join(blocker.name, "conduit")
        result = trickle_gateway_conduit(b"turn text", conduit_dir=bogus_dir)
        assert result is None


# =============================================================================
# VPS side: drain_gateway_conduit
# =============================================================================

def test_drain_gateway_conduit_absorbs_and_deletes_conduit_files(cc_ng, tmp_path, draining_hemisphere):
    """The golden end-to-end: real BTF frames trickled into per-batch conduit
    files -> drain_gateway_conduit absorbs each via the real drain_ingest_
    tract -> conversational nodes exist -> each fully-drained file is deleted."""
    import ng_tract

    conduit_dir = str(tmp_path / "conduit")
    os.makedirs(conduit_dir)

    for i, text in enumerate([
        "the laptop turn about the corpus callosum spec",
        "the laptop turn about retiring the lossy jsonl sync",
    ]):
        path = os.path.join(conduit_dir, f"laptop_cc_gateway.{1000 + i}_batch{i}.tract")
        ng_tract.deposit_experience(
            content=text.encode(),
            source="cc_gateway",
            tract_path=path,
            content_type="text",
        )

    state = {"last_forest_id": None}
    absorbed = drain_gateway_conduit(cc_ng.graph, cc_ng.vector_db, state, conduit_dir=conduit_dir,
                                    idle_steps=0, load_ceiling=999.0)
    assert absorbed == 2

    conv_nodes = [n for n in cc_ng.graph.nodes.values()
                  if n.metadata.get("creation_mode") == "conversational"]
    # >= 2, not == 2: on an environment with live TID reachable (e.g. the
    # VPS), dual-pass extracts real concept/tree nodes per turn alongside
    # the 2 forest nodes -- exactly Leg 1's actual purpose (the VPS is the
    # sole Arborist). A forest-only environment (no TID, e.g. the laptop
    # itself) degrades to exactly 2. Either is correct; what matters is
    # both turns landed, which `absorbed == 2` above already proved.
    assert len(conv_nodes) >= 2

    # Fully-drained (now-empty) conduit files are deleted -- repo-sync.sh
    # syncs the deletion back to the laptop, no shared mutable file crosses
    # the wire twice.
    assert glob.glob(os.path.join(conduit_dir, "laptop_cc_gateway.*.tract")) == []


def test_drain_gateway_conduit_missing_dir_is_noop(cc_ng, tmp_path, draining_hemisphere):
    conduit_dir = str(tmp_path / "does_not_exist")
    state = {"last_forest_id": None}
    absorbed = drain_gateway_conduit(cc_ng.graph, cc_ng.vector_db, state, conduit_dir=conduit_dir,
                                    idle_steps=0, load_ceiling=999.0)
    assert absorbed == 0


def test_drain_gateway_conduit_skips_corrupt_file_absorbs_rest(cc_ng, tmp_path, draining_hemisphere):
    """One corrupt/garbage conduit file must not abort the whole batch --
    fails soft on that file, keeps draining the others."""
    import ng_tract

    conduit_dir = str(tmp_path / "conduit")
    os.makedirs(conduit_dir)

    good_path = os.path.join(conduit_dir, "laptop_cc_gateway.1_good.tract")
    ng_tract.deposit_experience(
        content=b"a genuine turn that must still be absorbed",
        source="cc_gateway",
        tract_path=good_path,
        content_type="text",
    )
    bad_path = os.path.join(conduit_dir, "laptop_cc_gateway.2_bad.tract")
    with open(bad_path, "wb") as f:
        f.write(b"not a valid BTF tract at all")

    state = {"last_forest_id": None}
    absorbed = drain_gateway_conduit(cc_ng.graph, cc_ng.vector_db, state, conduit_dir=conduit_dir,
                                    idle_steps=0, load_ceiling=999.0)
    assert absorbed >= 1
    conv_nodes = [n for n in cc_ng.graph.nodes.values()
                  if n.metadata.get("creation_mode") == "conversational"]
    assert len(conv_nodes) >= 1


def test_drain_gateway_conduit_quarantines_unparseable_file_instead_of_retrying_forever(
        cc_ng, tmp_path, draining_hemisphere):
    """Finding 2 (law-enforcer review): a file that fails to even PARSE never
    reaches drain_ingest_tract's truncate step, so it would otherwise sit in
    the conduit dir unchanged forever, retried every pulse. It must instead
    be moved to <conduit_dir>/quarantine/ (not deleted -- the bytes are
    preserved for inspection) and removed from the conduit dir proper, so a
    laptop/VPS format skew can't silently pile up garbage in a git-synced
    directory."""
    conduit_dir = str(tmp_path / "conduit")
    os.makedirs(conduit_dir)
    bad_path = os.path.join(conduit_dir, "laptop_cc_gateway.1_bad.tract")
    garbage = b"not a valid BTF tract at all, and never will be"
    with open(bad_path, "wb") as f:
        f.write(garbage)

    state = {"last_forest_id": None}
    absorbed = drain_gateway_conduit(cc_ng.graph, cc_ng.vector_db, state, conduit_dir=conduit_dir,
                                    idle_steps=0, load_ceiling=999.0)

    assert absorbed == 0
    assert not os.path.exists(bad_path)  # gone from the conduit dir proper
    quarantined = os.path.join(conduit_dir, "quarantine", "laptop_cc_gateway.1_bad.tract")
    assert os.path.exists(quarantined)
    with open(quarantined, "rb") as f:
        assert f.read() == garbage  # bytes preserved, not truncated or lost


# =============================================================================
# Default path resolution
# =============================================================================

def test_cc_gateway_conduit_dir_default_and_env_override(monkeypatch):
    monkeypatch.delenv("CC_GATEWAY_CONDUIT_PATH", raising=False)
    assert cc_gateway_conduit_dir() == os.path.expanduser("~/docs/ng_topology")

    monkeypatch.setenv("CC_GATEWAY_CONDUIT_PATH", "/tmp/custom_conduit_dir")
    assert cc_gateway_conduit_dir() == "/tmp/custom_conduit_dir"


# =============================================================================
# drain_ingest_tract(return_consumed=True) -- the Finding-1 fix itself.
#
# A prior laptop-side wiring took an INDEPENDENT pre-drain snapshot of the
# local tract file, then called drain_ingest_tract() separately. If miniTID
# appended new bytes in the window between those two reads, drain absorbed
# AND truncated those bytes (they made it into the laptop's own forest) but
# the earlier snapshot never saw them -- so they were silently lost to the
# VPS conduit forever, even though they were already gone from the local
# file too. return_consumed=True closes that window by handing back the
# EXACT bytes drain_ingest_tract itself truncated -- one read, both sinks.
# =============================================================================

def test_drain_ingest_tract_default_return_is_unchanged_int(cc_ng, tmp_path):
    """return_consumed defaults to False -- every pre-existing caller
    (both hemispheres' local drain, tests/test_cc_dual_pass.py,
    tests/test_cc_refeed.py) keeps getting a plain int back, unchanged."""
    import ng_tract
    tract_path = str(tmp_path / "turns.tract")
    ng_tract.deposit_experience(
        content=b"a turn", source="cc_gateway", tract_path=tract_path, content_type="text",
    )
    state = {"last_forest_id": None}
    result = drain_ingest_tract(cc_ng.graph, cc_ng.vector_db, state, tract_path=tract_path)
    assert isinstance(result, int)
    assert result >= 1


def test_drain_ingest_tract_return_consumed_matches_what_was_truncated(cc_ng, tmp_path):
    """The consumed bytes returned must be EXACTLY what the file's own
    truncate-after-drain step removed -- verified by re-depositing them into
    a fresh tract file and confirming a second drain absorbs the same turn
    again (proving byte-for-byte fidelity, not a reconstruction/re-encoding)."""
    import ng_tract
    tract_path = str(tmp_path / "turns.tract")
    ng_tract.deposit_experience(
        content=b"a genuinely distinct turn for consumed-bytes fidelity",
        source="cc_gateway", tract_path=tract_path, content_type="text",
    )
    with open(tract_path, "rb") as f:
        original_bytes = f.read()

    state = {"last_forest_id": None}
    absorbed, consumed = drain_ingest_tract(
        cc_ng.graph, cc_ng.vector_db, state, tract_path=tract_path, return_consumed=True)

    assert absorbed >= 1
    assert consumed == original_bytes
    assert os.path.getsize(tract_path) == 0  # truncated, same as the non-consumed path

    # Byte-fidelity proof: replaying the consumed bytes into a fresh file
    # and draining again must absorb the same turn a second time.
    replay_path = str(tmp_path / "replay.tract")
    with open(replay_path, "wb") as f:
        f.write(consumed)
    replay_absorbed = drain_ingest_tract(cc_ng.graph, cc_ng.vector_db, state, tract_path=replay_path)
    assert replay_absorbed >= 1


def test_drain_ingest_tract_return_consumed_is_empty_on_missing_file(cc_ng, tmp_path):
    absorbed, consumed = drain_ingest_tract(
        cc_ng.graph, cc_ng.vector_db, {"last_forest_id": None},
        tract_path=str(tmp_path / "does_not_exist.tract"), return_consumed=True)
    assert absorbed == 0
    assert consumed == b""


def test_drain_ingest_tract_return_consumed_is_empty_on_parse_failure(cc_ng, tmp_path):
    """A file that fails to even PARSE never reaches the truncate step, so
    return_consumed must report b'' -- nothing was actually removed from
    the file, so nothing should be trickled anywhere for this pulse. The
    file survives untouched for the next retry (or eventual quarantine by
    drain_gateway_conduit, for the Leg-1 conduit case)."""
    bad_path = str(tmp_path / "garbage.tract")
    garbage = b"not a valid BTF tract, ever"
    with open(bad_path, "wb") as f:
        f.write(garbage)

    absorbed, consumed = drain_ingest_tract(
        cc_ng.graph, cc_ng.vector_db, {"last_forest_id": None},
        tract_path=bad_path, return_consumed=True)

    assert consumed == b""
    # File is untouched -- proves truncate never ran on this path.
    with open(bad_path, "rb") as f:
        assert f.read() == garbage


def test_drain_ingest_tract_return_consumed_empty_when_file_changed_underneath(
        cc_ng, tmp_path, monkeypatch):
    """2026-07-27 law-enforcer re-review: the FIRST fix for the data-loss
    Finding unconditionally returned `data` as `consumed` even when the
    file no longer started with `data` at truncate time (someone else wrote
    to it mid-drain) -- reporting bytes as gone that were never actually
    removed. That would DUPLICATE-trickle: the laptop would re-drain and
    re-send the same turn a pulse later, since the file still has it, while
    the caller already believes it was consumed. consumed must be b"" here."""
    import ng_tract
    tract_path = str(tmp_path / "turns.tract")
    ng_tract.deposit_experience(
        content=b"a turn whose file gets rewritten mid-drain",
        source="cc_gateway", tract_path=tract_path, content_type="text",
    )

    other_content = b"something else wrote this while we were mid-drain"
    real_dual_pass = cc_ng_organism.run_conversational_dual_pass

    def _mutate_file_then_pass_through(*args, **kwargs):
        with open(tract_path, "wb") as f:
            f.write(other_content)
        return real_dual_pass(*args, **kwargs)

    monkeypatch.setattr(cc_ng_organism, "run_conversational_dual_pass",
                         _mutate_file_then_pass_through)

    absorbed, consumed = drain_ingest_tract(
        cc_ng.graph, cc_ng.vector_db, {"last_forest_id": None},
        tract_path=tract_path, return_consumed=True)

    assert consumed == b""  # nothing of ours was actually removed
    with open(tract_path, "rb") as f:
        assert f.read() == other_content  # the mutation is preserved, not clobbered


def test_drain_ingest_tract_return_consumed_empty_when_truncate_write_fails(
        cc_ng, tmp_path, monkeypatch):
    """Same Finding, second edge case: the truncate step's own I/O can fail
    (disk full, permissions, etc.) after data was already absorbed into the
    dual-pass -- consumed must be b"" (the write never actually landed), not
    the unconditional `data` a naive fix would still report."""
    import builtins
    import ng_tract
    tract_path = str(tmp_path / "turns.tract")
    ng_tract.deposit_experience(
        content=b"a turn whose truncate write fails",
        source="cc_gateway", tract_path=tract_path, content_type="text",
    )

    real_open = builtins.open

    def _flaky_open(path, mode="r", *args, **kwargs):
        if str(path) == tract_path and mode == "wb":
            raise OSError("simulated disk-full on truncate write")
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", _flaky_open)

    absorbed, consumed = drain_ingest_tract(
        cc_ng.graph, cc_ng.vector_db, {"last_forest_id": None},
        tract_path=tract_path, return_consumed=True)

    assert absorbed >= 1        # the entry WAS absorbed into the dual-pass
    assert consumed == b""      # but the truncate write never landed



# =============================================================================
# FatherGraph absorption discipline (Findings 1 + 3)
#
# The drain must NOT bulk-dump. Finding 1: "the drain can't be a bulk dump...
# New topology must arrive gradually enough that the receiving topology's
# homeostatic regulation can absorb it without displacement." Finding 3:
# "After receiving a merge batch, run idle steps (~250) BEFORE accepting the
# next batch" -- measured 47%->74% accuracy, "not optional -- it's what makes
# merge work."
#
# These drive the REAL drain_gateway_conduit batching/sleep logic, stubbing
# only drain_ingest_tract (the per-file BTF parse + dual-pass, covered
# elsewhere) so the discipline itself is what's under test.
# =============================================================================

class _CountingGraph:
    """Records every graph.step() so consolidation can be asserted, not assumed."""
    def __init__(self):
        self.steps = 0
        self.step_marks = []          # steps-so-far at each consolidation boundary
        self._concurrent_lock = threading.RLock()

    def step(self):
        self.steps += 1


def _stub_drain(monkeypatch, turns_per_file):
    """Replace drain_ingest_tract with a stub that truncates the file (as the
    real one does on success) and reports a fixed turn count."""
    def _fake(graph, vector_db, state, tract_path=None, return_consumed=False):
        with open(tract_path, "wb") as f:
            f.write(b"")          # mimic truncate-after-successful-drain
        return turns_per_file
    monkeypatch.setattr(cc_ng_organism, "drain_ingest_tract", _fake)


def _make_conduit(tmp_path, n_files, prefix="laptop_cc_gateway"):
    conduit_dir = str(tmp_path / "conduit")
    os.makedirs(conduit_dir, exist_ok=True)
    for i in range(n_files):
        with open(os.path.join(conduit_dir, f"{prefix}.{1000+i}_b{i}.tract"), "wb") as f:
            f.write(b"payload")
    return conduit_dir


def test_drain_sleeps_between_batches_not_after_bulk_dump(tmp_path, draining_hemisphere, monkeypatch):
    """6 files x 1 turn, batch_size=2, idle_steps=5 -> consolidation must run
    after every 2nd turn (3 times), NOT once at the end. Proves the sleep is
    interleaved (Finding 3), not tacked on after a bulk dump (Finding 1)."""
    conduit_dir = _make_conduit(tmp_path, 6)
    _stub_drain(monkeypatch, turns_per_file=1)
    g = _CountingGraph()

    absorbed = drain_gateway_conduit(g, None, {}, conduit_dir=conduit_dir,
                                      batch_size=2, idle_steps=5, load_ceiling=999.0)

    assert absorbed == 6
    # 6 turns / batch 2 = 3 consolidation passes x 5 steps each
    assert g.steps == 15, f"expected 3 interleaved sleeps (15 steps), got {g.steps}"


def test_drain_consolidates_trailing_partial_batch(tmp_path, draining_hemisphere, monkeypatch):
    """5 turns with batch_size=2 -> two full-batch sleeps plus one trailing
    sleep for the remaining turn, so freshly-merged topology is never left
    unconsolidated when the run ends."""
    conduit_dir = _make_conduit(tmp_path, 5)
    _stub_drain(monkeypatch, turns_per_file=1)
    g = _CountingGraph()

    drain_gateway_conduit(g, None, {}, conduit_dir=conduit_dir, batch_size=2, idle_steps=5,
                            load_ceiling=999.0)

    # 2 full batches (2 sleeps) + 1 trailing turn (1 sleep) = 3 x 5 steps
    assert g.steps == 15


def test_drain_no_consolidation_when_nothing_absorbed(tmp_path, draining_hemisphere, monkeypatch):
    """An empty conduit must not spin the graph for no reason."""
    conduit_dir = _make_conduit(tmp_path, 0)
    _stub_drain(monkeypatch, turns_per_file=0)
    g = _CountingGraph()

    assert drain_gateway_conduit(g, None, {}, conduit_dir=conduit_dir,
                                  batch_size=2, idle_steps=5) == 0
    assert g.steps == 0


def test_drain_stops_and_leaves_files_when_load_is_high(tmp_path, draining_hemisphere, monkeypatch):
    """Backpressure (cc_refeed discipline): above the load ceiling the drain
    stops cleanly and leaves the remaining conduit files ON DISK for the next
    run -- durability IS the backpressure. Nothing may be lost or eaten."""
    conduit_dir = _make_conduit(tmp_path, 4)
    _stub_drain(monkeypatch, turns_per_file=1)
    g = _CountingGraph()

    import cc_refeed
    monkeypatch.setattr(cc_refeed, "should_pause_for_load", lambda *a, **k: True)

    absorbed = drain_gateway_conduit(g, None, {}, conduit_dir=conduit_dir,
                                      batch_size=2, idle_steps=5)

    # Throttle a flowing river, don't dam it before the first drop: the gate
    # applies BETWEEN batches, so file 1 always lands and files 2..N are held.
    # (Checked-BEFORE-the-first-file was the 2026-07-28 bug -- on any box above
    # the ceiling the whole callosum silently absorbed nothing and logged
    # nothing, letting the conduit grow forever while looking like success.)
    assert absorbed == 1, "first file must land; the gate throttles, not dams"
    remaining = glob.glob(os.path.join(conduit_dir, "laptop_cc_gateway.*.tract"))
    assert len(remaining) == 3, "the rest must survive for the next run"


def test_drain_skips_own_hemispheres_outgoing_files(tmp_path, leg1_enabled, monkeypatch):
    """exclude_prefix guard: a hemisphere must never eat its OWN outgoing
    files -- they're addressed to the far half and must survive until it has
    pulled them. Without this, running the drain on the producing machine
    would silently starve the other hemisphere (the turns are already absorbed
    locally, so it would look harmless)."""
    conduit_dir = _make_conduit(tmp_path, 2, prefix="laptop_cc_gateway")
    for i in range(3):
        with open(os.path.join(conduit_dir, f"vps_cc_gateway.{2000+i}_b{i}.tract"), "wb") as f:
            f.write(b"payload")
    _stub_drain(monkeypatch, turns_per_file=1)
    g = _CountingGraph()

    # Running ON the laptop: must drain vps_* and leave laptop_* alone.
    absorbed = drain_gateway_conduit(g, None, {}, conduit_dir=conduit_dir,
                                      batch_size=10, idle_steps=1,
                                      load_ceiling=999.0, exclude_prefix="laptop_")

    assert absorbed == 3, "should absorb only the far hemisphere's 3 files"
    assert len(glob.glob(os.path.join(conduit_dir, "laptop_cc_gateway.*.tract"))) == 2
    assert len(glob.glob(os.path.join(conduit_dir, "vps_cc_gateway.*.tract"))) == 0


def test_drain_guard_defaults_from_machine_id_without_being_passed(
        tmp_path, leg1_enabled, monkeypatch):
    """The self-consumption guard must NOT depend on every caller remembering
    to pass exclude_prefix. It is an invariant of the drain, so it defaults
    from this hemisphere's declared MACHINE_ID (LAW 4 -- enforce where the
    identity is known). Same scenario as the test above, with the argument
    omitted entirely: the outcome must be identical."""
    conduit_dir = _make_conduit(tmp_path, 2, prefix="laptop_cc_gateway")
    for i in range(3):
        with open(os.path.join(conduit_dir, f"vps_cc_gateway.{2000+i}_b{i}.tract"), "wb") as f:
            f.write(b"payload")
    _stub_drain(monkeypatch, turns_per_file=1)
    g = _CountingGraph()

    # MACHINE_ID=laptop from the fixture; NO exclude_prefix argument.
    absorbed = drain_gateway_conduit(g, None, {}, conduit_dir=conduit_dir,
                                      batch_size=10, idle_steps=1, load_ceiling=999.0)

    assert absorbed == 3
    assert len(glob.glob(os.path.join(conduit_dir, "laptop_cc_gateway.*.tract"))) == 2
    assert len(glob.glob(os.path.join(conduit_dir, "vps_cc_gateway.*.tract"))) == 0


def test_drain_refuses_entirely_when_machine_id_is_unset(tmp_path, leg1_enabled, monkeypatch):
    """With no declared identity the guard cannot be built, so the drain must
    refuse rather than run unguarded. Draining unguarded would absorb AND
    DELETE this hemisphere's own outgoing files before the far half pulled
    them -- silent one-way data loss that looks exactly like success. Mirrors
    trickle_gateway_conduit()'s refusal to write without MACHINE_ID."""
    monkeypatch.delenv("MACHINE_ID", raising=False)
    conduit_dir = _make_conduit(tmp_path, 3, prefix="laptop_cc_gateway")
    _stub_drain(monkeypatch, turns_per_file=1)
    g = _CountingGraph()

    absorbed = drain_gateway_conduit(g, None, {}, conduit_dir=conduit_dir,
                                      batch_size=10, idle_steps=1, load_ceiling=999.0)

    assert absorbed == 0
    # Nothing consumed, nothing deleted -- the files stay durable on disk.
    assert len(glob.glob(os.path.join(conduit_dir, "laptop_cc_gateway.*.tract"))) == 3


def test_drain_explicit_exclude_prefix_overrides_machine_id_default(
        tmp_path, leg1_enabled, monkeypatch):
    """An explicit argument is an override, not the guard -- a caller that does
    pass one still wins, so the existing cc-ng-sync.py call site is unaffected."""
    conduit_dir = _make_conduit(tmp_path, 2, prefix="laptop_cc_gateway")
    _stub_drain(monkeypatch, turns_per_file=1)
    g = _CountingGraph()

    # MACHINE_ID=laptop would skip these; an explicit non-matching prefix drains them.
    absorbed = drain_gateway_conduit(g, None, {}, conduit_dir=conduit_dir,
                                      batch_size=10, idle_steps=1,
                                      load_ceiling=999.0, exclude_prefix="vps_")

    assert absorbed == 2
