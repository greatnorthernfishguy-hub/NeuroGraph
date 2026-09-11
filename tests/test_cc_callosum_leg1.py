# tests/test_cc_callosum_leg1.py
#
# ---- Changelog ----
# [2026-09-11] Codex — isolate local/trickle tests; receipt gateway cases moved
#   to test_cc_gateway_durable.py. No real NG/model fixture.
# [2026-07-27] Claude Code (Sonnet 5) — CC Corpus Callosum Leg 1 (#70) tests
# What: Coverage for cc_ng_organism.trickle_gateway_conduit() (laptop-side
#   per-batch conduit write) and drain_gateway_conduit() (VPS-side drain +
#   delete of every conduit file), per docs/superpowers/plans/2026-07-27-cc-
#   corpus-callosum-leg1-spec.md §3: append-correctness (here: per-batch-file
#   correctness, since the built design uses one immutable file per trickle
#   rather than a shared append target -- see spec §2b race note), collision-
#   free filenames under rapid calls, VPS drain-and-delete, gate-off no-op on
#   both sides, and fail-soft on a missing/corrupt conduit dir.
# How: Real ng_tract.deposit_experience()/TractReader round-trips and fake
#   conversational learning (no graph/model construction) for the end-to-end drain proof;
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


# Leg1 is raw conversational experience: no synthetic topology consolidation.
# Old tests forced idle_steps=0 because 250 culled fresh toy-graph deposits;
# Josh's Sep11 correction identifies that behavior as the wrong path, not a toy artifact.

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
def cc_ng(monkeypatch):
    """Transport tests use fake learning, never construct a live NG/model."""
    from types import SimpleNamespace
    import ng_embed
    monkeypatch.setattr(ng_embed, 'embed', lambda text: None)
    monkeypatch.setattr(cc_ng_organism, 'run_conversational_dual_pass', lambda *args: True)
    return SimpleNamespace(graph=SimpleNamespace(_concurrent_lock=threading.RLock()),
                           vector_db=None)


# =============================================================================
# Gate-off default: inert on both sides
# =============================================================================

def test_trickle_gateway_conduit_is_noop_when_gate_off(tmp_path, monkeypatch):
    monkeypatch.setattr(cc_ng_organism, "_CC_CALLOSUM_LEG1_ENABLED", False)
    conduit_dir = str(tmp_path / "conduit")
    result = trickle_gateway_conduit(b"some raw tract bytes", conduit_dir=conduit_dir)
    assert result is None
    assert not os.path.exists(conduit_dir)




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
        raw=b"a turn", source="cc_gateway", tract_paths=[tract_path],
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
        raw=b"a genuinely distinct turn for consumed-bytes fidelity",
        source="cc_gateway", tract_paths=[tract_path],
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


# =============================================================================
# max_entries -- resource bounding and durable partial consumption.
# Real BTF files exercise byte offsets against the compiled TractReader.
# A bounded call must retain all unprocessed records for the next slice.
# =============================================================================

def _deposit_turns(tract_path, n, tag="cap"):
    import ng_tract
    for i in range(n):
        ng_tract.deposit_experience(
            raw=f"{tag} turn number {i}".encode(), source="cc_gateway",
            tract_paths=[tract_path],
        )


def test_drain_ingest_tract_max_entries_caps_absorption_and_keeps_remainder(cc_ng, tmp_path):
    """5 turns, max_entries=2 -> absorb exactly 2, and the file must still hold
    the other 3 as VALID, PARSEABLE tract data. This is the load-bearing claim:
    a partial truncate at a TractReader.position() offset leaves a remainder
    that is itself a well-formed tract, not a corrupt tail."""
    import ng_tract
    tract_path = str(tmp_path / "turns.tract")
    _deposit_turns(tract_path, 5)
    full_size = os.path.getsize(tract_path)
    state = {"last_forest_id": None}

    absorbed = drain_ingest_tract(cc_ng.graph, cc_ng.vector_db, state,
                                  tract_path=tract_path, max_entries=2)

    assert absorbed == 2, "cap must stop at exactly max_entries turns"
    remaining_size = os.path.getsize(tract_path)
    assert 0 < remaining_size < full_size, "remainder must survive, and be smaller"

    with open(tract_path, "rb") as f:
        tail = f.read()
    tail_contents = [e.content for e in ng_tract.TractReader(tail)]
    assert tail_contents == ["cap turn number 2", "cap turn number 3", "cap turn number 4"], (
        "the untouched remainder must parse standalone, in order, with nothing "
        f"dropped or duplicated -- got {tail_contents}")


def test_drain_ingest_tract_repeated_capped_calls_drain_every_turn_exactly_once(cc_ng, tmp_path):
    """Draining 5 turns two-at-a-time must yield 2+2+1 and then stop, with the
    file gone to zero. No turn absorbed twice, none stranded -- the property the
    conduit loop depends on when it comes back to a partially-drained file."""
    tract_path = str(tmp_path / "turns.tract")
    _deposit_turns(tract_path, 5)
    state = {"last_forest_id": None}

    counts = []
    for _ in range(4):
        if not os.path.exists(tract_path) or os.path.getsize(tract_path) == 0:
            break
        counts.append(drain_ingest_tract(cc_ng.graph, cc_ng.vector_db, state,
                                         tract_path=tract_path, max_entries=2))

    assert counts == [2, 2, 1], f"expected 2+2+1 across capped calls, got {counts}"
    assert os.path.getsize(tract_path) == 0, "file must be fully drained at the end"


def test_drain_ingest_tract_max_entries_zero_is_byte_identical_to_uncapped(cc_ng, tmp_path):
    """max_entries=0 (the default) must behave EXACTLY as before the cap
    existed: whole file drained, whole file consumed. Guards the ~dozen
    pre-existing callers that pass no cap at all."""
    tract_path = str(tmp_path / "turns.tract")
    _deposit_turns(tract_path, 4)
    with open(tract_path, "rb") as f:
        original = f.read()
    state = {"last_forest_id": None}

    absorbed, consumed = drain_ingest_tract(
        cc_ng.graph, cc_ng.vector_db, state, tract_path=tract_path,
        return_consumed=True, max_entries=0)

    assert absorbed == 4
    assert consumed == original, "uncapped drain must still consume the entire file"
    assert os.path.getsize(tract_path) == 0


def test_drain_ingest_tract_capped_consumed_bytes_are_exactly_the_removed_prefix(cc_ng, tmp_path):
    """A capped drain must report as `consumed` ONLY the prefix it removed --
    never the whole file it read. Over-reporting here is the duplicate-ingestion
    bug in reverse for Leg 1: the laptop would trickle turns to the VPS that it
    had not actually taken out of its own file, then take them again next pulse."""
    tract_path = str(tmp_path / "turns.tract")
    _deposit_turns(tract_path, 5)
    with open(tract_path, "rb") as f:
        original = f.read()
    state = {"last_forest_id": None}

    absorbed, consumed = drain_ingest_tract(
        cc_ng.graph, cc_ng.vector_db, state, tract_path=tract_path,
        return_consumed=True, max_entries=2)

    assert absorbed == 2
    assert consumed == original[:len(consumed)], "consumed must be a true prefix of what was read"
    assert len(consumed) < len(original), "a capped drain must NOT claim the whole file"
    with open(tract_path, "rb") as f:
        left_on_disk = f.read()
    assert consumed + left_on_disk == original, (
        "consumed bytes + bytes still on disk must reconstruct the original file "
        "exactly -- nothing lost, nothing double-claimed")





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
        raw=b"a turn whose file gets rewritten mid-drain",
        source="cc_gateway", tract_paths=[tract_path],
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
        raw=b"a turn whose truncate write fails",
        source="cc_gateway", tract_paths=[tract_path],
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
