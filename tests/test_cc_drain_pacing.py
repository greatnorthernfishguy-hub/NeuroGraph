# ---- Changelog ----
# [2026-10-01] Claude Sonnet 5.5 (Z12 builder, lane drain-pacing-d24, dispatch #12731) -- D24 (re-scoped, Exec P471/P472/P473/P476) NG-side tests
# What: tests for drain_ingest_tract's optional node-count pacing (batch_nodes) + out-parameter receipt,
#   and for _cc_callosum_consolidate's now-loud failure log.
# Why: Exec P471 (the first-drain cap is 25 NODES, a turn's dual pass is ATOMIC), P476 (two-phase: the drain stays
#   caller-locked and DEPOSIT-ONLY; the daemon consolidates after its lock is released), P476(e) (the
#   DEBUG-swallow in _cc_callosum_consolidate is fixed where the drain's phase 2 reuses it).
# How: a real neuro_foundation.Graph (tiny, in memory) with run_conversational_dual_pass stubbed to create a KNOWN
#   number of nodes per turn; tract bytes written through the real ng_tract; the base module for the byte-identity
#   proof is loaded from `git show e4ebf982:cc_ng_organism.py` into a temp dir. No checkpoint, no data/ path, no
#   daemon, no embedder (ng_embed.embed is stubbed).
# -------------------
"""D24 NG-side tests. Which tests FAIL on the base e4ebf982 and why:

  * every pacing / receipt / lock-contract test passes `batch_nodes=` / `receipt=`; the base signature has no such
    parameters -> TypeError (the feature is missing).
  * test_consolidate_failure_is_loud_*: the base logs the failure at DEBUG only -> no WARNING+ record.
  * GUARDS that PASS on both (they are identity proofs, not features): test_unset_parameters_are_byte_identical_*,
    test_consolidate_success_path_is_unchanged, test_drain_default_return_types_unchanged.
"""
import logging
import os
import subprocess
import sys
import threading
import types
import importlib.util
from pathlib import Path

import pytest

_WORKTREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_WORKTREE))

BASE_COMMIT = "e4ebf982b1989fd9066d610b94853bc68bf70d37"

import cc_ng_organism  # noqa: E402
import cc_topology_merge  # noqa: E402
import ng_tract  # noqa: E402
from neuro_foundation import Graph  # noqa: E402

# Exec P379/#770 printed-path preamble: say exactly which files these tests exercise.
print("[P379/#770 preamble] cc_ng_organism     ->", Path(cc_ng_organism.__file__).resolve())
print("[P379/#770 preamble] cc_topology_merge  ->", Path(cc_topology_merge.__file__).resolve())
print("[P379/#770 preamble] worktree root      ->", _WORKTREE)


def test_module_under_test_resolves_inside_worktree():
    for mod in (cc_ng_organism, cc_topology_merge):
        p = Path(mod.__file__).resolve()
        assert _WORKTREE in p.parents, f"{mod.__name__} loaded from {p}, not from the worktree {_WORKTREE}"


# ---------------------------------------------------------------- helpers

class _SpyLock:
    """An RLock that records acquire/release so ordering is observable."""

    def __init__(self, events=None):
        self._l = threading.RLock()
        self.events = events if events is not None else []
        self.depth = 0

    def acquire(self, blocking=True, timeout=-1):
        ok = self._l.acquire(blocking, timeout)
        if ok:
            self.depth += 1
            self.events.append("acquire")
        return ok

    def release(self):
        self._l.release()
        self.depth -= 1
        self.events.append("release")

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()
        return False

    @property
    def held(self):
        return self.depth > 0


def _fake_dual_pass_factory(observed_lock=None, held_log=None):
    """Turn text is `label|n|mode`. Creates n nodes `label::i` (none if the label already landed -- content-hashed
    ids, an exact-repeat turn creates nothing). mode 'bound' wires them with synapses, 'unbound' leaves them bare."""

    def dual_pass(graph, vdb, text, emb, state):
        if held_log is not None and observed_lock is not None:
            held_log.append(observed_lock.held)
        label, n, mode = str(text).split("|")
        n = int(n)
        ids = [f"{label}::{i}" for i in range(n)]
        for nid in ids:
            if nid not in graph.nodes:
                graph.create_node(node_id=nid, metadata={"cc": True})
        if mode == "bound":
            for nid in ids[1:]:
                graph.create_synapse(ids[0], nid, weight=0.2)
        return True

    return dual_pass


@pytest.fixture
def rig(tmp_path, monkeypatch):
    import ng_embed
    monkeypatch.setattr(ng_embed, "embed", lambda text: text)
    monkeypatch.setattr(cc_ng_organism, "run_conversational_dual_pass", _fake_dual_pass_factory())
    g = Graph()
    g._concurrent_lock = _SpyLock()
    path = tmp_path / "turns.tract"

    def write(*turns):
        """turns: (label, n_nodes) or (label, n_nodes, mode)"""
        for t in turns:
            label, n = t[0], t[1]
            mode = t[2] if len(t) > 2 else "bound"
            ng_tract.deposit_experience(raw=f"{label}|{n}|{mode}".encode(), source="cc_gateway",
                                        tract_paths=[str(path)])

    def remaining():
        data = path.read_bytes() if path.exists() else b""
        return [e.content for e in ng_tract.TractReader(data)] if data else []

    return types.SimpleNamespace(graph=g, path=path, write=write, remaining=remaining,
                                 state={"last_forest_id": None}, tmp=tmp_path)


def _drain(rig, **kw):
    return cc_ng_organism.drain_ingest_tract(rig.graph, None, rig.state, tract_path=str(rig.path), **kw)


# ---------------------------------------------------------------- pacing by NODES

def test_batch_ends_on_whole_turn_boundary_at_or_over_size(rig):
    rig.write(("a", 10), ("b", 10), ("c", 10), ("d", 10))
    receipt = {}
    absorbed = _drain(rig, batch_nodes=25, receipt=receipt)
    # a(10) b(20) c(30>=25): c is absorbed WHOLE (never split), the batch ends; d stays.
    assert absorbed == 3
    assert len(rig.graph.nodes) == 30
    assert receipt["nodes_created"] == 30
    assert receipt["ended_on_size"] is True
    assert receipt["reason"] == "size_reached"
    assert len(rig.remaining()) == 1 and rig.remaining()[0].startswith("d|")


def test_next_call_takes_the_remainder_and_reports_exhausted(rig):
    rig.write(("a", 10), ("b", 10), ("c", 10), ("d", 10))
    _drain(rig, batch_nodes=25, receipt={})
    receipt = {}
    absorbed = _drain(rig, batch_nodes=25, receipt=receipt)
    assert absorbed == 1
    assert receipt["nodes_created"] == 10
    assert receipt["ended_on_size"] is False
    assert receipt["reason"] == "tract_exhausted"
    assert rig.remaining() == []


def test_single_turn_alone_over_size_is_one_batch(rig):
    rig.write(("big", 40), ("small", 5))
    receipt = {}
    absorbed = _drain(rig, batch_nodes=25, receipt=receipt)
    assert absorbed == 1
    assert receipt["nodes_created"] == 40
    assert receipt["ended_on_size"] is True
    assert [t.split("|")[0] for t in rig.remaining()] == ["small"]


def test_exactly_size_ends_the_batch(rig):
    rig.write(("a", 25), ("b", 5))
    receipt = {}
    assert _drain(rig, batch_nodes=25, receipt=receipt) == 1
    assert receipt["ended_on_size"] is True
    assert len(rig.remaining()) == 1


def test_exact_repeat_turn_creates_no_nodes_and_does_not_advance_the_cap(rig):
    # content-hashed ids: the second 'a' lands on the same nodes -> delta 0.
    rig.write(("a", 20), ("a", 20), ("b", 20))
    receipt = {}
    absorbed = _drain(rig, batch_nodes=25, receipt=receipt)
    assert absorbed == 3          # a(20) a(+0 = 20) b(+20 = 40 >= 25)
    assert receipt["nodes_created"] == 40
    assert receipt["ended_on_size"] is True


def test_consumed_bytes_are_exactly_the_removed_span(rig):
    rig.write(("a", 10), ("b", 10), ("c", 10), ("d", 10))
    original = rig.path.read_bytes()
    absorbed, consumed = _drain(rig, batch_nodes=25, receipt={}, return_consumed=True)
    assert absorbed == 3
    assert consumed + rig.path.read_bytes() == original   # byte-exact: nothing lost, nothing claimed twice
    assert consumed and rig.path.read_bytes()


def test_max_entries_and_batch_nodes_both_apply_whichever_first(rig):
    rig.write(("a", 1), ("b", 1), ("c", 1), ("d", 1))
    receipt = {}
    assert _drain(rig, batch_nodes=25, max_entries=2, receipt=receipt) == 2
    assert receipt["ended_on_size"] is False
    assert receipt["reason"] == "entries_cap_reached"
    assert len(rig.remaining()) == 2


@pytest.mark.parametrize("bad", [0, -5, "nope", None])
def test_nonpositive_or_invalid_batch_nodes_is_unpaced(rig, bad):
    rig.write(("a", 30), ("b", 30))
    receipt = {}
    assert _drain(rig, batch_nodes=bad, receipt=receipt) == 2    # whole file, exactly as with no pacing
    assert receipt["ended_on_size"] is False
    assert receipt["reason"] == "tract_exhausted"
    assert rig.remaining() == []


# ---------------------------------------------------------------- the receipt

def test_receipt_arrivals_are_the_set_difference_of_node_ids(rig):
    rig.graph.create_node(node_id="pre-existing", metadata={})
    rig.write(("a", 3), ("b", 2))
    receipt = {}
    _drain(rig, batch_nodes=25, receipt=receipt)
    assert receipt["arrivals"] == {"a::0", "a::1", "a::2", "b::0", "b::1"}
    assert "pre-existing" not in receipt["arrivals"]
    assert receipt["turns_taken"] == 2


def test_receipt_is_reset_on_early_return_never_stale(rig):
    receipt = {"nodes_created": 99, "ended_on_size": True, "arrivals": {"stale"}, "reason": "size_reached"}
    absorbed = _drain(rig, batch_nodes=25, receipt=receipt)   # no tract file at all
    assert absorbed == 0
    assert receipt["nodes_created"] == 0
    assert receipt["ended_on_size"] is False
    assert receipt["arrivals"] == set()
    assert receipt["reason"] == "no_batch"


class _RaisingReceipt(dict):
    def __setitem__(self, k, v):
        raise RuntimeError("secret-detail-that-must-not-be-logged")

    def update(self, *a, **k):
        raise RuntimeError("secret-detail-that-must-not-be-logged")

    def clear(self):
        raise RuntimeError("secret-detail-that-must-not-be-logged")


def test_raising_receipt_changes_nothing_and_logs_class_only(rig, caplog):
    rig.write(("a", 10), ("b", 10), ("c", 10), ("d", 10))
    before = rig.path.read_bytes()
    with caplog.at_level(logging.DEBUG, logger=cc_ng_organism.logger.name):
        absorbed = _drain(rig, batch_nodes=25, receipt=_RaisingReceipt())
    assert absorbed == 3                                   # same result as with a good receipt
    assert len(rig.graph.nodes) == 30
    assert len(rig.remaining()) == 1                       # same truncation
    assert rig.path.read_bytes() != before
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "RuntimeError" in text and "secret-detail" not in text
    assert any(r.levelno >= logging.WARNING for r in caplog.records)


# ---------------------------------------------------------------- P476(a): caller-locked, deposit-only

def test_drain_is_deposit_only_and_never_touches_the_callers_lock(rig, monkeypatch):
    held = []
    monkeypatch.setattr(cc_ng_organism, "run_conversational_dual_pass",
                        _fake_dual_pass_factory(rig.graph._concurrent_lock, held))
    consolidations = []
    monkeypatch.setattr(cc_ng_organism, "_cc_callosum_consolidate",
                        lambda *a, **k: consolidations.append(a) or True)
    rig.write(("a", 15), ("b", 15), ("c", 15))
    lock = rig.graph._concurrent_lock
    steps_before = rig.graph.timestep
    with lock:                                   # the CALLER holds it, exactly as the daemon's autosave does
        absorbed = _drain(rig, batch_nodes=25, receipt={})
    assert absorbed == 2
    # The drain made ZERO acquire/release calls of its own: only the caller's pair.
    assert lock.events == ["acquire", "release"]
    # Every deposit ran while the caller's hold was in force (held for the deposit exactly as at the base).
    assert held and all(held)
    # Deposit-only: no step, no consolidation inside the drain, even though the batch ended on size.
    assert rig.graph.timestep == steps_before
    assert consolidations == []


def test_drain_default_return_types_unchanged(rig):
    rig.write(("a", 1))
    assert _drain(rig) == 1 and isinstance(_drain(rig), int)
    rig.write(("b", 1))
    out = _drain(rig, return_consumed=True)
    assert isinstance(out, tuple) and out[0] == 1 and isinstance(out[1], bytes)


def test_receipt_and_batch_nodes_do_not_change_the_return_shape(rig):
    rig.write(("c", 1))
    out = _drain(rig, batch_nodes=25, receipt={})
    assert out == 1 and isinstance(out, int)
    rig.write(("d", 1))
    out = _drain(rig, batch_nodes=25, receipt={}, return_consumed=True)
    assert isinstance(out, tuple) and out[0] == 1 and isinstance(out[1], bytes)


# ---------------------------------------------------------------- the guard, arrival-scoped (reuse of the merge's own predicate)

def test_receipt_arrivals_feed_the_merges_unbound_guard_and_scope_is_arrivals_only(rig):
    # A node that was ALREADY unbound before this drain (the laptop's ~806, CC-CALLOSUM-TRUTH.md:310) ...
    rig.graph.create_node(node_id="pre-existing-orphan", metadata={})
    rig.write(("ok", 4, "bound"))
    receipt = {}
    _drain(rig, batch_nodes=25, receipt=receipt)
    unbound = cc_topology_merge._unbound_nodes(rig.graph, receipt["arrivals"])
    # ... must NOT trip an arrival-scoped guard; a whole-graph guard WOULD (this is why the scope is the arrivals).
    assert unbound == set()
    assert cc_topology_merge._unbound_nodes(rig.graph, set(rig.graph.nodes)) == {"pre-existing-orphan"}


def test_unbound_arrival_is_visible_to_the_guard(rig):
    rig.write(("ok", 3, "bound"), ("bare", 2, "unbound"))
    receipt = {}
    _drain(rig, batch_nodes=25, receipt=receipt)
    assert cc_topology_merge._unbound_nodes(rig.graph, receipt["arrivals"]) == {"bare::0", "bare::1"}


# ---------------------------------------------------------------- P476(e): the DEBUG-swallow fix

class _StepBoom:
    """Minimal graph whose step() raises; the exception text is a canary that must never reach a log line."""
    timestep = 0

    def __init__(self):
        self._concurrent_lock = _SpyLock()
        self.calls = 0

    def step(self):
        self.calls += 1
        if self.calls >= 3:
            raise RuntimeError("canary-str-exc-must-not-be-logged")


def test_consolidate_failure_is_loud_with_class_name_and_still_returns_false(caplog, monkeypatch):
    monkeypatch.setenv("CC_CALLOSUM_LOCK_SLICE_STEPS", "25")
    g = _StepBoom()
    with caplog.at_level(logging.DEBUG, logger=cc_ng_organism.logger.name):
        result = cc_ng_organism._cc_callosum_consolidate(g, 10)
    assert result is False                                            # return value byte-identical
    loud = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert loud, "the consolidation failure was swallowed below WARNING (the silent class)"
    msg = "\n".join(r.getMessage() for r in loud)
    assert "RuntimeError" in msg
    assert "canary-str-exc" not in msg                                # class name only, never str(exc)
    assert not any(r.exc_info for r in loud)                          # no traceback carrying the message either


def test_consolidate_success_path_is_unchanged(caplog, monkeypatch):
    monkeypatch.setenv("CC_CALLOSUM_LOCK_SLICE_STEPS", "3")
    g = Graph()
    g._concurrent_lock = _SpyLock()
    t0 = g.timestep
    with caplog.at_level(logging.DEBUG, logger=cc_ng_organism.logger.name):
        assert cc_ng_organism._cc_callosum_consolidate(g, 7) is True
    assert g.timestep == t0 + 7
    # lock sliced 3+3+1, released between slices (3 acquire/release pairs), exactly as before
    assert g._concurrent_lock.events == ["acquire", "release"] * 3
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert cc_ng_organism._cc_callosum_consolidate(g, 0) is False     # idle_steps <= 0 -> no steps, False
    assert cc_ng_organism._cc_callosum_consolidate(None, 5) is False


# ---------------------------------------------------------------- byte-identity vs the base module

def _load_base_module(tmp_path):
    src = subprocess.run(["git", "show", f"{BASE_COMMIT}:cc_ng_organism.py"], cwd=str(_WORKTREE),
                         capture_output=True, check=True).stdout
    p = tmp_path / "cc_ng_organism_base_e4ebf982.py"
    p.write_bytes(src)
    spec = importlib.util.spec_from_file_location("cc_ng_organism_base_e4ebf982", str(p))
    mod = importlib.util.module_from_spec(spec)
    # dataclasses resolve string annotations through sys.modules[cls.__module__]; register before exec.
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return mod, p


def test_unset_parameters_are_byte_identical_to_the_base_module(tmp_path, monkeypatch):
    import ng_embed
    monkeypatch.setattr(ng_embed, "embed", lambda text: text)
    base, base_path = _load_base_module(tmp_path)
    print("[P379/#770 preamble] BASE module (git show %s) ->" % BASE_COMMIT[:8], base_path)
    assert base_path.parent == tmp_path
    for mod in (base, cc_ng_organism):
        monkeypatch.setattr(mod, "run_conversational_dual_pass", _fake_dual_pass_factory())
    assert base.drain_ingest_tract is not cc_ng_organism.drain_ingest_tract

    scenarios = [
        ("whole file", dict(), [("a", 10), ("b", 10), ("c", 10), ("d", 10)], True),
        ("max_entries=2", dict(max_entries=2), [("a", 3), ("b", 3), ("c", 3)], True),
        ("return_consumed", dict(return_consumed=True), [("a", 4), ("b", 4)], True),
        ("return_consumed+max_entries", dict(return_consumed=True, max_entries=1), [("a", 4), ("b", 4)], True),
        ("empty file", dict(), [], True),
        ("missing file", dict(), None, False),
    ]
    for name, kw, turns, create in scenarios:
        # ONE source tract per scenario (entries carry timestamps, so two independent writes differ in bytes);
        # base and tip each get a byte-identical copy.
        src_bytes = None
        if create:
            src = tmp_path / f"src_{name.replace(' ', '_').replace('+', '_').replace('=', '')}.tract"
            src.write_bytes(b"")
            for t in turns:
                ng_tract.deposit_experience(raw=f"{t[0]}|{t[1]}|bound".encode(), source="cc_gateway",
                                            tract_paths=[str(src)])
            src_bytes = src.read_bytes()
        outs = []
        for label, mod in (("base", base), ("tip", cc_ng_organism)):
            d = tmp_path / f"{name.replace(' ', '_').replace('+', '_').replace('=', '')}_{label}"
            d.mkdir()
            path = d / "turns.tract"
            if create:
                path.write_bytes(src_bytes)
            g = Graph()
            state = {"last_forest_id": None}
            ret = mod.drain_ingest_tract(g, None, state, tract_path=str(path), **kw)
            outs.append((ret, path.read_bytes() if path.exists() else None, sorted(g.nodes),
                         len(g.synapses), dict(state)))
        assert outs[0] == outs[1], f"tip diverges from base with params unset: {name}"

    # a corrupt file: parse failure -> file untouched, identical on both
    outs = []
    for label, mod in (("base", base), ("tip", cc_ng_organism)):
        d = tmp_path / f"corrupt_{label}"
        d.mkdir()
        path = d / "turns.tract"
        path.write_bytes(b"\xff\x00\x01garbage-not-a-tract")
        ret = mod.drain_ingest_tract(Graph(), None, {}, tract_path=str(path))
        outs.append((ret, path.read_bytes()))
    assert outs[0] == outs[1]
