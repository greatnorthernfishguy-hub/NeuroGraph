# ---- Changelog ----
# [2026-10-23] Claude Code (Opus 5.5, laptop) — #920 finished on the retry tract (Josh "go ahead" 2026-10-23; Exec P497/P498/P503)
# What: tests for (1) the bounded retry -> durable append-only QUARANTINE on the retry tract (drain_ingest_tract max_retries),
#   (2) compute OUTSIDE the lock / commit INSIDE (prepare_ingest_tract + drain_ingest_tract prepared/prepared_only), and
#   (3) the vendored ng_embed split (prepare_dual_record + dual_record_outcome(prepared=)) keeping R3 atomicity.
# Why: P497 (bound, quarantine never discard, loud, compute outside / commit inside); the chief's literal count (attempts =
#   max_retries + 1).
# How: the drain tests use the REAL ng_tract writer/reader on tmp files and a FAKE ng_embed + fake dual pass (no model, no
#   network). The R3 tests use the REAL ng_embed class with its TID call replaced in-process (never the live TID) and the
#   embedding replaced by a deterministic fake. Nothing under ~/.claude is opened.
# -------------------
import hashlib
import inspect
import json
import logging
import os
import sys
import threading
import types

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

import pytest

import cc_ng_organism
import ng_tract

_REAL_HOME = os.path.expanduser("~")


def _safe(p):
    p = os.path.realpath(str(p))
    assert not p.startswith(os.path.join(_REAL_HOME, ".claude") + os.sep), p
    assert "plugins/neurograph" not in p, p
    return p


def _frame(tmp, text, n):
    p = _safe(tmp / ("f%d.tract" % n))
    if os.path.exists(p):
        os.remove(p)
    ng_tract.deposit_experience(raw=text.encode(), source="cc_gateway", tract_paths=[p])
    with open(p, "rb") as f:
        return f.read()


def _write(path, frames):
    with open(_safe(path), "wb") as f:
        f.write(b"".join(frames))
    return str(path)


def _read(path):
    try:
        with open(path, "rb") as f:
            return f.read()
    except FileNotFoundError:
        return b""


def _frames_of(data):
    out, off = [], 0
    r = ng_tract.TractReader(data)
    for _e in r:
        end = r.position()
        out.append(data[off:end])
        off = end
    return out


class _FakeNGEmbed:
    """Fake of the split ng_embed API (the signature check in prepared_dual_pass_supported reads `prepared`)."""
    calls = []
    lock = None
    seen_locked = []

    @classmethod
    def get_instance(cls):
        return cls()

    def embed_windows(self, text):
        return types.SimpleNamespace(windows=())

    def prepare_dual_record(self, content):
        _FakeNGEmbed.calls.append(content)
        if _FakeNGEmbed.lock is not None:
            _FakeNGEmbed.seen_locked.append(_FakeNGEmbed.lock.locked())
        if content.startswith("POISON"):
            return {"concepts": None, "tree_embeddings": [], "failure_class": "ReadTimeout"}
        return {"concepts": ["c"], "tree_embeddings": [[0.0]], "failure_class": None}

    def dual_record_outcome(self, ecosystem, content, embedding, target_id, success, strength=1.0, metadata=None,
                            *, windows=None, prepared=None):
        raise AssertionError("not used by the fake dual pass")


@pytest.fixture
def env(tmp_path, monkeypatch):
    fake = types.ModuleType("ng_embed")
    fake.embed = lambda text: [0.0]
    fake.NGEmbed = _FakeNGEmbed
    monkeypatch.setitem(sys.modules, "ng_embed", fake)
    monkeypatch.setattr(cc_ng_organism, "cc_gateway_tract_path",
                        lambda: (_ for _ in ()).throw(AssertionError("default tract path used")))
    _FakeNGEmbed.calls = []
    _FakeNGEmbed.lock = None
    _FakeNGEmbed.seen_locked = []
    calls = []

    def dual(graph, vdb, text, emb, state, prepared=None):
        calls.append((text, prepared is not None))
        if prepared is not None:
            return bool(prepared.get("ok"))
        return not (text.startswith("POISON") or text.startswith("FALSE"))

    monkeypatch.setattr(cc_ng_organism, "run_conversational_dual_pass", dual)
    return types.SimpleNamespace(tmp=tmp_path, calls=calls)


def _drain(path, state=None, **kw):
    state = {"last_forest_id": None} if state is None else state
    r = cc_ng_organism.drain_ingest_tract(types.SimpleNamespace(nodes={}), None, state, tract_path=path, return_consumed=True, **kw)
    return r, state


def _retry_pass(path, state=None, max_retries=5, **kw):
    return _drain(path, state, hold_on_failure=True, retry_tract_path=path, max_retries=max_retries, **kw)


# ------------------------------------------------------------------ the bound, then quarantine
def test_bound_then_quarantine_after_max_retries_plus_one_attempts(env, caplog):
    poison = _frame(env.tmp, "POISON turn", 1)
    retry = _write(env.tmp / "retry.tract", [poison])
    caplog.set_level(logging.WARNING, logger="cc_ng_organism")
    state = {"last_forest_id": None}
    # attempt 1 was the main-tract failure that moved it here; retries 1..4 rotate, retry 5 quarantines
    for i in range(1, 5):
        _retry_pass(retry, state, max_retries=5)
        assert _read(retry) == poison, "retry %d must rotate, not drop" % i
        assert _read(retry + ".quarantine") == b""
        counts = json.load(open(retry + ".retries"))
        assert counts == {hashlib.sha256(poison).hexdigest(): i}
    _retry_pass(retry, state, max_retries=5)
    assert _read(retry) == b""                         # the tract moved past it
    assert _read(retry + ".quarantine") == poison      # WHOLE, byte-for-byte
    assert state["quarantined_total"] == 1
    assert json.load(open(retry + ".retries")) == {}   # its count ended
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1
    msg = errors[0].getMessage()
    assert "QUARANTINED" in msg and "class=ReadTimeout" not in msg   # fake dual pass: no prepared class here
    assert "after 6 attempts (= max_retries + 1), max_retries=5" in msg
    assert "turn=forest:" in msg and "entry:" + hashlib.sha256(poison).hexdigest()[:12] in msg
    assert "POISON" not in msg                         # never the text


def test_quarantine_error_carries_the_prepared_failure_class(env, caplog):
    poison = _frame(env.tmp, "POISON turn", 1)
    retry = _write(env.tmp / "retry.tract", [poison])
    caplog.set_level(logging.WARNING, logger="cc_ng_organism")
    cache = {}
    cc_ng_organism.prepare_ingest_tract(retry, cache, max_entries=4)
    _retry_pass(retry, max_retries=1, prepared=cache, prepared_only=True)
    err = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert len(err) == 1 and "class=ReadTimeout" in err[0]


def test_retry_count_is_durable_across_a_restart(env):
    poison = _frame(env.tmp, "POISON turn", 1)
    retry = _write(env.tmp / "retry.tract", [poison])
    for _ in range(2):
        _retry_pass(retry, {"last_forest_id": None}, max_retries=3)   # a fresh state each time = a restarted daemon
    assert _read(retry) == poison
    _retry_pass(retry, {"last_forest_id": None}, max_retries=3)
    assert _read(retry) == b"" and _read(retry + ".quarantine") == poison


def test_success_ends_the_count(env):
    good = _frame(env.tmp, "GOOD turn", 1)
    retry = _write(env.tmp / "retry.tract", [good])
    with open(retry + ".retries", "w") as f:
        json.dump({hashlib.sha256(good).hexdigest(): 3}, f)
    (n, consumed), _ = _retry_pass(retry)
    assert n == 1 and consumed == good and _read(retry) == b""
    assert json.load(open(retry + ".retries")) == {}


# ------------------------------------------------------------------ the quarantine file
def test_quarantine_is_append_only_and_idempotent(env):
    old = _frame(env.tmp, "OLD quarantined turn", 0)
    p1, p2 = _frame(env.tmp, "POISON one", 1), _frame(env.tmp, "POISON two", 2)
    retry = _write(env.tmp / "retry.tract", [p1, p2])
    with open(retry + ".quarantine", "wb") as f:
        f.write(old)
    _retry_pass(retry, max_retries=1)
    assert _read(retry + ".quarantine") == old + p1 + p2     # appended, never rewritten
    # crash between the quarantine fsync and the tract rewrite: the same frame comes round again
    retry2 = _write(env.tmp / "retry.tract", [p2])
    st = {"last_forest_id": None}
    _retry_pass(retry2, st, max_retries=1)
    assert _read(retry + ".quarantine") == old + p1 + p2     # not appended twice
    assert "quarantined_total" not in st                      # already present: not counted again


def test_quarantine_write_failure_rotates_and_never_drops(env, caplog, monkeypatch):
    poison = _frame(env.tmp, "POISON turn", 1)
    retry = _write(env.tmp / "retry.tract", [poison])
    caplog.set_level(logging.WARNING, logger="cc_ng_organism")
    monkeypatch.setattr(cc_ng_organism, "_drain_quarantine_frame", lambda p, f: (False, None, "OSError"))
    _retry_pass(retry, max_retries=1)
    assert _read(retry) == poison                              # still in the tract, whole
    msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert len(msgs) == 1 and "QUARANTINE FAILED" in msgs[0] and "quarantine_exc_type=OSError" in msgs[0]


def test_quarantine_fsyncs_file_and_directory_before_returning(env, monkeypatch):
    synced = []
    real = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (synced.append(fd), real(fd))[1])
    path = _write(env.tmp / "t.tract", [])
    ok, stored, _ = cc_ng_organism._drain_quarantine_frame(path, b"abc")
    assert ok and stored == "appended" and len(synced) == 2
    ok, stored, _ = cc_ng_organism._drain_quarantine_frame(path, b"abc")
    assert ok and stored == "already_present" and len(synced) == 4   # synced again even when present


# ------------------------------------------------------------------ nothing is ever dropped
def test_nothing_dropped_mixed_pass(env):
    texts = ["GOOD a", "POISON b", "GOOD c", "POISON d", "GOOD e"]
    frames = [_frame(env.tmp, t, i) for i, t in enumerate(texts)]
    retry = _write(env.tmp / "retry.tract", frames)
    absorbed_total = 0
    for _ in range(4):
        (n, _c), _ = _retry_pass(retry, max_retries=2)
        absorbed_total += n
    left = _frames_of(_read(retry))
    quarantined = _frames_of(_read(retry + ".quarantine"))
    assert absorbed_total == 3
    assert sorted(left + quarantined) == sorted([frames[1], frames[3]])   # every failing frame is somewhere, whole
    assert quarantined == [frames[1], frames[3]]


def test_one_attempt_per_entry_per_pass_even_when_the_batch_exceeds_the_tract(env):
    poison = _frame(env.tmp, "POISON turn", 1)
    retry = _write(env.tmp / "retry.tract", [poison])
    _retry_pass(retry, max_retries=5, max_entries=8)
    assert [t for t, _ in env.calls] == ["POISON turn"]       # rotated to the end, but not re-attempted this pass


# ------------------------------------------------------------------ batch cap
def test_retry_batch_cap(env):
    frames = [_frame(env.tmp, "GOOD %d" % i, i) for i in range(6)]
    retry = _write(env.tmp / "retry.tract", frames)
    (n, consumed), _ = _retry_pass(retry, max_entries=4)
    assert n == 4 and consumed == b"".join(frames[:4]) and _read(retry) == b"".join(frames[4:])


# ------------------------------------------------------------------ off / default behaviour
def test_default_no_max_retries_rotates_without_bound_and_writes_no_sidecar(env):
    poison = _frame(env.tmp, "POISON turn", 1)
    retry = _write(env.tmp / "retry.tract", [poison])
    for _ in range(10):
        _drain(retry, hold_on_failure=True, retry_tract_path=retry)
    assert _read(retry) == poison
    assert not os.path.exists(retry + ".retries") and not os.path.exists(retry + ".quarantine")


def test_max_retries_on_the_main_pass_does_not_count(env):
    poison = _frame(env.tmp, "POISON turn", 1)
    main = _write(env.tmp / "turns.tract", [poison])
    retry = str(env.tmp / "retry.tract")
    _drain(main, hold_on_failure=True, retry_tract_path=retry, max_retries=1)
    assert _read(main) == b"" and _read(retry) == poison      # moved: attempt 1
    assert not os.path.exists(main + ".retries") and not os.path.exists(retry + ".retries")


def test_max_retries_without_hold_is_ignored(env):
    poison = _frame(env.tmp, "POISON turn", 1)
    retry = _write(env.tmp / "retry.tract", [poison])
    _drain(retry, retry_tract_path=retry, max_retries=1)   # hold off: the historical drop path, untouched by #920
    assert not os.path.exists(retry + ".quarantine") and not os.path.exists(retry + ".retries")


@pytest.mark.parametrize("bad", [0, -1, "5", True, 2.0])
def test_bad_max_retries_means_no_bound_with_one_warning(env, caplog, bad):
    poison = _frame(env.tmp, "POISON turn", 1)
    retry = _write(env.tmp / "retry.tract", [poison])
    caplog.set_level(logging.WARNING, logger="cc_ng_organism")
    _retry_pass(retry, max_retries=bad)
    assert _read(retry) == poison and not os.path.exists(retry + ".quarantine")
    assert sum("max_retries is not a positive integer" in r.getMessage() for r in caplog.records) == 1


# ------------------------------------------------------------------ compute outside, commit inside
def test_prepared_only_stops_at_an_unprepared_turn_and_leaves_it_untouched(env):
    frames = [_frame(env.tmp, "GOOD %d" % i, i) for i in range(3)]
    path = _write(env.tmp / "turns.tract", frames)
    cache = {}
    s = cc_ng_organism.prepare_ingest_tract(path, cache, max_entries=2)
    assert s["prepared"] == 2 and s["supported"]
    rec = {}
    (n, consumed), _ = _drain(path, prepared=cache, prepared_only=True, receipt=rec, hold_on_failure=True)
    assert n == 2 and consumed == frames[0] + frames[1]
    assert _read(path) == frames[2] and rec["reason"] == "unprepared"
    assert env.calls == [("GOOD 0", True), ("GOOD 1", True)] and cache == {}   # used entries are removed


def test_prepared_only_with_hold_off_never_truncates_the_unprepared_turn(env):
    frames = [_frame(env.tmp, "GOOD %d" % i, i) for i in range(2)]
    path = _write(env.tmp / "turns.tract", frames)
    (n, consumed), _ = _drain(path, prepared={}, prepared_only=True)
    assert n == 0 and consumed == b"" and _read(path) == b"".join(frames)


def test_tid_runs_outside_the_lock_and_the_commit_inside(env):
    """The TID/embed compute (prepare) runs with the lock FREE; the drain under the lock calls no TID at all."""
    lock = threading.RLock()
    _FakeNGEmbed.lock = types.SimpleNamespace(locked=lambda: lock._is_owned())
    frames = [_frame(env.tmp, "GOOD %d" % i, i) for i in range(3)]
    path = _write(env.tmp / "turns.tract", frames)
    cache = {}
    cc_ng_organism.prepare_ingest_tract(path, cache, max_entries=3)        # no lock
    n_prepare_calls = len(_FakeNGEmbed.calls)
    with lock:
        (n, _c), _ = _drain(path, prepared=cache, prepared_only=True, hold_on_failure=True)
    assert n == 3
    assert n_prepare_calls == 3 and len(_FakeNGEmbed.calls) == 3           # nothing extracted under the lock
    assert _FakeNGEmbed.seen_locked == [False, False, False]


def test_prepare_stop_on_failure_bounds_tid_failures_to_one_per_pass(env):
    frames = [_frame(env.tmp, t, i) for i, t in enumerate(["POISON a", "POISON b", "GOOD c"])]
    path = _write(env.tmp / "retry.tract", frames)
    cache = {}
    s = cc_ng_organism.prepare_ingest_tract(path, cache, max_entries=3, stop_on_failure=True)
    assert s["failed"] == 1 and s["examined"] == 1 and s["classes"] == {"ReadTimeout": 1}
    # the drain then attempts exactly the one prepared (failed) turn and stops at the next
    (n, _c), _ = _retry_pass(path, max_retries=5, prepared=cache, prepared_only=True)
    assert n == 0 and _frames_of(_read(path)) == [frames[1], frames[2], frames[0]]


def test_prepare_skips_cached_successes_and_the_defer_size(env):
    frames = [_frame(env.tmp, "GOOD " + "x" * 50, 1), _frame(env.tmp, "GOOD short", 2)]
    path = _write(env.tmp / "turns.tract", frames)
    cache = {}
    s = cc_ng_organism.prepare_ingest_tract(path, cache, max_entries=4, defer_over_bytes=20)
    assert s["examined"] == 1 and _FakeNGEmbed.calls == ["GOOD short"]
    s = cc_ng_organism.prepare_ingest_tract(path, cache, max_entries=4, defer_over_bytes=20)
    assert s["cached"] == 1 and s["prepared"] == 0 and len(_FakeNGEmbed.calls) == 1


def test_prepare_never_writes_the_tract(env):
    frames = [_frame(env.tmp, "GOOD %d" % i, i) for i in range(2)]
    path = _write(env.tmp / "turns.tract", frames)
    before = os.stat(path)
    cc_ng_organism.prepare_ingest_tract(path, {}, max_entries=2)
    assert _read(path) == b"".join(frames) and os.stat(path).st_mtime_ns == before.st_mtime_ns


def test_prepare_unsupported_ng_embed_reports_unsupported(env, monkeypatch):
    mod = types.ModuleType("ng_embed")
    mod.NGEmbed = type("Old", (), {"dual_record_outcome": lambda self, ecosystem, content: None})
    monkeypatch.setitem(sys.modules, "ng_embed", mod)
    assert cc_ng_organism.prepared_dual_pass_supported() is False
    assert cc_ng_organism.prepare_ingest_tract(str(env.tmp / "x"), {}, 3)["supported"] is False


# ------------------------------------------------------------------ the vendored split + R3 (real ng_embed class)
@pytest.fixture
def real_embed(monkeypatch):
    sys.modules.pop("ng_embed", None)
    import importlib
    ne = importlib.import_module("ng_embed")
    assert os.path.realpath(ne.__file__).startswith(_REPO + os.sep)
    inst = ne.NGEmbed.__new__(ne.NGEmbed)
    ne.NGEmbed.__init__(inst, {"tid_endpoint": "http://127.0.0.1:9/never"})   # unroutable: the TID call is replaced below
    inst.embed_batch = lambda texts, **k: [[float(len(t))] for t in texts]
    return ne, inst


class _Eco:
    def __init__(self):
        self.writes = []

    def record_outcome(self, embedding, target_id, success, strength=1.0, metadata=None):
        self.writes.append(target_id)
        return {"ok": True}


def test_prepare_dual_record_is_pure_and_commit_writes_the_same(real_embed, monkeypatch):
    ne, inst = real_embed
    monkeypatch.setattr(inst, "_tid_extract_window", lambda content: ["alpha", "beta"])
    prep = inst.prepare_dual_record("some text")
    assert prep["concepts"] == ["alpha", "beta"] and prep["failure_class"] is None
    eco_a, eco_b = _Eco(), _Eco()
    monkeypatch.setattr(inst, "_tid_extract_window", lambda content: (_ for _ in ()).throw(AssertionError("TID in commit")))
    ra = inst.dual_record_outcome(eco_a, "some text", [0.0], "t", True, prepared=prep)
    monkeypatch.setattr(inst, "_tid_extract_window", lambda content: ["alpha", "beta"])
    rb = inst.dual_record_outcome(eco_b, "some text", [0.0], "t", True)
    assert eco_a.writes == eco_b.writes                       # the prepared commit writes exactly what the single call does
    assert eco_a.writes[0] == "t" and {"t::tree::alpha", "t::tree::beta"} <= set(eco_a.writes)
    assert ra["tree_ids"] == rb["tree_ids"]


def test_r3_prepared_failure_writes_nothing_and_names_the_class(real_embed, monkeypatch, caplog):
    ne, inst = real_embed
    import requests

    def boom(*a, **k):
        raise requests.exceptions.ReadTimeout("SECRET")
    monkeypatch.setattr(requests, "post", boom)
    prep = inst.prepare_dual_record("some text")
    assert prep == {"concepts": None, "tree_embeddings": [], "failure_class": "ReadTimeout"}
    assert inst.stats["dual_pass"]["last_failure_class"] == "ReadTimeout"
    eco = _Eco()
    caplog.set_level(logging.WARNING, logger="ng_embed")
    with pytest.raises(ne.DualPassIncompleteError):
        inst.dual_record_outcome(eco, "some text", [0.0], "t", True, prepared=prep)
    assert eco.writes == []                                              # R3: no forest-only deposit
    msg = " ".join(r.getMessage() for r in caplog.records)
    assert "last_failure=ReadTimeout" in msg and "SECRET" not in msg


def test_r3_prepared_length_mismatch_writes_nothing(real_embed):
    ne, inst = real_embed
    eco = _Eco()
    with pytest.raises(ne.DualPassIncompleteError):
        inst.dual_record_outcome(eco, "x", [0.0], "t", True,
                                 prepared={"concepts": ["a", "b"], "tree_embeddings": [[0.0]], "failure_class": None})
    assert eco.writes == []


def test_default_dual_record_outcome_signature_unchanged_for_old_callers(real_embed):
    ne, _inst = real_embed
    p = inspect.signature(ne.NGEmbed.dual_record_outcome).parameters
    assert p["prepared"].kind is inspect.Parameter.KEYWORD_ONLY and p["prepared"].default is None
    assert list(p)[:8] == ["self", "ecosystem", "content", "embedding", "target_id", "success", "strength", "metadata"]


# ------------------------------------------------------------------ R3 on a REAL graph (real dual pass, TID replaced in-process)
@pytest.fixture
def real_graph(monkeypatch, tmp_path):
    sys.modules.pop("ng_embed", None)
    import importlib
    ne = importlib.import_module("ng_embed")
    monkeypatch.setattr(ne.NGEmbed, "_tid_extract_window", lambda self, content: ["alpha concept", "beta concept"])
    from openclaw_hook import NeuroGraphMemory
    ng = NeuroGraphMemory(workspace_dir=str(tmp_path / "ws"),
                          config={"tonic": {"enabled": False}, "peer_bridge": {"enabled": False}})
    return ne, ng


def _counts(ng):
    return len(ng.graph.nodes), len(ng.graph.synapses), len(ng.graph.hyperedges)


def test_r3_real_graph_prepared_failure_leaves_the_graph_untouched(real_graph, monkeypatch):
    ne, ng = real_graph
    monkeypatch.setattr(ne.NGEmbed, "_tid_extract_window", lambda self, content: None)
    text = "a turn whose extraction times out"
    prep = cc_ng_organism.prepare_conversational_dual_pass(text)
    assert prep["ok"] is False and prep["failure_class"] == "extraction_failed"
    before = _counts(ng)
    state = {"last_forest_id": None}
    assert cc_ng_organism.run_conversational_dual_pass(
        ng.graph, ng.vector_db, text, prep["embedding"], state, prepared=prep) is False
    assert _counts(ng) == before and state == {"last_forest_id": None}


def test_r3_real_graph_prepared_commit_equals_the_single_call(real_graph, tmp_path):
    ne, ng = real_graph
    from openclaw_hook import NeuroGraphMemory
    ng2 = NeuroGraphMemory(workspace_dir=str(tmp_path / "ws2"),
                           config={"tonic": {"enabled": False}, "peer_bridge": {"enabled": False}})
    text = "the same turn, prepared outside the lock or computed inline"
    prep = cc_ng_organism.prepare_conversational_dual_pass(text)
    assert prep["ok"] is True
    a = cc_ng_organism.run_conversational_dual_pass(ng.graph, ng.vector_db, text, prep["embedding"],
                                                     {"last_forest_id": None}, prepared=prep)
    from ng_embed import embed
    b = cc_ng_organism.run_conversational_dual_pass(ng2.graph, ng2.vector_db, text, embed(text),
                                                     {"last_forest_id": None})
    assert a is b is True
    ids_a = sorted(n for n in ng.graph.nodes if str(n).startswith("cc:conv::"))
    ids_b = sorted(n for n in ng2.graph.nodes if str(n).startswith("cc:conv::"))
    assert ids_a == ids_b and len(ids_a) >= 3      # forest + both trees, identical ids either way
