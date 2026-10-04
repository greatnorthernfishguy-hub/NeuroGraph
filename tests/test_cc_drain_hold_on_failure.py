# ---- Changelog ----
# [2026-10-02] Claude Sonnet 5.5 (Z12 builder, lane ng-trial-chain-s4, dispatch #14658) — N8 (TRIAL branch, test-only): F-A, the stale
#   cross-change pin. test_signature_hold_on_failure_is_last_and_defaults_false pinned the exact pre-D24 signature; with D24 in the same tree
#   the signature is [..., max_entries, batch_nodes, receipt, hold_on_failure]. UPDATED to assert the prefix list (now with batch_nodes,
#   receipt), `params[-1] == "hold_on_failure"` and `default is False`: same protection, true on the integrated tree. No other test changed.
# [2026-09-30] Z12 worker (Claude Sonnet 5.5), lane ingest-tract-swallow-781 — #794
# What: tests for drain_ingest_tract(hold_on_failure=...) (Chief-003 ruling / Exec P386):
#   hold semantics, the single hardcoded warning, retry on the next call, and a
#   byte-identity proof for default callers against the BASE module (e4ebf982).
# Why: an entry whose absorb returned False/raised was truncated out of the tract
#   and lost (#794). The opt-in flag holds it; the default must not change at all.
# How: FAKES ONLY. ng_tract is the real installed writer/reader (tract bytes are built
#   in tmp files); ng_embed is a FAKE module injected into sys.modules (no model, no
#   network); run_conversational_dual_pass is monkeypatched per entry text; the graph /
#   vector_db / state are plain fakes. SAFETY: every drain call passes an explicit
#   tract_path under pytest's tmp dir; the fixture asserts it is NOT under the real
#   ~/.claude, ~/.et_modules or any data/ dir, and poisons cc_gateway_tract_path()
#   so a default-path call would fail before opening anything. The LIVE miniTID tract
#   (~/.claude/plugins/neurograph/tracts/cc_gateway/turns.tract) is never opened.
#   Exec P379 / #770: the module under test must be from THIS worktree; the check runs
#   at import (collection) time and fails the session otherwise.
# -------------------
import importlib.util
import inspect
import logging
import os
import pwd
import subprocess
import sys
import types

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

import pytest

import cc_ng_organism
import ng_tract

# --- Exec P379 / #770 session-start preamble (stderr, survives capture) -----------
_ORG_FILE = os.path.realpath(cc_ng_organism.__file__)
_PREAMBLE = (
    "[P379/#770] cc_ng_organism.__file__ = %s\n"
    "[P379/#770] worktree root           = %s\n"
    "[P379/#770] ng_tract.__file__       = %s (REAL installed writer/reader; tract bytes are tmp-file only)\n"
    "[P379/#770] ng_embed                = FAKED (types.ModuleType injected per test; the real one is never imported)\n"
    "[P379/#770] neurograph_rpc in sys.modules: %s\n"
    "[P379/#770] other NG modules in sys.modules: %s\n"
    "[P379/#770] HOME = %s\n" % (
        _ORG_FILE, _REPO, getattr(ng_tract, "__file__", "?"),
        "neurograph_rpc" in sys.modules,
        sorted(m for m in ("neuro_foundation", "openclaw_hook", "ng_lite", "ng_ecosystem",
                           "universal_ingestor", "cc_ng_host", "ng_embed") if m in sys.modules),
        os.environ.get("HOME")))
sys.__stderr__.write("\n" + _PREAMBLE)
if not _ORG_FILE.startswith(_REPO + os.sep):
    raise RuntimeError(
        "P379/#770 FAIL: cc_ng_organism was imported from %s, NOT from the worktree %s"
        % (_ORG_FILE, _REPO))

_REAL_HOME = pwd.getpwuid(os.getuid()).pw_dir
_BASE_SHA = "e4ebf982b1989fd9066d610b94853bc68bf70d37"
SECRET_EXC = "SECRET-EXC-TEXT"
ENTRY_MARK = "ENTRYTEXT"


# ------------------------------------------------------------------ fixtures
@pytest.fixture(scope="session", autouse=True)
def _p379_preamble_visible(request):
    """Print the P379/#770 preamble through the terminal reporter (bypasses capture)."""
    tr = request.config.pluginmanager.get_plugin("terminalreporter")
    if tr is not None:
        tr.write_line("")
        for line in _PREAMBLE.strip().splitlines():
            tr.write_line(line)
        tr.write_line("[P379/#770] worktree check: PASSED (cc_ng_organism is under the worktree)")


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Safety + fakes. Returns a namespace with the tmp dir and a call log."""
    fake_embed = types.ModuleType("ng_embed")
    fake_embed.embed = lambda text: [0.0]
    monkeypatch.setitem(sys.modules, "ng_embed", fake_embed)
    # A default-path call must die BEFORE any open(): poison the resolver and point
    # the env var at the tmp dir as well (covers the loaded BASE module too).
    def _no_default_path():
        raise AssertionError("SAFETY: drain called with the default tract path")
    monkeypatch.setattr(cc_ng_organism, "cc_gateway_tract_path", _no_default_path)
    monkeypatch.setenv("CC_GATEWAY_TRACT_PATH", str(tmp_path / "never-used.tract"))
    calls = []

    def dual(graph, vdb, text, emb, state):
        calls.append(text)
        if text.startswith("FALSE"):
            return False
        if text.startswith("RAISE"):
            raise RuntimeError(SECRET_EXC + " " + text)
        return True

    monkeypatch.setattr(cc_ng_organism, "run_conversational_dual_pass", dual)
    return types.SimpleNamespace(tmp=tmp_path, calls=calls, dual=dual, mp=monkeypatch)


def _check_safe(path):
    p = os.path.realpath(str(path))
    for bad in (os.path.join(_REAL_HOME, ".claude"), os.path.join(_REAL_HOME, ".et_modules")):
        assert not p.startswith(bad + os.sep), "SAFETY: %s is under %s" % (p, bad)
    assert "%sdata%s" % (os.sep, os.sep) not in p, "SAFETY: %s is under a data/ dir" % p
    assert "plugins/neurograph" not in p, "SAFETY: %s looks like the live tract" % p
    assert p.startswith(os.path.realpath("/tmp")) or "pytest" in p, "SAFETY: not under a pytest tmp dir: %s" % p
    return p


def _frame(tmp, kind, n):
    """Real BTF bytes for one entry, built via the real writer into a tmp file."""
    p = str(tmp / ("frame_%s_%d.tract" % (kind, n)))
    _check_safe(p)
    if os.path.exists(p):       # deposit_* APPENDS: never build onto a stale file
        os.remove(p)
    text = "%s %s-%d" % (kind.upper(), ENTRY_MARK, n)
    if kind in ("ok", "false", "raise"):
        ng_tract.deposit_experience(raw=text.encode(), source="cc_gateway", tract_paths=[p])
    elif kind == "skip_source":
        ng_tract.deposit_experience(raw=b"other source", source="other", tract_paths=[p])
    elif kind == "skip_empty":
        ng_tract.deposit_experience(raw=b"   ", source="cc_gateway", tract_paths=[p])
    elif kind == "skip_type":
        ng_tract.deposit_topology(raw=b"topo", source="cc_gateway", tract_paths=[p])
    else:
        raise ValueError(kind)
    with open(p, "rb") as f:
        return f.read()


def _write(env, name, frames, tail=b""):
    path = str(env.tmp / (name + ".tract"))
    _check_safe(path)
    with open(path, "wb") as f:
        f.write(b"".join(frames) + tail)
    return path


def _read(path):
    with open(path, "rb") as f:
        return f.read()


# scenario name -> (list of (kind, n), max_entries, tail-bytes)
POISON = b'{"poison":1}\n'
SCENARIOS = {
    "all_success":        ([("ok", 1), ("ok", 2), ("ok", 3)], 0, b""),
    "filter_skips_only":  ([("skip_source", 1), ("skip_empty", 2), ("skip_type", 3)], 0, b""),
    "fail_middle_false":  ([("ok", 1), ("false", 2), ("ok", 3)], 0, b""),
    "fail_middle_raise":  ([("ok", 1), ("raise", 2), ("ok", 3)], 0, b""),
    "fail_first":         ([("false", 1), ("ok", 2)], 0, b""),
    "fail_last":          ([("ok", 1), ("ok", 2), ("false", 3)], 0, b""),
    "cap_fail_inside":    ([("ok", 1), ("false", 2), ("ok", 3)], 2, b""),
    "cap_fail_after":     ([("ok", 1), ("ok", 2), ("false", 3), ("ok", 4)], 2, b""),
    "skips_around_fail":  ([("skip_source", 1), ("ok", 2), ("skip_empty", 3), ("false", 4),
                            ("skip_type", 5), ("ok", 6)], 0, b""),
    "parse_failure":      ([("ok", 1)], 0, POISON),
    "empty_file":         ([], 0, b""),
}


def _build(env, name):
    """Build a scenario's frames ONCE (frames carry timestamps: rebuilding changes the bytes)."""
    spec, cap, tail = SCENARIOS[name]
    return [_frame(env.tmp, k, n) for k, n in spec]


def _run(mod, env, name, rc, hold=None, frames=None, tag=""):
    """Write the scenario file, drain it, return (result, file_bytes_after, original, calls, records, wb_opens, frames)."""
    spec, cap, tail = SCENARIOS[name]
    if frames is None:
        frames = _build(env, name)
    path = _write(env, "%s_%s_%s_%s" % (name, rc, hold, tag), frames, tail)
    original = _read(path)
    env.calls.clear()
    kwargs = dict(tract_path=path, return_consumed=rc, max_entries=cap)
    if hold is not None:
        kwargs["hold_on_failure"] = hold
    records = []

    class _H(logging.Handler):
        def emit(self, rec):
            records.append((rec.levelname, rec.getMessage()))
    h = _H(level=logging.DEBUG)
    lg = logging.getLogger("cc_ng_organism")
    old = lg.level
    lg.addHandler(h)
    lg.setLevel(logging.DEBUG)
    import builtins
    real_open = builtins.open
    wb = []

    def spy(p, mode="r", *a, **k):
        if str(p) == path and "w" in mode:
            wb.append(mode)
        return real_open(p, mode, *a, **k)
    builtins.open = spy
    try:
        result = mod.drain_ingest_tract(object(), None, {"last_forest_id": None}, **kwargs)
    finally:
        builtins.open = real_open
        lg.removeHandler(h)
        lg.setLevel(old)
    return result, _read(path), original, list(env.calls), records, wb, frames


def _split(result, rc):
    return result if rc else (result, None)


# ------------------------------------------------- signature / call-shape guards
def test_signature_hold_on_failure_is_last_and_defaults_false():
    """FAILS on base: no such parameter (KeyError).

    N8 (NG trial integration): on a tree that also carries D24 the two D24 keyword
    parameters `batch_nodes, receipt` sit between `max_entries` and `hold_on_failure`.
    What this pins is unchanged: every parameter the old pin listed is still there, in
    order, and `hold_on_failure` is the LAST one and defaults to False. The prefix is
    therefore asserted explicitly (a dropped or reordered parameter fails), then the
    last-ness, then the default."""
    sig = inspect.signature(cc_ng_organism.drain_ingest_tract)
    params = list(sig.parameters)
    assert params[:-1] == ["graph", "vector_db", "state", "tract_path", "return_consumed",
                           "max_entries", "batch_nodes", "receipt",
                           "max_seconds",   # MVP 2026-10-03: the wall-time cap per pass
                           "retry_tract_path",   # MVP 2026-10-03: failed turns move aside, nothing dropped
                           "defer_tract_path", "defer_over_bytes"]   # MVP 2026-10-03: oversize turns wait aside
    assert params[-1] == "hold_on_failure"
    assert sig.parameters["hold_on_failure"].default is False


def test_existing_caller_shapes_still_bind():
    """Preservation guard (passes on base too): cc_ng_host.py:1526 passes 3 positionals;
    the daemon passes return_consumed=True."""
    sig = inspect.signature(cc_ng_organism.drain_ingest_tract)
    sig.bind(object(), None, {})
    sig.bind(object(), None, {}, return_consumed=True)


# ------------------------------------------------------------- hold semantics
def _expect_hold(env, name, rc, *, absorbed, remainder_from, calls_n, reason=None, exc_type="-"):
    result, after, original, calls, records, wb, frames = _run(cc_ng_organism, env, name, rc, hold=True)
    n_abs, consumed = _split(result, rc)
    consumed_bytes = b"".join(frames[:remainder_from])
    assert n_abs == absorbed
    assert after == b"".join(frames[remainder_from:]) + SCENARIOS[name][2]
    assert len(calls) == calls_n
    if rc:
        assert consumed == consumed_bytes          # EXACTLY the bytes removed from the file
        assert original == consumed + after
    warns = [m for lvl, m in records if lvl == "WARNING"]
    if reason:
        assert len(warns) == 1                      # ONE warning when the hold engages
        assert "reason=%s" % reason in warns[0] and "exc_type=%s" % exc_type in warns[0]
        # hardcoded text: no str(exc), entry text or path
        assert SECRET_EXC not in warns[0] and ENTRY_MARK not in warns[0]
        assert str(env.tmp) not in warns[0] and ".tract" not in warns[0]
    else:
        assert warns == []
    return records, wb, after, original


@pytest.mark.parametrize("rc", [False, True])
def test_hold_all_success_consumes_everything(env, rc):
    """FAILS on base: TypeError (unexpected keyword hold_on_failure)."""
    _, _, after, _ = _expect_hold(env, "all_success", rc, absorbed=3, remainder_from=3, calls_n=3)
    assert after == b""


@pytest.mark.parametrize("rc", [False, True])
def test_hold_filter_skips_are_consumed(env, rc):
    """FAILS on base: TypeError. Wrong type/source/empty text are 'looked at, nothing to
    absorb' and advance the offset even in hold mode."""
    _expect_hold(env, "filter_skips_only", rc, absorbed=0, remainder_from=3, calls_n=0)


@pytest.mark.parametrize("rc", [False, True])
def test_hold_false_in_middle_keeps_entry_and_everything_after(env, rc):
    """FAILS on base: TypeError; and on base the default would also consume the failed entry."""
    _expect_hold(env, "fail_middle_false", rc, absorbed=1, remainder_from=1, calls_n=2,
                 reason="absorb_returned_false")


@pytest.mark.parametrize("rc", [False, True])
def test_hold_raising_entry_keeps_it_and_names_only_the_class(env, rc):
    """FAILS on base: TypeError. The exception text must not reach the WARNING."""
    _expect_hold(env, "fail_middle_raise", rc, absorbed=1, remainder_from=1, calls_n=2,
                 reason="absorb_raised", exc_type="RuntimeError")


@pytest.mark.parametrize("rc", [False, True])
def test_hold_failure_at_first_entry_leaves_file_untouched_and_not_rewritten(env, rc):
    """FAILS on base: TypeError. safe_offset == 0 takes the existing early return, so the
    file is not even reopened for writing."""
    _, wb, after, original = _expect_hold(env, "fail_first", rc, absorbed=0, remainder_from=0,
                                          calls_n=1, reason="absorb_returned_false")
    assert after == original and wb == []


@pytest.mark.parametrize("rc", [False, True])
def test_hold_failure_at_last_entry(env, rc):
    """FAILS on base: TypeError."""
    _expect_hold(env, "fail_last", rc, absorbed=2, remainder_from=2, calls_n=3,
                 reason="absorb_returned_false")


@pytest.mark.parametrize("rc", [False, True])
def test_hold_cap_with_failure_inside_the_cap(env, rc):
    """FAILS on base: TypeError. A held entry counts as attempted; the hold breaks the loop."""
    _expect_hold(env, "cap_fail_inside", rc, absorbed=1, remainder_from=1, calls_n=2,
                 reason="absorb_returned_false")


@pytest.mark.parametrize("rc", [False, True])
def test_hold_cap_reached_before_the_failure(env, rc):
    """FAILS on base: TypeError. The cap stops the loop first, so the failing entry is never
    attempted and there is no warning."""
    _expect_hold(env, "cap_fail_after", rc, absorbed=2, remainder_from=2, calls_n=2)


@pytest.mark.parametrize("rc", [False, True])
def test_hold_filter_skips_before_the_failure_are_consumed_after_it_are_kept(env, rc):
    """FAILS on base: TypeError."""
    _expect_hold(env, "skips_around_fail", rc, absorbed=1, remainder_from=3, calls_n=2,
                 reason="absorb_returned_false")


@pytest.mark.parametrize("rc", [False, True])
def test_hold_parse_failure_is_unchanged(env, rc):
    """FAILS on base: TypeError. The parse-failure handler is untouched: whole file left
    alone, the existing WARNING (not the hold warning), consumed == b''."""
    pre = ng_tract.TractReader(POISON)
    assert [type(x).__name__ for x in pre] == ["bytes"]  # reader precondition for this scenario
    result, after, original, calls, records, wb, frames = _run(cc_ng_organism, env, "parse_failure", rc, hold=True)
    n_abs, consumed = _split(result, rc)
    assert after == original and wb == []
    if rc:
        assert consumed == b""
    warns = [m for lvl, m in records if lvl == "WARNING"]
    assert len(warns) == 1 and "parse failed" in warns[0] and "hold" not in warns[0]
    assert n_abs == 1


@pytest.mark.parametrize("rc", [False, True])
def test_hold_empty_file(env, rc):
    """FAILS on base: TypeError."""
    _expect_hold(env, "empty_file", rc, absorbed=0, remainder_from=0, calls_n=0)


def test_held_entry_is_retried_next_call_and_holds_again_if_still_failing(env):
    """FAILS on base: TypeError. (E) a held entry is retried each cycle."""
    spec = [("ok", 1), ("false", 2), ("ok", 3)]
    frames = [_frame(env.tmp, k, n) for k, n in spec]
    path = _write(env, "retry", frames)
    drain = cc_ng_organism.drain_ingest_tract
    a1 = drain(object(), None, {}, tract_path=path, hold_on_failure=True)
    assert a1 == 1 and _read(path) == b"".join(frames[1:])
    a2 = drain(object(), None, {}, tract_path=path, hold_on_failure=True)   # still failing
    assert a2 == 0 and _read(path) == b"".join(frames[1:])
    # the cause is fixed: the held entry and the one after it now land, file drains
    env.mp.setattr(cc_ng_organism, "run_conversational_dual_pass", lambda *a: True)
    a3 = drain(object(), None, {}, tract_path=path, hold_on_failure=True)
    assert a3 == 2 and _read(path) == b""


# --------------------------------------------- default behaviour is unchanged
@pytest.mark.parametrize("rc", [False, True])
def test_default_still_consumes_failed_entries_and_warns_nothing(env, rc):
    """Characterisation (passes on base too): with the flag off a failed entry IS consumed
    and no WARNING is emitted -- the historical behaviour #794 documents."""
    result, after, original, calls, records, wb, frames = _run(cc_ng_organism, env, "fail_middle_false", rc)
    n_abs, consumed = _split(result, rc)
    assert n_abs == 2 and after == b"" and len(calls) == 3
    assert [m for lvl, m in records if lvl == "WARNING"] == []
    if rc:
        assert consumed == original


def _load_base_module(tmp, mp):
    try:
        src = subprocess.run(["git", "-C", _REPO, "show", "%s:cc_ng_organism.py" % _BASE_SHA],
                             check=True, capture_output=True).stdout
    except Exception as exc:  # the proof is required: do not skip silently
        pytest.fail("cannot obtain BASE cc_ng_organism.py (%s) for the byte-identity proof: %s"
                    % (_BASE_SHA[:8], type(exc).__name__))
    name = "cc_ng_organism_base_e4ebf982"
    p = tmp / (name + ".py")
    p.write_bytes(src)
    spec = importlib.util.spec_from_file_location(name, str(p))
    mod = importlib.util.module_from_spec(spec)
    # dataclasses resolve their own module through sys.modules at class-creation time
    mp.setitem(sys.modules, name, mod)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("rc", [False, True])
@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_default_is_byte_identical_to_the_base_module(env, name, rc):
    """BYTE-IDENTITY PROOF (a preservation guard: it passes on the base by construction).
    The BASE module (git show e4ebf982) and the new module are fed the same tract bytes with
    hold_on_failure UNSET; return values, resulting file bytes, dual-pass call sequence,
    file-write opens and every captured log record (level, message) must be equal."""
    base = _load_base_module(env.tmp, env.mp)
    assert os.path.realpath(base.__file__) != _ORG_FILE
    env.mp.setattr(base, "run_conversational_dual_pass", env.dual)
    env.mp.setattr(base, "cc_gateway_tract_path", lambda: (_ for _ in ()).throw(
        AssertionError("SAFETY: default tract path used by BASE module")))
    frames = _build(env, name)
    new = _run(cc_ng_organism, env, name, rc, frames=frames, tag="new")
    old = _run(base, env, name, rc, frames=frames, tag="base")
    (r_new, a_new, o_new, c_new, l_new, w_new, _), (r_old, a_old, o_old, c_old, l_old, w_old, _) = new, old
    assert r_new == r_old
    assert a_new == a_old
    assert c_new == c_old
    assert l_new == l_old
    assert [m for m in w_new] == [m for m in w_old]


# ------------------------------------------------- retry tract (MVP 2026-10-03, Josh)
def _drain(env, path, **kw):
    return cc_ng_organism.drain_ingest_tract(object(), None, {"last_forest_id": None},
                                             tract_path=path, return_consumed=True, **kw)


@pytest.mark.parametrize("kind", ["false", "raise"])
def test_retry_tract_moves_the_failed_entry_whole_and_the_pass_continues(env, kind):
    frames = [_frame(env.tmp, "ok", 1), _frame(env.tmp, kind, 2), _frame(env.tmp, "ok", 3)]
    path = _write(env, "retry_" + kind, frames)
    retry = _check_safe(env.tmp / ("retry_" + kind + ".retry.tract"))
    absorbed, consumed = _drain(env, path, hold_on_failure=True, retry_tract_path=retry)
    assert absorbed == 2                      # both good turns landed; nothing held behind the failure
    assert _read(path) == b""                 # the main tract is fully drained
    assert _read(retry) == frames[1]          # the failed turn, byte-for-byte, nothing else
    assert consumed == b"".join(frames)


def test_retry_tract_rotates_a_failing_retry_to_its_own_end(env):
    frames = [_frame(env.tmp, "false", 1), _frame(env.tmp, "ok", 2)]
    path = _write(env, "retry_rotate", frames)
    absorbed, _ = _drain(env, path, hold_on_failure=True, retry_tract_path=path)
    assert absorbed == 1
    assert _read(path) == frames[0]           # the failure went round to the end; still there, whole


def test_retry_append_failure_falls_back_to_the_hold(env):
    frames = [_frame(env.tmp, "ok", 1), _frame(env.tmp, "false", 2), _frame(env.tmp, "ok", 3)]
    path = _write(env, "retry_unwritable", frames)
    bad = str(env.tmp / "no-such-dir" / "retry.tract")
    absorbed, _ = _drain(env, path, hold_on_failure=True, retry_tract_path=bad)
    assert absorbed == 1
    assert _read(path) == frames[1] + frames[2]   # held exactly as without a retry tract


def test_defer_moves_an_oversize_turn_whole_unattempted_and_the_rest_flow(env):
    frames = [_frame(env.tmp, "ok", 1), _frame(env.tmp, "false", 2), _frame(env.tmp, "ok", 3)]
    path = _write(env, "defer", frames)
    deferred = _check_safe(env.tmp / "defer.deferred.tract")
    # threshold just under the entry text length ("FALSE <mark>-2" is longer than "OK <mark>-1")
    th = len("OK %s-1" % ENTRY_MARK)
    absorbed, _ = _drain(env, path, hold_on_failure=True, defer_tract_path=deferred, defer_over_bytes=th)
    assert absorbed == 2
    assert _read(path) == b""
    assert _read(deferred) == frames[1]   # moved whole; never attempted (it would have failed)
