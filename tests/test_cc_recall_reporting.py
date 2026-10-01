# tests/test_cc_recall_reporting.py
#
# ---- Changelog ----
# [2026-09-30] Claude Code (Sonnet 5.5), Z12 worker seat, lane
#   daemon-recall-organism-756b (punchlist rows #756 / #779 / #780; slice B of the
#   #756 daemon recall-swallow chain) -- NEW file.
# What: tests for the optional recall-failure REPORTERS added to cc_ng_organism.py:
#   cc_assemble_recall(on_degraded=) reporting monitor_race,
#   pattern_completion_failed and pith_fallback; cc_pattern_completion_recall(
#   on_error=); render_constitutional_core(on_error=) / render_wants(on_error=).
#   Each reporting test states why it FAILS on the base (e4ebf982): the new
#   kwarg does not exist there, so the call raises TypeError --
#   test_base_rejects_every_new_kwarg proves that in the same run. A second
#   family proves the DEFAULT path (every new kwarg left unset) is byte-identical
#   to the base: the same inputs go through the base module (loaded from
#   `git show e4ebf982:cc_ng_organism.py` into a temp dir, never from the
#   worktree) and through the module under test, comparing return values,
#   captured log records, callback side effects and Pith metrics.
# Why: Josh P361/P370 (LAW 4: the origin must report a failed recall instead of
#   swallowing it); Chief-003 Decision 2 = YES with constraints (P381) and decision
#   B (pith_fallback in this slice); plan-001.md rev 1 sections 6 / 7 / 9 I6-I7.
#   The VPS half (cc_ng_host.py, Syl's process) imports this same file and passes
#   none of the new kwargs, so default-path byte-identity is the safety property.
# How: no pytest fixtures except the session-scoped P379/#770 preamble; every
#   fake is local; patching is unittest.mock.patch.object (auto-restored); log
#   capture is a local handler on the module's own logger (both the base copy and
#   the module under test use logging.getLogger("cc_ng_organism")).
#   P379 / Exec #770: neurograph_rpc.py:735-738 and cc-ng-daemon.py:540-543 put
#   the PRIMARY ~/NeuroGraph at sys.path[0], so a narrow worktree run can silently
#   test the primary's cc_ng_organism.py. This file prints cc_ng_organism.__file__
#   and the resolved path of every NG module in sys.modules at session start and
#   end, and aborts / errors the session if the module under test is not this
#   worktree's file or any NG module resolves into the primary checkout.
#   No graph is loaded, no daemon is started, no checkpoint or data/ path is read.
# -------------------
import contextlib
import hashlib
import importlib.util
import inspect
import logging
import os
import pwd
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if sys.path[:1] != [str(_ROOT)]:
    sys.path.insert(0, str(_ROOT))

import cc_ng_organism as _org  # noqa: E402  (must resolve to _ROOT; verified below)

_BASE_REV = 'e4ebf982b1989fd9066d610b94853bc68bf70d37'
_BASE_MODNAME = 'cc_ng_organism_base_e4ebf982'
_PRIMARY_NG = Path(pwd.getpwuid(os.getuid()).pw_dir) / 'NeuroGraph'
_NG_NAME_PREFIXES = ('ng_', 'cc_ng', 'cc_', 'neuro', 'openclaw', 'surface', 'ces_',
                     'stream_parser', 'surfacing', 'activation_persistence', 'universal_ingestor')


# ---------------------------------------------------------------- P379 / #770 preamble

def _under(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _ng_module_report():
    """(lines, problems): the resolved path of every NG-looking module in
    sys.modules, tagged by where it came from. A problem is: the module under
    test is not this worktree's file; or any module resolves into the PRIMARY
    ~/NeuroGraph checkout; or a name that exists at this worktree's root was
    loaded from somewhere else."""
    root_modules = {p.stem for p in _ROOT.glob('*.py')}
    primary = _PRIMARY_NG.resolve()
    lines, problems = [], []
    want = (_ROOT / 'cc_ng_organism.py').resolve()
    got = Path(getattr(sys.modules.get('cc_ng_organism'), '__file__', '') or '/nonexistent').resolve()
    if got != want:
        problems.append(f'cc_ng_organism resolves to {got}, not this worktree ({want})')
    if sys.modules.get('cc_ng_organism') is not _org:
        problems.append('sys.modules["cc_ng_organism"] is no longer the module this file imported')
    for name in sorted(sys.modules):
        file = getattr(sys.modules[name], '__file__', None)
        if not file:
            continue
        path = Path(file).resolve()
        in_wt, in_primary = _under(path, _ROOT), _under(path, primary)
        if not (in_wt or in_primary or name in root_modules or name.startswith(_NG_NAME_PREFIXES)):
            continue
        if name == _BASE_MODNAME:
            tag = f'BASE COPY (git show {_BASE_REV[:8]}, temp dir; used only for byte-identity)'
        elif in_wt:
            tag = 'worktree'
        elif in_primary:
            tag = 'PRIMARY CHECKOUT  <-- PROBLEM'
            problems.append(f'{name} resolves into the primary checkout: {path}')
        elif name in root_modules:
            tag = 'OUTSIDE WORKTREE  <-- PROBLEM'
            problems.append(f'{name} exists at this worktree root but was loaded from {path}')
        else:
            tag = 'external (not an NG-tree file)'
        lines.append(f'    {name:34s} {path}  [{tag}]')
    return lines, problems


def _preamble(stage, config=None):
    lines, problems = _ng_module_report()
    text = '\n'.join(
        [f'[P379/#770 preamble, {stage}] worktree root: {_ROOT}',
         f'  cc_ng_organism.__file__ = {_org.__file__}',
         f'  primary NG checkout (must NOT be used): {_PRIMARY_NG}',
         '  NG-looking modules in sys.modules:'] + lines
        + [f'  RESULT: {"FAIL -- " + "; ".join(problems) if problems else "PASS -- module under test is this worktree\'s file; no NG module from the primary checkout"}'])
    capman = config.pluginmanager.getplugin('capturemanager') if config is not None else None
    if capman is not None and hasattr(capman, 'global_and_fixture_disabled'):
        with capman.global_and_fixture_disabled():
            print(text, file=sys.__stderr__, flush=True)
    else:
        print(text, file=sys.__stderr__, flush=True)
    return problems


# ---------------------------------------------------------------- the pinned BASE module

_BASE_DIR = None


def _load_base():
    """The base module, from the git blob at e4ebf982 (never the worktree
    file), exec'd under its own module name from a temp dir. The blob hash is
    re-derived from the bytes and compared with git's own, so a wrong or
    truncated `git show` cannot pass silently."""
    global _BASE_DIR
    git = ['git', '-C', str(_ROOT)]
    blob = subprocess.run(git + ['show', f'{_BASE_REV}:cc_ng_organism.py'],
                          check=True, capture_output=True).stdout
    want = subprocess.run(git + ['rev-parse', f'{_BASE_REV}:cc_ng_organism.py'],
                          check=True, capture_output=True, text=True).stdout.strip()
    got = hashlib.sha1(b'blob %d\0' % len(blob) + blob).hexdigest()
    assert got == want, f'base blob mismatch: {got} != {want}'
    _BASE_DIR = Path(tempfile.mkdtemp(prefix='cc_ng_organism_base_'))
    path = _BASE_DIR / 'cc_ng_organism.py'
    path.write_bytes(blob)
    spec = importlib.util.spec_from_file_location(_BASE_MODNAME, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[_BASE_MODNAME] = module  # @dataclass resolves cls.__module__ through sys.modules
    spec.loader.exec_module(module)
    return module


_base = _load_base()

_startup_problems = _preamble('collection')
if _startup_problems:  # a collection error aborts the whole pytest session
    raise RuntimeError('P379/#770: module under test is not from this worktree: '
                       + '; '.join(_startup_problems))


@pytest.fixture(scope='session', autouse=True)
def _p379_worktree_preamble(request):
    problems = _preamble('session start', request.config)
    if problems:
        pytest.exit('P379/#770: ' + '; '.join(problems), returncode=3)
    yield
    # Re-check at the very end: the unification tests load the daemon, whose
    # module scope inserts ~/NeuroGraph at sys.path[0] (cc-ng-daemon.py:540-543);
    # this shows which organism and which sibling modules the WHOLE run used.
    problems = _preamble('session end (after every collected test, including the unification file)',
                         request.config)
    if _BASE_DIR is not None:
        shutil.rmtree(_BASE_DIR, ignore_errors=True)
    sys.modules.pop(_BASE_MODNAME, None)
    if problems:
        pytest.fail('P379/#770: ' + '; '.join(problems), pytrace=False)


# ---------------------------------------------------------------- local fakes

class _Records(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append((record.levelname, record.getMessage()))


@contextlib.contextmanager
def _captured(mod):
    """Capture every record `mod` logs, as (level, message). The base copy and
    the module under test share one logger object (getLogger("cc_ng_organism"))."""
    lg = mod.logger
    handler = _Records()
    old_level = lg.level
    lg.addHandler(handler)
    lg.setLevel(logging.DEBUG)
    try:
        yield handler.records
    finally:
        lg.removeHandler(handler)
        lg.setLevel(old_level)


class _Reports:
    """A reporter that records its calls."""

    def __init__(self):
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)


def _raiser(make_exc):
    def _f(*args, **kwargs):
        raise make_exc()
    return _f


def _boom(*args, **kwargs):
    raise RuntimeError('INJECTED_PITH_FAILURE')


class _Monitor:
    def __init__(self, items=(), make_exc=None):
        self._items = list(items)
        self._make_exc = make_exc

    def get_surfaced(self):
        if self._make_exc is not None:
            raise self._make_exc()
        return list(self._items)

    def format_context(self, items):
        if not items:
            return ''
        return '## Recent\n' + '\n'.join(f"- {it['content']}" for it in items)


class _Graph:
    def __init__(self, nodes=None):
        self.nodes = {} if nodes is None else nodes
        self.config = {}

    def _is_identity_protected(self, node_id):
        return False


class _Ng:
    def __init__(self, monitor=None, harvest=None):
        self.graph = _Graph()
        self._surfacing_monitor = monitor
        self._harvest = harvest

    def _harvest_associations(self, query, **kwargs):
        return self._harvest(query, **kwargs)


class _Node:
    def __init__(self, metadata, creation_time=0.0):
        self.metadata = metadata
        self.creation_time = creation_time


class _ExplodingGraph:
    @property
    def nodes(self):
        raise RuntimeError('INJECTED_NODES')


_MONITOR_ITEM = {'node_id': 'm1', 'score': 1.0, 'content': 'monitor hit'}
_PATTERN_ITEM = {'node_id': 'p1', 'score': 0.9, 'content': 'pattern hit'}
_MONITOR_TEXT = '## Recent\n- monitor hit'
# Same sentences the existing unification test feeds the REAL Pith pipeline
# (gate on, no failure injected): short strings could be filtered by stage 1.
_LONG_MONITOR_ITEM = {'node_id': 'recent', 'score': 1.0, 'content': 'a genuinely distinct monitor hit'}
_LONG_PATTERN_ITEM = {'node_id': 'novel', 'score': 0.8,
                      'content': 'a genuinely distinct pattern-completion hit'}


def _pattern_text(mod):
    return mod._format_cc_recall_block([_PATTERN_ITEM])


def _strict_pc(results):
    """Pattern-completion stub that accepts EXACTLY the pre-change call shape
    (ng, query, k, state=) -- an accidental extra kwarg on the default path
    raises TypeError here, which cc_assemble_recall swallows into an empty
    block, so the byte-identity comparison catches it."""
    return lambda ng, query, k, state=None: list(results)


def _loose_pc(results):
    return lambda ng, query, k, state=None, **kwargs: list(results)


def _wants_graph():
    return _Graph({'w1': _Node({'kind': 'want', 'provenance': 'cc_authored',
                                'want_text': 'learn the substrate'}, creation_time=5.0)})


def _core_graph():
    return _Graph({'c1': _Node({'constitutional': True, 'core_text': 'I am CC.', 'spine_order': 1})})


# ---------------------------------------------------------------- 1. cc_pattern_completion_recall(on_error=)

def test_pattern_completion_on_error_reports_the_swallowed_exception_and_still_returns_empty():
    # FAILS on base: TypeError, cc_pattern_completion_recall has no `on_error`.
    reports = _Reports()
    ng = _Ng(harvest=_raiser(lambda: RuntimeError('INJECTED_HARVEST')))
    with _captured(_org) as logs:
        out = _org.cc_pattern_completion_recall(ng, 'q', on_error=reports)
    assert out == []
    assert len(reports.calls) == 1 and len(reports.calls[0]) == 1
    exc = reports.calls[0][0]
    assert isinstance(exc, RuntimeError) and str(exc) == 'INJECTED_HARVEST'
    # the existing debug line is still written, once, unchanged
    assert logs == [('DEBUG', 'cc_pattern_completion_recall failed (non-fatal): INJECTED_HARVEST')]


def test_pattern_completion_raising_reporter_does_not_change_the_failsoft_return():
    # FAILS on base: TypeError (no `on_error`).
    def bad_reporter(exc):
        raise OSError('reporter blew up')

    ng = _Ng(harvest=_raiser(lambda: RuntimeError('INJECTED_HARVEST')))
    with _captured(_org) as logs:
        out = _org.cc_pattern_completion_recall(ng, 'q', on_error=bad_reporter)
    assert out == []
    assert logs == [('DEBUG', 'cc_pattern_completion_recall failed (non-fatal): INJECTED_HARVEST')]


def test_pattern_completion_does_not_report_a_legitimate_empty():
    # FAILS on base: TypeError (no `on_error`). An empty query / no graph is
    # "nothing to recall", not a failure: the reporter must stay silent.
    reports = _Reports()
    assert _org.cc_pattern_completion_recall(_Ng(), '', on_error=reports) == []
    assert _org.cc_pattern_completion_recall(None, 'q', on_error=reports) == []
    assert reports.calls == []


# ---------------------------------------------------------------- 2. render_*(on_error=)

def test_render_constitutional_core_distinguishes_nothing_to_render_from_raised():
    # FAILS on base: TypeError (no `on_error`); on base "" is ambiguous.
    raised, empty, populated = _Reports(), _Reports(), _Reports()
    with _captured(_org) as logs:
        out_raised = _org.render_constitutional_core(_ExplodingGraph(), on_error=raised)
    assert out_raised == ''
    assert len(raised.calls) == 1
    exc = raised.calls[0][0]
    assert isinstance(exc, RuntimeError) and str(exc) == 'INJECTED_NODES'
    assert logs == [('DEBUG', 'CC constitutional-core render error (non-fatal): INJECTED_NODES')]

    assert _org.render_constitutional_core(_Graph(), on_error=empty) == ''
    assert empty.calls == []  # same "" return, NOT reported
    assert _org.render_constitutional_core(_core_graph(), on_error=populated) == '## Who I Am\n- I am CC.'
    assert populated.calls == []


def test_render_wants_distinguishes_nothing_to_render_from_raised():
    # FAILS on base: TypeError (no `on_error`).
    raised, empty, none_graph, populated = _Reports(), _Reports(), _Reports(), _Reports()
    with _captured(_org) as logs:
        out_raised = _org.render_wants(_ExplodingGraph(), on_error=raised)
    assert out_raised == ''
    assert len(raised.calls) == 1
    exc = raised.calls[0][0]
    assert isinstance(exc, RuntimeError) and str(exc) == 'INJECTED_NODES'
    assert logs == [('DEBUG', 'CC want-render error (non-fatal): INJECTED_NODES')]

    assert _org.render_wants(_Graph(), on_error=empty) == ''
    assert empty.calls == []
    assert _org.render_wants(None, on_error=none_graph) == ''
    assert none_graph.calls == []
    assert _org.render_wants(_wants_graph(), on_error=populated) == '## What I Want\n- learn the substrate'
    assert populated.calls == []


def test_render_on_error_raising_reporter_does_not_change_the_return():
    # FAILS on base: TypeError (no `on_error`).
    def bad_reporter(exc):
        raise OSError('reporter blew up')

    assert _org.render_constitutional_core(_ExplodingGraph(), on_error=bad_reporter) == ''
    assert _org.render_wants(_ExplodingGraph(), on_error=bad_reporter) == ''
    assert _org.render_constitutional_core(_core_graph(), on_error=bad_reporter) == '## Who I Am\n- I am CC.'


def test_render_on_error_positional_provenance_still_binds_first():
    # render_wants(graph, provenance) is called positionally in the wild; the new
    # kwarg is appended AFTER provenance. FAILS on base: TypeError (no `on_error`).
    assert _org.render_wants(_wants_graph(), 'cc_authored', on_error=_Reports()) == \
        '## What I Want\n- learn the substrate'
    assert _org.render_wants(_wants_graph(), 'cc_emergent', on_error=_Reports()) == ''


# ---------------------------------------------------------------- 3. cc_assemble_recall(on_degraded=)

def test_assemble_reports_monitor_race():
    # FAILS on base: TypeError (no `on_degraded`); on base the RuntimeError has
    # no log, no callback, no counter.
    degraded = _Reports()
    ng = _Ng(monitor=_Monitor(make_exc=lambda: RuntimeError('INJECTED_RACE')))
    with mock.patch.object(_org, '_CC_PITH_ENABLED', False), \
            mock.patch.object(_org, 'cc_pattern_completion_recall', _loose_pc([_PATTERN_ITEM])):
        out = _org.cc_assemble_recall(ng, 'q', 5, {}, None, on_degraded=degraded)
    assert out == _pattern_text(_org)  # the monitor block is the only thing missing
    assert [(c[0], type(c[1]).__name__, str(c[1])) for c in degraded.calls] == \
        [('monitor_race', 'RuntimeError', 'INJECTED_RACE')]


def test_assemble_reports_pattern_completion_failed_from_the_swallow_below():
    # FAILS on base: TypeError (no `on_degraded`). Drives the REAL
    # cc_pattern_completion_recall, whose own `except` (not the one in
    # cc_assemble_recall) is where Active Recall actually dies.
    degraded = _Reports()
    ng = _Ng(monitor=_Monitor([_MONITOR_ITEM]),
             harvest=_raiser(lambda: RuntimeError('INJECTED_HARVEST')))
    with mock.patch.object(_org, '_CC_PITH_ENABLED', False), _captured(_org) as logs:
        out = _org.cc_assemble_recall(ng, 'q', 5, None, None, on_degraded=degraded)
    assert out == _MONITOR_TEXT  # Active Recall block silently absent, monitor block intact
    assert [(c[0], type(c[1]).__name__, str(c[1])) for c in degraded.calls] == \
        [('pattern_completion_failed', 'RuntimeError', 'INJECTED_HARVEST')]
    assert logs == [('DEBUG', 'cc_pattern_completion_recall failed (non-fatal): INJECTED_HARVEST')]


def test_assemble_reports_pattern_completion_failed_from_the_outer_except():
    # FAILS on base: TypeError (no `on_degraded`).
    degraded = _Reports()
    ng = _Ng(monitor=_Monitor([_MONITOR_ITEM]))
    with mock.patch.object(_org, '_CC_PITH_ENABLED', False), \
            mock.patch.object(_org, 'cc_pattern_completion_recall',
                              _raiser(lambda: ValueError('INJECTED_PC'))), \
            _captured(_org) as logs:
        out = _org.cc_assemble_recall(ng, 'q', 5, {}, None, on_degraded=degraded)
    assert out == _MONITOR_TEXT
    assert [(c[0], type(c[1]).__name__, str(c[1])) for c in degraded.calls] == \
        [('pattern_completion_failed', 'ValueError', 'INJECTED_PC')]
    assert logs == [('DEBUG', 'Pattern-completion recall failed (non-fatal): INJECTED_PC')]


_PITH_FAILURE_POINTS = {
    'CacheLine build': lambda org: mock.patch.object(org.CacheLine, 'from_surfaced', classmethod(_boom)),
    'victim_recover': lambda org: mock.patch.object(org, 'pith_victim_recover', _boom),
    'stage1': lambda org: mock.patch.object(org, 'pith_stage1', _boom),
    'stage3': lambda org: mock.patch.object(org, 'pith_stage3', _boom),
}


def test_assemble_reports_pith_fallback_and_still_calls_on_pith_failure_first():
    # FAILS on base: TypeError (no `on_degraded`). Amendment (Chief-003 decision B):
    # the Pith fallback is reported here, at the spot where on_pith_failure is
    # already called, so the daemon can pass cc_deposit_pith_failure UNCHANGED.
    for stage, patcher in _PITH_FAILURE_POINTS.items():
        order = []

        def on_pith(exc, order=order):
            order.append(('on_pith_failure', exc))

        def on_deg(code, exc, order=order):
            order.append((code, exc))

        ng = _Ng(monitor=_Monitor([_MONITOR_ITEM]))
        with mock.patch.object(_org, '_CC_PITH_ENABLED', True), \
                mock.patch.object(_org, '_last_pith_warn_ts', 0.0), \
                patcher(_org), \
                mock.patch.object(_org, 'cc_pattern_completion_recall', _loose_pc([_PATTERN_ITEM])), \
                _captured(_org) as logs:
            out = _org.cc_assemble_recall(ng, 'q', 5, {}, None,
                                          on_pith_failure=on_pith, on_degraded=on_deg)
        assert [o[0] for o in order] == ['on_pith_failure', 'pith_fallback'], stage
        assert order[0][1] is order[1][1], stage  # the same raw exception object
        assert isinstance(order[0][1], RuntimeError) and str(order[0][1]) == 'INJECTED_PITH_FAILURE', stage
        assert out == _MONITOR_TEXT + '\n\n' + _pattern_text(_org), stage  # un-Pithed fallback text
        assert any(lvl == 'WARNING' and 'falling back to un-Pithed rendering' in msg
                   for lvl, msg in logs), stage  # the existing rate-limited WARNING is still there


def test_assemble_pith_fallback_is_reported_even_without_on_pith_failure():
    # FAILS on base: TypeError (no `on_degraded`).
    degraded = _Reports()
    ng = _Ng(monitor=_Monitor([_MONITOR_ITEM]))
    with mock.patch.object(_org, '_CC_PITH_ENABLED', True), \
            mock.patch.object(_org, '_last_pith_warn_ts', 0.0), \
            mock.patch.object(_org, 'pith_stage1', _boom), \
            mock.patch.object(_org, 'cc_pattern_completion_recall', _loose_pc([])):
        out = _org.cc_assemble_recall(ng, 'q', 5, {}, None, on_degraded=degraded)
    assert out == _MONITOR_TEXT
    assert [c[0] for c in degraded.calls] == ['pith_fallback']


def test_assemble_all_three_codes_in_one_request_arrive_in_pipeline_order():
    # FAILS on base: TypeError (no `on_degraded`). The order is the contract the
    # daemon's precedence table sits on: monitor, then Active Recall, then Pith.
    degraded = _Reports()
    ng = _Ng(monitor=_Monitor(make_exc=lambda: RuntimeError('INJECTED_RACE')))
    with mock.patch.object(_org, '_CC_PITH_ENABLED', True), \
            mock.patch.object(_org, '_last_pith_warn_ts', 0.0), \
            mock.patch.object(_org, 'pith_stage1', _boom), \
            mock.patch.object(_org, 'cc_pattern_completion_recall',
                              _raiser(lambda: ValueError('INJECTED_PC'))):
        out = _org.cc_assemble_recall(ng, 'q', 5, {}, None, on_degraded=degraded)
    assert out == ''
    assert [c[0] for c in degraded.calls] == ['monitor_race', 'pattern_completion_failed', 'pith_fallback']


def test_assemble_raising_on_degraded_changes_nothing():
    # FAILS on base: TypeError (no `on_degraded`). The raising reporter must not
    # change the returned text, the logs, or the existing on_pith_failure call.
    def run(**extra):
        recorded = []
        ng = _Ng(monitor=_Monitor(make_exc=lambda: RuntimeError('INJECTED_RACE')))
        with mock.patch.object(_org, '_CC_PITH_ENABLED', True), \
                mock.patch.object(_org, '_last_pith_warn_ts', 0.0), \
                mock.patch.object(_org, 'pith_stage1', _boom), \
                mock.patch.object(_org, 'cc_pattern_completion_recall',
                                  _raiser(lambda: ValueError('INJECTED_PC'))), \
                _captured(_org) as logs:
            out = _org.cc_assemble_recall(ng, 'q', 5, {}, None,
                                          on_pith_failure=lambda exc: recorded.append(
                                              (type(exc).__name__, str(exc))),
                                          **extra)
        return out, list(logs), recorded

    def bad_reporter(code, exc):
        raise OSError('reporter blew up')

    without = run()
    with_bad = run(on_degraded=bad_reporter)
    assert with_bad == without
    assert without[2] == [('RuntimeError', 'INJECTED_PITH_FAILURE')]  # on_pith_failure ran, once


def test_assemble_clean_runs_report_nothing():
    # FAILS on base: TypeError (no `on_degraded`). No false positives: a healthy
    # request (gate off, and gate on through the real Pith pipeline) reports nothing.
    for gate in (False, True):
        degraded = _Reports()
        ng = _Ng(monitor=_Monitor([_LONG_MONITOR_ITEM]))
        with mock.patch.object(_org, '_CC_PITH_ENABLED', gate), \
                mock.patch.object(_org, 'cc_pattern_completion_recall', _loose_pc([_LONG_PATTERN_ITEM])):
            out = _org.cc_assemble_recall(ng, 'query text', 5, {}, None, on_degraded=degraded)
        assert 'distinct monitor hit' in out and 'distinct pattern-completion hit' in out, gate  # non-vacuous
        assert degraded.calls == [], gate


def test_assemble_pattern_completion_call_shape_is_unchanged_unless_on_degraded_is_set():
    # FAILS on base for the second half (no `on_degraded`). The first half is the
    # structural reason the default path is byte-identical: with the reporter
    # unset the pattern-completion call carries no new kwarg at all.
    seen = []

    def spy(ng, query, k, state=None, **kwargs):
        seen.append((query, k, state, dict(kwargs)))
        return []

    conv = {'marker': 1}
    with mock.patch.object(_org, '_CC_PITH_ENABLED', False), \
            mock.patch.object(_org, 'cc_pattern_completion_recall', spy):
        _org.cc_assemble_recall(_Ng(), 'q', 5, conv, None)
        _org.cc_assemble_recall(_Ng(), 'q', 5, conv, None, on_degraded=_Reports())
    assert seen[0] == ('q', 5, conv, {})
    assert seen[1][:3] == ('q', 5, conv) and set(seen[1][3]) == {'on_error'} and callable(seen[1][3]['on_error'])


# ---------------------------------------------------------------- 4. signatures

def _params(fn):
    return list(inspect.signature(fn).parameters.values())


def test_new_kwargs_are_appended_default_none_and_nothing_else_changed():
    # FAILS on base: the new parameter is absent there.
    cases = [
        ('cc_assemble_recall', 'on_degraded'),
        ('cc_pattern_completion_recall', 'on_error'),
        ('render_constitutional_core', 'on_error'),
        ('render_wants', 'on_error'),
    ]
    for name, new in cases:
        old_params, new_params = _params(getattr(_base, name)), _params(getattr(_org, name))
        assert new not in [p.name for p in old_params], name
        assert [(p.name, p.kind, p.default) for p in new_params[:-1]] == \
            [(p.name, p.kind, p.default) for p in old_params], name  # every existing param untouched, in order
        last = new_params[-1]
        assert (last.name, last.kind, last.default) == (new, inspect.Parameter.POSITIONAL_OR_KEYWORD, None), name


def test_base_rejects_every_new_kwarg():
    # The in-run proof that every reporting test above FAILS on the base: the
    # base module (git blob e4ebf982) raises TypeError for each new kwarg.
    ng = _Ng(monitor=_Monitor())
    calls = [
        ('on_degraded', lambda: _base.cc_assemble_recall(ng, 'q', 5, {}, None, on_degraded=_Reports())),
        ('on_error', lambda: _base.cc_pattern_completion_recall(ng, 'q', on_error=_Reports())),
        ('on_error', lambda: _base.render_constitutional_core(_Graph(), on_error=_Reports())),
        ('on_error', lambda: _base.render_wants(_Graph(), on_error=_Reports())),
    ]
    for kwarg, call in calls:
        with pytest.raises(TypeError, match=f"unexpected keyword argument '{kwarg}'"):
            call()


# ---------------------------------------------------------------- 5. default path == base, byte for byte

def _scenarios():
    """name -> fn(mod) -> (value, side_effects). Every call is made EXACTLY as
    the pre-change callers make it: no new kwarg. Fakes are rebuilt per call so
    the base and the module under test never share state."""
    S = {}

    def scenario(fn):
        S[fn.__name__] = fn
        return fn

    # -- cc_pattern_completion_recall
    @scenario
    def pc_empty_query(mod):
        return mod.cc_pattern_completion_recall(_Ng(), ''), None

    @scenario
    def pc_ng_none(mod):
        return mod.cc_pattern_completion_recall(None, 'q'), None

    @scenario
    def pc_harvest_raises(mod):
        ng = _Ng(harvest=_raiser(lambda: RuntimeError('INJECTED_HARVEST')))
        return mod.cc_pattern_completion_recall(ng, 'q'), None

    @scenario
    def pc_graph_without_config(mod):
        ng = _Ng()
        ng.graph = object()
        return mod.cc_pattern_completion_recall(ng, 'q'), None

    # -- render_wants / render_constitutional_core
    @scenario
    def wants_none_graph(mod):
        return mod.render_wants(None), None

    @scenario
    def wants_empty(mod):
        return mod.render_wants(_Graph()), None

    @scenario
    def wants_populated(mod):
        return mod.render_wants(_wants_graph()), None

    @scenario
    def wants_positional_provenance(mod):
        return mod.render_wants(_wants_graph(), 'cc_emergent'), None

    @scenario
    def wants_exploding_nodes(mod):
        return mod.render_wants(_ExplodingGraph()), None

    @scenario
    def wants_bad_creation_time_raises_midway(mod):
        g = _Graph({'w1': _Node({'kind': 'want', 'provenance': 'cc_authored', 'want_text': 'x'},
                                creation_time='not-a-number')})
        return mod.render_wants(g), None

    @scenario
    def core_empty(mod):
        return mod.render_constitutional_core(_Graph()), None

    @scenario
    def core_populated(mod):
        return mod.render_constitutional_core(_core_graph()), None

    @scenario
    def core_exploding_nodes(mod):
        return mod.render_constitutional_core(_ExplodingGraph()), None

    @scenario
    def core_unsortable_spine_order_raises_midway(mod):
        g = _Graph({'a': _Node({'constitutional': True, 'core_text': 'a', 'spine_order': 1}),
                    'b': _Node({'constitutional': True, 'core_text': 'b', 'spine_order': 'x'})})
        return mod.render_constitutional_core(g), None

    # -- cc_assemble_recall (strict pc stub: the default call shape is part of what is compared)
    def assemble(mod, ng, *, gate=False, pc=(_PATTERN_ITEM,), conv=None, extra=None, patches=(),
                 allow_pc=True, strict=True):
        got = []
        kwargs = dict(extra or {})
        for key in ('on_monitor_error', 'on_pith_failure'):
            if key in kwargs:
                cb = kwargs[key]
                kwargs[key] = (lambda exc, cb=cb, key=key:
                               (got.append((key, type(exc).__name__, str(exc))), cb(exc)))
        metrics_before = mod._PITH_METRICS.pith_failures
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(mod, '_CC_PITH_ENABLED', gate))
            stack.enter_context(mock.patch.object(mod, '_last_pith_warn_ts', 0.0))
            if pc is not None:
                stack.enter_context(mock.patch.object(
                    mod, 'cc_pattern_completion_recall', (_strict_pc if strict else _loose_pc)(pc)))
            for p in patches:
                stack.enter_context(p(mod))
            out = mod.cc_assemble_recall(ng, 'query text', 5, conv, None,
                                         allow_pattern_completion=allow_pc, **kwargs)
        return out, {'callbacks': got, 'pith_failures_delta': mod._PITH_METRICS.pith_failures - metrics_before}

    @scenario
    def assemble_clean_gate_off(mod):
        return assemble(mod, _Ng(monitor=_Monitor([_MONITOR_ITEM])))

    @scenario
    def assemble_allow_pattern_completion_false(mod):
        return assemble(mod, _Ng(monitor=_Monitor([_MONITOR_ITEM])), allow_pc=False)

    @scenario
    def assemble_monitor_runtimeerror_default(mod):
        return assemble(mod, _Ng(monitor=_Monitor(make_exc=lambda: RuntimeError('INJECTED_RACE'))))

    @scenario
    def assemble_monitor_valueerror_with_on_monitor_error(mod):
        return assemble(mod, _Ng(monitor=_Monitor(make_exc=lambda: ValueError('INJECTED_MON'))),
                        extra={'on_monitor_error': lambda exc: None})

    @scenario
    def assemble_monitor_valueerror_raising_on_monitor_error(mod):
        def bad(exc):
            raise OSError('cb blew up')
        return assemble(mod, _Ng(monitor=_Monitor(make_exc=lambda: ValueError('INJECTED_MON'))),
                        extra={'on_monitor_error': bad})

    @scenario
    def assemble_pc_stub_raises(mod):
        ng = _Ng(monitor=_Monitor([_MONITOR_ITEM]))
        return assemble(mod, ng, pc=None, patches=(
            lambda m: mock.patch.object(m, 'cc_pattern_completion_recall',
                                        _raiser(lambda: ValueError('INJECTED_PC'))),))

    @scenario
    def assemble_real_pattern_completion_swallow(mod):
        ng = _Ng(monitor=_Monitor([_MONITOR_ITEM]),
                 harvest=_raiser(lambda: RuntimeError('INJECTED_HARVEST')))
        return assemble(mod, ng, pc=None)

    for stage, patcher in _PITH_FAILURE_POINTS.items():
        def make(stage=stage, patcher=patcher):
            def sc(mod):
                return assemble(mod, _Ng(monitor=_Monitor([_MONITOR_ITEM])), gate=True, conv={},
                                patches=(patcher,), extra={'on_pith_failure': lambda exc: None})
            sc.__name__ = 'assemble_pith_fallback_' + stage.replace(' ', '_')
            return sc
        scenario(make())

    @scenario
    def assemble_pith_fallback_without_on_pith_failure(mod):
        return assemble(mod, _Ng(monitor=_Monitor([_MONITOR_ITEM])), gate=True, conv={},
                        patches=(_PITH_FAILURE_POINTS['stage1'],))

    @scenario
    def assemble_pith_fallback_raising_on_pith_failure(mod):
        def bad(exc):
            raise OSError('tract unwritable')
        return assemble(mod, _Ng(monitor=_Monitor([_MONITOR_ITEM])), gate=True, conv={},
                        patches=(_PITH_FAILURE_POINTS['stage1'],), extra={'on_pith_failure': bad})

    @scenario
    def assemble_gate_on_real_pipeline_success(mod):
        return assemble(mod, _Ng(monitor=_Monitor([_LONG_MONITOR_ITEM])), gate=True, conv={},
                        pc=(_LONG_PATTERN_ITEM,))

    return S


def _observe(mod, fn):
    with _captured(mod) as logs:
        try:
            result = ('returned', fn(mod))
        except Exception as exc:  # an escaping exception is part of the contract too
            result = ('raised', type(exc).__name__, str(exc))
    return result, list(logs)


def test_default_path_is_byte_identical_to_the_base_module():
    # Passes on base by construction (it IS the base): the guard that the
    # worktree's module, called exactly as every existing caller calls it, returns
    # the same values, writes the same log records, drives the same callbacks and
    # bumps the same Pith metrics as the git blob at e4ebf982.
    scenarios = _scenarios()
    assert len(scenarios) == 28, sorted(scenarios)  # a silently dropped scenario must not shrink the proof
    mismatches = []
    for name, fn in scenarios.items():
        want = _observe(_base, fn)
        got = _observe(_org, fn)
        if got != want:
            mismatches.append(f'{name}:\n    base: {want!r}\n    new:  {got!r}')
    assert not mismatches, 'default path diverged from base:\n  ' + '\n  '.join(mismatches)


def test_default_path_scenarios_exercise_the_paths_they_claim():
    # Non-vacuity: '' == '' would also "match". Pin the shape of the base's own
    # behaviour for the paths that matter, so the identity test cannot pass on an
    # accidentally dead scenario.
    S = _scenarios()
    obs = {name: _observe(_base, fn) for name, fn in S.items()}

    def value(name):
        return obs[name][0][1][0]

    def logs(name):
        return obs[name][1]

    def effects(name):
        return obs[name][0][1][1]

    assert value('wants_populated') == '## What I Want\n- learn the substrate'
    assert value('core_populated') == '## Who I Am\n- I am CC.'
    assert value('wants_exploding_nodes') == '' and \
        logs('wants_exploding_nodes') == [('DEBUG', 'CC want-render error (non-fatal): INJECTED_NODES')]
    assert value('core_exploding_nodes') == '' and logs('core_exploding_nodes')[0][0] == 'DEBUG'
    assert logs('wants_bad_creation_time_raises_midway') and logs('core_unsortable_spine_order_raises_midway')
    assert logs('pc_harvest_raises') == [
        ('DEBUG', 'cc_pattern_completion_recall failed (non-fatal): INJECTED_HARVEST')]
    assert value('assemble_clean_gate_off') == _MONITOR_TEXT + '\n\n' + _pattern_text(_base)
    assert value('assemble_monitor_runtimeerror_default') == _pattern_text(_base)
    assert logs('assemble_monitor_runtimeerror_default') == []  # the base is SILENT here (the bug)
    assert effects('assemble_monitor_valueerror_with_on_monitor_error')['callbacks'] == \
        [('on_monitor_error', 'ValueError', 'INJECTED_MON')]
    assert value('assemble_real_pattern_completion_swallow') == _MONITOR_TEXT
    for stage in _PITH_FAILURE_POINTS:
        name = 'assemble_pith_fallback_' + stage.replace(' ', '_')
        assert value(name) == _MONITOR_TEXT + '\n\n' + _pattern_text(_base), name
        assert effects(name)['pith_failures_delta'] == 1, name
        assert [r for r in logs(name) if r[0] == 'WARNING'], name
    assert any('Pith failure deposit failed: tract unwritable' in m
               for _, m in logs('assemble_pith_fallback_raising_on_pith_failure'))
    ok_text = value('assemble_gate_on_real_pipeline_success')
    assert 'genuinely distinct' in ok_text and \
        effects('assemble_gate_on_real_pipeline_success')['pith_failures_delta'] == 0


# ---------------------------------------------------------------- 6. P379 guards

def test_module_under_test_is_this_worktrees_file_and_the_base_is_not():
    assert Path(_org.__file__).resolve() == (_ROOT / 'cc_ng_organism.py').resolve()
    assert sys.modules['cc_ng_organism'] is _org
    base_file = Path(_base.__file__).resolve()
    assert not _under(base_file, _ROOT) and not _under(base_file, _PRIMARY_NG.resolve())
    assert _base is not _org and _base.cc_assemble_recall is not _org.cc_assemble_recall
    lines, problems = _ng_module_report()
    assert problems == [], problems
