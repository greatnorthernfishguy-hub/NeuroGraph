# ---- Changelog ----
# [2026-10-01] Claude Sonnet 5.5 (Z12 builder, lane drain-pacing-d24, dispatch #12929) — D24 FOLD: the committed REAL-FUNCTION seam test
# What: the real docs-daemon _autosave_loop + the real drain_ingest_tract + the real cc_topology_merge._unbound_nodes + the
#   real cc_ng_organism._cc_callosum_consolidate on a real in-memory neuro_foundation.Graph, with ONLY the embedder and the dual
#   pass stubbed. Covers: the paced MULTI-batch run (including a failing entry and a foreign-producer entry), the #896
#   behaviours (a pre-existing unbound node blocks the steps AND the deposit; binding it releases both), the loud wedge of an
#   unbindable node, the signature pins of the two reused functions, and the NG half of the CC_NG_IDLE_STEPS table.
# Why: le-044 P-a / checker-035 note 12: the docs-repo daemon suite installs FAKE NG modules (its no-real-NG rule), so nothing in
#   the repo bound the daemon to the real reused functions, and the one real-loop test (tests/test_cc_deposit_step.py::
#   test_autosave_loop_drains_the_tract_under_concurrent_lock_643) hangs (#754 class). Exec Packet 482 / #896 (rule 1: the guard
#   covers the whole unbound population; rule 2: no unpaced deposits).
# How: it lives in the NG repo because it needs the real NG modules. The daemon file is loaded by path from the docs repo:
#   Z12_D24_DAEMON_UNDER_TEST (a path), default ~/docs/scripts/cc-ng-daemon.py. A daemon that does not define the D24 pacing
#   (_drain_consolidate) cannot be bound to anything, so the file SKIPS with the reason printed (NeuroGraph merges BEFORE the
#   daemon, so on a primary checkout whose daemon is still older this is the expected state, not a pass). P379/#770: the printed
#   preamble lists every module under test and FAILS if an NG module resolves outside this worktree.
# -------------------
"""Which tests FAIL against an EARLIER daemon (run with Z12_D24_DAEMON_UNDER_TEST=<copy of git show <commit>:scripts/cc-ng-daemon.py>):
against the pre-fold tip 55fb0e1f the #896 tests (a pre-existing unbound node does not block the steps or the deposit there:
arrival-scoped guard, no pre-check) and the loud-wedge test; against the pre-batch tip e2c8092b the C9 tests (an unbound skip used
to CLEAR pending, so the owed steps were never run). The signature pins, the table pin and the plain multi-batch run pass on every
daemon (they pin facts that did not change). Tests marked 'needs #905' run the REAL shared _cc_callosum_consolidate with its
per-slice guard and need a NeuroGraph tree that has #905 (see C8 above)."""
import ast
import importlib.util
import logging
import os
import sys
import threading
import types
import time
from pathlib import Path

import pytest

_WORKTREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_WORKTREE))

import cc_ng_organism as org  # noqa: E402
import cc_topology_merge  # noqa: E402
import ng_embed  # noqa: E402
import ng_tract  # noqa: E402
import neuro_foundation  # noqa: E402
from neuro_foundation import Graph  # noqa: E402

DAEMON_PATH = Path(os.environ.get('Z12_D24_DAEMON_UNDER_TEST')
                   or os.path.expanduser('~/docs/scripts/cc-ng-daemon.py')).resolve()

print('[P379/#770 preamble] worktree root          ->', _WORKTREE)
for _m in (org, cc_topology_merge, ng_embed, neuro_foundation):
    print('[P379/#770 preamble] %-20s ->' % _m.__name__, Path(_m.__file__).resolve())
print('[P379/#770 preamble] ng_tract (site, Rust)  ->', getattr(ng_tract, '__file__', ng_tract))
print('[P379/#770 preamble] daemon under test      ->', DAEMON_PATH)


def test_modules_under_test_resolve_inside_the_worktree():
    for mod in (org, cc_topology_merge, ng_embed, neuro_foundation):
        assert _WORKTREE in Path(mod.__file__).resolve().parents, f'{mod.__name__} resolves outside {_WORKTREE}'


# C8 (le-047 / checker-036): these tests used to SKIP -- green -- when the daemon path lacked the D24 pacing or was unreadable, so a
# mistyped Z12_D24_DAEMON_UNDER_TEST or a primary checkout whose daemon is still older made the real-loop tests vanish from a green
# suite. Now they FAIL LOUD with the printed reason. The ONE opt-out is explicit and OFF by default: Z12_D24_SKIP_UNMERGED=1, for the
# window in which NeuroGraph is merged BEFORE the daemon (and before #905). The suite stays meaningful in that window: the 7
# daemon-independent tests (the module-path preamble, both signature pins, the CC_NG_IDLE_STEPS table) still run and pass, and the
# opt-out SKIPS print which pieces are not merged yet. A mistyped explicit Z12_D24_DAEMON_UNDER_TEST path ALWAYS fails, opt-out or not.
SKIP_UNMERGED = os.environ.get('Z12_D24_SKIP_UNMERGED') == '1'
DAEMON_EXPLICIT = 'Z12_D24_DAEMON_UNDER_TEST' in os.environ


def _daemon_state():
    try:
        text = DAEMON_PATH.read_text()
    except OSError:
        return 'unreadable'
    return 'ok' if '_drain_consolidate' in text else 'no_d24'


def _fail_or_skip_unmerged(msg, always_fail=False):
    msg = msg + ' [set Z12_D24_DAEMON_UNDER_TEST to the D24 daemon file; Z12_D24_SKIP_UNMERGED=1 opts out ONLY while NeuroGraph is merged before the daemon/#905]'
    if SKIP_UNMERGED and not always_fail:
        pytest.skip(msg)
    pytest.fail(msg, pytrace=False)


def _require_daemon():
    state = _daemon_state()
    if state == 'unreadable':
        _fail_or_skip_unmerged('the daemon file %s is UNREADABLE (a mistyped path?)' % DAEMON_PATH, always_fail=DAEMON_EXPLICIT)
    if state == 'no_d24':
        _fail_or_skip_unmerged('the daemon at %s defines no D24 pacing (_drain_consolidate): there is nothing to bind' % DAEMON_PATH)


def organism_has_905():
    import inspect
    return {'guard', 'progress'} <= set(inspect.signature(org._cc_callosum_consolidate).parameters) \
        and hasattr(cc_topology_merge, 'whole_graph_guard')


def _require_905():
    if not organism_has_905():
        _fail_or_skip_unmerged('the NeuroGraph tree under test has no #905 (_cc_callosum_consolidate guard=/progress= and '
                               'cc_topology_merge.whole_graph_guard): the per-slice guard cannot be exercised against the real function')


# ---------------------------------------------------------------- real-function pins (no daemon load needed)

def test_signatures_of_the_two_reused_functions():
    import inspect
    assert list(inspect.signature(cc_topology_merge._unbound_nodes).parameters) == ['graph', 'node_ids']
    # The positional (graph, idle_steps) contract is fixed. #905 (Exec P490) adds KEYWORD-ONLY guard=/progress= to the same function
    # (so this must hold both before and after it merges); anything else is a signature drift the daemon does not know about.
    params = inspect.signature(org._cc_callosum_consolidate).parameters
    assert list(params)[:2] == ['graph', 'idle_steps']
    extra = {n: p for n, p in params.items() if n not in ('graph', 'idle_steps')}
    assert set(extra) <= {'guard', 'progress'} and all(p.kind is p.KEYWORD_ONLY for p in extra.values()), list(params)


@pytest.mark.parametrize('env_value,expected', [(None, 250), ('300', 300), ('0', 0), ('-5', 0), ('abc', ValueError)])
def test_merge_idle_steps_expression_is_what_the_table_documents(monkeypatch, env_value, expected):
    """The MERGE half of the documented CC_NG_IDLE_STEPS table (the daemon half is in the docs repo's
    scripts/tests/test_cc_ng_daemon_drain_pacing.py): unset -> 250, a positive int -> itself, <= 0 -> 0 (no consolidation),
    non-integer -> ValueError. The source line is pinned by AST so a change to it fails here, then the table is evaluated."""
    tree = ast.parse(Path(cc_topology_merge.__file__).read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'merge_cc_topology')
    assigns = [n for n in ast.walk(fn) if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == 'idle_steps' for t in n.targets)
        and 'CC_NG_IDLE_STEPS' in ast.unparse(n.value)]
    assert len(assigns) == 1
    expr = ast.unparse(assigns[0].value)
    assert expr == "max(0, int(os.environ.get('CC_NG_IDLE_STEPS', '250')))", expr
    if env_value is None:
        monkeypatch.delenv('CC_NG_IDLE_STEPS', raising=False)
    else:
        monkeypatch.setenv('CC_NG_IDLE_STEPS', env_value)
    if expected is ValueError:
        with pytest.raises(ValueError):
            eval(expr, {'os': os, 'max': max, 'int': int})
    else:
        assert eval(expr, {'os': os, 'max': max, 'int': int}) == expected


# ---------------------------------------------------------------- the real-loop rig

class _SpyRLock:
    """A real RLock that remembers whether the CALLING thread owns it, so 'the lock is not held at the consolidation'
    is observable. Delegates to threading.RLock."""

    def __init__(self):
        self._l = threading.RLock()

    def acquire(self, blocking=True, timeout=-1):
        return self._l.acquire(blocking, timeout)

    def release(self):
        self._l.release()

    def __enter__(self):
        self._l.acquire()
        return self

    def __exit__(self, *a):
        self._l.release()
        return False

    def _is_owned(self):
        return self._l._is_owned()


_FAKE_DAEMON_COUNT = [0]


@pytest.fixture
def rig(tmp_path, monkeypatch):
    _require_daemon()
    tract = tmp_path / 'turns.tract'
    monkeypatch.setenv('CC_NG_BATCH_SIZE', '25')
    monkeypatch.setenv('CC_NG_IDLE_STEPS', '250')
    monkeypatch.setenv('CC_GATEWAY_TRACT_PATH', str(tract))
    # ISOLATION: the daemon's held section calls the REAL trickle_gateway_conduit, which WRITES a conduit file into
    # CC_GATEWAY_CONDUIT_PATH (default ~/docs/ng_topology, the git-synced live Leg 1 conduit) whenever
    # CC_CALLOSUM_LEG1_ENABLED and MACHINE_ID are set -- as they are in the laptop's shell. Redirect the path into
    # tmp AND replace the writer with a recorder, so this test can never leave a file in a live conduit directory.
    monkeypatch.setenv('CC_GATEWAY_CONDUIT_PATH', str(tmp_path / 'conduit'))
    trickled = []
    monkeypatch.setattr(org, 'trickle_gateway_conduit', lambda data, *a, **k: trickled.append(data))
    monkeypatch.setattr(ng_embed, 'embed', lambda t: t)

    seen = []

    def fake_dual(graph, vdb, text, emb, state):
        label, n, mode = str(text).split('|')
        n = int(n)
        seen.append(label)
        if mode == 'fail':                       # a failing entry that deposited nothing
            return False
        ids = [f'cc:conv::{label}::{i}' for i in range(n)]
        for nid in ids:
            if nid not in graph.nodes:
                graph.create_node(node_id=nid, metadata={})
        if mode == 'fail_partial':               # nodes landed, then the dual pass raised: they stay unbound
            raise RuntimeError('boom')
        if mode == 'bound':
            for nid in ids[1:]:
                graph.create_synapse(ids[0], nid, weight=0.2)
        return True

    monkeypatch.setattr(org, 'run_conversational_dual_pass', fake_dual)
    for name in ('cc_update_probation', 'surface_wants', 'generate_emergent_want', 'persist_cc_commons'):
        monkeypatch.setattr(org, name, lambda *a, **k: None)

    drain_calls = []
    real_drain = org.drain_ingest_tract

    def spy_drain(*a, **k):
        drain_calls.append(dict(k))
        return real_drain(*a, **k)

    monkeypatch.setattr(org, 'drain_ingest_tract', spy_drain)
    owned_at_consolidation = []
    real_cons = org._cc_callosum_consolidate

    import functools

    @functools.wraps(real_cons)             # inspect.signature follows __wrapped__: the daemon feature-detects guard=/progress=
    def spy_cons(graph, n, *a, **kw):
        owned_at_consolidation.append(graph._concurrent_lock._is_owned())
        return real_cons(graph, n, *a, **kw)

    monkeypatch.setattr(org, '_cc_callosum_consolidate', spy_cons)

    _FAKE_DAEMON_COUNT[0] += 1
    spec = importlib.util.spec_from_file_location('d24_seam_daemon_%d' % _FAKE_DAEMON_COUNT[0], str(DAEMON_PATH))
    d = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = d
    spec.loader.exec_module(d)
    # the daemon inserts a NeuroGraph path at import; the modules under test must STILL be the worktree's
    assert _WORKTREE in Path(sys.modules['cc_ng_organism'].__file__).resolve().parents
    assert _WORKTREE in Path(sys.modules['cc_topology_merge'].__file__).resolve().parents

    g = Graph()
    g._concurrent_lock = _SpyRLock()
    saves = []
    d.STATE = types.SimpleNamespace(
        running=True, lock=threading.Lock(), stats={}, conv_state={'last_forest_id': None},
        ng=types.SimpleNamespace(graph=g, vector_db=None), pending_consolidation=None, pending_consolidation_batch=None)
    monkeypatch.setattr(d, '_guarded_save', lambda *a, **k: saves.append(g.timestep) or True)
    cyc = {'n': 0, 'max': 1}

    def sleep(_):
        if cyc['n'] >= cyc['max']:
            d.STATE.running = False
        cyc['n'] += 1

    monkeypatch.setattr(d, 'time', types.SimpleNamespace(sleep=sleep, time=time.time, monotonic=time.monotonic))

    def write(*turns, source='cc_gateway'):
        for label, n, mode in turns:
            ng_tract.deposit_experience(raw=f'{label}|{n}|{mode}'.encode(), source=source, tract_paths=[str(tract)])

    def remaining():
        data = tract.read_bytes() if tract.exists() else b''
        return [e.content for e in ng_tract.TractReader(data)] if data else []

    def run(cycles):
        cyc['n'], cyc['max'] = 0, cycles
        d.STATE.running = True
        d._autosave_loop()

    return types.SimpleNamespace(d=d, graph=g, tract=tract, write=write, remaining=remaining, run=run, seen=seen,
                                 drain_calls=drain_calls, owned=owned_at_consolidation, saves=saves, trickled=trickled,
                                 state=d.STATE)


# ---------------------------------------------------------------- paced multi-batch, real functions

def test_paced_multibatch_run_with_a_failing_and_a_foreign_entry(rig):
    # cc_gateway turns a, fail, b, c, d, e with a foreign-producer entry between a and fail
    rig.write(('a', 10, 'bound'))
    rig.write(('foreign', 5, 'bound'), source='other_producer')
    rig.write(('fail', 0, 'fail'), ('b', 10, 'bound'), ('c', 10, 'bound'), ('d', 10, 'bound'), ('e', 10, 'bound'))
    rig.run(3)
    # every cc_gateway turn was seen exactly once, in order; the foreign entry was consumed but never absorbed
    assert rig.seen == ['a', 'fail', 'b', 'c', 'd', 'e']
    assert rig.remaining() == []
    # batch 1 = a, (foreign skipped), fail, b, c (30 nodes >= 25) -> 250 steps; batch 2 = d, e (20 nodes) -> 250 steps; cycle 3 empty
    assert [c['batch_nodes'] for c in rig.drain_calls] == [25, 25, 25]
    assert rig.graph.timestep == 500
    assert rig.owned == [False, False]                   # the autosave thread held NOTHING at either consolidation
    assert rig.state.pending_consolidation is None
    assert len(rig.graph.nodes) == 50
    assert rig.saves == [0, 250, 500]                    # the checkpoint save ran every cycle
    assert [bool(b) for b in rig.trickled] == [True, True, False]   # the drain's consumed bytes flow to the Leg 1 trickle, unchanged
    assert not (rig.tract.parent / 'conduit').exists()   # and nothing was written to any conduit directory


# ---------------------------------------------------------------- #896: the whole population gates BOTH the steps and the deposit

def test_preexisting_unbound_node_blocks_steps_and_deposit_and_binding_it_releases_both(rig, caplog):
    rig.graph.create_node(node_id='cc:conv::old-orphan', metadata={})     # the pre-existing cohort stand-in (no synapse, no hyperedge)
    rig.write(('a', 10, 'bound'), ('b', 10, 'bound'))
    before = rig.tract.read_bytes()
    t0, n0 = rig.graph.timestep, len(rig.graph.nodes)
    with caplog.at_level(logging.DEBUG, logger=rig.d.logger.name):
        rig.run(3)
    # NO unpaced deposit: the tract is byte-identical, nothing was drained, the clock did not move
    assert rig.tract.read_bytes() == before
    assert rig.drain_calls == [] and rig.seen == []
    assert rig.graph.timestep == t0 and len(rig.graph.nodes) == n0 and rig.owned == []
    assert rig.saves == [0, 0, 0]                                         # ... but the checkpoint save ran every cycle
    blocked = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR
               and 'consolidation_skipped_unbound_arrivals=1' in r.getMessage()]
    assert len(blocked) == 3 and all('1 pre-existing' in m for m in blocked)   # loud, EVERY cycle, names the cohort
    assert rig.state.stats['drain_consolidation_blocked_unbound'] == 3
    # Leg 2 binds it: the next cycle drains ONE batch and runs its steps
    rig.graph.create_node(node_id='cc:conv::bound-partner', metadata={})
    rig.graph.create_synapse('cc:conv::old-orphan', 'cc:conv::bound-partner', weight=0.2)
    rig.run(1)
    assert rig.seen == ['a', 'b'] and rig.remaining() == []
    assert rig.graph.timestep == 250 and rig.owned == [False]


def test_a_turn_that_fails_after_depositing_unbound_nodes_wedges_the_drain_loudly_not_silently(rig, caplog):
    """The consequence of rules 1+2, now with C9 (Exec P488): a node that cannot bind (a dual pass that raised after
    depositing nodes) blocks the steps AND every later batch until it binds -- loudly every cycle -- and the blocked batch's
    steps are RE-ARMED as pending, not dropped: when it binds, the owed pass runs FIRST and only then does the next batch
    deposit and get its OWN pass."""
    rig.write(('p', 3, 'fail_partial'), ('q', 10, 'bound'))
    with caplog.at_level(logging.DEBUG, logger=rig.d.logger.name):
        rig.run(1)
    assert rig.seen == ['p', 'q'] and rig.graph.timestep == 0           # the batch landed, its steps were BLOCKED
    assert rig.state.pending_consolidation and rig.state.pending_consolidation_steps == 250   # ... and re-armed, FULL pass
    rig.write(('r', 10, 'bound'))
    held = rig.tract.read_bytes()
    with caplog.at_level(logging.DEBUG, logger=rig.d.logger.name):
        rig.run(2)
    assert rig.tract.read_bytes() == held and rig.seen == ['p', 'q']    # r waits in the tract, untouched
    errs = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR
            and 'consolidation_skipped_unbound_arrivals=3' in r.getMessage()]
    assert len(errs) == 3                                               # one per blocked cycle (1 post-batch + 2 pending re-runs)
    assert 'turns=2' in errs[0] and 'run=fresh' in errs[0]              # the post-batch record names its batch (item 1)
    assert all('run=pending' in m for m in errs[1:]) and all('turns=2' in m for m in errs[1:])   # the pending run keeps it
    assert rig.state.stats['drain_consolidation_blocked_unbound'] == 3
    # #907 on the REAL path: no raw node id in any record (counts only without the helper, kind:12-hex with it)
    assert not [r for r in caplog.records if 'cc:conv::p' in r.getMessage() or 'cc:conv::q' in r.getMessage()]
    # Leg 2 binds p's stray nodes: the owed pass runs FIRST (250 steps), THEN r deposits and gets its OWN 250
    rig.graph.create_synapse('cc:conv::p::0', 'cc:conv::p::1', weight=0.2)
    rig.graph.create_synapse('cc:conv::p::0', 'cc:conv::p::2', weight=0.2)
    rig.run(1)
    assert rig.seen == ['p', 'q', 'r'] and rig.remaining() == []
    assert rig.graph.timestep == 500 and rig.owned == [False, False]    # two full passes, two separate consolidations
    assert rig.state.pending_consolidation is None and rig.state.pending_consolidation_steps is None


# ---------------------------------------------------------------- needs #905: the REAL shared function, the REAL contract

def test_real_consolidate_contract_complete_held_failed_and_unchanged_default(monkeypatch):
    """needs #905. The real _cc_callosum_consolidate on a real Graph: a complete pass, a pass HELD at a slice boundary by the guard
    (done/remaining tell how far, nothing aged past it), a RAISING guard (fail closed: held AND failed), and the positional
    default call behaving exactly as before."""
    _require_905()
    monkeypatch.setenv('CC_CALLOSUM_LOCK_SLICE_STEPS', '25')
    g = Graph()
    g._concurrent_lock = _SpyRLock()
    prog = {}
    assert org._cc_callosum_consolidate(g, 100, guard=lambda: set(), progress=prog) is True
    assert prog == {'done': 100, 'remaining': 0, 'held': False, 'failed': False} and g.timestep == 100
    calls = {'n': 0}

    def blocks_on_third():
        calls['n'] += 1
        return {'x'} if calls['n'] >= 3 else set()
    prog = {}
    assert org._cc_callosum_consolidate(g, 100, guard=blocks_on_third, progress=prog) is False
    assert prog == {'done': 50, 'remaining': 50, 'held': True, 'failed': False} and g.timestep == 150

    def raising():
        raise RuntimeError('boom')
    prog = {}
    assert org._cc_callosum_consolidate(g, 100, guard=raising, progress=prog) is False
    assert prog['held'] is True and prog['failed'] is True and prog['done'] == 0 and g.timestep == 150
    assert org._cc_callosum_consolidate(g, 30) is True and g.timestep == 180      # the positional call: unchanged


def test_real_daemon_cycle_with_a_mid_pass_hold_carries_exactly_the_remaining_steps(rig, monkeypatch):
    """needs #905. The real daemon loop + the real drain + the real shared consolidation, with the REAL per-slice guard
    constructor replaced by one that blocks at the third slice boundary: the pass is HELD after 50 of 250 steps, the pending
    carries exactly 200, and when the guard clears the owed 200 run FIRST, then the next batch deposits with its OWN 250."""
    _require_905()
    monkeypatch.setenv('CC_CALLOSUM_LOCK_SLICE_STEPS', '25')
    state = {'n': 0, 'block': True}

    def fake_guard_ctor(graph):
        def guard():
            state['n'] += 1
            return {'blocker'} if (state['block'] and state['n'] == 3) else set()
        return guard
    monkeypatch.setattr(cc_topology_merge, 'whole_graph_guard', fake_guard_ctor)
    rig.write(('a', 30, 'bound'), ('b', 10, 'bound'))                       # a alone crosses 25 nodes: batch 1 = a, batch 2 = b
    rig.run(1)
    assert rig.seen == ['a'] and rig.graph.timestep == 50
    assert rig.state.pending_consolidation and rig.state.pending_consolidation_steps == 200
    state['block'] = False                                                  # the guard clears
    rig.run(1)
    assert rig.seen == ['a', 'b'] and rig.remaining() == []
    assert rig.graph.timestep == 50 + 200 + 250                             # the owed 200, THEN b's own full 250
    assert rig.state.pending_consolidation is None and rig.state.pending_consolidation_steps is None
    assert rig.owned == [False, False, False]                               # nothing held at any of the three consolidations
