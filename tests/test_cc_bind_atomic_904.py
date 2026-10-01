# ---- Changelog ----
# [2026-10-02] Claude Sonnet 5.5 (Z12 builder, lane ng-trial-chain-s4, dispatch #14658) — N8 (TRIAL branch, test-only): F-B, the stale cross-change pin.
#   test_generate_emergent_want_is_untouched compared generate_emergent_want to the PRE-#905 base (c5684334), which cannot hold once #905 part A is in the
#   same tree (either merge order). RE-BASED (not retired) to compare against #905's own version (ee94f7d2) through the same `git show` mechanism, so it
#   still pins that #904 itself adds nothing to that function; the 'new' side is now the module under test (org.__file__), which is the worktree file in
#   a normal run and lets the existing Z12_904_ORG_UNDER_TEST hook feed a scratch mutant. A failed `git show` is now a loud, explained assert. No other test changed.
# [2026-10-01] Z12 builder (Claude Sonnet 5.5), lane dual-pass-atomic-904, dispatch #13437 — #904 ROUND 2: le-053 F1-1 (the verdict sentence says what was
#   KEPT), F1-2 (a probation sweep between the snapshot and the rollback is NOT erased; a key added by the call is deleted), F1-3 (the restore's
#   field list is tied to the deposit's write-set by AST), F1-4 (the unrestorable count is printed).
# [2026-10-01] Z12 builder (Claude Sonnet 5.5), lane dual-pass-atomic-904, dispatch #13259 — #904 FOLD-UP: tests for C3/F11 (a failed EXACT-REPEAT
#   turn failing after the hyperedge create: kills M20), C4/F1 (a failed attempt leaves NO trace on a pre-existing node, over N held cycles),
#   checker-038 corrections 2 (a raising vdb.get() must not orphan what the call inserted) and 3 (the bind COMPLETES, a LATER stage fails).
# [2026-10-01] Z12 builder (Claude Sonnet 5.5), lane dual-pass-atomic-904, dispatch #13058 — #904 (Exec Packets 487 + 489 + 491)
# What: tests for (A) the ATOMIC / loud / truthful dual pass, (B) the daemon honouring a held tract entry (real drain, real daemon loop,
#   real NG), (C) the #805 temp-graph retry-safety PROOF, the Packet 489 addendum (surface_wants / surface_wants_for_graph seed-synapse
#   swallow now LOUD) and the Packet 491 addendum: (i) the turn-sequence pointer survives a daemon restart (real NG + the docs daemon's
#   real _guarded_save / init_ng, all paths redirected into tmp), (ii) every bind site is loud, counted and truthful (per-site spies, the
#   truth rule), (iii) a forest born unbound is announced once.
# Why: #898: the 147 conversational nodes were never bound. The dual pass wrote the forest, trees and windows in separate locks, the
#   bind came later, any exception after the forest write skipped the bind at logger.debug and returned False (or, for a failure INSIDE
#   the bind, returned True), every caller ignored the result, and the drain truncated the failed turn. Each test below that pins the
#   fix FAILS on the BASE (c5684334) and passes on the fix; the golden / pre-write / pin tests are stated to pass on both.
# How: a REAL in-memory neuro_foundation.Graph + a REAL SimpleVectorDB + the REAL ng_embed.NGEmbed.dual_record_outcome (vendored, unedited) with
#   ONLY its model/TID seams replaced on the instance (_extract_concepts, embed_batch, embed_windows: deterministic hash vectors, scripted
#   concepts, so an attempt-2 can extract DIFFERENT concepts like the LLM does). The drain is the real drain_ingest_tract on real tract
#   bytes in tmp files (never the live tract). The daemon-loop tests load the docs daemon by path from Z12_904_DAEMON_UNDER_TEST (skipped,
#   with the reason, when unset). Z12_904_ORG_UNDER_TEST points the module under test at a SCRATCH copy of cc_ng_organism.py (the BASE, or
#   a mutation) so the same file can be run against the base and against each mutation. P379/#770: a preamble prints every module's path and
#   the session FAILS if an NG module (other than a scratch org under /tmp/z12-904-scratch) resolves outside this worktree.
# -------------------
import ast
import hashlib
import importlib.util
import inspect
import json
import logging
import os
import random
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest

_WORKTREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_WORKTREE))

import ng_embed  # noqa: E402
import ng_tract  # noqa: E402
import neuro_foundation  # noqa: E402
from neuro_foundation import Graph  # noqa: E402
from universal_ingestor import SimpleVectorDB  # noqa: E402

_SCRATCH_ROOTS = ('/tmp/z12-904-scratch', '/tmp/z12-904-fold-scratch')   # scratch copies of the organism (base / mutants) live here
_ORG_ALT = os.environ.get('Z12_904_ORG_UNDER_TEST')
if _ORG_ALT:
    _spec = importlib.util.spec_from_file_location('cc_ng_organism', _ORG_ALT)
    org = importlib.util.module_from_spec(_spec)
    sys.modules['cc_ng_organism'] = org
    _spec.loader.exec_module(org)
else:
    import cc_ng_organism as org  # noqa: E402

DAEMON_ENV = os.environ.get('Z12_904_DAEMON_UNDER_TEST')

_PREAMBLE = ['[P379/#770] worktree root           -> %s' % _WORKTREE]
for _m in (org, ng_embed, neuro_foundation):
    _PREAMBLE.append('[P379/#770] %-20s -> %s' % (_m.__name__, Path(_m.__file__).resolve()))
_PREAMBLE.append('[P379/#770] ng_tract (site, Rust)  -> %s' % getattr(ng_tract, '__file__', ng_tract))
_PREAMBLE.append('[P379/#770] daemon under test      -> %s' % (DAEMON_ENV or '(none: the daemon-loop tests are skipped)'))
_PREAMBLE.append('[P379/#770] org under test is a scratch copy: %s' % bool(_ORG_ALT))
sys.__stderr__.write('\n' + '\n'.join(_PREAMBLE) + '\n')


def _inside_worktree(mod):
    return _WORKTREE in Path(mod.__file__).resolve().parents


for _m in (ng_embed, neuro_foundation):
    if not _inside_worktree(_m):
        raise RuntimeError('P379/#770 FAIL: %s resolves outside the worktree %s' % (_m.__name__, _WORKTREE))
if _ORG_ALT:
    if not any(Path(r).resolve() in Path(org.__file__).resolve().parents for r in _SCRATCH_ROOTS):
        raise RuntimeError('P379/#770 FAIL: a scratch org must live under one of %s, got %s' % (_SCRATCH_ROOTS, org.__file__))
elif not _inside_worktree(org):
    raise RuntimeError('P379/#770 FAIL: cc_ng_organism resolves outside the worktree %s' % _WORKTREE)


@pytest.fixture(scope='session', autouse=True)
def _preamble_visible(request):
    tr = request.config.pluginmanager.get_plugin('terminalreporter')
    if tr is not None:
        tr.write_line('')
        for line in _PREAMBLE:
            tr.write_line(line)


SECRET = 'SECRET-EXC-TEXT-904'
TEXT_CANARY = 'TEXTCANARY904'
TURN_A = 'TURN-A alpha beta %s' % TEXT_CANARY
TURN_B = 'TURN-B gamma delta %s' % TEXT_CANARY
TURN_T1 = 'TURN-T1 first %s' % TEXT_CANARY
TURN_T2 = 'TURN-T2 second %s' % TEXT_CANARY
TURN_T3 = 'TURN-T3 third %s' % TEXT_CANARY
CONCEPTS_A = ['alpha concept one', 'alpha concept two']
CONCEPTS_B = ['gamma concept one', 'gamma concept two', 'gamma concept three']
HOLD_PREFIX = 'CC ingest-tract hold'
LOUD_MARK = 'dual-pass FAILED after the first write'


def _vec(text):
    d = hashlib.sha256(str(text).encode()).digest()
    return np.array([b / 255.0 + 0.05 for b in d[:16]], dtype=np.float32)


class FaultyVDB(SimpleVectorDB):
    """A REAL SimpleVectorDB whose insert can be made to fail for chosen ids (the write that raises midway)."""

    def __init__(self):
        super().__init__()
        self.fail_ids = lambda _id: False

    def insert(self, id, embedding, content='', metadata=None):
        if self.fail_ids(id):
            raise OSError(SECRET)
        return super().insert(id, embedding, content, metadata)


def _new_stores():
    return Graph(), FaultyVDB(), {'last_forest_id': None}


@pytest.fixture
def world(monkeypatch):
    w = types.SimpleNamespace()
    w.graph, w.vdb, w.state = _new_stores()
    w.concepts = {}      # text -> list[str] | None (None = pass-2 extraction failure)
    w.windows = {}       # text -> list[str]
    monkeypatch.setattr(random, 'randint', lambda a, b: 3)   # the bind's delays are random: pin them
    inst = ng_embed.NGEmbed()
    inst._extract_concepts = lambda text: w.concepts.get(text, [])
    inst.embed_batch = lambda concepts, **kw: [_vec(c) for c in concepts]
    inst.embed_windows = lambda text, **kw: types.SimpleNamespace(
        windows=[types.SimpleNamespace(text=t, embedding=_vec(t)) for t in w.windows.get(text, [])])
    monkeypatch.setattr(ng_embed.NGEmbed, 'get_instance', classmethod(lambda cls, config=None: inst))
    monkeypatch.setattr(ng_embed, 'embed', _vec)
    w.run = lambda text: org.run_conversational_dual_pass(w.graph, w.vdb, text, _vec(text), w.state)

    def clean_pass(texts):
        """One clean pass of `texts`, in order, on a FRESH graph/vdb/state, using the CURRENT extraction tables."""
        g, v, st = _new_stores()
        for t in texts:
            assert org.run_conversational_dual_pass(g, v, t, _vec(t), st) is True
        return types.SimpleNamespace(graph=g, vdb=v, state=st)

    w.clean_pass = clean_pass
    w.concepts.update({TURN_A: CONCEPTS_A, TURN_B: CONCEPTS_B})
    w.windows.update({TURN_A: ['A window one', 'A window two'], TURN_B: ['B window one', 'B window two', 'B window three']})
    return w


def snap(w):
    """Everything a dual pass can leave in the graph / vdb / state, comparable across runs (ids that embed a uuid are excluded)."""
    g = w.graph
    return {
        'nodes': sorted(g.nodes),
        'synapses': sorted((s.pre_node_id, s.post_node_id, round(float(s.weight), 6), int(s.delay)) for s in g.synapses.values()),
        'hyperedges': sorted((tuple(sorted(h.member_nodes)), h.metadata.get('creation_mode')) for h in g.hyperedges.values()),
        'vdb': sorted(w.vdb.all_ids()),
        'out_index': sorted(g._outgoing), 'in_index': sorted(g._incoming),
        'node_he_counts': {n: len(s) for n, s in sorted(g._node_hyperedges.items())},
        'timestep': int(g.timestep),
        'last_forest_id': w.state.get('last_forest_id'),
        'primed': sorted((w.state.get('primed_nodes') or {}).keys()),
    }


def _loud(caplog):
    return [r for r in caplog.records if r.name == org.logger.name and LOUD_MARK in r.getMessage()]


def _hold_records(caplog):
    return [r for r in caplog.records if r.name == org.logger.name and r.getMessage().startswith(HOLD_PREFIX)]


def _forest_id(text):
    return 'cc:conv::' + hashlib.sha1(text.encode()).hexdigest()


# ------------------------------------------------------------------ pins on the shape (pass on base AND fix)

def test_signatures_are_pinned():
    import inspect
    assert list(inspect.signature(org.run_conversational_dual_pass).parameters) == ['graph', 'vector_db', 'text', 'embedding', 'state']
    assert list(inspect.signature(org.drain_ingest_tract).parameters)[-1] == 'hold_on_failure'
    assert inspect.signature(org.drain_ingest_tract).parameters['hold_on_failure'].default is False


def test_ng_embed_is_unedited_vendored_file():
    """ng_embed.py is VENDORED (LAW 2): this build must not have touched it."""
    import subprocess
    out = subprocess.run(['git', '-C', str(_WORKTREE), 'diff', '--name-only', 'c568433434a99154b9f7652b77cbe43752d34324', '--',
                          'ng_embed.py', 'neuro_foundation.py', 'openclaw_hook.py', 'stream_parser.py', 'activation_persistence.py'],
                         capture_output=True, text=True)
    assert out.stdout.strip() == ''


# ------------------------------------------------------------------ A: the success path is byte-identical (golden: pass on base AND fix)

def _success_scenario(w):
    assert w.run(TURN_A) is True
    assert w.run(TURN_B) is True
    return snap(w)


# Generated by running _success_scenario against the BASE module (c5684334); see the return for the command.
GOLDEN_SHA = '569dfe325c21d19d53e54170dd37befa444537edfc94f07d27d703a09bdb3aab'
GOLDEN_COUNTS = {'nodes': 12, 'synapses': 24, 'hyperedges': 2, 'vdb': 7}


def test_success_path_is_golden_identical_to_base(world, caplog):
    with caplog.at_level(logging.DEBUG, logger=org.logger.name):
        s = _success_scenario(world)
    got = {k: len(s[k]) for k in GOLDEN_COUNTS}
    assert got == GOLDEN_COUNTS, got
    assert hashlib.sha256(json.dumps(s, sort_keys=True, default=list).encode()).hexdigest() == GOLDEN_SHA
    assert s['last_forest_id'] == _forest_id(TURN_B)
    # nothing on the success path is louder than debug (the bind used to log its own swallows; there are none to log)
    assert [r for r in caplog.records if r.name == org.logger.name and r.levelno >= logging.WARNING] == []


# ------------------------------------------------------------------ A: the partial-write reproducers (FAIL on base, PASS on fix)

CASES = [
    # case, stage, exception class name, bind site (or '-'), site writes attempted when it failed
    ('tree_vdb_insert', 'dual_record', 'OSError', '-', 0),
    ('tree_create_node', 'dual_record', 'RuntimeError', '-', 0),
    ('window_deposit', 'windows', 'RuntimeError', '-', 0),
    ('bind_tree_synapse', 'bind', 'RuntimeError', 'tree_synapse', 3),      # tree 1: 2 ok, tree 2: its first synapse fails
    ('bind_window_synapse', 'bind', 'RuntimeError', 'window_synapse', 3),  # window 1: 2 ok, window 2: its first synapse fails
    ('bind_hyperedge', 'bind', 'RuntimeError', 'hyperedge', 1),
    ('bind_window_chain', 'bind', 'RuntimeError', 'window_chain', 1),
    ('bind_prev_link', 'bind', 'RuntimeError', 'sequence_link', 1),
    ('bind_forest_absent', 'bind', 'RuntimeError', 'forest_absent', 1),    # the forest vanished between its write and the bind
]


def _inject(w, case, monkeypatch):
    g = w.graph
    forest_a = _forest_id(TURN_A)
    if case == 'tree_vdb_insert':          # the 2nd tree: its node IS created, then the recall insert raises
        w.vdb.fail_ids = lambda i: i.endswith('::tree::' + CONCEPTS_B[1])
    elif case == 'tree_create_node':       # the 2nd tree's create_node raises (earlier forest + tree 1 already written)
        real = g.create_node

        def create_node(node_id=None, metadata=None, **kw):
            if node_id and node_id.endswith('::tree::' + CONCEPTS_B[1]):
                raise RuntimeError(SECRET)
            return real(node_id=node_id, metadata=metadata, **kw)
        g.create_node = create_node
    elif case == 'bind_forest_absent':     # the forest is removed after the 1st window is deposited: the bind finds no forest
        real_dep = org._cc_deposit_memory_node

        def dep_then_drop_forest(graph, vdb, node_id, *a, **k):
            out = real_dep(graph, vdb, node_id, *a, **k)
            if node_id.endswith('::window::0'):
                graph.remove_node(_forest_id(TURN_B))
            return out
        monkeypatch.setattr(org, '_cc_deposit_memory_node', dep_then_drop_forest)
    elif case == 'window_deposit':         # forest + 3 trees written, then the 2nd window's deposit raises
        real_dep = org._cc_deposit_memory_node

        def dep(graph, vdb, node_id, *a, **k):
            if node_id.endswith('::window::1'):
                raise RuntimeError(SECRET)
            return real_dep(graph, vdb, node_id, *a, **k)
        monkeypatch.setattr(org, '_cc_deposit_memory_node', dep)
    else:
        real_syn, real_he = g.create_synapse, g.create_hyperedge

        def create_synapse(pre, post, *a, **k):
            if case == 'bind_tree_synapse' and post.endswith('::tree::' + CONCEPTS_B[1]) and pre == _forest_id(TURN_B):
                raise RuntimeError(SECRET)
            if case == 'bind_window_synapse' and pre == _forest_id(TURN_B) and post.endswith('::window::1'):
                raise RuntimeError(SECRET)
            if case == 'bind_window_chain' and '::window::' in pre and '::window::' in post:
                raise RuntimeError(SECRET)
            if case == 'bind_prev_link' and pre == forest_a and post == _forest_id(TURN_B):
                raise RuntimeError(SECRET)
            return real_syn(pre, post, *a, **k)

        def create_hyperedge(*a, **k):
            if case == 'bind_hyperedge':
                raise RuntimeError(SECRET)
            return real_he(*a, **k)
        g.create_synapse, g.create_hyperedge = create_synapse, create_hyperedge


@pytest.mark.parametrize('case,stage,exc_name,site,site_attempted', CASES)
def test_partial_write_is_rolled_back_loudly_and_truthfully(world, monkeypatch, caplog, case, stage, exc_name, site, site_attempted):
    w = world
    assert w.run(TURN_A) is True                       # a previous turn: so the prev -> current link has something to link
    before = snap(w)
    primed_before = w.state.get('primed_nodes')
    _inject(w, case, monkeypatch)
    with caplog.at_level(logging.DEBUG, logger=org.logger.name):
        ok = w.run(TURN_B)
    assert ok is False
    # ATOMIC: the graph, the vdb, the indices and the state are exactly as before the call (BASE: forest/trees/windows stay, unbound,
    # or, for a failure inside the bind, the call returns True with a half-bound turn)
    assert snap(w) == before
    assert w.state.get('primed_nodes') is primed_before
    # LOUD: exactly one record above debug, at WARNING (the rollback succeeded), with the fields and none of the forbidden text
    loud = _loud(caplog)
    assert len(loud) == 1, [r.getMessage() for r in caplog.records]
    rec = loud[0]
    assert rec.levelno == logging.WARNING
    msg = rec.getMessage()
    assert 'stage=%s' % stage in msg and 'exc_type=%s' % exc_name in msg
    assert 'site=%s ' % site in msg and 'site_attempted=%d ' % site_attempted in msg    # the failing write SITE is named (Exec P491 (ii))
    assert 'site_failed=%d ' % (0 if site == '-' else 1) in msg
    assert 'written forest=1 ' in msg and 'rolled back' in msg and 'NOT bound' in msg
    # COUNTED: the failure is tallied per site in the caller-owned state, and survives the rollback (BASE: no such key)
    if site != '-':
        assert w.state.get('bind_site_failures') == {site: 1}
    else:
        assert 'bind_site_failures' not in w.state
    for forbidden in (SECRET, TEXT_CANARY, 'cc:conv::', '::tree::', '::window::', 'concept'):
        assert forbidden not in msg, forbidden
    assert rec.exc_info is None                       # no traceback / str(exc) smuggled in through exc_info


def test_tree_failure_counts_what_was_written(world, monkeypatch, caplog):
    w = world
    _inject(w, 'tree_vdb_insert', monkeypatch)
    with caplog.at_level(logging.DEBUG, logger=org.logger.name):
        assert w.run(TURN_B) is False
    msg = _loud(caplog)[0].getMessage()
    assert 'written forest=1 trees=2 windows=0 synapses=0 hyperedges=0' in msg   # tree 1 whole, tree 2 written midway
    assert snap(w)['nodes'] == [] and snap(w)['vdb'] == []


def test_failure_in_the_window_stage_counts_windows(world, monkeypatch, caplog):
    w = world
    _inject(w, 'window_deposit', monkeypatch)
    with caplog.at_level(logging.DEBUG, logger=org.logger.name):
        assert w.run(TURN_B) is False
    assert 'written forest=1 trees=3 windows=2 ' in _loud(caplog)[0].getMessage()   # window 0 whole, window 1 raised midway


def test_rollback_failure_is_an_error_and_says_a_partial_write_remains(world, monkeypatch, caplog):
    w = world
    _inject(w, 'bind_hyperedge', monkeypatch)
    real_remove = w.graph.remove_node

    def remove_node(node_id):
        raise RuntimeError(SECRET)
    w.graph.remove_node = remove_node
    with caplog.at_level(logging.DEBUG, logger=org.logger.name):
        ok = w.run(TURN_B)
    assert ok is False                                  # still truthful
    loud = _loud(caplog)
    assert len(loud) == 1 and loud[0].levelno == logging.ERROR
    msg = loud[0].getMessage()
    assert 'ROLLBACK FAILED' in msg and 'REMAINS' in msg and SECRET not in msg and TEXT_CANARY not in msg
    assert _forest_id(TURN_B) in w.graph.nodes          # the partial write really is still there: the report is honest
    w.graph.remove_node = real_remove


def test_a_node_that_existed_before_the_call_is_never_removed(world, monkeypatch, caplog):
    """An exact-repeat turn lands on the SAME forest id (content-hashed). A failed repeat (with re-extracted, DIFFERENT concepts) must
    remove only what IT created, never the earlier success."""
    w = world
    assert w.run(TURN_A) is True
    before = snap(w)
    w.concepts[TURN_A] = [CONCEPTS_A[0], 'alpha concept three']       # one existing tree, one new
    _inject_hyperedge = w.graph.create_hyperedge
    w.graph.create_hyperedge = lambda *a, **k: (_ for _ in ()).throw(RuntimeError(SECRET))
    with caplog.at_level(logging.DEBUG, logger=org.logger.name):
        assert w.run(TURN_A) is False
    w.graph.create_hyperedge = _inject_hyperedge
    assert snap(w) == before
    assert _forest_id(TURN_A) in w.graph.nodes and _forest_id(TURN_A) in w.vdb.all_ids()
    assert not any(n.endswith('alpha concept three') for n in w.graph.nodes)


def test_an_orphan_vdb_entry_that_existed_before_is_not_deleted(world, monkeypatch, caplog):
    """The real vdb holds thousands of conversational entries whose graph node is gone (#898). A failed call must not delete an entry
    it did not insert (it re-wrote it with the same content, as a success would)."""
    w = world
    w.vdb.insert(_forest_id(TURN_B), _vec(TURN_B), TURN_B, {})
    _inject(w, 'bind_hyperedge', monkeypatch)
    with caplog.at_level(logging.DEBUG, logger=org.logger.name):
        assert w.run(TURN_B) is False
    assert w.vdb.all_ids() == [_forest_id(TURN_B)] and w.graph.nodes == {}


def test_failure_before_the_first_write_is_unchanged_debug_and_false(world, caplog):
    """Pass-2 extraction failure (TID down) raises DualPassIncompleteError BEFORE any write: nothing to undo, debug line, False.
    PASSES on base and fix (it pins what must not change)."""
    w = world
    w.concepts[TURN_B] = None
    before = snap(w)
    with caplog.at_level(logging.DEBUG, logger=org.logger.name):
        assert w.run(TURN_B) is False
    assert snap(w) == before
    mine = [r for r in caplog.records if r.name == org.logger.name]
    assert [r.levelno for r in mine] == [logging.DEBUG]
    assert 'dual-pass failed (non-fatal)' in mine[0].getMessage()


# ------------------------------------------------------------------ B + C: the drain HOLD, end to end on real tract bytes

def _frame_bytes(tmp_path, tag, text):
    p = str(tmp_path / ('frame_%s.tract' % tag))
    if os.path.exists(p):
        os.remove(p)
    ng_tract.deposit_experience(raw=text.encode(), source='cc_gateway', tract_paths=[p])
    with open(p, 'rb') as f:
        return f.read()


class Tract:
    def __init__(self, tmp_path, texts):
        self.path = str(tmp_path / 'turns.tract')
        self.frames = [_frame_bytes(tmp_path, 'f%d' % i, t) for i, t in enumerate(texts)]
        self.texts = list(texts)
        self.rewrite(b''.join(self.frames))

    def rewrite(self, data):
        assert 'plugins/neurograph' not in self.path and '/pytest-' in self.path, 'SAFETY: not a pytest tmp tract: %s' % self.path
        with open(self.path, 'wb') as f:
            f.write(data)

    def read(self):
        with open(self.path, 'rb') as f:
            return f.read()


def _drain(w, tract, hold):
    return org.drain_ingest_tract(w.graph, w.vdb, w.state, tract_path=tract.path, return_consumed=True, hold_on_failure=hold)


def _setup_three(w, tmp_path):
    w.concepts.update({TURN_T1: ['first concept one', 'first concept two'],
                       TURN_T2: ['second concept one', 'second concept two', 'second concept three'],
                       TURN_T3: ['third concept one', 'third concept two']})
    w.windows.update({TURN_T1: ['T1 window one', 'T1 window two'], TURN_T2: ['T2 window one', 'T2 window two'],
                      TURN_T3: ['T3 window one', 'T3 window two']})
    return Tract(tmp_path, [TURN_T1, TURN_T2, TURN_T3])


def test_drain_hold_end_to_end_failed_turn_is_held_not_truncated(world, tmp_path, caplog, monkeypatch):
    """Real tract bytes, real drain, REAL dual pass failing partway on turn 2: with the flag, the prefix (T1) is truncated, T2 and T3
    stay byte-identical, one drain warning and one dual-pass warning per cycle. FAILS on base (the failed turn leaves unbound nodes
    and no dual-pass record)."""
    w = world
    tr = _setup_three(w, tmp_path)
    w.vdb.fail_ids = lambda i: i.endswith('::tree::second concept two')
    with caplog.at_level(logging.DEBUG, logger=org.logger.name):
        absorbed, consumed = _drain(w, tr, True)
    assert absorbed == 1 and consumed == tr.frames[0]
    assert tr.read() == tr.frames[1] + tr.frames[2]
    assert len(_hold_records(caplog)) == 1 and len(_loud(caplog)) == 1
    assert 'reason=absorb_returned_false' in _hold_records(caplog)[0].getMessage()
    # nothing of the failed turn is left in the graph (no unbound stragglers): the graph is exactly "T1 only"
    only_t1 = w.clean_pass([TURN_T1])
    assert snap(w) == snap(only_t1)


def test_flag_off_truncates_a_failed_turn_as_before(world, tmp_path, caplog):
    """The #794 default (hold_on_failure=False) is unchanged: the failed turn is consumed (lost), the rest is absorbed. The tract-level
    assertions hold on base AND fix; the final graph assertion FAILS on base (the failed turn leaves unbound remnants there) and holds on
    the fix (T2 is lost with the flag off, but nothing of it remains in the graph)."""
    w = world
    tr = _setup_three(w, tmp_path)
    w.vdb.fail_ids = lambda i: i.endswith('::tree::second concept two')
    with caplog.at_level(logging.DEBUG, logger=org.logger.name):
        absorbed, consumed = _drain(w, tr, False)
    assert absorbed == 2 and consumed == b''.join(tr.frames) and tr.read() == b''
    assert _hold_records(caplog) == []
    assert snap(w) == snap(w.clean_pass([TURN_T1, TURN_T3]))      # T2 lost (flag off), but no unbound remnants of it


def test_805_proof_failed_turn_then_a_retry_with_different_concepts_equals_one_clean_pass(world, tmp_path, caplog):
    """THE #805 PROOF. hold_on_failure=True. Cycle 1: T2's dual pass fails partway (a tree write raises after the forest, one tree and
    the windows... are written). Cycle 2: the fault is gone AND the extractor now returns DIFFERENT concepts for T2 (LLM variance: the
    case a LOUD-only fix cannot survive). The final graph must equal ONE CLEAN PASS of the same turns: same node set (no orphan /
    straggler), no duplicate forest / tree / window, same synapses and hyperedge set; the tract is empty of the entry."""
    w = world
    tr = _setup_three(w, tmp_path)
    w.vdb.fail_ids = lambda i: i.endswith('::tree::second concept two')
    with caplog.at_level(logging.DEBUG, logger=org.logger.name):
        a1, c1 = _drain(w, tr, True)
    assert (a1, c1) == (1, tr.frames[0]) and tr.read() == tr.frames[1] + tr.frames[2]
    # cycle 2: healthy, different extraction for the held turn
    w.vdb.fail_ids = lambda i: False
    w.concepts[TURN_T2] = ['second concept four', 'second concept five']
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger=org.logger.name):
        a2, c2 = _drain(w, tr, True)
    assert (a2, c2) == (2, tr.frames[1] + tr.frames[2])
    assert tr.read() == b''                                           # the tract is empty of that entry
    assert _hold_records(caplog) == [] and _loud(caplog) == []        # a healthy cycle is quiet
    final = snap(w)
    clean = snap(w.clean_pass([TURN_T1, TURN_T2, TURN_T3]))
    assert final == clean
    # explicit: no node of the abandoned first attempt survives, no duplicate forest / tree / window
    assert not any('second concept one' in n or 'second concept two' in n or 'second concept three' in n for n in final['nodes'])
    assert len(final['nodes']) == len(set(final['nodes']))
    assert final['hyperedges'] == clean['hyperedges'] and final['synapses'] == clean['synapses']


def test_805_proof_a_turn_that_fails_every_cycle_stays_byte_identical_and_logs_per_cycle(world, tmp_path, caplog):
    """A turn that fails EVERY cycle stays in the tract byte-identical (with everything after it), leaks nothing into the graph, and
    logs exactly TWO WARNINGs per cycle (one from the dual pass, one from the drain hold): the held entry is retried, re-embedded and
    re-extracted EVERY cycle. VOLUME at the daemon's 60 s autosave: 2 WARNING lines per minute (about 2,880 a day) for as long as an
    entry is held, plus one full dual-pass attempt per minute. Then it recovers when the fault clears."""
    w = world
    tr = _setup_three(w, tmp_path)
    w.vdb.fail_ids = lambda i: i.endswith('::tree::second concept two')
    held_bytes = tr.frames[1] + tr.frames[2]
    graph_after_first = None
    for cycle in range(4):
        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger=org.logger.name):
            absorbed, consumed = _drain(w, tr, True)
        assert tr.read() == held_bytes, 'cycle %d' % cycle
        warns = [r for r in caplog.records if r.name == org.logger.name and r.levelno >= logging.WARNING
                 and not r.getMessage().startswith('CC recall insert failed')]   # that line is _cc_deposit_memory_node's OWN, pre-existing
        assert len(warns) == 2, [r.getMessage() for r in warns]                  # (it logs str(exc): reported, out of scope here)
        assert len(_hold_records(caplog)) == 1 and len(_loud(caplog)) == 1
        assert absorbed == (1 if cycle == 0 else 0) and consumed == (tr.frames[0] if cycle == 0 else b'')
        if graph_after_first is None:
            graph_after_first = snap(w)
        assert snap(w) == graph_after_first, 'the failing turn leaked into the graph on cycle %d' % cycle
    w.vdb.fail_ids = lambda i: False
    absorbed, consumed = _drain(w, tr, True)
    assert absorbed == 2 and tr.read() == b''
    assert snap(w) == snap(w.clean_pass([TURN_T1, TURN_T2, TURN_T3]))


# ------------------------------------------------------------------ B: the REAL daemon loop on REAL NG (gated on Z12_904_DAEMON_UNDER_TEST)

needs_daemon = pytest.mark.skipif(
    not DAEMON_ENV, reason='Z12_904_DAEMON_UNDER_TEST is unset: the daemon-loop tests need the docs daemon file (the #904 docs worktree)')
_DAEMON_COUNT = [0]


@pytest.fixture
def daemon_rig(world, tmp_path, monkeypatch):
    w = world
    tr = _setup_three(w, tmp_path)
    monkeypatch.setenv('CC_GATEWAY_TRACT_PATH', tr.path)
    monkeypatch.setenv('CC_GATEWAY_CONDUIT_PATH', str(tmp_path / 'conduit'))
    for var in ('CC_NG_BATCH_SIZE', 'CC_NG_IDLE_STEPS', 'CC_NG_DRAIN_HOLD_ON_FAILURE'):
        monkeypatch.delenv(var, raising=False)
    trickled = []
    monkeypatch.setattr(org, 'trickle_gateway_conduit', lambda data, *a, **k: trickled.append(data))
    for name in ('cc_update_probation', 'surface_wants', 'generate_emergent_want', 'persist_cc_commons'):
        monkeypatch.setattr(org, name, lambda *a, **k: None)
    drain_calls = []
    real_drain = org.drain_ingest_tract
    monkeypatch.setattr(org, 'drain_ingest_tract', lambda *a, **k: (drain_calls.append(dict(k)), real_drain(*a, **k))[1])
    _DAEMON_COUNT[0] += 1
    spec = importlib.util.spec_from_file_location('z12_904_daemon_%d' % _DAEMON_COUNT[0], DAEMON_ENV)
    d = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = d
    spec.loader.exec_module(d)
    assert d.DRAIN_PACING is None                 # this NG base predates D24 (no batch_nodes/receipt): the unpaced call is the real one
    w.graph._concurrent_lock = threading.RLock()
    d.STATE = types.SimpleNamespace(running=True, lock=threading.Lock(), stats={}, conv_state=w.state,
                                    ng=types.SimpleNamespace(graph=w.graph, vector_db=w.vdb),
                                    pending_consolidation=None, pending_consolidation_batch=None)
    monkeypatch.setattr(d, '_guarded_save', lambda *a, **k: True)
    cyc = {'n': 0, 'max': 1}

    def sleep(_):
        if cyc['n'] >= cyc['max']:
            d.STATE.running = False
        cyc['n'] += 1

    monkeypatch.setattr(d, 'time', types.SimpleNamespace(sleep=sleep, time=time.time, monotonic=time.monotonic))

    def run(cycles):
        cyc['n'], cyc['max'] = 0, cycles
        d.STATE.running = True
        d._autosave_loop()

    return types.SimpleNamespace(d=d, tract=tr, w=w, run=run, drain_calls=drain_calls, trickled=trickled)


@needs_daemon
def test_daemon_loop_with_the_flag_on_holds_the_failed_turn_and_the_retry_equals_one_clean_pass(daemon_rig, caplog):
    r, w = daemon_rig, daemon_rig.w
    r.d.DRAIN_HOLD_ON_FAILURE = True
    w.vdb.fail_ids = lambda i: i.endswith('::tree::second concept two')
    with caplog.at_level(logging.DEBUG, logger=org.logger.name):
        r.run(1)
    assert r.drain_calls == [{'return_consumed': True, 'hold_on_failure': True}]
    assert r.tract.read() == r.tract.frames[1] + r.tract.frames[2]
    assert r.trickled == [r.tract.frames[0]]            # the Leg 1 conduit gets ONLY the bytes that left the file, never the held turn
    assert len(_hold_records(caplog)) == 1 and len(_loud(caplog)) == 1
    w.vdb.fail_ids = lambda i: False
    w.concepts[TURN_T2] = ['second concept four', 'second concept five']
    r.run(1)
    assert r.tract.read() == b''
    assert snap(w) == snap(w.clean_pass([TURN_T1, TURN_T2, TURN_T3]))


@needs_daemon
def test_daemon_loop_with_the_flag_off_is_the_pre_904_call_and_truncates(daemon_rig, caplog):
    r, w = daemon_rig, daemon_rig.w
    assert r.d.DRAIN_HOLD_ON_FAILURE is False             # absent from the environment: off
    w.vdb.fail_ids = lambda i: i.endswith('::tree::second concept two')
    with caplog.at_level(logging.DEBUG, logger=org.logger.name):
        r.run(1)
    assert r.drain_calls == [{'return_consumed': True}]   # byte-identical call: the kwarg is not passed at all
    assert r.tract.read() == b'' and r.trickled == [b''.join(r.tract.frames)]
    assert _hold_records(caplog) == []


# ------------------------------------------------------------------ Exec Packet 491 (ii): the truth rule, the counter

def _lying_bind(world, mutate):
    """A bind that does the real writes but REPORTS less than a turn with trees needs (what a swallowing bind would report)."""
    real = org._cc_bind_conversational_topology

    def bind(graph, forest_id, result, emb, state, window_ids=None, journal=None):
        status = real(graph, forest_id, result, emb, state, window_ids=window_ids, journal=journal)
        return mutate(dict(status))
    return bind


@pytest.mark.parametrize('how,mutate', [
    ('no_hyperedge', lambda st: {**st, 'hyperedge': False}),
    ('short_tree_pairs', lambda st: {**st, 'tree_pairs': st['trees'] - 1}),
    ('no_tree_pairs', lambda st: {**st, 'tree_pairs': 0}),
])
def test_a_bind_that_wrote_less_than_a_turn_with_trees_needs_is_not_reported_true(world, monkeypatch, caplog, how, mutate):
    """FAILS on base (it ignores the bind entirely and returns True). The rule is ALL-OR-NOTHING: every forest<->tree pair AND the
    hyperedge, else the turn is rolled back and False is returned with its own loud record."""
    w = world
    assert w.run(TURN_A) is True
    before = snap(w)
    monkeypatch.setattr(org, '_cc_bind_conversational_topology', _lying_bind(w, mutate))
    with caplog.at_level(logging.DEBUG, logger=org.logger.name):
        ok = w.run(TURN_B)
    assert ok is False
    assert snap(w) == before                                  # rolled back: nothing of the half-reported turn remains
    loud = _loud(caplog)
    assert len(loud) == 1 and loud[0].levelno == logging.WARNING
    assert 'site=bind_postcondition ' in loud[0].getMessage() and w.state.get('bind_site_failures') == {'bind_postcondition': 1}


def test_a_truthful_full_status_is_true(world, monkeypatch, caplog):
    w = world
    monkeypatch.setattr(org, '_cc_bind_conversational_topology', _lying_bind(w, lambda st: st))
    with caplog.at_level(logging.DEBUG, logger=org.logger.name):
        assert w.run(TURN_B) is True
    assert _loud(caplog) == [] and 'bind_site_failures' not in w.state


def test_the_site_counter_accumulates_per_site(world, monkeypatch):
    w = world
    _inject(w, 'bind_hyperedge', monkeypatch)
    assert w.run(TURN_B) is False and w.run(TURN_B) is False
    assert w.state['bind_site_failures'] == {'hyperedge': 2}
    assert snap(w)['nodes'] == []


def test_the_success_path_leaves_no_failure_counter_and_adds_no_state_keys(world):
    w = world
    assert w.run(TURN_A) is True and w.run(TURN_B) is True
    assert set(w.state) <= {'last_forest_id', 'primed_nodes'}   # the success path adds nothing to the caller-owned state


# ------------------------------------------------------------------ Exec Packet 491 (iii): a forest born UNBOUND is announced

def _born(caplog):
    return [r for r in caplog.records if r.name == org.logger.name and 'forest_born_unbound' in r.getMessage()]


def test_a_first_forest_with_no_predecessor_no_trees_no_windows_logs_one_born_unbound_warning(world, caplog):
    """FAILS on base (no such record). The turn is still True and the forest still exists (the existing contract): it is announced."""
    w = world
    w.concepts[TURN_A], w.windows[TURN_A] = [], []
    with caplog.at_level(logging.DEBUG, logger=org.logger.name):
        assert w.run(TURN_A) is True
    recs = _born(caplog)
    assert len(recs) == 1 and recs[0].levelno == logging.WARNING
    msg = recs[0].getMessage()
    assert 'reason=forest_born_unbound' in msg and 'has_predecessor=False has_trees=False has_windows=False' in msg
    for forbidden in (TEXT_CANARY, 'cc:conv::', 'TURN-A'):
        assert forbidden not in msg
    assert _forest_id(TURN_A) in w.graph.nodes and w.graph.synapses.__len__() == 0     # really born unbound


@pytest.mark.parametrize('what', ['predecessor', 'trees', 'windows'])
def test_no_born_unbound_warning_when_the_forest_has_a_predecessor_or_trees_or_windows(world, caplog, what):
    w = world
    if what == 'predecessor':
        assert w.run(TURN_A) is True                          # turn A has trees: it gives turn B a predecessor
        w.concepts[TURN_B], w.windows[TURN_B] = [], []
    elif what == 'trees':
        w.concepts[TURN_B], w.windows[TURN_B] = ['gamma concept one'], []
    else:
        w.concepts[TURN_B], w.windows[TURN_B] = [], ['B window one']
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger=org.logger.name):
        assert w.run(TURN_B) is True
    assert _born(caplog) == []


def test_a_repeat_of_an_existing_forest_is_not_born(world, caplog):
    w = world
    w.concepts[TURN_A], w.windows[TURN_A] = [], []
    assert w.run(TURN_A) is True
    w.state['last_forest_id'] = None                           # as after a restart that lost the pointer
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger=org.logger.name):
        assert w.run(TURN_A) is True                           # the SAME turn again: the forest already exists
    assert _born(caplog) == []


# ------------------------------------------------------------------ Exec Packet 491 (i): the pointer survives a daemon restart (real NG + this daemon)

@pytest.fixture
def restart_rig(world, tmp_path, monkeypatch):
    w = world
    for var in ('CC_NG_BATCH_SIZE', 'CC_NG_IDLE_STEPS', 'CC_NG_DRAIN_HOLD_ON_FAILURE'):
        monkeypatch.delenv(var, raising=False)
    _DAEMON_COUNT[0] += 1
    spec = importlib.util.spec_from_file_location('z12_904_daemon_%d' % _DAEMON_COUNT[0], DAEMON_ENV)
    d = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = d
    spec.loader.exec_module(d)
    ckpt = tmp_path / 'checkpoints'
    ckpt.mkdir()
    assert '/pytest-' in str(ckpt)
    # NOTHING of the live checkpoint dir is read or written: every path the save/restore touches is redirected into tmp
    monkeypatch.setattr(d, '_CONV_POINTER_FILE', str(ckpt / '.cc_conv_last_forest'), raising=False)
    monkeypatch.setattr(d, '_HEALTHY_REF_FILE', str(ckpt / '.healthy_node_count'))
    monkeypatch.setattr(d, 'CC_NG_WORKSPACE', str(tmp_path / 'ws'))
    saves = []
    fake_ng = types.SimpleNamespace(graph=w.graph, vector_db=w.vdb, auto_save_interval=0,
                                    save=lambda **kw: saves.append(1) or True)
    hook = types.ModuleType('openclaw_hook')
    hook.NeuroGraphMemory = types.SimpleNamespace(get_instance=lambda workspace_dir=None, config=None: fake_ng)
    monkeypatch.setitem(sys.modules, 'openclaw_hook', hook)
    for name in ('bootstrap_lenia', 'bootstrap_trisynaptic', 'get_cc_commons', 'cc_stamp_missing_geometry', 'bootstrap_cc_modules'):
        monkeypatch.setattr(org, name, lambda *a, **k: None)
    st1 = d.DaemonState()
    st1.ng, st1.conv_state = fake_ng, w.state
    monkeypatch.setattr(d, 'STATE', st1)

    def restart():
        """A daemon restart: a FRESH DaemonState (a fresh conv_state dict) whose init_ng restores from the persisted state."""
        st2 = d.DaemonState()
        monkeypatch.setattr(d, 'STATE', st2)
        st2.init_ng()
        w.state = st2.conv_state
        return st2

    return types.SimpleNamespace(d=d, w=w, saves=saves, ckpt=ckpt, restart=restart)


@needs_daemon
def test_restart_round_trip_the_sequence_link_survives_a_restart(restart_rig, caplog):
    """FAILS on base: with no persistence the first forest after the restart has no predecessor, so A -> B is never written."""
    r, w, d = restart_rig, restart_rig.w, restart_rig.d
    assert w.run(TURN_A) is True
    forest_a, forest_b = _forest_id(TURN_A), _forest_id(TURN_B)
    assert w.state['last_forest_id'] == forest_a
    assert d._guarded_save('test') is True and r.saves == [1]
    with caplog.at_level(logging.DEBUG):
        st2 = r.restart()
    assert st2.conv_state['last_forest_id'] == forest_a                    # restored, not inferred from the graph
    assert w.run(TURN_B) is True
    link = [s for s in w.graph.synapses.values() if s.pre_node_id == forest_a and s.post_node_id == forest_b]
    assert len(link) == 1 and round(float(link[0].weight), 6) == 0.2 and 2 <= int(link[0].delay) <= 5


@needs_daemon
def test_a_rolled_back_turn_never_moves_the_persisted_pointer(restart_rig, monkeypatch):
    """The pointer is never moved to a forest that was not written: a failed turn restores it, and the save persists the OLD forest."""
    r, w, d = restart_rig, restart_rig.w, restart_rig.d
    assert w.run(TURN_A) is True
    _inject(w, 'bind_hyperedge', monkeypatch)
    assert w.run(TURN_B) is False
    assert w.state['last_forest_id'] == _forest_id(TURN_A)
    assert d._guarded_save('test') is True
    assert json.loads((r.ckpt / '.cc_conv_last_forest').read_text())['last_forest_id'] == _forest_id(TURN_A)


# ------------------------------------------------------------------ #904 FOLD-UP (dispatch #13259)

class GetRaisesVDB(FaultyVDB):
    """A REAL SimpleVectorDB whose get() raises for chosen ids (the probe the rollback's fresh-entry tracking relies on)."""

    def __init__(self):
        super().__init__()
        self.get_raises = lambda _id: False

    def get(self, id):
        if self.get_raises(id):
            raise OSError(SECRET)
        return super().get(id)


def _node_state(w):
    return {n: (repr(node.metadata), node.threshold, node.intrinsic_excitability) for n, node in sorted(w.graph.nodes.items())}


def _vdb_state(w):
    return {i: (w.vdb.embeddings[i].tobytes(), w.vdb.content[i], repr(w.vdb.metadata[i])) for i in sorted(w.vdb.all_ids())}


def _age_every_node(w):
    """Make the stamp observable: an aged state (probation almost done, graduated, non-default threshold/excitability)."""
    for i, node in enumerate(w.graph.nodes.values()):
        node.metadata['probation_remaining'] = 2
        node.metadata['graduated'] = True
        node.threshold = 0.5 + i / 100.0
        node.intrinsic_excitability = 0.9 - i / 100.0


# ---- C3 / F11: the failed EXACT-REPEAT turn failing AFTER the hyperedge create (kills M20)

def test_a_failed_exact_repeat_failing_after_the_hyperedge_create_removes_that_hyperedge_and_every_synapse_it_created(world, monkeypatch, caplog):
    """Every member PRE-EXISTS (so remove_node's cascade can never mask a stray hyperedge: the existing cases all use a NEW forest).
    The call creates its hyperedge and then fails at the window chain: the rollback must remove that hyperedge and every synapse the
    call created, and leave every pre-existing node intact. KILLS M20 (rollback leaves the hyperedge)."""
    w = world
    assert w.run(TURN_A) is True
    before = snap(w)
    n_he_before, n_syn_before = len(w.graph.hyperedges), len(w.graph.synapses)
    created = []
    real_he, real_syn = w.graph.create_hyperedge, w.graph.create_synapse

    def create_hyperedge(*a, **k):
        he = real_he(*a, **k)
        created.append(he.hyperedge_id)
        return he

    def create_synapse(pre, post, *a, **k):
        if '::window::' in pre and '::window::' in post:
            raise RuntimeError(SECRET)
        return real_syn(pre, post, *a, **k)
    w.graph.create_hyperedge, w.graph.create_synapse = create_hyperedge, create_synapse
    with caplog.at_level(logging.DEBUG, logger=org.logger.name):
        assert w.run(TURN_A) is False                     # the SAME turn again: forest, trees and windows all pre-exist
    assert len(created) == 1, 'the hyperedge was created BEFORE the failure (so the test can see a stray one)'
    assert created[0] not in w.graph.hyperedges           # the rollback removed it
    assert len(w.graph.hyperedges) == n_he_before and len(w.graph.synapses) == n_syn_before
    assert snap(w) == before
    assert 'site=window_chain ' in _loud(caplog)[0].getMessage()


# ---- C4 / F1: a failed attempt leaves NO trace on a PRE-EXISTING node

@pytest.mark.parametrize('fault', ['bind_hyperedge', 'new_tree_vdb_insert'])
def test_a_held_exact_repeat_poison_turn_leaves_the_pre_existing_nodes_byte_identical_for_N_cycles(world, tmp_path, monkeypatch, caplog, fault):
    """FAILS on build-001 (the probation re-stamp of the forest and its trees happened every cycle). A held exact-repeat turn is retried
    each autosave cycle: after EVERY cycle the pre-existing nodes' full metadata (repr, so key ORDER too), threshold, excitability and
    their vdb entries (embedding bytes, content, metadata) are identical to before the first attempt; the tract stays byte-identical."""
    w = world
    assert w.run(TURN_A) is True
    _age_every_node(w)
    nodes_before, vdb_before, graph_before = _node_state(w), _vdb_state(w), snap(w)
    w.concepts[TURN_A] = [CONCEPTS_A[0], 'alpha concept three']       # a retry extracts DIFFERENT concepts: one new tree
    if fault == 'bind_hyperedge':
        w.graph.create_hyperedge = lambda *a, **k: (_ for _ in ()).throw(RuntimeError(SECRET))
    else:
        w.vdb.fail_ids = lambda i: i.endswith('::tree::alpha concept three')   # forest + tree 1 re-stamped, the new tree fails midway
    tr = Tract(tmp_path, [TURN_A])
    for cycle in range(4):
        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger=org.logger.name):
            absorbed, consumed = _drain(w, tr, True)
        assert (absorbed, consumed) == (0, b'') and tr.read() == b''.join(tr.frames), 'cycle %d' % cycle
        assert len(_hold_records(caplog)) == 1 and len(_loud(caplog)) == 1
        assert _node_state(w) == nodes_before, 'a pre-existing node was re-stamped by a FAILED attempt (cycle %d)' % cycle
        assert _vdb_state(w) == vdb_before, 'a pre-existing vdb entry was changed by a FAILED attempt (cycle %d)' % cycle
        assert snap(w) == graph_before


def test_a_pre_existing_vdb_entry_that_DIFFERS_from_what_the_deposit_writes_is_put_back_exactly(world):
    """Kills F4/F6 (the vdb-entry restore). A checkpoint-loaded entry has its OWN metadata dict, older content and a non-unit embedding;
    a failed repeat re-writes all three. The rollback must put back the ORIGINAL objects: the same metadata dict, the same content, the
    embedding array bit-for-bit (SimpleVectorDB.insert re-normalises, so a plain re-insert is NOT enough)."""
    w = world
    assert w.run(TURN_A) is True
    fid = _forest_id(TURN_A)
    old_emb = (np.arange(16, dtype=np.float32) + 3.0)            # NOT unit length: a re-insert would normalise it
    old_meta = {'loaded': 'from-checkpoint', 'cc': True}
    w.vdb.embeddings[fid], w.vdb.content[fid], w.vdb.metadata[fid] = old_emb, 'OLDER CONTENT', old_meta
    before = (w.vdb.embeddings[fid].tobytes(), w.vdb.content[fid], repr(w.vdb.metadata[fid]))
    w.graph.create_hyperedge = lambda *a, **k: (_ for _ in ()).throw(RuntimeError(SECRET))
    assert w.run(TURN_A) is False
    assert (w.vdb.embeddings[fid].tobytes(), w.vdb.content[fid], repr(w.vdb.metadata[fid])) == before
    assert w.vdb.embeddings[fid] is old_emb and w.vdb.metadata[fid] is old_meta   # the ORIGINAL objects, not copies


def test_a_successful_exact_repeat_still_restamps_as_before(world):
    """The success path is UNCHANGED (passes on build-001 and on base): a SUCCESSFUL repeat re-stamps the probation exactly as always."""
    w = world
    assert w.run(TURN_A) is True
    _age_every_node(w)
    assert w.run(TURN_A) is True
    forest = w.graph.nodes[_forest_id(TURN_A)]
    assert forest.metadata['probation_remaining'] == org._CC_CONV_PROBATION_PERIOD
    assert forest.threshold == w.graph.config.get('default_threshold', 1.0) + org._CC_CONV_THRESHOLD_BOOST


def test_restore_covers_a_node_deposited_twice_in_one_call(world, monkeypatch):
    """The same node written twice in one call (a duplicated concept): last-in-first-out restores the ORIGINAL state, not the
    first re-stamp."""
    w = world
    assert w.run(TURN_A) is True
    _age_every_node(w)
    nodes_before = _node_state(w)
    w.concepts[TURN_A] = [CONCEPTS_A[0], CONCEPTS_A[0]]               # the same tree twice
    w.graph.create_hyperedge = lambda *a, **k: (_ for _ in ()).throw(RuntimeError(SECRET))
    assert w.run(TURN_A) is False
    assert _node_state(w) == nodes_before


# ---- checker-038 correction 2: a raising vdb.get() must not orphan what the call inserted

def test_a_raising_vdb_get_for_a_NEW_node_leaves_no_orphan_after_the_rollback(world, monkeypatch, caplog):
    w = world
    w.vdb = GetRaisesVDB()
    w.vdb.get_raises = lambda i: True
    _inject(w, 'bind_hyperedge', monkeypatch)
    with caplog.at_level(logging.DEBUG, logger=org.logger.name):
        assert w.run(TURN_B) is False
    assert w.vdb.all_ids() == [] and w.graph.nodes == {}              # BUILD-001: the forest and tree entries stay as ORPHANS
    assert 'vdb_kept_unprovable=0' in _loud(caplog)[0].getMessage()


def test_a_raising_vdb_get_for_a_PRE_EXISTING_node_never_deletes_what_may_pre_exist_and_says_so(world, monkeypatch, caplog):
    w = world
    w.vdb = GetRaisesVDB()
    assert w.run(TURN_A) is True
    before = snap(w)
    w.vdb.get_raises = lambda i: True                                  # now the probe raises for a node that PRE-EXISTS
    w.graph.create_hyperedge = lambda *a, **k: (_ for _ in ()).throw(RuntimeError(SECRET))
    with caplog.at_level(logging.DEBUG, logger=org.logger.name):
        assert w.run(TURN_A) is False
    assert snap(w) == before                                           # nothing of the earlier success was deleted
    assert 'vdb_kept_unprovable=3;' in _loud(caplog)[0].getMessage()   # the forest and its two trees (windows are not indexed)


# ---- checker-038 correction 3: the bind COMPLETES, a LATER stage fails

def test_when_the_bind_completes_and_a_later_stage_fails_the_pointer_is_restored(world, monkeypatch, caplog):
    """The nine partial-write cases all raise BEFORE the bind assigns state['last_forest_id']. Here the REAL bind completes (the
    pointer is moved to the new forest and primed_nodes replaced) and the post-condition then fails: both are restored."""
    w = world
    assert w.run(TURN_A) is True
    primed_before = w.state.get('primed_nodes')
    moved = []
    real = org._cc_bind_conversational_topology

    def bind(graph, forest_id, result, emb, state, window_ids=None, journal=None):
        status = real(graph, forest_id, result, emb, state, window_ids=window_ids, journal=journal)
        moved.append((state['last_forest_id'], state.get('primed_nodes') is not primed_before))
        return {**status, 'hyperedge': False}
    monkeypatch.setattr(org, '_cc_bind_conversational_topology', bind)
    with caplog.at_level(logging.DEBUG, logger=org.logger.name):
        assert w.run(TURN_B) is False
    assert moved == [(_forest_id(TURN_B), True)]                       # the bind really moved the pointer first
    assert w.state['last_forest_id'] == _forest_id(TURN_A) and w.state.get('primed_nodes') is primed_before


@needs_daemon
def test_when_the_bind_completes_and_a_later_stage_fails_the_persisted_sidecar_is_not_advanced(restart_rig, monkeypatch):
    r, w, d = restart_rig, restart_rig.w, restart_rig.d
    assert w.run(TURN_A) is True and d._guarded_save('test') is True
    real = org._cc_bind_conversational_topology
    monkeypatch.setattr(org, '_cc_bind_conversational_topology',
                        lambda *a, **k: {**real(*a, **k), 'hyperedge': False})
    assert w.run(TURN_B) is False
    assert d._guarded_save('test') is True
    assert json.loads((r.ckpt / '.cc_conv_last_forest').read_text())['last_forest_id'] == _forest_id(TURN_A)


# ------------------------------------------------------------------ #904 ROUND 2 (dispatch #13437): le-053 F1-1..F1-4

# ---- F1-2: the restore is NARROW (exactly what the failed call wrote), and conditional

def test_a_probation_sweep_between_the_snapshot_and_the_rollback_is_not_erased(world):
    """FAILS on build-002 (the whole-dict restore erased the sweep's changes). The sweep (cc_update_probation) takes the same lock
    on the autosave pulse and rewrites these fields on a node the dual pass does not own at that moment. Simulated INSIDE the failing
    call, after every deposit and before the rollback: the rollback must leave the sweep's changes and every key the deposit never
    wrote, and still restore the fields the deposit wrote and nobody touched since."""
    w = world
    assert w.run(TURN_A) is True
    _age_every_node(w)
    fid = _forest_id(TURN_A)
    f = w.graph.nodes[fid]
    f.metadata['probation_total'], f.metadata['novelty_dampening'], f.metadata['poincare_dir'] = 99, 0.77, b'ORIGINAL-POINCARE'

    def sweep_then_fail(*a, **k):
        with w.graph._step_lock:                                   # what the sweep does, under the same lock
            f.threshold, f.intrinsic_excitability = 0.123, 0.456
            f.metadata['probation_remaining'] = 7
            f.metadata['graduated'] = False                        # a key the deposit NEVER writes
            f.metadata['probation_expired_unfired'] = True         # a key that did not exist before
        raise RuntimeError(SECRET)
    w.graph.create_hyperedge = sweep_then_fail
    assert w.run(TURN_A) is False
    assert (f.threshold, f.intrinsic_excitability, f.metadata['probation_remaining']) == (0.123, 0.456, 7)   # the sweep's, kept
    assert f.metadata['graduated'] is False and f.metadata['probation_expired_unfired'] is True
    assert f.metadata['probation_total'] == 99 and f.metadata['novelty_dampening'] == 0.77        # the call's writes, nobody touched: restored
    assert f.metadata['poincare_dir'] == b'ORIGINAL-POINCARE'


def test_a_key_the_failed_call_added_is_deleted_and_the_key_order_is_preserved(world):
    w = world
    assert w.run(TURN_A) is True
    f = w.graph.nodes[_forest_id(TURN_A)]
    del f.metadata['novelty_dampening'], f.metadata['probation_total']        # absent BEFORE the call
    before = repr(f.metadata)
    w.graph.create_hyperedge = lambda *a, **k: (_ for _ in ()).throw(RuntimeError(SECRET))
    assert w.run(TURN_A) is False
    assert 'novelty_dampening' not in f.metadata and 'probation_total' not in f.metadata
    assert repr(f.metadata) == before                                          # byte-identical, order included


# ---- F1-3: the restore's field list is TIED to _cc_deposit_memory_node's write-set

def test_the_restores_field_list_is_tied_to_the_deposits_write_set():
    """Fails if _cc_deposit_memory_node starts writing an attribute or a metadata key the restore does not cover (or any write form
    this test cannot classify). `metadata.update(meta)` is covered generically (every key of the call's meta is snapshotted)."""
    tree = ast.parse(inspect.getsource(org._cc_deposit_memory_node).lstrip())
    attrs, keys, unknown = set(), set(), []
    for n in ast.walk(tree):
        targets = n.targets if isinstance(n, ast.Assign) else [n.target] if isinstance(n, (ast.AugAssign, ast.AnnAssign)) else []
        for t in targets:
            if isinstance(t, ast.Name):
                continue                                                      # a local
            if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == 'node':
                attrs.add(t.attr)
            elif isinstance(t, ast.Subscript) and ast.unparse(t.value) == 'node.metadata' and isinstance(t.slice, ast.Constant):
                keys.add(t.slice.value)
            else:
                unknown.append(ast.unparse(t))
    calls = [ast.unparse(c) for c in ast.walk(tree) if isinstance(c, ast.Call) and ast.unparse(c.func).startswith('node.')]
    assert calls == ['node.metadata.update(meta)'], calls
    assert unknown == [], unknown
    eco = org._CCConversationalDualPassEco
    assert attrs == set(eco._STAMPED_ATTRS), (attrs, eco._STAMPED_ATTRS)
    assert keys == set(eco._STAMPED_META_KEYS), (keys, eco._STAMPED_META_KEYS)


# ---- F1-1: the verdict sentence must not contradict vdb_kept_unprovable

@pytest.mark.parametrize('concepts,windows,n,noun', [(CONCEPTS_A, ['A window one', 'A window two'], 3, 'entries'),
                                                      ([], ['S window one'], 1, 'entry')])
def test_the_rolled_back_sentence_says_what_was_KEPT_when_a_vdb_entry_could_not_be_proven(world, caplog, concepts, windows, n, noun):
    """FAILS on build-002 (it said "rolled back, nothing of this turn remains" beside vdb_kept_unprovable=n>0)."""
    w = world
    w.vdb = GetRaisesVDB()
    w.concepts[TURN_A], w.windows[TURN_A] = list(concepts), list(windows)
    assert w.run(TURN_A) is True
    w.vdb.get_raises = lambda i: True
    w.graph.create_hyperedge = lambda *a, **k: (_ for _ in ()).throw(RuntimeError(SECRET))
    with caplog.at_level(logging.DEBUG, logger=org.logger.name):
        assert w.run(TURN_A) is False
    rec = _loud(caplog)[0]
    msg = rec.getMessage()
    assert rec.levelno == logging.WARNING                                    # the rollback itself succeeded
    assert 'rolled back; kept: %d pre-existing vdb %s that could not be proven' % (n, noun) in msg
    assert 'nothing of this turn remains' not in msg
    assert 'vdb_kept_unprovable=%d;' % n in msg                              # the earlier substring is unchanged


def test_the_nothing_remains_sentence_is_kept_when_nothing_was_kept(world, monkeypatch, caplog):
    w = world
    _inject(w, 'bind_hyperedge', monkeypatch)
    with caplog.at_level(logging.DEBUG, logger=org.logger.name):
        assert w.run(TURN_B) is False
    msg = _loud(caplog)[0].getMessage()
    assert 'rolled back, nothing of this turn remains' in msg and 'kept:' not in msg


# ---- F1-4: the unrestorable count is printed

def test_the_unrestorable_count_is_printed_and_a_snapshot_that_cannot_be_taken_is_an_error(world, monkeypatch, caplog):
    w = world
    assert w.run(TURN_A) is True
    w.graph.create_hyperedge = lambda *a, **k: (_ for _ in ()).throw(RuntimeError(SECRET))
    with caplog.at_level(logging.DEBUG, logger=org.logger.name):
        assert w.run(TURN_A) is False
    assert 'unrestorable=0;' in _loud(caplog)[0].getMessage()                # printed even when 0
    monkeypatch.setattr(org._CCConversationalDualPassEco, '_STAMPED_ATTRS', ('threshold', 'no_such_attribute'))
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger=org.logger.name):
        assert w.run(TURN_A) is False
    rec = _loud(caplog)[0]
    assert rec.levelno == logging.ERROR and 'ROLLBACK FAILED' in rec.getMessage()
    assert 'unrestorable=5;' in rec.getMessage()                             # the forest, 2 trees and 2 windows: 5 pre-existing nodes


# ------------------------------------------------------------------ Exec Packet 489: the want seed-synapse swallow is LOUD

def _want_world():
    g, v, _ = _new_stores()
    return g, v


def _seed_source(g, v, nid, text):
    g.create_node(node_id=nid, metadata={'creation_mode': 'conversational'})
    v.insert(nid, _vec(text), text, {})


def _spy_synapse(g, fail_when):
    real = g.create_synapse

    def create_synapse(pre, post, *a, **k):
        if fail_when(pre, post):
            raise KeyError(SECRET)
        return real(pre, post, *a, **k)
    g.create_synapse = create_synapse


WANT_TEXT = 'learn the thing WANTCANARY904'
WANT_FNS = [('surface_wants', 'cc:want::'), ('surface_wants_for_graph', 'want::')]


def _wants(fn_name, g, v):
    return getattr(org, fn_name)(g, v)


@pytest.mark.parametrize('fn_name,prefix', WANT_FNS)
def test_a_failed_want_seed_synapse_is_a_warning_and_the_want_is_kept(fn_name, prefix, caplog):
    """FAILS on base (silent `pass`: no record above debug); on the fix ONE WARNING with the fixed reason code, the counts and the
    exception CLASS only. The want is still created and still returned (the existing contract), unbound, never rolled back (H-1)."""
    g, v = _want_world()
    _seed_source(g, v, 'cc:conv::src', 'turn text [WANT]%s[/WANT] tail' % WANT_TEXT)
    _spy_synapse(g, lambda pre, post: post.startswith(prefix))
    with caplog.at_level(logging.DEBUG, logger=org.logger.name):
        out = _wants(fn_name, g, v)
    assert [w_['text'] for w_ in out] == [WANT_TEXT]                    # reported exactly as before
    want_id = out[0]['id']
    assert want_id in g.nodes and g.nodes[want_id].metadata['provenance'] == 'cc_authored'
    assert g.nodes[want_id].metadata['want_state'] == 'open' and g.nodes[want_id].metadata['source_node'] == 'cc:conv::src'
    assert not g._incoming[want_id] and not g._outgoing[want_id]       # unbound: no synapse
    loud = [r for r in caplog.records if r.name == org.logger.name and r.levelno >= logging.WARNING]
    assert len(loud) == 1 and loud[0].levelno == logging.WARNING
    msg = loud[0].getMessage()
    for need in ('reason=want_source_synapse_failed', 'fn=%s ' % fn_name, 'attempted=1', 'failed=1', 'exc_types=KeyError'):
        assert need in msg, (need, msg)
    for forbidden in (SECRET, 'WANTCANARY904', 'learn the thing', 'cc:conv::', want_id):
        assert forbidden not in msg, forbidden
    # a want that exists is skipped on the next pulse: the failure is reported ONCE, not per autosave
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger=org.logger.name):
        again = _wants(fn_name, g, v)
    assert [w_['id'] for w_ in again] == [want_id]
    assert [r for r in caplog.records if r.name == org.logger.name and r.levelno >= logging.WARNING] == []


@pytest.mark.parametrize('fn_name,prefix', WANT_FNS)
def test_want_synapse_counts_are_per_call_attempted_and_failed(fn_name, prefix, caplog):
    g, v = _want_world()
    _seed_source(g, v, 'cc:conv::s1', 'a [WANT]first want body[/WANT] b')
    _seed_source(g, v, 'cc:conv::s2', 'c [WANT]second want body[/WANT] d')
    _spy_synapse(g, lambda pre, post: pre == 'cc:conv::s2' and post.startswith(prefix))   # only the second source fails
    with caplog.at_level(logging.DEBUG, logger=org.logger.name):
        out = _wants(fn_name, g, v)
    assert len(out) == 2
    loud = [r for r in caplog.records if r.name == org.logger.name and r.levelno >= logging.WARNING]
    assert len(loud) == 1
    assert 'attempted=2' in loud[0].getMessage() and 'failed=1' in loud[0].getMessage()


@pytest.mark.parametrize('fn_name,prefix', WANT_FNS)
def test_want_success_path_is_golden_identical_and_quiet(fn_name, prefix, caplog):
    """Passes on base and fix: same return, same node, the same ONE synapse (source -> want, weight 0.3), no record above debug."""
    g, v = _want_world()
    _seed_source(g, v, 'cc:conv::src', 'turn text [WANT]%s[/WANT] tail' % WANT_TEXT)
    with caplog.at_level(logging.DEBUG, logger=org.logger.name):
        out = _wants(fn_name, g, v)
    want_id = prefix + hashlib.sha1(WANT_TEXT.encode()).hexdigest()[:16]
    assert out == [{'id': want_id, 'text': WANT_TEXT, 'provenance': 'cc_authored', 'state': 'open', 'source': 'cc:conv::src'}]
    assert [(s.pre_node_id, s.post_node_id, round(float(s.weight), 6)) for s in g.synapses.values()] == [('cc:conv::src', want_id, 0.3)]
    assert [r for r in caplog.records if r.name == org.logger.name and r.levelno >= logging.WARNING] == []


def test_generate_emergent_want_is_untouched():
    """#905 owns it: #904 must not alter generate_emergent_want. Its source must be byte-identical to #905's.

    N8 (NG trial integration): this used to compare against the PRE-#905 base (c5684334), which cannot hold once #905
    part A (the born-bound want) is in the same tree, in either merge order. RE-BASED, not retired: the function is now
    compared against #905's own version (ee94f7d2, read via the same `git show` mechanism), so what #904 owed is still
    pinned -- #904 itself adds nothing to this function beyond what #905 made. The 'new' side is the MODULE UNDER TEST
    (org.__file__; the worktree file unless Z12_904_ORG_UNDER_TEST points at a scratch copy)."""
    import subprocess
    new = Path(org.__file__).read_text()

    def body(src):
        start = src.index('def generate_emergent_want(')
        end = src.index('\ndef ', start + 10)
        return src[start:end]
    ref = subprocess.run(['git', '-C', str(_WORKTREE), 'show', 'ee94f7d2c516cce40ff81bd32e4febb9781e8e04:cc_ng_organism.py'],
                         capture_output=True, text=True)
    assert ref.returncode == 0 and ref.stdout, 'cannot read #905 (ee94f7d2) cc_ng_organism.py: %s' % ref.stderr.strip()
    assert body(new) == body(ref.stdout)
