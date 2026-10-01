# ---- Changelog ----
# [2026-10-01] Z12 builder (Claude Sonnet 5.5), lane poison-quarantine-920, dispatch #13417 (+ ADDENDUM 2 / Exec P501: the ERROR text and the
#   docstring state `attempts = max_retries + 1` in words; test_the_quarantine_error_states_attempts_equal_retries_plus_one asserts it)
# [2026-10-01] Z12 builder (Claude Sonnet 5.5), lane poison-quarantine-920, dispatch #13417 — #920 build-002 (Chief-003 RULINGS, ADDENDUM 1):
#   COUNT SEMANTICS = THE LITERAL NAME: max_retries is the number of RETRIES AFTER THE FIRST FAILED ATTEMPT, so an entry gets max_retries + 1
#   attempts (_attempts(n) below) and the (max_retries + 1)-th consecutive failure is quarantined in that cycle. Every test that encoded the old
#   boundary (the Nth failure quarantines) moved to the new one; none was weakened (the max_retries=1 special case of the first test is gone
#   because the boundary no longer lets T1 and the quarantine share a call, so its assertion is now uniform); NEW: the boundary pin, the
#   N-failures-then-success pin, the literal WARNING / ERROR number pins, the docstring pin (formula + the RAISE-the-variable lever).
# [2026-10-01] Z12 builder (Claude Sonnet 5.5), lane poison-quarantine-920, dispatch #13303 — #920 (Exec Packet 497 via Chief-003)
# What: tests for the POISON-TURN POLICY on drain_ingest_tract (bounded retry -> QUARANTINE, never discard -> LOUD): a turn that fails
#   every cycle is retried exactly max_retries times and then quarantined WHOLE (exact raw frame bytes) to the append-only
#   `<tract>.quarantine` with fsync BEFORE the tract advances, the entries behind it flow, one hash-form ERROR, the counter; a turn that
#   fails twice then succeeds is not quarantined; a failed quarantine write keeps the entry HELD; a crash between the quarantine fsync and
#   the truncate neither duplicates nor loses; max_retries=None / hold_on_failure=False are byte-identical to the #904 tip (differential
#   against the BASE module, live); the table is hard-capped; the quarantine file is never created when nothing is quarantined and is
#   never truncated across calls; and (gated on Z12_920_DAEMON_UNDER_TEST) the REAL docs daemon loop on REAL NG.
# Why: Exec Packet 497 (row #920): with hold_on_failure a turn that can NEVER succeed blocks the whole tract forever. LAW 7 / P370.
# How: the #904 test_805_proof_* pattern: a REAL in-memory neuro_foundation.Graph + REAL SimpleVectorDB + the REAL (vendored, unedited)
#   ng_embed.NGEmbed.dual_record_outcome with ONLY its model/TID seams replaced on the instance (deterministic hash vectors, scripted
#   concepts); the drain is the REAL drain_ingest_tract on REAL tract bytes (ng_tract writer) in pytest tmp files. The LIVE tract
#   (~/.claude/plugins/neurograph/...) and any checkpoint are never opened: every tract path is asserted to be a pytest tmp path.
#   Z12_920_ORG_UNDER_TEST points the module under test at a SCRATCH copy of cc_ng_organism.py (the BASE or a mutation) so this one file
#   runs against the base (failure counts) and against each mutation. P379/#770: the preamble prints every module path and the session
#   FAILS if an NG module (other than a scratch org under /tmp/z12-920-scratch) resolves outside this worktree.
# -------------------
import hashlib
import importlib.util
import json
import logging
import os
import random
import re
import subprocess
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

BASE_SHA = '895a809a60ab86afacd8ef193b95c4ab39f6893f'    # the #904 NG tip this build starts from
_SCRATCH_ROOT = Path('/tmp/z12-920-scratch').resolve()
_ORG_ALT = os.environ.get('Z12_920_ORG_UNDER_TEST')
if _ORG_ALT:
    _spec = importlib.util.spec_from_file_location('cc_ng_organism', _ORG_ALT)
    org = importlib.util.module_from_spec(_spec)
    sys.modules['cc_ng_organism'] = org
    _spec.loader.exec_module(org)
else:
    import cc_ng_organism as org  # noqa: E402

DAEMON_ENV = os.environ.get('Z12_920_DAEMON_UNDER_TEST')

_PREAMBLE = ['[P379/#770] worktree root           -> %s' % _WORKTREE]
for _m in (org, ng_embed, neuro_foundation):
    _PREAMBLE.append('[P379/#770] %-20s -> %s' % (_m.__name__, Path(_m.__file__).resolve()))
_PREAMBLE.append('[P379/#770] ng_tract (site, Rust)  -> %s' % getattr(ng_tract, '__file__', ng_tract))
_PREAMBLE.append('[P379/#770] daemon under test      -> %s' % (DAEMON_ENV or '(none: the daemon-loop test is skipped)'))
_PREAMBLE.append('[P379/#770] org under test is a scratch copy: %s' % bool(_ORG_ALT))
_PREAMBLE.append('[P379/#770] HOME=%s' % os.environ.get('HOME'))
sys.__stderr__.write('\n' + '\n'.join(_PREAMBLE) + '\n')


def _inside_worktree(mod):
    return _WORKTREE in Path(mod.__file__).resolve().parents


for _m in (ng_embed, neuro_foundation):
    if not _inside_worktree(_m):
        raise RuntimeError('P379/#770 FAIL: %s resolves outside the worktree %s' % (_m.__name__, _WORKTREE))
if _ORG_ALT:
    if _SCRATCH_ROOT not in Path(org.__file__).resolve().parents:
        raise RuntimeError('P379/#770 FAIL: a scratch org must live under %s, got %s' % (_SCRATCH_ROOT, org.__file__))
elif not _inside_worktree(org):
    raise RuntimeError('P379/#770 FAIL: cc_ng_organism resolves outside the worktree %s' % _WORKTREE)


@pytest.fixture(scope='session', autouse=True)
def _preamble_visible(request):
    tr = request.config.pluginmanager.get_plugin('terminalreporter')
    if tr is not None:
        tr.write_line('')
        for line in _PREAMBLE:
            tr.write_line(line)


TEXT_CANARY = 'TEXTCANARY920'
SECRET = 'SECRET-EXC-TEXT-920'
T1 = 'TURN-T1 first %s' % TEXT_CANARY
TP = 'TURN-POISON second %s' % TEXT_CANARY
T3 = 'TURN-T3 third %s' % TEXT_CANARY
TP2 = 'TURN-POISON-TWO fourth %s' % TEXT_CANARY
POISON_TREE_WORD = 'poisonconceptwordB'
CONCEPTS = {
    T1: ['first concept one', 'first concept two'],
    TP: ['poison concept A', POISON_TREE_WORD],
    T3: ['third concept one', 'third concept two'],
    TP2: ['poison two concept A', 'poison two concept B'],
}
WINDOWS = {T1: ['T1 window one', 'T1 window two'], TP: ['P window one', 'P window two'],
           T3: ['T3 window one', 'T3 window two'], TP2: ['P2 window one', 'P2 window two']}
QUAR_MARK = 'CC ingest-tract QUARANTINED'
QFAIL_MARK = 'CC ingest-tract QUARANTINE FAILED'
HOLD_PREFIX = 'CC ingest-tract hold'


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


def _poison(vdb, *words):
    """Make the dual pass of any turn whose tree concept ends with one of `words` fail partway (it rolls back, returns False)."""
    vdb.fail_ids = lambda i: any(i.endswith('::tree::' + w) for w in words)


@pytest.fixture
def world(monkeypatch):
    w = types.SimpleNamespace()
    w.graph, w.vdb, w.state = _new_stores()
    monkeypatch.setattr(random, 'randint', lambda a, b: 3)   # the bind's delays are random: pin them
    inst = ng_embed.NGEmbed()
    inst._extract_concepts = lambda text: list(CONCEPTS.get(text, []))
    inst.embed_batch = lambda concepts, **kw: [_vec(c) for c in concepts]
    inst.embed_windows = lambda text, **kw: types.SimpleNamespace(
        windows=[types.SimpleNamespace(text=t, embedding=_vec(t)) for t in WINDOWS.get(text, [])])
    monkeypatch.setattr(ng_embed.NGEmbed, 'get_instance', classmethod(lambda cls, config=None: inst))
    monkeypatch.setattr(ng_embed, 'embed', _vec)

    def clean_pass(texts):
        g, v, st = _new_stores()
        for t in texts:
            assert org.run_conversational_dual_pass(g, v, t, _vec(t), st) is True
        return types.SimpleNamespace(graph=g, vdb=v, state=st)

    w.clean_pass = clean_pass
    return w


def snap(w):
    """Everything a dual pass can leave in the graph / vdb / state, comparable across runs."""
    g = w.graph
    return {
        'nodes': sorted(g.nodes),
        'synapses': sorted((s.pre_node_id, s.post_node_id, round(float(s.weight), 6), int(s.delay)) for s in g.synapses.values()),
        'hyperedges': sorted((tuple(sorted(h.member_nodes)), h.metadata.get('creation_mode')) for h in g.hyperedges.values()),
        'vdb': sorted(w.vdb.all_ids()),
        'timestep': int(g.timestep),
        'last_forest_id': w.state.get('last_forest_id'),
    }


def _frame_bytes(tmp_path, tag, text):
    p = str(tmp_path / ('frame_%s.tract' % tag))
    if os.path.exists(p):
        os.remove(p)
    ng_tract.deposit_experience(raw=text.encode(), source='cc_gateway', tract_paths=[p])
    with open(p, 'rb') as f:
        return f.read()


class Tract:
    """A REAL tract file (real ng_tract frames) in a pytest tmp dir. Never the live tract."""

    def __init__(self, tmp_path, texts, name='turns.tract', frames=None):
        self.path = str(tmp_path / name)
        self.tmp = tmp_path
        # `frames` lets two tracts start from IDENTICAL bytes (a frame embeds a per-deposit stamp, so rebuilding differs)
        self.frames = list(frames) if frames is not None else [_frame_bytes(tmp_path, '%s_%d' % (name, i), t) for i, t in enumerate(texts)]
        self.rewrite(b''.join(self.frames))

    @property
    def qpath(self):
        return self.path + '.quarantine'

    def rewrite(self, data):
        assert 'plugins/neurograph' not in self.path and '/pytest-' in self.path, 'SAFETY: not a pytest tmp tract: %s' % self.path
        with open(self.path, 'wb') as f:
            f.write(data)

    def read(self):
        with open(self.path, 'rb') as f:
            return f.read()

    def qread(self):
        with open(self.qpath, 'rb') as f:
            return f.read()

    def qexists(self):
        return os.path.exists(self.qpath)


def _attempts(max_retries):
    """max_retries = RETRIES AFTER the first failed attempt (the literal name): an entry gets this many attempts in total."""
    return max_retries + 1


def _drain(w, tr, max_retries=None, hold=True, **kw):
    return org.drain_ingest_tract(w.graph, w.vdb, w.state, tract_path=tr.path, return_consumed=True,
                                  hold_on_failure=hold, max_retries=max_retries, **kw)


def _recs(caplog, prefix):
    return [r for r in caplog.records if r.name == org.logger.name and r.getMessage().startswith(prefix)]


def _tag(frame):
    return 'entry:%s' % hashlib.sha256(frame).hexdigest()[:12]


def _setup(w, tmp_path, texts=(T1, TP, T3)):
    _poison(w.vdb, POISON_TREE_WORD, 'poison two concept B')
    return Tract(tmp_path, list(texts))


# ------------------------------------------------------------------ pins (pass on the fix; the signature pin FAILS on base)

def test_signature_is_pinned():
    import inspect
    params = inspect.signature(org.drain_ingest_tract).parameters
    assert list(params)[-2:] == ['hold_on_failure', 'max_retries']
    assert params['hold_on_failure'].default is False and params['max_retries'].default is None


def test_vendored_and_protected_files_are_unedited():
    """ng_embed.py is VENDORED (LAW 2) and four files are protected: this build must not have touched any of them."""
    out = subprocess.run(['git', '-C', str(_WORKTREE), 'diff', '--name-only', BASE_SHA, '--', 'ng_embed.py', 'neuro_foundation.py',
                          'openclaw_hook.py', 'stream_parser.py', 'activation_persistence.py'], capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout.strip() == ''


# ------------------------------------------------------------------ THE POLICY, end to end on real tract bytes

@pytest.mark.parametrize('max_retries', [1, 3, 5])
def test_a_turn_that_fails_every_cycle_is_retried_max_retries_times_then_quarantined_whole(world, tmp_path, caplog, max_retries):
    """T1 ok, TP poison (its dual pass fails partway EVERY cycle, then rolls back), T3 ok. max_retries = RETRIES after the first failed
    attempt, so TP gets max_retries + 1 attempts: cycles 1..max_retries hold it (tract byte-identical, one hold WARNING each, nothing
    quarantined); the (max_retries + 1)-th failure quarantines it in that same cycle: the quarantine file is EXACTLY TP's raw frame, the
    tract no longer holds it, one ERROR, the counter is 1, no node/synapse of TP is in the graph. The next cycle drains T3. Nothing is
    discarded."""
    w = world
    tr = _setup(w, tmp_path)
    total = b''.join(tr.frames)
    removed = b''
    graph_after_t1 = None
    attempts = _attempts(max_retries)
    for cycle in range(1, attempts + 1):
        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger=org.logger.name):
            absorbed, consumed = _drain(w, tr, max_retries)
        removed += consumed
        if cycle < attempts:
            held = (tr.frames[1] + tr.frames[2])
            assert tr.read() == held, 'cycle %d: the held entry and everything behind it stay byte-identical' % cycle
            assert not tr.qexists()
            assert len(_recs(caplog, HOLD_PREFIX)) == 1 and _recs(caplog, QUAR_MARK) == []
            assert 'attempt=%d of %d (max_retries=%d)' % (cycle, attempts, max_retries) in _recs(caplog, HOLD_PREFIX)[0].getMessage()
            assert w.state['drain_retry_counts'] == {hashlib.sha256(tr.frames[1]).hexdigest(): cycle}
            assert w.state.get('quarantined_total', 0) == 0
            assert absorbed == (1 if cycle == 1 else 0) and consumed == (tr.frames[0] if cycle == 1 else b'')
            if graph_after_t1 is None:
                graph_after_t1 = snap(w)
            assert snap(w) == graph_after_t1, 'the failing turn leaked into the graph on cycle %d' % cycle
    # the (max_retries + 1)-th failure: quarantined in that same call
    assert tr.qread() == tr.frames[1], 'the quarantine file holds the EXACT raw frame bytes, nothing else'
    assert tr.read() == tr.frames[2], 'the tract advanced past the quarantined entry; the entry behind it is still there'
    assert (absorbed, consumed) == (0, tr.frames[1])
    errs = _recs(caplog, QUAR_MARK)
    assert len(errs) == 1 and errs[0].levelno == logging.ERROR and _recs(caplog, QFAIL_MARK) == []
    assert _recs(caplog, HOLD_PREFIX) == []                                   # the quarantining cycle logs the ERROR, not a hold WARNING
    msg = errs[0].getMessage()
    assert _tag(tr.frames[1]) in msg and 'failures=%d' % attempts in msg and 'max_retries=%d' % max_retries in msg
    assert 'quarantining after %d attempts (= max_retries + 1), max_retries=%d' % (attempts, max_retries) in msg, 'BOTH numbers + the relation'
    assert 'reason=absorb_returned_false' in msg and 'turns.tract.quarantine' in msg and 'stored=appended' in msg
    assert w.state['quarantined_total'] == 1 and w.state['drain_retry_counts'] == {}
    assert snap(w) == snap(w.clean_pass([T1])), 'no node/synapse of the poison turn remains (the #904 rollback)'
    # the entry behind it flows on the next cycle
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger=org.logger.name):
        a2, c2 = _drain(w, tr, max_retries)
    removed += c2
    assert (a2, c2) == (1, tr.frames[2]) and tr.read() == b''
    assert snap(w) == snap(w.clean_pass([T1, T3]))
    assert _recs(caplog, QUAR_MARK) == [] and w.state['quarantined_total'] == 1
    # NOTHING DISCARDED: every byte that ever left the tract is either absorbed or in the quarantine file
    assert removed == total and tr.qread() == tr.frames[1]


def test_the_quarantine_error_carries_only_the_hash_form_class_counts_and_file_name(world, tmp_path, caplog):
    """LAW 7 / #907: no text, no raw id, no tree concept word, no path, no exception text in the quarantine ERROR."""
    w = world
    tr = _setup(w, tmp_path)
    with caplog.at_level(logging.DEBUG, logger=org.logger.name):
        for _ in range(_attempts(2)):
            _drain(w, tr, 2)
    errs = _recs(caplog, QUAR_MARK)
    assert len(errs) == 1
    msg = errs[0].getMessage()
    for forbidden in (TEXT_CANARY, TP, POISON_TREE_WORD, 'poison concept', 'cc:conv::', '::tree::', SECRET, str(tmp_path), '/pytest-',
                      'TURN-'):
        assert forbidden not in msg, forbidden
    assert errs[0].args and all(not isinstance(a, bytes) for a in errs[0].args)
    # the entry tag is the 12-hex sha256 of the raw frame and nothing longer
    assert re.findall(r'entry:([0-9a-f]+)', msg) == [_tag(tr.frames[1])[len('entry:'):]]
    assert re.findall(r'[0-9a-f]{13,}', msg) == [], 'no long hex run (a full hash or an id) anywhere in the record'


def test_a_raising_dual_pass_is_quarantined_with_its_exception_class_name(world, tmp_path, caplog, monkeypatch):
    w = world
    tr = Tract(tmp_path, [T1, TP, T3])
    real = org.run_conversational_dual_pass

    def boom(graph, vdb, text, emb, state):
        if text == TP:
            raise ValueError(SECRET + ' ' + text)
        return real(graph, vdb, text, emb, state)

    monkeypatch.setattr(org, 'run_conversational_dual_pass', boom)
    with caplog.at_level(logging.DEBUG, logger=org.logger.name):
        for _ in range(_attempts(2)):
            _drain(w, tr, 2)
    msg = _recs(caplog, QUAR_MARK)[0].getMessage()
    assert 'reason=absorb_raised' in msg and 'exc_type=ValueError' in msg
    assert SECRET not in msg and TEXT_CANARY not in msg
    assert tr.qread() == tr.frames[1] and tr.read() == tr.frames[2]


def test_poison_as_the_very_first_entry_is_quarantined_and_the_tract_advances(world, tmp_path, caplog):
    """Nothing precedes the poison entry: the old 'never got past the first entry' early return must not strand it."""
    w = world
    tr = _setup(w, tmp_path, texts=(TP, T3))
    with caplog.at_level(logging.DEBUG, logger=org.logger.name):
        for _ in range(2):                                           # max_retries=2: attempts 1 and 2 are held
            a, c = _drain(w, tr, 2)
            assert (a, c) == (0, b'') and tr.read() == b''.join(tr.frames)
        a3, c3 = _drain(w, tr, 2)                                    # the 3rd (= max_retries + 1) failure quarantines it
    assert (a3, c3) == (0, tr.frames[0]) and tr.read() == tr.frames[1] and tr.qread() == tr.frames[0]
    a4, c4 = _drain(w, tr, 2)
    assert (a4, c4) == (1, tr.frames[1]) and tr.read() == b''


def test_fsync_happens_before_the_tract_is_truncated(world, tmp_path, monkeypatch):
    """Spy ordering: at EVERY os.fsync of the quarantine step the tract still holds the poison frame (not yet truncated) and the
    quarantine file already holds it (flushed). The truncate happens after the last fsync."""
    w = world
    tr = _setup(w, tmp_path)
    seen = []
    real_fsync = os.fsync

    def spy(fd):
        seen.append({'tract_has_poison': tr.frames[1] in tr.read(), 'tract_len': len(tr.read()),
                     'q': tr.qread() if tr.qexists() else None})
        return real_fsync(fd)

    for _ in range(2):                                              # max_retries=2: attempts 1 and 2 are held (no fsync spied yet)
        _drain(w, tr, 2)
    monkeypatch.setattr(os, 'fsync', spy)
    _drain(w, tr, 2)                                                # the 3rd attempt quarantines
    assert seen, 'the quarantine step never called os.fsync'
    assert all(s['tract_has_poison'] for s in seen), 'the tract was truncated BEFORE an fsync of the quarantine step'
    assert seen[0]['q'] == tr.frames[1], 'the frame was flushed into the quarantine file before its fsync'
    assert tr.frames[1] not in tr.read() and tr.qread() == tr.frames[1]          # and afterwards the tract did advance


def test_a_turn_that_fails_twice_then_succeeds_is_not_quarantined(world, tmp_path, caplog):
    w = world
    tr = _setup(w, tmp_path)
    for _ in range(2):
        _drain(w, tr, 5)
    assert w.state['drain_retry_counts'] == {hashlib.sha256(tr.frames[1]).hexdigest(): 2}
    w.vdb.fail_ids = lambda i: False                                  # the fault clears
    caplog.clear()                                                    # (caplog also holds the two held cycles' records)
    with caplog.at_level(logging.DEBUG, logger=org.logger.name):
        a, c = _drain(w, tr, 5)
    assert (a, c) == (2, tr.frames[1] + tr.frames[2]) and tr.read() == b''
    assert not tr.qexists() and w.state.get('quarantined_total', 0) == 0
    assert w.state['drain_retry_counts'] == {}, 'the count leaves with the success'
    assert _recs(caplog, QUAR_MARK) == [] and _recs(caplog, HOLD_PREFIX) == []
    assert snap(w) == snap(w.clean_pass([T1, TP, T3]))


def test_a_turn_that_succeeds_resets_so_a_later_failure_of_a_different_turn_starts_at_one(world, tmp_path):
    w = world
    tr = _setup(w, tmp_path, texts=(TP,))
    _drain(w, tr, 5)
    w.vdb.fail_ids = lambda i: False
    _drain(w, tr, 5)
    assert w.state['drain_retry_counts'] == {} and tr.read() == b''
    tr2 = _setup(w, tmp_path, texts=(TP2,))
    _drain(w, tr2, 5)
    assert w.state['drain_retry_counts'] == {hashlib.sha256(tr2.frames[0]).hexdigest(): 1}


# ------------------------------------------------------------------ a FAILED quarantine write keeps the entry HELD

def _break_quarantine(how, tr, monkeypatch):
    """Inject ONE fault into the quarantine write; returns restore() which clears ONLY that fault (never monkeypatch.undo(): it
    would also undo the dual-pass seams the world fixture installed)."""
    real_open, real_fsync = os.open, os.fsync
    if how == 'qpath_is_a_directory':
        os.mkdir(tr.qpath)                              # a real fault: the file cannot be opened for append
        return lambda: os.rmdir(tr.qpath)
    if how == 'open_raises':
        def bad_open(path, flags, *a, **k):
            if str(path).endswith('.quarantine'):
                raise PermissionError(SECRET)
            return real_open(path, flags, *a, **k)
        monkeypatch.setattr(os, 'open', bad_open)
        return lambda: monkeypatch.setattr(os, 'open', real_open)
    if how == 'fsync_raises':
        def bad_fsync(fd):
            raise OSError(SECRET)
        monkeypatch.setattr(os, 'fsync', bad_fsync)
        return lambda: monkeypatch.setattr(os, 'fsync', real_fsync)
    raise ValueError(how)


FAULTS = ['qpath_is_a_directory', 'open_raises', 'fsync_raises']


@pytest.mark.parametrize('how', FAULTS)
def test_a_failed_quarantine_write_keeps_the_entry_held_and_is_one_error(world, tmp_path, caplog, monkeypatch, how):
    w = world
    tr = _setup(w, tmp_path)
    for _ in range(2):                                              # max_retries=2: attempts 1 and 2 are held; the 3rd would quarantine
        _drain(w, tr, 2)
    held = tr.frames[1] + tr.frames[2]
    assert tr.read() == held
    _break_quarantine(how, tr, monkeypatch)
    for cycle in range(3):                                              # the entry is retried EVERY cycle and never advanced past
        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger=org.logger.name):
            a, c = _drain(w, tr, 2)
        assert (a, c) == (0, b'') and tr.read() == held, 'cycle %d: never advanced past it' % cycle
        errs = [r for r in caplog.records if r.name == org.logger.name and r.levelno == logging.ERROR
                and r.getMessage().startswith('CC ingest-tract')]
        assert len(errs) == 1 and errs[0].getMessage().startswith(QFAIL_MARK), [e.getMessage() for e in errs]
        m = errs[0].getMessage()
        assert _tag(tr.frames[1]) in m and 'nothing was discarded' in m and SECRET not in m and TEXT_CANARY not in m
        assert _recs(caplog, QUAR_MARK) == [] and _recs(caplog, HOLD_PREFIX) == []
        assert w.state.get('quarantined_total', 0) == 0 or how == 'fsync_raises'   # fsync_raises: the bytes were appended, only the sync failed
        if tr.qexists() and how != 'qpath_is_a_directory':
            assert tr.qread() == tr.frames[1], 'at most ONE copy however many attempts failed'
    assert snap(w) == snap(w.clean_pass([T1])), 'a held poison turn still leaves nothing in the graph'


def test_a_frame_found_at_the_tail_is_fsynced_again_before_the_tract_advances(world, tmp_path, monkeypatch):
    """THE HOLE this test was written for: attempt 1 WRITES the frame but its fsync fails; attempt 2 finds the identical frame
    already last in the file. It must NOT take 'already_present' as proof of durability: it fsyncs (file and directory) again,
    and if THAT fails the entry stays held. Only a successful fsync lets the tract advance."""
    w = world
    tr = _setup(w, tmp_path, texts=(TP, T3))
    real_fsync = os.fsync
    _drain(w, tr, 1)                                   # max_retries=1: attempt 1 is held (nothing quarantined yet)
    assert not tr.qexists()
    monkeypatch.setattr(os, 'fsync', lambda fd: (_ for _ in ()).throw(OSError(SECRET)))
    _drain(w, tr, 1)                                   # attempt 2 quarantines: appended, fsync fails -> held
    assert tr.qread() == tr.frames[0] and tr.read() == b''.join(tr.frames)
    _drain(w, tr, 1)                                   # attempt 3: frame already at the tail, fsync STILL fails -> still held
    assert tr.qread() == tr.frames[0] and tr.read() == b''.join(tr.frames)
    calls = []
    monkeypatch.setattr(os, 'fsync', lambda fd: (calls.append(fd), real_fsync(fd))[1])
    _drain(w, tr, 1)                                   # attempt 4: fsync works: it is called (file + directory) BEFORE the tract advances
    assert len(calls) == 2, 'the already_present path fsynced the file and the directory'
    assert tr.read() == tr.frames[1] and tr.qread() == tr.frames[0]


@pytest.mark.parametrize('how', FAULTS)
def test_recovery_after_a_failed_quarantine_write_quarantines_exactly_once(world, tmp_path, caplog, monkeypatch, how):
    w = world
    tr = _setup(w, tmp_path)
    for _ in range(2):                                              # attempts 1 and 2 held
        _drain(w, tr, 2)
    restore = _break_quarantine(how, tr, monkeypatch)
    _drain(w, tr, 2)                                                # the 3rd attempt: the quarantine write fails -> held
    assert tr.read() == tr.frames[1] + tr.frames[2]
    restore()
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger=org.logger.name):
        a, c = _drain(w, tr, 2)
    assert (a, c) == (0, tr.frames[1]) and tr.read() == tr.frames[2]
    assert tr.qread() == tr.frames[1], 'exactly one copy of the frame, however many attempts failed on the way'
    assert len(_recs(caplog, QUAR_MARK)) == 1


# ------------------------------------------------------------------ a crash between the quarantine fsync and the truncate

class _Crash(BaseException):
    """Stands in for the process dying: a BaseException passes every `except Exception` in the drain."""


class _TruncateFault:
    """Shadows `open` inside the module under test ONLY: opening the TRACT for write (the truncate step) raises `exc` while armed."""

    def __init__(self, monkeypatch, tract, exc):
        self.armed = True
        real_open = open

        def spy(path, mode='r', *a, **k):
            if self.armed and str(path) == tract.path and mode == 'wb':
                raise exc
            return real_open(path, mode, *a, **k)
        monkeypatch.setattr(org, 'open', spy, raising=False)


def test_crash_between_the_quarantine_fsync_and_the_truncate_neither_duplicates_nor_loses(world, tmp_path, caplog, monkeypatch):
    w = world
    tr = _setup(w, tmp_path)
    fault = _TruncateFault(monkeypatch, tr, _Crash())
    fault.armed = False                                # disarmed for attempt 1: its prefix truncate (T1) must succeed normally
    _drain(w, tr, 1)                                   # max_retries=1: absorbs T1, attempt 1 of TP is held
    original = tr.read()
    assert original == tr.frames[1] + tr.frames[2] and not tr.qexists()
    fault.armed = True
    with pytest.raises(_Crash):
        _drain(w, tr, 1)                               # attempt 2 quarantines TP durably, then 'dies' at the truncate
    assert tr.qread() == tr.frames[1], 'the frame was durable BEFORE the crash'
    assert tr.read() == original, 'the crash left the tract untouched: the entry is NOT lost'
    fault.armed = False
    # the restart: fresh process memory (new graph, new state), same files. The count restarts, so the entry takes its
    # attempts again (1 held, 2 quarantines) and is re-quarantined idempotently.
    w2 = types.SimpleNamespace()
    w2.graph, w2.vdb, w2.state = _new_stores()
    _poison(w2.vdb, POISON_TREE_WORD)
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger=org.logger.name):
        _drain(w2, tr, 1)                              # attempt 1 after the restart: held
        assert tr.read() == original and tr.qread() == tr.frames[1]
        _drain(w2, tr, 1)                              # attempt 2: quarantined again, idempotently
    assert tr.qread() == tr.frames[1], 'no duplicate: the identical frame already LAST in the file is not appended twice'
    assert 'stored=already_present' in _recs(caplog, QUAR_MARK)[0].getMessage()
    assert w2.state.get('quarantined_total', 0) == 0, 'not newly appended by this process'
    assert tr.read() == tr.frames[2]


def test_a_soft_truncate_failure_requarantines_idempotently_and_does_not_inflate_the_counter(world, tmp_path, monkeypatch):
    w = world
    tr = _setup(w, tmp_path, texts=(TP, T3))
    fault = _TruncateFault(monkeypatch, tr, PermissionError(SECRET))
    _drain(w, tr, 1)                                   # max_retries=1: attempt 1 is held (poison first: nothing to truncate)
    assert not tr.qexists() and tr.read() == b''.join(tr.frames)
    for _ in range(3):                                 # attempts 2, 3, 4: each (re-)quarantines, each truncate is refused
        _drain(w, tr, 1)
        assert tr.read() == b''.join(tr.frames), 'the tract could not be advanced, so it is untouched'
        assert tr.qread() == tr.frames[0], 'and the quarantine still holds exactly one copy'
    assert w.state['quarantined_total'] == 1, 'newly appended once; the retries are already_present and not counted again'
    fault.armed = False
    _drain(w, tr, 1)
    assert tr.read() == tr.frames[1] and tr.qread() == tr.frames[0]
    assert w.state['drain_retry_counts'] == {}, 'the count ends once the removal is confirmed'


# ------------------------------------------------------------------ THE BOUNDARY, pinned (build-002: the literal name wins)

@pytest.mark.parametrize('max_retries', [1, 2, 5])
def test_boundary_attempt_n_still_held_attempt_n_plus_1_is_quarantined_in_that_cycle(world, tmp_path, caplog, max_retries):
    """max_retries = RETRIES AFTER the first failed attempt. After exactly max_retries failing attempts the entry is STILL HELD (the tract
    is byte-identical, no quarantine file, the count is max_retries); the very next attempt (number max_retries + 1) quarantines it IN
    that cycle. max_retries=1 -> 2 attempts; the default 5 -> 6 attempts."""
    w = world
    tr = _setup(w, tmp_path, texts=(TP,))
    key = hashlib.sha256(tr.frames[0]).hexdigest()
    for attempt in range(1, max_retries + 1):
        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger=org.logger.name):
            assert _drain(w, tr, max_retries) == (0, b'')
        assert tr.read() == tr.frames[0] and not tr.qexists(), 'attempt %d of %d must still be HELD' % (attempt, max_retries + 1)
        assert w.state['drain_retry_counts'] == {key: attempt} and w.state.get('quarantined_total', 0) == 0
        assert _recs(caplog, QUAR_MARK) == []
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger=org.logger.name):
        assert _drain(w, tr, max_retries) == (0, tr.frames[0])          # attempt max_retries + 1: quarantined IN this cycle
    assert tr.read() == b'' and tr.qread() == tr.frames[0]
    assert len(_recs(caplog, QUAR_MARK)) == 1 and w.state['quarantined_total'] == 1
    assert 'failures=%d' % (max_retries + 1) in _recs(caplog, QUAR_MARK)[0].getMessage()


@pytest.mark.parametrize('max_retries', [1, 2, 5])
def test_boundary_max_retries_failures_then_a_success_on_the_last_attempt_is_absorbed_not_quarantined(world, tmp_path, caplog, max_retries):
    """The other side of the boundary: the entry fails max_retries times and the (max_retries + 1)-th attempt, its LAST allowed one,
    succeeds: it is absorbed, never quarantined, the count leaves, the graph equals one clean pass."""
    w = world
    tr = _setup(w, tmp_path, texts=(T1, TP, T3))
    for _ in range(max_retries):
        _drain(w, tr, max_retries)
    assert not tr.qexists() and tr.read() == tr.frames[1] + tr.frames[2]
    w.vdb.fail_ids = lambda i: False                                   # the fault clears before the last allowed attempt
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger=org.logger.name):
        a, c = _drain(w, tr, max_retries)
    assert (a, c) == (2, tr.frames[1] + tr.frames[2]) and tr.read() == b'' and not tr.qexists()
    assert w.state['drain_retry_counts'] == {} and w.state.get('quarantined_total', 0) == 0
    assert _recs(caplog, QUAR_MARK) == [] and snap(w) == snap(w.clean_pass([T1, TP, T3]))


def test_the_hold_warning_and_the_quarantine_error_state_the_numbers_literally(world, tmp_path, caplog):
    """Default-sized bound (5): the first hold WARNING reads 'attempt=1 of 6'; the 6th attempt's ERROR reads 'quarantining after 6 attempts,
    max_retries=5'. (No other text or id: the existing no-leak assertions stay in their own tests.)"""
    w = world
    tr = _setup(w, tmp_path, texts=(TP,))
    with caplog.at_level(logging.DEBUG, logger=org.logger.name):
        for _ in range(6):
            _drain(w, tr, 5)
    holds = [r.getMessage() for r in _recs(caplog, HOLD_PREFIX)]
    assert len(holds) == 5 and 'attempt=1 of 6 (max_retries=5)' in holds[0] and 'attempt=5 of 6 (max_retries=5)' in holds[4]
    (err,) = _recs(caplog, QUAR_MARK)
    assert 'quarantining after 6 attempts (= max_retries + 1), max_retries=5' in err.getMessage()


@pytest.mark.parametrize('max_retries', [1, 5])
def test_the_quarantine_error_states_attempts_equal_retries_plus_one(world, tmp_path, caplog, max_retries):
    """Exec P501's ONE condition on the literal count: the quarantine ERROR carries the relation IN WORDS, with the numbers it
    relates: `quarantining after <attempts> attempts (= max_retries + 1), max_retries=<N>`, where attempts is exactly N + 1."""
    w = world
    tr = _setup(w, tmp_path, texts=(TP,))
    with caplog.at_level(logging.DEBUG, logger=org.logger.name):
        for _ in range(_attempts(max_retries)):
            _drain(w, tr, max_retries)
    (err,) = _recs(caplog, QUAR_MARK)
    m = re.search(r'quarantining after (\d+) attempts \(= max_retries \+ 1\), max_retries=(\d+)', err.getMessage())
    assert m, err.getMessage()
    assert (int(m.group(1)), int(m.group(2))) == (max_retries + 1, max_retries) and int(m.group(1)) == int(m.group(2)) + 1


def test_the_docstring_states_the_formula_and_the_operator_lever():
    """Docstring only (no behaviour): the count formula and the lever (RAISE CC_NG_DRAIN_MAX_RETRIES while TID is flaky) are written down,
    and the infra-vs-data split is named as not built (#928)."""
    doc = ' '.join(org.drain_ingest_tract.__doc__.split())
    assert 'RETRIES AFTER THE FIRST FAILED ATTEMPT' in doc and 'max_retries + 1 attempts' in doc
    assert 'attempts = max_retries + 1' in doc, 'Exec P501: the relation is stated in words in the docstring'
    assert 'max_retries=5 means up to 6 attempts' in doc and 'max_retries=1 means 2 attempts' in doc
    assert 'RAISE CC_NG_DRAIN_MAX_RETRIES' in doc and '#928' in doc


# ------------------------------------------------------------------ never created, never truncated, append-only

def test_the_quarantine_file_is_never_created_when_nothing_is_quarantined(world, tmp_path):
    w = world
    for hold, mr in ((True, 5), (True, None), (False, None), (False, 5)):
        tr = Tract(tmp_path, [T1, T3], name='ok_%s_%s.tract' % (hold, mr))
        _drain(w, tr, mr, hold=hold)
        assert tr.read() == b'' and not tr.qexists()
    tr = _setup(w, tmp_path, texts=(T1, TP), )
    _drain(w, tr, 5)                                                     # held below the bound: still nothing quarantined
    assert not tr.qexists()
    assert sorted(p.name for p in tmp_path.iterdir() if p.name.endswith('.quarantine')) == []


def test_the_quarantine_file_is_append_only_and_never_truncated_across_calls(world, tmp_path):
    w = world
    tr = Tract(tmp_path, [TP, T3, TP2])
    _poison(w.vdb, POISON_TREE_WORD, 'poison two concept B')
    sizes = []
    for _ in range(8):
        _drain(w, tr, 2)
        if tr.qexists():
            sizes.append(len(tr.qread()))
    assert tr.read() == b''
    assert tr.qread() == tr.frames[0] + tr.frames[2], 'both poison frames, in order, whole, nothing between them'
    assert sizes == sorted(sizes) and sizes[0] == len(tr.frames[0]) and sizes[-1] == len(tr.frames[0]) + len(tr.frames[2])
    assert w.state['quarantined_total'] == 2


def test_the_quarantine_file_is_itself_a_valid_tract_of_the_quarantined_entries(world, tmp_path):
    """Why a deliberate re-ingest is easy later (NOT built here): the file is a flat run of the tract's own frames."""
    w = world
    tr = Tract(tmp_path, [TP, TP2])
    _poison(w.vdb, POISON_TREE_WORD, 'poison two concept B')
    for _ in range(6):
        _drain(w, tr, 1)
    entries = list(ng_tract.TractReader(tr.qread()))
    assert [e.content for e in entries] == [TP, TP2]


def test_the_quarantine_file_is_created_no_wider_than_the_tract_directory_and_file(world, tmp_path):
    w = world
    tr = _setup(w, tmp_path, texts=(TP,))
    os.chmod(tr.path, 0o640)
    os.chmod(str(tmp_path), 0o750)
    try:
        for _ in range(_attempts(2)):
            _drain(w, tr, 2)
        mode = os.stat(tr.qpath).st_mode & 0o777
        assert mode & ~0o640 == 0 and mode & ~0o750 == 0 and mode & 0o111 == 0, oct(mode)
    finally:
        os.chmod(str(tmp_path), 0o700)


# ------------------------------------------------------------------ the table: keyed by the sha256 of the raw bytes, hard-capped

def test_the_retry_key_is_the_sha256_of_the_raw_frame_bytes_never_the_text(world, tmp_path):
    w = world
    tr = _setup(w, tmp_path, texts=(TP,))
    _drain(w, tr, 5)
    (key,) = w.state['drain_retry_counts']
    assert key == hashlib.sha256(tr.frames[0]).hexdigest() and len(key) == 64
    assert TEXT_CANARY not in json.dumps(w.state['drain_retry_counts'])


def test_the_retry_table_is_hard_capped(world, tmp_path, monkeypatch):
    """300 DISTINCT failing frames across calls (each call sees a new first-failing entry): the table never exceeds its cap."""
    w = world
    monkeypatch.setattr(org, 'run_conversational_dual_pass', lambda *a, **k: False)
    cap = org._DRAIN_RETRY_TABLE_MAX
    assert 1 <= cap <= 1024
    for i in range(cap + 44):
        tr = Tract(tmp_path, ['distinct poison turn number %d' % i], name='cap_%d.tract' % i)
        _drain(w, tr, 10 ** 6)
        assert len(w.state['drain_retry_counts']) <= cap
    assert len(w.state['drain_retry_counts']) == cap


def test_a_restart_resets_the_count_so_the_bound_is_per_process_lifetime(world, tmp_path):
    w = world
    tr = _setup(w, tmp_path, texts=(TP,))
    _drain(w, tr, 3)
    _drain(w, tr, 3)
    assert list(w.state['drain_retry_counts'].values()) == [2]
    fresh = {'last_forest_id': None}                                     # a restarted daemon's conv_state
    org.drain_ingest_tract(w.graph, w.vdb, fresh, tract_path=tr.path, hold_on_failure=True, max_retries=3)
    assert list(fresh['drain_retry_counts'].values()) == [1] and not tr.qexists()


# ------------------------------------------------------------------ argument contract

@pytest.mark.parametrize('bad', [0, -1, 'x', 2.5, True, [3]])
def test_a_non_positive_or_non_int_bound_is_off_with_one_fixed_warning(world, tmp_path, caplog, bad):
    w = world
    tr = _setup(w, tmp_path, texts=(TP, T3))
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger=org.logger.name):
        for _ in range(3):
            _drain(w, tr, bad)
    warns = [r for r in caplog.records if r.name == org.logger.name and 'max_retries is not a positive integer' in r.getMessage()]
    assert len(warns) == 3                                               # one per call (the daemon never passes one)
    assert len({r.getMessage() for r in warns}) == 1, 'a FIXED text: no value is interpolated'
    assert not tr.qexists() and tr.read() == b''.join(tr.frames), 'unbounded: held forever, as before #920'
    assert 'drain_retry_counts' not in w.state


def test_max_retries_without_hold_on_failure_is_ignored(world, tmp_path, caplog):
    w = world
    tr = _setup(w, tmp_path)
    with caplog.at_level(logging.DEBUG, logger=org.logger.name):
        a, c = _drain(w, tr, 1, hold=False)
    assert (a, c) == (2, b''.join(tr.frames)) and tr.read() == b'' and not tr.qexists()    # flag off: the failed turn is consumed (#794 default)
    assert 'drain_retry_counts' not in w.state and 'quarantined_total' not in w.state
    assert [r for r in caplog.records if 'max_retries' in r.getMessage()] == []


def test_max_entries_still_counts_the_poison_attempt(world, tmp_path):
    w = world
    tr = _setup(w, tmp_path, texts=(TP, T3))
    a, c = _drain(w, tr, 1, max_entries=1)             # max_retries=1: attempt 1 is held
    assert (a, c) == (0, b'') and not tr.qexists() and tr.read() == b''.join(tr.frames)
    a, c = _drain(w, tr, 1, max_entries=1)             # attempt 2 quarantines
    assert (a, c) == (0, tr.frames[0]) and tr.qread() == tr.frames[0] and tr.read() == tr.frames[1]
    a, c = _drain(w, tr, 1, max_entries=1)
    assert (a, c) == (1, tr.frames[1]) and tr.read() == b''


# ------------------------------------------------------------------ DEFAULTS ARE BYTE-IDENTICAL TO THE #904 TIP (live differential vs the BASE module)

def _load_base_module():
    d = _SCRATCH_ROOT / 'base_org'
    d.mkdir(parents=True, exist_ok=True)
    src = subprocess.run(['git', '-C', str(_WORKTREE), 'show', '%s:cc_ng_organism.py' % BASE_SHA], capture_output=True, check=True).stdout
    p = d / 'cc_ng_organism.py'
    p.write_bytes(src)
    spec = importlib.util.spec_from_file_location('cc_ng_organism_base_920', str(p))
    mod = importlib.util.module_from_spec(spec)
    sys.modules['cc_ng_organism_base_920'] = mod          # dataclasses looks cls.__module__ up here
    spec.loader.exec_module(mod)
    return mod


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.rows = []

    def emit(self, record):
        if record.name == 'cc_ng_organism':
            self.rows.append((record.levelname, record.getMessage()))


def _scenario(mod, tmp_path, tag, texts, frames, hold, kw, fail_words):
    graph, vdb, state = _new_stores()
    _poison(vdb, *fail_words)
    tr = Tract(tmp_path, list(texts), name='%s.tract' % tag, frames=frames)
    cap = _Capture()
    root = logging.getLogger('cc_ng_organism')
    old = root.level
    root.setLevel(logging.DEBUG)
    root.addHandler(cap)
    try:
        outs = []
        for _ in range(3):
            outs.append(mod.drain_ingest_tract(graph, vdb, state, tract_path=tr.path, return_consumed=True, hold_on_failure=hold, **kw))
    finally:
        root.removeHandler(cap)
        root.setLevel(old)
    w = types.SimpleNamespace(graph=graph, vdb=vdb, state=state)
    return {'outs': outs, 'tract': tr.read(), 'snap': snap(w), 'log': cap.rows, 'qexists': tr.qexists(), 'state_keys': sorted(state)}


@pytest.mark.parametrize('name,texts,hold,kw_new,fail', [
    ('hold_off_failing_middle', (T1, TP, T3), False, {}, (POISON_TREE_WORD,)),
    ('hold_off_ignores_max_retries', (T1, TP, T3), False, {'max_retries': 3}, (POISON_TREE_WORD,)),
    ('hold_on_unbounded_failing_middle', (T1, TP, T3), True, {}, (POISON_TREE_WORD,)),
    ('hold_on_explicit_none_failing_middle', (T1, TP, T3), True, {'max_retries': None}, (POISON_TREE_WORD,)),
    ('hold_on_all_healthy', (T1, T3), True, {}, ()),
    ('hold_off_all_healthy', (T1, T3), False, {}, ()),
])
def test_defaults_are_byte_identical_to_the_904_tip(world, tmp_path, monkeypatch, name, texts, hold, kw_new, fail):
    """Same scenario, same fresh stores, run against the BASE module (895a809a, loaded live from git) and the module under test:
    the returns, the tract bytes, the graph/vdb snapshot, EVERY log record (level + text) and the state keys must be equal, and no
    quarantine file exists. (New kwargs are passed only to the module under test: base has no max_retries.)"""
    base = _load_base_module()
    frames = Tract(tmp_path, list(texts), name=name + '_proto.tract').frames          # ONE set of bytes for both runs
    got_base = _scenario(base, tmp_path, name + '_base', texts, frames, hold, {}, fail)
    got_new = _scenario(org, tmp_path, name + '_new', texts, frames, hold, kw_new, fail)
    assert got_new['outs'] == got_base['outs']
    assert got_new['tract'] == got_base['tract']
    assert got_new['snap'] == got_base['snap']
    assert got_new['log'] == got_base['log']
    assert got_new['state_keys'] == got_base['state_keys']
    assert not got_new['qexists'] and not got_base['qexists']
    assert not any(m.startswith('CC ingest-tract QUARANTINE') for _lvl, m in got_new['log'])
    # not vacuous: the compared logs really contained the #794 hold WARNING exactly when a failing turn met hold_on_failure=True
    assert any(m.startswith(HOLD_PREFIX) for _lvl, m in got_new['log']) == (hold and bool(fail))
    assert got_new['log'], 'an empty log would make the log comparison meaningless'


# ------------------------------------------------------------------ the REAL docs daemon loop on REAL NG (gated)

needs_daemon = pytest.mark.skipif(
    not DAEMON_ENV, reason='Z12_920_DAEMON_UNDER_TEST is unset: the daemon-loop test needs the docs daemon file (the #920 docs worktree)')
_DAEMON_COUNT = [0]


@pytest.fixture
def daemon_rig(world, tmp_path, monkeypatch):
    w = world
    tr = _setup(w, tmp_path)
    monkeypatch.setenv('CC_GATEWAY_TRACT_PATH', tr.path)
    monkeypatch.setenv('CC_GATEWAY_CONDUIT_PATH', str(tmp_path / 'conduit'))
    for var in ('CC_NG_BATCH_SIZE', 'CC_NG_IDLE_STEPS', 'CC_NG_DRAIN_HOLD_ON_FAILURE', 'CC_NG_DRAIN_MAX_RETRIES'):
        monkeypatch.delenv(var, raising=False)
    trickled = []
    monkeypatch.setattr(org, 'trickle_gateway_conduit', lambda data, *a, **k: trickled.append(data))
    for name in ('cc_update_probation', 'surface_wants', 'generate_emergent_want', 'persist_cc_commons'):
        monkeypatch.setattr(org, name, lambda *a, **k: None)
    drain_calls = []
    real_drain = org.drain_ingest_tract
    monkeypatch.setattr(org, 'drain_ingest_tract', lambda *a, **k: (drain_calls.append(dict(k)), real_drain(*a, **k))[1])
    _DAEMON_COUNT[0] += 1
    spec = importlib.util.spec_from_file_location('z12_920_daemon_%d' % _DAEMON_COUNT[0], DAEMON_ENV)
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
def test_daemon_loop_quarantines_a_poison_turn_after_the_bound_and_counts_it(daemon_rig, caplog):
    r, w = daemon_rig, daemon_rig.w
    r.d.DRAIN_HOLD_ON_FAILURE = True
    r.d.DRAIN_MAX_RETRIES = 3
    with caplog.at_level(logging.DEBUG):
        r.run(_attempts(3))                                          # max_retries=3: four autosave cycles = four attempts
    assert r.drain_calls == [{'return_consumed': True, 'hold_on_failure': True, 'max_retries': 3}] * 4
    assert r.tract.qread() == r.tract.frames[1] and r.tract.read() == r.tract.frames[2]
    assert r.d.STATE.stats['quarantined_total'] == 1
    assert r.trickled[0] == r.tract.frames[0] and r.trickled[1:3] == [b'', b''] and r.trickled[3] == r.tract.frames[1]
    errs = [x for x in caplog.records if x.getMessage().startswith(QUAR_MARK)]
    assert len(errs) == 1
    r.run(1)                                                         # the entry behind it flows
    assert r.tract.read() == b'' and r.d.STATE.stats['quarantined_total'] == 1
    assert snap(w) == snap(w.clean_pass([T1, T3]))


@needs_daemon
def test_daemon_loop_with_the_hold_flag_off_is_the_904_call_and_adds_no_stat(daemon_rig, caplog):
    r, w = daemon_rig, daemon_rig.w
    assert r.d.DRAIN_HOLD_ON_FAILURE is False and r.d.DRAIN_MAX_RETRIES is None
    with caplog.at_level(logging.DEBUG):
        r.run(1)
    assert r.drain_calls == [{'return_consumed': True}]                      # byte-identical call: neither kwarg is passed
    assert 'quarantined_total' not in r.d.STATE.stats and not r.tract.qexists()
