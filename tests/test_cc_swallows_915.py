# ---- Changelog ----
# [2026-10-01] Z12 builder (Claude Sonnet 5.5), lane held-clock-visible-901, dispatch #13433 (Exec P501 / le-052 #4) — tests for the READABLE counter and the
#   rate-limited want-create WARNING: cc_want_create_failure_count() totals both functions and rises by the failures; a second failure inside the interval is
#   COUNTED but not re-logged (per function); the first failure always warns; the warning returns after the interval; the success path is still quiet and
#   byte-identical; no str(exc) / id. An autouse fixture clears the limiter state so the earlier one-WARNING-per-call tests stay independent.
# [2026-10-01] Z12 builder (Claude Sonnet 5.5), lane held-clock-visible-901, dispatch #13220 — #915 (Exec Packet 495): tests for the three
#   NG swallows made loud
# What: (a) _cc_deposit_memory_node's recall-insert WARNING is class-name only (the function still re-raises, still no rollback);
#   (b) surface_wants guards create_node PER WANT (one raise no longer aborts the loop; counted; ONE WARNING per call, class names only);
#   (c) surface_wants_for_graph's outer swallow is a counted WARNING (class names only) instead of DEBUG + str(exc). H-1: nothing
#   existing (authored wants, the constitutional node) is deleted, re-tagged or edited by a failing pulse. Golden success path.
# Why: Exec Packet 495 / row #915 (the ONE RULE of Packet 494: exception text that can name a node never goes into a log).
# How: a REAL neuro_foundation.Graph + a REAL SimpleVectorDB; only create_node / insert are wrapped to raise for chosen ids. No model, no
#   embedder, no network, no live graph. Z12_915_ORG_UNDER_TEST points the module under test at a SCRATCH copy of cc_ng_organism.py (the
#   BASE 895a809a, or a mutant) so the same file runs against the base and against each mutation. P379/#770: a preamble prints every
#   module path and the session FAILS if an NG module (other than a scratch org under /tmp) resolves outside this worktree.
# -------------------
"""Which tests FAIL on the base 895a809a: the loop-continues, loud-WARNING, counter and class-only tests (surface_wants aborts on the
first raise; surface_wants_for_graph logs at DEBUG with str(exc); the deposit warning prints str(exc)). The golden, H-1 and
re-raise/no-rollback tests pass on both (they pin behaviour this slice must NOT change)."""
import copy
import hashlib
import importlib.util
import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

_WORKTREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_WORKTREE))

import neuro_foundation  # noqa: E402
from neuro_foundation import Graph  # noqa: E402
from universal_ingestor import SimpleVectorDB  # noqa: E402

_ORG_ALT = os.environ.get('Z12_915_ORG_UNDER_TEST')
if _ORG_ALT:
    _spec = importlib.util.spec_from_file_location('cc_ng_organism', _ORG_ALT)
    org = importlib.util.module_from_spec(_spec)
    sys.modules['cc_ng_organism'] = org
    _spec.loader.exec_module(org)
else:
    import cc_ng_organism as org  # noqa: E402

_TMP = Path(tempfile.gettempdir()).resolve()
_PREAMBLE = ['[P379/#770] worktree root       -> %s' % _WORKTREE,
             '[P379/#770] cc_ng_organism      -> %s' % Path(org.__file__).resolve(),
             '[P379/#770] neuro_foundation    -> %s' % Path(neuro_foundation.__file__).resolve(),
             '[P379/#770] org is a scratch copy: %s' % bool(_ORG_ALT)]
sys.__stderr__.write('\n' + '\n'.join(_PREAMBLE) + '\n')
if _WORKTREE not in Path(neuro_foundation.__file__).resolve().parents:
    raise RuntimeError('P379/#770 FAIL: neuro_foundation resolves outside the worktree %s' % _WORKTREE)
if _ORG_ALT:
    if _TMP not in Path(org.__file__).resolve().parents:
        raise RuntimeError('P379/#770 FAIL: a scratch org must live under %s, got %s' % (_TMP, org.__file__))
elif _WORKTREE not in Path(org.__file__).resolve().parents:
    raise RuntimeError('P379/#770 FAIL: cc_ng_organism resolves outside the worktree %s' % _WORKTREE)

SECRET = 'SECRET-EXC-TEXT-915'
WANT_A = 'learn the first thing WANTCANARY915A'
WANT_B = 'learn the second thing WANTCANARY915B'
TREE_WORDS = 'quarterly salary negotiation with my landlord'
TREE_ID = 'cc:conv::' + 'ab' * 20 + '::tree::' + TREE_WORDS
LEAKY = '%s %s' % (SECRET, TREE_ID)
WANT_FNS = [('surface_wants', 'cc:want::'), ('surface_wants_for_graph', 'want::')]


def _vec(text):
    d = hashlib.sha256(str(text).encode()).digest()
    return np.array([b / 255.0 + 0.05 for b in d[:16]], dtype=np.float32)


def _want_id(prefix, text):
    return prefix + hashlib.sha1(text.encode()).hexdigest()[:16]


def _stores():
    return Graph(), SimpleVectorDB()


def _seed_source(g, v, nid, text):
    g.create_node(node_id=nid, metadata={'creation_mode': 'conversational'})
    v.insert(nid, _vec(text), text, {})


def _fail_creates(g, fail_when, exc_cls=KeyError):
    real = g.create_node

    def create_node(node_id, metadata=None, *a, **k):
        if fail_when(node_id):
            raise exc_cls(LEAKY)
        return real(node_id=node_id, metadata=metadata, *a, **k)

    g.create_node = create_node
    g._real_create_node = real


def _run(fn_name, g, v):
    return getattr(org, fn_name)(g, v)


def _org_records(caplog):
    return [r for r in caplog.records if r.name == org.logger.name]


def _all_text(caplog):
    out = []
    for r in caplog.records:
        out.append(r.getMessage())
        out.append(repr(r.args))
    return '\n'.join(out)


def _assert_no_leak(text, *extra):
    for needle in (SECRET, TREE_ID, TREE_WORDS, 'WANTCANARY915', 'learn the', 'landlord', 'salary') + extra:
        assert needle not in text, 'leaked %r' % needle


def _two_wants(g, v):
    _seed_source(g, v, 'cc:conv::s1', 'a [WANT]%s[/WANT] b' % WANT_A)
    _seed_source(g, v, 'cc:conv::s2', 'c [WANT]%s[/WANT] d' % WANT_B)


# ------------------------------------------------------------------ (b) + (c): a failed want create

@pytest.mark.parametrize('fn_name,prefix', WANT_FNS)
def test_one_raising_create_does_not_abort_the_loop_later_wants_still_materialize(fn_name, prefix, caplog):
    """FAILS on the base for surface_wants (the unguarded create_node raised out of the function: the later want was never created).
    surface_wants_for_graph already continued on the base; it is here for the same contract."""
    g, v = _stores()
    _two_wants(g, v)
    first, second = _want_id(prefix, WANT_A), _want_id(prefix, WANT_B)
    _fail_creates(g, lambda nid: nid == first)
    with caplog.at_level(logging.DEBUG, logger=org.logger.name):
        out = _run(fn_name, g, v)                                    # must NOT raise
    assert [w['id'] for w in out] == [second]
    assert second in g.nodes and first not in g.nodes                  # not created, not half-created, retried next pulse


@pytest.mark.parametrize('fn_name,prefix', WANT_FNS)
def test_a_failed_create_is_a_counted_warning_with_the_class_name_only(fn_name, prefix, caplog):
    """FAILS on the base (surface_wants: no record, it raised; surface_wants_for_graph: DEBUG with str(exc), no WARNING)."""
    g, v = _stores()
    _two_wants(g, v)
    first = _want_id(prefix, WANT_A)
    _fail_creates(g, lambda nid: nid == first)
    before = dict(getattr(org, '_CC_WANT_CREATE_FAILURES', {}))
    with caplog.at_level(logging.DEBUG, logger=org.logger.name):
        _run(fn_name, g, v)
    loud = [r for r in _org_records(caplog) if r.levelno >= logging.WARNING]
    assert len(loud) == 1 and loud[0].levelno == logging.WARNING, [r.getMessage() for r in _org_records(caplog)]
    msg = loud[0].getMessage()
    for need in ('reason=want_create_failed', 'fn=%s ' % fn_name, 'attempted=2', 'failed=1', 'exc_types=KeyError'):
        assert need in msg, (need, msg)
    _assert_no_leak(_all_text(caplog), first)
    assert org._CC_WANT_CREATE_FAILURES[fn_name] == before.get(fn_name, 0) + 1          # COUNTED


@pytest.mark.parametrize('fn_name,prefix', WANT_FNS)
def test_every_want_failing_reports_once_per_call_with_the_total(fn_name, prefix, caplog):
    g, v = _stores()
    _two_wants(g, v)
    _fail_creates(g, lambda nid: nid.startswith(prefix), exc_cls=OSError)
    with caplog.at_level(logging.DEBUG, logger=org.logger.name):
        out = _run(fn_name, g, v)
    assert out == []
    loud = [r for r in _org_records(caplog) if r.levelno >= logging.WARNING]
    assert len(loud) == 1 and 'attempted=2' in loud[0].getMessage() and 'failed=2' in loud[0].getMessage()
    assert 'exc_types=OSError' in loud[0].getMessage()
    _assert_no_leak(_all_text(caplog))


@pytest.mark.parametrize('fn_name,prefix', WANT_FNS)
def test_a_failed_want_is_retried_on_the_next_pulse(fn_name, prefix, caplog):
    g, v = _stores()
    _two_wants(g, v)
    first = _want_id(prefix, WANT_A)
    _fail_creates(g, lambda nid: nid == first)
    _run(fn_name, g, v)
    g.create_node = g._real_create_node                               # the fault clears
    out = _run(fn_name, g, v)
    assert {w['id'] for w in out} == {first, _want_id(prefix, WANT_B)}


@pytest.mark.parametrize('fn_name,prefix', WANT_FNS)
def test_H1_a_failing_pulse_deletes_retags_and_edits_nothing_that_exists(fn_name, prefix, caplog):
    """Passes on base and fix for what it pins: the pre-existing authored wants and the constitutional node are byte-identical
    (ids, metadata, synapses) after a pulse in which a create raised."""
    g, v = _stores()
    g.create_node(node_id='cc:want::pre-existing', metadata={'kind': 'want', 'want_text': 'old authored want',
                                                              'want_state': 'open', 'provenance': 'cc_authored'})
    g.create_node(node_id='cc:constitutional::rim', metadata={'constitutional': True, 'core_text': 'who I am'})
    _two_wants(g, v)
    first = _want_id(prefix, WANT_A)
    snap = {nid: copy.deepcopy(n.metadata) for nid, n in g.nodes.items()}
    syn_before = sorted((s.pre_node_id, s.post_node_id) for s in g.synapses.values())
    _fail_creates(g, lambda nid: nid == first)
    try:
        _run(fn_name, g, v)
    except Exception:  # noqa: BLE001 -- the base's surface_wants raises here; what it left must still be untouched
        pass
    for nid, meta in snap.items():
        assert nid in g.nodes and g.nodes[nid].metadata == meta, nid
    assert g.nodes['cc:want::pre-existing'].metadata['provenance'] == 'cc_authored'
    assert g.nodes['cc:constitutional::rim'].metadata == {'constitutional': True, 'core_text': 'who I am'}
    assert set(syn_before) <= {(s.pre_node_id, s.post_node_id) for s in g.synapses.values()}


@pytest.mark.parametrize('fn_name,prefix', WANT_FNS)
def test_the_success_path_is_golden_identical_and_quiet(fn_name, prefix, caplog):
    """Passes on base and fix: the same return, the same node, the same ONE synapse (source -> want, 0.3), no record above DEBUG."""
    g, v = _stores()
    _seed_source(g, v, 'cc:conv::src', 'turn [WANT]%s[/WANT] tail' % WANT_A)
    with caplog.at_level(logging.DEBUG, logger=org.logger.name):
        out = _run(fn_name, g, v)
    wid = _want_id(prefix, WANT_A)
    assert out == [{'id': wid, 'text': WANT_A, 'provenance': 'cc_authored', 'state': 'open', 'source': 'cc:conv::src'}]
    assert [(s.pre_node_id, s.post_node_id, round(float(s.weight), 6)) for s in g.synapses.values()] == [('cc:conv::src', wid, 0.3)]
    assert g.nodes[wid].metadata['want_state'] == 'open' and g.nodes[wid].metadata['source_node'] == 'cc:conv::src'
    assert [r for r in _org_records(caplog) if r.levelno >= logging.WARNING] == []


def test_the_seed_synapse_swallow_of_904_is_unchanged(caplog):
    """The #904 contract (a failed seed synapse: want kept, unbound, ONE warning) still holds next to the new create guard."""
    g, v = _stores()
    _seed_source(g, v, 'cc:conv::src', 'turn [WANT]%s[/WANT] tail' % WANT_A)
    real = g.create_synapse

    def syn(pre, post, *a, **k):
        if post.startswith('cc:want::'):
            raise KeyError(SECRET)
        return real(pre, post, *a, **k)

    g.create_synapse = syn
    with caplog.at_level(logging.DEBUG, logger=org.logger.name):
        out = org.surface_wants(g, v)
    assert len(out) == 1
    loud = [r for r in _org_records(caplog) if r.levelno >= logging.WARNING]
    assert len(loud) == 1 and 'reason=want_source_synapse_failed' in loud[0].getMessage()


# ------------------------------------------------------------------ (a) _cc_deposit_memory_node

class _FailingVDB(SimpleVectorDB):
    def insert(self, id, embedding, content='', metadata=None):
        raise OSError(LEAKY)


def test_deposit_memory_node_warning_is_the_class_name_only_and_still_reraises(caplog):
    """FAILS on the base (the WARNING prints str(exc): the secret and the tree words)."""
    g = Graph()
    with caplog.at_level(logging.DEBUG, logger=org.logger.name):
        with pytest.raises(OSError):
            org._cc_deposit_memory_node(g, _FailingVDB(), TREE_ID, np.ones(16, dtype=np.float32), 'content', {})
    loud = [r for r in _org_records(caplog) if r.levelno >= logging.WARNING]
    assert len(loud) == 1
    msg = loud[0].getMessage()
    assert 'reason=recall_insert_failed' in msg and 'exc_type=OSError' in msg
    _assert_no_leak(_all_text(caplog))
    assert TREE_ID in g.nodes                                          # retained partial application: no fabricated rollback


def test_deposit_memory_node_success_is_unchanged_and_quiet(caplog):
    g, v = Graph(), SimpleVectorDB()
    with caplog.at_level(logging.DEBUG, logger=org.logger.name):
        node = org._cc_deposit_memory_node(g, v, 'cc:conv::x', np.ones(16, dtype=np.float32), 'content', {'k': 'v'})
    assert g.nodes['cc:conv::x'] is node and node.metadata['k'] == 'v'
    assert 'cc:conv::x' in v.content
    assert [r for r in _org_records(caplog) if r.levelno >= logging.WARNING] == []


def test_generate_emergent_want_is_untouched():
    """#905 owns it: its source must be byte-identical to the base."""
    import subprocess
    new = (_WORKTREE / 'cc_ng_organism.py').read_text()

    def body(src):
        start = src.index('def generate_emergent_want(')
        end = src.index('\ndef ', start + 10)
        return src[start:end]
    base = subprocess.run(['git', '-C', str(_WORKTREE), 'show', '895a809a60ab86afacd8ef193b95c4ab39f6893f:cc_ng_organism.py'],
                          capture_output=True, text=True).stdout
    assert base, 'cannot read the base; a pin must FAIL, not skip'
    assert body(new) == body(base)


# ------------------------------------------------------------------ Exec P501 / le-052 #4: the counter is READABLE, the WARNING is RATE-LIMITED

@pytest.fixture(autouse=True)
def _reset_want_create_limiter():
    """The limiter keeps module-level state; clear it (when it exists: the base has none) so each test starts un-throttled."""
    getattr(org, '_cc_want_create_warn_last', {}).clear()
    yield
    getattr(org, '_cc_want_create_warn_last', {}).clear()


def _fail_all_wants(g, prefix):
    _fail_creates(g, lambda nid: nid.startswith(prefix))


@pytest.mark.parametrize('fn_name,prefix', WANT_FNS)
def test_the_accessor_totals_both_functions_and_rises_by_the_failures(fn_name, prefix):
    """FAILS on 78810c1b (no accessor)."""
    before = org.cc_want_create_failure_count()
    g, v = _stores()
    _two_wants(g, v)
    _fail_all_wants(g, prefix)
    _run(fn_name, g, v)
    assert org.cc_want_create_failure_count() == before + 2
    assert org.cc_want_create_failure_count() == sum(org._CC_WANT_CREATE_FAILURES.values())


@pytest.mark.parametrize('fn_name,prefix', WANT_FNS)
def test_a_second_failing_call_inside_the_interval_is_counted_but_not_relogged(fn_name, prefix, caplog, monkeypatch):
    """FAILS on 78810c1b (the WARNING was unthrottled: one per call)."""
    monkeypatch.setattr(org, '_PITH_WARN_INTERVAL_S', 3600.0)
    g, v = _stores()
    _two_wants(g, v)
    _fail_all_wants(g, prefix)
    before = org.cc_want_create_failure_count()
    with caplog.at_level(logging.DEBUG, logger=org.logger.name):
        for _ in range(3):
            _run(fn_name, g, v)
    warns = [r for r in _org_records(caplog) if r.levelno >= logging.WARNING]
    assert len(warns) == 1                                                   # the FIRST failure always warns
    assert 'reason=want_create_failed' in warns[0].getMessage() and 'attempted=2' in warns[0].getMessage()
    assert org.cc_want_create_failure_count() == before + 6                  # 3 calls x 2 failures, ALL counted
    _assert_no_leak(_all_text(caplog))


@pytest.mark.parametrize('fn_name,prefix', WANT_FNS)
def test_the_warning_returns_after_the_interval_has_elapsed(fn_name, prefix, caplog, monkeypatch):
    monkeypatch.setattr(org, '_PITH_WARN_INTERVAL_S', 0.0)
    g, v = _stores()
    _two_wants(g, v)
    _fail_all_wants(g, prefix)
    with caplog.at_level(logging.DEBUG, logger=org.logger.name):
        _run(fn_name, g, v)
        _run(fn_name, g, v)
    assert len([r for r in _org_records(caplog) if r.levelno >= logging.WARNING]) == 2


def test_the_limiter_is_per_function_each_one_warns_once(caplog, monkeypatch):
    monkeypatch.setattr(org, '_PITH_WARN_INTERVAL_S', 3600.0)
    with caplog.at_level(logging.DEBUG, logger=org.logger.name):
        for fn_name, prefix in WANT_FNS:
            g, v = _stores()
            _two_wants(g, v)
            _fail_all_wants(g, prefix)
            _run(fn_name, g, v)
            _run(fn_name, g, v)
    msgs = [r.getMessage() for r in _org_records(caplog) if r.levelno >= logging.WARNING]
    assert len(msgs) == 2 and any('fn=surface_wants ' in m for m in msgs) and any('fn=surface_wants_for_graph ' in m for m in msgs)


@pytest.mark.parametrize('fn_name,prefix', WANT_FNS)
def test_a_success_path_moves_no_counter_and_stays_quiet(fn_name, prefix, caplog):
    before = org.cc_want_create_failure_count()
    g, v = _stores()
    _seed_source(g, v, 'cc:conv::src', 'turn [WANT]%s[/WANT] tail' % WANT_A)
    with caplog.at_level(logging.DEBUG, logger=org.logger.name):
        out = _run(fn_name, g, v)
    assert len(out) == 1
    assert org.cc_want_create_failure_count() == before
    assert [r for r in _org_records(caplog) if r.levelno >= logging.WARNING] == []
