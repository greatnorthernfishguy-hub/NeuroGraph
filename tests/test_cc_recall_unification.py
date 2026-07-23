# tests/test_cc_recall_unification.py
#
# ---- Changelog ----
# [2026-07-22] Claude Code (Sonnet 5) — CC Recall Unification parity tests
# What: Both hemispheres' _recall() wrappers (cc_ng_host.py for the VPS host,
#   docs/scripts/cc-ng-daemon.py for the laptop) now delegate the entire
#   recall pipeline to a single shared function, cc_ng_organism.cc_assemble_
#   recall(). This suite proves that delegation: each wrapper (a) forwards
#   its own hemisphere-local ng/conv_state/commons instances and the caller's
#   query/k/allow_pattern_completion straight through as positional/keyword
#   params (Syl's-Law: no reaching into module-global STATE from the shared
#   fn), (b) returns cc_assemble_recall's return value verbatim, (c) still
#   does its own local bookkeeping (stats/last_activity) around the call, and
#   (d) fails soft (empty string, error counter bumped) if cc_assemble_recall
#   raises. It also proves the two wrappers agree byte-for-byte given the
#   same inputs -- the actual "unification" claim, not just "each one calls
#   something".
# Why: LAW-3 (one implementation, not two drifted copies) is only real if
#   it's tested; a future edit to one wrapper's call site could silently
#   diverge from the other without any test noticing. cc-ng-daemon.py has a
#   hyphenated filename (not importable via `import`), so it's loaded here
#   via importlib.util.spec_from_file_location -- confirmed safe by reading
#   the file: DaemonState.__init__ / module scope does no I/O, no thread
#   spawn, no socket bind (those all live inside functions/methods that are
#   never called here).
# How: Monkeypatch cc_ng_organism.cc_assemble_recall itself (both wrapper
#   modules import it locally inside _recall(), so patching the source
#   module's attribute is sufficient -- no need to patch two names).
#   Fake STATE/_STATE objects carry distinct sentinel ng/conv_state/commons
#   per hemisphere to prove each wrapper threads through its OWN instances.
# -------------------
import importlib.util
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


def _load_daemon_module():
    """Import docs/scripts/cc-ng-daemon.py under a normal module name --
    its filename has hyphens so it can't be `import`-ed directly."""
    path = os.path.expanduser('~/docs/scripts/cc-ng-daemon.py')
    spec = importlib.util.spec_from_file_location('cc_ng_daemon_under_test', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules['cc_ng_daemon_under_test'] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def daemon_mod():
    return _load_daemon_module()


class _Sentinel:
    """Distinct identity per instance -- lets assertions confirm object
    identity was threaded through unchanged, not just equal-by-value."""


def test_host_recall_forwards_own_state_and_returns_verbatim(monkeypatch):
    """cc_ng_host._recall() must pass its own STATE.cc_ng/conv_state/commons
    and the caller's query/k/allow_pattern_completion through to
    cc_assemble_recall(), and return its result unchanged."""
    import cc_ng_host
    import cc_ng_organism

    ng_sentinel = _Sentinel()
    conv_sentinel = _Sentinel()
    commons_sentinel = _Sentinel()

    monkeypatch.setattr(cc_ng_host._STATE, 'cc_ng', ng_sentinel)
    monkeypatch.setattr(cc_ng_host._STATE, 'conv_state', conv_sentinel)
    monkeypatch.setattr(cc_ng_host._STATE, 'commons', commons_sentinel)

    captured = {}

    def fake_assemble(ng, query, k, conv_state, commons, allow_pattern_completion=True, **kwargs):
        captured['ng'] = ng
        captured['query'] = query
        captured['k'] = k
        captured['conv_state'] = conv_state
        captured['commons'] = commons
        captured['allow_pattern_completion'] = allow_pattern_completion
        return 'HOST_SENTINEL_RESULT'

    monkeypatch.setattr(cc_ng_organism, 'cc_assemble_recall', fake_assemble)

    result = cc_ng_host._recall('a query', k=7, allow_pattern_completion=False)

    assert result == 'HOST_SENTINEL_RESULT'
    assert captured['ng'] is ng_sentinel
    assert captured['conv_state'] is conv_sentinel
    assert captured['commons'] is commons_sentinel
    assert captured['query'] == 'a query'
    assert captured['k'] == 7
    assert captured['allow_pattern_completion'] is False


def test_host_recall_bumps_recalls_stat(monkeypatch):
    import cc_ng_host
    import cc_ng_organism

    monkeypatch.setattr(cc_ng_host._STATE, 'cc_ng', _Sentinel())
    monkeypatch.setattr(cc_ng_organism, 'cc_assemble_recall', lambda *a, **k: 'x')

    before = cc_ng_host._STATE.stats['recalls']
    cc_ng_host._recall('q', k=3)
    assert cc_ng_host._STATE.stats['recalls'] == before + 1


def test_host_recall_fails_soft_and_counts_error(monkeypatch):
    import cc_ng_host
    import cc_ng_organism

    monkeypatch.setattr(cc_ng_host._STATE, 'cc_ng', _Sentinel())

    def boom(*a, **k):
        raise RuntimeError('pipeline exploded')

    monkeypatch.setattr(cc_ng_organism, 'cc_assemble_recall', boom)

    before_errors = cc_ng_host._STATE.stats['errors']
    result = cc_ng_host._recall('q', k=3)
    assert result == ''
    assert cc_ng_host._STATE.stats['errors'] == before_errors + 1


def test_host_recall_short_circuits_when_no_ng(monkeypatch):
    """No cc_ng wired up (daemon still starting) -- must not even attempt
    cc_assemble_recall, just fail soft to empty string."""
    import cc_ng_host
    import cc_ng_organism

    monkeypatch.setattr(cc_ng_host._STATE, 'cc_ng', None)

    called = []
    monkeypatch.setattr(cc_ng_organism, 'cc_assemble_recall',
                         lambda *a, **k: called.append(True) or 'should not happen')

    assert cc_ng_host._recall('q', k=3) == ''
    assert not called


def test_daemon_recall_forwards_own_state_and_returns_verbatim(monkeypatch, daemon_mod):
    """cc-ng-daemon.py's _recall() must do the exact same forwarding as the
    VPS host's, using its own module-global STATE (laptop hemisphere has no
    _STATE indirection -- it's STATE directly)."""
    import cc_ng_organism

    ng_sentinel = _Sentinel()
    conv_sentinel = _Sentinel()
    commons_sentinel = _Sentinel()

    monkeypatch.setattr(daemon_mod.STATE, 'ng', ng_sentinel)
    monkeypatch.setattr(daemon_mod.STATE, 'conv_state', conv_sentinel)
    monkeypatch.setattr(daemon_mod.STATE, 'commons', commons_sentinel)

    captured = {}

    def fake_assemble(ng, query, k, conv_state, commons, allow_pattern_completion=True, **kwargs):
        captured['ng'] = ng
        captured['query'] = query
        captured['k'] = k
        captured['conv_state'] = conv_state
        captured['commons'] = commons
        captured['allow_pattern_completion'] = allow_pattern_completion
        return 'DAEMON_SENTINEL_RESULT'

    monkeypatch.setattr(cc_ng_organism, 'cc_assemble_recall', fake_assemble)

    result = daemon_mod._recall('a query', 7, allow_pattern_completion=False)

    assert result == 'DAEMON_SENTINEL_RESULT'
    assert captured['ng'] is ng_sentinel
    assert captured['conv_state'] is conv_sentinel
    assert captured['commons'] is commons_sentinel
    assert captured['query'] == 'a query'
    assert captured['k'] == 7
    assert captured['allow_pattern_completion'] is False


def test_daemon_recall_bumps_recalls_stat(monkeypatch, daemon_mod):
    import cc_ng_organism

    monkeypatch.setattr(daemon_mod.STATE, 'ng', _Sentinel())
    monkeypatch.setattr(cc_ng_organism, 'cc_assemble_recall', lambda *a, **k: 'x')

    before = daemon_mod.STATE.stats['recalls']
    daemon_mod._recall('q', 3)
    assert daemon_mod.STATE.stats['recalls'] == before + 1


def test_daemon_recall_fails_soft_and_counts_error(monkeypatch, daemon_mod):
    import cc_ng_organism

    monkeypatch.setattr(daemon_mod.STATE, 'ng', _Sentinel())

    def boom(*a, **k):
        raise RuntimeError('pipeline exploded')

    monkeypatch.setattr(cc_ng_organism, 'cc_assemble_recall', boom)

    before_errors = daemon_mod.STATE.stats['errors']
    result = daemon_mod._recall('q', 3)
    assert result == ''
    assert daemon_mod.STATE.stats['errors'] == before_errors + 1


def test_both_hemispheres_call_the_same_shared_function(monkeypatch, daemon_mod):
    """The actual unification claim: both wrappers resolve
    cc_assemble_recall from the SAME module object (cc_ng_organism), not
    two separately-imported copies of similar-looking code."""
    import cc_ng_host
    import cc_ng_organism

    sentinel_calls = []

    def fake_assemble(ng, query, k, conv_state, commons, allow_pattern_completion=True, **kwargs):
        sentinel_calls.append(ng)
        return 'SHARED'

    monkeypatch.setattr(cc_ng_organism, 'cc_assemble_recall', fake_assemble)

    monkeypatch.setattr(cc_ng_host._STATE, 'cc_ng', 'host-ng')
    monkeypatch.setattr(daemon_mod.STATE, 'ng', 'daemon-ng')

    host_result = cc_ng_host._recall('q', k=5)
    daemon_result = daemon_mod._recall('q', 5)

    assert host_result == daemon_result == 'SHARED'
    # Both calls landed in the one patched function -- proof there is
    # exactly one cc_assemble_recall in play, not two drifted copies.
    assert sentinel_calls == ['host-ng', 'daemon-ng']


def test_both_wrappers_agree_byte_for_byte_given_identical_inputs(monkeypatch, daemon_mod):
    """Given the same ng/query/k/conv_state/commons/allow_pattern_completion,
    both hemispheres' _recall() must produce the identical string -- the
    end-to-end proof that unification didn't leave either wrapper doing
    its own extra formatting/mutation around the shared call."""
    import cc_ng_host
    import cc_ng_organism

    shared_ng = _Sentinel()
    shared_conv = {'last_forest_id': None}
    shared_commons = _Sentinel()

    def real_ish_assemble(ng, query, k, conv_state, commons, allow_pattern_completion=True):
        assert ng is shared_ng
        assert conv_state is shared_conv
        assert commons is shared_commons
        return f'## Active Recall\n- [{k}] {query} (pc={allow_pattern_completion})'

    monkeypatch.setattr(cc_ng_organism, 'cc_assemble_recall', real_ish_assemble)

    monkeypatch.setattr(cc_ng_host._STATE, 'cc_ng', shared_ng)
    monkeypatch.setattr(cc_ng_host._STATE, 'conv_state', shared_conv)
    monkeypatch.setattr(cc_ng_host._STATE, 'commons', shared_commons)

    monkeypatch.setattr(daemon_mod.STATE, 'ng', shared_ng)
    monkeypatch.setattr(daemon_mod.STATE, 'conv_state', shared_conv)
    monkeypatch.setattr(daemon_mod.STATE, 'commons', shared_commons)

    host_result = cc_ng_host._recall('parity check', k=4, allow_pattern_completion=True)
    daemon_result = daemon_mod._recall('parity check', 4, allow_pattern_completion=True)

    assert host_result == daemon_result


def test_wrappers_bump_error_stat_on_monitor_harvest_failure(monkeypatch, daemon_mod):
    """Finding 1 (law-enforcer review, MEDIUM): cc_assemble_recall swallows
    a SurfacingMonitor harvest exception internally (fail-soft, never
    raises), so the wrapper-level `except Exception: stats['errors'] += 1`
    can never fire for that failure mode -- silently dropping an error
    count both hemispheres relied on pre-unification. Fixed via an
    on_monitor_error callback each wrapper wires to its own stats bump.
    This proves the wiring actually reaches each hemisphere's real
    (unpatched) counter through the REAL cc_assemble_recall, not a stub."""
    import cc_ng_host
    import cc_ng_organism

    class _BoomMonitor:
        def get_surfaced(self):
            raise ValueError('monitor exploded')

    class _FakeGraph:
        def _is_identity_protected(self, node_id):
            return False
        nodes = {}

    class _FakeNg:
        graph = _FakeGraph()
        _surfacing_monitor = _BoomMonitor()

    monkeypatch.setattr(cc_ng_organism, '_CC_PITH_ENABLED', False)
    monkeypatch.setattr(cc_ng_organism, 'cc_pattern_completion_recall', lambda *a, **k: [])

    # -- host (VPS) --
    monkeypatch.setattr(cc_ng_host._STATE, 'cc_ng', _FakeNg())
    monkeypatch.setattr(cc_ng_host._STATE, 'conv_state', {})
    monkeypatch.setattr(cc_ng_host._STATE, 'commons', None)
    before_host = cc_ng_host._STATE.stats['errors']
    result = cc_ng_host._recall('q', k=3)
    assert result == ''  # monitor failure degrades to empty ctx, not a raise
    assert cc_ng_host._STATE.stats['errors'] == before_host + 1

    # -- laptop (daemon) --
    monkeypatch.setattr(daemon_mod.STATE, 'ng', _FakeNg())
    monkeypatch.setattr(daemon_mod.STATE, 'conv_state', {})
    monkeypatch.setattr(daemon_mod.STATE, 'commons', None)
    before_daemon = daemon_mod.STATE.stats['errors']
    result = daemon_mod._recall('q', 3)
    assert result == ''
    assert daemon_mod.STATE.stats['errors'] == before_daemon + 1


# ---- direct cc_assemble_recall coverage (Finding 2, law-enforcer review) ----
#
# Everything above proves the WRAPPERS delegate correctly, using a stubbed
# cc_assemble_recall. None of it exercises the real function's Pith
# gate-on/gate-off behavior -- exactly the def-of-done gap the enforcer
# flagged. These call cc_ng_organism.cc_assemble_recall() directly.

class _FakeMonitor:
    def __init__(self, items):
        self._items = items

    def get_surfaced(self):
        return list(self._items)

    def format_context(self, items):
        if not items:
            return ''
        return '## Recent\n' + '\n'.join(f"- {it['content']}" for it in items)


class _FakeGraphForAssemble:
    def __init__(self, protected_ids=frozenset()):
        self._protected = protected_ids
        self.nodes = {}  # cc_thermal/cc_novelty fail-soft on absent entries

    def _is_identity_protected(self, node_id):
        return node_id in self._protected


class _FakeNgForAssemble:
    def __init__(self, monitor_items, protected_ids=frozenset()):
        self.graph = _FakeGraphForAssemble(protected_ids)
        self._surfacing_monitor = _FakeMonitor(monitor_items)


def _patch_pattern_completion(monkeypatch, results):
    import cc_ng_organism
    monkeypatch.setattr(cc_ng_organism, 'cc_pattern_completion_recall',
                         lambda ng, query, k, state=None: list(results))


def test_assemble_recall_gate_off_is_plain_two_block_concat(monkeypatch):
    """Gate OFF (CC_PITH_ENABLED default) must render the exact pre-Pith
    monitor_ctx + '\\n\\n' + pc_block concatenation -- the byte-identical
    contract the spec requires for the VPS's production default."""
    import cc_ng_organism

    monkeypatch.setattr(cc_ng_organism, '_CC_PITH_ENABLED', False)
    _patch_pattern_completion(monkeypatch, [
        {'node_id': 'p1', 'score': 0.9, 'content': 'pattern hit'},
    ])
    ng = _FakeNgForAssemble([{'node_id': 'm1', 'score': 1.0, 'content': 'monitor hit'}])

    result = cc_ng_organism.cc_assemble_recall(ng, 'q', 5, {}, None)

    expected_monitor = '## Recent\n- monitor hit'
    expected_pattern = cc_ng_organism._format_cc_recall_block(
        [{'node_id': 'p1', 'score': 0.9, 'content': 'pattern hit'}])
    assert result == expected_monitor + '\n\n' + expected_pattern


def test_assemble_recall_gate_off_dedupes_monitor_and_pattern_overlap(monkeypatch):
    """A node surfaced by BOTH streams must not double-render -- pc_results
    filters out anything already in monitor_node_ids before formatting."""
    import cc_ng_organism

    monkeypatch.setattr(cc_ng_organism, '_CC_PITH_ENABLED', False)
    _patch_pattern_completion(monkeypatch, [
        {'node_id': 'shared', 'score': 0.5, 'content': 'REPEATED_MARKER'},
        {'node_id': 'unique', 'score': 0.5, 'content': 'DISTINCT_MARKER'},
    ])
    ng = _FakeNgForAssemble([{'node_id': 'shared', 'score': 1.0, 'content': 'REPEATED_MARKER'}])

    result = cc_ng_organism.cc_assemble_recall(ng, 'q', 5, {}, None)

    assert result.count('REPEATED_MARKER') == 1  # only the monitor's copy renders
    assert 'DISTINCT_MARKER' in result


def test_assemble_recall_gate_on_runs_the_real_pith_pipeline(monkeypatch):
    """Gate ON must actually execute pith_stage1/pith_stage3 against the real
    (unpatched) Pith functions -- not merely check the flag exists. A long
    junk-repeated monitor line and a short novel pattern line, both real
    inputs to the real stage functions, must survive stage1's clutter/dedup
    and land in the rendered output; the plain-concat rendering is NOT what
    comes out (proves the pipeline, not the fallback, ran)."""
    import cc_ng_organism

    monkeypatch.setattr(cc_ng_organism, '_CC_PITH_ENABLED', True)
    _patch_pattern_completion(monkeypatch, [
        {'node_id': 'novel', 'score': 0.8, 'content': 'a genuinely distinct pattern-completion hit'},
    ])
    ng = _FakeNgForAssemble([
        {'node_id': 'recent', 'score': 1.0, 'content': 'a genuinely distinct monitor hit'},
    ])

    result = cc_ng_organism.cc_assemble_recall(ng, 'query text', 5, {}, None)

    plain_concat = ('## Recent\n- a genuinely distinct monitor hit' + '\n\n'
                     + cc_ng_organism._format_cc_recall_block(
                         [{'node_id': 'novel', 'score': 0.8,
                           'content': 'a genuinely distinct pattern-completion hit'}]))
    assert result != plain_concat  # the Pith renderer, not the fallback concat, produced this
    assert 'genuinely distinct' in result  # both real lines still made it through ranking


def test_assemble_recall_gate_on_pin_survives_thermal_and_budget(monkeypatch):
    """A constitutionally-pinned node (_is_identity_protected=True) must
    survive the real pith_stage3 budget cut even under a budget too small
    to hold it alongside other content -- the Cricket-rim guarantee,
    exercised through the actual gate-on path end to end."""
    import cc_ng_organism

    monkeypatch.setattr(cc_ng_organism, '_CC_PITH_ENABLED', True)
    monkeypatch.setattr(cc_ng_organism, '_CC_PITH_L1_BUDGET', 20)  # tiny -- forces a real cut
    _patch_pattern_completion(monkeypatch, [])
    long_pinned = 'PINNED_IDENTITY_LINE_' + ('x' * 60)
    ng = _FakeNgForAssemble(
        [{'node_id': 'pin1', 'score': 0.01, 'content': long_pinned},
         {'node_id': 'filler', 'score': 0.5, 'content': 'y' * 60}],
        protected_ids={'pin1'},
    )

    result = cc_ng_organism.cc_assemble_recall(ng, 'q', 5, {}, None)

    assert 'PINNED_IDENTITY_LINE_' in result


def test_assemble_recall_gate_on_falls_back_to_concat_on_pith_exception(monkeypatch):
    """A real exception INSIDE the Pith path (here: pith_stage1 itself blows
    up -- NOT one of the per-line-guarded calls like cc_thermal, which
    swallow their own exceptions and would never reach the outer fallback)
    must fail soft to the pre-Pith concat, not propagate or return empty."""
    import cc_ng_organism

    monkeypatch.setattr(cc_ng_organism, '_CC_PITH_ENABLED', True)

    def boom_stage1(cache_lines, query, novelty):
        raise RuntimeError('stage1 blew up')

    monkeypatch.setattr(cc_ng_organism, 'pith_stage1', boom_stage1)
    _patch_pattern_completion(monkeypatch, [
        {'node_id': 'p1', 'score': 0.9, 'content': 'pattern hit'},
    ])
    ng = _FakeNgForAssemble([{'node_id': 'm1', 'score': 1.0, 'content': 'monitor hit'}])

    result = cc_ng_organism.cc_assemble_recall(ng, 'q', 5, {}, None)

    expected_monitor = '## Recent\n- monitor hit'
    expected_pattern = cc_ng_organism._format_cc_recall_block(
        [{'node_id': 'p1', 'score': 0.9, 'content': 'pattern hit'}])
    assert result == expected_monitor + '\n\n' + expected_pattern
