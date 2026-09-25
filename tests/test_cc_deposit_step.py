# ---- Changelog ----
# [2026-09-25] Z2 zone manager (Claude Opus 5.5, Claude Code) — lane C (ii-a)
# What: tests for cc_ng_organism.cc_deposit_step and its two wrappers
#   (cc_ng_host._deposit, docs/scripts/cc-ng-daemon.py _deposit).
# Why: the conversational deposit steps again (LAW 3 restore of on_message()'s
#   step + baseline reward, #543 discovery on the step's own fired set), behind
#   CC_NG_DEPOSIT_STEP (default off). Chief rulings D1-D4.
# How: fakes record lock and call order; one real Graph checks the clock moves.
#   CC_NG_DAEMON_SCRIPT points the daemon tests at a worktree copy of the script
#   (default ~/docs/scripts/cc-ng-daemon.py).
# -------------------
import importlib.util
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cc_ng_organism  # noqa: E402


class _RecordingLock:
    """Context-manager lock that logs enter/exit into a shared event list."""

    def __init__(self, name, events):
        self.name = name
        self.events = events
        self.held = False

    def __enter__(self):
        self.events.append(('enter', self.name))
        self.held = True
        return self

    def __exit__(self, *exc):
        self.held = False
        self.events.append(('exit', self.name))
        return False


class _FakeGraph:
    def __init__(self, fired=('n1', 'n2'), three_factor=True, step_raises=None):
        self.events = []
        self._step_lock = _RecordingLock('step', self.events)
        self._concurrent_lock = _RecordingLock('concurrent', self.events)
        self.config = {'three_factor_enabled': three_factor}
        self._fired = list(fired)
        self._step_raises = step_raises
        self.timestep = 7
        self._recent_spikes = {'stale': [7], 'older': [3]}
        self.discovered = []
        self.rewards = []

    def step(self):
        self.events.append(('step', self._step_lock.held))
        if self._step_raises:
            raise self._step_raises
        self.timestep += 1
        return types.SimpleNamespace(fired_node_ids=list(self._fired))

    def inject_reward(self, strength, scope=None):
        self.events.append(('reward', self._step_lock.held))
        self.rewards.append(strength)

    def discover_hyperedges(self, fired):
        self.events.append(('discover', self._step_lock.held))
        self.discovered.append(list(fired))
        return []


# ---- cc_deposit_step ----

def test_default_flag_is_off():
    env = os.environ.get('CC_NG_DEPOSIT_STEP')
    if env not in (None, '', '0', 'false', 'False'):
        pytest.skip('CC_NG_DEPOSIT_STEP set in this environment')
    assert cc_ng_organism._CC_NG_DEPOSIT_STEP is False


def test_steps_rewards_then_discovers_on_the_steps_fired_set_under_step_lock():
    g = _FakeGraph(fired=('n1', 'n2'))
    result = cc_ng_organism.cc_deposit_step(g)
    assert result.fired_node_ids == ['n1', 'n2']
    assert g.events == [
        ('enter', 'step'), ('step', True), ('reward', True),
        ('discover', True), ('exit', 'step'),
    ]
    assert g.rewards == [0.1]
    assert g.discovered == [['n1', 'n2']]


def test_reward_is_three_factor_gated():
    g = _FakeGraph(three_factor=False)
    cc_ng_organism.cc_deposit_step(g)
    assert g.rewards == []
    assert ('step', True) in g.events


def test_nothing_fired_means_no_discovery():
    g = _FakeGraph(fired=())
    result = cc_ng_organism.cc_deposit_step(g)
    assert result.fired_node_ids == []
    assert g.discovered == []


def test_step_failure_is_soft_and_skips_reward_and_discovery():
    g = _FakeGraph(step_raises=RuntimeError('boom'))
    assert cc_ng_organism.cc_deposit_step(g) is None
    assert g.rewards == [] and g.discovered == []
    assert g.events[-1] == ('exit', 'step')


def test_none_graph_returns_none():
    assert cc_ng_organism.cc_deposit_step(None) is None


def test_real_graph_clock_advances_once_per_deposit_step():
    from neuro_foundation import Graph
    g = Graph()
    g.create_node(node_id='a')
    before = g.timestep
    result = cc_ng_organism.cc_deposit_step(g)
    assert g.timestep == before + 1
    assert hasattr(result, 'fired_node_ids')


# ---- cc_ng_host._deposit ----

@pytest.fixture
def host(monkeypatch):
    import cc_ng_host
    import ng_embed
    g = _FakeGraph()
    order = []

    def fake_dual_pass(graph, vdb, text, emb, state):
        g.events.append(('dual_pass', graph._concurrent_lock.held))
        order.append('dual_pass')
        return fake_dual_pass.ok
    fake_dual_pass.ok = True

    monkeypatch.setattr(cc_ng_host._STATE, 'cc_ng',
                        types.SimpleNamespace(graph=g, vector_db=None))
    monkeypatch.setattr(ng_embed, 'embed', lambda text: [0.0])
    monkeypatch.setattr(cc_ng_organism, 'run_conversational_dual_pass', fake_dual_pass)
    monkeypatch.setattr(cc_ng_organism, 'deposit_cc_experience', lambda *a, **k: None)
    monkeypatch.setattr(cc_ng_organism, 'surface_wants_for_graph', lambda *a, **k: [])
    return types.SimpleNamespace(mod=cc_ng_host, graph=g, dual_pass=fake_dual_pass)


def test_host_flag_on_steps_after_dual_pass_inside_concurrent_lock(host, monkeypatch):
    monkeypatch.setattr(cc_ng_organism, '_CC_NG_DEPOSIT_STEP', True)
    host.mod._deposit('a turn')
    ev = host.graph.events
    assert ev[:3] == [('enter', 'concurrent'), ('dual_pass', True), ('enter', 'step')]
    assert ev[-2:] == [('exit', 'step'), ('exit', 'concurrent')]
    # #543: discovery gets the step's fired set, not the stale _recent_spikes read.
    assert host.graph.discovered == [['n1', 'n2']]


def test_host_flag_on_steps_even_when_the_dual_pass_fails(host, monkeypatch):
    monkeypatch.setattr(cc_ng_organism, '_CC_NG_DEPOSIT_STEP', True)
    host.dual_pass.ok = False
    host.mod._deposit('a turn')
    assert ('step', True) in host.graph.events


def test_host_flag_off_is_the_previous_path(host, monkeypatch):
    monkeypatch.setattr(cc_ng_organism, '_CC_NG_DEPOSIT_STEP', False)
    host.mod._deposit('a turn')
    ev = host.graph.events
    assert not any(e[0] == 'step' for e in ev)
    assert host.graph.rewards == []
    # Previous discovery: nodes whose last spike equals the current timestep.
    assert host.graph.discovered == [['stale']]
    assert ev == [('enter', 'concurrent'), ('dual_pass', True),
                  ('discover', False), ('exit', 'concurrent')]


def test_host_step_failure_does_not_count_a_deposit_error(host, monkeypatch):
    monkeypatch.setattr(cc_ng_organism, '_CC_NG_DEPOSIT_STEP', True)
    host.graph._step_raises = RuntimeError('boom')
    before = host.mod._STATE.stats['errors']
    host.mod._deposit('a turn')
    assert host.mod._STATE.stats['errors'] == before


# ---- laptop daemon _deposit ----

def _load_daemon():
    path = os.environ.get('CC_NG_DAEMON_SCRIPT',
                          os.path.expanduser('~/docs/scripts/cc-ng-daemon.py'))
    spec = importlib.util.spec_from_file_location('cc_ng_daemon_deposit_step', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules['cc_ng_daemon_deposit_step'] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def daemon(monkeypatch):
    mod = _load_daemon()
    import ng_embed
    g = _FakeGraph()

    def fake_dual_pass(graph, vdb, text, emb, state):
        g.events.append(('dual_pass', graph._concurrent_lock.held))
        return True

    monkeypatch.setattr(mod.STATE, 'ng', types.SimpleNamespace(graph=g, vector_db=None))
    monkeypatch.setattr(ng_embed, 'embed', lambda text: [0.0])
    monkeypatch.setattr(cc_ng_organism, 'run_conversational_dual_pass', fake_dual_pass)
    monkeypatch.setattr(cc_ng_organism, 'deposit_cc_experience', lambda *a, **k: None)
    return types.SimpleNamespace(mod=mod, graph=g)


def test_daemon_flag_on_steps_after_dual_pass_inside_concurrent_lock(daemon, monkeypatch):
    monkeypatch.setattr(cc_ng_organism, '_CC_NG_DEPOSIT_STEP', True)
    daemon.mod._deposit('a turn')
    ev = daemon.graph.events
    assert ev[:3] == [('enter', 'concurrent'), ('dual_pass', True), ('enter', 'step')]
    assert ev[-2:] == [('exit', 'step'), ('exit', 'concurrent')]
    assert daemon.graph.rewards == [0.1]
    assert daemon.graph.discovered == [['n1', 'n2']]


def test_daemon_flag_off_is_the_previous_path(daemon, monkeypatch):
    monkeypatch.setattr(cc_ng_organism, '_CC_NG_DEPOSIT_STEP', False)
    daemon.mod._deposit('a turn')
    assert daemon.graph.events == [('enter', 'concurrent'), ('dual_pass', True),
                                   ('exit', 'concurrent')]
