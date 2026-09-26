# ---- Changelog ----
# [2026-09-26] openrouter/deepseek/deepseek-v4.1-flash (OpenCode harness on T3 Code),
#   lane z2-one-step-per-turn-001 — Exec P240(2)/P241: one step per turn
# What: host-section tests updated for the `step` parameter on cc_ng_host._deposit.
#   The three former test_host_flag_on_* tests are renamed test_host_stop_side_*
#   and call `_deposit('a turn', step=True)`. New:
#   test_host_prompt_side_does_not_step_even_with_flag_on,
#   test_one_turn_steps_exactly_once (UserPromptSubmit + Stop -> exactly one step),
#   test_post_tool_use_never_steps. test_host_flag_off_is_the_previous_path also
#   asserts the flag-off non-step path for step=True.
# Why: Chief-p240-commission-001 row z2-one-step-per-turn-001; P241 (Lanes 1 and 2
#   land together); Z11 return (PostToolUse does not step); canonical cardinality:
#   Syl's handle_after_turn does exactly one graph.step() per turn.
# How: drives the real _deposit and dispatch table with an inline Thread that
#   forwards kwargs; host-level _nudge/_recall and organism-level
#   render_constitutional_core/render_wants are patched to no-ops. The daemon
#   section is deliberately untouched (its twin cc-ng-daemon.py is Z12's item).
# -------------------
# [2026-09-26] GLM (z-ai/glm-5.3-flash, OpenCode harness on T3 Code),
#   lane z2-b3-kiss-drain-step-001 — the drains step + #643 lock scope
# What: new tests for the two drains calling cc_deposit_step once per APPLIED
#   record under _CC_NG_DEPOSIT_STEP (flag on: exactly one timestep per
#   absorbed/applied record, none on a failed dual pass or an already-applied
#   re-drain; flag off: timestep unchanged), and for punchlist #643: the
#   autosave loop's drain_ingest_tract call runs under
#   graph._concurrent_lock (cross-thread acquire(blocking=False) probe inside
#   the wrapped dual pass; that assertion fails on 2e58509).
# Why: Chief B3 ruling 001 (docs 0ef6dac1) R1/R2/R3; P153(4) Q3; KISS.md:38
#   (op 1 at Apprentice, the Delta Gate on graph data); punchlist #643;
#   assignment z2-b3-kiss-drain-step-001.
# How: every test runs the real cc_ng_organism drain functions and a real
#   graph; run_conversational_dual_pass is faked (TID extraction is
#   unreachable in the test env, the same condition behind the pre-existing
#   dual-pass failures) because it is a dependency, not the code under test.
#   The Leg-1 tests mirror tests/test_cc_gateway_durable.py's durable rig
#   setup but call the real module function: the durable rig AST-extracts
#   the drain into a bare namespace the mandated module-level
#   monkeypatch.setattr cannot reach, and its SimpleNamespace graph has no
#   step clock. cc_ng is imported from tests/test_cc_dual_pass.py (same
#   package-relative pattern as tests/test_cc_durable_integration.py).
# -------------------
# [2026-09-25] Z2 zone manager (Claude Opus 5.5, Claude Code) — lane C (ii-a)
# What: tests for cc_ng_organism.cc_deposit_step and its two wrappers
#   (cc_ng_host._deposit, docs/scripts/cc-ng-daemon.py _deposit).
# Why: the conversational deposit steps again (LAW 3 restore of on_message()'s
#   step + baseline reward, #543 discovery on the step's own fired set), behind
#   CC_NG_DEPOSIT_STEP (default off). Chief rulings D1-D4.
# How: fakes record lock and call order; one real Graph checks the clock moves.
#   R1 (Chief): the reward is on_message()'s success-path form -- a failed dual
#   pass still steps but earns no reward (pinned in the organism, host and daemon).
#   CC_NG_DAEMON_SCRIPT points the daemon tests at a worktree copy of the script
#   (default ~/docs/scripts/cc-ng-daemon.py).
# -------------------
import importlib.util
import os
import sys
import threading
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cc_ng_organism  # noqa: E402
from .test_cc_dual_pass import cc_ng  # noqa: E402,F401  (pytest fixture)


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
    result = cc_ng_organism.cc_deposit_step(g, True)
    assert result.fired_node_ids == ['n1', 'n2']
    assert g.events == [
        ('enter', 'step'), ('step', True), ('reward', True),
        ('discover', True), ('exit', 'step'),
    ]
    assert g.rewards == [0.1]
    assert g.discovered == [['n1', 'n2']]


def test_reward_is_three_factor_gated():
    g = _FakeGraph(three_factor=False)
    cc_ng_organism.cc_deposit_step(g, True)
    assert g.rewards == []
    assert ('step', True) in g.events


def test_failed_deposit_still_steps_but_earns_no_reward():
    # R1: on_message()'s success-path form (openclaw_hook:1226).
    g = _FakeGraph(fired=('n1',))
    result = cc_ng_organism.cc_deposit_step(g, False)
    assert g.rewards == []
    assert g.events == [
        ('enter', 'step'), ('step', True), ('discover', True), ('exit', 'step'),
    ]
    assert result.fired_node_ids == ['n1']


def test_nothing_fired_means_no_discovery():
    g = _FakeGraph(fired=())
    result = cc_ng_organism.cc_deposit_step(g, True)
    assert result.fired_node_ids == []
    assert g.discovered == []


def test_step_failure_is_soft_and_skips_reward_and_discovery():
    g = _FakeGraph(step_raises=RuntimeError('boom'))
    assert cc_ng_organism.cc_deposit_step(g, True) is None
    assert g.rewards == [] and g.discovered == []
    assert g.events[-1] == ('exit', 'step')


def test_none_graph_returns_none():
    assert cc_ng_organism.cc_deposit_step(None, True) is None


def test_real_graph_clock_advances_once_per_deposit_step():
    from neuro_foundation import Graph
    g = Graph()
    g.create_node(node_id='a')
    before = g.timestep
    result = cc_ng_organism.cc_deposit_step(g, True)
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


def test_host_stop_side_steps_after_dual_pass_inside_concurrent_lock(host, monkeypatch):
    monkeypatch.setattr(cc_ng_organism, '_CC_NG_DEPOSIT_STEP', True)
    host.mod._deposit('a turn', step=True)
    ev = host.graph.events
    assert ev[:3] == [('enter', 'concurrent'), ('dual_pass', True), ('enter', 'step')]
    assert ev[-2:] == [('exit', 'step'), ('exit', 'concurrent')]
    # #543: discovery gets the step's fired set, not the stale _recent_spikes read.
    assert host.graph.discovered == [['n1', 'n2']]


def test_host_stop_side_steps_even_when_the_dual_pass_fails(host, monkeypatch):
    monkeypatch.setattr(cc_ng_organism, '_CC_NG_DEPOSIT_STEP', True)
    host.dual_pass.ok = False
    host.mod._deposit('a turn', step=True)
    assert ('step', True) in host.graph.events
    # R1: the failed turn is still a timestep, but earns no reward.
    assert host.graph.rewards == []


def test_host_stop_side_rewards_a_landed_turn(host, monkeypatch):
    monkeypatch.setattr(cc_ng_organism, '_CC_NG_DEPOSIT_STEP', True)
    host.mod._deposit('a turn', step=True)
    assert host.graph.rewards == [0.1]


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
    # Flag off stays the non-step path even when the Stop side asks for a step.
    host.graph.events.clear()
    host.graph.discovered.clear()
    host.graph.rewards.clear()
    host.mod._deposit('a turn', step=True)
    ev = host.graph.events
    assert not any(e[0] == 'step' for e in ev)
    assert host.graph.rewards == []
    assert host.graph.discovered == [['stale']]
    assert ev == [('enter', 'concurrent'), ('dual_pass', True),
                  ('discover', False), ('exit', 'concurrent')]


def test_host_prompt_side_does_not_step_even_with_flag_on(host, monkeypatch):
    monkeypatch.setattr(cc_ng_organism, '_CC_NG_DEPOSIT_STEP', True)
    host.mod._deposit('a turn')
    ev = host.graph.events
    assert not any(e[0] == 'step' for e in ev)
    assert host.graph.rewards == []
    assert ev == [('enter', 'concurrent'), ('dual_pass', True),
                  ('discover', False), ('exit', 'concurrent')]


def test_one_turn_steps_exactly_once(host, monkeypatch):
    monkeypatch.setattr(cc_ng_organism, '_CC_NG_DEPOSIT_STEP', True)
    monkeypatch.setattr(host.mod, '_nudge', lambda text: None)
    monkeypatch.setattr(host.mod, '_recall', lambda *a, **k: "")
    monkeypatch.setattr(cc_ng_organism, 'render_constitutional_core',
                        lambda *a, **k: "")
    monkeypatch.setattr(cc_ng_organism, 'render_wants', lambda *a, **k: [])

    original_thread = threading.Thread
    thread_calls = []

    def inline_thread(target=None, args=(), daemon=False, kwargs=None, **extra):
        if target:
            target(*args, **(kwargs or {}))
        thread_obj = original_thread()
        thread_calls.append((target, args, kwargs))
        return thread_obj

    monkeypatch.setattr(host.mod.threading, 'Thread', inline_thread)

    host.mod._DISPATCH["UserPromptSubmit"]({"prompt": "a prompt"})
    host.mod._DISPATCH["Stop"]({"last_assistant_message": "a reply"})

    step_enters = [e for e in host.graph.events if e == ('enter', 'step')]
    assert len(step_enters) == 1
    # The one step came from the Stop side (kwargs step=True), not the prompt.
    assert not thread_calls[0][2]
    assert thread_calls[1][2] == {"step": True}


def test_post_tool_use_never_steps(host, monkeypatch):
    monkeypatch.setattr(cc_ng_organism, '_CC_NG_DEPOSIT_STEP', True)
    monkeypatch.setattr(cc_ng_organism, 'deposit_cc_experience',
                        lambda *a, **k: None)
    host.mod._DISPATCH["PostToolUse"]({
        "tool_name": "Read",
        "tool_input": {"file_path": "/tmp/x"},
        "tool_response": "contents",
    })
    assert not any(e[0] == 'step' for e in host.graph.events)


def test_host_step_failure_does_not_count_a_deposit_error(host, monkeypatch):
    monkeypatch.setattr(cc_ng_organism, '_CC_NG_DEPOSIT_STEP', True)
    host.graph._step_raises = RuntimeError('boom')
    before = host.mod._STATE.stats['errors']
    host.mod._deposit('a turn', step=True)
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
        return fake_dual_pass.ok
    fake_dual_pass.ok = True

    monkeypatch.setattr(mod.STATE, 'ng', types.SimpleNamespace(graph=g, vector_db=None))
    monkeypatch.setattr(ng_embed, 'embed', lambda text: [0.0])
    monkeypatch.setattr(cc_ng_organism, 'run_conversational_dual_pass', fake_dual_pass)
    monkeypatch.setattr(cc_ng_organism, 'deposit_cc_experience', lambda *a, **k: None)
    return types.SimpleNamespace(mod=mod, graph=g, dual_pass=fake_dual_pass)


def test_daemon_flag_on_steps_after_dual_pass_inside_concurrent_lock(daemon, monkeypatch):
    monkeypatch.setattr(cc_ng_organism, '_CC_NG_DEPOSIT_STEP', True)
    daemon.mod._deposit('a turn')
    ev = daemon.graph.events
    assert ev[:3] == [('enter', 'concurrent'), ('dual_pass', True), ('enter', 'step')]
    assert ev[-2:] == [('exit', 'step'), ('exit', 'concurrent')]
    assert daemon.graph.rewards == [0.1]
    assert daemon.graph.discovered == [['n1', 'n2']]


def test_daemon_flag_on_failed_dual_pass_steps_without_reward(daemon, monkeypatch):
    monkeypatch.setattr(cc_ng_organism, '_CC_NG_DEPOSIT_STEP', True)
    daemon.dual_pass.ok = False
    daemon.mod._deposit('a turn')
    assert ('step', True) in daemon.graph.events
    assert daemon.graph.rewards == []


def test_daemon_flag_off_is_the_previous_path(daemon, monkeypatch):
    monkeypatch.setattr(cc_ng_organism, '_CC_NG_DEPOSIT_STEP', False)
    daemon.mod._deposit('a turn')
    assert daemon.graph.events == [('enter', 'concurrent'), ('dual_pass', True),
                                   ('exit', 'concurrent')]


# ---- the drains step (B3: assignment z2-b3-kiss-drain-step-001) ----

@pytest.fixture
def leg1(tmp_path, monkeypatch):
    """Real-module Leg-1 conduit rig. Mirrors the durable rig setup in
    tests/test_cc_gateway_durable.py (conduit dir, journal, MACHINE_ID,
    refused save) but calls the real cc_ng_organism.drain_gateway_conduit
    with a real neuro_foundation.Graph: the durable rig AST-extracts the
    drain into a bare namespace that the mandated
    monkeypatch.setattr(cc_ng_organism, '_CC_NG_DEPOSIT_STEP', ...) cannot
    reach, and its SimpleNamespace graph has no step clock to assert on."""
    import ng_embed
    monkeypatch.setattr(cc_ng_organism, '_CC_CALLOSUM_LEG1_ENABLED', True)
    monkeypatch.setattr(ng_embed, 'embed', lambda text: text)
    monkeypatch.setitem(sys.modules, 'cc_refeed', types.SimpleNamespace(
        should_pause_for_load=lambda ceiling: False))
    monkeypatch.setenv('MACHINE_ID', 'vps')
    calls = []

    def dual_pass(graph, vdb, text, emb, state):
        calls.append(text)
        return True

    monkeypatch.setattr(cc_ng_organism, 'run_conversational_dual_pass', dual_pass)

    from neuro_foundation import Graph
    g = Graph()
    if not hasattr(g, '_concurrent_lock'):
        g._concurrent_lock = threading.RLock()

    def refused_save():
        return {'accepted': False, 'outcome': 'retry'}

    def add(texts=('leg one turn one', 'leg one turn two'),
            name='laptop_cc_gateway.1.tract'):
        import ng_tract
        path = tmp_path / 'conduit' / name
        for text in texts:
            ng_tract.deposit_experience(raw=text.encode(), source='cc_gateway',
                                        tract_paths=[str(path)])
        return path

    conduit = tmp_path / 'conduit'
    conduit.mkdir()
    journal = tmp_path / 'local' / 'delivery.sqlite3'
    state = {}

    def drain(**kw):
        args = dict(conduit_dir=str(conduit), journal_path=str(journal),
                    save_callback=refused_save)
        args.update(kw)
        return cc_ng_organism.drain_gateway_conduit(g, None, state, **args)

    return types.SimpleNamespace(graph=g, conduit=conduit, journal=journal,
                                 add=add, drain=drain, state=state, calls=calls)


def test_flag_on_drain_ingest_tract_steps_once_per_absorbed_entry(cc_ng, tmp_path, monkeypatch):
    import ng_tract
    monkeypatch.setattr(cc_ng_organism, '_CC_NG_DEPOSIT_STEP', True)
    tract_path = str(tmp_path / 'turns.tract')
    for raw in (b'turn one lands', b'turn two lands', b'turn three fails the dual pass'):
        ng_tract.deposit_experience(raw=raw, source='cc_gateway', tract_paths=[tract_path])

    def dual_pass(graph, vdb, text, emb, state):
        if 'fails the dual pass' in str(text):
            return False
        return True

    monkeypatch.setattr(cc_ng_organism, 'run_conversational_dual_pass', dual_pass)

    if not hasattr(cc_ng.graph, '_concurrent_lock'):
        cc_ng.graph._concurrent_lock = threading.RLock()
    before = cc_ng.graph.timestep
    with cc_ng.graph._concurrent_lock:
        absorbed = cc_ng_organism.drain_ingest_tract(
            cc_ng.graph, cc_ng.vector_db, {'last_forest_id': None},
            tract_path=tract_path)
    assert absorbed == 2
    assert cc_ng.graph.timestep == before + 2


def test_flag_off_drain_ingest_tract_does_not_step(cc_ng, tmp_path, monkeypatch):
    import ng_tract
    monkeypatch.setattr(cc_ng_organism, '_CC_NG_DEPOSIT_STEP', False)
    tract_path = str(tmp_path / 'turns.tract')
    for raw in (b'flag off turn one', b'flag off turn two'):
        ng_tract.deposit_experience(raw=raw, source='cc_gateway', tract_paths=[tract_path])
    monkeypatch.setattr(cc_ng_organism, 'run_conversational_dual_pass', lambda *a: True)

    if not hasattr(cc_ng.graph, '_concurrent_lock'):
        cc_ng.graph._concurrent_lock = threading.RLock()
    before = cc_ng.graph.timestep
    with cc_ng.graph._concurrent_lock:
        absorbed = cc_ng_organism.drain_ingest_tract(
            cc_ng.graph, cc_ng.vector_db, {'last_forest_id': None},
            tract_path=tract_path)
    assert absorbed == 2
    assert cc_ng.graph.timestep == before


def test_flag_on_drain_gateway_conduit_steps_once_per_applied_record(leg1, monkeypatch):
    monkeypatch.setattr(cc_ng_organism, '_CC_NG_DEPOSIT_STEP', True)
    leg1.add()
    before = leg1.graph.timestep
    result = leg1.drain()
    assert result['applied'] == 2
    assert leg1.graph.timestep == before + 2
    second = leg1.drain()
    assert second['applied'] == 0
    assert second['uncertain'] == 0
    assert len(leg1.calls) == 2
    assert leg1.graph.timestep == before + 2


def test_flag_off_drain_gateway_conduit_does_not_step(leg1, monkeypatch):
    monkeypatch.setattr(cc_ng_organism, '_CC_NG_DEPOSIT_STEP', False)
    leg1.add(texts=('flag off leg one turn',))
    before = leg1.graph.timestep
    result = leg1.drain()
    assert result['applied'] == 1
    assert leg1.graph.timestep == before


def test_autosave_loop_drains_the_tract_under_concurrent_lock_643(cc_ng, tmp_path, monkeypatch):
    import ng_tract
    import cc_ng_host
    tract_path = str(tmp_path / 'turns.tract')
    ng_tract.deposit_experience(raw=b'autosave drain lock probe turn',
                                source='cc_gateway', tract_paths=[tract_path])
    monkeypatch.setenv('CC_GATEWAY_TRACT_PATH', tract_path)

    monkeypatch.setattr(cc_ng_organism, 'cc_update_probation', lambda *a, **k: None)
    monkeypatch.setattr(cc_ng_organism, 'surface_wants', lambda *a, **k: None)
    monkeypatch.setattr(cc_ng_organism, 'generate_emergent_want', lambda *a, **k: None)
    monkeypatch.setattr(cc_ng_organism, 'persist_cc_commons', lambda *a, **k: None)

    monkeypatch.setattr(cc_ng_host._STATE, 'cc_ng', cc_ng)
    monkeypatch.setattr(cc_ng_host._STATE, 'conv_state', {'last_forest_id': None})
    monkeypatch.setattr(cc_ng_host._STATE, 'running', True)
    monkeypatch.setattr(cc_ng_host, 'time', types.SimpleNamespace(sleep=lambda _s: None))

    real_dual_pass = cc_ng_organism.run_conversational_dual_pass
    probe = {'ran': False, 'acquired': None}

    def probing_dual_pass(graph, vdb, text, emb, state):
        probe['ran'] = True
        seen = {}

        def attempt():
            seen['ok'] = graph._concurrent_lock.acquire(blocking=False)
            if seen['ok']:
                graph._concurrent_lock.release()

        thread = threading.Thread(target=attempt)
        thread.start()
        thread.join()
        probe['acquired'] = seen['ok']
        outcome = real_dual_pass(graph, vdb, text, emb, state)
        cc_ng_host._STATE.running = False
        return outcome

    monkeypatch.setattr(cc_ng_organism, 'run_conversational_dual_pass', probing_dual_pass)

    if not hasattr(cc_ng.graph, '_concurrent_lock'):
        cc_ng.graph._concurrent_lock = threading.RLock()

    cc_ng_host._autosave_loop()

    assert probe['ran'] is True
    assert probe['acquired'] is False
