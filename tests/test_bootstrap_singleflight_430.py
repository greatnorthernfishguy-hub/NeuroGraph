# ---- Changelog ----
# [2026-09-12] Codex — #430 concurrency and partial-initialization regressions.
# What: Execute actual bootstrap boundary AST with fake graph constructors.
# Why: A duplicate bootstrap cleared the graph behind a listening CC socket.
# How: Event-controlled overlap, fake sockets/workers; no NG/model imports.
# -------------------
import ast
import logging
from pathlib import Path
import threading
import time
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[1]


def functions(file, names, ns):
    tree = ast.parse((ROOT / file).read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ROOT / file), 'exec'), ns)
    return ns


@pytest.fixture
def cc(tmp_path, monkeypatch):
    import sys
    state = NS(cc_ng=None, stats={}, concept_queue=[], running=False, server_sock=None)
    counts = {'construct': 0, 'organism': 0, 'worker': 0, 'closed': 0}
    control = NS(entered=threading.Event(), release=threading.Event(), pause=False,
                 fail=None)

    class Memory:
        def __init__(self, **kw):
            counts['construct'] += 1
            control.entered.set()
            if control.pause:
                assert control.release.wait(3)
            if control.fail == 'constructor':
                raise RuntimeError('constructor failed')
            self.graph = NS(nodes={}, synapses={}, timestep=0)
            self.vector_db = object()

    class Socket:
        def bind(self, path):
            if control.fail == 'bind':
                raise RuntimeError('bind failed')
        def listen(self, n):
            pass
        def close(self):
            counts['closed'] += 1

    class Worker:
        def __init__(self, **kw):
            pass
        def start(self):
            counts['worker'] += 1
            if control.fail == 'worker':
                raise RuntimeError('worker failed')

    def organism(*a, **kw):
        counts['organism'] += 1

    monkeypatch.setitem(sys.modules, 'openclaw_hook', NS(NeuroGraphMemory=Memory))
    monkeypatch.setitem(sys.modules, 'cc_ng_organism', NS(
        bootstrap_lenia=organism, bootstrap_trisynaptic=lambda *a, **kw: None,
        get_cc_commons=lambda *a: None, cc_stamp_missing_geometry=lambda *a: 0))
    ns = dict(_cc_init_lock=threading.Lock(), _cc_init_attempted=False,
              _cc_init_failed=False, logger=logging.getLogger('test'), _STATE=state,
              Path=Path, CC_NG_WORKSPACE=str(tmp_path), _CC_SNN_CONFIG={},
              threading=NS(RLock=threading.RLock, Thread=Worker), time=time,
              _cleanup_stale_socket=lambda: None,
              socket=NS(socket=lambda *a: Socket(), AF_UNIX=1, SOCK_STREAM=1),
              SOCKET_PATH=str(tmp_path / 'fake'), os=NS(chmod=lambda *a: None),
              _write_refcount=lambda *a: None, _serve_loop=lambda: None,
              _autosave_loop=lambda: None, _start_cc_tonic_idle_watcher=lambda: None,
              _start_cc_dream_consolidation_pulse=lambda: None)
    functions('cc_ng_host.py', {'init_cc_host', '_init_cc_host_once'}, ns)
    return ns, state, counts, control


def test_cc_overlap_constructs_once_and_preserves_graph(cc):
    ns, state, counts, ctl = cc
    ctl.pause = True
    results = []
    second_entered = threading.Event()
    def run(second=False):
        if second:
            second_entered.set()
        results.append(ns['init_cc_host']())
    first = threading.Thread(target=run)
    second = threading.Thread(target=run, args=(True,))
    first.start()
    assert ctl.entered.wait(3)
    second.start()
    assert second_entered.wait(3)
    ctl.release.set()
    first.join(3); second.join(3)
    assert not first.is_alive() and not second.is_alive()
    assert results == [True, True]
    graph = state.cc_ng
    assert ns['init_cc_host']() is True and state.cc_ng is graph
    assert counts == {'construct': 1, 'organism': 1, 'worker': 2, 'closed': 0}


@pytest.mark.parametrize('failure', ['constructor', 'bind', 'worker'])
def test_cc_partial_failure_never_reconstructs_or_claims_ready(cc, failure):
    ns, state, counts, ctl = cc
    ctl.fail = failure
    if failure == 'worker':
        with pytest.raises(RuntimeError):
            ns['init_cc_host']()
    else:
        assert ns['init_cc_host']() is False
    graph, before = state.cc_ng, counts.copy()
    ctl.fail = None
    assert ns['init_cc_host']() is False
    assert state.cc_ng is graph and counts == before
    assert counts['construct'] == 1
    if failure != 'constructor':
        assert graph is not None
    if failure == 'bind':
        assert counts['closed'] == 1


def test_cc_before_constructor_failure_can_retry(cc):
    ns, state, counts, ctl = cc
    real_path = ns['Path']
    class BadPath:
        def __init__(self, *a): pass
        def mkdir(self, **kw): raise OSError('workspace unavailable')
    ns['Path'] = BadPath
    with pytest.raises(OSError): ns['init_cc_host']()
    ns['Path'] = real_path
    assert ns['init_cc_host']() is True
    assert counts['construct'] == 1


def bootstrap_ns():
    ns = dict(Dict=dict, Any=object, _bootstrap_lock=threading.Lock(),
              _bootstrap_incomplete=False, _bootstrap_construction_attempted=False,
              _memory=None)
    functions('neurograph_rpc.py', {'handle_bootstrap'}, ns)
    return ns


def test_rpc_overlap_returns_initializing_without_running_second_body():
    ns = bootstrap_ns()
    entered, release = threading.Event(), threading.Event()
    calls = []
    def body(params):
        calls.append(params)
        entered.set()
        assert release.wait(3)
        ns['_memory'] = object()
        return {'bootstrapped': True}
    ns['_handle_bootstrap_once'] = body
    result = []
    thread = threading.Thread(target=lambda: result.append(ns['handle_bootstrap']({})))
    thread.start()
    assert entered.wait(3)
    try:
        with pytest.raises(RuntimeError, match='bootstrap initializing'):
            ns['handle_bootstrap']({})
        assert len(calls) == 1
    finally:
        release.set(); thread.join(3)
    assert result == [{'bootstrapped': True}]


@pytest.mark.parametrize('published', [False, True])
def test_rpc_constructor_attempt_failure_is_latched(published):
    ns = bootstrap_ns()
    graph = object() if published else None
    def body(params):
        ns['_bootstrap_construction_attempted'] = True
        ns['_memory'] = graph
        raise RuntimeError('partial initialization')
    ns['_handle_bootstrap_once'] = body
    with pytest.raises(RuntimeError): ns['handle_bootstrap']({})
    ns['_handle_bootstrap_once'] = lambda p: pytest.fail('must not retry partial bootstrap')
    with pytest.raises(RuntimeError, match='initialization_failed'):
        ns['handle_bootstrap']({})
    assert ns['_memory'] is graph


def test_rpc_preconstruction_failure_releases_claim():
    ns = bootstrap_ns()
    def body(params): raise OSError('preconstruction')
    ns['_handle_bootstrap_once'] = body
    with pytest.raises(OSError): ns['handle_bootstrap']({})
    ns['_handle_bootstrap_once'] = lambda p: {'bootstrapped': True}
    assert ns['handle_bootstrap']({}) == {'bootstrapped': True}


def test_actual_rpc_constructor_boundary_marks_attempt_before_call():
    tree = ast.parse((ROOT / 'neurograph_rpc.py').read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == '_handle_bootstrap_once')
    assignment_index = next(i for i, n in enumerate(fn.body)
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
        and ast.unparse(n.value.func) == 'NeuroGraphMemory.get_instance')
    marker = fn.body[assignment_index - 1]
    assert ast.unparse(marker) == '_bootstrap_construction_attempted = True'
    ns = {'_bootstrap_construction_attempted': False,
          'NeuroGraphMemory': NS(get_instance=lambda: (_ for _ in ()).throw(RuntimeError()))}
    with pytest.raises(RuntimeError):
        exec(compile(ast.Module(body=[marker, fn.body[assignment_index]], type_ignores=[]),
                     '<actual-constructor-boundary>', 'exec'), ns)
    assert ns['_bootstrap_construction_attempted'] is True


def test_cc_waiting_caller_observes_first_failure_without_reconstructing(cc):
    ns, state, counts, ctl = cc
    ctl.pause, ctl.fail = True, 'constructor'
    results = []
    entered_second = threading.Event()
    def run(second=False):
        if second:
            entered_second.set()
        results.append(ns['init_cc_host']())
    first = threading.Thread(target=run)
    second = threading.Thread(target=run, args=(True,))
    first.start()
    assert ctl.entered.wait(3)
    second.start()
    assert entered_second.wait(3)
    ctl.release.set()
    first.join(3); second.join(3)
    assert not first.is_alive() and not second.is_alive()
    assert results == [False, False]
    assert counts['construct'] == 1 and counts['worker'] == 0


@pytest.mark.parametrize('condition', ['initializing', 'initialization_failed', 'ready'])
def test_actual_rpc_envelope_rejects_incomplete_bootstrap(condition):
    import json
    import traceback
    from typing import Optional
    ns = bootstrap_ns()
    ns.update(json=json, traceback=traceback, Optional=Optional,
              logger=logging.getLogger('test'))
    ns['_handle_bootstrap_once'] = lambda p: {'bootstrapped': True}
    ns['METHODS'] = {'bootstrap': ns['handle_bootstrap']}
    functions('neurograph_rpc.py', {'process_request'}, ns)
    if condition == 'initializing':
        ns['_bootstrap_lock'].acquire()
    elif condition == 'initialization_failed':
        ns['_bootstrap_incomplete'] = True
    try:
        response = json.loads(ns['process_request'](json.dumps(
            {'jsonrpc': '2.0', 'id': 7, 'method': 'bootstrap', 'params': {}})))
    finally:
        if condition == 'initializing':
            ns['_bootstrap_lock'].release()
    assert response['id'] == 7
    if condition == 'ready':
        assert response['result'] == {'bootstrapped': True}
        assert 'error' not in response
    else:
        assert response['error']['code'] == -32000
        assert condition in response['error']['message']
        assert 'result' not in response
