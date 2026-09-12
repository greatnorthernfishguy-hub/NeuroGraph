"""#423: actual host, receiver and save receipt compose without losing input."""
# ---- Changelog ----
# [2026-09-11] Codex — Exercise the complete persistence/acknowledgment boundary.
# What: real production functions, actual BTF/SQLite/guardian files, fake learning.
# Why: individually correct components must agree on refusal and save retry.
# How: AST host extraction joins existing disposable source fixtures; no NG model.
import ast
import logging
import os
from pathlib import Path
import sys
import threading
from types import MethodType, SimpleNamespace

import pytest

from .test_cc_gateway_durable import rig
from .test_save_receipt_423 import FakeSelf, NS, save


@pytest.mark.parametrize('failure', ['vectors', 'quarantine', 'flush'])
def test_host_retains_input_and_retries_save_without_relearning(rig, monkeypatch, failure):
    checkpoint = rig.tmp_path / 'checkpoints'
    checkpoint.mkdir()
    ng = FakeSelf(checkpoint)
    ng.graph._concurrent_lock = threading.RLock()
    ng.save = MethodType(save, ng)
    calls = []
    def learn(graph, vectors, text, embedding, state):
        calls.append(text)
        graph.payload += text.encode()
        return True
    rig.ns['run_conversational_dual_pass'] = learn
    path = rig.add(texts=('new experience',))
    raw = path.read_bytes()
    monkeypatch.setitem(sys.modules, 'cc_ng_organism', SimpleNamespace(
        drain_gateway_conduit=rig.ns['drain_gateway_conduit']))
    source = Path(__file__).parents[1] / 'cc_ng_host.py'
    tree = ast.parse(source.read_text())
    handler = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                   and n.name == '_handle_drain_conduit')
    ns = dict(_STATE=SimpleNamespace(cc_ng=ng, conv_state={}), os=os,
              CC_NG_WORKSPACE=str(rig.tmp_path / 'host'), logger=logging.getLogger('fixture'))
    exec(compile(ast.Module(body=[handler], type_ignores=[]), str(source), 'exec'), ns)
    invoke = lambda: ns['_handle_drain_conduit']({'conduit_dir': str(rig.conduit)})
    if failure == 'vectors':
        ng.vector_db.fail = True
    if failure == 'quarantine':
        ng._save_gate._permit = False
    with monkeypatch.context() as scoped:
        if failure == 'flush':
            def fail_flush(components):
                raise OSError('storage did not confirm checkpoint persistence')
            scoped.setitem(NS, '_sync_receipt_artifacts', fail_flush)
        result = invoke()
    assert not result['accepted'] and result['retained'] == 1
    assert path.read_bytes() == raw
    assert calls == ['new experience']
    ng.vector_db.fail = False
    ng._save_gate._permit = True
    result = invoke()
    assert result['accepted'] and result['all_done']
    assert not path.exists()
    assert b'new experience' in ng._checkpoint_path.read_bytes()
    assert calls == ['new experience']
    assert invoke()['accepted_files'] == result['accepted_files']
    assert calls == ['new experience']
