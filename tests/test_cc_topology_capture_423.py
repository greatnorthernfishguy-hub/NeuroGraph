# ---- Changelog ----
# [2026-09-11] Codex — #423 topology capture interleaving regression tests
# What: execute canonical merge AST with fake graphs and in-memory conduit frames.
# Why: capture must observe completed node/vector/binding batches, with I/O outside.
# How: deterministic thread barriers; no Graph, model or checkpoint constructors.
# -------------------
import ast
import logging
import os
from pathlib import Path
import sys
import threading
from types import SimpleNamespace, ModuleType
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pytest


class FakeGraph:
    def __init__(self):
        self._step_lock = threading.RLock()
        self.nodes = {}
        self.synapses = {}
        self.hyperedges = {}
        self._outgoing = {}
        self._incoming = {}
        self._node_hyperedges = {}
        self.entered = threading.Event()
        self.release = threading.Event()
        self.pause = False

    def create_node(self, node_id, metadata):
        assert self._step_lock._is_owned()
        self.nodes[node_id] = SimpleNamespace(metadata=metadata)
        if self.pause:
            self.pause = False
            self.entered.set()
            assert self.release.wait(3)
        return self.nodes[node_id]

    def create_synapse(self, **kw):
        assert self._step_lock._is_owned()
        sid = str(len(self.synapses))
        self.synapses[sid] = SimpleNamespace(**kw)
        self._outgoing.setdefault(kw['pre_node_id'], set()).add(sid)
        self._incoming.setdefault(kw['post_node_id'], set()).add(sid)

    def create_hyperedge(self, member_node_ids, **kw):
        assert self._step_lock._is_owned()
        edge = SimpleNamespace(member_nodes=member_node_ids, level=0, **kw)
        hid = kw['hyperedge_id'] or str(len(self.hyperedges))
        self.hyperedges[hid] = edge
        for nid in member_node_ids:
            self._node_hyperedges.setdefault(nid, set()).add(hid)
        return edge


def load_merge(monkeypatch, graph, vectors, frames):
    source = Path(__file__).parents[1] / 'cc_topology_merge.py'
    tree = ast.parse(source.read_text())
    names = {'merge_cc_topology', 'cc_current_membership', '_unbound_nodes',
             '_synapse_exists', '_hyperedge_exists', 'TopologyMergeAbort'}
    selected = ast.Module(body=[n for n in tree.body
                              if getattr(n, 'name', None) in names], type_ignores=[])
    calls = {'consolidation': [], 'membership': [], 'decode': 0}

    def deposit(g, v, nid, emb, content, meta):
        g.create_node(nid, dict(meta))
        assert g._step_lock._is_owned()
        v[nid] = emb.copy()

    def consolidate(g, steps):
        assert not g._step_lock._is_owned(), 'step -> concurrent inversion'
        calls['consolidation'].append(steps)
        return True

    def decode(raw):
        assert not graph._step_lock._is_owned()
        calls['decode'] += 1
        return iter(frames)

    def membership(path, ids):
        assert not graph._step_lock._is_owned(), 'disk I/O under step lock'
        calls['membership'].append(ids)

    organism = ModuleType('cc_ng_organism')
    organism._cc_deposit_memory_node = deposit
    organism._cc_callosum_consolidate = consolidate
    monkeypatch.setitem(sys.modules, 'cc_ng_organism', organism)
    ns = dict(Any=Any, Dict=Dict, List=List, Optional=Optional, Set=Set,
              os=os, np=np, logger=logging.getLogger(__name__),
              _DEFAULT_MAX_NODES_PER_CALL=25,
              _load_membership=lambda p: set(), _write_membership=membership,
              read_topology_frames=decode, is_cc_provenance=lambda nid, meta: True,
              _synapse_type=lambda value: value)
    exec(compile(selected, str(source), 'exec'), ns)
    return ns['merge_cc_topology'], calls


def frames(bound=True):
    records = [dict(id=nid, metadata={'cc': True}, embedding_dim=2,
                    embedding=np.array([1, 2], dtype=np.float32).tobytes())
               for nid in ['a', 'b']]
    return [dict(kind='header', machine_id='vps', embedding_model='test'),
            dict(kind='batch', nodes=records,
                 synapses=[dict(pre='a', post='b', delay=7)] if bound else [],
                 hyperedges=[dict(id='he', members=['a', 'b'], level=3)] if bound else [])]


def run_merge(merge, graph, vectors, path, **kw):
    return merge(graph, vectors, str(path), local_machine_id='laptop',
                 expected_embedding_model='test', **kw)


def test_capture_waits_for_complete_topology_and_vector_batch(monkeypatch, tmp_path):
    graph, vectors = FakeGraph(), {}
    merge, calls = load_merge(monkeypatch, graph, vectors, frames())
    path = tmp_path / 'conduit'
    path.write_bytes(b'fake frames')
    graph.pause = True
    errors, result, snapshot = [], [], []
    capture_started, capture_done = threading.Event(), threading.Event()

    def receiver():
        try:
            result.append(run_merge(merge, graph, vectors, path))
        except BaseException as exc:
            errors.append(exc)

    def capture():
        capture_started.set()
        with graph._step_lock:
            snapshot.append((set(graph.nodes), set(vectors),
                             len(graph.synapses), len(graph.hyperedges)))
        capture_done.set()

    worker = threading.Thread(target=receiver)
    reader = threading.Thread(target=capture)
    worker.start()
    try:
        assert graph.entered.wait(3)
        reader.start()
        assert capture_started.wait(3)
        assert not capture_done.wait(.05)
    finally:
        graph.release.set()
        worker.join(3)
        if reader.ident is not None:
            reader.join(3)
    assert not worker.is_alive() and not reader.is_alive()
    assert errors == []
    assert snapshot == [({'a', 'b'}, {'a', 'b'}, 1, 1)]
    assert result[0]['consolidation_steps'] == 250
    assert calls['consolidation'] == [250]
    assert calls['membership'] == [{'a', 'b'}]
    assert next(iter(graph.synapses.values())).delay == 7
    assert graph.hyperedges['he'].level == 3


@pytest.mark.parametrize('budget, bound, expected_nodes, expected_steps', [
    (1, True, 1, 0), (25, False, 2, 0), (25, True, 2, 250),
])
def test_budget_and_unbound_arrival_guard_unchanged(
        monkeypatch, tmp_path, budget, bound, expected_nodes, expected_steps):
    graph, vectors = FakeGraph(), {}
    merge, calls = load_merge(monkeypatch, graph, vectors, frames(bound))
    path = tmp_path / 'conduit'
    path.write_bytes(b'fake frames')
    result = run_merge(merge, graph, vectors, path, max_nodes_per_call=budget)
    assert result['absorbed_nodes'] == expected_nodes
    assert result['consolidation_steps'] == expected_steps
    assert calls['decode'] == 1
    assert calls['membership'] == [set(graph.nodes)]
    assert result['deferred_by_budget'] == 2 - expected_nodes
