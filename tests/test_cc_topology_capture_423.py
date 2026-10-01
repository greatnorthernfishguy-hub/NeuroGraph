# ---- Changelog ----
# [2026-11-16] DudeMan CC (Claude Fable 5.2, laptop; NG suite repair #761) -- harness follows the merge, NO assertion changed:
#   the AST-extracted names now include `_drop_banned_meta` and the namespace carries the real `_BANNED_META` (imported from
#   cc_topology_export, as the merge does). Commit 5637eb9 (2026-10-02, "Drop banned metadata on topology merge") made
#   merge_cc_topology call `_drop_banned_meta(meta)` per inbound node; without it the extracted merge raised NameError
#   (4 tests). The test frames carry only {'cc': True}, so nothing is dropped and every pinned value is as before.
#   Likewise `find_synapse_id` / `find_hyperedge_id` join the extracted names (and `Callable` the namespace): commit 4aa8c82
#   (2026-10-23, "#1173 fix set (NG): reinforce on recurrence ...") split the two idempotency guards into lookups that
#   return the id; `_synapse_exists` / `_hyperedge_exists` are now `find_*_id(...) is not None` ("this guard's answer is unchanged").
# [2026-10-01] Claude Sonnet 5.5 (Z12 builder, lane ack-bound-918, dispatch #13930) — #918 fold-up: harness follows the merge
# What: the AST-extracted names now also include `cc_ack_membership` (the ack writer's new function; it calls the extracted
#   `cc_current_membership`) and `_reset_reoffer_streaks` (the one streak reset, N4). No assertion changed.
# [2026-10-01] Claude Sonnet 5.5 (Z12 builder, lane ack-bound-918, dispatch #13352) — #918: harness follows the merge; ack = BOUND
# What: (1) the AST-extracted merge now also needs `_track_reoffers` (extracted) and the re-offer constants/state (namespace).
#   (2) test_budget_and_unbound_arrival_guard_unchanged: its membership expectation was `[set(graph.nodes)]` -- it encoded "the
#   ack includes unbound nodes". Under #918 the ack is the nodes held BOUND, so the expectation is now a parameter:
#   budget 1 (a landed, its edges skipped because b was deferred) -> set(); unbound arrivals -> set(); bound -> {'a','b'}.
# Why: #918 (Exec Packet 496): the ack means "I hold it BOUND". Nothing else asserted here changed; the first test's
#   membership assertion ([{'a','b'}], both bound) is untouched and still passes.
# [2026-10-02] Claude Sonnet 5.5 (Z12 builder, lane in-transit-hold-905d, dispatch #14474) — #905-DELTA: harness follows the merge, NO assertion changed
# What: the AST-extracted merge now calls cc_topology_merge.held_unbound_nodes (the batch-end check and whole_graph_guard route through it),
#   so it joins the extracted names, and the namespace gets `_in_transit_cohort` / `_IN_TRANSIT_VALID`: the harness runs the merge with the
#   in-transit variable UNSET (the real function's UNSET state: held == _unbound_nodes), which is exactly the behaviour these tests pin.
# Why: Exec P547/P548. Without it the extracted merge raises NameError('held_unbound_nodes'): a harness gap, not a behaviour change.
# How: names += held_unbound_nodes; ns += the UNSET answer. `_unbound_nodes` keeps its (graph, node_ids) arity: still the pin this file is for.
# [2026-10-01] Claude Sonnet 5.5 (Z12 builder, lane emergent-want-bound-905, dispatch #13138) — #905: harness follows the merge
# What: the AST-extracted merge now references cc_topology_merge.whole_graph_guard and redact_node_id (so they join the
#   extracted names, and hashlib/re join the namespace) and passes guard=/progress= to the consolidation, so the stub accepts them.
# Why: Part D (the merge passes the per-slice guard) and Part C (the ERROR sample is redacted). Nothing the tests assert changed;
#   `_unbound_nodes` keeps its (graph, node_ids) arity, which is the pin this file exists for.
# [2026-10-01] Claude Sonnet 5.5 (Z12 builder, lane model-guard-910, dispatch #13145) — #910: harness namespace
# What: load_merge() injects the two new names merge_cc_topology references (is_resolved_model_id -- the REAL predicate --
#   and a resolve_embedding_model stub that fails if reached). No assertion changed.
# Why: the harness exec()s only AST-selected names, so a new module-level dependency of merge_cc_topology is a NameError
#   here. This test did NOT encode the dead guard: it always passes expected_embedding_model='test' and a 'test' header.
# [2026-09-11] Codex — #423 topology capture interleaving regression tests
# What: execute canonical merge AST with fake graphs and in-memory conduit frames.
# Why: capture must observe completed node/vector/binding batches, with I/O outside.
# How: deterministic thread barriers; no Graph, model or checkpoint constructors.
# -------------------
import ast
import hashlib
import logging
import os
import re
from pathlib import Path
import sys
import threading
from types import SimpleNamespace, ModuleType
from typing import Any, Callable, Dict, List, Optional, Set

import numpy as np
import pytest

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from cc_topology_export import is_resolved_model_id  # noqa: E402


def _resolver_must_not_run():
    raise AssertionError('resolve_embedding_model reached: expected_embedding_model was not honoured')


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
    names = {'merge_cc_topology', 'cc_current_membership', 'cc_ack_membership', '_reset_reoffer_streaks', '_unbound_nodes',
             'whole_graph_guard', 'redact_node_id', '_track_reoffers', 'held_unbound_nodes',
             '_synapse_exists', '_hyperedge_exists', 'TopologyMergeAbort', '_drop_banned_meta',
             'find_synapse_id', 'find_hyperedge_id'}
    selected = ast.Module(body=[n for n in tree.body
                              if getattr(n, 'name', None) in names], type_ignores=[])
    calls = {'consolidation': [], 'membership': [], 'decode': 0}

    def deposit(g, v, nid, emb, content, meta):
        g.create_node(nid, dict(meta))
        assert g._step_lock._is_owned()
        v[nid] = emb.copy()

    def consolidate(g, steps, *, guard=None, progress=None):
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
    from cc_topology_export import _BANNED_META   # the real constant _drop_banned_meta reads (5637eb9)
    ns = dict(Any=Any, Callable=Callable, Dict=Dict, List=List, Optional=Optional, Set=Set, _BANNED_META=_BANNED_META,
              os=os, re=re, hashlib=hashlib, np=np, logger=logging.getLogger(__name__),
              _DEFAULT_MAX_NODES_PER_CALL=25,
              _REOFFER_WARN_STREAK_DEFAULT=5, _REOFFER_WARN_STREAK_ENV='CC_TOPOLOGY_REOFFER_WARN_STREAK',
              _REOFFER_TABLE_CAP=1024, _reoffer_streaks={},
              _load_membership=lambda p: set(), _write_membership=membership,
              read_topology_frames=decode, is_cc_provenance=lambda nid, meta: True,
              # #910: merge_cc_topology now calls the shared model-id predicate/resolver.
              # The REAL predicate; the resolver is never reached (every caller here passes
              # expected_embedding_model) so a stub that fails loudly if it is.
              is_resolved_model_id=is_resolved_model_id,
              resolve_embedding_model=_resolver_must_not_run,
              _synapse_type=lambda value: value,
              _in_transit_cohort=lambda: ('unset', frozenset()), _IN_TRANSIT_VALID='valid')   # #905-DELTA: variable UNSET
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


# #918: the ack is the set the receiver holds BOUND, so an unbound arrival (budget 1 leaves `a` without its edges;
# bound=False carries none) is NOT in it. This column used to be `set(graph.nodes)`.
@pytest.mark.parametrize('budget, bound, expected_nodes, expected_steps, expected_ack', [
    (1, True, 1, 0, set()), (25, False, 2, 0, set()), (25, True, 2, 250, {'a', 'b'}),
])
def test_budget_and_unbound_arrival_guard_unchanged(
        monkeypatch, tmp_path, budget, bound, expected_nodes, expected_steps, expected_ack):
    graph, vectors = FakeGraph(), {}
    merge, calls = load_merge(monkeypatch, graph, vectors, frames(bound))
    path = tmp_path / 'conduit'
    path.write_bytes(b'fake frames')
    result = run_merge(merge, graph, vectors, path, max_nodes_per_call=budget)
    assert result['absorbed_nodes'] == expected_nodes
    assert result['consolidation_steps'] == expected_steps
    assert calls['decode'] == 1
    assert calls['membership'] == [expected_ack]
    assert result['deferred_by_budget'] == 2 - expected_nodes
