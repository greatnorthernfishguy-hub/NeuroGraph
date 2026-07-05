# tests/test_graph_substrate_race.py
# [2026-06-18] CC — #88 regression: "dictionary changed size during iteration" in the
# bootstrap path. The racy iteration is _build_adjacency's PYTHON-level loop over the LIVE
# graph.synapses dict (GIL releases between iterations); a concurrent graph.step()
# (prune/sprout mutates synapses UNDER _step_lock) changes the dict mid-iteration.
# (_rebuild_index's sorted(nodes.keys()) is a single C call = GIL-atomic, but we snapshot
# it under the lock too as cheap insurance.) Fix: snapshot under the Graph's _step_lock.
import threading
from neuro_foundation import Graph
from lenia.graph_substrate import NeuroGraphSubstrate


def test_build_adjacency_no_race_under_concurrent_step_mutation():
    g = Graph()
    for i in range(200):
        g.create_node(node_id=f"n{i}")
    syn = None
    for i in range(0, 198):
        syn = g.create_synapse(f"n{i}", f"n{i + 1}")  # real synapses -> real Python loop
    sub = NeuroGraphSubstrate(g, None)

    stop = threading.Event()
    errors = []

    def writer():
        # Mimic graph.step() churning graph.synapses UNDER the step lock, continuously.
        i = 100000
        while not stop.is_set():
            with g._step_lock:
                g.synapses[f"s{i}"] = syn
                old = f"s{i - 60}"
                if old in g.synapses:
                    del g.synapses[old]
            i += 1

    def reader():
        try:
            for _ in range(5000):
                sub._build_adjacency()
                sub._rebuild_index()
        except Exception as e:  # noqa: BLE001 — catching the race is the point
            errors.append(repr(e))

    wt = threading.Thread(target=writer, daemon=True)
    rt = threading.Thread(target=reader, daemon=True)
    wt.start()
    rt.start()
    rt.join(timeout=30)
    stop.set()
    wt.join(timeout=5)

    assert not rt.is_alive(), "reader did not finish (possible deadlock)"
    assert errors == [], f"concurrent mutation raced: {errors[:3]}"


# [2026-07-05] CC (laptop) — regression test for the Jul 3 11:36:42 live crash:
# RuntimeError: dictionary changed size during iteration, graph_substrate.py:285,
# in _hyperedge_similarity. That method iterated graph.hyperedges.values() directly
# on every call — the one distance-vector helper that never got the snapshot fix
# _build_adjacency (#88, above) and _synaptic_distance (#341) already have. Fixed
# by precomputing entity->hyperedge-id-set membership once (_build_hyperedge_membership),
# snapshotting under the same _step_lock pattern.
def test_hyperedge_membership_no_race_under_concurrent_he_mutation():
    g = Graph()
    for i in range(50):
        g.create_node(node_id=f"n{i}")
    he = g.create_hyperedge(member_node_ids={f"n{i}" for i in range(10)})
    sub = NeuroGraphSubstrate(g, None)

    stop = threading.Event()
    errors = []

    def writer():
        # Mimic graph.step() churning graph.hyperedges UNDER the step lock,
        # continuously — same shape as the existing synapse writer above.
        i = 100000
        while not stop.is_set():
            with g._step_lock:
                g.hyperedges[f"h{i}"] = he
                old = f"h{i - 60}"
                if old in g.hyperedges:
                    del g.hyperedges[old]
            i += 1

    def reader():
        try:
            for _ in range(5000):
                sub._build_hyperedge_membership()
                sub._rebuild_index()
        except Exception as e:  # noqa: BLE001 — catching the race is the point
            errors.append(repr(e))

    wt = threading.Thread(target=writer, daemon=True)
    rt = threading.Thread(target=reader, daemon=True)
    wt.start()
    rt.start()
    rt.join(timeout=30)
    stop.set()
    wt.join(timeout=5)

    assert not rt.is_alive(), "reader did not finish (possible deadlock)"
    assert errors == [], f"concurrent mutation raced: {errors[:3]}"
