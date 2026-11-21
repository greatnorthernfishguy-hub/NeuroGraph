# ---- Changelog ----
# [2026-11-21] DudeMan CC (Claude Fable 5.2, laptop) -- step()'s propagation and delivery as native passes
#   (config native_propagation_enabled; ng_tract SynapseStore.propagate_step / deliver_step; SQ10, punch #1221).
# What: twin runs of THIS engine from the same start, same wheel, same seeded streams: side A with the key absent (the
#       existing Python loop and `_delay_buffer`), side B with the key on (the native passes, the queue in the store).
#       Compared after every step: the same nodes fire, the same arrivals are queued for the same ticks, currents and
#       voltages equal to rounding. With every node given a `poincare_dir` (hyperbolic layers 0-2 and spherical ones)
#       the manifold attenuation is compared against the Python's NumPy float32 arithmetic. Also: the key going off
#       moves the queue back; the checkpoint carries the queue; a deleted target is dropped at delivery; the legacy
#       list form of `poincare_dir`; a shape mismatch raises; the receiver reads its pending input from the store.
# Why:  docs superpowers/specs/2026-11-20-recall-without-the-step-lock-design.md §2b / §7. Declared difference (Josh
#       2026-11-21 "still acceptable"): rows ascending instead of Python set order and one f64 dot product instead of
#       three float32 ones, so the bar is "equal to rounding", not bytes. Key absent = the engine before the change is
#       covered by the twin runs against the engine from git (tests/test_native_tick_m1.py, _m2.py, test_stability_s*).
# How:  tests.test_hyperedge_store_h2 Side (identical uuid / random streams) and its built() start graphs.
# -------------------
import os
import random
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import neuro_foundation as NF  # noqa: E402
from tests import test_hyperedge_store_h2 as H2   # noqa: E402
from tests import test_stability_s as S            # noqa: E402
from tests import test_stability_s18 as S18        # noqa: E402

HAVE = getattr(getattr(NF.ng_tract, "SynapseStore", None), "propagate_round", 0) >= 1
pytestmark = pytest.mark.skipif(not HAVE, reason="installed ng_tract has no propagate_step")

KEY = {"native_propagation_enabled": True}
DIM = 24


def _queue(g):
    """{tick: sorted [(target, current)]} from whichever side holds the queue."""
    q = {t: list(v) for t, v in g._delay_buffer.items() if v}
    if g.__dict__.get("_prop_is_native"):
        for t, v in g.synapses.delay_export():
            q.setdefault(t, []).extend(v)
    return {t: sorted(v) for t, v in q.items()}


def _twins(seed, extra=None, stamp=False):
    start = H2.built(seed)
    a, b = S.restore_as(NF, "native", start), S.restore_as(NF, "native", start)
    for g in (a, b):
        g.config.update(dict(S18.ARM, **(extra or {})))
        if stamp:
            rng = np.random.RandomState(seed)
            for i, nid in enumerate(sorted(g.nodes.keys())):
                d = rng.standard_normal(DIM).astype(np.float32)
                d /= np.linalg.norm(d)
                n = g.nodes[nid]
                md = dict(n.metadata or {})
                md["poincare_dir"] = NF.pack_poincare_dir(d)
                n.metadata = md
                n.diffpc_layer = i % 3
                if i % 5 == 0:
                    n.manifold_type = "spherical"
    b.config.update(KEY)
    return H2.Side(a, seed), H2.Side(b, seed)


def _kick(g, rng, k=12, amount=1.5):
    ids = sorted(g.nodes.keys())
    for nid in rng.sample(ids, min(k, len(ids))):
        g.nodes[nid].voltage += amount


def _close(x, y, rel):
    return abs(x - y) <= rel * max(1.0, abs(x), abs(y))


def _compare(a, b, rel, where):
    qa, qb = _queue(a), _queue(b)
    assert sorted(qa) == sorted(qb), (where, sorted(qa), sorted(qb))
    for t in qa:
        assert [x for x, _ in qa[t]] == [x for x, _ in qb[t]], (where, t)
        for (tid, ca), (_, cb) in zip(qa[t], qb[t]):
            assert _close(ca, cb, rel), (where, t, tid, ca, cb)
    for nid in a.nodes.keys():
        assert _close(a.nodes[nid].voltage, b.nodes[nid].voltage, rel), (where, nid)


def _run(seed, steps, extra=None, stamp=False, rel=1e-9):
    sa, sb = _twins(seed, extra, stamp)
    fired_total = queued = 0
    for i in range(steps):
        ra = sa(lambda g: (_kick(g, random.Random(seed * 100 + i)), g.step())[1])
        rb = sb(lambda g: (_kick(g, random.Random(seed * 100 + i)), g.step())[1])
        assert sorted(ra.fired_node_ids) == sorted(rb.fired_node_ids), (seed, i)
        fired_total += len(rb.fired_node_ids)
        _compare(sa.g, sb.g, rel, (seed, i))
        queued += sb.g.synapses.delay_len()[1]
    assert not sb.g._delay_buffer and sb.g.__dict__["_prop_is_native"] and not sa.g.__dict__.get("_prop_is_native")
    assert sa.g.synapses.delay_len() == (0, 0)
    assert fired_total > 0 and queued > 0          # not vacuous
    return sa.g, sb.g


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_same_firing_same_arrivals_without_positions(seed):
    """No node has a poincare_dir here: currents are weight x sign, so the two sides differ only in order."""
    _run(seed, 30)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_same_firing_same_arrivals_with_depression(seed):
    _run(seed, 30, {"syn_std_enabled": True})


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_manifold_attenuation_matches_the_numpy_arithmetic(seed):
    """Every node has a direction; layers 0-2 and one node in five spherical. The Python side computes in float32."""
    a, b = _run(seed, 30, stamp=True, rel=2e-5)
    plain = sorted(c for v in _queue(_run(seed, 30)[1]).values() for _, c in v)
    assert sorted(c for v in _queue(b).values() for _, c in v) != plain      # the attenuation did act


def test_the_key_going_off_moves_the_queue_back():
    _, b = _run(1, 8)
    before = _queue(b)
    assert before and not b._delay_buffer
    b.config.pop("native_propagation_enabled")
    assert b._prop_native() is False
    assert b.synapses.delay_len() == (0, 0) and not b.__dict__["_prop_is_native"]
    assert {t: sorted(v) for t, v in b._delay_buffer.items()} == before
    b.config.update(KEY)
    assert b._prop_native() is True and not b._delay_buffer and _queue(b) == before


def test_the_checkpoint_carries_the_queue(tmp_path):
    a, b = _run(2, 8)
    before = _queue(b)
    path = str(tmp_path / "c.msgpack")
    b.checkpoint(path)
    g = S.graph_of(NF, "native")
    g.restore(path)
    kept = {t: v for t, v in before.items() if t > b.timestep}
    assert kept and {t: sorted(v) for t, v in g._delay_buffer.items()} == kept
    b.restore(path)                                    # in place: the store's queue is emptied, the field reloaded
    assert not b.__dict__.get("_prop_is_native") and b.synapses.delay_len() == (0, 0)
    assert {t: sorted(v) for t, v in b._delay_buffer.items()} == kept


def test_a_target_deleted_in_flight_is_dropped_at_delivery():
    g = S.restore_as(NF, "native", H2.built(0))
    g.config.update(dict(S18.ARM, **KEY))
    ids = sorted(g.nodes.keys())
    live = ids[0]
    v0 = g.nodes[live].voltage
    g.synapses.delay_import([(g.timestep + 1, [("no-such-node", 0.7), (live, 0.25)])])
    assert g.synapses.delay_arrivals(g.timestep + 1) == [("no-such-node", 0.7), (live, 0.25)]
    assert g.synapses.delay_pending(g.timestep, 3) == {"no-such-node": [0.7, 0.0, 0.0], live: [0.25, 0.0, 0.0]}
    assert g.synapses.deliver_step(g.nodes, g.timestep + 1) == (1, 1)
    assert g.nodes[live].voltage == v0 + 0.25 * g.nodes[live].intrinsic_excitability
    assert g.synapses.delay_len() == (0, 0)


def _two_nodes(pd_a, pd_b):
    g = S.graph_of(NF, "native")
    g.config.update(KEY)
    a = g.create_node(node_id="A")
    b = g.create_node(node_id="B")
    g.create_synapse("A", "B", weight=0.5)
    for n, pd in ((a, pd_a), (b, pd_b)):
        if pd is not None:
            md = dict(g.nodes[n.node_id].metadata or {})
            md["poincare_dir"] = pd
            g.nodes[n.node_id].metadata = md
    return g


def _gsg():
    return (NF._GSG_MSG_DECAY, NF._GSG_MSG_DECAY_SPHER, NF._GSG_KAPPA_L2, *NF._GSG_LAYER_NORMS_NF)


def test_the_legacy_list_form_and_a_missing_direction():
    u, v = [0.6, 0.8, 0.0], [0.0, 0.6, 0.8]
    packed = _two_nodes(NF.pack_poincare_dir(u), NF.pack_poincare_dir(v))
    listed = _two_nodes(list(u), list(v))
    none = _two_nodes(NF.pack_poincare_dir(u), None)
    out = []
    for g in (packed, listed, none):
        assert g.synapses.propagate_step(g.nodes, ["A"], 0, None, _gsg()) == 1
        ((_, entries),) = g.synapses.delay_export()
        out.append(entries[0][1])
    assert out[0] == out[1] and 0.0 < out[0] < 0.5 and out[2] == 0.5
    # the Python expression on float32 positions (each node's own layer radius)
    ra, rb = (NF._GSG_LAYER_NORMS_NF[max(0, min(2, packed.nodes[n].diffpc_layer))] for n in ("A", "B"))
    x, y = np.asarray(u, np.float32) * ra, np.asarray(v, np.float32) * rb
    nx2, ny2 = min(float(np.dot(x, x)), 0.9999), min(float(np.dot(y, y)), 0.9999)
    d = x - y
    hd = np.arccosh(max(1.0, 1.0 + 2.0 * float(np.dot(d, d)) / max((1.0 - nx2) * (1.0 - ny2), 1e-9)))
    want = 0.5 * np.exp(-NF._GSG_MSG_DECAY * ((1.0 / max(1.0 - nx2, 1e-6)) / NF._GSG_KAPPA_L2) * hd)
    assert _close(out[0], float(want), 1e-6)


def test_directions_of_different_sizes_raise_as_numpy_does():
    g = _two_nodes(NF.pack_poincare_dir([1.0, 0.0]), NF.pack_poincare_dir([1.0, 0.0, 0.0]))
    with pytest.raises(ValueError):
        g.synapses.propagate_step(g.nodes, ["A"], 0, None, _gsg())
    g = _two_nodes(b"\x00\x00\x00", None)
    with pytest.raises(ValueError):
        g.synapses.propagate_step(g.nodes, ["A"], 0, None, _gsg())


def test_a_wheel_without_the_pass_logs_once_and_stays_on_the_python_loop(monkeypatch, caplog):
    g = S.restore_as(NF, "native", H2.built(0))
    g.config.update(dict(S18.ARM, **KEY))
    monkeypatch.setattr(NF.ng_tract, "NodeStore", (), raising=False)
    with caplog.at_level("ERROR"):
        assert g._prop_native() is False and g._prop_native() is False
    assert sum("native_propagation_enabled" in r.message for r in caplog.records) == 1
    assert not g.__dict__.get("_prop_is_native")


@pytest.mark.parametrize("seed", [0, 1])
def test_the_receiver_reads_its_pending_input_from_the_store(seed):
    extra = dict(S18.RPE, **S18.ON, prediction_receiver_chain_floor=0.0, prediction_receiver_rho0=0.12,
                 prediction_receiver_rho_min=0.1)
    a, b = _run(seed, 40, extra)
    assert b._recv_stats["registered"] > 0
    for k in ("registered", "merged", "dropped", "chains", "read_skipped"):
        assert a._recv_stats[k] == b._recv_stats[k], k
    assert sorted(p.target_node_id for p in a.active_predictions.values()) == \
           sorted(p.target_node_id for p in b.active_predictions.values())
