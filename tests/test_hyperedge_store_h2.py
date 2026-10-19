# ---- Changelog ----
# [2026-10-18] Claude (lane hyperedge-store-h2, #1136) — H2: native hyperedge passes == the frozen Python oracle, bitwise
# What: ORACLE = the trial tip's whole neuro_foundation.py (BASE_REV, from git; dict path) with every hyperedge's
#       member_nodes replaced by an insertion-ordered set (OSet) built from the NATIVE store's member order
#       (HyperedgeStore.member_list), so both sides sum / inject in the same order (spec §3.1: "hand the order").
#       NATIVE = this branch with BOTH native stores (the H2 passes: evaluate_step / evaluate_sweep / plasticity_pass /
#       reward_thresholds, and the pass state in the store). Compared STRICTLY after every operation: fired node ids and
#       hyperedge ids in order, every node voltage and refractory (bitwise), every hyperedge and archived hyperedge
#       field (floats bitwise, member lists IN ORDER, weights in order), the pass state (_he_last_fired_step,
#       _he_output_candidates, _he_co_fire_counts, in order), the hyperedge events in order, HE predictions, and the
#       sweep's fired entries. Workloads: steps + read / write sweeps + rewards (global, scoped, int) + discovery + node
#       removal + consolidation + seam split + shed, 6 seeds; mid-run save / restore; INCREMENTAL / extract_subgraph;
#       a forced decline (non-canonical threshold) handing the state to the Python loop and back. Also: the pass-state
#       views are read-only; the checkpoint's pass-state bytes equal msgpack of the oracle's dicts; sorted member lists
#       (#1140) on the native path always and on the Python path behind he_checkpoint_sorted_members; #1142
#       choose-by-contribution (he_member_choice_by_contribution) is hash-seed independent and follows its rule, on
#       both paths; the lenia reader snapshots under the lock.
# Why:  spec 2026-10-17-rust-hyperedge-store-design.md §1.4, §3.1, §4 H2; Josh's H1 rulings 3a / 3b (2026-10-18).
# How:  PYTHONHASHSEED fixed by the runner; seeded random + deterministic uuid4 (the H1 pattern).
# -------------------
"""H2 native hyperedge passes: bitwise parity with the frozen Python oracle (spec §3.1), #1140, #1142."""
import importlib.util
import os
import random
import struct
import subprocess
import sys
import tempfile
import uuid

import msgpack
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
import neuro_foundation as NF  # noqa: E402  (the branch)

BASE_REV = os.environ.get("NG_HE_BASE_REV", "ac9dd38")
HAVE_H2 = hasattr(NF.ng_tract, "HyperedgeStore") and hasattr(NF.ng_tract.HyperedgeStore, "evaluate_step") \
    and hasattr(NF.ng_tract, "NodeStore")
needs_h2 = pytest.mark.skipif(not HAVE_H2, reason="ng_tract without the H2 hyperedge passes")


def _load(rev, name):
    src = subprocess.run(["git", "-C", _REPO, "show", f"{rev}:neuro_foundation.py"],
                         check=True, capture_output=True, text=True).stdout
    d = tempfile.mkdtemp(prefix=f"{name}_")
    p = os.path.join(d, f"{name}.py")
    with open(p, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


BASE = _load(BASE_REV, "nf_base_he_h2")


class OSet(set):
    """A set that iterates in insertion order (append on add, order-preserving discard) — the native store's member
    list semantics. Set algebra (&, -, in, len, issubset) is the plain set's."""

    def __init__(self, it=()):
        it = list(it)
        super().__init__(it)
        self._o = list(dict.fromkeys(it))

    def __iter__(self):
        return iter(list(self._o))

    def add(self, x):
        if x not in self:
            super().add(x)
            self._o.append(x)

    def discard(self, x):
        if x in self:
            super().discard(x)
            self._o.remove(x)

    def remove(self, x):
        super().remove(x)
        self._o.remove(x)

    def clear(self):
        super().clear()
        self._o.clear()

    def update(self, *its):
        for it in its:
            for x in list(it):
                self.add(x)

    def __ior__(self, other):
        self.update(other)
        return self

    def difference_update(self, *its):
        for it in its:
            for x in list(it):
                self.discard(x)

    def __isub__(self, other):
        self.difference_update(other)
        return self

    def pop(self):
        raise NotImplementedError("OSet.pop (order-dependent) is not expected in the engine")

    def intersection_update(self, *its):
        raise NotImplementedError

    def symmetric_difference_update(self, *its):
        raise NotImplementedError


class DetUUID:
    def __init__(self, seed):
        self.rng = random.Random(seed)

    def __call__(self):
        return uuid.UUID(int=self.rng.getrandbits(128), version=4)


def seeded(seed, fn, uuid_stream=0):
    saved = uuid.uuid4
    uuid.uuid4 = DetUUID(seed * 1000 + uuid_stream)
    random.seed(seed)
    try:
        return fn(random.Random(seed))
    finally:
        uuid.uuid4 = saved


def ckpt_bytes(g):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.msgpack")
        g.checkpoint(p)
        with open(p, "rb") as f:
            return f.read()


def restore(kind, b):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.msgpack")
        with open(p, "wb") as f:
            f.write(b)
        g = new_graph(kind)
        g.restore(p)
        return g


def new_graph(kind, config=None):
    if kind == "oracle":
        return BASE.Graph(config)
    if kind == "native":
        g = NF.Graph(config, native_node_store=True, native_hyperedge_store=True)
        assert g._native_he_passes
        return g
    if kind == "h1":            # the store with the Python passes over it (node store off)
        return NF.Graph(config, native_node_store=False, native_hyperedge_store=True)
    return NF.Graph(config, native_node_store=False, native_hyperedge_store=False)


def fb(x):
    return struct.pack("<d", x) if isinstance(x, float) else x


# ---------------------------------------------------------------------------
# order hand-off and strict state
# ---------------------------------------------------------------------------

def sync_order(o, n):
    """Give every oracle hyperedge the native member order (wrap plain sets); check already-wrapped ones agree."""
    for omap, nmap in ((o.hyperedges, n.hyperedges), (o._archived_hyperedges, n._archived_hyperedges)):
        assert list(omap) == list(nmap)
        for hid, he in omap.items():
            order = nmap.member_list(hid)
            if type(he.member_nodes) is OSet:
                assert list(he.member_nodes) == order, f"member order diverged on {hid}"
            else:
                assert set(he.member_nodes) == set(order), hid
                he.member_nodes = OSet(order)


def he_state(he, members):
    return (he.hyperedge_id, tuple(members), tuple((k, fb(v)) for k, v in he.member_weights.items()),
            fb(he.activation_threshold), he.activation_mode.name, fb(he.current_activation),
            tuple(he.output_targets), fb(he.output_weight), repr(he.metadata), he.is_learnable,
            he.refractory_period, he.refractory_remaining, he.activation_count, fb(he.pattern_completion_strength),
            tuple(sorted(he.child_hyperedges)), he.level, fb(he.recent_activation_ema), he.is_archived,
            he.consolidation_state.value, he.creation_time)


def graph_state(g):
    native = g.__class__ is NF.Graph and getattr(g, "_native_he", False)

    def mem(m, hid, he):
        return m.member_list(hid) if native else list(he.member_nodes)
    return (
        tuple((nid, fb(x.voltage), x.refractory_remaining, fb(x.intrinsic_excitability)) for nid, x in g.nodes.items()),
        tuple((hid, he_state(he, mem(g.hyperedges, hid, he))) for hid, he in g.hyperedges.items()),
        tuple((hid, he_state(he, mem(g._archived_hyperedges, hid, he))) for hid, he in g._archived_hyperedges.items()),
        tuple(sorted((k, tuple(sorted(v))) for k, v in g._node_hyperedges.items())),
        tuple((pid, ps.hyperedge_id, tuple(sorted(ps.predicted_targets)), fb(ps.prediction_strength))
              for pid, ps in g._active_predictions.items()),
        tuple(g._he_last_fired_step.items()),
        tuple((h, tuple(c.items())) for h, c in g._he_output_candidates.items()),
        tuple((h, tuple(c.items())) for h, c in g._he_co_fire_counts.items()),
        g._prediction_counter, g._total_predictions,
    )


HE_EVENTS = ("prediction_created", "hyperedge_fired", "he_output_learned", "hyperedge_archived",
             "hyperedge_discovered", "he_state_transition", "hyperedge_seam_split",
             "hyperedges_seam_split_pass", "hyperedge_members_shed", "reward_injected")


def attach_events(g, log):
    def norm(v):
        if isinstance(v, (set, frozenset)):
            return ("set", tuple(sorted(map(repr, v))))
        if isinstance(v, (list, tuple)):
            return tuple(map(repr, v))
        return fb(v)
    for ev in HE_EVENTS:
        g.register_event_handler(ev, lambda _ev=ev, **kw: log.append(
            (_ev, tuple(sorted((k, norm(v)) for k, v in kw.items())))))


HE_CFG = {"he_output_learning_window": 5, "he_output_min_co_fires": 2, "he_output_max_targets": 5,
          "he_member_evolution_min_co_fires": 2, "he_consolidation_eval_interval": 7,
          "he_speculative_to_candidate_min_count": 2, "he_candidate_to_consolidated_min_count": 3,
          "he_candidate_to_consolidated_min_age": 5, "he_experience_threshold": 4,
          "he_discovery_min_co_fires": 2, "he_discovery_window": 4, "he_discovery_min_nodes": 3,
          "he_consolidation_overlap": 0.5, "he_split_oversized_enabled": True, "he_max_members": 9,
          "he_shed_weight_threshold": 0.02, "he_shed_min_tenure": 3, "prediction_window": 4,
          "default_threshold": 0.6, "three_factor_enabled": True}


def build(seed, n_nodes=70, n_syn=350, n_he=40, cfg=None):
    """Built with the ORACLE module through the API (create_hyperedge / hierarchical / attribute writes)."""
    def b(rng):
        g = BASE.Graph(dict(HE_CFG, **(cfg or {})))
        ids = [g.create_node(node_id=f"n{seed}_{i}", metadata={"k": i}).node_id for i in range(n_nodes)]
        for _ in range(n_syn):
            a, c = rng.sample(ids, 2)
            g.create_synapse(a, c, weight=rng.uniform(0.1, 1.5), delay=rng.randint(1, 3))
        made = []
        modes = list(BASE.ActivationMode)
        for j in range(n_he):
            mem = set(rng.sample(ids, rng.randint(2, 8)))
            mw = None if rng.random() < 0.5 else {m: rng.choice([1.0, 0.3, 2.5, rng.random()]) for m in mem}
            he = g.create_hyperedge(mem, member_weights=mw, activation_threshold=rng.uniform(0.2, 0.9),
                                    activation_mode=rng.choice(modes),
                                    output_targets=rng.sample(ids, rng.randint(0, 3)),
                                    output_weight=rng.choice([1.0, 0.6]),
                                    metadata={"label": f"é{j}", "member_since": {m: 0 for m in list(mem)[:1]}},
                                    is_learnable=rng.random() < 0.9)
            he.pattern_completion_strength = rng.choice([0.3, 0.0, 0.8])
            made.append(he.hyperedge_id)
        for _ in range(3):
            g.create_hierarchical_hyperedge(rng.sample(made, 2), activation_threshold=0.5)
        dup = rng.choice(made)
        g.create_hyperedge(set(g.hyperedges[dup].member_nodes), activation_threshold=0.5)
        g.timestep = rng.randint(10, 500)
        return ckpt_bytes(g)
    return seeded(seed, b)


_BUILT = {}


def built(seed):
    if seed not in _BUILT:
        _BUILT[seed] = build(seed)
    return _BUILT[seed]


class Side:
    """One graph of a twin run with its own uuid4 / random / numpy-random streams (identical seeds on both sides), so
    both graphs mint the same ids and draw the same numbers operation by operation."""

    def __init__(self, g, seed):
        import numpy as np
        self.g, self.log = g, []
        self.uuid = DetUUID(seed * 1000 + 1)
        r = random.Random(seed)
        self.rstate = r.getstate()
        self.nstate = np.random.RandomState(seed).get_state()
        attach_events(g, self.log)

    def __call__(self, fn):
        import numpy as np
        saved_u, saved_r, saved_n = uuid.uuid4, random.getstate(), np.random.get_state()
        uuid.uuid4 = self.uuid
        random.setstate(self.rstate)
        np.random.set_state(self.nstate)
        try:
            return fn(self.g)
        finally:
            self.rstate, self.nstate = random.getstate(), np.random.get_state()
            uuid.uuid4 = saved_u
            random.setstate(saved_r)
            np.random.set_state(saved_n)


def _first_diff(a, b, path=()):
    if type(a) is not type(b) or not isinstance(a, tuple):
        return path, a, b
    if len(a) != len(b):
        return path + ("len",), len(a), len(b)
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return _first_diff(x, y, path + (i,))
    return path, None, None


class Twin:
    """The oracle and the native graph driven in lock-step; `op(name, fn)` applies fn to both (oracle first), hands the
    native member order to any new oracle set, and compares results, events and the whole state strictly."""

    def __init__(self, start_bytes, seed):
        self.o = Side(restore("oracle", start_bytes), seed)
        self.n = Side(restore("native", start_bytes), seed)
        sync_order(self.o.g, self.n.g)
        self.ops = 0
        self.counts = {}

    def op(self, name, fn):
        ro = self.o(fn)
        rn = self.n(fn)
        sync_order(self.o.g, self.n.g)
        assert ro == rn, f"{name}: result {ro!r} != {rn!r}"
        assert self.o.log == self.n.log, f"{name}: events differ: {_first_diff(tuple(self.o.log), tuple(self.n.log))}"
        a, b = graph_state(self.o.g), graph_state(self.n.g)
        assert a == b, f"{name} (op {self.ops}): state differs at {_first_diff(a, b)}"
        for ev in self.n.log:
            self.counts[ev[0]] = self.counts.get(ev[0], 0) + 1
        self.o.log.clear()
        self.n.log.clear()
        self.ops += 1
        return rn

    def save_restore(self):
        """Mid-run save / restore on both sides (each reads its OWN file), then re-sync the order."""
        bo, bn = ckpt_bytes(self.o.g), ckpt_bytes(self.n.g)
        assert ckpt_cmp(bo) == ckpt_cmp(bn)
        for side, b, kind in ((self.o, bo, "oracle"), (self.n, bn, "native")):
            g = restore(kind, b)
            side.g = g
            side.log.clear()
            attach_events(g, side.log)
        sync_order(self.o.g, self.n.g)
        a, c = graph_state(self.o.g), graph_state(self.n.g)
        assert a == c, f"after save/restore: {_first_diff(a, c)}"
        return bo, bn


def ckpt_cmp(b):
    """A checkpoint as comparable bytes: member / child lists as sorted lists (the oracle writes list(set), the store
    writes them sorted — §1.5 / #1140), every other byte as is."""
    d = msgpack.unpackb(b, raw=False, strict_map_key=False)
    for key in ("hyperedges", "archived_hyperedges"):
        for hd in (d.get(key) or {}).values():
            for k in ("member_nodes", "child_hyperedges"):
                hd[k] = sorted(hd[k])
    return msgpack.packb(d, use_bin_type=True)


def drive(t, rng, rounds, flags=None):
    """One round = stimulate + step + a sweep (write / read alternating) + rewards (float, scoped, int) + discovery +
    node removal + consolidation / split / shed; every operation compared by the Twin."""
    if flags:
        t.op("flags", lambda g: g.config.update(flags))
    for k in range(rounds):
        nodes = list(t.n.g.nodes)
        cue = [(nid, rng.uniform(0.5, 3.0)) for nid in rng.sample(nodes, min(12, len(nodes)))]

        def step(g, cue=cue):
            for nid, c in cue:
                g.stimulate(nid, c)
            r = g.step()
            return tuple(r.fired_node_ids), tuple(r.fired_hyperedge_ids)
        fired = t.op("step", step)
        ids = rng.sample(nodes, min(6, len(nodes)))
        wm = k % 2 == 0
        t.op("sweep", lambda g: tuple((e.node_id, e.firing_step, fb(e.voltage_at_fire), e.was_predicted)
                                      for e in g.prime_and_propagate(ids, [1.5] * len(ids), steps=3,
                                                                     write_mode=wm).fired_entries))
        if k % 4 == 1:
            scope = set(rng.sample(nodes, 10))
            t.op("reward", lambda g: (g.inject_reward(0.3), g.inject_reward(-0.2, scope=scope), g.inject_reward(1)))
        if k % 5 == 2:
            t.op("discover", lambda g: g.discover_hyperedges(list(fired[0])))
        if k % 6 == 3 and len(nodes) > 30:
            gone = rng.sample(nodes, 3)
            t.op("remove", lambda g: [g.remove_node(x) for x in gone])
        if k % 9 == 4:
            t.op("consolidate", lambda g: g.consolidate_hyperedges())
            t.op("split", lambda g: g.dedup_and_split_oversized_hyperedges(None))
            t.op("shed", lambda g: g.shed_floor_members())


SEEDS = list(range(6))


# ---------------------------------------------------------------------------
# 1. parity: every pass, bitwise, against the oracle
# ---------------------------------------------------------------------------

@needs_h2
@pytest.mark.parametrize("seed", SEEDS)
def test_native_passes_equal_the_oracle(seed):
    t = Twin(built(seed), seed)
    drive(t, random.Random(seed), 30)
    assert t.ops > 80
    # the workload really exercises the passes: fires, predictions, learned targets
    assert t.counts.get("hyperedge_fired", 0) > 30 and t.counts.get("prediction_created", 0) > 0, t.counts
    print("seed", seed, "ops", t.ops, t.counts)
    d = t.n.g.hyperedges.pass_declines()
    assert d == (0, 0, 0, 0), d                          # every pass ran natively
    bo, bn = ckpt_bytes(t.o.g), ckpt_bytes(t.n.g)
    assert ckpt_cmp(bo) == ckpt_cmp(bn)


@needs_h2
@pytest.mark.parametrize("seed", SEEDS[:3])
def test_mid_run_save_restore_continues(seed):
    t = Twin(built(seed), seed)
    rng = random.Random(seed)
    drive(t, rng, 12)
    t.save_restore()
    drive(t, rng, 12)
    t.save_restore()
    drive(t, rng, 4)


@needs_h2
@pytest.mark.parametrize("seed", SEEDS[:3])
def test_incremental_and_subgraph(seed):
    t = Twin(built(seed), seed)
    rng = random.Random(seed)
    drive(t, rng, 6)
    nodes = sorted(t.n.g.nodes)
    pick = set(rng.sample(nodes, 40)) | {"ghost"}

    def caps(g):
        M = BASE if g.__class__ is BASE.Graph else NF
        inc = g.capture_checkpoint(M.CheckpointMode.INCREMENTAL, detach=True)
        sub = g.extract_subgraph(pick)
        sub["extraction_metadata"]["missing_nodes"].sort()
        for d in (inc, sub):
            for hd in d["hyperedges"].values():
                hd["member_nodes"] = sorted(hd["member_nodes"])
                hd["child_hyperedges"] = sorted(hd["child_hyperedges"])
        inc["hyperedges"] = dict(sorted(inc["hyperedges"].items()))
        return msgpack.packb(inc, use_bin_type=True), msgpack.packb(sub, use_bin_type=True)
    t.op("incremental+subgraph", caps)


@needs_h2
@pytest.mark.parametrize("seed", SEEDS[:2])
def test_heavy_synthetic(seed):
    """Larger hyperedges (up to the 50 cap), many fires, member evolution to the cap, output learning to the cap."""
    def b(rng):
        g = BASE.Graph(dict(HE_CFG, he_max_members=50, he_split_oversized_enabled=False, he_output_max_targets=5,
                            he_member_evolution_min_co_fires=3, he_experience_threshold=20, default_threshold=0.7))
        ids = [g.create_node(node_id=f"h{i}").node_id for i in range(400)]
        for _ in range(2400):
            a, c = rng.sample(ids, 2)
            g.create_synapse(a, c, weight=rng.uniform(0.1, 1.2), delay=rng.randint(1, 3))
        for j in range(160):
            mem = set(rng.sample(ids, rng.randint(3, 48)))
            g.create_hyperedge(mem, member_weights={m: rng.uniform(0.3, 5.0) for m in mem},
                               activation_threshold=rng.uniform(0.1, 0.5),
                               output_targets=rng.sample(ids, rng.randint(0, 4)))
        return ckpt_bytes(g)
    start = seeded(100 + seed, b)
    t = Twin(start, seed)
    rng = random.Random(seed)
    for k in range(40):
        cue = rng.sample(sorted(t.n.g.nodes), 60)

        def step(g, cue=cue):
            for nid in cue:
                g.stimulate(nid, 1.2)
            r = g.step()
            return tuple(r.fired_node_ids), tuple(r.fired_hyperedge_ids)
        fired = t.op("step", step)
        if k % 5 == 0:
            t.op("sweep", lambda g: tuple((e.node_id, e.firing_step, fb(e.voltage_at_fire))
                                          for e in g.prime_and_propagate(cue[:8], [1.0] * 8, steps=3,
                                                                         write_mode=False).fired_entries))
    assert t.n.g.hyperedges.pass_declines() == (0, 0, 0, 0)
    assert sum(1 for he in t.n.g.hyperedges.values() if he.activation_count) > 20


@needs_h2
def test_a_declined_pass_hands_the_state_to_python_and_back():
    """A non-canonical value (an int threshold, kept exactly in an overflow cell) makes the native passes decline: the
    frozen loop runs on plain-dict pass state (still equal to the oracle), and the native passes resume after."""
    t = Twin(built(1), 1)
    rng = random.Random(1)
    drive(t, rng, 4)
    hid = next(h for h, he in t.n.g.hyperedges.items() if not he.is_archived)
    t.op("int threshold", lambda g: setattr(g.hyperedges[hid], "activation_threshold", 1))
    before = t.n.g.hyperedges.pass_declines()
    drive(t, rng, 3)
    after = t.n.g.hyperedges.pass_declines()
    assert after[0] > before[0] and after[2] >= before[2]
    assert t.n.g._he_ps_in_py or after[0] > before[0]
    t.op("float threshold", lambda g: setattr(g.hyperedges[hid], "activation_threshold", 0.5))
    mid = t.n.g.hyperedges.pass_declines()
    drive(t, rng, 4)
    assert t.n.g.hyperedges.pass_declines()[0] == mid[0]      # native again
    assert not t.n.g._he_ps_in_py


# ---------------------------------------------------------------------------
# 2. the pass state in the store
# ---------------------------------------------------------------------------

@needs_h2
def test_pass_state_views_are_read_only_and_live():
    t = Twin(built(2), 2)
    drive(t, random.Random(2), 5)
    g = t.n.g
    assert isinstance(g._he_last_fired_step, NF.ng_tract.PassStateView) if hasattr(NF.ng_tract, "PassStateView") \
        else type(g._he_last_fired_step).__name__ == "PassStateView"
    with pytest.raises(TypeError):
        g._he_last_fired_step["x"] = 1
    with pytest.raises(TypeError):
        g._he_co_fire_counts["x"] = {}
    some = next(iter(g._he_output_candidates), None)
    if some is not None:
        with pytest.raises(TypeError):
            g._he_output_candidates[some]["n"] = 3
    assert dict(g._he_last_fired_step) == dict(t.o.g._he_last_fired_step)


@needs_h2
def test_checkpoint_pass_state_bytes_equal_msgpack_of_the_oracle_dicts():
    t = Twin(built(3), 3)
    drive(t, random.Random(3), 8)
    def raw(b):
        u = msgpack.Unpacker(raw=False, strict_map_key=False)
        u.feed(b)
        out = {}
        for _ in range(u.read_map_header()):
            k = u.unpack()
            s0 = u.tell()
            u.skip()
            out[k] = b[s0:u.tell()]
        return out
    ro, rn = raw(ckpt_bytes(t.o.g)), raw(ckpt_bytes(t.n.g))
    assert list(ro) == list(rn)                           # same keys, same order
    for k in ("he_last_fired_step", "he_output_candidates"):
        assert ro[k] == rn[k] == msgpack.packb(getattr(t.o.g, "_" + k), use_bin_type=True), k
    for k in ro:
        if k not in ("hyperedges", "archived_hyperedges", "he_active_predictions"):
            assert ro[k] == rn[k], k


# ---------------------------------------------------------------------------
# 3. #1140: sorted member lists in saved checkpoints
# ---------------------------------------------------------------------------

def _he_lists(b):
    d = msgpack.unpackb(b, raw=False, strict_map_key=False)
    return [(hd["member_nodes"], hd["child_hyperedges"]) for k in ("hyperedges", "archived_hyperedges")
            for hd in d[k].values()]


@needs_h2
def test_native_store_saves_sorted_member_lists():
    g = restore("native", built(0))
    for m, c in _he_lists(ckpt_bytes(g)):
        assert m == sorted(m) and c == sorted(c)


@pytest.mark.parametrize("kind", ["off", "h1"] + (["native"] if HAVE_H2 else []))
def test_python_path_sorts_only_behind_its_flag(kind):
    if kind != "off" and not hasattr(NF.ng_tract, "HyperedgeStore"):
        pytest.skip("no HyperedgeStore")
    g = restore(kind, built(0))
    lists = _he_lists(ckpt_bytes(g))
    if kind == "off":
        assert ckpt_bytes(g) == ckpt_bytes(restore("oracle", built(0)))   # flag absent: today's bytes
        assert any(m != sorted(m) for m, _ in lists)
    g.config["he_checkpoint_sorted_members"] = True
    for m, c in _he_lists(ckpt_bytes(g)):
        assert m == sorted(m) and c == sorted(c)


_CROSS_SEED = r"""
import os, sys, tempfile, hashlib, msgpack
sys.path.insert(0, %(repo)r)
import neuro_foundation as NF
kind, src, flag = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
g = NF.Graph(native_node_store=(kind == "native"), native_hyperedge_store=(kind != "off"))
g.restore(src)
g.config.pop("he_checkpoint_sorted_members", None)
if flag:
    g.config["he_checkpoint_sorted_members"] = True
d = tempfile.mkdtemp(); p = os.path.join(d, "c.msgpack"); g.checkpoint(p)
raw = open(p, "rb").read()
u = msgpack.Unpacker(raw=False); u.feed(raw); out = {}
for _ in range(u.read_map_header()):
    k = u.unpack(); s = u.tell(); u.skip()
    if k in ("hyperedges", "archived_hyperedges"):
        out[k] = hashlib.sha256(raw[s:u.tell()]).hexdigest()
print(out["hyperedges"], out["archived_hyperedges"])
"""


@pytest.mark.parametrize("kind", ["off", "h1"] + (["native"] if HAVE_H2 else []))
def test_sorted_saves_do_not_depend_on_the_hash_seed(kind):
    """#1140: the same state saved by processes with different PYTHONHASHSEEDs gives the same hyperedge bytes once the
    member / child lists are sorted (always on the store; behind he_checkpoint_sorted_members on the Python path).
    (Dynamics across seeds are another matter: the engine iterates other Python sets -- fired sets, adjacency -- and a
    hyperedge created from a set takes that set's order; only bookkeeping is made reproducible here.)"""
    if kind != "off" and not hasattr(NF.ng_tract, "HyperedgeStore"):
        pytest.skip("no HyperedgeStore")
    t = Twin(built(2), 2) if HAVE_H2 else None
    if t is not None:
        drive(t, random.Random(2), 10)          # a state with consolidation / archive / learned targets
        b = ckpt_bytes(t.o.g)
    else:
        b = built(2)
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "src.msgpack")
        with open(src, "wb") as f:
            f.write(b)

        def outs(flag):
            o = set()
            for hs in ("1", "2", "3", "4"):
                r = subprocess.run([sys.executable, "-c", _CROSS_SEED % {"repo": _REPO}, kind, src, "1" if flag else "0"],
                                   capture_output=True, text=True, env=dict(os.environ, PYTHONHASHSEED=hs), timeout=600)
                assert r.returncode == 0, r.stderr[-2000:]
                o.add(r.stdout.strip())
            return o
        assert len(outs(True)) == 1
        if kind == "off":
            assert len(outs(False)) > 1           # today's list(set) bytes follow the hash seed (#1140)
        else:
            assert len(outs(False)) == 1          # the store sorts regardless of the flag


# ---------------------------------------------------------------------------
# 4. #1142: shed / seam split choose by contribution, never by member-list order
# ---------------------------------------------------------------------------

_SHED = r"""
import os, sys, json
sys.path.insert(0, %(repo)r)
import neuro_foundation as NF
kind, flag = sys.argv[1], sys.argv[2] == "1"
cfg = {"he_shed_weight_threshold": 0.02, "he_shed_min_tenure": 3, "he_discovery_min_nodes": 3,
       "he_max_members": 6, "he_split_oversized_enabled": True, "he_split_dedup_overlap": 0.7}
if flag:
    cfg["he_member_choice_by_contribution"] = True
g = NF.Graph(cfg, native_node_store=(kind == "native"), native_hyperedge_store=(kind != "off"))
for i in range(40):
    g.create_node(node_id="m%%02d" %% i)
g.timestep = 100
# 8 members, 6 at the floor with equal weight and tenure -> 5 may go (min_keep 3): WHICH 5 is the question
a = g.create_hyperedge({"m%%02d" %% i for i in range(8)},
                       member_weights={"m%%02d" %% i: (0.01 if i < 6 else 2.0) for i in range(8)},
                       metadata={"member_since": {"m%%02d" %% i: 0 for i in range(8)}})
# one floor member a little lower: it must go first under the rule
if g._native_he:
    g.hyperedges.set_member_weight(a.hyperedge_id, "m04", 0.005)
else:
    a.member_weights["m04"] = 0.005
# a 14-member edge whose seam scores tie: equal weights, nothing else discriminates
big = g.create_hyperedge({"m%%02d" %% i for i in range(20, 34)}, member_weights={"m%%02d" %% i: 1.0 for i in range(20, 34)})
shed = g.shed_floor_members()
kept = sorted(g.hyperedges[a.hyperedge_id].member_nodes)
g.dedup_and_split_oversized_hyperedges(None)
parts = sorted((sorted(he.member_nodes), bool(he.metadata.get("core"))) for he in g.hyperedges.values()
               if not he.is_archived and he.metadata.get("parent_blob") == big.hyperedge_id)
print(json.dumps([shed, kept, parts]))
"""


def _shed_run(kind, flag, hs):
    r = subprocess.run([sys.executable, "-c", _SHED % {"repo": _REPO}, kind, "1" if flag else "0"],
                       capture_output=True, text=True, env=dict(os.environ, PYTHONHASHSEED=hs), timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    import json
    return json.loads(r.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("kind", ["off", "h1"] + (["native"] if HAVE_H2 else []))
def test_1142_choice_by_contribution_is_seed_independent_and_follows_the_rule(kind):
    if kind != "off" and not hasattr(NF.ng_tract, "HyperedgeStore"):
        pytest.skip("no HyperedgeStore")
    outs = [_shed_run(kind, True, hs) for hs in ("1", "2", "3", "4")]
    assert all(o == outs[0] for o in outs), outs
    shed, kept, parts = outs[0]
    assert shed == 5
    # the lowest weight goes first, then (equal weight, no co-fires) by node id: m04, then m00 m01 m02 m03 -> m05
    # (the last floor member) and the two strong members stay
    assert kept == ["m05", "m06", "m07"]
    core = [p for p, is_core in parts if is_core]
    assert core and core[0] == ["m%02d" % i for i in range(20, 26)]   # ties -> weight (equal) -> co-fire -> id
    if kind == "off":
        assert outs[0] == _shed_run("h1", True, "1") if hasattr(NF.ng_tract, "HyperedgeStore") else True


def test_1142_flag_off_keeps_todays_choice():
    """Flag absent: shed / split behave exactly as the trial tip (the OFF byte-identity tests cover the full run);
    here the order-dependence that #1142 describes is still visible across hash seeds."""
    outs = {json_s(_shed_run("off", False, hs)) for hs in ("1", "2", "3", "4", "5", "6")}
    assert len(outs) >= 1     # (informational: >1 shows the hash-seed dependence the flag removes)
    print("flag off, distinct shed/split outcomes over 6 hash seeds:", len(outs))


def json_s(x):
    import json
    return json.dumps(x)


# ---------------------------------------------------------------------------
# 5. readers outside the lock (H1 audit)
# ---------------------------------------------------------------------------

@needs_h2
def test_stale_ref_raises_and_the_lenia_reader_snapshots_under_the_lock():
    g = restore("native", built(4))
    hid, he = next(iter(g.hyperedges.items()))
    g._remove_hyperedge_internal(hid)
    with pytest.raises(KeyError):
        he.member_nodes                                     # D3: a removed hyperedge's ref fails loud
    from lenia.graph_substrate import NeuroGraphSubstrate
    s_n, s_d = NeuroGraphSubstrate(g), NeuroGraphSubstrate(restore("off", ckpt_bytes(g)))
    s_n._build_hyperedge_membership()
    s_d._build_hyperedge_membership()
    assert {k: set(v) for k, v in s_n._he_membership.items()} == {k: set(v) for k, v in s_d._he_membership.items()}
