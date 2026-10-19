# ---- Changelog ----
# [2026-10-18] Claude (lane hyperedge-store-h2, #1136) — keep this file on the H1 path
# What: with both native stores on, new_graph turns the graph's H2 native passes off (the Python passes over the store).
# Why:  the H2 passes sum in the store's member order, so voltages match an oracle only when the order is handed to it;
#       that is test_hyperedge_store_h2.py's job. This file keeps proving H1 as it was.
# How:  _he_ps_to_python() then _native_he_passes = False, right after construction.
# -------------------
# [2026-10-18] Claude (lane hyperedge-store-h1, #1136) — equivalence: native hyperedge store (H1) ON / OFF vs the TRIAL TIP
"""H1 native hyperedge store: byte- and trace-equivalence against the trial tip (spec 2026-10-17 §1.5, §3.1).

BASE = the WHOLE neuro_foundation.py of the trial tip (BASE_REV, read from git, imported as a separate module; its
Graph.hyperedges is a dict of Hyperedge). BRANCH = this worktree, run OFF (native_hyperedge_store=False: the dict
path) and ON (ng_tract.HyperedgeStore; skipped when the installed wheel has none), each with the native node store
off and (when the wheel has it) on.

OFF must equal BASE bit for bit: per-step traces and checkpoint BYTES. ON must equal BASE on every hyperedge read and
write the engine makes: per step the fired node ids (order), fired hyperedge ids (order), every node voltage (bitwise),
the full state of every hyperedge and archived hyperedge (floats bitwise; member / child sets as sets; weights and
targets in order), the events the hyperedge code emits (in order) and the active HE predictions; and checkpoints equal
with member_nodes / child_hyperedges compared as sets (the one §1.5 exception), every other byte identical, and an
ON read -> write -> read -> write cycle byte-stable. Cross-reads: an ON-written file is read by OFF, BASE and the VPS
`main` module (VPS_REV) and re-emitted equal (set order aside).

Optional real-checkpoint pass: NG_HE_CKPT=<a COPY of a checkpoint> (the live plugin dir is refused).
Run with a fixed PYTHONHASHSEED (set iteration order is part of what is compared).
"""
import gc
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

BASE_REV = os.environ.get("NG_HE_BASE_REV", "ac9dd38")     # trial tip this branch was cut from
VPS_REV = os.environ.get("NG_HE_VPS_REV", "origin/main")    # Syl's line (reads an ON-written file)
HAVE_HE = hasattr(NF.ng_tract, "HyperedgeStore")
HAVE_NODES = hasattr(NF.ng_tract, "NodeStore")
LIVE_DIR = "/.claude/plugins/neurograph/"


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


BASE = _load(BASE_REV, "nf_base_he_h1")
MODES = ["off"] + (["on"] if HAVE_HE else [])
NODE_MODES = [False] + ([True] if HAVE_NODES else [])


def new_graph(mode, config=None, native_nodes=False):
    if mode == "base":
        return BASE.Graph(config)
    g = NF.Graph(config, native_node_store=native_nodes, native_hyperedge_store=(mode == "on"))
    if getattr(g, "_native_he_passes", False):
        # [2026-10-18] H2: this file proves H1 (the frozen Python passes over the store); the native passes, which
        # need the member order handed to the oracle for bitwise voltages, are proven in test_hyperedge_store_h2.py
        g._he_ps_to_python()
        g._native_he_passes = False
    assert g._native_he == (mode == "on")
    assert isinstance(g.hyperedges, dict) == (mode != "on")
    return g


def mod_of(mode):
    return BASE if mode == "base" else NF


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


def ckpt_bytes(g, mode=None):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.msgpack")
        g.checkpoint(p)
        with open(p, "rb") as f:
            return f.read()


def restore(mode, b, native_nodes=False):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.msgpack")
        with open(p, "wb") as f:
            f.write(b)
        g = new_graph(mode, native_nodes=native_nodes)
        g.restore(p)
        return g


def canon_ckpt(b):
    """§1.5 bar: member_nodes / child_hyperedges compared as sets, every other byte as is."""
    d = msgpack.unpackb(b, raw=False, strict_map_key=False)
    for key in ("hyperedges", "archived_hyperedges"):
        for hd in (d.get(key) or {}).values():
            for k in ("member_nodes", "child_hyperedges"):
                if isinstance(hd.get(k), list):
                    hd[k] = sorted(hd[k], key=repr)
    return msgpack.packb(d, use_bin_type=True)


def other_keys(b):
    """{top-level key: raw value bytes} for every key except the two hyperedge maps."""
    u = msgpack.Unpacker(raw=False, strict_map_key=False)
    u.feed(b)
    out = {}
    for _ in range(u.read_map_header()):
        k = u.unpack()
        s = u.tell()
        u.skip()
        if k not in ("hyperedges", "archived_hyperedges"):
            out[k] = b[s:u.tell()]
    return out


def fb(x):
    return struct.pack("<d", x) if isinstance(x, float) else x


class A(float):
    """An activation value: Σ over a member SET, so its last bits depend on set iteration order (§1.5)."""


class W(tuple):
    """member_weights items: key order follows member-set iteration where entries were added by iterating one."""


def he_state(he):
    return (he.hyperedge_id, tuple(sorted(he.member_nodes, key=repr)),
            W((k, fb(v)) for k, v in he.member_weights.items()),
            fb(he.activation_threshold), he.activation_mode.name, A(he.current_activation),
            tuple(he.output_targets), fb(he.output_weight), repr(he.metadata), he.is_learnable,
            he.refractory_period, he.refractory_remaining, he.activation_count, fb(he.pattern_completion_strength),
            tuple(sorted(he.child_hyperedges, key=repr)), he.level, fb(he.recent_activation_ema), he.is_archived,
            he.consolidation_state.value, he.creation_time)


def graph_state(g):
    return (
        tuple((nid, fb(n.voltage), n.refractory_remaining) for nid, n in g.nodes.items()),
        tuple((hid, he_state(he)) for hid, he in g.hyperedges.items()),
        tuple((hid, he_state(he)) for hid, he in g._archived_hyperedges.items()),
        tuple(sorted((hid, tuple(sorted(s))) for hid, s in g._node_hyperedges.items())),
        tuple((pid, ps.hyperedge_id, tuple(sorted(ps.predicted_targets)), A(ps.prediction_strength))
              for pid, ps in g._active_predictions.items()),
        tuple((h, s) for h, s in g._he_last_fired_step.items()),
        tuple((h, tuple(c.items())) for h, c in g._he_output_candidates.items()),
        tuple((h, tuple(c.items())) for h, c in g._he_co_fire_counts.items()),
    )


HE_EVENTS = ("prediction_created", "hyperedge_fired", "he_output_learned", "hyperedge_archived",
             "hyperedge_discovered", "he_state_transition", "hyperedge_seam_split",
             "hyperedges_seam_split_pass", "hyperedge_members_shed")


def attach_events(g, log):
    def norm(k, v):
        if isinstance(v, (list, set, tuple, frozenset)):
            return tuple(sorted(map(repr, v)))
        if k == "activation":
            return A(v)
        return fb(v)
    for ev in HE_EVENTS:
        g.register_event_handler(ev, lambda _ev=ev, **kw: log.append(
            (_ev, tuple(sorted((k, norm(k, v)) for k, v in kw.items())))))


STATS = {"act_not_bitwise": 0, "act_max_abs": 0.0, "weights_order_only": 0, "compared": 0}


def same(x, y, strict=False):
    """Exact equality, except the two documented set-order effects (counted in STATS): an activation's last bits
    (|diff| <= 1e-12) and member_weights key order (same items)."""
    STATS["compared"] += 1
    if isinstance(x, A) and isinstance(y, A):
        if struct.pack("<d", x) == struct.pack("<d", y):
            return True
        d = abs(float(x) - float(y))
        if d <= 1e-12 and not strict:
            STATS["act_not_bitwise"] += 1
            STATS["act_max_abs"] = max(STATS["act_max_abs"], d)
            return True
        return False
    if isinstance(x, W) and isinstance(y, W):
        if tuple(x) == tuple(y):
            return True
        if sorted(x) == sorted(y) and not strict:
            STATS["weights_order_only"] += 1
            return True
        return False
    if isinstance(x, tuple) and isinstance(y, tuple):
        return len(x) == len(y) and all(same(a, b, strict) for a, b in zip(x, y))
    if isinstance(x, list) and isinstance(y, list):
        return len(x) == len(y) and all(same(a, b, strict) for a, b in zip(x, y))
    return type(x) is type(y) and x == y


def ckpt_obj(b):
    """A checkpoint as comparable data: hyperedge sets as sets, weights / activations as W / A (see `same`)."""
    d = msgpack.unpackb(b, raw=False, strict_map_key=False)
    out = []
    for k, v in d.items():
        if k in ("hyperedges", "archived_hyperedges"):
            rows = []
            for hid, hd in v.items():
                row = []
                for f, x in hd.items():
                    if f in ("member_nodes", "child_hyperedges"):
                        x = tuple(sorted(x, key=repr))
                    elif f == "member_weights":
                        x = W((a, fb(c)) for a, c in x.items())
                    elif f == "current_activation":
                        x = A(x)
                    else:
                        x = repr(fb(x))
                    row.append((f, x))
                rows.append((hid, tuple(row)))
            out.append((k, tuple(rows)))
        elif k == "he_active_predictions":
            out.append((k, tuple((pid, tuple((f, A(x) if f == "prediction_strength" else repr(x))
                                             for f, x in pd.items())) for pid, pd in v.items())))
        else:
            out.append((k, msgpack.packb(v, use_bin_type=True)))
    return tuple(out)


HE_CFG = {"he_output_learning_window": 5, "he_output_min_co_fires": 2, "he_output_max_targets": 5,
          "he_member_evolution_min_co_fires": 2, "he_consolidation_eval_interval": 7,
          "he_speculative_to_candidate_min_count": 2, "he_candidate_to_consolidated_min_count": 3,
          "he_candidate_to_consolidated_min_age": 5, "he_experience_threshold": 4,
          "he_discovery_min_co_fires": 2, "he_discovery_window": 4, "he_discovery_min_nodes": 3,
          "he_consolidation_overlap": 0.5, "he_split_oversized_enabled": True, "he_max_members": 9,
          # shed stays order-free here (it truncates its candidate list in member-SET order when more members
          # qualify than may go: Python's own choice then depends on the hash seed); see test_shed_and_split
          "he_shed_weight_threshold": 0.02, "he_shed_min_tenure": 3, "prediction_window": 4,
          "default_threshold": 0.6, "three_factor_enabled": True}


def build(mode, seed, native_nodes=False, n_nodes=70, n_syn=350, n_he=40):
    def b(rng):
        g = new_graph(mode, dict(HE_CFG), native_nodes=native_nodes)
        M = mod_of(mode)
        ids = [g.create_node(node_id=f"n{seed}_{i}", metadata={"k": i}).node_id for i in range(n_nodes)]
        for _ in range(n_syn):
            a, c = rng.sample(ids, 2)
            g.create_synapse(a, c, weight=rng.uniform(0.1, 1.5), delay=rng.randint(1, 3))
        made = []
        modes = list(M.ActivationMode)
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
            kids = rng.sample(made, 2)
            g.create_hierarchical_hyperedge(kids, activation_threshold=0.5)
        dup = rng.choice(made)                         # an exact duplicate -> consolidation merges / subsumes
        g.create_hyperedge(set(g.hyperedges[dup].member_nodes), activation_threshold=0.5)
        g.timestep = rng.randint(10, 500)
        return g
    return seeded(seed, b)


def workload(rounds):
    def act(g, rng):
        log, snaps = [], []
        attach_events(g, log)
        for k in range(rounds):
            for nid in rng.sample(list(g.nodes), min(12, len(g.nodes))):
                g.stimulate(nid, rng.uniform(0.5, 3.0))
            r = g.step()
            log.append(("step", tuple(r.fired_node_ids), tuple(r.fired_hyperedge_ids)))
            log.append(graph_state(g))
            ids = rng.sample(list(g.nodes), min(6, len(g.nodes)))
            p = g.prime_and_propagate(ids, [1.5] * len(ids), steps=3, write_mode=(k % 2 == 0))
            log.append(tuple((e.node_id, e.firing_step, fb(e.voltage_at_fire)) for e in p.fired_entries))
            log.append(graph_state(g))
            if k % 4 == 1:
                g.inject_reward(0.3)
                g.inject_reward(-0.2, scope=set(rng.sample(list(g.nodes), 10)))
            if k % 5 == 2:
                g.discover_hyperedges(list(r.fired_node_ids))
            if k % 6 == 3 and len(g.nodes) > 30:              # removal mid-run (members, targets, emptied HEs)
                for nid in rng.sample(list(g.nodes), 3):
                    g.remove_node(nid)
            if k % 9 == 4:
                log.append(("consolidate", g.consolidate_hyperedges()))
                log.append(("split", g.dedup_and_split_oversized_hyperedges(None)))
                log.append(graph_state(g))
            if k % 10 == 9:
                snaps.append(ckpt_bytes(g))
        return log, snaps
    return act


def run(mode, start, act, seed, native_nodes=False):
    g = restore(mode, start, native_nodes=native_nodes)
    out = seeded(seed, lambda rng: act(g, rng), uuid_stream=1)
    return ckpt_bytes(g), out, g


SEEDS = list(range(6))
_BUILT = {}


def built(mode, seed, native_nodes=False):
    key = (mode, seed, native_nodes)
    if key not in _BUILT:
        _BUILT[key] = ckpt_bytes(build(mode, seed, native_nodes))
    return _BUILT[key]


# ---------------------------------------------------------------------------
# 1. graphs BUILT through the API (create_hyperedge / hierarchical / ref writes)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("native_nodes", NODE_MODES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("mode", MODES)
def test_built_graph(seed, mode, native_nodes):
    b = built(mode, seed, native_nodes)
    ref = built("base", seed)
    if mode == "off":
        assert b == ref                                   # OFF: byte-identical
    else:
        assert same(ckpt_obj(b), ckpt_obj(ref))           # ON: §1.5 bar (set order aside)
        g = restore("on", b)                              # ON read -> write -> read -> write: byte-stable
        b2 = ckpt_bytes(g)
        assert ckpt_bytes(restore("on", b2)) == b2


# ---------------------------------------------------------------------------
# 2. whole runs: every hyperedge path, ON / OFF against the trial tip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("native_nodes", NODE_MODES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("mode", MODES)
def test_whole_run(mode, seed, native_nodes):
    act = workload(30)
    start = built("base", seed)
    a, (la, sa), _ = run("base", start, act, seed)
    b, (lb, sb), _ = run(mode, start, act, seed, native_nodes)
    strict = mode == "off"
    assert len(la) == len(lb)
    for i, (x, y) in enumerate(zip(la, lb)):
        assert same(x, y, strict), f"trace item {i} differs"
    if strict:
        assert sa == sb and a == b
    else:
        assert len(sa) == len(sb) and all(same(ckpt_obj(x), ckpt_obj(y)) for x, y in zip(sa, sb))
        assert same(ckpt_obj(a), ckpt_obj(b))


@pytest.mark.parametrize("seed", SEEDS[:3])
@pytest.mark.parametrize("mode", MODES)
def test_save_restore_mid_run_continues(mode, seed):
    """run 12 -> save -> restore (same mode) -> run 12 more == the trial tip doing the same."""
    act = workload(12)

    def two_legs(m):
        b1, (l1, _), _ = run(m, built("base", seed), act, seed)
        b2, (l2, _), _ = run(m, b1, act, seed + 100)
        return b2, l1 + l2
    a, la = two_legs("base")
    b, lb = two_legs(mode)
    assert same(la, lb, mode == "off")
    assert same(ckpt_obj(a), ckpt_obj(b), mode == "off")
    if mode == "off":
        assert a == b


@pytest.mark.parametrize("seed", SEEDS[:3])
@pytest.mark.parametrize("mode", MODES)
def test_incremental_and_subgraph(mode, seed):
    def act(g, rng):
        for k in range(5):
            for nid in rng.sample(list(g.nodes), 10):
                g.stimulate(nid, 2.0)
            g.step()
        inc = g.capture_checkpoint(mod_of("base" if g.__class__ is BASE.Graph else "off").CheckpointMode.INCREMENTAL,
                                   detach=True)
        sub = g.extract_subgraph(set(rng.sample(list(g.nodes), 40)) | {"ghost"})
        sub["extraction_metadata"]["missing_nodes"].sort()
        for d in (inc, sub):
            for hd in d["hyperedges"].values():
                hd["member_nodes"].sort(); hd["child_hyperedges"].sort()
        inc["hyperedges"] = dict(sorted(inc["hyperedges"].items()))   # dirty-set order
        return (msgpack.packb(inc, use_bin_type=True), msgpack.packb(sub, use_bin_type=True),
                repr(g.get_telemetry()), [he.hyperedge_id for he in g.get_hyperedges(next(iter(g.nodes)))],
                sorted(g.get_archived_hyperedges()))
    start = built("base", seed)
    a, ra, _ = run("base", start, act, seed)
    b, rb, _ = run(mode, start, act, seed)
    assert ra[2:] == rb[2:]
    for x, y in zip(ra[:2], rb[:2]):     # INCREMENTAL / subgraph dicts (lists sorted above)
        assert same(ckpt_obj(x), ckpt_obj(y), mode == "off")


# ---------------------------------------------------------------------------
# 3. cross-mode reads: ON-written files read by OFF / BASE / VPS main; old files read by ON
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAVE_HE, reason="ng_tract without HyperedgeStore")
@pytest.mark.parametrize("seed", SEEDS)
def test_cross_mode_round_trips(seed):
    act = workload(10)
    on_bytes, _, _ = run("on", built("base", seed), act, seed)
    base_bytes, _, _ = run("base", built("base", seed), act, seed)
    assert same(ckpt_obj(on_bytes), ckpt_obj(base_bytes))
    def he_maps(b):
        d = msgpack.unpackb(canon_ckpt(b), raw=False, strict_map_key=False)
        return msgpack.packb([d["hyperedges"], d["archived_hyperedges"]], use_bin_type=True)
    for reader in ("base", "off", "on"):
        again = ckpt_bytes(restore(reader, on_bytes))
        assert he_maps(again) == he_maps(on_bytes), reader               # every hyperedge byte kept (set order aside)
        # restore() itself normalizes some non-hyperedge keys (he_active_predictions, delay_buffer) for ANY file:
        # compare against the same reader re-saving the trial tip's own file
        assert same(ckpt_obj(again), ckpt_obj(ckpt_bytes(restore(reader, base_bytes)))), reader
    def raw_he(b):
        u = msgpack.Unpacker(raw=False, strict_map_key=False)
        u.feed(b)
        out = {}
        for _ in range(u.read_map_header()):
            k = u.unpack()
            st = u.tell()
            u.skip()
            if k in ("hyperedges", "archived_hyperedges"):
                out[k] = b[st:u.tell()]
        return out
    b1 = ckpt_bytes(restore("on", on_bytes))
    b2 = ckpt_bytes(restore("on", b1))
    assert raw_he(b1) == raw_he(on_bytes) == raw_he(b2)              # Rust read -> write -> read -> write: stable
    # (he_active_predictions' predicted_targets is a set saved as list(set): restore -> save re-orders it on EVERY
    #  reader, the trial tip included, so the whole file has no fixed point; that key is not hyperedge state)
    p1 = ckpt_bytes(restore("base", on_bytes))
    assert other_keys(b1).keys() == other_keys(p1).keys()
    assert all(other_keys(b1)[k] == other_keys(p1)[k] for k in other_keys(p1) if k != "he_active_predictions")
    assert he_maps(ckpt_bytes(restore("on", base_bytes))) == he_maps(base_bytes)   # Rust reads old files


VPS = None


def _vps():
    global VPS
    if VPS is None:
        VPS = _load(VPS_REV, "nf_vps_main_he_h1")
    return VPS


@pytest.mark.skipif(not HAVE_HE, reason="ng_tract without HyperedgeStore")
@pytest.mark.parametrize("seed", SEEDS[:2])
def test_vps_main_reads_an_on_written_file(seed):
    on_bytes, _, _ = run("on", built("base", seed), workload(6), seed)
    V = _vps()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.msgpack")
        with open(p, "wb") as f:
            f.write(on_bytes)
        g = V.Graph()
        g.restore(p)
        assert set(g.hyperedges) == set(msgpack.unpackb(on_bytes, raw=False)["hyperedges"])
        q = os.path.join(d, "d.msgpack")
        g.checkpoint(q)
        with open(q, "rb") as f:
            again = f.read()
    da = msgpack.unpackb(canon_ckpt(again), raw=False)
    db = msgpack.unpackb(canon_ckpt(on_bytes), raw=False)
    assert da["hyperedges"] == db["hyperedges"] and da["archived_hyperedges"] == db["archived_hyperedges"]


# ---------------------------------------------------------------------------
# 4. the engine contract with the store ON
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAVE_HE, reason="ng_tract without HyperedgeStore")
def test_engine_ref_contract():
    g = new_graph("on")
    for i in range(4):
        g.create_node(node_id=f"n{i}")
    he = g.create_hyperedge({"n0", "n1", "n2"}, output_targets=["n0", "n3"])
    assert isinstance(he, NF.ng_tract.HyperedgeRef) and g.hyperedges[he.hyperedge_id] is he
    assert he.creation_time == g.timestep and isinstance(he.member_nodes, frozenset)
    with pytest.raises(AttributeError):
        he.member_nodes.add("n3")                        # fail loud, not silent (§1.3)
    he.level = 2
    assert g.hyperedges[he.hyperedge_id].level == 2      # write-through
    g.remove_node("n0")                                  # a member that is also a target (as the dict path)
    assert he.member_nodes == {"n1", "n2"} and "n0" not in he.member_weights and he.output_targets == ("n3",)
    hid = he.hyperedge_id
    g.remove_node("n1")
    g.remove_node("n2")                                  # last member: the hyperedge goes, its ref is stale (D3)
    assert hid not in g.hyperedges
    with pytest.raises(KeyError):
        he.level


@pytest.mark.skipif(not HAVE_HE, reason="ng_tract without HyperedgeStore")
def test_archive_aliasing_matches_the_dicts():
    """consolidate's seatbelt / subsumption put the SAME object in both dicts; later live writes show in both."""
    def drive(mode):
        g = new_graph(mode, dict(HE_CFG, he_max_members=3, he_consolidation_overlap=0.5))
        for i in range(6):
            g.create_node(node_id=f"n{i}")
        a = g.create_hyperedge({"n0", "n1", "n2"})
        b = g.create_hyperedge({"n1", "n2", "n3"})
        b.refractory_remaining = 2
        g.consolidate_hyperedges()                       # union 4 > 3: b archived, kept in both dicts
        out = [(hid, he_state(he)) for hid, he in g._archived_hyperedges.items()]
        g.step()
        g.step()                                         # the live object's refractory decrements...
        g.remove_node("n3")                              # ...and remove_node edits its members
        out.append([(hid, he_state(he)) for hid, he in g._archived_hyperedges.items()])
        out.append([(hid, he_state(he)) for hid, he in g.hyperedges.items()])
        return out, ckpt_obj(ckpt_bytes(g))
    ra = seeded(5, lambda rng: drive("base"))
    rb = seeded(5, lambda rng: drive("on"))
    assert same(ra, rb)


@pytest.mark.parametrize("mode", MODES)
def test_shed_and_split(mode):
    """The dream-side member removals / additions (shed_floor_members, the dedup fold and the seam split's orphan
    safety) through the store, on cases whose outcome does not depend on member-set iteration order."""
    def drive(m):
        g = new_graph(m, dict(HE_CFG, he_max_members=6, he_shed_weight_threshold=0.02, he_shed_min_tenure=3,
                             he_discovery_min_nodes=3, he_split_dedup_overlap=0.7))
        for i in range(30):
            g.create_node(node_id=f"n{i:02d}")
        g.timestep = 100
        a = g.create_hyperedge({f"n{i:02d}" for i in range(8)},
                               member_weights={f"n{i:02d}": (0.01 if i < 3 else 1.0 + i / 10) for i in range(8)},
                               metadata={"member_since": {f"n{i:02d}": 0 for i in range(8)}})
        big = {f"n{i:02d}" for i in range(10, 24)}
        b = g.create_hyperedge(big, member_weights={n: 1.0 + k / 7 for k, n in enumerate(sorted(big))})
        c = g.create_hyperedge(big | {"n25"}, member_weights={n: 0.5 + k / 9 for k, n in enumerate(sorted(big | {"n25"}))},
                               metadata={"member_since": {"n25": 7}})
        out = [g.shed_floor_members()]
        out.append([(h, he_state(x)) for h, x in g.hyperedges.items()])
        out.append(g.dedup_and_split_oversized_hyperedges(None))
        out.append(sorted((tuple(sorted(x.member_nodes)), W(sorted((k, fb(v)) for k, v in x.member_weights.items())),
                           x.is_archived, repr(sorted(x.metadata.items(), key=repr)))
                          for x in g.hyperedges.values()))
        out.append(sorted(tuple(sorted(x.member_nodes)) for x in g._archived_hyperedges.values()))
        return out
    ra = seeded(3, lambda rng: drive("base"))
    rb = seeded(3, lambda rng: drive(mode))
    assert same(ra, rb, mode == "off")
    assert ra[0] == 3 and ra[2] > 0                   # the case really sheds and splits


@pytest.mark.skipif(not HAVE_HE, reason="ng_tract without HyperedgeStore")
@pytest.mark.parametrize("seed", SEEDS[:3])
def test_hot_readers_match(seed):
    """surfacing._score_node, the daemon's salience score and tonic_thread._build_he_index read the same counts
    through membership_count / member_index as through the scan they replace."""
    from ces_config import CESConfig
    from surfacing import SurfacingMonitor
    from tonic_thread import TonicThread
    import syl_daemon
    on_bytes, _, _ = run("on", built("base", seed), workload(8), seed)
    g_on, g_off = restore("on", on_bytes), restore("off", on_bytes)
    for g in (g_on, g_off):
        g.remove_node(next(iter(g.nodes)))
    sm_on, sm_off = SurfacingMonitor(g_on, None, CESConfig()), SurfacingMonitor(g_off, None, CESConfig())
    sr_on, sr_off = syl_daemon.SalienceReader(g_on, None), syl_daemon.SalienceReader(g_off, None)
    for nid in list(g_off.nodes):
        assert sm_on._score_node(nid, g_on.nodes[nid]) == sm_off._score_node(nid, g_off.nodes[nid])
        assert g_on.hyperedges.membership_count(nid) == sum(1 for he in g_off.hyperedges.values() if nid in he.member_nodes)
        assert sr_on._salience_score(nid, g_on.nodes[nid]) == sr_off._salience_score(nid, g_off.nodes[nid])
    assert TonicThread(g_on, None)._build_he_index() == TonicThread(g_off, None)._build_he_index()


def test_opt_in_resolution():
    prev = NF.set_native_hyperedge_store_default(False)
    try:
        assert isinstance(NF.Graph().hyperedges, dict)
        assert NF.set_native_hyperedge_store_default(True) is False
        assert isinstance(NF.Graph().hyperedges, dict) == (not HAVE_HE)
        assert isinstance(NF.Graph(native_hyperedge_store=False).hyperedges, dict)
        NF.set_native_hyperedge_store_default(False)
        assert isinstance(NF.Graph().hyperedges, dict)
        cfg = NF.Graph(native_hyperedge_store=True).config
        assert "native_hyperedge_store" not in cfg and cfg == NF.Graph().config   # never saved in a checkpoint
    finally:
        NF.set_native_hyperedge_store_default(prev)


# ---------------------------------------------------------------------------
# 5. optional: a COPY of the real checkpoint (sequential, memory-bounded)
# ---------------------------------------------------------------------------

CKPT = os.environ.get("NG_HE_CKPT")


@pytest.mark.skipif(not (CKPT and HAVE_HE), reason="set NG_HE_CKPT to a checkpoint COPY (and a wheel with HyperedgeStore)")
def test_checkpoint_copy_migration():
    assert LIVE_DIR not in os.path.abspath(CKPT), "refusing the live checkpoint"
    with open(CKPT, "rb") as f:
        raw = f.read()
    g = restore("on", raw)
    on_state = (tuple((h, he_state(x)) for h, x in g.hyperedges.items()),
                tuple((h, he_state(x)) for h, x in g._archived_hyperedges.items()),
                {k: frozenset(v) for k, v in g._node_hyperedges.items()},
                dict(g._he_output_candidates), dict(g._he_last_fired_step))
    on_bytes = ckpt_bytes(g)
    del g
    gc.collect()
    on2 = ckpt_bytes(restore("on", on_bytes))
    assert on2 == on_bytes                                # Rust read -> write -> read -> write: byte-stable
    del on2
    gc.collect()
    b = restore("base", raw)
    base_state = (tuple((h, he_state(x)) for h, x in b.hyperedges.items()),
                  tuple((h, he_state(x)) for h, x in b._archived_hyperedges.items()),
                  {k: frozenset(v) for k, v in b._node_hyperedges.items()},
                  dict(b._he_output_candidates), dict(b._he_last_fired_step))
    base_bytes = ckpt_bytes(b)
    del b
    gc.collect()
    assert on_state == base_state                         # every HE and archived HE, field by field; indices equal
    assert other_keys(on_bytes) == other_keys(base_bytes)
    assert canon_ckpt(on_bytes) == canon_ckpt(base_bytes)  # nothing re-ordered but the member / child lists


def test_zz_report_set_order_effects():
    """Not a check: records how often the two documented set-order effects were seen (NG_HE_STATS=<file>)."""
    rep = os.environ.get("NG_HE_STATS")
    if rep:
        import json
        with open(rep, "a") as f:
            f.write(json.dumps(dict(STATS, modes=MODES)) + "\n")
