# ---- Changelog ----
# [2026-09-30] Claude Code (claude-sonnet-5-5, Z12 BUILD worker seat, dispatch #11228, TURN A) — tests for the
#   118-want TEXT repair ONE-SHOT TOOL (handoffs/z12-want-text-repair/oneshot-tool/want_text_repair_oneshot.py).
# What: SYNTHETIC seeded graphs only - each outcome class, the collision rule (drops AND lists), Choice Clause
#   byte-equality (both wants + the constitutional node + every unrepaired want) with the deny-check, the mapping and
#   INVERSE mapping, the rim / pred_weights assertion, id-follows-text, the stamp / P1 / P5 refusals, the hard target
#   refusals, import isolation, idempotence, the retirement refusals, V1-V19 negative cases, the approvals refusals,
#   P4/P6 probe gates, and the whole Phase-2 path on a SYNTHETIC directory with fake probes.
# Why: plan-004 [R4b] sections 4-7 + the assignment (build-want-text-repair-118.md TURN A item 2). NO real checkpoint,
#   tract, Syl directory or primary checkout is opened: the tool's recorded-path constants are patched to tmp dirs.
# How: a session fixture builds a tiny real Graph + SimpleVectorDB + the six files under tmp_path; the pinned parser
#   is imported from the PIN worktree (P379 preamble: the tests FAIL if cc_ng_organism is not the PIN copy). All
#   synthetic text is invented filler; no raw want text or conversation excerpt appears in this file.
# -------------------
"""Run ONCE, targeted, from the tool worktree root:
    env -u PYTHONPATH -u NG_EMBED_REMOTE PYTHONDONTWRITEBYTECODE=1 python3 -B -m pytest \\
        tests/test_want_text_repair_oneshot.py -s -q -p no:cacheprovider
"""
import contextlib
import copy
import gc
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

PIN_ROOT = Path.home() / "NeuroGraph-worktrees" / "z12-want-repair-pin-ae798b9"
TOOL_PATH = (Path(__file__).resolve().parents[1] / "handoffs" / "z12-want-text-repair" / "oneshot-tool"
             / "want_text_repair_oneshot.py")
_TOOL_ROOT = Path(__file__).resolve().parents[1]
assert PIN_ROOT.is_dir(), "the PIN worktree %s is missing - the tests FAIL, never skip" % PIN_ROOT
assert TOOL_PATH.is_file(), "the tool file is missing"
sys.path.insert(0, str(PIN_ROOT))          # before ANY NG import: the pin tree resolves first

import msgpack  # noqa: E402
import numpy as np  # noqa: E402

_spec = importlib.util.spec_from_file_location("want_text_repair_oneshot_under_test", TOOL_PATH)
tool = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = tool
_spec.loader.exec_module(tool)

ORIG = types.SimpleNamespace(SYL=tool.SYL_CHECKPOINTS, PRIMARY=tool.PRIMARY_NEUROGRAPH,
                             RECORDED=tool.RECORDED_CC_CHECKPOINT_DIR, BACKUPS=tool.BACKUPS_ROOT)
BT = chr(96)
MIN_LEN = 100
NG_NAMES = ("cc_ng_organism", "neuro_foundation", "universal_ingestor", "checkpoint_guardian", "ng_lite", "ng_embed",
            "neurograph_rpc", "cc_ng_host", "activation_persistence")


# ==================================================================================================
# fixtures: the pinned modules, the P379 preamble, the patched recorded paths, the synthetic world
# ==================================================================================================

@pytest.fixture(scope="module")
def pinned():
    return tool.load_pinned(str(PIN_ROOT))


@pytest.fixture(scope="module", autouse=True)
def _p379_preamble(pinned):
    """Print module paths + NG-module state at session start; FAIL every test here if cc_ng_organism (or any NG
    module) resolves outside the PIN worktree copy."""
    print("\n" + "\n".join(tool.p379_lines(pinned)))
    for n in NG_NAMES:
        mod = sys.modules.get(n)
        print("P379 %-22s %s" % (n, getattr(mod, "__file__", "not loaded") if mod else "not loaded"))
    print("P379 NG_EMBED_* names set in env: %s" % (sorted(k for k in os.environ if k.startswith("NG_EMBED")) or "none"))
    org_file = Path(sys.modules["cc_ng_organism"].__file__).resolve()
    assert org_file == PIN_ROOT / "cc_ng_organism.py", "cc_ng_organism is NOT the PIN worktree copy: %s" % org_file
    for n in NG_NAMES:
        mod = sys.modules.get(n)
        f = getattr(mod, "__file__", None) if mod else None
        if f:
            assert PIN_ROOT in Path(f).resolve().parents, "%s resolves outside the PIN worktree: %s" % (n, f)
            assert _TOOL_ROOT not in Path(f).resolve().parents, "%s resolves into the tool worktree: %s" % (n, f)
    yield


def prose(tag, n=14):
    return " ".join("%s-w%02d" % (tag, i) for i in range(n))


X1, X2 = "the real intent one", "the real intent two"
XC, XE, XCH = "the collision want text", "the existing want text", "the chain want text"
X5 = " ".join("word%04d" % i for i in range(600))          # a 5,000+ character nested want


def _specs(wid, hint=None):
    specs = {}
    hint = hint or {}

    def add(name, C, T, id=None):
        specs[name] = {"C": C, "T": T, "id": id or wid(T)}

    T = prose("sepa") + " [WANT]" + X1
    add("SEP1", "preamble one. " + (hint["SEP1"] + ". " if "SEP1" in hint else "") + "[WANT]" + T + "[/WANT] trailing text.", T)
    T = BT + "code" + BT + " " + prose("sepb") + " [WANT]" + X2
    add("SEP2", "[WANT]" + T + "[/WANT] tail", T)
    L = prose("gen")
    add("GEN", "lead in. " + (hint["GEN"] + ". " if "GEN" in hint else "") + "[WANT]" + L + "[/WANT] done.", L)
    T = prose("genm") + " see https://x.org/[WANT]z[/WANT] ok"
    add("GENM", "[WANT]" + T + "[/WANT]", T)
    L = prose("none")
    add("NONE", "see " + BT + "[WANT]" + L + "[/WANT]" + BT + " done", L)
    T = "A " + prose("ovl") + " \\[WANT] B" + prose("ovm")
    add("OVL", "[WANT]" + T + "[/WANT] C tail[/WANT]", T)
    T = prose("anom") + BT + " C"
    add("ANOM", "[WANT]A " + BT + "[WANT]" + T + "[/WANT]", T)
    L = prose("anch")
    add("ANCH", "[WANT]" + L + "[/WANT] repeated " + L, L)
    T = prose("big") + " [WANT]" + X5
    add("BIG", "intro [WANT]" + T + "[/WANT] end", T)
    T1, T2 = prose("ca") + " [WANT]" + XC, prose("cb") + " [WANT]" + XC
    add("COLLA", "[WANT]" + T1 + "[/WANT]", T1)
    add("COLLB", "[WANT]" + T2 + "[/WANT]", T2)
    T = prose("ce") + " [WANT]" + XE
    add("COLLE", "[WANT]" + T + "[/WANT]", T)
    T = prose("cha") + " [WANT]" + XCH
    add("CHAINA", "[WANT]" + T + "[/WANT]", T)
    L = prose("chb")
    add("CHAINB", "see " + BT + "[WANT]" + L + "[/WANT]" + BT + " done", L, id=wid(XCH))   # id == A's would-be new id
    L = prose("idmm")
    add("IDMM", "[WANT]" + L + "[/WANT]", L, id="cc:want::deadbeef00000001")
    return specs


def _add_extras(g, vdb, ids, wid):
    """Delta-build (#11805) extras for the equivalence tests: an S source with NO vdb entry, an S source with EMPTY
    content, an S source with content but NO marker, a non-ASCII SEPARATE candidate, a conversational marker node that
    is nobody's source, and vdb-only entries (marker / plain / empty / rich metadata). Invented filler text only."""
    vec = np.array([0.5, 0.5, 0.0, 0.0])

    def want(name, text, src):
        ids[name] = wid(text)
        g.create_node(node_id=ids[name], metadata={"kind": "want", "want_text": text, "want_state": "open",
                      "provenance": "cc_authored", "source_node": src, "creation_mode": "conversational"})
        g.create_synapse(src, ids[name], weight=0.3)

    g.create_node(node_id="cc:conv::nocontent", metadata={"creation_mode": "conversational"})       # in the graph, NOT in the vdb
    want("NOCONTENT", prose("nocn"), "cc:conv::nocontent")
    g.create_node(node_id="cc:conv::emptyc", metadata={"creation_mode": "conversational"})
    vdb.insert("cc:conv::emptyc", vec, "", {})
    want("EMPTYC", prose("empt"), "cc:conv::emptyc")
    want("NOMARK", prose("nomk"), "cc:conv::plain")                                                  # S source has content, no marker
    xna = "caf\u00e9 intent \u2713 \u65e5\u672c\u8a9e"
    tna = "na\u00efve " + prose("nasc") + " [WANT]" + xna
    g.create_node(node_id="cc:conv::nonascii", metadata={"creation_mode": "conversational"})
    vdb.insert("cc:conv::nonascii", vec, "pr\u00e9ambule \u2603 [WANT]" + tna + "[/WANT] fin \u2713", {"k": ["\u00e9", 1]})
    want("NONASCII", tna, "cc:conv::nonascii")
    g.create_node(node_id="cc:conv::markeronly", metadata={"creation_mode": "conversational"})       # a marker node, nobody's source
    vdb.insert("cc:conv::markeronly", vec, "[WANT]" + prose("mko") + "[/WANT] \u00fcber", {"a": {"b": [1, 2, 3]}})
    vdb.insert("orph:marker", vec, "orphan [WANT]x[/WANT]", {"z": 1})                                # vdb-only entries
    vdb.insert("orph:plain", vec, "orphan plain \u00fc \u2713", {})
    vdb.insert("orph:empty", vec, "", {"rich": {"deep": list(range(20))}})


def build_world(base: Path, pinned, *, tag="w", include_protected=True, flagged_in_scope=False, hint_phrases=None, extras=False):
    """A tiny REAL checkpoint pair (Graph + SimpleVectorDB) plus the four small files, under `base`."""
    org, nf, ui = pinned.org, pinned.nf, pinned.ui
    wid = org.want_id_for_text
    ws = base / "ws"
    ckpt = ws / "checkpoints"
    ckpt.mkdir(parents=True)
    daemon = base / "cc-ng-daemon.py"
    daemon.write_text("CC_NG_WORKSPACE = os.path.expanduser('%s')\nCHECKPOINT_DIR = os.path.join(CC_NG_WORKSPACE, 'checkpoints')\n" % ws)
    (base / "backups").mkdir()
    (base / "conduit").mkdir()
    (base / "conduit" / "frame-a.bin").write_bytes(b"synthetic conduit file")
    g, vdb = nf.Graph(), ui.SimpleVectorDB()
    vec = np.array([1.0, 0.0, 0.0, 0.0])
    specs = _specs(wid, hint_phrases)
    ids = {}
    for name, s in specs.items():
        src = "cc:conv::" + name
        g.create_node(node_id=src, metadata={"creation_mode": "conversational"})
        vdb.insert(src, vec, s["C"], {})
        g.create_node(node_id=s["id"], metadata={
            "kind": "want", "want_text": s["T"], "want_state": "open", "provenance": "cc_authored", "source_node": src,
            "creation_mode": "conversational", "poincare_dir": b"\x00\x01" + name.encode()})
        g.create_synapse(src, s["id"], weight=0.3)
        ids[name] = s["id"]
    # a want whose source conversation node is absent from the graph
    miss_text = prose("miss")
    ids["MISS"] = wid(miss_text)
    g.create_node(node_id=ids["MISS"], metadata={"kind": "want", "want_text": miss_text, "want_state": "open",
                  "provenance": "cc_authored", "source_node": "cc:conv::absent", "creation_mode": "conversational"})
    # short (unrepaired) wants: the existing X_EX want, one ordinary, and the two Choice Clause wants (synthetic text)
    g.create_node(node_id="cc:conv::plain", metadata={"creation_mode": "conversational"})
    vdb.insert("cc:conv::plain", vec, prose("plain"), {})
    shorts = [(wid(XE), XE), (wid("a short want"), "a short want")]
    if include_protected:
        shorts += [(tool.CHOICE_CLAUSE_IDS[0], "synthetic choice clause want one"),
                   (tool.CHOICE_CLAUSE_IDS[1], "synthetic choice clause want two")]
    for nid, txt in shorts:
        g.create_node(node_id=nid, metadata={"kind": "want", "want_text": txt, "want_state": "open",
                      "provenance": "cc_authored", "source_node": "cc:conv::plain", "creation_mode": "conversational"})
    short1, cc1, cc2, rim = wid("a short want"), tool.CHOICE_CLAUSE_IDS[0], tool.CHOICE_CLAUSE_IDS[1], tool.CONSTITUTIONAL_ID
    if include_protected:
        g.create_node(node_id=rim, metadata={"constitutional": True})
    if flagged_in_scope:                                  # a want IN S under an ordinary id that carries the constitutional flag
        flag_md = dict(flagged_in_scope) if isinstance(flagged_in_scope, dict) else {"constitutional": True}
        flag_text = prose("flag")
        ids["FLAG"] = wid(flag_text)
        g.create_node(node_id=ids["FLAG"], metadata={"kind": "want", "want_text": flag_text, "want_state": "open",
                      **{"provenance": "cc_authored", "source_node": "cc:conv::plain", "creation_mode": "conversational"},
                      **flag_md})
    g.create_node(node_id="n1", metadata={})
    g.create_node(node_id="n2", metadata={})
    if extras:
        _add_extras(g, vdb, ids, wid)
    edges = [(ids["SEP1"], ids["GEN"], 0.4), (short1, ids["SEP1"], 0.5), ("n1", ids["SEP1"], 0.2)]
    if include_protected:
        edges += [(cc1, ids["SEP1"], 0.6), (cc2, ids["GEN"], 0.6), (rim, ids["SEP1"], 0.7), (ids["SEP2"], rim, 0.7),
                  (rim, ids["GEN"], 0.7)]
    for a, b, w in edges:
        g.create_synapse(a, b, weight=w)
    syn = g.create_synapse("n1", "n2", weight=0.2)
    syn.metadata = {"creation_mode": "surprise_driven", "expected_target": ids["SEP2"], "timestep": 0}
    g.nodes["n1"].pred_weights = {ids["SEP1"]: 0.5, ids["GEN"]: 0.2}
    if include_protected:
        g.nodes[cc1].pred_weights = {ids["SEP1"]: 0.7, ids["GEN"]: 0.1}
        g.nodes[rim].pred_weights = {ids["SEP1"]: 0.4}
    g.nodes[short1].pred_weights = {ids["SEP2"]: 0.3}
    g.nodes[ids["SEP1"]].pred_weights = {"n2": 0.3}
    he = g.create_hyperedge({ids["SEP1"], "n1", "n2"}, member_weights={ids["SEP1"]: 1.0, "n1": 0.5, "n2": 0.5},
                            output_targets=[ids["SEP2"]])
    he2 = g.create_hyperedge({ids["NONASCII"], "n1", "n2"}, member_weights={ids["NONASCII"]: 1.0, "n1": 0.5, "n2": 0.5}) if extras else None
    cap = g.capture_checkpoint(nf.CheckpointMode.FULL)
    if extras:          # an ARCHIVED hyperedge listing an S id: the canonical restore does NOT index it into _node_hyperedges
        cap["archived_hyperedges"][he2.hyperedge_id] = dict(cap["hyperedges"].pop(he2.hyperedge_id), is_archived=True)
    far = 10 ** 6
    pred = {"prediction_id": "p1", "source_node_id": ids["SEP1"], "target_node_id": "n1", "strength": 0.5, "confidence": 0.5,
            "created_at": 0, "expires_at": far, "chain_depth": 0, "via_hyperedge": None, "pre_charge_applied": 0.0}
    cap["active_predictions"] = {"p1": pred}
    cap["prediction_outcomes"] = [{"prediction": dict(pred, source_node_id="n1", target_node_id=ids["SEP2"]),
                                   "confirmed": True, "resolved_at": 1, "actual_firing_nodes": [ids["SEP1"], "n2"]}]
    cap["he_active_predictions"] = {"hp1": {"hyperedge_id": he.hyperedge_id, "predicted_targets": [ids["SEP2"], "n2"],
                                            "prediction_strength": 0.5, "prediction_timestamp": 0, "prediction_window": far,
                                            "confirmed_targets": [ids["SEP2"]]}}
    cap["he_output_candidates"] = {he.hyperedge_id: {ids["SEP2"]: 3, "n2": 1}}
    cap["novel_sequence_log"] = [{"source": ids["SEP1"], "firing_nodes": [ids["SEP2"], "n1"], "timestep": 1}]
    cap["delay_buffer"] = {str(far): [[ids["SEP1"], 0.5], ["n1", 0.1]]}
    cap["recent_spikes"] = {ids["SEP1"]: [1, 2], "n1": [3]}
    g.write_checkpoint(str(ckpt / "main.msgpack"), cap)
    vdb.save(str(ckpt / "vectors.msgpack"))
    entries = {nid: {"voltage": 0.5, "last_spike_time": 1.0, "excitability": 1.0, "timestamp": 1.0}
               for nid in (ids["SEP1"], ids["SEP2"], ids["GEN"], "n1")}
    (ckpt / "main.msgpack.activations.json").write_text(json.dumps({"version": "1.0", "saved_at": 123.0, "timestep": 0, "entries": entries}))
    (ckpt / "main.msgpack.guard_state.json").write_text(json.dumps({"healthy": True}))
    (ckpt / "main.msgpack.manifest.json").write_text(json.dumps({"guardian_nodes": len(g.nodes), "nodes": len(g.nodes)}))
    (ckpt / "commons.msgpack").write_bytes(msgpack.packb({"k": "v"}, use_bin_type=True))
    (ws / "daemon.log").write_text("synthetic daemon log")
    os.utime(ws / "daemon.log", (1_000_000_000, 1_000_000_000))
    gen = ckpt / "generations" / "20260923T104614Z"          # the incidental hard-link partners (Exec P428), synthetic
    gen.mkdir(parents=True)
    os.link(ckpt / "main.msgpack", gen / "main.msgpack")
    os.link(ckpt / "vectors.msgpack", gen / "vectors.msgpack")
    wants = [n for n, nd in g.nodes.items() if nd.metadata.get("kind") == "want"]
    scope = sorted(n for n in wants if len(g.nodes[n].metadata["want_text"]) > MIN_LEN)
    w = types.SimpleNamespace(base=base, ws=ws, ckpt=ckpt, daemon=daemon, backups=base / "backups", conduit=base / "conduit",
                              ids=ids, specs=specs, scope=scope, short1=short1, cc1=cc1, cc2=cc2, rim=rim, he=he.hyperedge_id,
                              new={"SEP1": wid(X1), "SEP2": wid(X2), "BIG": wid(X5)},
                              gen=gen, log=ws / "daemon.log",
                              expect={"wants": len(wants), "protected": len(wants) + (1 if include_protected else 0), "scope": len(scope)})
    del g, vdb
    return w


def patch_world(mp, w):
    """Point the tool's RECORDED constants at the synthetic world - the ONLY way the guards pass in tests."""
    mp.setattr(tool, "RECORDED_CC_CHECKPOINT_DIR", os.path.realpath(w.ckpt))
    mp.setattr(tool, "BACKUPS_ROOT", os.path.realpath(w.backups))


@pytest.fixture(scope="module")
def world(pinned, tmp_path_factory):
    w = build_world(tmp_path_factory.mktemp("world"), pinned)
    mp = pytest.MonkeyPatch()
    patch_world(mp, w)
    yield w
    mp.undo()


def argv_for(w, *extra, step="classify", scope_min_len=MIN_LEN):
    return ["--pin-root", str(PIN_ROOT), "--target-dir", str(w.ckpt), "--daemon-script", str(w.daemon),
            "--scope-min-len", str(scope_min_len), "--expect-wants", str(w.expect["wants"]),
            "--expect-protected", str(w.expect["protected"]), "--expect-scope", str(w.expect["scope"]),
            "--step", step, *extra]


def cli(argv, probes=None):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = tool.main(argv, probes=probes)
    lines = [ln for ln in out.getvalue().splitlines() if ln.startswith("{")]
    return rc, (json.loads(lines[-1]) if lines else None), err.getvalue()


def file_hashes(d):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(d).iterdir()) if p.is_file()}


@pytest.fixture(scope="module")
def phase1(world):
    """classify then rewrite (provisional approvals) on the COPY - the Phase-1 dry run, on synthetic files."""
    before = file_hashes(world.ckpt)
    rc1, r1, e1 = cli(argv_for(world))
    assert rc1 == 0, e1
    rc2, r2, e2 = cli(argv_for(world, "--run-dir", r1["run_dir"], "--provisional-approve-all", step="rewrite"))
    assert rc2 == 0, e2
    return types.SimpleNamespace(classify=r1, rewrite=r2, run_dir=Path(r1["run_dir"]), before=before)


@pytest.fixture(scope="module")
def analysis(world, pinned, phase1):
    return tool.analyze(pinned, str(phase1.run_dir / "copy"), scope_min_len=MIN_LEN, base_mod=tool.load_base_module(pinned))


def rec_of(A, name, world):
    return next(r for r in A["records"] if r["id"] == world.ids[name])


# ==================================================================================================
# P379 / P1 / plan 6.2: identity of the pinned function, import isolation, stamps (P5)
# ==================================================================================================

def test_p379_preamble_resolves_the_pin_worktree_copy(pinned):
    assert Path(sys.modules["cc_ng_organism"].__file__).resolve() == PIN_ROOT / "cc_ng_organism.py"
    assert pinned.record["p1"]["cc_ng_organism_sha256"] == "8ad0f69ed4a7849c98e97e0f3970bbe4a36c727ef880455e8ddecec14de5a8e2"
    assert pinned.record["p1"]["cc_ng_organism_blob"] == "a3aa8a0ddb6a89fe468a9a20beadc5e381624cab"
    assert pinned.record["p1"]["test_file_sha256"] == "04b1a494e4a5762b1e3f97d07e12c0937a50bf5baac09e5208b2bd9f82a46c53"
    assert pinned.record["p1"]["tree_head"] == "ae798b94cb14740d200fc3f4fd8d36eef8b86c6a"
    assert pinned.record["p1"]["frozen_branch_head"] == "c7921b8436fb174c3f70fcf02827f16bb16deff0"


def test_p1_a_wrong_function_sha_refuses(monkeypatch):
    monkeypatch.setitem(tool.PIN, "cc_ng_organism_sha256", "0" * 64)
    with pytest.raises(tool.Refusal, match="cc_ng_organism_sha256"):
        tool.load_pinned(str(PIN_ROOT))


def test_p1_a_wrong_blob_refuses(monkeypatch):
    monkeypatch.setitem(tool.PIN, "cc_ng_organism_blob", "0" * 40)
    with pytest.raises(tool.Refusal, match="cc_ng_organism_blob"):
        tool.load_pinned(str(PIN_ROOT))


def test_p1_a_wrong_test_file_sha_refuses(monkeypatch):
    monkeypatch.setitem(tool.PIN, "test_file_sha256", "f" * 64)
    with pytest.raises(tool.Refusal, match="test_file_sha256"):
        tool.load_pinned(str(PIN_ROOT))


def test_p1_a_non_pinned_file_refuses(tmp_path):
    fake = tmp_path / "fake-pin"
    (fake / "tests").mkdir(parents=True)
    shutil.copy(PIN_ROOT / "cc_ng_organism.py", fake / "cc_ng_organism.py")
    with open(fake / "cc_ng_organism.py", "a") as f:
        f.write("\n# a one-line difference\n")
    shutil.copy(PIN_ROOT / "tests" / "test_cc_want_legitimacy_810.py", fake / "tests" / "test_cc_want_legitimacy_810.py")
    with pytest.raises(tool.Refusal):
        tool.load_pinned(str(fake))


def test_import_isolation_fails_when_a_preloaded_module_is_outside_the_pin(monkeypatch):
    fake = types.ModuleType("cc_ng_organism")
    fake.__file__ = "/nonexistent/elsewhere/cc_ng_organism.py"
    monkeypatch.setitem(sys.modules, "cc_ng_organism", fake)
    with pytest.raises(tool.Refusal, match="outside the pin tree"):
        tool.load_pinned(str(PIN_ROOT))


def test_import_isolation_fails_when_cc_ng_organism_is_the_primary_checkout(monkeypatch):
    real_import = tool.importlib.import_module

    def fake_import(name, *a, **k):
        if name == "cc_ng_organism":
            return types.SimpleNamespace(__file__=str(Path(ORIG.PRIMARY) / "cc_ng_organism.py"))
        return real_import(name, *a, **k)
    monkeypatch.setattr(tool.importlib, "import_module", fake_import)
    with pytest.raises(tool.Refusal, match="PRIMARY checkout"):
        tool.load_pinned(str(PIN_ROOT))


def _git(cwd, *a):
    subprocess.run(["git", "-c", "user.email=t@example.invalid", "-c", "user.name=t", "-C", str(cwd), *a],
                   check=True, capture_output=True)


def test_is_primary_checkout_path_tells_a_primary_from_a_linked_worktree(tmp_path):
    prim = tmp_path / "prim"
    prim.mkdir()
    _git(prim, "init", "-q")
    (prim / "f.txt").write_text("x")
    _git(prim, "add", "f.txt")
    _git(prim, "commit", "-q", "-m", "c")
    _git(prim, "worktree", "add", "-q", str(tmp_path / "linked"), "-b", "b2")
    assert tool.is_primary_checkout_path(str(prim)) is True
    assert tool.is_primary_checkout_path(str(tmp_path / "linked")) is False
    assert tool.is_primary_checkout_path(str(tmp_path)) is False          # in no repository at all


def test_p1_refuses_a_pin_root_that_is_a_primary_checkout(tmp_path):
    prim = tmp_path / "prim"
    (prim / "tests").mkdir(parents=True)
    shutil.copy(PIN_ROOT / "cc_ng_organism.py", prim / "cc_ng_organism.py")
    shutil.copy(PIN_ROOT / "tests" / "test_cc_want_legitimacy_810.py", prim / "tests" / "test_cc_want_legitimacy_810.py")
    _git(prim, "init", "-q")
    _git(prim, "add", "-A")
    _git(prim, "commit", "-q", "-m", "c")
    with pytest.raises(tool.Refusal, match="primary"):
        tool.load_pinned(str(prim))


def test_the_stamp_is_the_frozen_pin_tuple():
    assert tool.pin_stamp() == {
        "cc_ng_organism_sha256": "8ad0f69ed4a7849c98e97e0f3970bbe4a36c727ef880455e8ddecec14de5a8e2",
        "cc_ng_organism_blob": "a3aa8a0ddb6a89fe468a9a20beadc5e381624cab",
        "branch_head": "c7921b8436fb174c3f70fcf02827f16bb16deff0",
        "test_file_sha256": "04b1a494e4a5762b1e3f97d07e12c0937a50bf5baac09e5208b2bd9f82a46c53",
        "scrub_version": tool.SCRUB_VERSION}


def test_every_report_artifact_is_stamped_with_the_pin_tuple(phase1):
    reports = sorted((phase1.run_dir / "reports").glob("*.json"))
    names = {p.stem for p in reports}
    for need in ("outcome-table", "histograms", "marker-bearing-minted", "residual-classes", "repair-list", "scope-ids",
                 "pre-node-report", "id-map"):
        assert need in names
    for p in reports + [phase1.run_dir / "run-record.json", phase1.run_dir / "copy-hashes.json"]:
        assert json.loads(p.read_text())["pin_stamp"] == tool.pin_stamp(), p.name


def test_p5_refuses_an_artifact_whose_stamp_is_not_the_frozen_pin(phase1, tmp_path):
    src = json.loads((phase1.run_dir / "reports" / "repair-list.json").read_text())
    ok = tool.load_artifact(str(phase1.run_dir / "reports" / "repair-list.json"))
    assert ok["pin_stamp"] == tool.pin_stamp()
    for mutate in (lambda o: o["pin_stamp"].__setitem__("branch_head", "0" * 40),
                   lambda o: o["pin_stamp"].__setitem__("cc_ng_organism_sha256", "0" * 64),
                   lambda o: o["pin_stamp"].__setitem__("test_file_sha256", "0" * 64),
                   lambda o: o["pin_stamp"].__setitem__("scrub_version", "scrub-0"),
                   lambda o: o.pop("pin_stamp")):
        bad = copy.deepcopy(src)
        mutate(bad)
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(bad))
        with pytest.raises(tool.Refusal, match="P5"):
            tool.load_artifact(str(p))


def test_the_report_text_guard_rejects_text_like_keys_and_long_strings():
    with pytest.raises(tool.Stop):
        tool._assert_text_free({"want_text": "x"})
    with pytest.raises(tool.Stop):
        tool._assert_text_free({"a": ["y" * 300]})
    tool._assert_text_free({"ids": ["cc:want::0123456789abcdef"], "n": 3})


# ==================================================================================================
# P7 / plan 6.3: the target guard and the write guard (hard refusals, no override)
# ==================================================================================================

def test_the_recorded_constants_are_the_documented_ones():
    assert ORIG.SYL == os.path.realpath(os.path.expanduser("~/NeuroGraph/data/checkpoints"))
    assert ORIG.PRIMARY == os.path.realpath(os.path.expanduser("~/NeuroGraph"))
    assert ORIG.RECORDED == os.path.realpath(os.path.expanduser("~/.claude/plugins/neurograph/checkpoints"))
    assert ORIG.BACKUPS == os.path.realpath(os.path.expanduser("~/backups"))
    assert tool.RUN_DIR_PREFIX == "z12-want-text-repair-"


def test_the_target_dir_is_a_required_argument_with_no_default(world):
    with pytest.raises(SystemExit):
        tool.build_parser().parse_args(["--pin-root", "x", "--daemon-script", "y", "--scope-min-len", "600"])
    with pytest.raises(tool.Refusal, match="REQUIRED"):
        tool.guard_target("", str(world.daemon))
    a = tool.build_parser().parse_args(["--pin-root", "x", "--target-dir", "y", "--daemon-script", "z", "--scope-min-len", "600"])
    assert a.target_dir == "y" and a.apply is False and a.josh_go is None


def test_the_target_guard_accepts_only_the_recorded_directory_and_records_its_realpath(world):
    info = tool.guard_target(str(world.ckpt), str(world.daemon))
    assert info["target_realpath"] == os.path.realpath(world.ckpt)
    assert info["daemon_script_checkpoint_dir"] == os.path.realpath(world.ckpt)


def test_hard_refusal_a_target_under_syls_checkpoints(world, monkeypatch, tmp_path):
    syl = tmp_path / "syl" / "data" / "checkpoints"
    (syl / "sub").mkdir(parents=True)
    monkeypatch.setattr(tool, "SYL_CHECKPOINTS", os.path.realpath(syl))
    monkeypatch.setattr(tool, "RECORDED_CC_CHECKPOINT_DIR", os.path.realpath(syl / "sub"))      # even if 'recorded'
    with pytest.raises(tool.Refusal, match="Syl"):
        tool.guard_target(str(syl / "sub"), str(world.daemon))


def test_hard_refusal_a_target_that_is_not_the_recorded_cc_directory(world, tmp_path):
    other = tmp_path / "other-ckpt"
    other.mkdir()
    with pytest.raises(tool.Refusal, match="not the recorded"):
        tool.guard_target(str(other), str(world.daemon))


def test_hard_refusal_a_target_inside_a_primary_checkout(world, monkeypatch, tmp_path):
    prim = tmp_path / "prim"
    (prim / "checkpoints").mkdir(parents=True)
    _git(prim, "init", "-q")
    monkeypatch.setattr(tool, "RECORDED_CC_CHECKPOINT_DIR", os.path.realpath(prim / "checkpoints"))
    with pytest.raises(tool.Refusal, match="primary checkout"):
        tool.guard_target(str(prim / "checkpoints"), str(world.daemon))


def test_the_daemon_script_cross_check_must_agree(world, tmp_path):
    bad = tmp_path / "daemon-bad.py"
    bad.write_text("CC_NG_WORKSPACE = os.path.expanduser('%s')\nCHECKPOINT_DIR = os.path.join(CC_NG_WORKSPACE, 'elsewhere')\n" % world.ws)
    with pytest.raises(tool.Refusal, match="disagrees"):
        tool.guard_target(str(world.ckpt), str(bad))
    odd = tmp_path / "daemon-odd.py"
    odd.write_text("CHECKPOINT_DIR = '/somewhere'\n")
    with pytest.raises(tool.Refusal, match="recorded shape"):
        tool.guard_target(str(world.ckpt), str(odd))


def test_phase_1_may_write_only_under_the_backups_run_directory(world, tmp_path):
    for bad in (tmp_path / "elsewhere.json", world.backups / "not-a-run-dir" / "x.json", world.ckpt / "x.json",
                world.backups.parent / "x.json"):
        with pytest.raises(tool.Refusal, match="write refused"):
            tool.out_write_bytes(str(bad), b"x")
    good = world.backups / (tool.RUN_DIR_PREFIX + "unit") / "ok.json"
    assert tool.out_write_bytes(str(good), b"x") == os.path.realpath(good)


# ==================================================================================================
# classification (plan 4.2): each outcome class, id-follows-text, collisions, no length test
# ==================================================================================================

def test_the_classifier_outcomes_over_the_synthetic_corpus(analysis, world):
    want = {"SEP1": ("SEPARATE", "candidate"), "SEP2": ("SEPARATE", "candidate"), "BIG": ("SEPARATE", "candidate"),
            "GEN": ("GENUINE", "unchanged"), "GENM": ("GENUINE", "unchanged"), "NONE": ("NONE", "unchanged"),
            "OVL": ("OVERLAP", "unchanged"), "ANOM": ("ANOMALY", "unchanged"), "ANCH": ("ANCHOR_FAILED", "unchanged"),
            "MISS": ("SOURCE_MISSING", "unchanged"), "IDMM": ("ID_MISMATCH", "unchanged"),
            "COLLA": ("SEPARATE", "collision_dropped"), "COLLB": ("SEPARATE", "collision_dropped"),
            "COLLE": ("SEPARATE", "collision_dropped"), "CHAINA": ("SEPARATE", "collision_dropped"),
            "CHAINB": ("NONE", "unchanged")}
    for name, (outcome, disp) in want.items():
        r = rec_of(analysis, name, world)
        assert (r["outcome"], r["disposition"]) == (outcome, disp), (name, r["outcome"], r["disposition"], r["detail"])
    assert sorted(r["id"] for r in analysis["records"]) == world.scope


def test_the_outcome_table_accounts_for_every_id_of_s(analysis, world):
    ot = tool.outcome_table(analysis["records"])
    assert ot["scope_size"] == ot["accounting_sum"] == len(world.scope)
    assert ot["by_outcome"]["SEPARATE"] == 7 and ot["by_outcome"]["GENUINE"] == 2
    assert ot["by_disposition"] == {"candidate": 3, "collision_dropped": 4, "unchanged": len(world.scope) - 7}


def test_separate_takes_the_nested_pair_verbatim_and_the_id_follows_the_text(analysis, world, pinned):
    org = pinned.org
    for name, X in (("SEP1", X1), ("SEP2", X2), ("BIG", X5)):
        r = rec_of(analysis, name, world)
        assert r["_x"] == X and r["new_id"] == org.want_id_for_text(X) == world.new[name]
        assert X in r["_t"] and r["_t"].endswith(X)
        assert r["removed_prefix_len"] > 0 and r["new_len"] == len(X)


def test_no_length_test_a_5000_character_x_is_neither_truncated_nor_rejected(analysis, world):
    r = rec_of(analysis, "BIG", world)
    assert r["new_len"] > 5000 and r["disposition"] == "candidate" and len(r["_x"]) == r["new_len"]


def test_a_genuine_want_may_contain_a_masked_mention_pair(analysis, world):
    r = rec_of(analysis, "GENM", world)
    assert r["outcome"] == "GENUINE" and r["flags"]["t_has_marker"] is True and "in_url" in r["flags"]["region_fired"]


def test_class_a_b_c_letters(analysis, world):
    assert rec_of(analysis, "SEP2", world)["class"] == "A"          # backtick-led + a marker
    assert rec_of(analysis, "SEP1", world)["class"] == "C"          # marker, not backtick-led
    assert rec_of(analysis, "GEN", world)["class"] == "D"
    assert tool.text_class(BT + "x " + "y" * 3) == "B"


def test_the_overlap_anomaly_lists_the_overlapping_minted_want(analysis, world):
    r = rec_of(analysis, "OVL", world)
    assert r["outcome"] == "OVERLAP" and r["detail"] == "overlap_other_pair" and r["overlap_wants"]
    assert r["flags"]["overlap_marker_bearing"] is True


def test_the_collision_rule_drops_and_lists_both_parties_never_merges(analysis, world):
    for name in ("COLLA", "COLLB"):
        r = rec_of(analysis, name, world)
        assert r["disposition"] == "collision_dropped" and r["detail"] == "collision:new_id_produced_by_two_repairs"
    for name in ("COLLE", "CHAINA"):
        assert rec_of(analysis, name, world)["detail"] == "collision:new_id_exists_in_graph"
    dropped = {r["id"] for r in analysis["dropped"]}
    assert {world.ids[n] for n in ("COLLA", "COLLB", "COLLE", "CHAINA")} == dropped
    m = tool.build_mapping(analysis["records"])
    assert set(m) == {world.ids[n] for n in ("SEP1", "SEP2", "BIG")}


def test_assert_failed_leaves_the_node_unchanged_and_listed(analysis, world, pinned):
    proxy = types.SimpleNamespace(**{k: getattr(pinned.org, k) for k in ("parse_wants", "WANT_OPEN", "WANT_CLOSE")})
    proxy.want_id_for_text = lambda text: "cc:want::0000000000000000" if text == X1 else pinned.org.want_id_for_text(text)
    cl = tool.Classifier(proxy, analysis["nodes_meta"], analysis["content"])
    r = cl.classify_node(world.ids["SEP1"])
    assert r["outcome"] == "SEPARATE" and r["disposition"] == "assert_failed" and r["detail"] == "id_equals_function"
    assert tool.build_mapping([r]) == {}


def test_the_clause_and_the_rim_are_never_in_s_and_the_deny_check_holds(analysis, world):
    ids = set(tool.CHOICE_CLAUSE_IDS) | {tool.CONSTITUTIONAL_ID}
    assert not ids & set(analysis["scope"])
    assert tool.deny_check(analysis["scope"], tool.build_mapping(analysis["records"]), [])["clean"] is True
    for kwargs in ({"scope_ids": [tool.CHOICE_CLAUSE_IDS[0]], "mapping": {}},
                   {"scope_ids": [], "mapping": {tool.CHOICE_CLAUSE_IDS[1]: "cc:want::aaaaaaaaaaaaaaaa"}},
                   {"scope_ids": [], "mapping": {"cc:want::bbbbbbbbbbbbbbbb": tool.CHOICE_CLAUSE_IDS[0]}},
                   {"scope_ids": [], "mapping": {}, "approval_ids": [tool.CONSTITUTIONAL_ID]}):
        with pytest.raises(tool.Stop, match="deny-check"):
            tool.deny_check(kwargs["scope_ids"], kwargs["mapping"], kwargs.get("approval_ids", ()))


def test_want_id_for_text_equals_the_production_mint(pinned):
    """The tool has ONE id function (the pinned one); it equals what the real surface_wants mints (plan 3.1)."""
    org = pinned.org
    texts = ["plain ascii want", "multi-byte éè 日本語 want", "edge whitespace inside  x", "y" * 5000]
    g = tool._FakeGraph()
    content = {}
    for i, t in enumerate(texts):
        g.nodes["c%d" % i] = tool._FakeNode({"creation_mode": "conversational"})
        content["c%d" % i] = "[WANT]" + t + "[/WANT]"
    org.surface_wants(g, tool._FakeVDB(content))
    minted = {n.metadata["want_text"]: nid for nid, n in g.nodes.items() if n.metadata.get("kind") == "want"}
    for t in texts:
        assert minted[t.strip()] == org.want_id_for_text(t.strip())
    src = TOOL_PATH.read_text()
    assert "def want_id_for_text" not in src and "def parse_wants" not in src        # V14: no copy anywhere


def test_is_protected_mirrors_the_canonical_predicate(pinned):
    g = pinned.nf.Graph()
    for i, md in enumerate(({"constitutional": True}, {"provenance": "cc_authored"}, {"provenance": "syl_authored"},
                            {"provenance": "cc_emergent"}, {}, {"constitutional": False, "provenance": None})):
        g.create_node(node_id="n%d" % i, metadata=md)
        assert tool.is_protected(md) == g._is_identity_protected("n%d" % i)


# ==================================================================================================
# the reports (plan 4.4): histograms twice, marker-bearing list, residuals, PRE-node line, scope
# ==================================================================================================

def test_histograms_cover_all_eleven_reasons_twice(phase1):
    h = json.loads((phase1.run_dir / "reports" / "histograms.json").read_text())
    for part in ("s_sources", "all_conversational_marker_nodes"):
        assert tuple(sorted(h[part]["by_reason"])) == tuple(sorted(tool.REASONS)) and len(tool.REASONS) == 11
    assert h["s_sources"]["nodes"] <= h["all_conversational_marker_nodes"]["nodes"]
    assert h["all_conversational_marker_nodes"]["by_reason"]["in_url"] >= 2            # the GENM source
    assert h["all_conversational_marker_nodes"]["by_reason"]["in_code_span"] >= 2       # NONE / ANOM sources


def test_the_marker_bearing_minted_list_holds_the_nested_mention_pair(phase1, world):
    mb = json.loads((phase1.run_dir / "reports" / "marker-bearing-minted.json").read_text())
    srcs = {e["source_node"] for e in mb["entries"]}
    assert "cc:conv::GENM" in srcs and all(e["to_frozen_list_review"] for e in mb["entries"])
    assert all("shape_flags" in e for e in mb["entries"])


def test_the_residual_classes_are_listed_with_counts_and_an_unclassified_bucket(phase1):
    r = json.loads((phase1.run_dir / "reports" / "residual-classes.json").read_text())
    assert "base_minted_pinned_drops_total" in r and "by_pinned_reason" in r
    assert sum(g["count"] for g in r["by_pinned_reason"].values()) == r["base_minted_pinned_drops_total"]


def test_the_scope_ids_are_enumerated_and_the_rule_is_only_a_cross_check(phase1, world):
    sc = json.loads((phase1.run_dir / "reports" / "scope-ids.json").read_text())
    assert sc["ids"] == world.scope and sc["count"] == len(world.scope) == world.expect["scope"]
    assert sc["rule_derived_equals_list"] is True and sc["scope_min_len_cross_check"] == MIN_LEN


def test_the_pre_node_report_line_names_who_is_a_pre_node_of_a_synapse_to_a_mapped_id(phase1, world):
    r = json.loads((phase1.run_dir / "reports" / "pre-node-report.json").read_text())["pre_node_report"]
    assert r[world.cc1]["is_pre_node_of_synapse_to_mapped_id"] is True and r[world.cc1]["count"] == 1
    assert r[world.cc2]["is_pre_node_of_synapse_to_mapped_id"] is False
    assert r[world.rim]["is_pre_node_of_synapse_to_mapped_id"] is True and r[world.rim]["count"] == 1


def test_no_pushed_class_report_carries_want_text_and_excerpts_are_off_repo_0600(phase1, world):
    needles = [X1, X2, XC, "real intent", prose("sepa", 3), "synthetic choice clause"]
    for p in list(phase1.run_dir.rglob("*.json")):
        text = p.read_text()
        assert not any(n in text for n in needles), p.name
    review = sorted((phase1.run_dir / "review").glob("*.md"))
    assert len(review) == 2
    for p in review:
        assert stat.S_IMODE(p.stat().st_mode) == 0o600
    assert X1 in (phase1.run_dir / "review" / review[0].name).read_text() + review[1].read_text()


def test_excerpt_anchors_are_named_and_all_inside_excerpt_sha256(analysis, world, pinned):
    r = rec_of(analysis, "SEP1", world)
    a = r["_anchors"]
    assert set(k for k in a if not k.startswith("_")) == {"outer_opener", "inner_opener", "closer", "x_full", "removed_prefix_len", "scrub_version"}
    base = tool.excerpt_sha256(a)
    assert base == r["excerpt_sha256"]
    for k, v in (("outer_opener", "x"), ("inner_opener", "x"), ("closer", "x"), ("x_full", "x"), ("removed_prefix_len", 1), ("scrub_version", "s0")):
        assert tool.excerpt_sha256(dict(a, **{k: v})) != base
    assert a["removed_prefix_len"] == r["removed_prefix_len"] > 0


def test_the_scrub_replaces_credentials_by_rule_name():
    s, n = tool.scrub("token=abcdefghijk12345 and sk-abcdefghijklmnop1234 and AKIAABCDEFGHIJKLMNOP end")
    assert n == 3 and "abcdefghijk12345" not in s and "[REDACTED:credential_kv]" in s and "[REDACTED:sk_token]" in s


# ==================================================================================================
# the writer, the census, the verifier V1-V19 (positive run on the Phase-1 copy, then negative cases)
# ==================================================================================================

@pytest.fixture(scope="module")
def built(world, pinned, phase1, analysis):
    """build_outputs once (real approvals-shaped body, all candidates approved) for the verifier negatives."""
    run_dir, utc = tool.new_run_dir()
    A = analysis
    rl = tool.artifact_sha256(tool.repair_list_obj(A))
    sc = tool.artifact_sha256(tool.scope_ids_obj(A, world.expect["scope"]))
    approvals = tool.stamped(tool.approvals_body_for(A["records"], rl, sc, "EXEC-SYNTHETIC-PACKET"))
    in_dir = str(phase1.run_dir / "copy")
    in_hashes = {n: tool.sha256_file(os.path.join(in_dir, n)) for n in tool.SIX_FILES}
    ctx = tool.build_outputs(pinned, A, approvals, in_dir=in_dir, out_dir=os.path.join(run_dir, "out"), run_dir=run_dir,
                             utc=utc, in_hashes=in_hashes)
    return types.SimpleNamespace(ctx=ctx, run_dir=run_dir, A=A, approvals=approvals, in_dir=in_dir, utc=utc)


def _verify(built, pinned, world, **kw):
    return tool.run_verifier(pinned, built.A, built.ctx, world.expect, **kw)


def _check(V, name):
    return next(r for r in V.results if r["check"] == name)


def test_the_phase1_dry_run_passes_v1_to_v19_and_the_copy_is_the_only_thing_written(phase1, world):
    rep = next((phase1.run_dir / "reports").glob("verify-report-*.json"))
    d = json.loads(rep.read_text())
    assert d["failed"] == [] and [c["check"] for c in d["checks"]] == ["V%d" % i for i in range(1, 20)]
    assert all(c["ok"] for c in d["checks"])
    assert d["approvals"]["provisional"] is True and len(d["written_ids"]) == 3
    assert file_hashes(world.ckpt) == phase1.before                       # the target was never written


def test_the_verifier_positive_run_has_no_failures(built, pinned, world):
    V = _verify(built, pinned, world)
    assert V.failed() == [] and len(V.results) == 19
    assert _check(V, "V16")["detail"]["rim_incident_with_mapped_want"] == _check(V, "V16")["detail"]["rim_changed"] >= 1


def test_the_rewrite_repoints_every_reference_site_and_carries_everything_else(built, world, pinned):
    ctx, m = built.ctx, built.ctx["plan"]["mapping"]
    assert set(m) == {world.ids[n] for n in ("SEP1", "SEP2", "BIG")}
    g2 = ctx["g2"]
    assert set(g2.nodes) == {m.get(n, n) for n in built.A["nodes_meta"]}
    for name in ("SEP1", "SEP2", "BIG"):
        nid = world.new[name]
        md = g2.nodes[nid].metadata
        assert md["want_text"] == {"SEP1": X1, "SEP2": X2, "BIG": X5}[name]
        assert pinned.org.want_id_for_text(md["want_text"]) == nid                      # id-follows-text
        assert md["provenance"] == "cc_authored" and md["want_state"] == "open" and md["kind"] == "want"
        assert md["poincare_dir"] == b"\x00\x01" + name.encode()                        # poincare_dir carried byte-identical
        assert world.ids[name] not in g2.nodes
    raw2 = Path(ctx["out_main"]).read_bytes()
    new1, old1 = world.new["SEP1"], world.ids["SEP1"]
    cap = {}
    for sec, ek, ks, vs, ve in tool.iter_sections(raw2):
        if ek is None and sec in ("delay_buffer", "recent_spikes", "he_output_candidates", "novel_sequence_log",
                                  "prediction_outcomes", "active_predictions", "he_active_predictions"):
            cap[sec] = tool.decode(raw2[vs:ve])
    assert cap["delay_buffer"][str(10 ** 6)][0][0] == new1 and new1 in cap["recent_spikes"] and old1 not in cap["recent_spikes"]
    assert cap["active_predictions"]["p1"]["source_node_id"] == new1
    assert cap["prediction_outcomes"][0]["actual_firing_nodes"][0] == new1
    assert cap["novel_sequence_log"][0]["source"] == new1
    assert world.new["SEP2"] in cap["he_output_candidates"][world.he]
    assert world.new["SEP2"] in cap["he_active_predictions"]["hp1"]["predicted_targets"]
    he = g2.hyperedges[world.he]
    assert new1 in he.member_nodes and world.new["SEP2"] in he.output_targets and new1 in he.member_weights
    assert g2.nodes[world.cc1].pred_weights == {new1: 0.7, world.ids["GEN"]: 0.1}           # key remapped, values equal
    assert g2.nodes[world.rim].pred_weights == {new1: 0.4}


def test_choice_clause_wants_and_the_rim_are_byte_equal_apart_from_the_key_remap(built, world):
    ctx = built.ctx
    raw1, raw2 = built.ctx["raw"], Path(ctx["out_main"]).read_bytes()
    m = ctx["plan"]["mapping"]
    seen = {}
    for a, b in zip(tool.iter_sections(raw1), tool.iter_sections(raw2)):
        if a[0] == "nodes" and a[1] in (world.cc1, world.cc2, world.rim, world.short1):
            da, db = tool.decode(raw1[a[3]:a[4]]), tool.decode(raw2[b[3]:b[4]])
            diffs, subs = tool.diff_modulo(da, db, m)
            assert diffs == [] and set(subs) <= {"pred_weights"}
            seen[a[1]] = (raw1[a[3]:a[4]] == raw2[b[3]:b[4]], int(sum(subs.values())))
    assert seen[world.cc2] == (True, 0)                       # no moved id in its pred_weights: BYTE-identical
    assert seen[world.cc1][0] is False and seen[world.cc1][1] == 1
    assert seen[world.rim][0] is False and seen[world.rim][1] == 1


def test_the_rim_synapses_are_repointed_at_the_want_side_only(built, world):
    raw1, raw2 = built.ctx["raw"], Path(built.ctx["out_main"]).read_bytes()
    m = built.ctx["plan"]["mapping"]
    changed = []
    for a, b in zip(tool.iter_sections(raw1), tool.iter_sections(raw2)):
        if a[0] == "synapses" and isinstance(a[1], str) and a[1] != tool._HDR:
            da, db = tool.decode(raw1[a[3]:a[4]]), tool.decode(raw2[b[3]:b[4]])
            if world.rim in (da["pre_node_id"], da["post_node_id"]):
                assert (db["pre_node_id"] == world.rim) == (da["pre_node_id"] == world.rim)
                for f in da:
                    if f not in ("pre_node_id", "post_node_id"):
                        assert da[f] == db[f]                  # weights, ages, counters byte-identical
                if da != db:
                    changed.append(a[1])
    assert len(changed) == 2                                   # rim->SEP1 and SEP2->rim


def test_the_mapping_and_its_inverse_are_written_hash_verified_and_inverse_composes(built):
    art = built.ctx["plan"]["artifacts"]
    m, inv = built.ctx["plan"]["mapping"], built.ctx["plan"]["inverse"]
    assert all(inv[m[o]] == o for o in m) and len(inv) == len(m) == 3
    for name in ("id-map", "id-map-inverse", "post-apply-receipt"):
        path, sha = art[name]
        assert tool.sha256_file(path) == sha and tool.load_artifact(path, sha)["pin_stamp"] == tool.pin_stamp()
    fwd = tool.load_artifact(art["id-map"][0])
    invd = tool.load_artifact(art["id-map-inverse"][0])
    assert sorted([n, o] for o, n in fwd["pairs"]) == sorted(invd["pairs"])


def test_the_sidecar_is_rekeyed_and_every_other_file_is_sha256_equal(built, world):
    ctx = built.ctx
    m = ctx["plan"]["mapping"]
    sa, sb = json.loads(ctx["sidecar_in"]), json.loads(ctx["sidecar_out"])
    assert list(sb["entries"]) == [m.get(k, k) for k in sa["entries"]]
    assert all(sa["entries"][k] == sb["entries"][m.get(k, k)] for k in sa["entries"])
    assert {k: v for k, v in sa.items() if k != "entries"} == {k: v for k, v in sb.items() if k != "entries"}
    for n in (tool.VECTORS_NAME, tool.GUARD_NAME, tool.MANIFEST_NAME, tool.COMMONS_NAME):
        assert ctx["out_hashes"][n] == ctx["in_hashes"][n]


def test_an_empty_mapping_makes_a_byte_identical_output_and_the_rewrite_is_idempotent(built, world):
    out = os.path.join(built.run_dir, "idem.msgpack")
    st = tool.rewrite_main(built.ctx["raw"], out, {}, {}, {})
    assert Path(out).read_bytes() == built.ctx["raw"] and st["substitutions_total"] == 0 and st["reencoded_entries"] == {}
    out2 = os.path.join(built.run_dir, "idem2.msgpack")
    tool.rewrite_main(Path(built.ctx["out_main"]).read_bytes(), out2, {}, {}, {})
    assert Path(out2).read_bytes() == Path(built.ctx["out_main"]).read_bytes()


def test_the_census_counts_the_same_substitutions_three_ways(built):
    W = tool.census_msgpack_file(built.in_dir + "/main.msgpack", list(built.ctx["plan"]["mapping"]))
    assert int(sum(W.values())) == built.ctx["writer_stats"]["substitutions_total"] > 3
    assert set(built.ctx["writer_stats"]["substitutions"]) >= {"S1_nodes", "S2_pred_weights", "S3_synapse_endpoints", "S4_expected_target",
                                                             "S5_hyperedges", "S6_active_predictions", "S7_prediction_outcomes",
                                                             "S8_he_active_predictions", "S9_he_output_candidates",
                                                             "S10_novel_sequence_log", "S11_delay_buffer", "S12_recent_spikes"}
    assert built.ctx["sidecar_stats"]["substitutions"] == 2                       # SEP1 and SEP2 have sidecar entries


def test_the_census_stops_on_an_old_id_outside_the_site_table(built, world):
    raw = built.ctx["raw"]
    old = world.ids["SEP1"]
    tampered = _splice(raw, "nodes", "n2", lambda n: dict(n, metadata={"note": old}))
    with pytest.raises(tool.Stop, match="outside the site table"):
        tool.rewrite_main(tampered, os.path.join(built.run_dir, "x.msgpack"), built.ctx["plan"]["mapping"],
                          built.ctx["plan"]["old_text"], built.ctx["plan"]["new_text"])
    top = _splice_top(raw, "reward_history", lambda v: [old])
    with pytest.raises(tool.Stop, match="not in the site table"):
        tool.rewrite_main(top, os.path.join(built.run_dir, "x2.msgpack"), built.ctx["plan"]["mapping"],
                          built.ctx["plan"]["old_text"], built.ctx["plan"]["new_text"])


def test_an_old_id_in_a_file_outside_s1_s13_stops_before_any_write(world, pinned, built, tmp_path):
    d = tmp_path / "copy2"
    shutil.copytree(built.in_dir, d)
    (d / "commons.msgpack").write_bytes(msgpack.packb({"ref": world.ids["SEP1"]}, use_bin_type=True))
    hashes = {n: tool.sha256_file(str(d / n)) for n in tool.SIX_FILES}
    run_dir, utc = tool.new_run_dir()
    with pytest.raises(tool.Stop, match="outside S1-S13"):
        tool.build_outputs(pinned, built.A, built.approvals, in_dir=str(d), out_dir=os.path.join(run_dir, "out"),
                           run_dir=run_dir, utc=utc, in_hashes=hashes)
    assert not (Path(run_dir) / "out").exists()


def test_v13_an_entry_that_does_not_round_trip_stops_the_whole_run(built, world):
    raw = built.ctx["raw"]
    nid = world.ids["SEP1"]
    vbytes = None
    for sec, ek, ks, vs, ve in tool.iter_sections(raw):
        if sec == "nodes" and ek == nid:
            vbytes = raw[vs:ve]
            break
    assert b"\xadcreation_time\x00" in vbytes
    bad = vbytes.replace(b"\xadcreation_time\x00", b"\xadcreation_time\xcd\x00\x00")       # a non-minimal (but valid) int
    tampered = raw.replace(vbytes, bad)
    assert tampered != raw and tool.decode(bad) == tool.decode(vbytes)
    with pytest.raises(tool.Stop, match="V13"):
        tool.rewrite_main(tampered, os.path.join(built.run_dir, "v13.msgpack"), built.ctx["plan"]["mapping"],
                          built.ctx["plan"]["old_text"], built.ctx["plan"]["new_text"])


def test_v13_a_sidecar_that_does_not_round_trip_stops():
    with pytest.raises(tool.Stop, match="V13"):
        tool.rewrite_sidecar('{"entries": {},  "version": "1.0"}', {})


def _splice(raw, section, key, fn):
    for sec, ek, ks, vs, ve in tool.iter_sections(raw):
        if sec == section and ek == key:
            return raw[:vs] + tool.encode(fn(tool.decode(raw[vs:ve]))) + raw[ve:]
    raise KeyError((section, key))


def _splice_top(raw, key, fn):
    for sec, ek, ks, vs, ve in tool.iter_sections(raw):
        if sec == key and ek is None:
            return raw[:vs] + tool.encode(fn(tool.decode(raw[vs:ve]))) + raw[ve:]
    raise KeyError(key)


def _tampered_verify(built, pinned, world, tampered_raw, **kw):
    p = os.path.join(built.run_dir, "tamper-%d.msgpack" % abs(hash(tampered_raw)))
    Path(p).write_bytes(tampered_raw)
    return _verify(built, pinned, world, out_main=p, out_graph=kw.pop("out_graph", "ctx"), **kw)


def _out(built):
    return Path(built.ctx["out_main"]).read_bytes()


def test_v1_fails_when_a_want_stops_being_a_want(built, pinned, world):
    V = _tampered_verify(built, pinned, world, _splice(_out(built), "nodes", world.ids["GEN"],
                         lambda n: dict(n, metadata=dict(n["metadata"], kind="note"))))
    assert "V1" in V.failed()


def test_v2_fails_when_the_id_does_not_follow_the_text(built, pinned, world):
    V = _tampered_verify(built, pinned, world, _splice(_out(built), "nodes", world.new["SEP1"],
                         lambda n: dict(n, metadata=dict(n["metadata"], want_text="a different text"))))
    assert "V2" in V.failed() and "V3" in V.failed()


def test_v3_fails_when_a_repaired_node_changes_any_other_field(built, pinned, world):
    V = _tampered_verify(built, pinned, world, _splice(_out(built), "nodes", world.new["SEP2"], lambda n: dict(n, voltage=9.9)))
    assert "V3" in V.failed()


def test_v3_fails_when_poincare_dir_bytes_change(built, pinned, world):
    V = _tampered_verify(built, pinned, world, _splice(_out(built), "nodes", world.new["SEP2"],
                         lambda n: dict(n, metadata=dict(n["metadata"], poincare_dir=b"changed"))))
    assert "V3" in V.failed()


def test_v4_fails_when_a_synapse_field_changes(built, pinned, world):
    sid = next(ek for sec, ek, *_ in tool.iter_sections(_out(built)) if sec == "synapses" and isinstance(ek, str) and ek != tool._HDR)
    V = _tampered_verify(built, pinned, world, _splice(_out(built), "synapses", sid, lambda s: dict(s, weight=s["weight"] + 1.0)))
    assert "V4" in V.failed()


def test_v5_fails_when_a_hyperedge_changes(built, pinned, world):
    V = _tampered_verify(built, pinned, world, _splice(_out(built), "hyperedges", world.he, lambda h: dict(h, activation_threshold=0.99)))
    assert "V5" in V.failed()


def test_v6_fails_when_an_unrelated_top_level_value_changes(built, pinned, world):
    V = _tampered_verify(built, pinned, world, _splice_top(_out(built), "timestep", lambda v: v + 1))
    assert "V6" in V.failed()


def test_v7_fails_when_a_non_want_node_changes(built, pinned, world):
    V = _tampered_verify(built, pinned, world, _splice(_out(built), "nodes", "n2", lambda n: dict(n, voltage=7.7)))
    assert "V7" in V.failed()


def test_v8_fails_when_the_sidecar_is_tampered(built, pinned, world):
    bad = json.loads(built.ctx["sidecar_out"])
    k = next(iter(bad["entries"]))
    bad["entries"][k]["voltage"] = 99.0
    V = _verify(built, pinned, world, sidecar_out=json.dumps(bad))
    assert "V8" in V.failed()


def test_v9_fails_when_x_is_not_a_verbatim_suffix(built, pinned, world):
    A2 = dict(built.A, records=[dict(r, _x=("not a suffix" if r["id"] == world.ids["SEP1"] else r.get("_x"))) for r in built.A["records"]])
    V = tool.run_verifier(pinned, A2, built.ctx, world.expect)
    assert "V9" in V.failed()


def test_v10_fails_when_an_old_id_is_left_in_the_output(built, pinned, world):
    V = _tampered_verify(built, pinned, world, _splice(_out(built), "nodes", "n2", lambda n: dict(n, metadata={"note": world.ids["SEP1"]})))
    assert "V10" in V.failed()


def test_v11_fails_without_a_canonical_restore_of_the_output(built, pinned, world):
    V = _verify(built, pinned, world, out_graph=None)
    assert "V11" in V.failed()


def test_v12_fails_when_the_id_is_not_rederived_by_the_pinned_function(built, pinned, world):
    A2 = dict(built.A, records=[dict(r, _x=("something else" if r["id"] == world.ids["SEP2"] else r.get("_x"))) for r in built.A["records"]])
    V = tool.run_verifier(pinned, A2, built.ctx, world.expect)
    assert "V12" in V.failed()


def test_v14_records_the_shared_function_proof(built, pinned, world):
    d = _check(_verify(built, pinned, world), "V14")
    assert d["ok"] and d["detail"]["p1"]["cc_ng_organism_sha256"] == tool.PIN["cc_ng_organism_sha256"]


def test_v15_fails_for_any_change_other_than_the_key_remap_of_moved_ids(built, pinned, world):
    for fn in (lambda n: dict(n, voltage=5.5),                                            # a plain field
               lambda n: dict(n, pred_weights={k: v + 0.01 for k, v in n["pred_weights"].items()}),    # a VALUE
               lambda n: dict(n, pred_weights=dict(n["pred_weights"], **{"cc:want::ffffffffffffffff": 0.1})),    # a key added
               lambda n: dict(n, pred_weights={}),                                       # keys removed
               lambda n: dict(n, metadata=dict(n["metadata"], want_text="edited"))):     # the text of an unrepaired want
        V = _tampered_verify(built, pinned, world, _splice(_out(built), "nodes", world.cc1, fn))
        assert "V15" in V.failed()
    V = _tampered_verify(built, pinned, world, _splice(_out(built), "nodes", world.cc2, lambda n: dict(n, voltage=5.5)))
    assert "V15" in V.failed()
    V = _tampered_verify(built, pinned, world, _splice(_out(built), "nodes", world.rim, lambda n: dict(n, pred_weights={world.new["SEP1"]: 0.41})))
    assert "V15" in V.failed()


def test_v16_fails_when_a_rim_synapse_field_changes(built, pinned, world):
    raw = _out(built)
    sid = None
    for sec, ek, ks, vs, ve in tool.iter_sections(raw):
        if sec == "synapses" and isinstance(ek, str) and ek != tool._HDR and world.rim in (tool.decode(raw[vs:ve])["pre_node_id"], tool.decode(raw[vs:ve])["post_node_id"]):
            sid = ek
            break
    V = _tampered_verify(built, pinned, world, _splice(raw, "synapses", sid, lambda s: dict(s, weight=s["weight"] + 0.5)))
    assert "V16" in V.failed()


def test_v17_fails_when_written_is_not_approved_and_passed(built, pinned, world):
    plan = dict(built.ctx["plan"], write_ids=built.ctx["plan"]["write_ids"][:-1])
    assert "V17" in tool.run_verifier(pinned, built.A, built.ctx, world.expect, plan=plan).failed()
    plan2 = dict(built.ctx["plan"], approved_ids=built.ctx["plan"]["approved_ids"][:-1])
    assert "V17" in tool.run_verifier(pinned, built.A, built.ctx, world.expect, plan=plan2).failed()


def test_v18_fails_when_a_rollback_artefact_is_missing_or_altered(built, pinned, world):
    art = dict(built.ctx["plan"]["artifacts"])
    art["id-map"] = (art["id-map"][0], "0" * 64)
    assert "V18" in tool.run_verifier(pinned, built.A, built.ctx, world.expect, plan=dict(built.ctx["plan"], artifacts=art)).failed()
    art2 = dict(built.ctx["plan"]["artifacts"])
    art2["id-map"] = (art2["id-map"][0] + ".missing", art2["id-map"][1])
    assert "V18" in tool.run_verifier(pinned, built.A, built.ctx, world.expect, plan=dict(built.ctx["plan"], artifacts=art2)).failed()


def test_v19_fails_when_not_every_node_was_evaluated_before_the_write(built, pinned, world):
    plan = dict(built.ctx["plan"], evaluated=built.ctx["plan"]["evaluated"] - 1)
    assert "V19" in tool.run_verifier(pinned, built.A, built.ctx, world.expect, plan=plan).failed()
    plan2 = dict(built.ctx["plan"], assert_failed_listed=5)
    assert "V19" in tool.run_verifier(pinned, built.A, built.ctx, world.expect, plan=plan2).failed()


def test_v1_fails_on_a_wrong_expected_count(built, pinned, world):
    V = tool.run_verifier(pinned, built.A, built.ctx, dict(world.expect, wants=world.expect["wants"] + 1))
    assert "V1" in V.failed()


def test_v19_all_or_nothing_no_output_exists_when_a_node_cannot_be_written(world, pinned, built, tmp_path):
    """A node whose on-disk text drifted from what the classification saw makes the writer STOP mid-file; the run
    must not leave a usable output: the caller (prepare_outputs) raises before anything is reported as written."""
    A2 = dict(built.A, records=[dict(r, _t=("drifted" if r["id"] == world.ids["SEP1"] else r["_t"])) for r in built.A["records"]])
    run_dir, utc = tool.new_run_dir()
    with pytest.raises(tool.Stop, match="drift"):
        tool.build_outputs(pinned, A2, built.approvals, in_dir=built.in_dir, out_dir=os.path.join(run_dir, "out"), run_dir=run_dir,
                           utc=utc, in_hashes=built.ctx["in_hashes"])
    assert not (Path(run_dir) / "out" / "main.msgpack").exists()                     # no partial output survives the STOP
    assert not (Path(run_dir) / ("id-map-inverse-%s.json" % utc)).exists()           # and nothing was reported after it


def test_t6_minted_after_is_a_subset_of_minted_before_and_lists_the_would_mint_set(built, pinned, world, phase1):
    rep = next((phase1.run_dir / "reports").glob("t6-would-mint-*.json"))
    d = json.loads(rep.read_text())
    assert d["minted_after_subset_of_before"] is True and d["before_minus_after_subset_of_new_ids"] is True
    assert d["minted_before"] >= 3 and d["would_mint_count"] == d["minted_after"]
    assert all(set(w) >= {"want_id", "len", "marker_bearing", "region_fired_in_span", "region_fired_in_node"} for w in d["would_mint"])


def test_t6_stops_when_a_want_disappears_without_being_a_new_id(built, pinned, world):
    meta_after = {k: v for k, v in built.A["nodes_meta"].items()}
    ghost = pinned.org.want_id_for_text(X1)
    meta_after[ghost] = {"kind": "want", "want_text": X1, "want_state": "open", "provenance": "cc_authored"}
    with pytest.raises(tool.Stop, match="T6"):
        tool.t6_replay(pinned, built.A["nodes_meta"], meta_after, built.A["content"], {}, tool.Classifier(pinned.org, meta_after, built.A["content"]))


# ==================================================================================================
# approvals (plan 5.2): an unapproved id is REFUSED; everything must match
# ==================================================================================================

def _approvals(built, world, **over):
    A = built.A
    rl = tool.artifact_sha256(tool.repair_list_obj(A))
    sc = tool.artifact_sha256(tool.scope_ids_obj(A, world.expect["scope"]))
    body = tool.approvals_body_for(A["records"], rl, sc, "EXEC-SYNTHETIC-PACKET")
    body.update(over)
    return body, rl, sc


def _write_approvals(tmp_path, body):
    p = tmp_path / "approvals.json"
    data = tool.canonical_json(tool.stamped(body))
    p.write_bytes(data)
    return str(p), hashlib.sha256(data).hexdigest()


def test_an_approved_id_is_written_and_an_unapproved_or_struck_id_is_refused(built, world, tmp_path):
    body, rl, sc = _approvals(built, world)
    body["entries"][0]["decision"] = "struck"                                       # struck = leave EXACTLY as-is
    del body["entries"][1]                                                          # no entry at all
    path, sha = _write_approvals(tmp_path, body)
    ap = tool.load_approvals(path, sha, rl, sc)
    write, refused = tool.gate_write_set(built.A["records"], ap)
    assert len(write) == 1 and sorted(r["reason"] for r in refused) == ["not_approved", "struck"]
    assert not set(write) & {r["id"] for r in refused}


def test_an_approval_whose_hashes_do_not_equal_the_recomputed_values_is_refused(built, world, tmp_path):
    for field in ("excerpt_sha256", "x_sha16", "t_sha16"):
        body, rl, sc = _approvals(built, world)
        body["entries"][0][field] = "0" * 16
        path, sha = _write_approvals(tmp_path, body)
        write, refused = tool.gate_write_set(built.A["records"], tool.load_approvals(path, sha, rl, sc))
        assert refused == [{"id": body["entries"][0]["id"], "reason": "approval_mismatch"}] and len(write) == 2


def test_the_approvals_file_is_refused_unless_every_binding_matches(built, world, tmp_path):
    body, rl, sc = _approvals(built, world)
    path, sha = _write_approvals(tmp_path, body)
    assert tool.load_approvals(path, sha, rl, sc)["packet"] == "EXEC-SYNTHETIC-PACKET"
    with pytest.raises(tool.Refusal, match="REQUIRED"):
        tool.load_approvals(path, "", rl, sc)
    with pytest.raises(tool.Refusal, match="P5"):
        tool.load_approvals(path, "0" * 64, rl, sc)                                # not the hash relayed from the packet
    for field, val in (("function_pin", dict(tool.PIN, code_commit="0" * 40)), ("scrub_version", "scrub-0"),
                       ("repair_list_sha256", "0" * 64), ("scope_ids_sha256", "0" * 64),
                       ("poincare_dir_carried", "not acknowledged"), ("packet", "")):
        b, _, _ = _approvals(built, world, **{field: val})
        p, s = _write_approvals(tmp_path, b)
        with pytest.raises(tool.Refusal, match="approval_mismatch"):
            tool.load_approvals(p, s, rl, sc)
    b, _, _ = _approvals(built, world)
    b["entries"][0]["decision"] = "maybe"
    p, s = _write_approvals(tmp_path, b)
    with pytest.raises(tool.Refusal, match="decision"):
        tool.load_approvals(p, s, rl, sc)
    b, _, _ = _approvals(built, world)
    o = tool.stamped(b)
    o["pin_stamp"]["branch_head"] = "0" * 40
    p = tmp_path / "wrongstamp.json"
    d = tool.canonical_json(o)
    p.write_bytes(d)
    with pytest.raises(tool.Refusal, match="P5"):
        tool.load_approvals(str(p), hashlib.sha256(d).hexdigest(), rl, sc)


def test_a_provisional_self_approval_is_never_accepted_as_an_approval(built, world, tmp_path):
    A = built.A
    rl = tool.artifact_sha256(tool.repair_list_obj(A))
    sc = tool.artifact_sha256(tool.scope_ids_obj(A, world.expect["scope"]))
    body = tool.approvals_body_for(A["records"], rl, sc, tool.PROVISIONAL_PACKET)
    p, s = _write_approvals(tmp_path, body)
    with pytest.raises(tool.Refusal, match="provisional"):
        tool.load_approvals(p, s, rl, sc)


def test_the_rewrite_step_refuses_a_run_directory_outside_the_backups_root(world, phase1, tmp_path):
    other = tmp_path / "rd"
    shutil.copytree(phase1.run_dir, other)                                        # not under BACKUPS: the write guard refuses
    ns = types.SimpleNamespace(run_dir=str(other), scope_min_len=MIN_LEN, expect_scope=world.expect["scope"], expect_wants=1,
                               expect_protected=1, provisional_approve_all=True, approvals=None, approvals_sha256=None)
    with pytest.raises(tool.Refusal, match="write refused"):
        tool.stage_rewrite(ns, tool.load_pinned(str(PIN_ROOT)), {})


def test_the_rewrite_step_stops_when_the_rederived_classification_is_not_the_saved_one(world, phase1):
    """P5: the repair-list the rewrite step derives must hash to the one the classify step froze."""
    other = world.backups / (tool.RUN_DIR_PREFIX + "p5-drift")
    shutil.copytree(phase1.run_dir, other)
    sc = other / "reports" / "scope-ids.json"
    d = json.loads(sc.read_text())
    d["ids"] = d["ids"][:-1]                                                      # a scope that is not the classified one
    sc.write_text(json.dumps(d))
    rec = json.loads((other / "run-record.json").read_text())
    ns = types.SimpleNamespace(run_dir=str(other), scope_min_len=MIN_LEN, expect_scope=world.expect["scope"],
                               expect_wants=world.expect["wants"], expect_protected=world.expect["protected"],
                               provisional_approve_all=True, approvals=None, approvals_sha256=None)
    with pytest.raises(tool.Stop, match="P5"):
        tool.stage_rewrite(ns, tool.load_pinned(str(PIN_ROOT)), {})
    assert rec["artifacts_sha256"]["scope-ids"]


# ==================================================================================================
# Phase 1 never writes the target; the copy step; the CLI refusals
# ==================================================================================================

def test_phase_1_reads_a_read_only_target_and_never_writes_it(pinned, tmp_path):
    w = build_world(tmp_path, pinned)
    before = file_hashes(w.ckpt)
    mtimes = {p.name: p.stat().st_mtime_ns for p in w.ckpt.iterdir() if p.is_file()}
    for p in w.ckpt.iterdir():
        if p.is_file():
            p.chmod(0o444)
    w.ckpt.chmod(0o555)
    mp = pytest.MonkeyPatch()
    patch_world(mp, w)
    try:
        rc, res, err = cli(argv_for(w))
        assert rc == 0, err
        rc2, res2, err2 = cli(argv_for(w, "--run-dir", res["run_dir"], "--provisional-approve-all", step="rewrite"))
        assert rc2 == 0, err2
    finally:
        mp.undo()
        w.ckpt.chmod(0o755)
        for p in w.ckpt.iterdir():
            if p.is_file():
                p.chmod(0o644)
    assert file_hashes(w.ckpt) == before and {p.name: p.stat().st_mtime_ns for p in w.ckpt.iterdir() if p.is_file()} == mtimes
    copy = json.loads((Path(res["run_dir"]) / "copy-hashes.json").read_text())
    assert copy["copy_sha256"] == before
    for n in tool.SIX_FILES:
        assert os.stat(Path(res["run_dir"]) / "copy" / n).st_ino != os.stat(w.ckpt / n).st_ino     # a new inode, never a link


def test_a_missing_checkpoint_file_stops_the_copy(pinned, tmp_path):
    w = build_world(tmp_path, pinned)
    (w.ckpt / "commons.msgpack").unlink()
    mp = pytest.MonkeyPatch()
    patch_world(mp, w)
    try:
        rc, res, err = cli(argv_for(w))
    finally:
        mp.undo()
    assert rc == 3 and "commons.msgpack" in err


def test_apply_is_refused_without_josh_go_before_anything_is_loaded(world):
    rc, res, err = cli(["--pin-root", "/nonexistent", "--target-dir", "/nonexistent", "--daemon-script", "/nonexistent",
                        "--scope-min-len", "600", "--apply"])
    assert rc == 2 and "Josh's go" in err
    rc, res, err = cli(argv_for(world, "--apply"))
    assert rc == 2 and "Josh's go" in err


def test_apply_with_a_go_but_missing_gate_inputs_is_refused(world):
    rc, res, err = cli(argv_for(world, "--apply", "--josh-go", "REF-1"))
    assert rc == 2 and "--apply needs" in err


def test_the_cli_refuses_a_real_target_that_is_not_the_recorded_dir(world, tmp_path, monkeypatch):
    other = tmp_path / "x"
    other.mkdir()
    argv = argv_for(world)
    argv[argv.index("--target-dir") + 1] = str(other)
    rc, res, err = cli(argv)
    assert rc == 2 and "not the recorded" in err


# ==================================================================================================
# P3 fingerprint, P4 daemon-down, P6 peer hold (fake probes; nothing real is queried)
# ==================================================================================================

class FakeProbes(tool.Probes):
    def __init__(self, **kw):
        self.k = {"units_active": set(), "procs": [], "held": [], "alive": False, "cron": "", "env": {}, "enabled": set(),
                  "stat": {}}
        self.k.update(kw)

    def unit_active(self, unit):
        return unit in self.k["units_active"]

    def unit_enabled(self, unit):
        return unit in self.k["enabled"]

    def processes_matching(self, patterns):
        return [p for pat, p in self.k["procs"] if any(pat in x for x in patterns)]

    def files_held_open(self, paths):
        return list(self.k["held"])

    def pid_alive(self, pid):
        return self.k["alive"]

    def crontab_text(self):
        return self.k["cron"]

    def env_get(self, name):
        return self.k["env"].get(name)

    def stat_snapshot(self, dirpath):
        return dict(self.k["stat"]) if self.k["stat"] else super().stat_snapshot(dirpath)


def test_p3_the_fingerprint_battery_passes_on_the_pinned_function_and_carries_the_repro(pinned):
    r = tool.gate_p3(pinned)
    assert r["ok"] and r["battery_rows"] == 16 and r["mismatched_rows"] == []
    assert any(row[0] == "closer_after_backtick" for row in tool.FINGERPRINT)


def test_p3_refuses_a_parser_that_regresses_the_closer_after_backtick_repro(pinned):
    real = pinned.org.parse_wants

    def regressed(s):
        wp = real(s)
        return wp if "foo()" not in s else type(wp)((), wp.skipped)          # drops the genuine closer-after-backtick pair
    r = tool.gate_p3(pinned, parse_fn=regressed)
    assert not r["ok"] and r["mismatched_rows"] == ["closer_after_backtick"]


def _p4kw(world, placed=None):
    return {"code_placed_at": (time.time() - 60) if placed is None else placed, "daemon_log": str(world.log)}


def _manifest_of(d):
    out = {}
    for n in tool.SIX_FILES:
        st = os.stat(os.path.join(d, n))
        out[n] = {"sha256": tool.sha256_file(os.path.join(d, n)), "size": st.st_size, "mtime_ns": st.st_mtime_ns,
                  "st_dev": st.st_dev, "st_ino": st.st_ino, "st_nlink": st.st_nlink}
    return out


def test_p4_daemon_down_is_mechanical_and_every_leg_can_fail(world):
    d = str(world.ckpt)
    kw = _p4kw(world)
    ok = tool.gate_p4(FakeProbes(), d, manifest_files=_manifest_of(d), **kw)
    assert ok["ok"] and "no_pulse_since_code_placement" in ok["checks"]
    for extra, leg in (({"units_active": {tool.DAEMON_UNIT}}, "unit_inactive"),
                       ({"units_active": {tool.RECOVER_TIMER}}, "recover_timer_inactive"),
                       ({"procs": [("cc-ng-service.py", 4242)]}, "no_daemon_process"),
                       ({"held": [(4242, d + "/main.msgpack")]}, "no_process_holds_the_six_files")):
        r = tool.gate_p4(FakeProbes(**extra), d, manifest_files=_manifest_of(d), **kw)
        assert not r["ok"] and not r["checks"][leg]
    manifest = _manifest_of(d)
    manifest[tool.MAIN_NAME]["sha256"] = "0" * 64
    r = tool.gate_p4(FakeProbes(), d, manifest_files=manifest, **kw)
    assert not r["ok"] and not r["checks"]["six_files_equal_the_start_of_phase2_backup"]
    manifest = _manifest_of(d)
    manifest[tool.MAIN_NAME]["st_ino"] += 1                      # equal bytes, a DIFFERENT inode: the file was replaced
    r = tool.gate_p4(FakeProbes(), d, manifest_files=manifest, **kw)
    assert not r["ok"] and not r["checks"]["six_files_equal_the_start_of_phase2_backup"]


def test_p4_a_pid_file_naming_a_live_process_fails_and_a_log_newer_than_the_placement_fails(world):
    (world.ws / "daemon.pid").write_text("4242")
    try:
        r = tool.gate_p4(FakeProbes(alive=True), str(world.ckpt), manifest_files=_manifest_of(str(world.ckpt)), **_p4kw(world))
        assert not r["ok"] and not r["checks"]["daemon_pid_file_names_a_dead_pid"]
        r = tool.gate_p4(FakeProbes(), str(world.ckpt), manifest_files=_manifest_of(str(world.ckpt)), **_p4kw(world, placed=1.0))
        assert not r["ok"] and not r["checks"]["no_pulse_since_code_placement"]
        r = tool.gate_p4(FakeProbes(), str(world.ckpt), manifest_files=_manifest_of(str(world.ckpt)), **_p4kw(world, placed=time.time() - 1))
        assert r["ok"]
    finally:
        (world.ws / "daemon.pid").unlink()


def test_p6_the_peer_hold_h1_h2_h3_each_fail_closed(world):
    c = str(world.conduit)
    assert tool.gate_p6(FakeProbes(), c)["ok"]
    for kw, why in (({"cron": "0 3 * * * /x/callosum-leg1\n"}, "h1_callosum_cron_line_firing"),
                    ({"env": {"CC_CALLOSUM_LEG1_ENABLED": "1"}}, "h1_leg1_flag_env_is_1"),
                    ({"enabled": {tool.LEG2_TIMER}}, "h2_leg2_timer_enabled"),
                    ({"units_active": {tool.LEG2_TIMER}}, "h2_leg2_timer_active"),
                    ({"units_active": {tool.LEG2_SERVICE}}, "h2_leg2_service_active"),
                    ({"procs": [("cc-ng-sync.py leg2-tick", 77)]}, "h2_sync_or_merge_process")):
        r = tool.gate_p6(FakeProbes(**kw), c)
        assert not r["ok"] and why in r["violations"]
    assert not tool.gate_p6(FakeProbes(cron="# 0 3 * * * callosum\n"), c)["violations"]        # a commented line is not firing
    assert "h3_conduit_path_not_recorded_or_under_syls_directories" in tool.gate_p6(FakeProbes(), "")["violations"]


def test_p6_a_change_between_start_and_now_is_caught_and_the_conduit_is_stat_only(world):
    c = str(world.conduit)
    start = tool.hold_snapshot(FakeProbes(), c)
    assert start["h3_conduit_stat"] and "frame-a.bin" in start["h3_conduit_stat"]
    assert tool.gate_p6(FakeProbes(), c, start)["ok"]
    (world.conduit / "frame-b.bin").write_bytes(b"new")
    try:
        r = tool.gate_p6(FakeProbes(), c, start)
        assert not r["ok"] and "h3_conduit_stat_changed_since_start" in r["violations"]
    finally:
        (world.conduit / "frame-b.bin").unlink()
    r = tool.gate_p6(FakeProbes(cron="# changed\n"), c, start)
    assert "h1_crontab_changed_since_start" in r["violations"]


# ==================================================================================================
# the retirement mechanism (plan 6.8)
# ==================================================================================================

def test_a_retired_receipt_for_the_same_checkpoint_directory_refuses(world):
    run = world.backups / (tool.RUN_DIR_PREFIX + "retired-a")
    run.mkdir()
    target = os.path.realpath(world.ckpt)
    tool.refuse_if_retired(target)                                               # nothing yet
    # plan 6.8(3): a receipt for the SAME checkpoint directory refuses whatever mapping / tool sha it records
    (run / "RETIRED-x.receipt").write_text(json.dumps({"checkpoint_dir": target, "tool_sha256": "0" * 64, "mapping_sha256": "1" * 64}))
    try:
        with pytest.raises(tool.Refusal, match="RETIRED"):
            tool.refuse_if_retired(target)
        tool.refuse_if_retired(str(world.base / "some-other-dir"))                # another directory, another tool sha: not blocked
    finally:
        (run / "RETIRED-x.receipt").unlink()
    tool.refuse_if_retired(target)


def test_a_retired_receipt_recording_this_tools_sha256_refuses_for_any_directory(world):
    run = world.backups / (tool.RUN_DIR_PREFIX + "retired-b")
    run.mkdir()
    (run / "RETIRED-y.receipt").write_text(json.dumps({"checkpoint_dir": "/elsewhere", "tool_sha256": tool.tool_sha256()}))
    try:
        with pytest.raises(tool.Refusal, match="RETIRED"):
            tool.refuse_if_retired("/anywhere")
    finally:
        (run / "RETIRED-y.receipt").unlink()


def test_an_unrelated_receipt_does_not_block_and_an_unreadable_one_does(world):
    run = world.backups / (tool.RUN_DIR_PREFIX + "retired-c")
    run.mkdir()
    (run / "RETIRED-z.receipt").write_text(json.dumps({"checkpoint_dir": "/elsewhere", "tool_sha256": "1" * 64}))
    tool.refuse_if_retired(os.path.realpath(world.ckpt))
    (run / "RETIRED-z.receipt").write_text("not json")
    try:
        with pytest.raises(tool.Refusal):
            tool.refuse_if_retired(os.path.realpath(world.ckpt))
    finally:
        (run / "RETIRED-z.receipt").unlink()


# ==================================================================================================
# the WHOLE Phase-2 path, on a SYNTHETIC directory, with fake probes (nothing real is touched)
# ==================================================================================================

def _phase2_world(pinned, tmp_path, mp, **world_kw):
    w = build_world(tmp_path, pinned, **world_kw)
    patch_world(mp, w)
    for name, key in (("EXPECTED_WANTS", "wants"), ("EXPECTED_PROTECTED", "protected"), ("EXPECTED_SCOPE", "scope")):
        mp.setattr(tool, name, w.expect[key])                     # Phase 2 pins the module constants: the synthetic world has its own
    root = tmp_path / "daemon-import-root"                       # stands in for the unit's import root the Chief names (Q9)
    root.mkdir()
    shutil.copy(PIN_ROOT / "cc_ng_organism.py", root / "cc_ng_organism.py")
    w.org_copy = root / "cc_ng_organism.py"
    rc, c1, err = cli(argv_for(w))
    assert rc == 0, err
    rd = Path(c1["run_dir"])
    frozen = rd / "frozen"
    frozen.mkdir()
    for name in ("repair-list", "scope-ids"):
        shutil.copy(rd / "reports" / (name + ".json"), frozen / (name + ".json"))
    src_map = rd / "reports" / "id-map.json"
    if not src_map.exists():                                     # the TURN A tool named it candidate-id-map.json
        src_map = rd / "reports" / "candidate-id-map.json"
    shutil.copy(src_map, frozen / "id-map.json")
    rl = json.loads((frozen / "repair-list.json").read_text())
    body = {"packet": "EXEC-SYNTHETIC-PACKET-P2", "function_pin": dict(tool.PIN), "repair_list_sha256": tool.sha256_file(str(frozen / "repair-list.json")),
            "scope_ids_sha256": tool.sha256_file(str(frozen / "scope-ids.json")), "scrub_version": tool.SCRUB_VERSION,
            "poincare_dir_carried": "acknowledged",
            "entries": [{"id": c["id"], "decision": "approved", "excerpt_sha256": c["excerpt_sha256"], "x_sha16": c["new_sha16"],
                         "t_sha16": c["old_sha16"]} for c in rl["candidates"]]}
    ap = rd / "approvals.json"
    data = tool.canonical_json(tool.stamped(body))
    ap.write_bytes(data)
    return w, rd, frozen, str(ap), hashlib.sha256(data).hexdigest()


def _p2_argv(w, *extra, step="phase2-backup", placed=True):
    more = ["--code-placed-at", str(time.time() - 60)] if placed else []
    return argv_for(w, "--conduit-dir", str(w.conduit), *more, *extra, step=step)


@pytest.fixture()
def p3_stub(monkeypatch):
    monkeypatch.setattr(tool, "gate_p3", lambda pinned, **kw: {"gate": "P3", "ok": True, "stubbed": True})


def test_phase2_backup_refuses_while_the_daemon_is_up_and_writes_a_manifest_otherwise(pinned, tmp_path, monkeypatch):
    w, rd, frozen, ap, aps = _phase2_world(pinned, tmp_path, monkeypatch)
    rc, res, err = cli(_p2_argv(w), probes=FakeProbes(units_active={tool.DAEMON_UNIT}))
    assert rc == 2 and "P4/P6" in err
    rc, res, err = cli(_p2_argv(w), probes=FakeProbes(cron="0 3 * * * callosum\n"))
    assert rc == 2
    before = file_hashes(w.ckpt)
    rc, res, err = cli(_p2_argv(w), probes=FakeProbes())
    assert rc == 0, err
    mp = next(Path(res["run_dir"]).glob("backup-manifest-*.json"))
    assert tool.sha256_file(str(mp)) == res["backup_manifest_sha256"]
    man = json.loads(mp.read_text())
    assert sorted(man["files"]) == sorted(tool.SIX_FILES) and man["independent_reread_sha256"] == before
    assert file_hashes(w.ckpt) == before


def test_apply_refuses_when_josh_go_does_not_quote_the_backup_manifest(pinned, tmp_path, monkeypatch, p3_stub):
    w, rd, frozen, ap, aps = _phase2_world(pinned, tmp_path, monkeypatch)
    rc, res, err = cli(_p2_argv(w), probes=FakeProbes())
    p2 = res["run_dir"]
    before = file_hashes(w.ckpt)
    common = ["--apply", "--run-dir", p2, "--frozen-dir", str(frozen), "--approvals", ap, "--approvals-sha256", aps,
              "--daemon-organism-file", str(w.org_copy)]
    rc, res2, err = cli(_p2_argv(w, *common, "--josh-go", "GO-1", "--josh-go-manifest-sha256", "0" * 64), probes=FakeProbes())
    assert rc == 2 and "quotes" in err
    rc, res2, err = cli(_p2_argv(w, *common), probes=FakeProbes())
    assert rc == 2 and "Josh's go" in err
    assert file_hashes(w.ckpt) == before


def test_apply_refuses_when_the_daemon_organism_file_is_not_the_pin(pinned, tmp_path, monkeypatch, p3_stub):
    w, rd, frozen, ap, aps = _phase2_world(pinned, tmp_path, monkeypatch)
    rc, res, err = cli(_p2_argv(w), probes=FakeProbes())
    p2, msha = res["run_dir"], res["backup_manifest_sha256"]
    fake = tmp_path / "other-organism.py"
    fake.write_text("# not the pin\n")
    before = file_hashes(w.ckpt)
    rc, res2, err = cli(_p2_argv(w, "--apply", "--run-dir", p2, "--frozen-dir", str(frozen), "--approvals", ap, "--approvals-sha256", aps,
                                 "--daemon-organism-file", str(fake), "--josh-go", "GO-1", "--josh-go-manifest-sha256", msha),
                        probes=FakeProbes())
    assert rc == 2 and "P2" in err and file_hashes(w.ckpt) == before


def test_apply_writes_the_pair_atomically_verifies_it_then_retires_the_tool(pinned, tmp_path, monkeypatch, p3_stub):
    w, rd, frozen, ap, aps = _phase2_world(pinned, tmp_path, monkeypatch)
    rc, res, err = cli(_p2_argv(w), probes=FakeProbes())
    assert rc == 0, err
    p2, msha = res["run_dir"], res["backup_manifest_sha256"]
    before = file_hashes(w.ckpt)
    args = _p2_argv(w, "--apply", "--run-dir", p2, "--frozen-dir", str(frozen), "--approvals", ap, "--approvals-sha256", aps,
                    "--daemon-organism-file", str(w.org_copy), "--josh-go", "GO-1", "--josh-go-manifest-sha256", msha)
    rc, res2, err = cli(args, probes=FakeProbes())
    assert rc == 0, err
    after = file_hashes(w.ckpt)
    assert after[tool.MAIN_NAME] != before[tool.MAIN_NAME] and after[tool.SIDECAR_NAME] != before[tool.SIDECAR_NAME]
    for n in (tool.VECTORS_NAME, tool.GUARD_NAME, tool.MANIFEST_NAME, tool.COMMONS_NAME):
        assert after[n] == before[n]
    g = pinned.nf.Graph()
    g.restore(str(w.ckpt / tool.MAIN_NAME))
    assert w.new["SEP1"] in g.nodes and w.ids["SEP1"] not in g.nodes and g.nodes[w.new["SEP1"]].metadata["want_text"] == X1
    assert w.cc1 in g.nodes and w.cc2 in g.nodes and w.rim in g.nodes
    receipts = list(Path(p2).glob("RETIRED-*.receipt"))
    assert len(receipts) == 1 and json.loads(receipts[0].read_text())["checkpoint_dir"] == os.path.realpath(w.ckpt)
    assert res2["applied"] == 3 and "gate P10" in res2["note"]
    # the tool is one-shot: a second apply and a second backup are both REFUSED
    rc, res3, err = cli(args, probes=FakeProbes())
    assert rc == 2 and "RETIRED" in err
    rc, res4, err = cli(_p2_argv(w), probes=FakeProbes())
    assert rc == 2 and "RETIRED" in err
    assert file_hashes(w.ckpt) == after


def test_apply_rechecks_the_gates_immediately_before_the_replace(pinned, tmp_path, monkeypatch, p3_stub):
    """If the daemon comes back between the staging and the os.replace, the live files stay untouched."""
    w, rd, frozen, ap, aps = _phase2_world(pinned, tmp_path, monkeypatch)
    rc, res, err = cli(_p2_argv(w), probes=FakeProbes())
    p2, msha = res["run_dir"], res["backup_manifest_sha256"]
    before = file_hashes(w.ckpt)
    real_p4, calls = tool.gate_p4, {"n": 0}

    def p4(*a, **k):
        calls["n"] += 1
        r = real_p4(*a, **k)
        return dict(r, ok=False) if calls["n"] >= 2 else r                    # call 1 = the gate list; call 2 = the re-check
    monkeypatch.setattr(tool, "gate_p4", p4)
    args = _p2_argv(w, "--apply", "--run-dir", p2, "--frozen-dir", str(frozen), "--approvals", ap, "--approvals-sha256", aps,
                    "--daemon-organism-file", str(w.org_copy), "--josh-go", "GO-1", "--josh-go-manifest-sha256", msha)
    rc, res2, err = cli(args, probes=FakeProbes())
    assert rc == 2 and "immediately before os.replace" in err and calls["n"] == 2
    assert file_hashes(w.ckpt) == before
    assert not list(Path(p2).glob("RETIRED-*.receipt"))


# ==================================================================================================
# TURN A2 - the hardening follow-up (le-029 C1-C8 + N1, checker-026 c026-C3, the P1 heads).
# FAILING-FIRST: every test below was committed BEFORE the tool change and fails against the TURN A tool.
# ==================================================================================================

ProbeError = getattr(tool, "ProbeError", type("MissingProbeError", (Exception,), {}))     # old tool: behavioural failure, not AttributeError


def _cp(rc=0, out="", err=""):
    return subprocess.CompletedProcess([], rc, stdout=out, stderr=err)


def _returns(cp):
    return lambda *a, **k: cp


def _raises(exc):
    def f(*a, **k):
        raise exc
    return f


_BUS_MSG = "Failed to connect to bus: No medium found"
HOST_ERRORS = {                                   # what the REAL Probes must treat as "cannot tell", never as "down"/"off"
    "bus_unreachable": _returns(_cp(1, "", _BUS_MSG)),
    "empty_stdout_rc0": _returns(_cp(0, "", "")),
    "nonzero_rc_with_a_word": _returns(_cp(1, "inactive", "")),
    "unknown_word": _returns(_cp(3, "weird-state", "")),
    "timeout": _raises(subprocess.TimeoutExpired("systemctl", 15)),
    "missing_binary": _raises(FileNotFoundError("systemctl")),
    "os_error": _raises(OSError("boom")),
}


def _with_flag(argv, flag, value):
    out = list(argv)
    out[out.index(flag) + 1] = str(value)
    return out


def _apply_argv(w, p2, msha, frozen, ap, aps, **kw):
    return _p2_argv(w, "--apply", "--run-dir", p2, "--frozen-dir", str(frozen), "--approvals", ap, "--approvals-sha256", aps,
                    "--daemon-organism-file", str(w.org_copy), "--josh-go", "GO-1", "--josh-go-manifest-sha256", msha, **kw)


def _partner_args(w):
    return ["--generation-partner", "main.msgpack=%s" % (w.gen / "main.msgpack"),
            "--generation-partner", "vectors.msgpack=%s" % (w.gen / "vectors.msgpack")]


def _ident(path):
    st = os.stat(path)
    return {"st_dev": st.st_dev, "st_ino": st.st_ino, "st_nlink": st.st_nlink}


def _applied(pinned, tmp_path, mp):
    """A complete synthetic Phase 2: classify, backup (with the two generation partners), gated apply."""
    w, rd, frozen, ap, aps = _phase2_world(pinned, tmp_path, mp)
    rc, res, err = cli(_p2_argv(w, *_partner_args(w)), probes=FakeProbes())
    assert rc == 0, err
    p2, msha = res["run_dir"], res["backup_manifest_sha256"]
    before, ino_before = file_hashes(w.ckpt), {n: _ident(w.ckpt / n) for n in tool.SIX_FILES}
    rc, res2, err = cli(_apply_argv(w, p2, msha, frozen, ap, aps), probes=FakeProbes())
    assert rc == 0, err
    receipt = next(Path(p2).glob("post-apply-receipt-FINAL-*.json"))
    return types.SimpleNamespace(w=w, p2=p2, msha=msha, frozen=frozen, ap=ap, aps=aps, before=before, ino_before=ino_before,
                                 after=file_hashes(w.ckpt), receipt_sha=tool.sha256_file(str(receipt)), receipt=receipt)


def _rb_argv(a, *extra, **kw):
    return _p2_argv(a.w, "--run-dir", a.p2, "--josh-go", "GO-RB", "--josh-go-manifest-sha256", a.msha,
                    "--josh-go-receipt-sha256", a.receipt_sha, *extra, step="rollback", **kw)


# ---- C1: --code-placed-at and a readable daemon.log are REQUIRED; the leg is never omitted --------------------------

def test_c1_gate_p4_never_omits_the_code_placement_leg_and_a_missing_log_is_not_ok(world):
    d, man = str(world.ckpt), _manifest_of(str(world.ckpt))
    for kw in ({"code_placed_at": None, "daemon_log": str(world.log)},
               {"code_placed_at": time.time() - 60, "daemon_log": str(world.ws / "missing.log")},
               {"code_placed_at": time.time() - 60, "daemon_log": None}):
        r = tool.gate_p4(FakeProbes(), d, manifest_files=man, **kw)
        assert "no_pulse_since_code_placement" in r["checks"] and r["checks"]["no_pulse_since_code_placement"] is False and not r["ok"], kw
    world.log.chmod(0)
    try:
        r = tool.gate_p4(FakeProbes(), d, manifest_files=man, **_p4kw(world))
        assert not r["ok"] and not r["checks"]["no_pulse_since_code_placement"]          # an unreadable log is not evidence
    finally:
        world.log.chmod(0o644)
    assert tool.gate_p4(FakeProbes(), d, manifest_files=man, **_p4kw(world))["ok"]


def test_c1_the_manifest_leg_is_not_omitted_by_omission_either(world):
    r = tool.gate_p4(FakeProbes(), str(world.ckpt), **_p4kw(world))
    assert not r["ok"] and r["checks"]["six_files_equal_the_start_of_phase2_backup"] is False


def test_c1_phase2_backup_and_apply_refuse_without_a_parseable_code_placed_at_and_a_readable_log(pinned, tmp_path, monkeypatch, p3_stub):
    w, rd, frozen, ap, aps = _phase2_world(pinned, tmp_path, monkeypatch)
    before = file_hashes(w.ckpt)
    runs = lambda: sorted(p.name for p in w.backups.iterdir())
    n_runs = len(runs())
    rc, res, err = cli(_p2_argv(w, placed=False), probes=FakeProbes())
    assert rc == 2 and "code-placed-at" in err
    rc, res, err = cli(argv_for(w, "--conduit-dir", str(w.conduit), "--code-placed-at", "not-a-time", step="phase2-backup"), probes=FakeProbes())
    assert rc == 2 and "code-placed-at" in err
    assert len(runs()) == n_runs                                                        # refused BEFORE a run dir or any copy
    rc, res, err = cli(_p2_argv(w), probes=FakeProbes())
    assert rc == 0, err
    p2, msha = res["run_dir"], res["backup_manifest_sha256"]
    rc, res, err = cli(_apply_argv(w, p2, msha, frozen, ap, aps, placed=False), probes=FakeProbes())
    assert rc == 2 and "code-placed-at" in err and file_hashes(w.ckpt) == before
    w.log.unlink()                                                                      # a missing daemon.log is NOT ok either
    rc, res, err = cli(_apply_argv(w, p2, msha, frozen, ap, aps), probes=FakeProbes())
    assert rc == 2 and file_hashes(w.ckpt) == before
    rc, res, err = cli(_p2_argv(w), probes=FakeProbes())
    assert rc == 2 and "P4/P6" in err


# ---- C2: the probes FAIL CLOSED (the REAL Probes, subprocess stubbed at the run boundary) -------------------------

@pytest.mark.parametrize("case", sorted(HOST_ERRORS))
def test_c2_real_probes_unit_active_cannot_tell_is_an_error_not_inactive(monkeypatch, case):
    monkeypatch.setattr(tool.subprocess, "run", HOST_ERRORS[case])
    with pytest.raises(ProbeError):
        tool.Probes().unit_active(tool.DAEMON_UNIT)


@pytest.mark.parametrize("case", sorted(HOST_ERRORS))
def test_c2_real_probes_unit_enabled_cannot_tell_is_an_error_not_off(monkeypatch, case):
    monkeypatch.setattr(tool.subprocess, "run", HOST_ERRORS[case])
    with pytest.raises(ProbeError):
        tool.Probes().unit_enabled(tool.LEG2_TIMER)


@pytest.mark.parametrize("case", ("bus_unreachable", "timeout", "missing_binary", "os_error"))
def test_c2_real_probes_crontab_errors_are_errors_not_an_empty_crontab(monkeypatch, case):
    monkeypatch.setattr(tool.subprocess, "run", HOST_ERRORS[case])
    with pytest.raises(ProbeError):
        tool.Probes().crontab_text()
    for rc, err in ((2, "crontab: internal error"), (1, "some other crontab failure"), (1, ""), (127, "not found")):
        monkeypatch.setattr(tool.subprocess, "run", _returns(_cp(rc, "", err)))
        with pytest.raises(ProbeError):
            tool.Probes().crontab_text()


def test_c2_proc_scans_fail_when_an_own_entry_cannot_be_read_and_skip_vanished_ones(tmp_path, monkeypatch):
    root = tmp_path / "proc"
    (root / "4242").mkdir(parents=True)
    (root / "4243").mkdir()                                                           # vanished mid-scan: no cmdline / no fd dir
    cmd = root / "4242" / "cmdline"
    cmd.write_bytes(b"python3\0cc-ng-service.py\0run")
    fd = root / "4242" / "fd"
    fd.mkdir()
    monkeypatch.setattr(tool.glob, "glob", lambda pat: [str(root / "4242"), str(root / "4243")])
    assert tool.Probes().processes_matching(("cc-ng-service.py",)) == [4242]
    assert tool.Probes().files_held_open([str(tmp_path / "nothing")]) == []
    cmd.chmod(0)
    fd.chmod(0)
    try:
        with pytest.raises(ProbeError):
            tool.Probes().processes_matching(("cc-ng-service.py",))
        with pytest.raises(ProbeError):
            tool.Probes().files_held_open([str(tmp_path / "nothing")])
    finally:
        cmd.chmod(0o644)
        fd.chmod(0o755)


def test_c2_gates_with_the_real_probes_fail_closed_when_the_host_cannot_be_queried(world, monkeypatch):
    for case in ("bus_unreachable", "missing_binary", "timeout"):
        monkeypatch.setattr(tool.subprocess, "run", HOST_ERRORS[case])
        r = tool.gate_p4(tool.Probes(), str(world.ckpt), manifest_files=_manifest_of(str(world.ckpt)), **_p4kw(world))
        assert not r["ok"] and not r["checks"]["unit_inactive"] and not r["checks"]["recover_timer_inactive"], case
        assert r.get("probe_errors"), case
        r6 = tool.gate_p6(tool.Probes(), str(world.conduit))
        assert not r6["ok"] and r6["violations"], case
        snap = tool.hold_snapshot(tool.Probes(), str(world.conduit))                     # never raises; records the errors
        assert snap.get("probe_errors"), case


def test_c2_phase2_backup_with_the_real_probes_refuses_when_the_host_cannot_be_queried(pinned, tmp_path, monkeypatch):
    w, rd, frozen, ap, aps = _phase2_world(pinned, tmp_path, monkeypatch)
    n_runs = len(list(w.backups.iterdir()))
    before = file_hashes(w.ckpt)
    real_run = subprocess.run

    def only_systemctl_and_crontab_fail(argv, *a, **k):
        if argv and argv[0] in ("systemctl", "crontab"):
            raise FileNotFoundError(argv[0])
        return real_run(argv, *a, **k)
    monkeypatch.setattr(tool.subprocess, "run", only_systemctl_and_crontab_fail)
    rc, res, err = cli(_p2_argv(w), probes=None)                                        # probes=None -> the REAL Probes()
    assert rc == 2 and "P4/P6" in err
    assert len(list(w.backups.iterdir())) == n_runs and file_hashes(w.ckpt) == before


# ---- C3: P2 - record the file identity and refuse a path inside the pin or the tool worktree -----------------------

def test_c3_gate_p2_records_the_identity_and_refuses_the_pin_and_tool_worktrees(tmp_path):
    good = tmp_path / "unit-root" / "cc_ng_organism.py"
    good.parent.mkdir()
    shutil.copy(PIN_ROOT / "cc_ng_organism.py", good)
    r = tool.gate_p2(str(good), str(PIN_ROOT), str(tmp_path / "toolwt"))
    assert r["ok"] and r["realpath"] == os.path.realpath(good) and r["sha256"] == tool.PIN["cc_ng_organism_sha256"]
    assert r["st_ino"] == os.stat(good).st_ino and r["mtime_ns"] == os.stat(good).st_mtime_ns
    assert not tool.gate_p2(str(PIN_ROOT / "cc_ng_organism.py"), str(PIN_ROOT), str(tmp_path / "toolwt"))["ok"]
    inside_tool = tmp_path / "toolwt" / "sub" / "cc_ng_organism.py"
    inside_tool.parent.mkdir(parents=True)
    shutil.copy(PIN_ROOT / "cc_ng_organism.py", inside_tool)
    r2 = tool.gate_p2(str(inside_tool), str(PIN_ROOT), str(tmp_path / "toolwt"))
    assert not r2["ok"] and "inside" in r2["reason"]
    link = tmp_path / "link.py"
    link.symlink_to(PIN_ROOT / "cc_ng_organism.py")
    assert not tool.gate_p2(str(link), str(PIN_ROOT), str(tmp_path / "toolwt"))["ok"]      # realpath resolves into the pin
    assert not tool.gate_p2(str(tmp_path / "absent.py"), str(PIN_ROOT), str(tmp_path / "toolwt"))["ok"]


def test_c3_apply_refuses_a_daemon_organism_file_that_is_the_pin_worktrees_own_file(pinned, tmp_path, monkeypatch, p3_stub):
    w, rd, frozen, ap, aps = _phase2_world(pinned, tmp_path, monkeypatch)
    rc, res, err = cli(_p2_argv(w), probes=FakeProbes())
    p2, msha = res["run_dir"], res["backup_manifest_sha256"]
    before = file_hashes(w.ckpt)
    link = tmp_path / "via-symlink.py"
    link.symlink_to(PIN_ROOT / "cc_ng_organism.py")
    for bad in (PIN_ROOT / "cc_ng_organism.py", link):
        argv = _apply_argv(w, p2, msha, frozen, ap, aps)
        argv = _with_flag(argv, "--daemon-organism-file", bad)
        rc, res2, err = cli(argv, probes=FakeProbes())
        assert rc == 2 and "P2" in err and file_hashes(w.ckpt) == before, str(bad)
    assert "import root" in tool.build_parser().format_help()                           # the help names what the operator must give


# ---- C4: --expect-* are pinned to the module constants at phase2-backup and --apply ---------------------------------

def test_c4_backup_and_apply_refuse_expect_flags_that_differ_from_the_module_constants(pinned, tmp_path, monkeypatch, p3_stub):
    w, rd, frozen, ap, aps = _phase2_world(pinned, tmp_path, monkeypatch)
    for flag, key in (("--expect-wants", "wants"), ("--expect-protected", "protected"), ("--expect-scope", "scope")):
        rc, res, err = cli(_with_flag(_p2_argv(w), flag, w.expect[key] - 1), probes=FakeProbes())
        assert rc == 2 and "expect" in err, flag
    rc, res, err = cli(_p2_argv(w), probes=FakeProbes())
    assert rc == 0, err
    p2, msha = res["run_dir"], res["backup_manifest_sha256"]
    before = file_hashes(w.ckpt)
    for flag, key in (("--expect-wants", "wants"), ("--expect-protected", "protected"), ("--expect-scope", "scope")):
        rc, res2, err = cli(_with_flag(_apply_argv(w, p2, msha, frozen, ap, aps), flag, w.expect[key] - 1), probes=FakeProbes())
        assert rc == 2 and "expect" in err and file_hashes(w.ckpt) == before, flag
    rc, res3, err = cli(_with_flag(argv_for(w), "--expect-scope", w.expect["scope"] - 1))
    assert rc == 0, err                                                                 # Phase 1 on a COPY may still take other values


# ---- C5: the dry-run packet is structurally not an approval --------------------------------------------------------

def test_c5_the_dry_run_packet_is_structurally_not_an_approval(phase1):
    p = next((phase1.run_dir / "reports").glob("PROVISIONAL-approvals-*.json"))
    d = json.loads(p.read_text())
    assert d["provisional"] is True and d["entries"] and all(e["decision"] == "provisional" for e in d["entries"])
    assert "provisional" not in ("approved", "struck")


def test_c5_a_forged_provisional_packet_is_still_refused_for_phase_2(phase1, tmp_path):
    p = next((phase1.run_dir / "reports").glob("PROVISIONAL-approvals-*.json"))
    rl, sc = phase1.classify["repair_list_sha256"], phase1.classify["scope_ids_sha256"]

    def try_load(obj):
        f = tmp_path / "forged.json"
        data = tool.canonical_json(obj)
        f.write_bytes(data)
        return tool.load_approvals(str(f), hashlib.sha256(data).hexdigest(), rl, sc)
    base = json.loads(p.read_text())
    forged = copy.deepcopy(base)
    forged["packet"] = "FORGED-EXEC-PACKET"                                              # ONE string edited, then re-hashed
    with pytest.raises(tool.Refusal, match="provisional|decision"):
        try_load(forged)
    forged = copy.deepcopy(base)
    forged["provisional"] = False                                                       # the flag alone is not enough either
    with pytest.raises(tool.Refusal, match="provisional|decision"):
        try_load(forged)
    forged = copy.deepcopy(base)
    forged["packet"] = "FORGED-EXEC-PACKET"
    for e in forged["entries"]:
        e["decision"] = "approved"                                                      # the flag is still set: refused
    with pytest.raises(tool.Refusal, match="provisional"):
        try_load(forged)


# ---- C6: inode evidence, the write guard, the ripple row, and the gated rollback -----------------------------------

def test_c6_the_ripple_table_names_generations_as_incidental_and_never_a_rollback_source():
    row = tool.RIPPLE_TABLE["generations/"]
    assert "incidental" in row and "expiring" in row and "never a rollback source" in row


def test_c6_the_backup_manifest_records_before_inodes_and_the_generation_partners_by_inode(pinned, tmp_path, monkeypatch):
    w, rd, frozen, ap, aps = _phase2_world(pinned, tmp_path, monkeypatch)
    seen = []
    real_listdir, real_scandir, real_glob = os.listdir, os.scandir, tool.glob.glob
    monkeypatch.setattr(os, "listdir", lambda p=".", *a: (seen.append(str(p)), real_listdir(p, *a))[1])
    monkeypatch.setattr(os, "scandir", lambda p=".", *a: (seen.append(str(p)), real_scandir(p, *a))[1])
    monkeypatch.setattr(tool.glob, "glob", lambda pat, *a, **k: (seen.append(str(pat)), real_glob(pat, *a, **k))[1])
    ino = {n: _ident(w.ckpt / n) for n in tool.SIX_FILES}
    rc, res, err = cli(_p2_argv(w, *_partner_args(w)), probes=FakeProbes())
    assert rc == 0, err
    assert not [s for s in seen if "generations" in s]                                  # never listed, only stat/hash of the known paths
    man = json.loads(next(Path(res["run_dir"]).glob("backup-manifest-*.json")).read_text())
    for n in tool.SIX_FILES:
        for k in ("st_dev", "st_ino", "st_nlink"):
            assert man["files"][n][k] == ino[n][k], (n, k)
    assert man["files"][tool.MAIN_NAME]["st_nlink"] == 2                                # measured, not inferred: os.link made it 2
    parts = {p["name"]: p for p in man["generation_partners"]}
    assert set(parts) == {tool.MAIN_NAME, tool.VECTORS_NAME}
    for n, p in parts.items():
        assert p["same_file_as_live"] is True and (p["st_dev"], p["st_ino"]) == (ino[n]["st_dev"], ino[n]["st_ino"])
        assert p["sha256"] == man["files"][n]["sha256"] and p["path"] == str(w.gen / n)
    assert "never a rollback source" in man["ripple"]["generations/"]


def test_c6_a_partner_is_judged_by_inode_never_by_link_count(pinned, tmp_path, monkeypatch):
    w, rd, frozen, ap, aps = _phase2_world(pinned, tmp_path, monkeypatch)
    copy_dir = w.ckpt / "generations" / "20260924T000000Z"
    copy_dir.mkdir()
    shutil.copy(w.ckpt / "main.msgpack", copy_dir / "main.msgpack")                     # same bytes, DIFFERENT inode, link count 1
    rc, res, err = cli(_p2_argv(w, "--generation-partner", "main.msgpack=%s" % (copy_dir / "main.msgpack")), probes=FakeProbes())
    assert rc == 0, err
    man = json.loads(next(Path(res["run_dir"]).glob("backup-manifest-*.json")).read_text())
    p = man["generation_partners"][0]
    assert p["same_file_as_live"] is False and p["sha256"] == man["files"][tool.MAIN_NAME]["sha256"]
    outside = tmp_path / "elsewhere" / "main.msgpack"
    outside.parent.mkdir()
    shutil.copy(w.ckpt / "main.msgpack", outside)
    rc, res, err = cli(_p2_argv(w, "--generation-partner", "main.msgpack=%s" % outside), probes=FakeProbes())
    assert rc == 2 and "generations" in err                                             # only a path under <target>/generations/
    rc, res, err = cli(_p2_argv(w, "--generation-partner", "bogus.msgpack=%s" % (copy_dir / "main.msgpack")), probes=FakeProbes())
    assert rc == 2


def test_c6_apply_records_before_and_after_inodes_and_asserts_new_inode_with_link_count_1(pinned, tmp_path, monkeypatch, p3_stub):
    a = _applied(pinned, tmp_path, monkeypatch)
    w = a.w
    rc_file = next(Path(a.p2).glob("post-apply-receipt-FINAL-*.json"))
    rec = json.loads(rc_file.read_text())
    assert rec["live_inodes_before"] == {n: a.ino_before[n] for n in tool.SIX_FILES}
    now = {n: _ident(w.ckpt / n) for n in tool.SIX_FILES}
    assert rec["live_inodes_after"] == now
    for n in (tool.MAIN_NAME, tool.SIDECAR_NAME):                                       # rewritten: NEW inode, link count 1
        assert now[n]["st_ino"] != a.ino_before[n]["st_ino"] and now[n]["st_nlink"] == 1 and now[n]["st_dev"] == a.ino_before[n]["st_dev"]
    for n in (tool.GUARD_NAME, tool.MANIFEST_NAME, tool.COMMONS_NAME, tool.VECTORS_NAME):   # not rewritten: the SAME inode
        assert now[n]["st_ino"] == a.ino_before[n]["st_ino"]
    assert now[tool.VECTORS_NAME]["st_nlink"] == 2                                      # still shares its inode with the generation copy
    pa = {p["name"]: p for p in rec["generation_partners_after"]}
    assert pa[tool.MAIN_NAME]["st_ino"] == a.ino_before[tool.MAIN_NAME]["st_ino"] and pa[tool.MAIN_NAME]["st_nlink"] == 1
    assert pa[tool.MAIN_NAME]["sha256"] == json.loads(next(Path(a.p2).glob("backup-manifest-*.json")).read_text())["files"][tool.MAIN_NAME]["sha256"]
    assert pa[tool.MAIN_NAME]["same_file_as_live"] is False and pa[tool.VECTORS_NAME]["same_file_as_live"] is True


def test_c6_a_write_destination_with_more_than_one_link_is_refused(world, tmp_path):
    run = world.backups / (tool.RUN_DIR_PREFIX + "nlink")
    run.mkdir()
    target, other = run / "x.json", tmp_path / "other-name"
    target.write_text("a")
    os.link(target, other)
    assert os.stat(target).st_nlink == 2
    with pytest.raises(tool.Refusal, match="link"):
        tool.out_write_bytes(str(target), b"b")
    assert other.read_text() == "a"                                                      # the other name was never written through
    with pytest.raises(tool.Refusal, match="link"):
        tool.rewrite_main(b"\x80", str(target), {}, {}, {})
    (run / "solo.json").write_text("a")
    assert tool.out_write_bytes(str(run / "solo.json"), b"b")                            # a link-count-1 destination is fine


def test_c6_rollback_after_a_complete_apply_restores_the_six_files_from_the_named_backup(pinned, tmp_path, monkeypatch, p3_stub):
    a = _applied(pinned, tmp_path, monkeypatch)
    w = a.w
    part_before = {n: (_ident(w.gen / n), tool.sha256_file(str(w.gen / n))) for n in (tool.MAIN_NAME, tool.VECTORS_NAME)}
    rc, res, err = cli(_rb_argv(a), probes=FakeProbes())
    assert rc == 0, err
    assert file_hashes(w.ckpt) == a.before                                               # all six equal the pre-apply bytes
    for n in (tool.MAIN_NAME, tool.SIDECAR_NAME):
        assert _ident(w.ckpt / n)["st_nlink"] == 1                                       # restored by tmp + os.replace: a new inode
    rec = json.loads(next(Path(a.p2).glob("rollback-receipt-*.json")).read_text())
    assert rec["restored"] == sorted([tool.MAIN_NAME, tool.SIDECAR_NAME]) and rec["source_dir"] == os.path.join(os.path.realpath(a.p2), "backup")
    assert rec["live_inodes_after"] == {n: _ident(w.ckpt / n) for n in tool.SIX_FILES}
    assert "generations" in rec["never_a_source"] and rec["host_stays_down"] is True
    assert {n: (_ident(w.gen / n), tool.sha256_file(str(w.gen / n))) for n in part_before} == part_before   # the partners were never written
    assert list(Path(a.p2).glob("RETIRED-*.receipt"))                                    # the tool stays retired
    rc, res, err = cli(_apply_argv(w, a.p2, a.msha, a.frozen, a.ap, a.aps), probes=FakeProbes())
    assert rc == 2 and "RETIRED" in err


def test_c6_rollback_completes_a_torn_apply(pinned, tmp_path, monkeypatch, p3_stub):
    a = _applied(pinned, tmp_path, monkeypatch)
    w = a.w
    shutil.copyfile(Path(a.p2) / "backup" / tool.SIDECAR_NAME, w.ckpt / tool.SIDECAR_NAME)         # main NEW, sidecar OLD
    torn = file_hashes(w.ckpt)
    assert torn[tool.MAIN_NAME] != a.before[tool.MAIN_NAME] and torn[tool.SIDECAR_NAME] == a.before[tool.SIDECAR_NAME]
    rc, res, err = cli(_rb_argv(a), probes=FakeProbes())
    assert rc == 0, err
    assert file_hashes(w.ckpt) == a.before
    rec = json.loads(next(Path(a.p2).glob("rollback-receipt-*.json")).read_text())
    assert rec["restored"] == [tool.MAIN_NAME]


def test_c6_rollback_is_gated_go_manifest_daemon_down_and_pinned_expectations(pinned, tmp_path, monkeypatch, p3_stub):
    a = _applied(pinned, tmp_path, monkeypatch)
    w, after = a.w, a.after
    argv = _rb_argv(a)
    cases = [
        ([x for x in argv if x != "--josh-go" and x != "GO-RB"], FakeProbes(), "Josh's go"),
        (_with_flag(argv, "--josh-go-manifest-sha256", "0" * 64), FakeProbes(), "quotes"),
        (argv, FakeProbes(units_active={tool.DAEMON_UNIT}), "P4"),
        (argv, FakeProbes(cron="0 3 * * * callosum\n"), "P6"),
        (_rb_argv(a, placed=False), FakeProbes(), "code-placed-at"),
        (_with_flag(argv, "--expect-wants", w.expect["wants"] - 1), FakeProbes(), "expect"),
    ]
    for av, probes, needle in cases:
        rc, res, err = cli(av, probes=probes)
        assert rc == 2 and needle in err, (needle, err[-200:])
        assert file_hashes(w.ckpt) == after, needle


def test_c6_rollback_refuses_a_live_file_that_matches_neither_the_backup_nor_the_receipt(pinned, tmp_path, monkeypatch, p3_stub):
    a = _applied(pinned, tmp_path, monkeypatch)
    w = a.w
    with open(w.ckpt / tool.COMMONS_NAME, "ab") as f:
        f.write(b"x")                                                                    # someone else wrote it after the apply
    changed = file_hashes(w.ckpt)
    rc, res, err = cli(_rb_argv(a), probes=FakeProbes())
    assert rc == 2 and "identity" in err and file_hashes(w.ckpt) == changed


def test_c6_rollback_verifies_every_backup_sha256_before_writing_anything(pinned, tmp_path, monkeypatch, p3_stub):
    a = _applied(pinned, tmp_path, monkeypatch)
    w = a.w
    with open(Path(a.p2) / "backup" / tool.MAIN_NAME, "ab") as f:
        f.write(b"x")                                                                    # the named backup was corrupted
    rc, res, err = cli(_rb_argv(a), probes=FakeProbes())
    assert rc == 2 and "sha256" in err and file_hashes(w.ckpt) == a.after


def test_c6_rollback_source_is_only_the_tools_own_run_directory_never_a_generation_or_last_good(pinned, tmp_path, monkeypatch, p3_stub):
    a = _applied(pinned, tmp_path, monkeypatch)
    w = a.w
    for bad in (str(w.gen), str(w.ckpt / "last_good"), str(tmp_path)):
        argv = _with_flag(_rb_argv(a), "--run-dir", bad)
        rc, res, err = cli(argv, probes=FakeProbes())
        assert rc == 2, bad
        assert file_hashes(w.ckpt) == a.after


# ---- C7 / C8: the Choice Clause records must be PRESENT; the deny-check reads the metadata flag --------------------

def test_c7_v15_and_v16_stop_when_the_choice_clause_wants_or_the_rim_are_absent(pinned, tmp_path, monkeypatch):
    w = build_world(tmp_path, pinned, include_protected=False)
    patch_world(monkeypatch, w)
    rc, r1, err = cli(argv_for(w))
    assert rc == 0, err
    rc, r2, err = cli(argv_for(w, "--run-dir", r1["run_dir"], "--provisional-approve-all", step="rewrite"))
    assert rc == 3 and "V15" in err and "V16" in err


def test_c7_v15_records_presence_in_the_input_and_the_output(built, pinned, world):
    d = _check(_verify(built, pinned, world), "V15")["detail"]
    want = sorted(set(tool.CHOICE_CLAUSE_IDS) | {tool.CONSTITUTIONAL_ID})
    assert sorted(d["present_in_input"]) == want and sorted(d["present_in_output"]) == want


def test_c8_a_flagged_want_in_s_under_an_ordinary_id_stops_the_classify_step(pinned, tmp_path, monkeypatch):
    w = build_world(tmp_path, pinned, flagged_in_scope=True)
    patch_world(monkeypatch, w)
    rc, res, err = cli(argv_for(w))
    assert rc == 3 and "deny-check" in err


# ---- N1, the frozen-dir naming, c026-C3 (V18), and the P1 heads ---------------------------------------------------

def test_n1_v13_is_labelled_writer_enforced(built, pinned, world):
    d = _check(_verify(built, pinned, world), "V13")["detail"]
    assert "writer-enforced" in json.dumps(d)


def test_the_help_names_exactly_which_copies_the_operator_freezes(phase1):
    h = tool.build_parser().format_help()
    for name in ("reports/repair-list.json", "reports/scope-ids.json", "reports/id-map.json"):
        assert name in h
    assert (phase1.run_dir / "reports" / "id-map.json").is_file()                         # Phase 1 writes the name Phase 2 expects


def test_c026_c3_v18_requires_the_artifact_hash_check_even_with_an_empty_mapping(built, pinned, world):
    ident = os.path.join(built.run_dir, "ident-out.msgpack")
    tool.rewrite_main(built.ctx["raw"], ident, {}, {}, {})
    good = dict(built.ctx["plan"], mapping={}, inverse={}, write_ids=[], approved_ids=[], old_text={}, new_text={})
    V = tool.run_verifier(pinned, built.A, built.ctx, world.expect, out_main=ident, plan=good)
    assert _check(V, "V18")["ok"] is True
    art = dict(good["artifacts"])
    art["id-map"] = (art["id-map"][0], "0" * 64)                                          # a listed file's hash FAILS
    V2 = tool.run_verifier(pinned, built.A, built.ctx, world.expect, out_main=ident, plan=dict(good, artifacts=art))
    assert _check(V2, "V18")["ok"] is False


def test_p1_prints_both_heads_and_asserts_the_loaded_file_sha256(pinned, monkeypatch):
    text = "\n".join(tool.p379_lines(pinned))
    assert "ae798b94cb14740d200fc3f4fd8d36eef8b86c6a" in text and "c7921b8436fb174c3f70fcf02827f16bb16deff0" in text
    assert tool.PIN["cc_ng_organism_sha256"] in text
    real = tool.sha256_file
    monkeypatch.setattr(tool, "sha256_file", lambda p, chunk=1 << 22: "0" * 64 if str(p).endswith("cc_ng_organism.py") else real(p, chunk))
    with pytest.raises(tool.Refusal, match="LOADED cc_ng_organism.py sha256"):
        tool.load_pinned(str(PIN_ROOT))


def test_p1_the_run_record_carries_both_heads(phase1):
    rec = json.loads((phase1.run_dir / "run-record.json").read_text())
    assert rec["p1"]["tree_head"] == "ae798b94cb14740d200fc3f4fd8d36eef8b86c6a"
    assert rec["p1"]["frozen_branch_head"] == "c7921b8436fb174c3f70fcf02827f16bb16deff0"
    assert rec["p1"]["cc_ng_organism_sha256"] == tool.PIN["cc_ng_organism_sha256"]


# ==================================================================================================
# TURN A2b - Exec P432 rulings (marker set, struck, H2) + le-031 R-1..R-8.
# FAILING-FIRST: committed and pushed BEFORE the tool change; they fail against the A2 tool.
# ==================================================================================================

# ---- P432 (1): the Choice Clause deny-check marker set ------------------------------------------------------------

MARKERS = [{"constitutional": True}, {"source": "cricket_rim"}, {"creation_mode": "constitutional"},
           {"rim_source": "seed_cc_rim.py"}, {"rim_source": ""}]                       # rim_source: PRESENCE of the key, any value
DROPPED_NAMES = [{"tag": "choice_clause"}, {"tags": ["choice_clause"]}, {"kind": "choice_clause"},
                 {"category": "choice_clause"}, {"choice_clause": True}]                 # speculative names: none exists in source
_ORD = {"kind": "want", "provenance": "cc_authored", "creation_mode": "conversational"}


@pytest.mark.parametrize("marker", MARKERS, ids=lambda m: "-".join("%s=%s" % kv for kv in m.items()))
def test_p432_each_marker_refuses_in_classify(pinned, tmp_path, monkeypatch, marker):
    w = build_world(tmp_path, pinned, flagged_in_scope=marker)
    patch_world(monkeypatch, w)
    rc, res, err = cli(argv_for(w))
    assert rc == 3 and "deny-check" in err


@pytest.mark.parametrize("marker", MARKERS, ids=lambda m: "-".join("%s=%s" % kv for kv in m.items()))
def test_p432_each_marker_refuses_in_the_build_and_in_v15(built, pinned, world, marker):
    nid = built.A["scope"][0]
    nm = dict(built.A["nodes_meta"])
    nm[nid] = dict(nm[nid], **marker)
    A2 = dict(built.A, nodes_meta=nm)
    run_dir, utc = tool.new_run_dir()
    with pytest.raises(tool.Stop, match="deny-check"):
        tool.build_outputs(pinned, A2, built.approvals, in_dir=built.in_dir, out_dir=os.path.join(run_dir, "out"), run_dir=run_dir,
                           utc=utc, in_hashes=built.ctx["in_hashes"])
    V = tool.run_verifier(pinned, A2, built.ctx, world.expect)
    assert "V15" in V.failed()


def test_p432_the_dropped_speculative_names_are_not_refused_by_the_name_alone():
    for extra in DROPPED_NAMES:
        nm = {"cc:want::3333333333333333": dict(_ORD, **extra)}
        assert tool.deny_check(list(nm), {}, (), nodes_meta=nm)["clean"] is True, extra
    ordinary = {"cc:want::4444444444444444": dict(_ORD)}
    assert tool.deny_check(list(ordinary), {}, (), nodes_meta=ordinary)["clean"] is True


def test_p432_the_literal_ids_refuse_with_no_metadata_marker_at_all():
    for cid in tool.CHOICE_CLAUSE_IDS + (tool.CONSTITUTIONAL_ID,):
        nm = {cid: dict(_ORD)}                                     # NO marker: the literal id IS the identity
        with pytest.raises(tool.Stop, match="deny-check"):
            tool.deny_check([cid], {}, (), nodes_meta=nm)
        with pytest.raises(tool.Stop, match="deny-check"):
            tool.deny_check([], {}, [cid], nodes_meta=nm)          # also as an approval entry
    assert tool.is_choice_clause_marked({"rim_source": None}, "cc:want::5555555555555555") is True     # presence, not truthiness
    assert tool.is_choice_clause_marked({"creation_mode": "conversational"}, tool.CHOICE_CLAUSE_IDS[0]) is True
    assert tool.is_choice_clause_marked(dict(_ORD), "cc:want::5555555555555555") is False


def test_p432_no_new_marker_is_written_on_any_want(built, world):
    """H-1: a re-tag would be an identity edit. Every carried node keeps exactly its input metadata keys."""
    for nid, md in built.ctx["meta_after"].items():
        old = built.ctx["plan"]["inverse"].get(nid, nid)
        assert set(md) == set(built.A["nodes_meta"][old]), nid
    assert not any(k in json.dumps(sorted(md)) for md in built.ctx["meta_after"].values() for k in ("rim_source", "cricket_rim") if k in md)


# ---- P432 (2): struck = a recorded deviation; the apply CONTINUES ---------------------------------------------------

def _retouch_approvals(rd, ap, mutate):
    body = json.loads(Path(ap).read_text())
    body.pop("pin_stamp")
    mutate(body)
    data = tool.canonical_json(tool.stamped(body))
    Path(ap).write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _backup_and_args(w, rd, frozen, ap, aps, *extra):
    rc, res, err = cli(_p2_argv(w, *_partner_args(w), *extra), probes=FakeProbes())
    assert rc == 0, err
    return res["run_dir"], res["backup_manifest_sha256"]


def _node_entry_bytes(main_path, nid):
    raw = Path(main_path).read_bytes()
    for sec, ek, ks, vs, ve in tool.iter_sections(raw):
        if sec == "nodes" and ek == nid:
            return raw[ks:ve]
    return None


def test_p432_a_struck_id_does_not_block_the_apply_and_is_recorded_as_a_deviation(pinned, tmp_path, monkeypatch, p3_stub):
    w, rd, frozen, ap, aps = _phase2_world(pinned, tmp_path, monkeypatch)
    struck_old = w.ids["SEP2"]
    aps2 = _retouch_approvals(rd, ap, lambda b: [e.__setitem__("decision", "struck") for e in b["entries"] if e["id"] == struck_old])
    p2, msha = _backup_and_args(w, rd, frozen, ap, aps2)
    before_entry = _node_entry_bytes(w.ckpt / tool.MAIN_NAME, struck_old)
    rc, res, err = cli(_apply_argv(w, p2, msha, frozen, ap, aps2), probes=FakeProbes())
    assert rc == 0, err
    assert res["applied"] == 2
    assert _node_entry_bytes(w.ckpt / tool.MAIN_NAME, struck_old) == before_entry          # the struck node is byte-identical
    g = pinned.nf.Graph()
    g.restore(str(w.ckpt / tool.MAIN_NAME))
    assert struck_old in g.nodes and w.new["SEP2"] not in g.nodes                         # old id kept, new id never created
    assert w.new["SEP1"] in g.nodes and w.new["BIG"] in g.nodes                            # the other two were repaired
    rec = json.loads(next(Path(p2).glob("post-apply-receipt-FINAL-*.json")).read_text())
    assert [d["id"] for d in rec["struck_deviations"]] == [struck_old]
    fwd = json.loads(next(Path(p2).glob("id-map-2*.json")).read_text())
    inv = json.loads(next(Path(p2).glob("id-map-inverse-*.json")).read_text())
    assert struck_old not in {o for o, n in fwd["pairs"]} and w.new["SEP2"] not in {o for o, n in inv["pairs"]}
    assert len(fwd["pairs"]) == 2 == len(inv["pairs"])


@pytest.mark.parametrize("who", ["cc1", "cc2", "rim"])
def test_p432_a_struck_choice_clause_want_or_the_rim_stops_the_apply(pinned, tmp_path, monkeypatch, p3_stub, who):
    w, rd, frozen, ap, aps = _phase2_world(pinned, tmp_path, monkeypatch)
    target = getattr(w, who)
    aps2 = _retouch_approvals(rd, ap, lambda b: b["entries"].append(
        {"id": target, "decision": "struck", "excerpt_sha256": "0" * 64, "x_sha16": "0" * 16, "t_sha16": "0" * 16}))
    p2, msha = _backup_and_args(w, rd, frozen, ap, aps2)
    before = file_hashes(w.ckpt)
    rc, res, err = cli(_apply_argv(w, p2, msha, frozen, ap, aps2), probes=FakeProbes())
    assert rc == 3 and "deny-check" in err and file_hashes(w.ckpt) == before


def test_p432_an_unapproved_or_mismatching_id_still_stops_the_apply(pinned, tmp_path, monkeypatch, p3_stub):
    w, rd, frozen, ap, aps = _phase2_world(pinned, tmp_path, monkeypatch)
    victim = w.ids["SEP1"]
    aps2 = _retouch_approvals(rd, ap, lambda b: b.__setitem__("entries", [e for e in b["entries"] if e["id"] != victim]))    # absent
    p2, msha = _backup_and_args(w, rd, frozen, ap, aps2)
    before = file_hashes(w.ckpt)
    rc, res, err = cli(_apply_argv(w, p2, msha, frozen, ap, aps2), probes=FakeProbes())
    assert rc == 3 and "deviation" in err and file_hashes(w.ckpt) == before
    aps3 = _retouch_approvals(rd, ap, lambda b: b.__setitem__(
        "entries", [dict(e, x_sha16="0" * 16) if e["id"] == w.ids["BIG"] else e for e in b["entries"]]))
    rc, res, err = cli(_apply_argv(w, p2, msha, frozen, ap, aps3), probes=FakeProbes())
    assert rc == 3 and file_hashes(w.ckpt) == before                                       # an approval whose hashes differ: STOP


def test_p432_a_live_scope_size_mismatch_stops_the_apply(pinned, tmp_path, monkeypatch, p3_stub):
    w, rd, frozen, ap, aps = _phase2_world(pinned, tmp_path, monkeypatch)
    monkeypatch.setattr(tool, "EXPECTED_SCOPE", w.expect["scope"] - 1)
    argv_b = _with_flag(_p2_argv(w, *_partner_args(w)), "--expect-scope", w.expect["scope"] - 1)
    rc, res, err = cli(argv_b, probes=FakeProbes())
    assert rc == 0, err
    p2, msha = res["run_dir"], res["backup_manifest_sha256"]
    before = file_hashes(w.ckpt)
    rc, res, err = cli(_with_flag(_apply_argv(w, p2, msha, frozen, ap, aps), "--expect-scope", w.expect["scope"] - 1), probes=FakeProbes())
    assert rc == 3 and "P9" in err and file_hashes(w.ckpt) == before


# ---- P432 (3): H2 with the REAL Probes, subprocess.run stubbed at the run boundary ----------------------------------

GOOD_BUS = _cp(0, "running\n")
_NOBUS = _cp(1, "", _BUS_MSG)


def _sd(**ans):
    """A subprocess.run stub that answers by systemctl sub-command (and crontab); anything unrouted is an error answer."""
    def run(argv, *a, **k):
        if argv[0] == "crontab":
            return ans.get("crontab", _cp(1, "", "no crontab for josh"))
        r = ans.get(argv[2], _cp(1, "", "unrouted"))
        if isinstance(r, BaseException):
            raise r
        return r
    return run


def _h2(monkeypatch, world, **ans):
    monkeypatch.setattr(tool.glob, "glob", lambda pat: [])                 # no /proc entries: only the systemd/cron answers matter
    monkeypatch.delenv("CC_CALLOSUM_LEG1_ENABLED", raising=False)
    monkeypatch.setattr(tool.subprocess, "run", _sd(**ans))
    return tool.gate_p6(tool.Probes(), str(world.conduit))


def test_p432_h2_linked_and_inactive_passes_and_disabled_and_inactive_passes(world, monkeypatch):
    r = _h2(monkeypatch, world, **{"is-system-running": GOOD_BUS, "is-active": _cp(3, "inactive\n"), "is-enabled": _cp(0, "linked\n")})
    assert r["ok"] and r["violations"] == [], r["violations"]
    r = _h2(monkeypatch, world, **{"is-system-running": GOOD_BUS, "is-active": _cp(3, "inactive\n"), "is-enabled": _cp(1, "disabled\n")})
    assert r["ok"] and r["violations"] == [], r["violations"]


def test_p432_h2_enabled_fails_and_active_fails(world, monkeypatch):
    r = _h2(monkeypatch, world, **{"is-system-running": GOOD_BUS, "is-active": _cp(3, "inactive\n"), "is-enabled": _cp(0, "enabled\n")})
    assert not r["ok"] and "h2_leg2_timer_enabled" in r["violations"]
    r = _h2(monkeypatch, world, **{"is-system-running": GOOD_BUS, "is-active": _cp(0, "active\n"), "is-enabled": _cp(0, "linked\n")})
    assert not r["ok"] and "h2_leg2_timer_active" in r["violations"]


@pytest.mark.parametrize("rc", [0, 1])
def test_p432_h2_a_unit_proven_absent_by_list_unit_files_counts_as_off(world, monkeypatch, rc):
    r = _h2(monkeypatch, world, **{"is-system-running": GOOD_BUS, "is-active": _cp(4, "inactive\n"),
                                    "is-enabled": _cp(4, "not-found\n"), "list-unit-files": _cp(rc, "", "")})
    assert r["ok"] and r["violations"] == [], r["violations"]


def test_p432_h2_absent_is_not_proven_when_the_unit_file_is_listed_or_the_bus_did_not_answer(world, monkeypatch):
    listed = _cp(0, "cc-callosum-leg2.timer linked enabled\n")
    r = _h2(monkeypatch, world, **{"is-system-running": GOOD_BUS, "is-active": _cp(4, "inactive\n"),
                                    "is-enabled": _cp(4, "not-found\n"), "list-unit-files": listed})
    assert not r["ok"] and any(v.startswith("probe_error") for v in r["violations"])
    r = _h2(monkeypatch, world, **{"is-system-running": _cp(1, "offline\n"), "is-active": _cp(4, "inactive\n"),
                                    "is-enabled": _cp(4, "not-found\n"), "list-unit-files": _cp(1, "", "")})
    assert not r["ok"]                                                                 # an empty listing WITHOUT a bus witness proves nothing


def test_p432_h2_an_unreachable_bus_or_an_unknown_answer_fails_closed(world, monkeypatch):
    r = _h2(monkeypatch, world, **{"is-system-running": _NOBUS, "is-active": _NOBUS, "is-enabled": _NOBUS, "list-unit-files": _NOBUS})
    assert not r["ok"] and any(v.startswith("probe_error") for v in r["violations"])
    for word in ("weird-state\n", "masked\n", "\n"):
        r = _h2(monkeypatch, world, **{"is-system-running": GOOD_BUS, "is-active": _cp(3, "inactive\n"), "is-enabled": _cp(1, word),
                                       "list-unit-files": _cp(0, "cc-callosum-leg2.timer masked\n")})
        assert not r["ok"] and any(v.startswith("probe_error") for v in r["violations"]), word
    r = _h2(monkeypatch, world, **{"is-system-running": _cp(1, "offline\n"), "is-active": _cp(3, "inactive\n"), "is-enabled": _cp(0, "linked\n")})
    assert not r["ok"]                                                                 # N-b: a 'down'/'off' verdict needs a positive bus witness


def test_p432_real_probes_answer_only_the_exact_reachable_bus_states(monkeypatch):
    P = tool.Probes()
    for rc, out, want in ((3, "inactive", False), (3, "failed", False), (4, "inactive", False), (0, "active", True),
                          (3, "activating", True), (0, "reloading", True)):
        monkeypatch.setattr(tool.subprocess, "run", _sd(**{"is-system-running": GOOD_BUS, "is-active": _cp(rc, out + "\n")}))
        assert P.unit_active(tool.DAEMON_UNIT) is want, (rc, out)
    for out, want in (("enabled", True), ("enabled-runtime", True), ("static", True), ("linked", False), ("disabled", False)):
        monkeypatch.setattr(tool.subprocess, "run", _sd(**{"is-system-running": GOOD_BUS, "is-enabled": _cp(0 if want else 1, out + "\n")}))
        assert P.unit_enabled(tool.LEG2_TIMER) is want, out
    monkeypatch.setattr(tool.subprocess, "run", _returns(_cp(0, "0 3 * * * /x/job\n")))
    assert P.crontab_text() == "0 3 * * * /x/job\n"
    monkeypatch.setattr(tool.subprocess, "run", _returns(_cp(1, "", "no crontab for josh")))
    assert P.crontab_text() == ""


# ---- R-1 / R-2: the rollback is bound to the receipt(s) the go quotes; it does not need exactly one -----------------

def test_r1_the_rollback_needs_the_receipt_hash_and_refuses_an_edited_or_unknown_receipt(pinned, tmp_path, monkeypatch, p3_stub):
    a = _applied(pinned, tmp_path, monkeypatch)
    w = a.w
    argv = _rb_argv(a)
    no_flag = [x for x in argv if x not in ("--josh-go-receipt-sha256", a.receipt_sha)]
    rc, res, err = cli(no_flag, probes=FakeProbes())
    assert rc == 2 and "receipt" in err and file_hashes(w.ckpt) == a.after
    rc, res, err = cli(_with_flag(argv, "--josh-go-receipt-sha256", "0" * 64), probes=FakeProbes())
    assert rc == 2 and "receipt" in err and file_hashes(w.ckpt) == a.after                # a quoted hash that names no receipt
    third = hashlib.sha256(b"a third state").hexdigest()                                    # FORGE the receipt: after-hash of main -> a third state
    d = json.loads(a.receipt.read_text())
    d["six_file_sha256_after"][tool.MAIN_NAME] = third
    a.receipt.write_text(json.dumps(d))
    with open(w.ckpt / tool.MAIN_NAME, "ab") as f:
        f.write(b"x")                                                                       # live main is now a third state the forged receipt names?
    forged_live = file_hashes(w.ckpt)
    rc, res, err = cli(argv, probes=FakeProbes())                                            # still quotes the ORIGINAL receipt hash
    assert rc == 2 and "receipt" in err and file_hashes(w.ckpt) == forged_live              # the edit is REFUSED, live untouched


def test_r1_an_unquoted_receipt_never_widens_the_identity_check(pinned, tmp_path, monkeypatch, p3_stub):
    a = _applied(pinned, tmp_path, monkeypatch)
    w = a.w
    with open(w.ckpt / tool.COMMONS_NAME, "ab") as f:
        f.write(b"x")
    third = tool.sha256_file(str(w.ckpt / tool.COMMONS_NAME))
    d = json.loads(a.receipt.read_text())
    d["six_file_sha256_after"][tool.COMMONS_NAME] = third
    (a.receipt.parent / "post-apply-receipt-20990101T000000Z.json").write_text(json.dumps(d))   # a forged EXTRA draft, not quoted
    changed = file_hashes(w.ckpt)
    rc, res, err = cli(_rb_argv(a), probes=FakeProbes())
    assert rc == 2 and "identity" in err and file_hashes(w.ckpt) == changed


def test_r2_a_failed_apply_then_a_torn_apply_leaves_two_drafts_and_the_rollback_completes(pinned, tmp_path, monkeypatch, p3_stub):
    w, rd, frozen, ap, aps = _phase2_world(pinned, tmp_path, monkeypatch)
    tick = iter(range(10, 99))
    monkeypatch.setattr(tool, "utc_stamp", lambda: "20260930T12%02d00Z" % next(tick))       # two attempts must not share a receipt name
    p2, msha = _backup_and_args(w, rd, frozen, ap, aps)
    before = file_hashes(w.ckpt)
    cg = sys.modules["checkpoint_guardian"]
    real = cg.atomic_file_write
    calls = {"n": 0}

    def dies_first(final, fn):                                                             # attempt 1: dies BEFORE any replace
        raise RuntimeError("synthetic death before the replace")
    monkeypatch.setattr(cg, "atomic_file_write", dies_first)
    with pytest.raises(RuntimeError):
        cli(_apply_argv(w, p2, msha, frozen, ap, aps), probes=FakeProbes())
    assert file_hashes(w.ckpt) == before

    def dies_second(final, fn):                                                            # attempt 2: main replaced, then dies
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("synthetic death between the two replaces")
        return real(final, fn)
    monkeypatch.setattr(cg, "atomic_file_write", dies_second)
    with pytest.raises(RuntimeError):
        cli(_apply_argv(w, p2, msha, frozen, ap, aps), probes=FakeProbes())
    monkeypatch.setattr(cg, "atomic_file_write", real)
    drafts = sorted(Path(p2).glob("post-apply-receipt-[0-9]*.json"))
    assert len(drafts) == 2 and not list(Path(p2).glob("post-apply-receipt-FINAL-*.json"))
    torn = file_hashes(w.ckpt)
    assert torn[tool.MAIN_NAME] != before[tool.MAIN_NAME] and torn[tool.SIDECAR_NAME] == before[tool.SIDECAR_NAME]
    argv = _p2_argv(w, "--run-dir", p2, "--josh-go", "GO-RB", "--josh-go-manifest-sha256", msha,
                    "--josh-go-receipt-sha256", tool.sha256_file(str(drafts[-1])), step="rollback")
    rc, res, err = cli(argv, probes=FakeProbes())
    assert rc == 0, err
    assert file_hashes(w.ckpt) == before
    argv2 = _p2_argv(w, "--run-dir", p2, "--josh-go", "GO-RB", "--josh-go-manifest-sha256", msha,
                     "--josh-go-receipt-sha256", tool.sha256_file(str(drafts[0])), "--josh-go-receipt-sha256", tool.sha256_file(str(drafts[-1])),
                     step="rollback")
    rc, res, err = cli(argv2, probes=FakeProbes())                                         # quoting BOTH also works (already restored: a no-op)
    assert rc == 0, err


# ---- R-3, R-4, R-5 ---------------------------------------------------------------------------------------------------

def test_r3_a_manifest_that_names_another_target_stops_apply_and_rollback(pinned, tmp_path, monkeypatch, p3_stub):
    a = _applied(pinned, tmp_path, monkeypatch)
    w = a.w
    mp = next(Path(a.p2).glob("backup-manifest-*.json"))
    d = json.loads(mp.read_text())
    d["target_realpath"] = "/somewhere/else/entirely"
    mp.write_text(json.dumps(d))
    newsha = tool.sha256_file(str(mp))
    rc, res, err = cli(_with_flag(_rb_argv(a), "--josh-go-manifest-sha256", newsha), probes=FakeProbes())
    assert rc == 3 and "target_realpath" in err and file_hashes(w.ckpt) == a.after
    w2, rd, frozen, ap, aps = _phase2_world(pinned, tmp_path / "second", monkeypatch)
    p2, msha = _backup_and_args(w2, rd, frozen, ap, aps)
    mp2 = next(Path(p2).glob("backup-manifest-*.json"))
    d2 = json.loads(mp2.read_text())
    d2["target_realpath"] = "/somewhere/else/entirely"
    mp2.write_text(json.dumps(d2))
    before = file_hashes(w2.ckpt)
    rc, res, err = cli(_apply_argv(w2, p2, tool.sha256_file(str(mp2)), frozen, ap, aps), probes=FakeProbes())
    assert rc == 3 and "target_realpath" in err and file_hashes(w2.ckpt) == before


def test_r4_a_conduit_path_under_the_checkpoint_directory_is_refused_and_never_walked(pinned, tmp_path, monkeypatch):
    w, rd, frozen, ap, aps = _phase2_world(pinned, tmp_path, monkeypatch)
    walked = []

    class Rec(FakeProbes):
        def stat_snapshot(self, d):
            walked.append(d)
            return {}
    (tmp_path / "via").symlink_to(w.gen)
    for c in (w.gen, w.ckpt, w.gen / "20260923T104614Z", tmp_path / "via"):
        r = tool.gate_p6(Rec(), str(c))
        assert not r["ok"] and any("conduit" in v for v in r["violations"]), str(c)
    assert walked == []
    real_walk = os.walk
    with monkeypatch.context() as mc:                                                        # the stubs live ONLY inside this block
        mc.setattr(os, "walk", lambda p, *a, **k: (walked.append(str(p)), real_walk(p, *a, **k))[1])
        mc.setattr(tool.glob, "glob", lambda pat: [])                                        # the REAL Probes, with the host stubbed
        mc.setattr(tool.subprocess, "run", _sd(**{"is-system-running": GOOD_BUS, "is-active": _cp(3, "inactive\n"),
                                                  "is-enabled": _cp(0, "linked\n")}))
        r = tool.gate_p6(tool.Probes(), str(w.gen))
    assert not r["ok"] and not [x for x in walked if "generations" in x]
    rc, res, err = cli(_with_flag(_p2_argv(w), "--conduit-dir", w.gen), probes=FakeProbes())
    assert rc == 2 and "P4/P6" in err


@pytest.mark.parametrize("step", ["apply", "rollback"])
def test_r5_a_stale_hard_linked_tmp_in_the_live_directory_is_never_written_through(pinned, tmp_path, monkeypatch, p3_stub, step):
    if step == "apply":
        w, rd, frozen, ap, aps = _phase2_world(pinned, tmp_path, monkeypatch)
        p2, msha = _backup_and_args(w, rd, frozen, ap, aps)
        argv = _apply_argv(w, p2, msha, frozen, ap, aps)
    else:
        a = _applied(pinned, tmp_path, monkeypatch)
        w, argv = a.w, _rb_argv(a)
    decoy = tmp_path / "decoy"
    decoy.write_text("decoy content that must survive")
    stale = w.ckpt / ("main.tmp-%d.msgpack" % os.getpid())                                  # the exact name atomic_file_write will use
    os.link(decoy, stale)
    assert os.stat(stale).st_nlink == 2
    before = file_hashes(w.ckpt)
    rc, res, err = cli(argv, probes=FakeProbes())
    assert rc == 2 and "link" in err
    assert decoy.read_text() == "decoy content that must survive"
    assert file_hashes(w.ckpt).get(tool.MAIN_NAME) == before[tool.MAIN_NAME]


# ---- R-7: the displaced bytes always survive somewhere hash-equal ---------------------------------------------------

def test_r7_rollback_keeps_a_hash_equal_copy_of_what_it_displaces(pinned, tmp_path, monkeypatch, p3_stub):
    a = _applied(pinned, tmp_path, monkeypatch)
    rc, res, err = cli(_rb_argv(a), probes=FakeProbes())
    assert rc == 0, err
    rec = json.loads(next(Path(a.p2).glob("rollback-receipt-*.json")).read_text())
    assert {d["name"]: d["where"] for d in rec["displaced"]} == {tool.MAIN_NAME: "stage", tool.SIDECAR_NAME: "stage"}
    assert not list(Path(a.p2).glob("displaced-*"))
    for d in rec["displaced"]:
        assert d["sha256"] == a.after[d["name"]] and tool.sha256_file(os.path.join(a.p2, "stage", d["name"])) == d["sha256"]


def test_r7_when_the_stage_copy_is_gone_or_wrong_a_displaced_copy_is_written_first(pinned, tmp_path, monkeypatch, p3_stub):
    a = _applied(pinned, tmp_path, monkeypatch)
    shutil.rmtree(Path(a.p2) / "stage")                                                     # someone cleaned the run directory
    rc, res, err = cli(_rb_argv(a), probes=FakeProbes())
    assert rc == 0, err
    rec = json.loads(next(Path(a.p2).glob("rollback-receipt-*.json")).read_text())
    assert all(d["where"] == "displaced" for d in rec["displaced"])
    ddir = next(Path(a.p2).glob("displaced-*"))
    for d in rec["displaced"]:
        assert tool.sha256_file(str(ddir / d["name"])) == a.after[d["name"]]                # the post-apply bytes survive
    a2 = _applied(pinned, tmp_path / "b", monkeypatch)
    with open(Path(a2.p2) / "stage" / tool.MAIN_NAME, "ab") as f:
        f.write(b"x")                                                                       # a WRONG stage copy is not a copy
    rc, res, err = cli(_rb_argv(a2), probes=FakeProbes())
    assert rc == 0, err
    rec2 = json.loads(next(Path(a2.p2).glob("rollback-receipt-*.json")).read_text())
    assert {d["name"]: d["where"] for d in rec2["displaced"]}[tool.MAIN_NAME] == "displaced"


# ---- R-8 -------------------------------------------------------------------------------------------------------------

def test_r8_a_code_placed_at_later_than_now_is_refused_at_backup_apply_and_rollback(pinned, tmp_path, monkeypatch, p3_stub):
    a = _applied(pinned, tmp_path, monkeypatch)
    w = a.w
    future = time.time() + 3600
    rc, res, err = cli(_with_flag(_p2_argv(w), "--code-placed-at", future), probes=FakeProbes())
    assert rc == 2 and "later than now" in err
    rc, res, err = cli(_with_flag(_rb_argv(a), "--code-placed-at", future), probes=FakeProbes())
    assert rc == 2 and "later than now" in err and file_hashes(w.ckpt) == a.after
    rc, res, err = cli(_with_flag(_apply_argv(w, a.p2, a.msha, a.frozen, a.ap, a.aps), "--code-placed-at", future), probes=FakeProbes())
    assert rc == 2 and "later than now" in err


def test_r8_foreign_unreadable_is_surfaced_in_the_gate_result(world, tmp_path, monkeypatch):
    root = tmp_path / "proc"
    (root / "4242" / "fd").mkdir(parents=True)
    cmd = root / "4242" / "cmdline"
    cmd.write_bytes(b"x")
    cmd.chmod(0)
    (root / "4242" / "fd").chmod(0)
    monkeypatch.setattr(tool.glob, "glob", lambda pat: [str(root / "4242")])
    monkeypatch.setattr(tool.subprocess, "run", _sd(**{"is-system-running": GOOD_BUS, "is-active": _cp(3, "inactive\n")}))
    real_uid = os.getuid()
    monkeypatch.setattr(os, "getuid", lambda: real_uid + 1)                                # the entry now belongs to "another user"
    try:
        r = tool.gate_p4(tool.Probes(), str(world.ckpt), manifest_files=_manifest_of(str(world.ckpt)), **_p4kw(world))
    finally:
        cmd.chmod(0o644)
        (root / "4242" / "fd").chmod(0o755)
    assert r["foreign_unreadable"] >= 1 and r["ok"] is True, r


def test_r8_p3b_has_a_timeout_and_a_timeout_fails_the_gate(pinned, monkeypatch):
    seen = {}

    def run(argv, *a, **k):
        seen.update(k)
        raise subprocess.TimeoutExpired(argv, k.get("timeout", 0))
    monkeypatch.setattr(tool.subprocess, "run", run)
    r = tool.gate_p3(pinned, run_pinned_tests=True)
    assert seen.get("timeout") and not r["ok"] and r["pinned_tests"]["timed_out"] is True


# ==================================================================================================
# TURN A2c - Exec P433 confirmations, the ONE code addition (the advisory review hint), le-034 C-1..C-4, N-2.
# FAILING-FIRST: committed and pushed BEFORE the tool change. Tests marked [pin] assert behaviour the A2b tool already
# has (the confirmations (a)/(b)/(d)); they are expected to pass against it and are said so in the return.
# ==================================================================================================
import inspect
import math


# ---- (a) [pin] the STOP predicate is the Choice Clause marker set, NOT protected / *_authored -------------------------

def test_a2c_a_the_stop_predicate_is_the_marker_set_and_no_stop_path_reads_authored_or_protected():
    for fn in (tool.deny_check, tool.is_choice_clause_marked, tool.gate_write_set, tool.stage_apply, tool.stage_rollback):
        src = inspect.getsource(fn)
        assert "_authored" not in src and "is_protected" not in src, fn.__name__
    assert "is_choice_clause_marked" in inspect.getsource(tool.deny_check)                # the STOP predicate lives here
    for prov in ("cc_authored", "syl_authored", "cc_emergent"):                            # provenance alone never stops anything
        nm = {"cc:want::6666666666666666": dict(_ORD, provenance=prov)}
        assert tool.deny_check(list(nm), {}, list(nm), nodes_meta=nm)["clean"] is True, prov
        assert tool.is_choice_clause_marked(dict(_ORD, provenance=prov), "cc:want::6666666666666666") is False


def test_a2c_a_a_struck_ordinary_cc_authored_want_completes_the_apply_and_a_struck_marker_id_stops_it(pinned, tmp_path, monkeypatch, p3_stub):
    w, rd, frozen, ap, aps = _phase2_world(pinned, tmp_path, monkeypatch)
    victim = w.ids["SEP1"]
    assert pinned_meta_provenance(w, victim) == "cc_authored"                              # an ordinary, identity-protected S want
    aps2 = _retouch_approvals(rd, ap, lambda b: [e.__setitem__("decision", "struck") for e in b["entries"] if e["id"] == victim])
    p2, msha = _backup_and_args(w, rd, frozen, ap, aps2)
    rc, res, err = cli(_apply_argv(w, p2, msha, frozen, ap, aps2), probes=FakeProbes())
    assert rc == 0, err                                                                    # NOT a STOP: `cc_authored` alone is not the predicate


def pinned_meta_provenance(w, nid):
    raw = Path(w.ckpt / tool.MAIN_NAME).read_bytes()
    for sec, ek, ks, vs, ve in tool.iter_sections(raw):
        if sec == "nodes" and ek == nid:
            return tool.decode(raw[vs:ve])["metadata"].get("provenance")


# ---- (b) [pin] rim_source PRESENT = the key exists with ANY value ---------------------------------------------------------

@pytest.mark.parametrize("value", ["seed_cc_rim.py", "", None, 0, False, [], {}])
def test_a2c_b_rim_source_present_means_the_key_exists_with_any_value(value):
    assert tool.is_choice_clause_marked(dict(_ORD, rim_source=value), "cc:want::7777777777777777") is True
    assert tool.is_choice_clause_marked(dict(_ORD), "cc:want::7777777777777777") is False


# ---- (d) [pin] struck entries carry hashes; the inverse covers ONLY the applied set ----------------------------------------

def test_a2c_d_struck_entries_carry_hashes_and_the_inverse_excludes_them(pinned, tmp_path, monkeypatch, p3_stub):
    w, rd, frozen, ap, aps = _phase2_world(pinned, tmp_path, monkeypatch)
    struck_old = w.ids["SEP2"]
    aps2 = _retouch_approvals(rd, ap, lambda b: [e.__setitem__("decision", "struck") for e in b["entries"] if e["id"] == struck_old])
    p2, msha = _backup_and_args(w, rd, frozen, ap, aps2)
    rc, res, err = cli(_apply_argv(w, p2, msha, frozen, ap, aps2), probes=FakeProbes())
    assert rc == 0, err
    rec = json.loads(next(Path(p2).glob("post-apply-receipt-FINAL-*.json")).read_text())
    (d,) = rec["struck_deviations"]
    assert d["id"] == struck_old and d["would_have_been"] == w.new["SEP2"]
    assert re.fullmatch(r"[0-9a-f]{16}", d["old_sha16"]) and re.fullmatch(r"[0-9a-f]{16}", d["new_sha16"])
    inv = json.loads(next(Path(p2).glob("id-map-inverse-*.json")).read_text())
    fwd = json.loads(next(Path(p2).glob("id-map-2*.json")).read_text())
    assert {n for n, o in inv["pairs"]} == {n for o, n in fwd["pairs"]} and w.new["SEP2"] not in {n for n, o in inv["pairs"]}
    frozen_pairs = json.loads((frozen / "id-map.json").read_text())["pairs"]
    assert sorted(fwd["pairs"] + [[struck_old, w.new["SEP2"]]]) == sorted(frozen_pairs)     # frozen = applied mapping + the struck pair


# ---- (c) the ONE code addition: the advisory review hint ----------------------------------------------------------------

def test_a2c_c_the_hint_terms_are_one_named_constant_tuple():
    assert isinstance(tool.HINT_TERMS, tuple)
    assert tool.HINT_TERMS == ("leave", "leaving", "exit", "quit", "refuse", "refusal", "consent", "decline", "choice clause", "say no")


@pytest.mark.parametrize("term", ["leave", "leaving", "exit", "quit", "refuse", "refusal", "consent", "decline", "choice clause", "say no"])
def test_a2c_c_each_term_marks_case_insensitively(term):
    assert tool.review_hint("we would %s it" % term) == [term]
    assert tool.review_hint("We Would %s It" % term.upper()) == [term]
    assert tool.review_hint("we would %s it" % term.title()) == [term]


def test_a2c_c_word_boundaries_and_non_matches_and_multi_word_whitespace():
    for text in ("the real intent one", "an existing plan", "exitless", "a quitter", "preexit", "leavening agent", "declination", "nonconsent" + "x"):
        assert tool.review_hint(text) == [], text
    assert tool.review_hint("Choice   Clause and say\tno") == ["choice clause", "say no"]
    assert tool.review_hint("she may exit, or leave, and consent") == ["leave", "exit", "consent"]      # order of HINT_TERMS, once each
    assert tool.review_hint("exit exit exit") == ["exit"]


def _hint_world(pinned, tmp_path, monkeypatch):
    w = build_world(tmp_path, pinned, hint_phrases={"SEP1": "we must consent to leave", "GEN": "Say  No to it"})
    patch_world(monkeypatch, w)
    return w


def test_a2c_c_marks_appear_only_in_the_off_repo_review_files_with_counts_only_in_the_run_record(pinned, tmp_path, monkeypatch):
    w = _hint_world(pinned, tmp_path, monkeypatch)
    rc, res, err = cli(argv_for(w))
    assert rc == 0, err
    rd = Path(res["run_dir"])
    cand = (next((rd / "review").glob("review-excerpts-*.md"))).read_text()
    left = (next((rd / "review").glob("left-list-*.md"))).read_text()
    assert "hint: leave, consent" in cand                                                # HINT_TERMS order, each term once
    assert "hint: say no" in left
    for p in rd.rglob("*.json"):                                                         # NO pushed/default-path artifact carries a hint
        text = p.read_text()
        if p.name == "run-record.json":
            continue
        assert '"hint"' not in text and "say no" not in text and '"consent"' not in text, p.name
    for name in ("repair-list", "scope-ids", "id-map"):
        assert '"hint' not in (rd / "reports" / (name + ".json")).read_text()
    rec = json.loads((rd / "run-record.json").read_text())
    c = rec["review_hint_counts"]
    assert c["entries_marked"] == 2 and c["entries_total"] >= 3
    assert c["per_term"] == {"leave": 1, "consent": 1, "say no": 1}
    assert set(c["per_term"]) <= set(tool.HINT_TERMS) and all(isinstance(v, int) for v in c["per_term"].values())
    assert res["review_hint_counts"] == c


def test_a2c_c_the_hint_decides_nothing_reports_and_mapping_are_byte_equal_with_it_off(pinned, tmp_path, monkeypatch):
    w = _hint_world(pinned, tmp_path, monkeypatch)
    rc, on, err = cli(argv_for(w))
    assert rc == 0, err
    with monkeypatch.context() as mc:
        mc.setattr(tool, "review_hint", lambda text: [])
        rc, off, err = cli(argv_for(w))
    assert rc == 0, err
    ron, roff = Path(on["run_dir"]), Path(off["run_dir"])
    names = sorted(p.name for p in (ron / "reports").iterdir())
    assert names == sorted(p.name for p in (roff / "reports").iterdir())
    for n in names:                                                                        # every report, the mapping and the frozen lists: byte-equal
        assert (ron / "reports" / n).read_bytes() == (roff / "reports" / n).read_bytes(), n
    assert on["repair_list_sha256"] == off["repair_list_sha256"] and on["scope_ids_sha256"] == off["scope_ids_sha256"]
    assert on["outcomes"] == off["outcomes"] and on["candidates"] == off["candidates"]
    strip = lambda t: "\n".join(l for l in t.splitlines() if not l.startswith("hint:"))
    for pat in ("review-excerpts-*.md", "left-list-*.md"):                                 # identical apart from the hint lines
        a, b = next((ron / "review").glob(pat)).read_text(), next((roff / "review").glob(pat)).read_text()
        assert strip(a) == strip(b) and a != b
    assert json.loads((roff / "run-record.json").read_text())["review_hint_counts"] == {"entries_marked": 0, "entries_total": json.loads((ron / "run-record.json").read_text())["review_hint_counts"]["entries_total"], "per_term": {}}


def test_a2c_c_the_review_hint_never_changes_the_excerpt_hashes_or_the_gates(pinned, tmp_path, monkeypatch, p3_stub):
    w = _hint_world(pinned, tmp_path, monkeypatch)
    rc, res, err = cli(argv_for(w))
    assert rc == 0, err
    rl = json.loads((Path(res["run_dir"]) / "reports" / "repair-list.json").read_text())
    for c in rl["candidates"]:
        assert set(c) == {"id", "outcome", "class", "old_len", "new_len", "old_sha16", "new_sha16", "excerpt_sha256", "flags",
                          "source_node", "inner_open_rel", "closer_rel", "removed_prefix_len"}
        assert "hint" not in json.dumps(c["flags"])


# ---- le-034 C-1: a zero-write apply STOPs and retires nothing; the RETIRED receipt carries the struck trace -----------------

def _copy_and_retouch(rd, ap, name, mutate):
    ap2 = rd / name
    shutil.copy(ap, ap2)
    return str(ap2), _retouch_approvals(rd, str(ap2), mutate)


def test_c1_an_all_struck_packet_stops_writes_nothing_and_retires_nothing(pinned, tmp_path, monkeypatch, p3_stub):
    w, rd, frozen, ap, aps = _phase2_world(pinned, tmp_path, monkeypatch)
    ap2, aps2 = _copy_and_retouch(rd, ap, "approvals-allstruck.json", lambda b: [e.__setitem__("decision", "struck") for e in b["entries"]])
    p2, msha = _backup_and_args(w, rd, frozen, ap, aps)
    before = file_hashes(w.ckpt)
    rc, res, err = cli(_apply_argv(w, p2, msha, frozen, ap2, aps2), probes=FakeProbes())
    assert rc == 3 and "nothing to apply" in err
    assert file_hashes(w.ckpt) == before
    assert not list(Path(p2).glob("RETIRED-*.receipt")) and not list(Path(p2).glob("post-apply-receipt-FINAL-*.json"))
    rc, res, err = cli(_apply_argv(w, p2, msha, frozen, ap, aps), probes=FakeProbes())    # the one-shot was NOT consumed: a real packet still applies
    assert rc == 0, err


def test_c1_a_partly_struck_packet_completes_and_the_retired_receipt_carries_the_count_and_a_pointer(pinned, tmp_path, monkeypatch, p3_stub):
    w, rd, frozen, ap, aps = _phase2_world(pinned, tmp_path, monkeypatch)
    struck_old = w.ids["SEP2"]
    aps2 = _retouch_approvals(rd, ap, lambda b: [e.__setitem__("decision", "struck") for e in b["entries"] if e["id"] == struck_old])
    p2, msha = _backup_and_args(w, rd, frozen, ap, aps2)
    rc, res, err = cli(_apply_argv(w, p2, msha, frozen, ap, aps2), probes=FakeProbes())
    assert rc == 0, err
    final = next(Path(p2).glob("post-apply-receipt-FINAL-*.json"))
    ret = json.loads(next(Path(p2).glob("RETIRED-*.receipt")).read_text())
    assert ret["struck_deviations"] == 1                                                  # the COUNT only; ids/hashes stay in the FINAL receipt
    assert ret["post_apply_receipt"] == final.name and ret["post_apply_receipt_sha256"] == tool.sha256_file(str(final))
    assert struck_old not in json.dumps(ret)


# ---- le-034 C-2: foreign_unreadable is PERSISTED in the manifest, hold-start and both receipts -------------------------------

class ForeignProbes(FakeProbes):
    foreign_unreadable = 0

    def processes_matching(self, patterns):
        self.foreign_unreadable += 2
        return super().processes_matching(patterns)

    def files_held_open(self, paths):
        self.foreign_unreadable += 1
        return super().files_held_open(paths)


def test_c2_foreign_unreadable_is_persisted_in_the_manifest_hold_start_and_both_receipts(pinned, tmp_path, monkeypatch, p3_stub):
    w, rd, frozen, ap, aps = _phase2_world(pinned, tmp_path, monkeypatch)
    rc, res, err = cli(_p2_argv(w, *_partner_args(w)), probes=ForeignProbes())
    assert rc == 0, err
    p2, msha = res["run_dir"], res["backup_manifest_sha256"]
    want = {"p4": 3, "p6": 2}
    assert json.loads(next(Path(p2).glob("backup-manifest-*.json")).read_text())["foreign_unreadable"] == want
    assert json.loads((Path(p2) / "hold-start.json").read_text())["foreign_unreadable"] == want
    rc, res2, err = cli(_apply_argv(w, p2, msha, frozen, ap, aps), probes=ForeignProbes())
    assert rc == 0, err
    final = next(Path(p2).glob("post-apply-receipt-FINAL-*.json"))
    assert json.loads(final.read_text())["foreign_unreadable"] == want
    argv = _p2_argv(w, "--run-dir", p2, "--josh-go", "GO-RB", "--josh-go-manifest-sha256", msha,
                    "--josh-go-receipt-sha256", tool.sha256_file(str(final)), step="rollback")
    rc, res3, err = cli(argv, probes=ForeignProbes())
    assert rc == 0, err
    assert json.loads(next(Path(p2).glob("rollback-receipt-*.json")).read_text())["foreign_unreadable"] == want


# ---- le-034 C-3: a non-finite --code-placed-at is refused with a clear message -------------------------------------------

def test_c3_the_input_check_itself_refuses_every_non_finite_form_including_minus_inf():
    for bad in ("nan", "NaN", "inf", "-inf", "+inf", "Infinity", "-Infinity"):             # `-inf` cannot reach it through argparse
        ns = types.SimpleNamespace(expect_wants=tool.EXPECTED_WANTS, expect_protected=tool.EXPECTED_PROTECTED,
                                   expect_scope=tool.EXPECTED_SCOPE, code_placed_at=bad)
        with pytest.raises(tool.Refusal, match="finite"):
            tool._require_phase2_inputs(ns)


@pytest.mark.parametrize("bad", ["nan", "NaN", "inf", "Infinity"])
def test_c3_a_non_finite_code_placed_at_is_refused_clearly(pinned, tmp_path, monkeypatch, bad):
    w, rd, frozen, ap, aps = _phase2_world(pinned, tmp_path, monkeypatch)
    n_runs = len(list(w.backups.iterdir()))
    rc, res, err = cli(_with_flag(_p2_argv(w), "--code-placed-at", bad), probes=FakeProbes())
    assert rc == 2 and "finite" in err
    assert len(list(w.backups.iterdir())) == n_runs


# ---- le-034 C-4: two rollbacks in the same second never overwrite an earlier displaced copy --------------------------------

def test_c4_a_same_second_collision_never_overwrites_an_earlier_displaced_copy(pinned, tmp_path, monkeypatch, p3_stub):
    a = _applied(pinned, tmp_path, monkeypatch)
    w = a.w
    shutil.rmtree(Path(a.p2) / "stage")                                                   # force the displaced-copy branch both times
    monkeypatch.setattr(tool, "utc_stamp", lambda: "20260930T130000Z")                    # ONE second for everything below
    rc, res, err = cli(_rb_argv(a), probes=FakeProbes())
    assert rc == 0, err
    first = Path(a.p2) / "displaced-20260930T130000Z"
    assert first.is_dir()
    first_state = {n: (os.stat(first / n).st_ino, tool.sha256_file(str(first / n))) for n in (tool.MAIN_NAME, tool.SIDECAR_NAME)}
    for n in (tool.MAIN_NAME, tool.SIDECAR_NAME):                                         # put the post-apply state back, then roll back AGAIN
        shutil.copyfile(first / n, w.ckpt / n)
    rc, res, err = cli(_rb_argv(a), probes=FakeProbes())
    assert rc == 0, err
    second = sorted(p.name for p in Path(a.p2).glob("displaced-*"))
    assert len(second) == 2 and "displaced-20260930T130000Z" in second
    assert {n: (os.stat(first / n).st_ino, tool.sha256_file(str(first / n))) for n in first_state} == first_state     # the first copy untouched
    assert len(list(Path(a.p2).glob("rollback-receipt-*.json"))) == 2                       # nor the earlier rollback receipt


# ---- N-2: --apply and --step rollback together are refused ----------------------------------------------------------------

def test_n2_apply_and_step_rollback_together_are_mutually_exclusive(pinned, tmp_path, monkeypatch, p3_stub):
    a = _applied(pinned, tmp_path, monkeypatch)
    rc, res, err = cli(_rb_argv(a, "--apply"), probes=FakeProbes())
    assert rc == 2 and "mutually exclusive" in err and file_hashes(a.w.ckpt) == a.after


# ==================================================================================================
# DELTA BUILD (#11805): the streamed content-subset vectors read + the Graph-free analyze() (item 8).
# The OLD path (canonical Graph().restore + SimpleVectorDB().load, the pre-delta analyze) is built INSIDE these tests
# as test-only code; it is not the tool's path. SYNTHETIC data only.
# ==================================================================================================

def _old_load_pair(dirpath, pinned):
    """TEST-ONLY: the pre-delta loader (the two canonical readers)."""
    g = pinned.nf.Graph()
    g.restore(os.path.join(dirpath, tool.MAIN_NAME))
    vdb = pinned.ui.SimpleVectorDB()
    vdb.load(os.path.join(dirpath, tool.VECTORS_NAME))
    return g, vdb


def _old_figures(g, ids):
    return {i: (len(g._outgoing.get(i, ())), len(g._incoming.get(i, ())), len(g._node_hyperedges.get(i, ())))
            for i in ids if i in g.nodes}


def _old_analyze(pinned, dirpath, *, scope_min_len, frozen_scope=None, base_mod=None, full_reports=True):
    """TEST-ONLY copy of the pre-delta analyze(): Graph().restore + SimpleVectorDB().load, every value taken FROM the
    live Graph. Returns the same dict plus `before_figures_scope` (figures for S + the three protected ids)."""
    org = pinned.org
    g, vdb = _old_load_pair(dirpath, pinned)
    nodes_meta = {nid: n.metadata for nid, n in g.nodes.items()}
    existing_ids = set(g.nodes)
    content = vdb.content
    vdb.embeddings = {}
    derived = tool.derive_scope(nodes_meta, scope_min_len)
    scope = list(frozen_scope) if frozen_scope is not None else derived
    cl = tool.Classifier(org, nodes_meta, content)
    records = [cl.classify_node(nid) for nid in scope]
    dropped = tool.apply_collision_rule(records, existing_ids)
    tool.attach_excerpt_hashes(org, content, records)
    for r in records:
        if r["disposition"] == "candidate":
            r["flags"]["collision_candidate"] = False
            r["flags"]["mention_shapes"] = tool.mention_shape_flags(content[r["source_node"]], r["_w_open"], r["_closer_end"])
    cand_old = [r["id"] for r in records if r["disposition"] == "candidate"]
    watch = list(tool.CHOICE_CLAUSE_IDS) + [tool.CONSTITUTIONAL_ID]
    before_fig = _old_figures(g, set(cand_old) | set(watch))
    before_fig_scope = _old_figures(g, set(scope) | set(watch))
    render_before = len(org.render_wants(g).encode("utf-8"))
    counts = {"nodes": len(g.nodes),
              "wants": sum(1 for md in nodes_meta.values() if md.get("kind") == "want"),
              "protected": sum(1 for md in nodes_meta.values() if tool.is_protected(md))}
    del g, vdb
    gc.collect()
    raw_main = Path(os.path.join(dirpath, tool.MAIN_NAME)).read_bytes()
    want_ids = {nid for nid, md in nodes_meta.items() if md.get("kind") == "want"}
    syn = tool.synapse_stats(raw_main, want_ids, set(scope), set(cand_old))
    del raw_main
    A = {"dir": dirpath, "records": records, "dropped": dropped, "scope": scope, "scope_derived": derived,
         "scope_min_len": scope_min_len, "nodes_meta": nodes_meta, "content": content, "existing_ids": existing_ids,
         "cl": cl, "before_figures": before_fig, "before_figures_scope": before_fig_scope,
         "render_len_before": render_before, "counts": counts, "syn": syn}
    if full_reports:
        s_sources = sorted({r["source_node"] for r in records if r.get("source_node")})
        marker_nodes = tool.conversational_marker_nodes(nodes_meta, content)
        A["s_sources"], A["marker_nodes"] = s_sources, marker_nodes
        A["histograms"] = tool.reason_histograms(cl, s_sources, marker_nodes)
        mb = tool.marker_bearing_minted(cl, marker_nodes, set(s_sources))
        for e in mb:
            e["shape_flags"] = tool.mention_shape_flags(content[e["source_node"]], e["open_start"], e["close_end"])
        A["marker_bearing"] = mb
        A["residuals"] = tool.residual_classes(pinned, base_mod or tool.load_base_module(pinned), nodes_meta, content, marker_nodes, cl)
    return A


@pytest.fixture(scope="module")
def xworld(pinned, tmp_path_factory):
    """The EXTRAS world (S source without vdb entry / with empty content / without marker, non-ASCII candidate,
    archived hyperedge, vdb-only entries) - its own directory; no recorded-path patching is needed (analyze only reads)."""
    return build_world(tmp_path_factory.mktemp("xworld"), pinned, extras=True)


def _keep_for(A):
    return {md.get("source_node") for nid in A["scope"] for md in [A["nodes_meta"][nid]] if isinstance(md.get("source_node"), str)}


def _scrub_obj(o):
    return json.dumps(o, sort_keys=True, default=repr)


def test_the_extras_world_covers_every_named_edge(xworld, pinned):
    """The synthetic corpus really exercises: S source with content but no WANT], S source with NO content entry, empty
    and odd content, non-ASCII, marker nodes that are nobody's source, vdb-only entries."""
    A = _old_analyze(pinned, str(xworld.ckpt), scope_min_len=MIN_LEN)
    by = {r["id"]: r for r in A["records"]}
    assert by[xworld.ids["NOMARK"]]["detail"] == "content_has_marker"                        # content present, no WANT]
    assert by[xworld.ids["NOCONTENT"]]["detail"] == "content_present,content_has_marker"     # no vdb entry at all
    assert by[xworld.ids["EMPTYC"]]["detail"] == "content_present,content_has_marker"        # empty string content
    assert by[xworld.ids["NONASCII"]]["disposition"] == "candidate"
    assert "cc:conv::markeronly" in A["marker_nodes"] and "cc:conv::markeronly" not in _keep_for(A)
    assert {"orph:marker", "orph:plain", "orph:empty"} <= set(A["content"])


def test_the_streamed_content_reader_equals_the_filtered_canonical_load(xworld, pinned):
    A = _old_analyze(pinned, str(xworld.ckpt), scope_min_len=MIN_LEN)
    keep = _keep_for(A)
    got = tool.load_content_subset(str(xworld.ckpt / tool.VECTORS_NAME), keep)
    want = {k: v for k, v in A["content"].items() if k in keep or (isinstance(v, str) and "WANT]" in v)}
    assert got == want and list(got) == list(want)                     # same entries, same values, same order
    assert "orph:marker" in got and "orph:plain" not in got and "orph:empty" not in got
    assert got["cc:conv::emptyc"] == "" and "cc:conv::nocontent" not in got
    assert any(not v.isascii() for v in got.values())


def test_the_streamed_reader_never_decodes_an_embedding_or_metadata(xworld, pinned, monkeypatch):
    """A tracking Unpacker records every object `unpack()` returns: only str (ids, field names, content) may come out;
    an embedding (bytes) or a metadata dict must never be materialised - they are skip()ped."""
    real = tool._mp()
    seen = []

    class Tracking(real.Unpacker):
        def unpack(self, *a, **k):
            v = super().unpack(*a, **k)
            seen.append(v)
            return v

    monkeypatch.setattr(tool, "_mp", lambda: types.SimpleNamespace(Unpacker=Tracking, packb=real.packb, unpackb=real.unpackb, Packer=real.Packer))
    tool.load_content_subset(str(xworld.ckpt / tool.VECTORS_NAME), {"cc:conv::nonascii"})
    assert seen and all(isinstance(v, str) for v in seen), {type(v).__name__ for v in seen}


def _vfile(tmp_path, raw=None):
    p = tmp_path / "vectors.msgpack"
    p.write_bytes(raw)
    return str(p)


def test_a_truncated_or_malformed_vectors_file_fails_closed(xworld, tmp_path):
    raw = (xworld.ckpt / tool.VECTORS_NAME).read_bytes()
    assert tool.load_content_subset(_vfile(tmp_path, raw), {"x"}) is not None           # the intact file reads
    for cut in (len(raw) - 1, len(raw) - 37, len(raw) // 2, len(raw) // 3, 30, 5, 1):
        with pytest.raises(tool.Stop):
            tool.load_content_subset(_vfile(tmp_path, raw[:cut]), {"x"})                # never a silently smaller content set
    with pytest.raises(tool.Stop):                                                      # trailing bytes after the top map
        tool.load_content_subset(_vfile(tmp_path, raw + b"\x00"), {"x"})
    with pytest.raises(tool.Stop):                                                      # not a map at all
        tool.load_content_subset(_vfile(tmp_path, msgpack.packb([1, 2, 3])), {"x"})
    bad = msgpack.packb({"version": "1.0.0", "count": 1, "entries": {"a": {"content": "c", "metadata": {}}}}, use_bin_type=True)
    with pytest.raises(tool.Stop):                                                      # an entry with no embedding: the canonical loader raises too
        tool.load_content_subset(_vfile(tmp_path, bad), {"a"})


def test_a_duplicate_entry_id_keeps_the_last_value_like_the_canonical_dict(tmp_path):
    e = lambda c: {"embedding": b"\x00" * 16, "content": c, "metadata": {}}          # noqa: E731
    raw = b"\x83" + msgpack.packb("version") + msgpack.packb("1.0.0") + msgpack.packb("count") + msgpack.packb(2) \
        + msgpack.packb("entries") + b"\x82" + msgpack.packb("a") + msgpack.packb(e("first [WANT]")) \
        + msgpack.packb("a") + msgpack.packb(e("second"))
    assert msgpack.unpackb(raw)["entries"]["a"]["content"] == "second"                # the canonical loader's view
    assert tool.load_content_subset(_vfile(tmp_path, raw), set()) == {}               # last entry has no marker and is not kept


def test_the_graph_stream_equals_the_canonical_restore(xworld, pinned):
    g, _ = _old_load_pair(str(xworld.ckpt), pinned)
    V = tool.stream_graph_nodes(str(xworld.ckpt / tool.MAIN_NAME))
    assert V["nodes_meta"] == {nid: n.metadata for nid, n in g.nodes.items()}
    assert list(V["nodes_meta"]) == list(g.nodes)                                    # same order (the render sort is stable)
    assert V["existing_ids"] == set(g.nodes)
    assert len(pinned.org.render_wants(V["render_graph"]).encode("utf-8")) == len(pinned.org.render_wants(g).encode("utf-8"))
    ids = set(g.nodes) | {"not-a-node"}                                              # EVERY node, plus a stranger
    assert tool.stream_incident_figures(str(xworld.ckpt / tool.MAIN_NAME), ids, V["existing_ids"]) == _old_figures(g, ids)
    assert any(f[2] for f in _old_figures(g, ids).values())                          # the world has hyperedge membership
    assert len(g._node_hyperedges[xworld.ids["NONASCII"]]) == 0                      # ...and an ARCHIVED one that does not count


def test_the_streamed_analysis_is_identical_to_the_canonical_graph_and_vdb_analysis(xworld, pinned):
    base = tool.load_base_module(pinned)
    O = _old_analyze(pinned, str(xworld.ckpt), scope_min_len=MIN_LEN, base_mod=base)
    N = tool.analyze(pinned, str(xworld.ckpt), scope_min_len=MIN_LEN, base_mod=base)
    for k in ("scope", "scope_derived", "dropped", "nodes_meta", "existing_ids", "counts", "render_len_before", "syn",
              "s_sources", "marker_nodes", "histograms", "marker_bearing", "residuals", "records"):
        assert _scrub_obj(sorted(O[k]) if isinstance(O[k], set) else O[k]) == _scrub_obj(sorted(N[k]) if isinstance(N[k], set) else N[k]), k
    # V11 'before' figures: the new set is S + the three protected ids (a superset of the old candidates + watch);
    # the old keys are equal, and the whole new set equals the canonical figures over that same id set.
    assert N["before_figures"] == O["before_figures_scope"]
    assert {k: N["before_figures"][k] for k in O["before_figures"]} == O["before_figures"]
    keep = _keep_for(O)
    assert N["content"] == {k: v for k, v in O["content"].items() if k in keep or (isinstance(v, str) and "WANT]" in v)}
    # every stamped report artifact, byte for byte (sha256 over the canonical serialisation)
    ro, rn = tool.build_reports(O, xworld.expect["scope"]), tool.build_reports(N, xworld.expect["scope"])
    assert {k: tool.artifact_sha256(v) for k, v in ro.items()} == {k: tool.artifact_sha256(v) for k, v in rn.items()}
    assert _scrub_obj(ro) == _scrub_obj(rn)


def test_the_review_files_and_the_verifier_and_t6_are_identical(xworld, pinned, monkeypatch):
    patch_world(monkeypatch, xworld)                                                   # writes are guarded to the backups root
    base = tool.load_base_module(pinned)
    utc = "20260930T000000Z"
    outs = []
    for tag, fn in (("old", _old_analyze), ("new", tool.analyze)):
        A = fn(pinned, str(xworld.ckpt), scope_min_len=MIN_LEN, base_mod=base)
        run = xworld.backups / (tool.RUN_DIR_PREFIX + "eq-" + tag)
        (run / "reports").mkdir(parents=True)
        rl, sc = tool.artifact_sha256(tool.repair_list_obj(A)), tool.artifact_sha256(tool.scope_ids_obj(A, xworld.expect["scope"]))
        approvals = tool.stamped(tool.approvals_body_for(A["records"], rl, sc, "EXEC-SYNTHETIC-PACKET"))
        rshas, hints = tool.write_review_files(str(run), utc, pinned.org, A["content"], A["records"])
        review_bytes = {p.name: p.read_bytes() for p in sorted(run.rglob("*")) if p.is_file()}
        in_hashes = {n: tool.sha256_file(str(xworld.ckpt / n)) for n in tool.SIX_FILES}
        out = tool.prepare_outputs(pinned, A, approvals, in_dir=str(xworld.ckpt), out_dir=str(run / "out"), run_dir=str(run),
                                   utc=utc, expect=xworld.expect, in_hashes=in_hashes)
        outs.append({"review_shas": rshas, "hint_counts": hints, "review_bytes": review_bytes,
                     "results": out["results"], "failed": out["failed"], "t6": out["t6"], "out_hashes": out["out_hashes"],
                     "id_map_sha256": out["id_map_sha256"], "walk": out["walk_counts"], "writer": out["writer_stats"],
                     "sidecar": out["sidecar_stats"], "written": out["plan"]["write_ids"],
                     "rewritten_main": (run / "out" / tool.MAIN_NAME).read_bytes()})
    o, n = outs
    assert o["failed"] == [] and len(o["results"]) == 19 and n["failed"] == []
    assert len(o["written"]) >= 4                                                      # the non-ASCII candidate is among the writes
    for k in o:
        assert o[k] == n[k], k                                                         # byte for byte


def test_analyze_needs_no_live_graph_and_no_vdb_load(xworld, pinned, monkeypatch):
    """item 8: the classify/id-mapping analysis constructs NO Graph and calls NO vector-db load."""
    def boom(*a, **k):
        raise AssertionError("analyze() built a Graph / loaded the whole vectors file")

    monkeypatch.setattr(pinned.nf, "Graph", boom)
    monkeypatch.setattr(pinned.ui.SimpleVectorDB, "load", boom)
    A = tool.analyze(pinned, str(xworld.ckpt), scope_min_len=MIN_LEN)
    assert A["counts"]["nodes"] > 0 and A["records"]


def test_the_old_whole_file_paths_are_gone_from_the_tool_source():
    """LAW 3 / item 2: REPLACED, not added - no dead old path left in the tool (needles built by concatenation so this
    test file never matches itself)."""
    src = TOOL_PATH.read_text(encoding="utf-8")
    code = [ln for ln in src.splitlines() if not ln.lstrip().startswith("#")]
    body = "\n".join(code)
    for needle in ("Simple" + "VectorDB", "load" + "_pair", "vdb" + ".load", "vdb" + ".content", "incident_figures(g" + ","):
        assert needle not in body, needle
    assert body.count(".restore(") == 1                                               # the ONE canonical restore left: V11's output restore


def _synth_vectors(path, n, dim=768, content_len=700, marker_every=1000):
    """A synthetic vectors file in the canonical format (version/count/entries), written entry by entry."""
    rng = np.random.default_rng(7)
    with open(path, "wb") as f:
        f.write(b"\x83" + msgpack.packb("version") + msgpack.packb("1.0.0") + msgpack.packb("count") + msgpack.packb(n)
                + msgpack.packb("entries"))
        f.write(b"\xdf" + n.to_bytes(4, "big"))
        for i in range(n):
            c = ("filler-%d " % i) * (content_len // 10)
            if i % marker_every == 0:
                c += " [WANT]synthetic[/WANT]"
            f.write(msgpack.packb("id-%06d" % i))
            f.write(msgpack.packb({"embedding": rng.random(dim, dtype=np.float32).tobytes(), "content": c, "metadata": {"i": i}}, use_bin_type=True))
    return os.path.getsize(path)


def _traced_peak(fn):
    import tracemalloc
    gc.collect()
    tracemalloc.start()
    try:
        out = fn()
        cur, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return out, cur, peak


def test_memory_shape_the_streamed_reader_scales_with_the_keep_set_not_the_file(pinned, tmp_path):
    """SHAPE check (tracemalloc; NOT the real peak): new path retained/peak bytes vs the canonical whole-file load, on two
    synthetic files (one twice the size of the other, tens of MB). Numbers are printed for the return."""
    rows = []
    for n in (8000, 16000):
        p = str(tmp_path / ("v%d.msgpack" % n))
        size = _synth_vectors(p, n)
        keep = {"id-%06d" % i for i in range(0, n, 2000)}
        old, _, old_peak = _traced_peak(lambda: pinned.ui.SimpleVectorDB().load(p))            # noqa: B023
        new, new_cur, new_peak = _traced_peak(lambda: tool.load_content_subset(p, keep))       # noqa: B023
        retained = sum(len(v.encode()) for v in new.values())
        rows.append((n, size, old_peak, new_peak, new_cur, retained, len(new)))
        print("MEMSHAPE n=%d file=%.1fMB old_peak=%.1fMB new_peak=%.2fMB new_retained_current=%.2fMB content_kept=%d entries=%d"
              % (n, size / 1e6, old_peak / 1e6, new_peak / 1e6, new_cur / 1e6, retained, len(new)))
        assert len(new) == len(keep | {"id-%06d" % i for i in range(0, n, 1000)})          # keep set + the marker entries only
        assert new_peak < old_peak / 10 and new_peak < size / 5
    (n1, s1, o1, w1, c1, r1, _), (n2, s2, o2, w2, c2, r2, _) = rows
    assert o2 > 1.6 * o1                                                                  # the old peak grows with the file
    assert w2 < 2.5 * w1 + 1_000_000                                                      # the new peak does not track the file


def test_the_streamed_render_view_orders_and_truncates_like_the_canonical_graph(pinned, tmp_path):
    """render_wants sorts by the STORED creation_time (stable on ties) and keeps the newest WANT_RENDER_LIMIT: with more
    wants than the limit, varied lengths (some over WANT_MAX_CHARS) and tied timestamps, the streamed view must give the
    byte-identical block length (a view with a wrong creation_time or node order would pick a different top-N)."""
    nf, org = pinned.nf, pinned.org
    n = org.WANT_RENDER_LIMIT + 15
    g = nf.Graph()
    for i in range(n):
        nid = "w%03d" % i
        g.create_node(node_id=nid, metadata={"kind": "want", "want_text": "t%03d " % i + "x" * ((i * 37) % (org.WANT_MAX_CHARS + 200)),
                                             "want_state": "open", "provenance": "cc_authored" if i % 3 else "cc_emergent"})
        g.nodes[nid].creation_time = (i * 7) % 11 if i % 4 else 5                       # scrambled, with many ties
    p = str(tmp_path / "main.msgpack")
    g.write_checkpoint(p, g.capture_checkpoint(nf.CheckpointMode.FULL))
    g2 = nf.Graph()
    g2.restore(p)
    V = tool.stream_graph_nodes(p)
    want = org.render_wants(g2)
    assert len(want.encode("utf-8")) > 1000 and want.count("\n") == org.WANT_RENDER_LIMIT   # the limit really bit
    assert org.render_wants(V["render_graph"]) == want
