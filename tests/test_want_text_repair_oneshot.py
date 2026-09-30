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
import hashlib
import importlib.util
import io
import json
import os
import shutil
import stat
import subprocess
import sys
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


def _specs(wid):
    specs = {}

    def add(name, C, T, id=None):
        specs[name] = {"C": C, "T": T, "id": id or wid(T)}

    T = prose("sepa") + " [WANT]" + X1
    add("SEP1", "preamble one. [WANT]" + T + "[/WANT] trailing text.", T)
    T = BT + "code" + BT + " " + prose("sepb") + " [WANT]" + X2
    add("SEP2", "[WANT]" + T + "[/WANT] tail", T)
    L = prose("gen")
    add("GEN", "lead in. [WANT]" + L + "[/WANT] done.", L)
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


def build_world(base: Path, pinned, *, tag="w"):
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
    specs = _specs(wid)
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
    for nid, txt in ((wid(XE), XE), (wid("a short want"), "a short want"),
                     (tool.CHOICE_CLAUSE_IDS[0], "synthetic choice clause want one"),
                     (tool.CHOICE_CLAUSE_IDS[1], "synthetic choice clause want two")):
        g.create_node(node_id=nid, metadata={"kind": "want", "want_text": txt, "want_state": "open",
                      "provenance": "cc_authored", "source_node": "cc:conv::plain", "creation_mode": "conversational"})
    short1, cc1, cc2, rim = wid("a short want"), tool.CHOICE_CLAUSE_IDS[0], tool.CHOICE_CLAUSE_IDS[1], tool.CONSTITUTIONAL_ID
    g.create_node(node_id=rim, metadata={"constitutional": True})
    g.create_node(node_id="n1", metadata={})
    g.create_node(node_id="n2", metadata={})
    for a, b, w in ((ids["SEP1"], ids["GEN"], 0.4), (short1, ids["SEP1"], 0.5), (cc1, ids["SEP1"], 0.6), (cc2, ids["GEN"], 0.6),
                    (rim, ids["SEP1"], 0.7), (ids["SEP2"], rim, 0.7), (rim, ids["GEN"], 0.7), ("n1", ids["SEP1"], 0.2)):
        g.create_synapse(a, b, weight=w)
    syn = g.create_synapse("n1", "n2", weight=0.2)
    syn.metadata = {"creation_mode": "surprise_driven", "expected_target": ids["SEP2"], "timestep": 0}
    g.nodes["n1"].pred_weights = {ids["SEP1"]: 0.5, ids["GEN"]: 0.2}
    g.nodes[cc1].pred_weights = {ids["SEP1"]: 0.7, ids["GEN"]: 0.1}
    g.nodes[rim].pred_weights = {ids["SEP1"]: 0.4}
    g.nodes[short1].pred_weights = {ids["SEP2"]: 0.3}
    g.nodes[ids["SEP1"]].pred_weights = {"n2": 0.3}
    he = g.create_hyperedge({ids["SEP1"], "n1", "n2"}, member_weights={ids["SEP1"]: 1.0, "n1": 0.5, "n2": 0.5},
                            output_targets=[ids["SEP2"]])
    cap = g.capture_checkpoint(nf.CheckpointMode.FULL)
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
    wants = [n for n, nd in g.nodes.items() if nd.metadata.get("kind") == "want"]
    scope = sorted(n for n in wants if len(g.nodes[n].metadata["want_text"]) > MIN_LEN)
    w = types.SimpleNamespace(base=base, ws=ws, ckpt=ckpt, daemon=daemon, backups=base / "backups", conduit=base / "conduit",
                              ids=ids, specs=specs, scope=scope, short1=short1, cc1=cc1, cc2=cc2, rim=rim, he=he.hyperedge_id,
                              new={"SEP1": wid(X1), "SEP2": wid(X2), "BIG": wid(X5)},
                              expect={"wants": len(wants), "protected": len(wants) + 1, "scope": len(scope)})
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
                 "pre-node-report", "candidate-id-map"):
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
    assert len(write) == 1 and sorted(r["reason"] for r in refused) == ["not_approved", "not_approved"]
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
    assert tool.load_approvals(p, s, rl, sc, allow_provisional=True)["packet"] == tool.PROVISIONAL_PACKET


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
    mtimes = {p.name: p.stat().st_mtime_ns for p in w.ckpt.iterdir()}
    for p in w.ckpt.iterdir():
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
            p.chmod(0o644)
    assert file_hashes(w.ckpt) == before and {p.name: p.stat().st_mtime_ns for p in w.ckpt.iterdir()} == mtimes
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


def test_p4_daemon_down_is_mechanical_and_every_leg_can_fail(world):
    d = str(world.ckpt)
    assert tool.gate_p4(FakeProbes(), d)["ok"]
    for kw, leg in (({"units_active": {tool.DAEMON_UNIT}}, "unit_inactive"),
                    ({"units_active": {tool.RECOVER_TIMER}}, "recover_timer_inactive"),
                    ({"procs": [("cc-ng-service.py", 4242)]}, "no_daemon_process"),
                    ({"held": [(4242, d + "/main.msgpack")]}, "no_process_holds_the_six_files")):
        r = tool.gate_p4(FakeProbes(**kw), d)
        assert not r["ok"] and not r["checks"][leg]
    manifest = {n: {"sha256": tool.sha256_file(os.path.join(d, n)), "size": os.stat(os.path.join(d, n)).st_size,
                    "mtime_ns": os.stat(os.path.join(d, n)).st_mtime_ns} for n in tool.SIX_FILES}
    assert tool.gate_p4(FakeProbes(), d, manifest_files=manifest)["ok"]
    manifest[tool.MAIN_NAME]["sha256"] = "0" * 64
    r = tool.gate_p4(FakeProbes(), d, manifest_files=manifest)
    assert not r["ok"] and not r["checks"]["six_files_equal_the_start_of_phase2_backup"]


def test_p4_a_pid_file_naming_a_live_process_fails_and_a_log_newer_than_the_placement_fails(world):
    (world.ws / "daemon.pid").write_text("4242")
    (world.ws / "daemon.log").write_text("x")
    try:
        r = tool.gate_p4(FakeProbes(alive=True), str(world.ckpt))
        assert not r["ok"] and not r["checks"]["daemon_pid_file_names_a_dead_pid"]
        r = tool.gate_p4(FakeProbes(), str(world.ckpt), code_placed_at=1.0, daemon_log=str(world.ws / "daemon.log"))
        assert not r["ok"] and not r["checks"]["no_pulse_since_code_placement"]
        r = tool.gate_p4(FakeProbes(), str(world.ckpt), code_placed_at=10 ** 12, daemon_log=str(world.ws / "daemon.log"))
        assert r["ok"]
    finally:
        (world.ws / "daemon.pid").unlink()
        (world.ws / "daemon.log").unlink()


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

def _phase2_world(pinned, tmp_path, mp):
    w = build_world(tmp_path, pinned)
    patch_world(mp, w)
    rc, c1, err = cli(argv_for(w))
    assert rc == 0, err
    rd = Path(c1["run_dir"])
    frozen = rd / "frozen"
    frozen.mkdir()
    for name in ("repair-list", "scope-ids"):
        shutil.copy(rd / "reports" / (name + ".json"), frozen / (name + ".json"))
    shutil.copy(rd / "reports" / "candidate-id-map.json", frozen / "id-map.json")
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


def _p2_argv(w, *extra, step="phase2-backup"):
    return argv_for(w, "--conduit-dir", str(w.conduit), *extra, step=step)


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
              "--daemon-organism-file", str(PIN_ROOT / "cc_ng_organism.py")]
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


def test_apply_stops_on_any_deviation_from_the_approved_list_and_writes_nothing(pinned, tmp_path, monkeypatch, p3_stub):
    w, rd, frozen, ap, aps = _phase2_world(pinned, tmp_path, monkeypatch)
    body = json.loads(Path(ap).read_text())
    body.pop("pin_stamp")
    body["entries"][0]["decision"] = "struck"
    data = tool.canonical_json(tool.stamped(body))
    Path(ap).write_bytes(data)
    aps2 = hashlib.sha256(data).hexdigest()
    rc, res, err = cli(_p2_argv(w), probes=FakeProbes())
    p2, msha = res["run_dir"], res["backup_manifest_sha256"]
    before = file_hashes(w.ckpt)
    rc, res2, err = cli(_p2_argv(w, "--apply", "--run-dir", p2, "--frozen-dir", str(frozen), "--approvals", ap, "--approvals-sha256", aps2,
                                 "--daemon-organism-file", str(PIN_ROOT / "cc_ng_organism.py"), "--josh-go", "GO-1",
                                 "--josh-go-manifest-sha256", msha), probes=FakeProbes())
    assert rc == 3 and "deviation" in err
    assert file_hashes(w.ckpt) == before
    assert not list(Path(p2).glob("RETIRED-*.receipt"))


def test_apply_writes_the_pair_atomically_verifies_it_then_retires_the_tool(pinned, tmp_path, monkeypatch, p3_stub):
    w, rd, frozen, ap, aps = _phase2_world(pinned, tmp_path, monkeypatch)
    rc, res, err = cli(_p2_argv(w), probes=FakeProbes())
    assert rc == 0, err
    p2, msha = res["run_dir"], res["backup_manifest_sha256"]
    before = file_hashes(w.ckpt)
    args = _p2_argv(w, "--apply", "--run-dir", p2, "--frozen-dir", str(frozen), "--approvals", ap, "--approvals-sha256", aps,
                    "--daemon-organism-file", str(PIN_ROOT / "cc_ng_organism.py"), "--josh-go", "GO-1", "--josh-go-manifest-sha256", msha)
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
                    "--daemon-organism-file", str(PIN_ROOT / "cc_ng_organism.py"), "--josh-go", "GO-1", "--josh-go-manifest-sha256", msha)
    rc, res2, err = cli(args, probes=FakeProbes())
    assert rc == 2 and "immediately before os.replace" in err and calls["n"] == 2
    assert file_hashes(w.ckpt) == before
    assert not list(Path(p2).glob("RETIRED-*.receipt"))
