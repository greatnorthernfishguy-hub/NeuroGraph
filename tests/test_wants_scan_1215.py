# ---- Changelog ----
# [2026-11-06] DudeMan CC (#1215; Josh 2026-11-06 "Please dive into that.") — CREATE
# What: the incremental want scan (CC_NG_WANTS_SCAN_NATIVE): (1) off = e2f93d4's surface_wants exactly, across a
#       scripted sequence of passes; (2) on = off after every pass of the same sequence (want nodes, their synapses, the
#       returned open wants as a set, the create / synapse failure reports, the skipped-marker detail lines): new text,
#       an unchanged pass, a rewritten text, a deleted want node, a create_node that fails once, a text with more skips
#       than the per-call detail cap, a checkpoint restore; (3) on: an unchanged text is not parsed again; (4) the record
#       follows rows through delete + compaction, and clear() empties it; (5) on without the native NodeStore raises.
# Why:  spec docs/superpowers/specs/2026-11-06-wants-scan-incremental-design.md §4.1.
# How:  a real neuro_foundation.Graph (native NodeStore) and universal_ingestor.SimpleVectorDB; the base organism is
#       loaded from git (NG_WANTS_SCAN_BASE_REV, default e2f93d4). Needs ng_tract with NodeStore.wants_scan.
# -------------------
import importlib.util
import logging
import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
import cc_ng_organism as ORG  # noqa: E402  (the branch)
import neuro_foundation as NF  # noqa: E402
from universal_ingestor import SimpleVectorDB  # noqa: E402

BASE_REV = os.environ.get("NG_WANTS_SCAN_BASE_REV", "e2f93d4")
NATIVE = hasattr(NF.ng_tract.NodeStore, "wants_scan")
needs_native = pytest.mark.skipif(not NATIVE, reason="installed ng_tract has no NodeStore.wants_scan")


def _load_base():
    src = subprocess.run(["git", "-C", _REPO, "show", f"{BASE_REV}:cc_ng_organism.py"],
                         check=True, capture_output=True, text=True).stdout
    d = tempfile.mkdtemp(prefix="wants_scan_base_")
    p = os.path.join(d, "org_base_wants_scan.py")
    with open(p, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location("org_base_wants_scan", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["org_base_wants_scan"] = mod
    spec.loader.exec_module(mod)
    return mod


BASE = _load_base()

EMB = np.ones(4, dtype=np.float32)
C1 = "intro\n\n[WANT] revisit the pth cleanup [/WANT]\n\nmore\n\n[WANT] a second want [/WANT]\n"
C1_REWRITTEN = C1 + "\n\n[WANT] a third want [/WANT]\n"
C2 = "\n\n".join('note %d: the "[WANT]" tag opens and the "[/WANT]" tag closes it' % i for i in range(3))
C6 = "later\n\n[WANT] a flaky want [/WANT]\n"


def _build():
    g = NF.Graph(native_node_store=True)
    vdb = SimpleVectorDB(dim=4)
    for nid, mode, text in (("c1", "conversational", C1), ("c2", "conversational", C2),
                            ("c3", "conversational", "plain text, no marker"),
                            ("c4", "ingested", "[WANT] not a conversation [/WANT]"),
                            ("c5", "conversational", None)):
        g.create_node(node_id=nid, metadata={"creation_mode": mode})
        if text is not None:
            vdb.insert(id=nid, embedding=EMB, content=text, metadata={})
    g.create_node(node_id="want::open0", metadata={"kind": "want", "want_text": "old open", "want_state": "open",
                                                   "provenance": "cc_authored", "source_node": "c3"})
    g.create_node(node_id="want::closed0", metadata={"kind": "want", "want_text": "old closed",
                                                     "want_state": "closed", "provenance": "cc_authored"})
    return g, vdb


def _want_id(mod, text):
    return [w.want_id for w in mod.parse_wants(text).wants]


def _snap(g, ret, reports, details):
    wants = {}
    for nid, node in g.nodes.items():
        md = node.metadata or {}
        if md.get("kind") == "want":
            wants[nid] = (md.get("want_text"), md.get("want_state"), md.get("provenance"), md.get("source_node"))
    syn = sorted((s.pre_node_id, s.post_node_id, round(s.weight, 9)) for s in g.synapses.values()
                 if s.post_node_id in wants)
    rset = sorted(tuple(sorted(w.items())) for w in ret)
    return dict(wants=wants, syn=syn, ret=rset, reports=list(reports), details=list(details))


def _run(mod, native, monkeypatch, caplog):
    """The scripted sequence; returns one snapshot per pass."""
    mod._WANT_SKIP_SEEN.clear()
    mod._WANT_SKIP_STATE.update(last_key=None, last_emit=None)
    monkeypatch.setattr(mod, "WANT_SKIP_DETAIL_PER_CALL_MAX", 2)
    if hasattr(mod, "WANTS_SCAN_NATIVE"):
        monkeypatch.setattr(mod, "WANTS_SCAN_NATIVE", native)
    reports = []
    for name in ("_cc_report_want_create_failures", "_cc_report_want_synapse_failures"):
        real = getattr(mod, name)
        monkeypatch.setattr(mod, name, (lambda real, name: lambda *a: (reports.append((name,) + a), real(*a)))(real, name))
    g, vdb = _build()
    snaps = []
    flaky = _want_id(mod, C6)[0]

    def one():
        reports.clear()
        caplog.clear()
        with caplog.at_level(logging.INFO, logger=mod.logger.name):
            ret = mod.surface_wants(g, vdb)
        details = [r.getMessage() for r in caplog.records if "node=" in r.getMessage()]
        snaps.append(_snap(g, ret, reports, details))

    one()                                                         # P1: first pass
    one()                                                         # P2: nothing changed
    vdb.insert(id="c1", embedding=EMB, content=C1_REWRITTEN, metadata={})
    one()                                                         # P3: c1 rewritten (a third want)
    g.remove_node(_want_id(mod, C1)[0])
    one()                                                         # P4: a want node deleted: recreated
    g.create_node(node_id="c6", metadata={"creation_mode": "conversational"})
    vdb.insert(id="c6", embedding=EMB, content=C6, metadata={})
    real_create = g.create_node

    def failing(node_id=None, **kw):
        if node_id == flaky:
            raise ValueError("injected")
        return real_create(node_id=node_id, **kw)
    g.create_node = failing
    one()                                                         # P5: c6's want fails to create
    g.create_node = real_create
    one()                                                         # P6: retried and created
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.msgpack")
        g.checkpoint(p)
        g.restore(p)
    one()                                                         # P7: after a restore
    one()                                                         # P8: nothing changed
    return snaps, g


@needs_native
def test_off_is_the_base_exactly(monkeypatch, caplog):
    a, _ = _run(ORG, False, monkeypatch, caplog)
    b, _ = _run(BASE, False, monkeypatch, caplog)
    assert a == b
    assert len(a[-1]["wants"]) == 6            # 2 old + 3 from c1 (after the rewrite) + 1 from c6; c4 is not conversational


@needs_native
def test_on_equals_off_after_every_pass(monkeypatch, caplog):
    off, _ = _run(ORG, False, monkeypatch, caplog)
    on, g = _run(ORG, True, monkeypatch, caplog)
    for i, (x, y) in enumerate(zip(off, on), start=1):
        assert x == y, f"pass P{i} differs"
    # the sequence did what it says: the third want appeared (P3), the deleted one came back (P4), the flaky one
    # failed once (P5) and was created next pass (P6)
    texts = lambda s: sorted(t for t, *_ in s["wants"].values())  # noqa: E731
    assert "a third want" in texts(on[2]) and "a third want" not in texts(on[1])
    assert "revisit the pth cleanup" in texts(on[3])
    assert any(r[0] == "_cc_report_want_create_failures" and r[3] == 1 for r in on[4]["reports"])
    assert "a flaky want" not in texts(on[4]) and "a flaky want" in texts(on[5])
    assert sum(len(s["details"]) for s in on) == 6     # c2's six skipped markers, two per pass, none twice


@needs_native
def test_an_unchanged_text_is_not_parsed_again(monkeypatch, caplog):
    calls = []
    real = ORG.parse_wants
    monkeypatch.setattr(ORG, "parse_wants", lambda c: (calls.append(c), real(c))[1])
    snaps, g = _run(ORG, True, monkeypatch, caplog)
    st = g.nodes.wants_scan_stats()
    assert st["todo"] == 0 and st["recorded"] >= 3     # P8: nothing to parse; c1, c2, c6 recorded
    assert st["walked"] == len(g.nodes)


@needs_native
def test_the_record_follows_rows_and_clear_empties_it():
    g, vdb = _build()
    ORG._surface_wants_native(g, vdb)
    before = g.nodes.wants_scan_stats()["recorded"]
    assert before >= 1
    for i in range(200):                                # enough deletes to trigger a compaction
        g.create_node(node_id=f"tmp{i}", metadata={"creation_mode": "conversational"})
    for i in range(200):
        g.remove_node(f"tmp{i}")
    g.nodes.compact_now()
    assert g.nodes.wants_scan_stats()["recorded"] == before
    _, todo = g.nodes.wants_scan(vdb.content)
    assert todo == []                                   # records still match their rows after the compaction
    g.nodes.clear()
    assert g.nodes.wants_scan_stats()["recorded"] == 0


def test_on_without_the_native_node_store_raises(monkeypatch):
    monkeypatch.setattr(ORG, "WANTS_SCAN_NATIVE", True)
    g = NF.Graph(native_node_store=False)
    with pytest.raises(RuntimeError, match="native NodeStore"):
        ORG.surface_wants(g, SimpleVectorDB(dim=4))
