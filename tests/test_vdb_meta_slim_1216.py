# ---- Changelog ----
# [2026-11-06] DudeMan CC (#1216 S1; Josh 2026-11-06 "As long as ... the text is still available, I'm fine with that")
#   — CREATE
# What: CC_NG_VDB_META_SLIM: (1) off = e2f93d4 exactly (graph checkpoint bytes; vector-DB entries, the entry metadata IS
#       the node's dict); (2) on: the graph checkpoint is byte-identical to off's, every vector-DB entry's text and
#       embedding equal off's, its metadata is {creation_mode, source}, and the text stays available to a direct read and
#       a search (Josh's condition); (3) cc_slim_vector_metadata trims existing entries, leaves text / embeddings alone,
#       is idempotent, and the trim survives save + load; (4) a deposit rolled back on a pre-existing node restores the
#       vector-DB entry with the switch on.
# Why:  spec docs/superpowers/specs/2026-11-06-node-metadata-and-text-in-rust-design.md S1 / §5.
# How:  a real neuro_foundation.Graph (native NodeStore) + universal_ingestor.SimpleVectorDB, deposits through the
#       journalled _CCConversationalDualPassEco; the base organism is loaded from git (NG_VDB_SLIM_BASE_REV, e2f93d4).
# [2026-11-06] DudeMan CC — review (Grok 4.7) I1: _tree_concept kept (Lenia's tree/forest split reads it from the
#   vector DB); the tests pin it per entry.
# -------------------
import importlib.util
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

BASE_REV = os.environ.get("NG_VDB_SLIM_BASE_REV", "e2f93d4")
DIM = 8


def _load_base():
    src = subprocess.run(["git", "-C", _REPO, "show", f"{BASE_REV}:cc_ng_organism.py"],
                         check=True, capture_output=True, text=True).stdout
    d = tempfile.mkdtemp(prefix="vdb_slim_base_")
    p = os.path.join(d, "org_base_vdb_slim.py")
    with open(p, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location("org_base_vdb_slim", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["org_base_vdb_slim"] = mod
    spec.loader.exec_module(mod)
    return mod


BASE = _load_base()
TURN = "a whole turn of conversation — Ünïcødé, several sentences long. " * 12


def _emb(i):
    v = np.zeros(DIM, dtype=np.float32)
    v[i % DIM] = 1.0
    v[(i + 3) % DIM] = 0.5
    return v


def _ckpt(g):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.msgpack")
        g.checkpoint(p)
        with open(p, "rb") as f:
            return f.read()


def _deposit_run(mod, slim, monkeypatch):
    if hasattr(mod, "VDB_META_SLIM"):
        monkeypatch.setattr(mod, "VDB_META_SLIM", slim)
    g = NF.Graph(native_node_store=True)
    vdb = SimpleVectorDB(dim=DIM)
    eco = mod._CCConversationalDualPassEco(g, vdb)
    forest = {"creation_mode": "conversational", "source": "cc_gateway", "cc": True, "_forest_content": TURN}
    eco.deposit("cc:conv::f1", _emb(0), TURN, forest, kind="forest")
    for i in range(3):
        tree = dict(forest, _tree_concept=True, _concept=f"concept {i}", _forest_target_id="cc:conv::f1")
        eco.deposit(f"cc:conv::f1::tree::{i}", _emb(i + 1), f"concept {i}", tree, kind="tree")
    eco.deposit("cc:conv::f1::window::0", _emb(5), "", dict(forest, _window=True, _window_index=0,
                                                              _forest_id="cc:conv::f1"),
                index_in_recall=False, kind="window")
    return g, vdb


def _entries(vdb):
    return {k: (vdb.embeddings[k].tobytes(), vdb.content[k], dict(vdb.metadata[k])) for k in vdb.content}


def test_off_is_the_base_exactly(monkeypatch):
    ga, va = _deposit_run(ORG, False, monkeypatch)
    gb, vb = _deposit_run(BASE, False, monkeypatch)
    assert _ckpt(ga) == _ckpt(gb)
    assert _entries(va) == _entries(vb)
    assert all(va.metadata[k] is ga.nodes[k].metadata for k in va.content)     # off: the node's own dict, as before


def test_on_keeps_the_graph_and_the_text_and_slims_only_the_entry_metadata(monkeypatch):
    goff, voff = _deposit_run(ORG, False, monkeypatch)
    gon, von = _deposit_run(ORG, True, monkeypatch)
    assert _ckpt(gon) == _ckpt(goff)                                    # the graph is untouched
    off, on = _entries(voff), _entries(von)
    assert set(on) == set(off) == {"cc:conv::f1"} | {f"cc:conv::f1::tree::{i}" for i in range(3)}
    for k in on:
        assert on[k][:2] == off[k][:2]                                  # embedding and text equal
        assert on[k][2] == {kk: off[k][2][kk] for kk in ("creation_mode", "source", "_tree_concept") if kk in off[k][2]}
        assert von.get_metadata(k).get("_tree_concept") == voff.get_metadata(k).get("_tree_concept")   # Lenia
        assert "_forest_content" in gon.nodes[k].metadata               # every key still on the node
    # Josh's condition: the text is still there for a direct read and a search
    assert von.get_content("cc:conv::f1") == TURN
    assert von.get("cc:conv::f1")["content"] == TURN
    hits = von.search(_emb(0), k=1)
    assert hits and hits[0][0] == "cc:conv::f1"


def test_the_load_trim_slims_old_entries_and_keeps_text(monkeypatch):
    g, vdb = _deposit_run(ORG, False, monkeypatch)                      # full metadata, as today's file has
    before = {k: (vdb.embeddings[k].tobytes(), vdb.content[k]) for k in vdb.content}
    n, trimmed = ORG.cc_slim_vector_metadata(vdb)
    assert (n, trimmed) == (4, 4)
    assert all(set(vdb.metadata[k]) <= {"creation_mode", "source", "_tree_concept"} for k in vdb.content)
    assert [k for k in vdb.content if vdb.metadata[k].get("_tree_concept")] == [f"cc:conv::f1::tree::{i}" for i in range(3)]
    assert {k: (vdb.embeddings[k].tobytes(), vdb.content[k]) for k in vdb.content} == before
    assert "_forest_content" in g.nodes["cc:conv::f1"].metadata         # the node's dict is not the one trimmed
    assert ORG.cc_slim_vector_metadata(vdb) == (4, 0)                   # idempotent
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "vectors.msgpack")
        vdb.save(p)
        v2 = SimpleVectorDB(dim=DIM)
        v2.load(p)
    assert {k: v2.metadata[k] for k in v2.content} == {k: vdb.metadata[k] for k in vdb.content}
    assert v2.get_content("cc:conv::f1") == TURN


def test_a_rolled_back_deposit_restores_the_entry_with_the_switch_on(monkeypatch):
    g, vdb = _deposit_run(ORG, True, monkeypatch)
    before = _entries(vdb)
    eco = ORG._CCConversationalDualPassEco(g, vdb)
    meta = {"creation_mode": "conversational", "source": "elsewhere", "_forest_content": "rewritten"}
    eco.deposit("cc:conv::f1", _emb(7), "rewritten", meta, kind="forest")      # a pre-existing node, re-stamped
    assert vdb.get_content("cc:conv::f1") == "rewritten"
    assert eco.rollback()
    assert _entries(vdb) == before


def test_slim_entry_metadata_keeps_only_present_keys():
    assert ORG.cc_vdb_entry_metadata({"creation_mode": "x", "other": 1}) == {"creation_mode": "x"}
    assert ORG.cc_vdb_entry_metadata(None) == {}
    assert ORG.cc_slim_vector_metadata(object()) == (0, 0)
