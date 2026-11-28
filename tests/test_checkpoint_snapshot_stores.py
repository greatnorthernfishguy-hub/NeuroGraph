# ---- Changelog ----
# [2026-11-27] DudeMan CC (Claude Fable 5.2, laptop; quest SQ11, punch #1201) -- CREATE
# What: pins Graph.checkpoint_snapshot_stores: off, the capture holds the packed synapse bytes as before; on, it
#       holds a detached snapshot and the file written is byte-identical to the off file; changes made to the graph
#       between capture and write are not in the file; a restore of the on-file gives the same graph; the organism's
#       enabler needs the env, the engine attribute and a wheel with checkpoint_snapshot.
# Why:  a save can then copy the synapse store inside the step lock and pack it after.
# How:  the real Graph (native synapse store) with nodes, synapses of all three kinds and synapse metadata.
#       Needs ng_tract with SynapseStore.checkpoint_snapshot (0fa35d9 or later); skipped on an older wheel.
# -------------------
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import neuro_foundation as nf
from neuro_foundation import CheckpointMode, Graph, SynapseType


def _graph():
    g = Graph()
    # step 2: something in the prediction book's two maps, so the packed path has content to get wrong
    for i in range(12):
        g.create_node(node_id=f"n{i}", metadata={"k": i})
    kinds = (SynapseType.EXCITATORY, SynapseType.INHIBITORY, SynapseType.MODULATORY)
    for i in range(30):
        syn = g.create_synapse(f"n{i % 12}", f"n{(i * 5 + 1) % 12}", weight=0.05 + 0.01 * i, synapse_type=kinds[i % 3])
        if i % 7 == 0 and syn is not None:
            sid = getattr(syn, "synapse_id", None)
            if sid is not None:
                g.synapses[sid].metadata = {"origin": "test", "i": i}
    # step 2: content in the prediction book's two maps (real synapse and node ids, so a restore keeps them)
    for i, sid in enumerate(list(g.synapses.keys())[:25]):
        g._pb.hist_set(sid, [(i + j) % 2 == 0 for j in range(1 + i % 7)])
    for i in range(6):
        g._pb.spike_set(f"n{i}", [40 + i, 41 + i, 42 + i][: 1 + i % 3], 16)
    return g


pytestmark = pytest.mark.skipif(not hasattr(Graph().synapses, "checkpoint_snapshot"),
                                reason="ng_tract without SynapseStore.checkpoint_snapshot")


def _file(g, path):
    g.checkpoint(str(path), CheckpointMode.FULL)
    return open(path, "rb").read()


def test_default_is_off_and_the_capture_holds_bytes(tmp_path):
    g = _graph()
    assert Graph.checkpoint_snapshot_stores is False and g.checkpoint_snapshot_stores is False
    cap = g.capture_checkpoint(CheckpointMode.FULL)
    assert isinstance(cap["synapses"], (bytes, bytearray))


def test_on_the_capture_holds_a_snapshot_and_the_file_is_byte_identical(tmp_path):
    g = _graph()
    off = _file(g, tmp_path / "off.msgpack")
    g.checkpoint_snapshot_stores = True
    cap = g.capture_checkpoint(CheckpointMode.FULL)
    assert not isinstance(cap["synapses"], (bytes, bytearray)) and hasattr(cap["synapses"], "to_msgpack")
    assert len(cap["synapses"]) == len(g.synapses)
    g.write_checkpoint(str(tmp_path / "on.msgpack"), cap, CheckpointMode.FULL)
    assert open(tmp_path / "on.msgpack", "rb").read() == off
    g.write_checkpoint(str(tmp_path / "on2.msgpack"), cap, CheckpointMode.FULL)      # a retry writes the same file
    assert open(tmp_path / "on2.msgpack", "rb").read() == off


def test_on_the_file_is_the_instant_of_the_capture_not_of_the_write(tmp_path):
    g = _graph()
    g.checkpoint_snapshot_stores = True
    cap = g.capture_checkpoint(CheckpointMode.FULL)
    g.checkpoint_snapshot_stores = False
    at_capture = _file(g, tmp_path / "ref.msgpack")
    # the graph moves on after the capture
    sid = next(iter(g.synapses.keys()))
    g.synapses[sid].weight = 4.5
    g.create_synapse("n0", "n11", weight=0.9)
    g.write_checkpoint(str(tmp_path / "later.msgpack"), cap, CheckpointMode.FULL)
    assert open(tmp_path / "later.msgpack", "rb").read() == at_capture


def test_on_file_restores_to_the_same_graph(tmp_path):
    g = _graph()
    g.checkpoint_snapshot_stores = True
    path = tmp_path / "on.msgpack"
    g.checkpoint(str(path), CheckpointMode.FULL)
    h = Graph()
    h.restore(str(path))
    assert len(h.synapses) == len(g.synapses) and len(h.nodes) == len(g.nodes)
    assert bytes(h.synapses.to_checkpoint_msgpack()) == bytes(g.synapses.to_checkpoint_msgpack())


def test_the_step_lock_is_released_after_a_snapshot_capture():
    g = _graph()
    g.checkpoint_snapshot_stores = True
    g.capture_checkpoint(CheckpointMode.FULL)
    assert not g._step_lock._is_owned()


def test_organism_enabler(monkeypatch):
    import cc_ng_organism as org
    g = _graph()
    monkeypatch.setattr(org, "CKPT_SNAPSHOT", False)
    assert org.cc_enable_checkpoint_snapshot(g) is False and g.checkpoint_snapshot_stores is False
    monkeypatch.setattr(org, "CKPT_SNAPSHOT", True)
    assert org.cc_enable_checkpoint_snapshot(None) is False
    with pytest.raises(RuntimeError):
        org.cc_enable_checkpoint_snapshot(object())                       # no engine attribute
    assert org.cc_enable_checkpoint_snapshot(g) is True and g.checkpoint_snapshot_stores is True
    assert Graph.checkpoint_snapshot_stores is False                      # the instance, not the class


# ---- [2026-11-27] step 2: the prediction book's two maps as packed bytes ----

def test_off_the_two_book_maps_are_dicts_as_before():
    g = _graph()
    cap = g.capture_checkpoint(CheckpointMode.FULL)
    assert cap["synapse_confirmation_history"] == {sid: list(h) for sid, h in g._pb.hist_items()}
    assert len(cap["synapse_confirmation_history"]) == 25
    assert cap["recent_spikes"] == {nid: list(sp) for nid, sp, _c in g._pb.spike_items(True)}
    assert cap["recent_spikes"]["n1"] == [41, 42]


def test_on_the_two_book_maps_are_bytes_and_decode_to_the_same_dicts():
    import msgpack
    g = _graph()
    off = g.capture_checkpoint(CheckpointMode.FULL)
    g.checkpoint_snapshot_stores = True
    on = g.capture_checkpoint(CheckpointMode.FULL)
    for key in ("synapse_confirmation_history", "recent_spikes"):
        assert isinstance(on[key], (bytes, bytearray))
        assert bytes(on[key]) == msgpack.packb(off[key], use_bin_type=True)
    assert list(on) == list(off)                                   # the same keys in the same order


def test_on_restore_gives_back_the_history_and_the_spikes(tmp_path):
    g = _graph()
    g.checkpoint_snapshot_stores = True
    path = tmp_path / "on.msgpack"
    g.checkpoint(str(path), CheckpointMode.FULL)
    g.checkpoint_snapshot_stores = False
    ref = tmp_path / "off.msgpack"
    g.checkpoint(str(ref), CheckpointMode.FULL)
    assert open(path, "rb").read() == open(ref, "rb").read()
    h, r = Graph(), Graph()
    h.restore(str(path))
    r.restore(str(ref))
    assert h._pb.hist_items() == r._pb.hist_items() and len(h._pb.hist_items()) == 25
    assert h._pb.spike_items(True) == r._pb.spike_items(True) and h._pb.spike_items(True)
