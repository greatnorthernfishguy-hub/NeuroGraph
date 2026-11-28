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
    for i in range(12):
        g.create_node(node_id=f"n{i}", metadata={"k": i})
    kinds = (SynapseType.EXCITATORY, SynapseType.INHIBITORY, SynapseType.MODULATORY)
    for i in range(30):
        syn = g.create_synapse(f"n{i % 12}", f"n{(i * 5 + 1) % 12}", weight=0.05 + 0.01 * i, synapse_type=kinds[i % 3])
        if i % 7 == 0 and syn is not None:
            sid = getattr(syn, "synapse_id", None)
            if sid is not None:
                g.synapses[sid].metadata = {"origin": "test", "i": i}
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
