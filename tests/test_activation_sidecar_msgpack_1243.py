# ---- Changelog ----
# [2026-11-18 15:45] DudeMan CC (#1243) -- rewritten for Josh's ruling of the same day: msgpack is the only format
#   written or verified; JSON is read only as a temporary catchment for the old file still on disk; the file name stays.
# [2026-11-18] DudeMan CC (#1243; Josh 2026-11-18 "you have my go and Proceed for ... #1243") -- created.
# What: write_state() writes msgpack under the unchanged <checkpoint>.activations.json name, bit-exact floats, the
#       same envelope; restore() reads it; the old JSON sidecar is still read (catchment) and the next save replaces
#       it with msgpack; openclaw_hook._sidecar_defect accepts the msgpack file and rejects a JSON one.
# Why:  saves are msgpack, never JSON (Josh 2026-10-13; Format-for-Purpose).
# How:  a plain dict of node objects as the graph (the node-loop capture path) and a numpy `columns` double for the
#       NodeStore path; _sidecar_defect is run from its own source (importing openclaw_hook loads the whole stack).
# -------------------
"""#1243: the activation sidecar is msgpack; the old JSON file is read once."""

import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import msgpack

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import activation_persistence as mod  # noqa: E402
from activation_persistence import ActivationPersistence  # noqa: E402
from ces_config import CESConfig  # noqa: E402

SUFFIX = ".activations.json"      # the historical name; kept


def _graph():
    nodes = {
        "a": SimpleNamespace(voltage=0.1 + 0.2, resting_potential=0.0, last_spike_time=41.0, intrinsic_excitability=1.25),
        "b": SimpleNamespace(voltage=-3.75, resting_potential=0.0, last_spike_time=float("-inf"), intrinsic_excitability=0.5),
        "rest": SimpleNamespace(voltage=0.0, resting_potential=0.0, last_spike_time=0.0, intrinsic_excitability=1.0),
        "ünï::tree::x y": SimpleNamespace(voltage=18.01178581431049, resting_potential=0.0, last_spike_time=673706.0,
                                          intrinsic_excitability=4.098172631326843),
    }
    return SimpleNamespace(nodes=nodes, timestep=673708)


def _fresh_graph():
    g = _graph()
    for n in g.nodes.values():
        n.voltage = 0.0
        n.intrinsic_excitability = 1.0
    return g


def _ap():
    return ActivationPersistence(CESConfig())


def _sidecar_defect():
    """openclaw_hook._sidecar_defect, compiled from its own source."""
    src = (ROOT / "openclaw_hook.py").read_text()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "_sidecar_defect")
    ns = {"json": json}
    exec("from typing import Any, Optional\n" + ast.get_source_segment(src, fn), ns)
    return ns["_sidecar_defect"]


def test_the_name_is_unchanged_and_the_content_is_msgpack(tmp_path):
    ap = _ap()
    ckpt = str(tmp_path / "main.msgpack")
    captured = ap.capture_state(_graph())
    path = ap.write_state(ckpt, captured)
    assert path == ckpt + SUFFIX
    raw = Path(path).read_bytes()
    assert raw == msgpack.packb(captured, use_bin_type=True)
    assert raw[:1] == b"\x84"                                    # the four-key envelope as a msgpack map, not "{"
    assert msgpack.unpackb(raw, raw=False) == captured            # every key, every float64, -inf, the non-ASCII id
    assert sorted(captured["entries"]) == ["a", "b", "ünï::tree::x y"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["main.msgpack" + SUFFIX]   # one file, no JSON twin


def test_restore_is_bit_exact_when_no_time_has_passed(tmp_path, monkeypatch):
    ap = _ap()
    ckpt = str(tmp_path / "main.msgpack")
    src = _graph()
    captured = ap.capture_state(src)
    ap.write_state(ckpt, captured)
    monkeypatch.setattr(mod.time, "time", lambda: captured["saved_at"])
    dst = _fresh_graph()
    assert _ap().restore(dst, ckpt) == 3
    for nid in ("a", "b", "ünï::tree::x y"):
        assert dst.nodes[nid].voltage == src.nodes[nid].voltage
        assert dst.nodes[nid].intrinsic_excitability == src.nodes[nid].intrinsic_excitability
    assert dst.nodes["rest"].voltage == 0.0


def test_receipt_and_stats_after_a_msgpack_write(tmp_path):
    ap = _ap()
    ckpt = str(tmp_path / "main.msgpack")
    captured = ap.capture_state(_graph())
    assert ap.write_state(ckpt, captured, with_receipt=True) == {"path": ckpt + SUFFIX, "saved_at": captured["saved_at"]}
    assert ap.get_stats()["entries_saved"] == 3 and ap.get_stats()["last_save_time"] == captured["saved_at"]


def test_old_json_sidecar_is_read_once_then_replaced_by_msgpack(tmp_path, monkeypatch):
    """The temporary catchment: the old engine's JSON file restores, and the next save leaves msgpack only."""
    ckpt = str(tmp_path / "main.msgpack")
    src = _graph()
    captured = _ap().capture_state(src)
    old = Path(ckpt + SUFFIX)
    old.write_text(json.dumps(captured))                          # what the old engine wrote
    monkeypatch.setattr(mod.time, "time", lambda: captured["saved_at"])
    dst = _fresh_graph()
    ap = _ap()
    assert ap.restore(dst, ckpt) == 3
    assert dst.nodes["ünï::tree::x y"].voltage == src.nodes["ünï::tree::x y"].voltage
    assert old.read_text() == json.dumps(captured)                # restore writes nothing
    ap.write_state(ckpt, ap.capture_state(dst))
    raw = old.read_bytes()
    assert raw[:1] == b"\x84" and b"{" != raw.lstrip()[:1]
    assert sorted(msgpack.unpackb(raw, raw=False)["entries"]) == ["a", "b", "ünï::tree::x y"]


def test_old_json_with_leading_whitespace_still_reads(tmp_path):
    ckpt = str(tmp_path / "main.msgpack")
    Path(ckpt + SUFFIX).write_text("\n  " + json.dumps(_ap().capture_state(_graph())))
    assert _ap().restore(_fresh_graph(), ckpt) == 3


def test_unreadable_sidecar_restores_nothing_and_does_not_raise(tmp_path):
    ckpt = str(tmp_path / "main.msgpack")
    for bad in (b"\x84\xa7version", b"{incomplete", b"", b"\x7b", msgpack.packb([1, 2, 3])):
        Path(ckpt + SUFFIX).write_bytes(bad)
        g = _fresh_graph()
        assert _ap().restore(g, ckpt) == 0
        assert all(n.voltage == 0.0 for n in g.nodes.values())


def test_empty_entries_round_trip(tmp_path):
    ap = _ap()
    ckpt = str(tmp_path / "main.msgpack")
    empty = dict(ap.capture_state(_graph()), entries={})
    ap.write_state(ckpt, empty)
    assert msgpack.unpackb(Path(ckpt + SUFFIX).read_bytes(), raw=False) == empty
    assert _ap().restore(_fresh_graph(), ckpt) == 0


def test_failed_write_raises_and_leaves_file_and_bookkeeping(tmp_path):
    ap = _ap()
    ckpt = str(tmp_path / "main.msgpack")
    good = ap.capture_state(_graph())
    ap.write_state(ckpt, good)
    before_stats, before_bytes = ap.get_stats(), Path(ckpt + SUFFIX).read_bytes()
    bad = dict(good, saved_at=good["saved_at"] + 5.0, timestep=2 ** 64)   # msgpack cannot pack this int
    try:
        ap.write_state(ckpt, bad)
    except Exception:
        pass
    else:
        raise AssertionError("a failed write must raise")
    assert ap.get_stats() == before_stats
    assert Path(ckpt + SUFFIX).read_bytes() == before_bytes


def test_decay_is_the_same_from_msgpack_and_from_the_old_json(tmp_path, monkeypatch):
    ckpt_j, ckpt_m = str(tmp_path / "j.msgpack"), str(tmp_path / "m.msgpack")
    captured = _ap().capture_state(_graph())
    monkeypatch.setattr(mod.time, "time", lambda: captured["saved_at"] + 7200.0)   # two hours later
    Path(ckpt_j + SUFFIX).write_text(json.dumps(captured))
    _ap().write_state(ckpt_m, captured)
    gj, gm = _fresh_graph(), _fresh_graph()
    assert _ap().restore(gj, ckpt_j) == _ap().restore(gm, ckpt_m) == 3
    for nid in ("a", "b"):
        assert gj.nodes[nid].voltage == gm.nodes[nid].voltage == _graph().nodes[nid].voltage * (0.9 ** 2.0)


def test_native_columns_capture_packs_as_msgpack(tmp_path):
    """The NodeStore path hands numpy columns; .tolist() makes plain floats, which msgpack takes."""
    import numpy as np
    g = _graph()
    ids = list(g.nodes)

    class Nodes(dict):
        @staticmethod
        def columns(names):
            return (ids, *[np.array([getattr(g.nodes[i], n) for i in ids], dtype=np.float64) for n in names])

    ap = _ap()
    ckpt = str(tmp_path / "main.msgpack")
    captured = ap.capture_state(SimpleNamespace(nodes=Nodes(g.nodes), timestep=g.timestep))
    assert sorted(captured["entries"]) == ["a", "b", "ünï::tree::x y"]
    ap.write_state(ckpt, captured)
    assert msgpack.unpackb(Path(ckpt + SUFFIX).read_bytes(), raw=False) == captured


def test_sidecar_defect_accepts_the_msgpack_file_and_rejects_json(tmp_path):
    defect = _sidecar_defect()
    ap = _ap()
    ckpt = str(tmp_path / "main.msgpack")
    captured = ap.capture_state(_graph())
    receipt = ap.write_state(ckpt, captured, with_receipt=True)
    assert defect(receipt["path"], receipt["saved_at"]) is None
    assert "not the writer" in defect(receipt["path"], receipt["saved_at"] + 1.0)
    assert defect(receipt["path"], None) is not None
    Path(receipt["path"]).write_text(json.dumps(captured))        # a JSON file after a save is a defect now
    assert "unreadable" in defect(receipt["path"], receipt["saved_at"]) or "entries" in defect(receipt["path"], receipt["saved_at"])
    Path(receipt["path"]).write_bytes(msgpack.packb({"saved_at": receipt["saved_at"]}))
    assert "entries" in defect(receipt["path"], receipt["saved_at"])
