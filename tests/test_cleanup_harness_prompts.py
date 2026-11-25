# ---- Changelog ----
# [2026-11-25] Claude (Fable 5.2, laptop; punch #412 bucket) -- the cleaner's harness-prompt finder and selection
# What: the harness's own flag decides (a user entry's promptSource 'system', a queued task-notification); a text
#       somebody also typed is not the harness's alone; a node is matched on its exact text; scope 'unconnected'
#       keeps a concept node that has a synapse to anything outside the matched set; unreadable input is skipped.
# Why:  Josh 2026-11-25: clean the existing harness nodes "after a dry run".
# How:  small JSONL files and a plain stand-in graph; no engine graph is built.
# -------------------
import json
from types import SimpleNamespace

import pytest

import cleanup_cc_tool_noise as C

NOTE = "<task-notification>\n<task-id>b1</task-id>\n</task-notification>"


def _write(tmp_path, name, *rows):
    p = tmp_path / name
    p.write_text("".join((r if isinstance(r, str) else json.dumps(r)) + "\n" for r in rows))
    return str(p)


def _user(text, source=None, blocks=False):
    d = {"type": "user", "message": {"content": [{"type": "text", "text": text}] if blocks else text}}
    if source:
        d["promptSource"] = source
    return d


def _queued(text, kind=None):
    a = {"type": "queued_command", "prompt": text}
    if kind:
        a["origin"] = {"kind": kind}
    return {"type": "attachment", "attachment": a}


def test_the_harness_flag_decides_and_a_text_also_typed_is_not_the_harness_alone(tmp_path):
    t1 = _write(tmp_path, "a.jsonl", _user(NOTE, "system"), _user("hello", "sdk"), _queued("queued note", "task-notification"),
                _queued("typed while working"), _user("blocks", "system", blocks=True), "not json", [1],
                {"type": "user", "promptSource": "system", "message": {"content": [{"type": "tool_result", "content": "x"}]}},
                {"type": "user", "promptSource": "system", "message": {"content": [{"type": "text", "text": None}]}})
    t2 = _write(tmp_path, "b.jsonl", _user("both", "system"), _user("both", "sdk"), _user(NOTE, None))
    flagged, other = C.harness_prompt_hashes([t1, str(tmp_path / "missing.jsonl"), str(tmp_path)])
    h = C.harness_hash
    assert flagged == {h(NOTE), h("queued note"), h("blocks")} and other == {h("hello"), h("typed while working")}
    flagged, other = C.harness_prompt_hashes([t1, t2])
    assert h("both") in flagged and h("both") in other and h(NOTE) in other     # typed too, in another session

    def node(text, **meta):
        return SimpleNamespace(metadata=dict(meta, _forest_content=text) if text is not None else dict(meta))
    g = SimpleNamespace(nodes={"n1": node("queued note"), "n2": node("queued note", _tree_concept=True), "n3": node("hello"),
                               "n4": node("both"), "n5": node(NOTE), "n6": node(None), "n7": node("queued note and more"),
                               "n8": SimpleNamespace(metadata=None)})
    assert sorted(C.find_harness_prompt_nodes(g, flagged, other)) == ["n1", "n2"]


def _graph():
    def node(**meta):
        return SimpleNamespace(metadata=meta)
    nodes = {"turn": node(), "turn::window::0": node(), "w2": node(_window=True), "c_in": node(_tree_concept=True),
             "c_out": node(_tree_concept=True), "c_alone": node(_tree_concept=True), "real": node(), "real_c": node(_tree_concept=True)}
    syn = lambda a, b: SimpleNamespace(pre_node_id=a, post_node_id=b)
    synapses = {1: syn("turn", "c_in"), 2: syn("c_in", "turn::window::0"), 3: syn("real_c", "c_out"), 4: syn("turn", "c_out"),
                5: syn("real", "turn"), 6: syn("real", "real_c")}
    return SimpleNamespace(nodes=nodes, synapses=synapses)


def test_scope_all_takes_every_matched_node_and_unconnected_keeps_a_concept_tied_outside():
    g = _graph()
    matched = ["turn", "turn::window::0", "w2", "c_in", "c_out", "c_alone", "gone"]
    r = C.select_harness_prompt_nodes(g, matched, "all")
    assert sorted(r["node_ids"]) == sorted(matched[:-1]) and r["by_kind"] == {"turn": 1, "window": 2, "concept": 3}
    assert r["concepts_kept_connected"] == 0
    r = C.select_harness_prompt_nodes(g, matched, "unconnected")
    assert sorted(r["node_ids"]) == ["c_alone", "c_in", "turn", "turn::window::0", "w2"]    # the turn goes though 'real' links to it
    assert r["by_kind"] == {"turn": 1, "window": 2, "concept": 2} and r["concepts_kept_connected"] == 1
    with pytest.raises(ValueError):
        C.select_harness_prompt_nodes(g, matched, "everything")
