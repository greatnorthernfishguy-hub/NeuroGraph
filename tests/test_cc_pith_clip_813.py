# tests/test_cc_pith_clip_813.py
#
# ---- Changelog ----
# [2026-09-30] Z12 worker (Claude Sonnet 5.5, Claude Code) — #813 Pith clip removal tests
# What: covers (1) a well-formed short node renders byte-identically to BASE e4ebf982
#   (golden), (2) an over-700 node renders WHOLE, (3) when the budget binds whole
#   lowest-relevance assemblies are dropped and ONE INFO line reports count + total
#   size, (4) no keyframe/word-cut call remains in any budgeted path (a keyframe
#   without its delta is a cut), (5) the CC_PITH_PROVIDER_NODE_CHARS knob is gone from
#   the config surface and the host contract, (6) the staged .bashrc script is
#   reversible, name-only and refuses ambiguous matches.
# Why: Exec P411/P413 (Josh: no truncation); assignment build-813-pith-clip.md steps 2-3.
# How: fake in-memory graph only (tests/pith_clip_813_scenarios.py) -- never a live path,
#   checkpoint, daemon or the real ~/.bashrc. P379 preamble first: NG_EMBED_* scrubbed,
#   and the run FAILS if cc_ng_organism is not this worktree's copy.
# -------------------
import ast
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace

# --- P379 preamble: resolve the module under test, scrub embed env, then import ---
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _k in [k for k in os.environ if k.startswith("NG_EMBED_")]:
    del os.environ[_k]
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "tests"))

import pytest

import cc_ng_organism as pith
from pith_clip_813_scenarios import FakeGraph, _core, _run, build_scenarios

_ORGANISM_SRC = os.path.join(_ROOT, "cc_ng_organism.py")


def test_p379_module_under_test_is_the_worktree_copy():
    resolved = os.path.realpath(pith.__file__)
    print("cc_ng_organism resolved to:", resolved)
    assert resolved == os.path.realpath(_ORGANISM_SRC), (
        "cc_ng_organism was imported from %s, not this worktree" % resolved)
    assert not any(k.startswith("NG_EMBED_") for k in os.environ)


@pytest.fixture(autouse=True)
def _reset_metrics():
    pith._PITH_METRICS.reset()
    yield
    pith._PITH_METRICS.reset()


def _drop_records(caplog, marker):
    return [r for r in caplog.records
            if r.levelno == logging.INFO and marker in r.getMessage()]


# ---------------------------------------------------------------- golden vs BASE

def test_short_wellformed_nodes_render_exactly_as_base():
    golden = json.load(open(os.path.join(
        _ROOT, "tests", "fixtures", "pith_clip_813_golden_base.json")))
    assert golden["base_commit"].startswith("e4ebf982")
    now = build_scenarios(pith)
    assert set(now) == set(golden["scenarios"]) and len(now) == 4
    for name, expected in golden["scenarios"].items():
        assert now[name] == expected, name


# ------------------------------------------------------------ nodes render WHOLE

def test_over_700_node_text_is_returned_whole():
    long_text = "word " * 600                      # 3000 chars
    node = SimpleNamespace(metadata={"_forest_content": long_text})
    assert pith._pith_node_text(node) == long_text.strip()


def test_over_700_root_and_relation_render_whole_through_provider_context():
    g = FakeGraph()
    _core(g)
    root_text = "root " + ("alpha detail. " * 100)           # ~1400 chars, > 700
    rel_text = "member " + ("beta detail. " * 150)           # ~1950 chars, > 700
    g.node("r", root_text, source="cc_gateway")
    g.node("m", rel_text, source="cc_gateway")
    g.synapse("s", "r", "m", 0.9)
    result = _run(pith, g, [{"node_id": "r", "score": 1.0}], "go on", budget_chars=8000)
    assert result["state"] == "ok"
    assert root_text.strip() in result["context"]
    assert rel_text.strip() in result["context"]
    assert "⋯" not in result["context"] and " …" not in result["context"]
    assert len(result["context"]) <= 8000


def test_source_label_longer_than_80_is_not_cut():
    label = "src-" + "x" * 116                                # 120 chars
    g = FakeGraph()
    _core(g)
    g.node("r", "a short root", source=label)
    result = _run(pith, g, [{"node_id": "r", "score": 1.0}], "go on")
    assert "- Sources: " + label in result["context"]


# ------------------------------------------- budget bound: whole items, loud drop

def _four_assemblies():
    g = FakeGraph()
    _core(g)
    surfaced = []
    for index in range(4):
        g.node(f"r{index}", f"root{index} " + ("situation " * 30), source="cc_gateway")
        g.node(f"m{index}", f"member{index} " + ("relationship " * 25), source="cc_gateway")
        g.synapse(f"s{index}", f"r{index}", f"m{index}", 0.9)
        surfaced.append({"node_id": f"r{index}", "score": 4.0 - index})
    return g, surfaced


def test_budget_binding_drops_lowest_ranked_whole_assemblies_with_one_info_line(caplog):
    g, surfaced = _four_assemblies()
    lines = pith.pith_connected_activation_basins(g, surfaced)      # ranked, unmodified
    assert [l.node_id for l in lines] == ["r0", "r1", "r2", "r3"]
    renders = [pith._pith_render_connected_line(l) for l in lines]
    core = pith.render_constitutional_core(g)
    # Choose a total budget that admits exactly the top two.  The organism reserves the
    # section/alert delimiters for ALL candidates before admission, so mirror that.
    shell, _ = pith._pith_provider_sections(core, lines, [""] * len(lines))
    budget = len(shell) + len(renders[0]) + 2 + len(renders[1])
    assert 500 <= budget <= 40000

    with caplog.at_level(logging.INFO, logger=pith.logger.name):
        result = _run(pith, g, surfaced, "go on", budget_chars=budget)

    assert result["state"] == "ok" and result["assemblies"] == 2
    assert len(result["context"]) <= budget
    for kept in (0, 1):                                              # whole, byte for byte
        assert renders[kept] in result["context"]
    for dropped in (2, 3):
        assert f"root{dropped}" not in result["context"]
        assert f"member{dropped}" not in result["context"]

    records = _drop_records(caplog, "whole assemblies")
    assert len(records) == 1, [r.getMessage() for r in caplog.records]
    message = records[0].getMessage()
    expected_chars = len(renders[2]) + len(renders[3])
    assert "dropping 2 whole assemblies" in message
    assert f"({expected_chars} chars rendered)" in message
    assert "kept 2" in message


def test_no_drop_no_info_line(caplog):
    g, surfaced = _four_assemblies()
    with caplog.at_level(logging.INFO, logger=pith.logger.name):
        result = _run(pith, g, surfaced, "go on", budget_chars=40000)
    assert result["assemblies"] == 4
    assert _drop_records(caplog, "whole assemblies") == []


def _cl(node_id, score, pad, member="tail"):
    return pith.CacheLine(
        node_id=node_id, content=f"{node_id} " + ("x" * pad), score=score,
        member_node_ids=[node_id, member],
        relations=[{"from": node_id, "to": member, "kind": "learned successor",
                    "content": f"{member} of {node_id}"}],
        sources=["cc_gateway"], stream="connected")


def test_admit_is_strict_rank_prefix_and_a_dropped_line_is_never_shortened(caplog):
    a, b, c = _cl("a", 3.0, 100), _cl("b", 2.0, 400), _cl("c", 1.0, 50)
    ra, rb, rc = (len(pith._pith_render_connected_line(x)) for x in (a, b, c))
    budget = ra + 2 + rb - 1            # b fits an empty envelope but not the remainder
    assert rb <= budget and rc + 2 <= budget - ra                    # c WOULD fit the remainder
    with caplog.at_level(logging.INFO, logger=pith.logger.name):
        kept, blocks = pith._pith_provider_admit([c, b, a], budget)
    assert [x.node_id for x in kept] == ["a"]                        # c does not jump b
    assert blocks == [pith._pith_render_connected_line(a)]
    (record,) = _drop_records(caplog, "whole assemblies")
    assert "dropping 2 whole assemblies" in record.getMessage()
    assert f"({rb + rc} chars rendered)" in record.getMessage()


def test_admit_skips_an_assembly_that_can_never_fit_and_keeps_the_rest(caplog):
    giant, small1, small2 = _cl("g", 9.0, 5000), _cl("s1", 2.0, 60), _cl("s2", 1.0, 60)
    rg = len(pith._pith_render_connected_line(giant))
    r1 = len(pith._pith_render_connected_line(small1))
    r2 = len(pith._pith_render_connected_line(small2))
    budget = r1 + 2 + r2
    assert rg > budget
    with caplog.at_level(logging.INFO, logger=pith.logger.name):
        kept, _blocks = pith._pith_provider_admit([giant, small1, small2], budget)
    assert [x.node_id for x in kept] == ["s1", "s2"]
    (record,) = _drop_records(caplog, "whole assemblies")
    assert "dropping 1 whole assemblies" in record.getMessage()
    assert f"({rg} chars rendered)" in record.getMessage()


def test_nothing_fits_is_still_the_closed_capacity_empty_state():
    only = _cl("a", 1.0, 9000)
    kept, blocks = pith._pith_provider_admit([only], 600)
    assert kept == [] and blocks == []


# ------------------------------------------------- Stage 3: whole or dropped

def test_stage3_over_budget_line_is_dropped_whole_not_keyframed(caplog):
    top = pith.CacheLine.from_surfaced("top", "A" * 50, score=10.0, stream="pattern")
    mid_text = ("the checkpoint save cadence changed after the rebuild. " * 8).strip()
    mid = pith.CacheLine.from_surfaced("mid", mid_text, score=5.0, stream="pattern")
    low = pith.CacheLine.from_surfaced("low", "z" * 10, score=1.0, stream="pattern")
    with caplog.at_level(logging.INFO, logger=pith.logger.name):
        out = pith.pith_stage3([top, mid, low], budget_chars=300)
    assert [l.node_id for l in out] == ["top"]
    assert mid.content == mid_text                                   # never rewritten in place
    assert pith._PITH_METRICS.compressed_count == 0
    assert pith._PITH_METRICS.chars_saved == 0
    (record,) = _drop_records(caplog, "whole items")
    assert "dropping 2 whole items" in record.getMessage()
    assert f"({len(mid_text) + 10} chars)" in record.getMessage()


def test_stage3_never_returns_an_empty_l1_and_logs_nothing_when_nothing_dropped(caplog):
    big = pith.CacheLine.from_surfaced("big", "B" * 900, score=1.0, stream="pattern")
    with caplog.at_level(logging.INFO, logger=pith.logger.name):
        out = pith.pith_stage3([big], budget_chars=500)
    assert [l.node_id for l in out] == ["big"] and out[0].content == "B" * 900
    assert _drop_records(caplog, "whole items") == []


# --------------------------------------- recall snippet: whole for the provider only

def _fake_ng(text):
    graph = FakeGraph()
    graph.node("n", text)
    ng = SimpleNamespace(graph=graph)
    ng._harvest_associations = lambda *a, **k: [{"node_id": "n", "strength": 1.0}]
    return ng


def test_recall_default_still_bounds_the_snippet_and_whole_content_does_not(monkeypatch):
    monkeypatch.setattr(pith, "cc_gsg_rescore", lambda surfaced, *_a, **_k: surfaced)
    text = "sentence " * 150                                          # 1350 chars
    default = pith.cc_pattern_completion_recall(_fake_ng(text), "q", 5)
    whole = pith.cc_pattern_completion_recall(_fake_ng(text), "q", 5, whole_content=True)
    assert default[0]["content"].endswith("…") and len(default[0]["content"]) <= 301
    assert whole[0]["content"] == text.strip()


def test_provider_context_asks_recall_for_whole_content(monkeypatch):
    seen = {}

    def fake_recall(*_args, **kwargs):
        seen.update(kwargs)
        return []

    monkeypatch.setattr(pith, "cc_pattern_completion_recall", fake_recall)
    g = FakeGraph()
    _core(g)
    pith.pith_provider_context(SimpleNamespace(graph=g), "hello")
    assert seen.get("whole_content") is True


# ------------------------------------------------ structural: no cut in budgeted paths

_BUDGETED = ("pith_stage3", "_pith_node_text", "_pith_fit_connected_line",
             "_pith_provider_admit", "pith_provider_context",
             "cc_pattern_completion_recall", "pith_connected_activation_basins",
             "_pith_node_sources")
_CUTTERS = {"pith_stage2_keyframe", "_pith_cut_at_word_boundary", "_pith_fit_statement"}


def test_no_keyframe_or_word_cut_call_in_any_budgeted_path():
    tree = ast.parse(open(_ORGANISM_SRC).read())
    defs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "_pith_fit_statement" not in defs
    for name in _BUDGETED:
        called = {c.func.id for c in ast.walk(defs[name])
                  if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
        assert not (called & _CUTTERS), (name, called & _CUTTERS)


def test_node_chars_knob_is_gone_from_the_organism_config_surface():
    assert not hasattr(pith, "_CC_PITH_PROVIDER_NODE_CHARS")
    cfg = pith.pith_effective_config()
    # Retired: nothing in this process reads it. It is reported as not-a-live-setting
    # (None + authority "retired") rather than deleted, because env == resolved ==
    # _PITH_CONFIG_KEYS == the VPS host allow-list is asserted elsewhere and the host is
    # not edited in this change. A stale export therefore shows as env set / resolved None.
    assert cfg["resolved"]["CC_PITH_PROVIDER_NODE_CHARS"] is None
    assert cfg["authority"]["CC_PITH_PROVIDER_NODE_CHARS"].startswith("retired")
    assert "CC_PITH_PROVIDER_NODE_CHARS" in pith._PITH_CONFIG_KEYS
    assert "CC_PITH_PROVIDER_NODE_CHARS" in cfg["env"]
    # the other two provider caps are REJECT-LOUDLY guards and stay
    assert "CC_PITH_PROVIDER_MAX_INSTRUCTION_CHARS" in cfg["resolved"]
    assert "CC_PITH_PROVIDER_MAX_QUEST_CHARS" in cfg["resolved"]


def test_reject_loudly_guards_still_refuse_the_whole_request():
    g = FakeGraph()
    _core(g)
    ng = SimpleNamespace(graph=g)
    big = "x" * (pith._CC_PITH_PROVIDER_MAX_INSTRUCTION_CHARS + 1)
    r = pith.pith_provider_context(ng, big)
    assert r["state"] == "unavailable" and r["warnings"] == ["instruction_too_large"]
    r = pith.pith_provider_context(ng, "ok", "q" * (pith._CC_PITH_PROVIDER_MAX_QUEST_CHARS + 1))
    assert r["state"] == "unavailable" and r["warnings"] == ["invalid_quest_focus"]


# ------------------------------------------------------------------ the host contract

def test_host_contract_lists_five_exports_and_no_shortening_promise():
    text = open(os.path.join(_ROOT, "docs", "PITH_HOST_CONTRACT.md")).read()
    block = re.search(r"```bash\n(export CC_PITH_PROVIDER_ROOTS.*?)```", text, re.S).group(1)
    exports = re.findall(r"^export (CC_[A-Z_]+)=", block, re.M)
    assert exports == ["CC_PITH_PROVIDER_ROOTS", "CC_PITH_PROVIDER_MEMBERS",
                       "CC_PITH_PROVIDER_DEPTH", "CC_PITH_PROVIDER_MAX_INSTRUCTION_CHARS",
                       "CC_PITH_PROVIDER_MAX_QUEST_CHARS"]
    assert "may shorten each member" not in text


# --------------------------------------------- staged .bashrc script (temp file only)

_SCRIPT = os.path.join(_ROOT, "handoffs", "z12-pith-clip-813", "returns",
                       "bashrc-drop-node-chars.sh")
_BASHRC = ("# fake\nexport A_KEEP=1\nexport CC_PITH_PROVIDER_ROOTS=8\n"
           "export CC_PITH_PROVIDER_NODE_CHARS=700\nexport CC_PITH_PROVIDER_MEMBERS=6\n"
           "export CC_PITH_PROVIDER_DEPTH=2\nexport CC_PITH_PROVIDER_MAX_INSTRUCTION_CHARS=8000\n"
           "export CC_PITH_PROVIDER_MAX_QUEST_CHARS=8000\nexport TOKEN_LIKE=SECRETVALUE123\n")


def _sh(tmp_path, *args):
    env = dict(os.environ, BASHRC=str(tmp_path / "bashrc"))
    return subprocess.run(["bash", _SCRIPT, *args], env=env, capture_output=True, text=True)


def test_bashrc_script_apply_verify_reverse_roundtrip_is_byte_exact_and_prints_no_values(tmp_path):
    rc = tmp_path / "bashrc"
    rc.write_text(_BASHRC)
    applied = _sh(tmp_path, "apply")
    assert applied.returncode == 0, applied.stderr
    after = rc.read_text()
    assert "CC_PITH_PROVIDER_NODE_CHARS" not in after
    assert after == _BASHRC.replace("export CC_PITH_PROVIDER_NODE_CHARS=700\n", "")
    assert list(tmp_path.glob("bashrc.bak-813-*")), "timestamped backup missing"
    verify = _sh(tmp_path, "verify")
    assert verify.returncode == 0, verify.stderr
    for out in (applied.stdout + applied.stderr, verify.stdout + verify.stderr):
        assert "SECRETVALUE123" not in out and "=8000" not in out and "=700" not in out
    reversed_ = _sh(tmp_path, "reverse")
    assert reversed_.returncode == 0, reversed_.stderr
    assert rc.read_text() == _BASHRC


@pytest.mark.parametrize("content", [
    "export A=1\n",
    "export CC_PITH_PROVIDER_NODE_CHARS=1\nexport CC_PITH_PROVIDER_NODE_CHARS=2\n",
])
def test_bashrc_script_refuses_zero_or_ambiguous_matches_and_changes_nothing(tmp_path, content):
    rc = tmp_path / "bashrc"
    rc.write_text(content)
    result = _sh(tmp_path, "apply")
    assert result.returncode != 0
    assert rc.read_text() == content
    assert not list(tmp_path.glob("bashrc.bak-813-*"))


def test_bashrc_script_reverse_does_not_clobber_later_batch_edits(tmp_path):
    rc = tmp_path / "bashrc"
    rc.write_text(_BASHRC)
    assert _sh(tmp_path, "apply").returncode == 0
    with open(rc, "a") as f:                       # another S4 batch change lands after apply
        f.write("export SOME_LATER_BATCH_CHANGE=1\n")
    result = _sh(tmp_path, "reverse")
    assert result.returncode == 0, result.stderr
    text = rc.read_text()
    assert "export SOME_LATER_BATCH_CHANGE=1\n" in text
    assert text.count("export CC_PITH_PROVIDER_NODE_CHARS=700\n") == 1
    assert "SECRETVALUE123" not in result.stdout + result.stderr
