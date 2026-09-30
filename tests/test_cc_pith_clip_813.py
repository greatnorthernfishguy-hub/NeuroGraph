# tests/test_cc_pith_clip_813.py
#
# ---- Changelog ----
# [2026-09-30] Z12 worker (Claude Sonnet 5.5, Claude Code) — #813 TURN 2 (1b): #816 + ONE rule
# What: regression tests for the pair's HIGH (F1/C1): a >300-char pattern item and a >240-char
#   monitor item survive Stage 3 WHOLE or are dropped WHOLE with the INFO line, on Pith-ON, on
#   the gate-off path and on the Pith-failure fallback; the CC-only monitor route (shared
#   surfacing.py / surface_resolver.py untouched); the ONE budget rule (F2, C3, F8); the AST guard
#   now walks the COMPLETE caller set (C4, F6); golden vs BASE for cc_assemble_recall.
# Why: checker-019 C1-C4, le-017 F1/F2/F6/F8; Exec P410(c)/P416.
# How: fake in-memory ng/graph; recall swapped by an honest fake that cuts at 300 unless the
#   caller asks whole_content=True (mirroring the real function), so a caller that forgets to ask
#   FAILS the test. No live path.
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
    pith._PITH_DROP_SEEN.clear()          # "first-time-seen" ids are process state by design
    yield
    pith._PITH_METRICS.reset()
    pith._PITH_DROP_SEEN.clear()


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
    # turn 2 (ONE rule): mid cannot fit even an empty 300-char budget, so it is skipped WHOLE and
    # does not end the prefix; the small lower-ranked line that fits is kept.
    assert [l.node_id for l in out] == ["top", "low"]
    assert mid.content == mid_text                                   # never rewritten in place
    assert pith._PITH_METRICS.compressed_count == 0
    assert pith._PITH_METRICS.chars_saved == 0
    (record,) = _drop_records(caplog, "whole items")
    assert "dropping 1 whole items" in record.getMessage()
    assert f"({len(mid_text)} chars)" in record.getMessage()
    assert "mid" in record.getMessage() and "never-fit" in record.getMessage()


def test_stage3_logs_nothing_when_nothing_is_dropped(caplog):
    a = pith.CacheLine.from_surfaced("a", "A" * 200, score=2.0, stream="pattern")
    b = pith.CacheLine.from_surfaced("b", "B" * 200, score=1.0, stream="pattern")
    with caplog.at_level(logging.INFO, logger=pith.logger.name):
        out = pith.pith_stage3([a, b], budget_chars=500)
    assert [l.node_id for l in out] == ["a", "b"]
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


# ------------------------------------------------ structural (turn 2: complete-caller-set tests below)


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


# ------------------------------------------------- turn 2: F3a (rollback) / F3b (symlink)

def test_bashrc_script_apply_rolls_back_when_its_own_verify_fails(tmp_path):
    # le-017's reproduction: the sole export sits inside `if true; then ... fi`, so deleting
    # it makes `bash -n` fail. The old script left the file broken; it must now restore it.
    broken_on_delete = ("export A_KEEP=1\nif true; then\nexport CC_PITH_PROVIDER_NODE_CHARS=700\nfi\n"
                        "export CC_PITH_PROVIDER_ROOTS=8\n")
    rc = tmp_path / "bashrc"
    rc.write_text(broken_on_delete)
    result = _sh(tmp_path, "apply")
    assert result.returncode != 0, result.stdout + result.stderr
    assert rc.read_text() == broken_on_delete                        # byte-identical: rolled back
    assert "rolled back" in result.stderr
    assert not list(tmp_path.glob("bashrc.bak-813.*")), "pointer files must not outlive a rollback"
    assert len(list(tmp_path.glob("bashrc.bak-813-*"))) == 1        # the backup itself is kept
    assert "SECRET" not in result.stdout + result.stderr


def test_bashrc_script_follows_a_symlinked_bashrc_and_keeps_it_a_symlink(tmp_path):
    real = tmp_path / "dotfiles_bashrc"
    real.write_text(_BASHRC)
    link = tmp_path / "bashrc"
    link.symlink_to(real)
    applied = _sh(tmp_path, "apply")
    assert applied.returncode == 0, applied.stderr
    assert link.is_symlink() and link.resolve() == real.resolve()   # F3b: still a symlink
    assert "CC_PITH_PROVIDER_NODE_CHARS" not in real.read_text()    # the TARGET was edited
    assert _sh(tmp_path, "reverse").returncode == 0
    assert link.is_symlink() and real.read_text() == _BASHRC


def test_bashrc_script_apply_prints_the_s4_checklist_reminder(tmp_path):
    (tmp_path / "bashrc").write_text(_BASHRC)
    out = _sh(tmp_path, "apply")
    assert out.returncode == 0
    assert "S4 checklist" in out.stdout and "cannot check" in out.stdout


# =================================================================== TURN 2 (1b): #816 + ONE rule
import surfacing as _shared_surfacing                                   # the REAL shared module
from pith_clip_813_scenarios import FakeVectorDB, fake_ng, run_recall, build_recall_scenarios


def _honest_recall(items_by_full):
    """A recall fake that behaves like the real one: 300-char cut unless whole_content=True."""
    seen = {}

    def fake(_ng, _query, _k, *_a, **kwargs):
        seen.update(kwargs)
        out = []
        for item in items_by_full:
            item = dict(item)
            if not kwargs.get("whole_content") and len(item["content"]) > 300:
                item["content"] = item["content"][:299].rstrip() + "…"
            out.append(item)
        return out
    return fake, seen


def _recall(pith, ng, pc_items, pith_on, monkeypatch, commons=None):
    fake, seen = _honest_recall(pc_items)
    monkeypatch.setattr(pith, "cc_pattern_completion_recall", fake)
    monkeypatch.setattr(pith, "cc_novelty", lambda *_a, **_k: 0.0)
    monkeypatch.setattr(pith, "_CC_PITH_ENABLED", pith_on)
    pith._PITH_VICTIM.clear()
    try:
        return pith.cc_assemble_recall(ng, "what next", 5, {}, commons), seen
    finally:
        pith._PITH_VICTIM.clear()


def _long_world(long_len=900, count=1):
    """A graph whose monitor node and pattern node are LONG; the fake monitor hands out the
    240-char-cut text the SHARED surfacing.py produces."""
    g = FakeGraph()
    _core(g)
    mon_full = "MONITOR-" + ("m" * (long_len - 8))
    g.node("mon", mon_full)
    monitor_items = [{"node_id": "mon", "content": mon_full[:239].rstrip() + "…", "score": 1.5}]
    pc_full = []
    pat = []
    for i in range(count):
        text = f"PATTERN{i}-" + ("p" * (long_len - 10))
        g.node(f"pat{i}", text)
        pat.append({"node_id": f"pat{i}", "score": 100.0 - i, "content": text,
                    "prefetch_origin": False})
        pc_full.append(text)
    return g, mon_full, monitor_items, pat


@pytest.mark.parametrize("pith_on", [True, False], ids=["pith_on", "gate_off"])
def test_816_long_pattern_and_monitor_items_are_rendered_whole(monkeypatch, pith_on):
    g, mon_full, monitor_items, pat = _long_world(900)
    out, seen = _recall(pith, fake_ng(g, monitor_items), pat, pith_on, monkeypatch)
    assert seen.get("whole_content") is True, "the caller must ask recall for whole content"
    assert pat[0]["content"] in out                                   # >300-char pattern item whole
    assert mon_full in out                                            # >240-char monitor item whole
    assert "…" not in out and "..." not in out


@pytest.mark.parametrize("pith_on", [True, False], ids=["pith_on", "gate_off"])
def test_816_when_the_budget_binds_whole_items_are_dropped_with_an_info_line(monkeypatch, caplog,
                                                                              pith_on):
    g, mon_full, monitor_items, pat = _long_world(1500, count=3)     # 4 items x ~1500 > 4000
    with caplog.at_level(logging.INFO, logger=pith.logger.name):
        out, _seen = _recall(pith, fake_ng(g, monitor_items), pat, pith_on, monkeypatch)
    fulls = [mon_full] + [p["content"] for p in pat]
    kept = [f for f in fulls if f in out]
    dropped = [f for f in fulls if f not in out]
    assert 1 <= len(kept) < len(fulls) and dropped
    for f in dropped:                                                 # whole or ABSENT, never a piece
        assert f[:20] not in out
    assert "…" not in out
    records = _drop_records(caplog, "whole items")
    assert len(records) == 1, [r.getMessage() for r in caplog.records]
    message = records[0].getMessage()
    assert f"dropping {len(dropped)} whole items" in message
    assert f"({sum(len(f) for f in dropped)} chars)" in message


def test_816_failure_fallback_is_also_whole_and_budgeted(monkeypatch, caplog):
    g, mon_full, monitor_items, pat = _long_world(1500, count=3)
    monkeypatch.setattr(pith, "pith_stage1", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with caplog.at_level(logging.INFO, logger=pith.logger.name):
        out, _ = _recall(pith, fake_ng(g, monitor_items), pat, True, monkeypatch)
    fulls = [mon_full] + [p["content"] for p in pat]
    assert any(f in out for f in fulls) and not all(f in out for f in fulls)
    assert "…" not in out
    assert _drop_records(caplog, "whole items")


def test_816_shared_surfacing_and_resolver_are_not_edited():
    changed = subprocess.run(
        ["git", "-C", _ROOT, "diff", "--name-only", "e4ebf982b1989fd9066d610b94853bc68bf70d37"],
        capture_output=True, text=True, check=True).stdout.split()
    for shared in ("surfacing.py", "surface_resolver.py", "neurograph_rpc.py", "kiss_filter.py",
                   "tonic_thread.py"):
        assert shared not in changed, shared


def test_cc_monitor_block_is_layout_identical_to_the_shared_format_context():
    items = [{"node_id": "a", "content": "short one", "score": 1.7321},
             {"node_id": "b", "content": "", "score": 1.1, "image_ref": "/tmp/x.png"},
             {"node_id": "c", "content": "x" * 200, "score": 0.8}]              # exactly at the shared cut
    shared = _shared_surfacing.SurfacingMonitor.format_context(SimpleNamespace(), items)
    assert pith._format_cc_monitor_block(items) == shared
    assert pith._format_cc_monitor_block([]) == ""


def test_monitor_items_are_re_resolved_whole_by_node_id_and_fail_soft():
    g = FakeGraph()
    full = "z" * 700
    g.node("a", full)
    ng = fake_ng(g, [], vdb=FakeVectorDB({"v": {"content": "from the vdb " + "q" * 400}}))
    g.node("v", "")                                                   # substrate empty -> vdb fallback
    items = [{"node_id": "a", "content": full[:239] + "…", "score": 1.0},
             {"node_id": "v", "content": "cut…", "score": 0.9},
             {"node_id": "gone", "content": "kept as-is", "score": 0.5}]
    out = pith._cc_monitor_items_whole(ng, items)
    assert out[0]["content"] == full
    assert out[1]["content"] == "from the vdb " + "q" * 400
    assert out[2]["content"] == "kept as-is"                          # unknown node: never dropped
    assert [o["score"] for o in out] == [1.0, 0.9, 0.5]


def test_recall_short_items_render_exactly_as_base():
    golden = json.load(open(os.path.join(
        _ROOT, "tests", "fixtures", "pith_clip_813_golden_base.json")))["recall_scenarios"]
    now = build_recall_scenarios(pith)
    assert set(now) == set(golden) and len(now) == 6
    for name, expected in golden.items():
        assert now[name] == expected, name


# ------------------------------------------------------------------- the ONE rule

def test_one_rule_stage3_skips_a_never_fit_line_and_names_it(caplog):
    top = pith.CacheLine.from_surfaced("top", "A" * 50, score=10.0, stream="pattern")
    giant = pith.CacheLine.from_surfaced("GIANT-ID", "G" * 5000, score=5.0, stream="pattern")
    small = pith.CacheLine.from_surfaced("small", "s" * 30, score=1.0, stream="pattern")
    with caplog.at_level(logging.INFO, logger=pith.logger.name):
        out = pith.pith_stage3([top, giant, small], budget_chars=300)
    assert [l.node_id for l in out] == ["top", "small"]              # never-fit does not end the prefix
    assert giant.content == "G" * 5000
    (record,) = _drop_records(caplog, "whole items")
    assert "GIANT-ID" in record.getMessage() and "never-fit" in record.getMessage()


def test_one_rule_stage3_never_emits_over_budget_F2(caplog):
    big = pith.CacheLine.from_surfaced("big", "B" * 900, score=1.0, stream="pattern")
    with caplog.at_level(logging.INFO, logger=pith.logger.name):
        out = pith.pith_stage3([big], budget_chars=500)
    assert out == []                                                  # no silent overrun (F2)
    (record,) = _drop_records(caplog, "whole items")
    assert "big" in record.getMessage() and "900" in record.getMessage()


def test_one_rule_stage3_pinned_lines_stay_outside_the_budget():
    pin = pith.CacheLine.from_surfaced("pin", "P" * 2000, score=0.0, pinned=True, stream="pattern")
    out = pith.pith_stage3([pin], budget_chars=500)
    assert [l.node_id for l in out] == ["pin"] and out[0].content == "P" * 2000


def test_one_rule_provider_admit_names_never_fit_ids_once_flood_safe(caplog):
    pith._PITH_DROP_SEEN.clear()
    giant, small = _cl("NEVERFIT-1", 9.0, 5000), _cl("s1", 2.0, 60)
    budget = len(pith._pith_render_connected_line(small)) + 10
    with caplog.at_level(logging.INFO, logger=pith.logger.name):
        pith._pith_provider_admit([giant, small], budget)
        pith._pith_provider_admit([giant, small], budget)
    first, second = _drop_records(caplog, "whole assemblies")
    assert "NEVERFIT-1" in first.getMessage() and "never-fit" in first.getMessage()
    assert "NEVERFIT-1" not in second.getMessage()                    # first-time-seen only
    assert "already reported" in second.getMessage()                  # but the drop is still counted


# ------------------------------------------- F6 / C4: the COMPLETE caller set, not a name list

def _module_calls(tree):
    """[(enclosing function name, called name)] for Name AND Attribute calls, whole module."""
    out = []

    class V(ast.NodeVisitor):
        def __init__(self):
            self.stack = []

        def visit_FunctionDef(self, node):
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_Call(self, node):
            f = node.func
            name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
            if name:
                out.append((self.stack[-1] if self.stack else "<module>", name))
            self.generic_visit(node)
    V().visit(tree)
    return out


def test_complete_caller_set_of_the_cutters_is_empty_outside_the_keyframe_primitive():
    tree = ast.parse(open(_ORGANISM_SRC).read())
    calls = _module_calls(tree)
    keyframe_callers = {fn for fn, name in calls if name == "pith_stage2_keyframe"}
    word_cut_callers = {fn for fn, name in calls if name == "_pith_cut_at_word_boundary"}
    assert keyframe_callers <= {"pith_compress_history"}, keyframe_callers   # #817 removes it
    assert word_cut_callers <= {"pith_stage2_keyframe"}, word_cut_callers
    assert "_pith_fit_statement" not in {name for _fn, name in calls}


def test_no_resolve_surface_content_call_passes_a_literal_character_cap():
    tree = ast.parse(open(_ORGANISM_SRC).read())
    seen = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", None)
            if name in ("resolve_surface_content", "resolve_surface_item"):
                seen += 1
                for kw in node.keywords:
                    if kw.arg == "max_chars":
                        assert not isinstance(kw.value, ast.Constant), ast.dump(kw.value)
    assert seen >= 2                                                  # recall + the CC monitor route


def test_cc_assemble_recall_asks_recall_for_whole_content_by_literal_true():
    tree = ast.parse(open(_ORGANISM_SRC).read())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "cc_assemble_recall")
    hits = [c for c in ast.walk(fn) if isinstance(c, ast.Call)
            and getattr(c.func, "id", None) == "cc_pattern_completion_recall"]
    assert hits and all(any(k.arg == "whole_content" and isinstance(k.value, ast.Constant)
                            and k.value.value is True for k in c.keywords) for c in hits)


def test_unpithed_renderer_fails_open_loudly_when_its_own_budget_step_breaks(monkeypatch, caplog):
    # Josh 2026-09-26: when Pith fails there HAS to be pass-through.  The fallback must not
    # depend on the machinery that failed, and must never raise or cut: whole, unbudgeted, LOUD.
    g, mon_full, monitor_items, pat = _long_world(900)
    monkeypatch.setattr(pith, "_pith_unified_rank",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("rank broke")))
    with caplog.at_level(logging.WARNING, logger=pith.logger.name):
        out, _ = _recall(pith, fake_ng(g, monitor_items), pat, False, monkeypatch)
    assert pat[0]["content"] in out and mon_full in out and "…" not in out
    assert any("rendering every item whole and unbudgeted" in r.getMessage() for r in caplog.records)
