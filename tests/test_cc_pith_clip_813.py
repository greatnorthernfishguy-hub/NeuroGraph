# tests/test_cc_pith_clip_813.py
#
# ---- Changelog ----
# [2026-09-30] Z12 worker (Claude Sonnet 5.5, Claude Code) — #813 TURN 6 (dispatch #11238): le-025 C-1..C-5
# What: C-2 the measured provider node limit must never drop or reference a node that fit whole before
#   (tight-budget cases + a sweep of (core, budget, node) against BASE e4ebf982 AND the turn-5 parent
#   b2f3d18: old-whole implies new-whole, and no cut ever); C-5 the identity-pin guard FAILS CLOSED
#   (raises/missing -> PINNED + a WARNING, id and exception type only) on Stage 3 and the un-Pithed
#   renderer; C-1 the pin probe is compared with BASE's closure (identical except the deliberate C-5
#   change); C-4 one shared alert-coherence constant drives both the renderer and the measured limit;
#   C-3 the INFO line says "above the reference limit L".
# Why: Chief ruling docs e19962de on le-025 (PASS-WITH-NOTES). C-2 and C-5 tests were written FIRST and
#   shown failing on the turn-5 head.
# How: fake in-memory graph; older modules loaded from `git show <commit>:cc_ng_organism.py` (a missing
#   commit FAILS the test, it never skips).
# [2026-09-30] Z12 worker (Claude Sonnet 5.5, Claude Code) — #813 TURN 5 (dispatch #11114): le-022 N1-N3 + LOWs
# What: N1 the un-Pithed (gate-off / Pith-failure) renderer exempts identity-protected items from
#   the budget exactly as Stage 3 does (pins outside the budget, rendered whole, never dropped, never
#   swapped for a reference, and never lost to the strict-prefix stop); N2 the whole-node reference
#   line tells the truth about what follows (0 / partial / full trees, pre-PASS-2 coverage); N3 the
#   contract documents C2/C3/the DROP_LOG knobs; LOW the shell allowance is MEASURED, not a constant.
# Why: Chief ruling docs 3604cfb1 on le-022 (PASS-WITH-NOTES). N1 test was written FIRST and shown failing.
# How: fake in-memory graph; FakeGraph._is_identity_protected = constitutional or *_authored provenance.
# [2026-09-30] Z12 worker (Claude Sonnet 5.5, Claude Code) — #813 TURN 4 (dispatch #11061): checker-022 C1-C3
# What: C1 pin the pith_stage3 docstring to what the body does (an over-budget FIRST unpinned line is
#   skipped, never kept); C2 a monitor item whose whole-content re-resolve RAISES is dropped
#   (whole-or-absent, never the shared 240-char snippet) with a WARNING naming node id + exception type
#   and no text, and its pattern-stream twin is NOT deduped away with it; C3 one INFO line when the #819
#   reference form leaves concept trees out (count only).
# Why: Chief ruling docs 2a3b5fbf on checker-022 PASS-WITH-NOTES; D8 confirmed (guard stays removed).
# How: fake in-memory graph; a resolver that raises; caplog.
# [2026-09-30] Z12 worker (Claude Sonnet 5.5, Claude Code) — #813 TURN 2 (2c) / #819: over-budget node
# What: a node whose WHOLE text cannot fit the usable envelope surfaces through its TREES (whole,
#   each small) plus a ONE-LINE whole-node reference (id, size, date, tree count) and one INFO line,
#   on the provider path, the Pith-ON L1 path and the un-Pithed path. NO split at ingest; the node
#   is never modified and still participates fully in activation/learning; only its RENDERING changes.
# Why: Exec P417 (LAW 7: raw means complete; Josh P360: a long turn stays ONE node/one forest);
#   brief TURN 2 item 4; le-017 F8 (a never-fit assembly was permanently absent and unidentified).
# How: fake in-memory graph with forest -> tree synapses (the real link, _cc_bind_conversational_topology).
# [2026-09-30] Z12 worker (Claude Sonnet 5.5, Claude Code) — #813 TURN 2 (2b) / #818: every drop is loud
# What: member_limit / depth_limit / overlap / roots drops each log ONE INFO line per call with
#   count, total chars and reason; ids named first-time-seen only (flood-safe).
# Why: brief TURN 2 item 3 (Exec P416): silent member/overlap drops.
# How: fake in-memory graphs; counts assert the candidates the walk REACHED and declined.
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
    # #817 is DEFERRED (turn 3): pith_compress_history has two live Python callers (host + laptop
    # daemon handlers), so it remains the ONLY caller until it is removed TOGETHER with both.
    assert keyframe_callers <= {"pith_compress_history"}, keyframe_callers
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


# ====================================================================== TURN 2 (2b): #818
def _star(n_leaves, leaf_len=40, prefix="L"):
    g = FakeGraph()
    _core(g)
    g.node("root", "the root situation")
    for i in range(n_leaves):
        g.node(f"{prefix}{i}", f"{prefix}{i} " + ("x" * leaf_len))
        g.synapse(f"s{prefix}{i}", "root", f"{prefix}{i}", 1.0 - i * 0.01)
    return g


def _drop_lines(caplog, reason):
    return [r.getMessage() for r in caplog.records
            if r.levelno == logging.INFO and "dropped" in r.getMessage() and reason in r.getMessage()]


def test_818_member_limit_drops_are_counted_sized_and_named_once(caplog):
    g = _star(5)                                             # root + 5 leaves, room for 3 members
    surfaced = [{"node_id": "root", "score": 1.0}]
    with caplog.at_level(logging.INFO, logger=pith.logger.name):
        (line,) = pith.pith_connected_activation_basins(g, surfaced, max_members=3)
    assert len(line.member_node_ids) == 3
    dropped_ids = [f"L{i}" for i in range(5) if f"L{i}" not in line.member_node_ids]
    (message,) = _drop_lines(caplog, "member_limit")
    assert f"dropped {len(dropped_ids)} " in message
    assert f"({sum(len(g.nodes[i].metadata['_forest_content']) for i in dropped_ids)} chars)" in message
    assert all(i in message for i in dropped_ids)
    assert "CC_PITH_PROVIDER_MEMBERS" in message
    # second identical call: still counted, ids not repeated (flood-safe)
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=pith.logger.name):
        pith.pith_connected_activation_basins(g, surfaced, max_members=3)
    (again,) = _drop_lines(caplog, "member_limit")
    assert f"dropped {len(dropped_ids)} " in again
    assert not any(i in again for i in dropped_ids) and "already reported" in again


def test_818_depth_limit_drops_the_neighbours_the_walk_declined(caplog):
    g = FakeGraph()
    _core(g)
    for nid in ("root", "a", "b"):
        g.node(nid, f"node {nid} " + "y" * 30)
    g.synapse("s1", "root", "a", 0.9)
    g.synapse("s2", "a", "b", 0.9)                           # b is 2 hops away; depth limit 1
    with caplog.at_level(logging.INFO, logger=pith.logger.name):
        (line,) = pith.pith_connected_activation_basins(
            g, [{"node_id": "root", "score": 1.0}], max_depth=1, max_members=6)
    assert line.member_node_ids == ["root", "a"]
    (message,) = _drop_lines(caplog, "depth_limit")
    assert "dropped 1 " in message and "b" in message and "CC_PITH_PROVIDER_DEPTH" in message


def test_818_overlapping_basins_are_dropped_loudly(caplog):
    g = _star(3)
    g.node("root2", "a second root that fires")
    for i in range(3):
        g.synapse(f"t{i}", "root2", f"L{i}", 0.9)            # same three leaves -> heavy overlap
    surfaced = [{"node_id": "root", "score": 2.0}, {"node_id": "root2", "score": 1.0}]
    with caplog.at_level(logging.INFO, logger=pith.logger.name):
        lines = pith.pith_connected_activation_basins(g, surfaced, max_members=6)
    assert [l.node_id for l in lines] == ["root"]
    (message,) = _drop_lines(caplog, "overlap")
    assert "dropped 1 basins" in message and "root2" in message


def test_818_roots_beyond_k_are_dropped_loudly(monkeypatch, caplog):
    monkeypatch.setattr(pith, "cc_gsg_rescore", lambda surfaced, *_a, **_k: surfaced)
    g = FakeGraph()
    for i in range(3):
        g.node(f"n{i}", f"root {i} " + "z" * 50)
    ng = SimpleNamespace(graph=g)
    ng._harvest_associations = lambda *a, **k: [{"node_id": f"n{i}", "strength": 3.0 - i} for i in range(3)]
    with caplog.at_level(logging.INFO, logger=pith.logger.name):
        out = pith.cc_pattern_completion_recall(ng, "q", 1)
    assert len(out) == 1
    (message,) = _drop_lines(caplog, "roots")
    assert "dropped 2 " in message and "n1" in message and "n2" in message


def test_818_nothing_dropped_logs_nothing(caplog):
    g = _star(2)
    with caplog.at_level(logging.INFO, logger=pith.logger.name):
        pith.pith_connected_activation_basins(g, [{"node_id": "root", "score": 1.0}], max_members=6)
    assert [r for r in caplog.records if "dropped" in r.getMessage()] == []


def test_818_a_one_member_budget_still_reports_and_does_not_crash(caplog):
    g = _star(2)
    with caplog.at_level(logging.INFO, logger=pith.logger.name):
        (line,) = pith.pith_connected_activation_basins(
            g, [{"node_id": "root", "score": 1.0}], max_members=1)
    assert line.member_node_ids == ["root"]
    (message,) = _drop_lines(caplog, "member_limit")
    assert "dropped 2 " in message


# ====================================================================== TURN 2 (2c): #819
def _giant_world(giant_len=30000, trees=3, with_meta_anchor=True):
    g = FakeGraph()
    _core(g)
    text = "GIANT-START " + "/home/josh/a/b/c.py " * 5 + ("filler words " * (giant_len // 13)) + " GIANT-END"
    meta = {"path": "/exact/metadata/anchor.txt"} if with_meta_anchor else {}
    big = g.node("cc:conv::bigforest", text, **meta)
    big.creation_time = 1790000000.0                                   # 2026-09-21 in UTC
    tree_texts = []
    for i in range(trees):
        tt = f"concept {i}: the checkpoint cadence decision number {i}"
        tree_texts.append(tt)
        g.node(f"tree{i}", tt, _tree_concept=True, _concept=tt)
        g.synapse(f"f{i}", "cc:conv::bigforest", f"tree{i}", 0.2)
        g.synapse(f"b{i}", f"tree{i}", "cc:conv::bigforest", 0.15)
    return g, text, tree_texts


def _ref_records(caplog):
    return [r.getMessage() for r in caplog.records
            if r.levelno == logging.INFO and "above the reference limit" in r.getMessage()]


def test_819_provider_over_budget_node_surfaces_through_its_trees_plus_one_reference_line(caplog):
    g, text, tree_texts = _giant_world()
    with caplog.at_level(logging.INFO, logger=pith.logger.name):
        result = _run(pith, g, [{"node_id": "cc:conv::bigforest", "score": 1.0}], "go on",
                      budget_chars=4000)
    ctx = result["context"]
    assert result["state"] == "ok" and result["assemblies"] == 1 and len(ctx) <= 4000
    assert "GIANT-START" not in ctx and "GIANT-END" not in ctx        # the whole is NOT shown ...
    ref_lines = [l for l in ctx.splitlines() if "cc:conv::bigforest" in l and "long node" in l]
    assert len(ref_lines) == 1                                        # ... one line points to it
    ref = ref_lines[0]
    assert "cc:conv::bigforest" in ref and "3 concept trees" in ref and "2026-" in ref
    assert re.search(r"≈\d+k", ref) or re.search(r"≈\d{3,}", ref)
    for tt in tree_texts:                                             # its concepts follow, WHOLE
        assert tt in ctx
    (message,) = _ref_records(caplog)
    assert "1 node above the reference limit" in message and "cc:conv::bigforest" in message


def test_819_the_node_itself_is_never_modified_or_split():
    g, text, _ = _giant_world()
    before = dict(g.nodes["cc:conv::bigforest"].metadata)
    _run(pith, g, [{"node_id": "cc:conv::bigforest", "score": 1.0}], "go on", budget_chars=4000)
    assert g.nodes["cc:conv::bigforest"].metadata == before          # LAW 7: raw stays complete
    assert len(g.nodes) == 1 + 1 + 3                                  # core + giant + trees: no new nodes


def test_819_text_derived_anchors_of_the_unshown_whole_are_not_extracted_but_metadata_ones_are():
    g, _text, _ = _giant_world()
    result = _run(pith, g, [{"node_id": "cc:conv::bigforest", "score": 1.0}], "go on",
                  budget_chars=4000)
    assert "/exact/metadata/anchor.txt" in result["anchors"]
    assert "/home/josh/a/b/c.py" not in result["anchors"]


def test_819_no_trees_still_gives_the_reference_and_says_zero():
    g, _text, _ = _giant_world(trees=0)
    result = _run(pith, g, [{"node_id": "cc:conv::bigforest", "score": 1.0}], "go on",
                  budget_chars=4000)
    assert result["state"] == "ok" and "0 concept trees" in result["context"]
    assert "GIANT-START" not in result["context"]


def test_819_a_node_that_fits_is_rendered_whole_not_referenced():
    g, _text, _ = _giant_world(giant_len=2500)
    result = _run(pith, g, [{"node_id": "cc:conv::bigforest", "score": 1.0}], "go on",
                  budget_chars=8000)
    assert "GIANT-START" in result["context"] and "GIANT-END" in result["context"]
    assert "long node" not in result["context"]


def test_819_reference_is_one_line_and_undated_when_there_is_no_timestamp():
    g, _text, _ = _giant_world()
    del g.nodes["cc:conv::bigforest"].creation_time
    node = g.nodes["cc:conv::bigforest"]
    ref = pith._pith_whole_node_reference(g, "cc:conv::bigforest", node, "x" * 300000)
    assert "\n" not in ref and "undated" in ref and "≈300k" in ref and "3 concept trees" in ref


@pytest.mark.parametrize("pith_on", [True, False], ids=["pith_on", "gate_off"])
def test_819_l1_and_unpithed_paths_use_the_same_reference_form(monkeypatch, caplog, pith_on):
    g, text, tree_texts = _giant_world()
    pat = [{"node_id": "cc:conv::bigforest", "score": 50.0, "content": text, "prefetch_origin": False}]
    with caplog.at_level(logging.INFO, logger=pith.logger.name):
        out, _ = _recall(pith, fake_ng(g, []), pat, pith_on, monkeypatch)
    assert "GIANT-START" not in out and "GIANT-END" not in out
    assert "long node" in out and "cc:conv::bigforest" in out
    assert all(tt in out for tt in tree_texts)
    assert _ref_records(caplog)


# ====================================================================== TURN 4: C1 / C2 / C3
def test_c1_stage3_docstring_describes_the_skip_not_the_removed_guard():
    doc = pith.pith_stage3.__doc__
    assert "is\n       still kept if nothing has been added yet" not in doc
    assert "still kept if nothing has been added" not in doc
    assert "never emit an empty L1" not in doc
    assert "skipped" in doc and "whole" in doc.lower() and "INFO" in doc


def test_c1_a_first_unpinned_line_longer_than_the_budget_is_absent_with_the_info_line(caplog):
    giant = pith.CacheLine.from_surfaced("FIRST-GIANT", "G" * 900, score=10.0, stream="pattern")
    small = pith.CacheLine.from_surfaced("small", "s" * 40, score=1.0, stream="pattern")
    pin = pith.CacheLine.from_surfaced("pin", "P" * 2000, score=0.0, pinned=True, stream="pattern")
    with caplog.at_level(logging.INFO, logger=pith.logger.name):
        out = pith.pith_stage3([pin, giant, small], budget_chars=500)
    assert [l.node_id for l in out] == ["pin", "small"]              # rank-1 giant absent, pin untouched
    (record,) = _drop_records(caplog, "whole items")
    assert "FIRST-GIANT" in record.getMessage() and "900" in record.getMessage()


class _Boom(Exception):
    pass


def _raising_resolver(bad_ids, secret="SECRET-NODE-TEXT-DO-NOT-LOG"):
    import surface_resolver
    real = surface_resolver.resolve_surface_content

    def fake(node, entry, *a, **k):
        if getattr(node, "node_id", None) in bad_ids:
            raise _Boom(secret)
        return real(node, entry, *a, **k)
    return fake


def test_c2_a_monitor_item_whose_re_resolve_raises_is_dropped_and_warned_without_text(monkeypatch, caplog):
    import surface_resolver
    pith._PITH_DROP_SEEN.clear()
    g = FakeGraph()
    g.node("ok", "fine " * 100)
    g.node("bad", "BAD-NODE-BODY " * 40)
    monkeypatch.setattr(surface_resolver, "resolve_surface_content", _raising_resolver({"bad"}))
    ng = fake_ng(g, [])
    items = [{"node_id": "ok", "content": "cut…", "score": 1.0},
             {"node_id": "bad", "content": "the shared 240-char snippet…", "score": 0.9}]
    with caplog.at_level(logging.WARNING, logger=pith.logger.name):
        out = pith._cc_monitor_items_whole(ng, items)
    assert [o["node_id"] for o in out] == ["ok"]                     # whole-or-ABSENT: no cut item kept
    assert out[0]["content"] == ("fine " * 100).strip()
    (record,) = [r for r in caplog.records if r.levelno >= logging.WARNING]
    msg = record.getMessage()
    assert "bad" in msg and "_Boom" in msg and "1 item" in msg
    assert "SECRET-NODE-TEXT-DO-NOT-LOG" not in msg and "BAD-NODE-BODY" not in msg
    assert "shared 240-char snippet" not in msg


def test_c2_repeated_failures_still_warn_but_name_the_id_once(monkeypatch, caplog):
    import surface_resolver
    pith._PITH_DROP_SEEN.clear()
    g = FakeGraph()
    g.node("bad", "x" * 300)
    monkeypatch.setattr(surface_resolver, "resolve_surface_content", _raising_resolver({"bad"}))
    with caplog.at_level(logging.WARNING, logger=pith.logger.name):
        pith._cc_monitor_items_whole(fake_ng(g, []), [{"node_id": "bad", "content": "c…", "score": 1.0}])
        pith._cc_monitor_items_whole(fake_ng(g, []), [{"node_id": "bad", "content": "c…", "score": 1.0}])
    first, second = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert "bad" in first and "bad" not in second and "already reported" in second


@pytest.mark.parametrize("pith_on", [True, False], ids=["pith_on", "gate_off"])
def test_c2_a_dropped_monitor_item_does_not_take_its_pattern_twin_with_it(monkeypatch, pith_on):
    import surface_resolver
    g = FakeGraph()
    _core(g)
    full = "TWIN-WHOLE-TEXT " + ("t" * 500)
    g.node("twin", full)
    monkeypatch.setattr(surface_resolver, "resolve_surface_content", _raising_resolver({"twin"}))
    monitor_items = [{"node_id": "twin", "content": full[:239] + "…", "score": 1.5}]
    pat = [{"node_id": "twin", "score": 90.0, "content": full, "prefetch_origin": False}]
    out, _ = _recall(pith, fake_ng(g, monitor_items), pat, pith_on, monkeypatch)
    assert out.count(full) == 1                                      # present once, via the pattern stream
    assert "…" not in out


def _tree_world(n_trees, tree_len=200):
    g = FakeGraph()
    big = g.node("cc:conv::big", "B" * 30000)
    big.creation_time = 1790000000.0
    for i in range(n_trees):
        txt = f"tree{i} " + ("c" * tree_len)
        g.node(f"t{i}", txt, _tree_concept=True, _concept=txt)
        g.synapse(f"f{i}", "cc:conv::big", f"t{i}", 1.0 - i * 0.01)
    return g


def _tree_lines(caplog):
    return [r.getMessage() for r in caplog.records
            if r.levelno == logging.INFO and "concept trees" in r.getMessage() and "left out" in r.getMessage()]


def test_c3_trees_left_out_by_the_budget_get_one_info_line_with_counts_only(caplog):
    g = _tree_world(5)
    ref_len = len(pith._pith_whole_node_reference(g, "cc:conv::big", g.nodes["cc:conv::big"], "B" * 30000))
    budget = ref_len + 2 * (len("- concept: ") + 206 + 1) + 5          # reference + exactly two trees
    with caplog.at_level(logging.INFO, logger=pith.logger.name):
        text = pith._pith_reference_text(g, "cc:conv::big", "B" * 30000, budget)
    assert text.count("- concept: ") == 2
    (message,) = _tree_lines(caplog)
    assert "cc:conv::big" in message and "2 of 5" in message and "3 left out" in message
    assert "tree0" not in message and "ccccc" not in message         # ids/counts only, no tree text


def test_c3_trees_left_out_by_the_member_cap_are_counted_too(monkeypatch, caplog):
    monkeypatch.setattr(pith, "_CC_PITH_PROVIDER_MEMBERS", 3)
    g = _tree_world(5, tree_len=20)
    with caplog.at_level(logging.INFO, logger=pith.logger.name):
        text = pith._pith_reference_text(g, "cc:conv::big", "B" * 30000, 40000)
    assert text.count("- concept: ") == 3
    (message,) = _tree_lines(caplog)
    assert "3 of 5" in message and "2 left out" in message


def test_c3_no_line_when_every_tree_is_included(caplog):
    g = _tree_world(2, tree_len=20)
    with caplog.at_level(logging.INFO, logger=pith.logger.name):
        text = pith._pith_reference_text(g, "cc:conv::big", "B" * 30000, 40000)
    assert text.count("- concept: ") == 2 and _tree_lines(caplog) == []


# ====================================================================== TURN 5: N1 (pins in the un-Pithed renderer)
def _n1_world(third_len=1200):
    """le-022's probe: a 2500 @9.0, b 2500 @8.0, an identity-protected 1200 @0.1; budget 4000."""
    g = FakeGraph()
    _core(g)
    a, b, ident = "A-ITEM " + "a" * 2493, "B-ITEM " + "b" * 2493, "IDENT-ITEM " + "i" * (third_len - 11)
    g.node("a", a)
    g.node("b", b)
    g.node("ident", ident, provenance="cc_authored")                 # -> _is_identity_protected
    pat = [{"node_id": "a", "score": 9.0, "content": a, "prefetch_origin": False},
           {"node_id": "b", "score": 8.0, "content": b, "prefetch_origin": False},
           {"node_id": "ident", "score": 0.1, "content": ident, "prefetch_origin": False}]
    return g, a, b, ident, pat


def test_n1_identity_protected_item_is_exempt_from_the_gate_off_budget(monkeypatch, caplog):
    g, a, b, ident, pat = _n1_world()
    assert g._is_identity_protected("ident") and not g._is_identity_protected("a")
    with caplog.at_level(logging.INFO, logger=pith.logger.name):
        out, _ = _recall(pith, fake_ng(g, []), pat, False, monkeypatch)
    assert ident in out                                              # pin: whole, outside the budget
    assert a in out                                                  # top-ranked unpinned item kept
    assert b not in out                                              # the ordinary drop still happens ...
    (record,) = _drop_records(caplog, "whole items")
    assert "dropping 1 whole items" in record.getMessage()          # ... and counts ONLY b, never the pin
    assert "kept 1 (2500 chars)" in record.getMessage()             # the pin is outside the kept budget too


def test_n1_a_pin_larger_than_the_whole_budget_is_still_rendered_whole_never_referenced(monkeypatch):
    g, a, b, _ident, pat = _n1_world()
    huge = "HUGE-PIN " + "h" * 6000                                # > the 4000 budget
    g.node("hugepin", huge, provenance="cc_authored")
    pat = pat + [{"node_id": "hugepin", "score": 0.05, "content": huge, "prefetch_origin": False}]
    out, _ = _recall(pith, fake_ng(g, []), pat, False, monkeypatch)
    assert huge in out                                               # never dropped, never a reference
    assert "long node" not in out


def test_n1_the_pin_exemption_also_holds_on_the_pith_failure_fallback(monkeypatch):
    g, a, b, ident, pat = _n1_world()
    monkeypatch.setattr(pith, "pith_stage1", lambda *a_, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    out, _ = _recall(pith, fake_ng(g, []), pat, True, monkeypatch)
    assert ident in out and a in out and b not in out


def test_n1_pith_on_and_gate_off_agree_on_which_items_survive(monkeypatch):
    g, a, b, ident, pat = _n1_world()
    on, _ = _recall(pith, fake_ng(g, []), pat, True, monkeypatch)
    off, _ = _recall(pith, fake_ng(g, []), pat, False, monkeypatch)
    for text in (a, b, ident):
        assert (text in on) == (text in off), text[:8]


# ====================================================================== TURN 5: N2 (truthful reference) / LOW (measured limit)
def _ref_world(n_trees):
    g = FakeGraph()
    big = g.node("cc:conv::big", "B" * 30000)
    big.creation_time = 1790000000.0
    for i in range(n_trees):
        txt = f"tree{i} " + ("c" * 40)
        g.node(f"t{i}", txt, _tree_concept=True, _concept=txt)
        g.synapse(f"f{i}", "cc:conv::big", f"t{i}", 1.0 - i * 0.01)
    return g


def _ref(g, shown=None):
    return pith._pith_whole_node_reference(g, "cc:conv::big", g.nodes["cc:conv::big"], "B" * 30000, shown=shown)


def test_n2_a_node_with_no_trees_does_not_promise_that_concepts_follow():
    line = _ref(_ref_world(0))
    assert "0 concept trees" in line and "follow" not in line and "concepts" not in line.split("(")[1].split(")")[1]
    assert line.endswith("too large to render whole here.")
    assert "\n" not in line


def test_n2_partial_and_full_coverage_are_stated_exactly():
    g = _ref_world(5)
    assert "3 of 5 concept trees follow" in _ref(g, shown=3)
    assert "5 concept trees follow" in _ref(g, shown=5) and "of 5" not in _ref(g, shown=5)
    assert "None of its concept trees fit here" in _ref(g, shown=0)
    assert "1 concept tree follows" in _ref(_ref_world(1), shown=1)              # grammar
    for shown in (3, 5):
        assert "may cover only part of it" in _ref(g, shown=shown)               # the honest hedge


def test_n2_when_the_caller_cannot_know_how_many_fit_it_says_where_they_fit():
    line = _ref(_ref_world(4))                                                  # provider path: shown=None
    assert "4 concept trees" in line and "follow where they fit" in line
    assert "so its concepts follow" not in line


def test_n2_the_l1_form_reports_exactly_the_trees_it_placed(caplog):
    g = _ref_world(5)
    ref_len = len(pith._pith_whole_node_reference(g, "cc:conv::big", g.nodes["cc:conv::big"], "B" * 30000, shown=2))
    budget = ref_len + 2 * (len("- concept: ") + 46 + 1) + 3
    text = pith._pith_reference_text(g, "cc:conv::big", "B" * 30000, budget)
    assert len(text) <= budget and text.count("- concept: ") == 2
    assert "2 of 5 concept trees follow" in text


def test_n2_the_provider_context_line_carries_no_false_promise_for_a_treeless_giant():
    g, _t, _ = _giant_world(trees=0)
    ctx = _run(pith, g, [{"node_id": "cc:conv::bigforest", "score": 1.0}], "go on", budget_chars=4000)["context"]
    assert "0 concept trees" in ctx and "concepts follow" not in ctx


def test_low_the_node_limit_is_measured_from_the_renderer_not_a_constant():
    assert not hasattr(pith, "_PITH_SHELL_ALLOWANCE")
    core = pith.render_constitutional_core(_giant_world()[0])
    limit = pith._pith_provider_node_limit(core, 4000)
    assert 1 <= limit < 4000 - len(core)
    # Turn 6 (C-2): the limit is the OPTIMISTIC bound -- the room a single ORDINARY line has (no
    # alert, no sources line, no anchors/relations).  A node of exactly `limit` chars in such a line
    # fits the budget exactly; an alert/sources/anchor-bearing assembly may need more and is then
    # caught at admit as a loud never-fit (whole-or-absent).
    line = pith.CacheLine(node_id="n", content="x" * limit, sources=[], stream="connected",
                          coherence="shared")
    fitted = pith._pith_provider_sections(core, [line], [pith._pith_render_connected_line(line)])[0]
    assert len(fitted) == 4000
    assert pith._pith_provider_node_limit(core, 10) == 1               # never 0 (0 would disable the form)
    assert pith._pith_provider_node_limit(core, 8000) - limit == 4000  # 1:1 with the budget

def test_n3_the_contract_states_the_log_knobs_with_the_defaults_and_clamps_the_code_uses():
    text = open(os.path.join(_ROOT, "docs", "PITH_HOST_CONTRACT.md")).read()
    assert f"| `CC_PITH_DROP_LOG_IDS_PER_CALL` | `{os.environ.get('CC_PITH_DROP_LOG_IDS_PER_CALL', '8')}` | `[1, 64]` |" in text
    assert f"| `CC_PITH_DROP_LOG_SEEN_MAX` | `{os.environ.get('CC_PITH_DROP_LOG_SEEN_MAX', '4096')}` | `[16, 65536]` |" in text
    assert pith._CC_PITH_DROP_LOG_IDS_PER_CALL == 8 and pith._CC_PITH_DROP_LOG_SEEN_MAX == 4096
    src = open(_ORGANISM_SRC).read()
    assert 'max(1, min(64, int(os.environ.get("CC_PITH_DROP_LOG_IDS_PER_CALL", "8"))))' in src
    assert 'max(16, min(65536, int(os.environ.get("CC_PITH_DROP_LOG_SEEN_MAX", "4096"))))' in src


def test_n3_the_contract_describes_c2_c3_n1_and_the_truthful_reference():
    body = open(os.path.join(_ROOT, "docs", "PITH_HOST_CONTRACT.md")).read().split("-->", 1)[1]
    assert "re-resolve *raises*" in body and "dropped" in body and "WARNING" in body
    assert "pith reference form" in body and "left out" in body
    assert "Identity is outside the recall budget" in body
    assert "so its concepts follow" not in body


# ====================================================================== TURN 6
import functools
import importlib.util
import tempfile

_BASE_COMMIT = "e4ebf982b1989fd9066d610b94853bc68bf70d37"
_TURN5_PARENT = "b2f3d183c8922302ecbc6e204141d229a02ddcdc"        # turn-5 head: the OLD 800/200 constants


@functools.lru_cache(maxsize=None)
def _module_at(commit):
    """cc_ng_organism as it was at `commit` (a missing commit raises -> the test FAILS, never skips)."""
    src = subprocess.check_output(["git", "-C", _ROOT, "show", f"{commit}:cc_ng_organism.py"])
    path = os.path.join(tempfile.mkdtemp(prefix="cc_ng_organism_old_"), f"cc_ng_organism_{commit[:8]}.py")
    open(path, "wb").write(src)
    name = f"cc_ng_organism_{commit[:8]}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _tight(mod, core_len, budget, node_len):
    """One provider_context render on `mod` with a core of `core_len` chars and ONE node."""
    g = FakeGraph()
    g.node("core", "K" * max(1, core_len), constitutional=True)
    text = "N" * node_len
    g.node("x", text)
    result = _run(mod, g, [{"node_id": "x", "score": 1.0}], "go on", budget_chars=budget)
    ctx = result.get("context") or ""
    return result, ctx, text


def _is_whole(ctx, text):
    return text in ctx


_TIGHT_CASES = [(800, 1500, 60), (800, 1500, 120), (800, 1200, 30), (800, 1200, 60),
                (400, 1000, 30), (400, 1000, 60), (400, 1000, 120)]


@pytest.mark.parametrize("core_len,budget,node_len", _TIGHT_CASES)
def test_c2_a_node_that_fit_whole_before_stays_whole_in_the_tight_regime(core_len, budget, node_len):
    _r_old, ctx_old, text = _tight(_module_at(_TURN5_PARENT), core_len, budget, node_len)
    assert _is_whole(ctx_old, text), "precondition: the turn-5 parent rendered this node whole"
    result, ctx, text = _tight(pith, core_len, budget, node_len)
    assert result["state"] == "ok", result["warnings"]
    assert _is_whole(ctx, text) and "long node" not in ctx           # WHOLE: not dropped, not a reference


def test_c2_sweep_old_whole_implies_new_whole_and_nothing_is_ever_cut():
    base, parent = _module_at(_BASE_COMMIT), _module_at(_TURN5_PARENT)
    checked = old_whole_points = 0
    for core_len in (0, 200, 400, 800):
        for budget in (500, 1000, 1200, 1500, 4000):
            for node_len in (30, 60, 120, 600):
                res = {}
                for label, mod in (("base", base), ("parent", parent), ("head", pith)):
                    res[label] = _tight(mod, core_len, budget, node_len)
                text = res["head"][2]
                _r, head_ctx, _t = res["head"]
                old_whole = _is_whole(res["base"][1], text) or _is_whole(res["parent"][1], text)
                if old_whole:
                    old_whole_points += 1
                    assert _is_whole(head_ctx, text), (core_len, budget, node_len, "old-whole but head is not")
                # no cut, ever: the node is whole, or absent, or represented by the reference line
                assert "⋯" not in head_ctx and " …" not in head_ctx, (core_len, budget, node_len)
                partial = ("N" * max(2, node_len // 2)) in head_ctx and not _is_whole(head_ctx, text)
                assert not partial, (core_len, budget, node_len, "a cut node")
                assert len(head_ctx) <= budget, (core_len, budget, node_len)
                checked += 1
    assert checked == 80 and old_whole_points >= 20                     # the grid really exercises the invariant


def test_c2_the_limit_is_the_optimistic_bound_a_lone_ordinary_line_is_never_rejected():
    core = pith.render_constitutional_core(_giant_world()[0])
    for budget in (500, 1000, 1500, 4000):
        limit = pith._pith_provider_node_limit(core, budget)
        # the room a single ORDINARY connected line (no alert, no correction, no sources line) has
        line = pith.CacheLine(node_id="n", content="x" * max(0, limit), stream="connected",
                              coherence="shared", sources=[])
        room = pith._pith_provider_sections(core, [line], [pith._pith_render_connected_line(line)])[0]
        assert limit == 1 or len(room) <= budget, (budget, limit, len(room))


# ------------------------------------------------------------------ C-5: the pin guard fails CLOSED
def _ng_with_guard(fn):
    g = FakeGraph()
    _core(g)
    g._is_identity_protected = fn
    return SimpleNamespace(graph=g)


def test_c5_a_raising_guard_is_treated_as_pinned_with_a_warning_naming_id_and_type_only(caplog):
    pith._PITH_DROP_SEEN.clear()

    def boom(node_id):
        raise RuntimeError("SECRET-GUARD-TEXT-DO-NOT-LOG " + node_id)
    probe = pith._cc_pin_probe(_ng_with_guard(boom))
    with caplog.at_level(logging.WARNING, logger=pith.logger.name):
        assert probe("ident-1") is True                                  # identity fails toward KEEPING content
        assert probe("ident-1") is True                                  # ... every time,
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1, warnings                                  # ... but names the id ONCE (no flood)
    assert "ident-1" in warnings[0] and "RuntimeError" in warnings[0] and "PINNED" in warnings[0]
    assert "SECRET-GUARD-TEXT-DO-NOT-LOG" not in warnings[0]


def test_c5_a_missing_guard_is_treated_as_pinned_too(caplog):
    pith._PITH_DROP_SEEN.clear()
    with caplog.at_level(logging.WARNING, logger=pith.logger.name):
        assert pith._cc_pin_probe(SimpleNamespace())("n1") is True                      # no .graph at all
        assert pith._cc_pin_probe(SimpleNamespace(graph=SimpleNamespace()))("n2") is True  # graph without the guard
    assert len([r for r in caplog.records if r.levelno >= logging.WARNING]) == 2


def test_c5_a_working_guard_is_unchanged():
    probe = pith._cc_pin_probe(_ng_with_guard(lambda nid: nid == "yes"))
    assert probe("yes") is True and probe("no") is False


@pytest.mark.parametrize("pith_on", [True, False], ids=["pith_on", "gate_off"])
def test_c5_with_a_raising_guard_identity_content_is_kept_on_both_paths(monkeypatch, caplog, pith_on):
    pith._PITH_DROP_SEEN.clear()
    g, a, b, ident, pat = _n1_world()

    def boom(node_id):
        raise RuntimeError("guard down")
    g._is_identity_protected = boom
    with caplog.at_level(logging.WARNING, logger=pith.logger.name):
        out, _ = _recall(pith, fake_ng(g, []), pat, pith_on, monkeypatch)
    assert ident in out and a in out and b in out                        # nothing budget-dropped: all treated as pinned
    assert any("PINNED" in r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)


# ------------------------------------------------------------------ C-1: the probe vs BASE's closure
def _closure_pinned(src, outer):
    tree = ast.parse(src)
    outer_fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == outer)
    return next(n for n in ast.walk(outer_fn) if isinstance(n, ast.FunctionDef) and n.name == "_pinned")


def test_c1_the_pin_probe_is_the_base_closure_except_the_deliberate_c5_change():
    base_src = subprocess.check_output(["git", "-C", _ROOT, "show", f"{_BASE_COMMIT}:cc_ng_organism.py"]).decode()
    base_fn = _closure_pinned(base_src, "cc_assemble_recall")
    head_fn = _closure_pinned(open(_ORGANISM_SRC).read(), "_cc_pin_probe")
    assert base_fn.args.args[0].arg == head_fn.args.args[0].arg == "node_id"
    base_try, head_try = base_fn.body[0], head_fn.body[0]
    assert isinstance(base_try, ast.Try) and isinstance(head_try, ast.Try)
    # the guarded call is IDENTICAL to base
    assert ast.dump(base_try.body[0]) == ast.dump(head_try.body[0])
    # the handler is the ONE deliberate difference: base failed soft to False (DEBUG), head fails closed
    base_ret = [n for n in ast.walk(base_try.handlers[0]) if isinstance(n, ast.Return)][-1]
    head_ret = [n for n in ast.walk(head_try.handlers[0]) if isinstance(n, ast.Return)][-1]
    assert isinstance(base_ret.value, ast.Constant) and base_ret.value.value is False
    assert (isinstance(head_ret.value, ast.Constant) and head_ret.value.value is True) or \
        "True" in ast.dump(head_ret.value) or "_pin_guard_failed" in ast.dump(head_ret.value)
    assert ast.dump(base_try.handlers[0]) != ast.dump(head_try.handlers[0])


def test_c1_executed_on_the_le025_cases_identical_except_when_the_guard_raises():
    base_src = subprocess.check_output(["git", "-C", _ROOT, "show", f"{_BASE_COMMIT}:cc_ng_organism.py"]).decode()
    seg = ast.get_source_segment(base_src, _closure_pinned(base_src, "cc_assemble_recall"))
    import textwrap

    def base_probe(ng):
        scope = {"ng": ng, "logger": logging.getLogger("base-probe")}
        exec(textwrap.dedent(seg), scope)
        return scope["_pinned"]
    def raiser(exc):
        return lambda nid: (_ for _ in ()).throw(exc)
    cases = [("true", lambda n: True, False), ("false", lambda n: False, False), ("truthy str", lambda n: "yes", False),
             ("none", lambda n: None, False), ("empty list", lambda n: [], False), ("zero", lambda n: 0, False),
             ("raises Runtime", raiser(RuntimeError("x")), True), ("raises Key", raiser(KeyError("k")), True)]
    pith._PITH_DROP_SEEN.clear()
    for label, fn, raises in cases:
        ng = _ng_with_guard(fn)
        b, h = base_probe(ng)("n"), pith._cc_pin_probe(ng)("n")
        if raises:
            assert (b, h) == (False, True), label                        # the deliberate C-5 change
        else:
            assert b == h, label                                          # identical to base
    for ng in (SimpleNamespace(), SimpleNamespace(graph=None)):          # a missing guard raises too
        assert (base_probe(ng)("n"), pith._cc_pin_probe(ng)("n")) == (False, True)


# ------------------------------------------------------------------ C-4: ONE shared alert constant
def test_c4_one_alert_constant_drives_both_the_renderer_and_the_measured_limit(monkeypatch):
    assert pith._PITH_ALERT_COHERENCE == ("conflict", "stale", "uncertain", "unknown")
    core = "## Who I Am\n- k"
    line = pith.CacheLine(node_id="n", content="x", coherence="shared", stream="connected")
    _ctx, warnings = pith._pith_provider_sections(core, [line], [pith._pith_render_connected_line(line)])
    assert warnings == []
    before = pith._pith_provider_node_limit(core, 4000)
    monkeypatch.setattr(pith, "_PITH_ALERT_COHERENCE", pith._PITH_ALERT_COHERENCE + ("shared",))
    _ctx, warnings = pith._pith_provider_sections(core, [line], [pith._pith_render_connected_line(line)])
    assert warnings == ["shared_material"]                               # the renderer follows the constant
    assert pith._pith_provider_node_limit(core, 4000) < before           # ... and so does the measured limit
    src = open(_ORGANISM_SRC).read()
    assert src.count('("conflict", "stale", "uncertain", "unknown")') == 1   # defined once, not re-listed


# ------------------------------------------------------------------ C-3: the wording tells the truth
def test_c3_the_info_line_says_above_the_reference_limit_not_over_budget(caplog):
    g, _text, _ = _giant_world()
    with caplog.at_level(logging.INFO, logger=pith.logger.name):
        _run(pith, g, [{"node_id": "cc:conv::bigforest", "score": 1.0}], "go on", budget_chars=4000)
    lines = [r.getMessage() for r in caplog.records if "whole-node reference" in r.getMessage()]
    assert lines and all("above the reference limit" in l and "over-budget" not in l for l in lines)


def test_c3_the_limit_docstring_no_longer_claims_admit_catches_the_between_band():
    doc = pith._pith_provider_node_limit.__doc__
    assert "conservative (worst-case shell) direction" not in doc
    assert "optimistic" in doc.lower() and "never-fit" in doc.lower()


def test_turn6_the_contract_states_the_optimistic_limit_and_the_fail_closed_guard():
    body = open(os.path.join(_ROOT, "docs", "PITH_HOST_CONTRACT.md")).read().split("-->", 1)[1]
    assert "The guard fails closed" in body and "treated as pinned" in body
    assert "smallest* overhead" in body and "optimistic" in body and "never-fit" in body
    assert "above the reference limit" in body
