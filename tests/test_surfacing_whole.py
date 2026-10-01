# tests/test_surfacing_whole.py
#
# #812 turn 1 — the shared CES surfacing path renders WHOLE (LAW 4: fix at the source).
#
# ---- Changelog ----
# [2026-10-01] Claude Sonnet 5.5 (Z12 lane surfacing-whole-812, dispatch #12618) — new file
# What: Pins the #812 contract. (1) surface_resolver.resolve_surface_content / _item return a
#   node's text WHOLE by default (no 240-char cut, no ellipsis); an explicit max_chars bound
#   still works unchanged. (2) SurfacingMonitor.format_context renders whole (no 200-char cut).
#   (3) The two max_chars=300 sites render whole: the Active Recall block of
#   neurograph_rpc.handle_assemble and cc_ng_organism.cc_pattern_completion_recall.
#   (4) Everything that is NOT the cut is byte-identical to the base e4ebf982 modules.
# Why:  Exec P468 / Josh: "We fix stuff correctly, not monkey patch or work around."
#   Chief-003 addendum 1: the two 300-char cuts are the same lossy-clipping class.
# How:  Pure-function tests + in-process fakes (no daemon, no graph load, no checkpoint, no
#   data/ path, no network, no embedding model: ng_embed is stubbed). Exec P379 / #770:
#   neurograph_rpc.py hard-codes ~/NeuroGraph at sys.path[0], so a narrow worktree run can
#   silently test the PRIMARY checkout. A session-scoped fixture therefore PRINTS the
#   __file__ of every NG module under test at session start and FAILS the session if any
#   is not under this file's own repo root (re-checked at teardown to cover modules other
#   test files import later). Each test says why it FAILS on the base.
# -------------------

import importlib.util
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
BASE_SHA = "e4ebf982b1989fd9066d610b94853bc68bf70d37"

# Worktree first (same idiom as the sibling tests). neurograph_rpc re-prepends ~/NeuroGraph
# on import, so the root is re-asserted below once it is loaded (#770).
sys.path.insert(0, str(_ROOT))

import surface_resolver  # noqa: E402
import surfacing  # noqa: E402
import neurograph_rpc as rpc  # noqa: E402
import cc_ng_organism as cc  # noqa: E402

sys.path.insert(0, str(_ROOT))  # undo neurograph_rpc's sys.path[0] = ~/NeuroGraph

from ces_config import CESConfig  # noqa: E402
from surface_resolver import resolve_surface_content, resolve_surface_item  # noqa: E402
from surfacing import SurfacingMonitor  # noqa: E402

_CORE_MODULES = ("surface_resolver", "surfacing", "neurograph_rpc", "cc_ng_organism", "ces_config")
_ROOT_PY_NAMES = {p.name for p in _ROOT.glob("*.py")}


# ── #770 printed-path preamble ───────────────────────────────────────────────

def _ng_module_rows():
    """(name, file) for every loaded module that is an NG root module by file name."""
    rows = []
    for name, mod in sorted(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if f and Path(f).name in _ROOT_PY_NAMES:
            rows.append((name, str(Path(f).resolve())))
    return rows


def _paths_not_under_root():
    bad = []
    for name in _CORE_MODULES:
        f = getattr(sys.modules.get(name), "__file__", None)
        if not f or Path(f).resolve().parent != _ROOT:
            bad.append((name, f))
    for name, f in _ng_module_rows():
        if Path(f).parent != _ROOT:
            bad.append((name, f))
    return bad


_expect = os.environ.get("EXPECT_NG_ROOT")
if _expect and Path(_expect).resolve() != _ROOT:
    raise AssertionError(f"EXPECT_NG_ROOT={_expect} but this test file's root is {_ROOT}")
_bad_at_import = _paths_not_under_root()
if _bad_at_import:
    raise AssertionError(f"NG modules under test are NOT under {_ROOT}: {_bad_at_import}")


@pytest.fixture(scope="session", autouse=True)
def _printed_path_preamble(request):
    tr = request.config.pluginmanager.get_plugin("terminalreporter")
    write = tr.write_line if tr is not None else print
    write(f"[#770 preamble] test root (must contain every NG module under test): {_ROOT}")
    for name in _CORE_MODULES:
        write(f"[#770 preamble]   {name}.__file__ = {getattr(sys.modules.get(name), '__file__', None)}")
    others = [(n, f) for n, f in _ng_module_rows() if n not in _CORE_MODULES]
    for n, f in others:
        write(f"[#770 preamble]   {n}.__file__ = {f}")
    bad = _paths_not_under_root()
    if bad:
        pytest.exit(f"[#770] NG module(s) NOT under {_ROOT}: {bad}", returncode=3)
    write(f"[#770 preamble] OK: all {len(_ng_module_rows())} loaded NG root modules are under the test root")
    yield
    late_bad = _paths_not_under_root()
    if late_bad:  # a later-imported module came from outside the root
        pytest.fail(f"[#770] late-imported NG module(s) NOT under {_ROOT}: {late_bad}")


def test_ng_modules_under_test_are_under_this_root():
    """The guard itself, as a visible test: nothing under test came from another checkout."""
    assert _paths_not_under_root() == []
    for name in _CORE_MODULES:
        assert Path(sys.modules[name].__file__).resolve().parent == _ROOT, name


# ── fixtures / helpers ───────────────────────────────────────────────────────

# Deterministic, whitespace-bearing texts that contain no "..." / "…" of their own.
OVER_240 = " ".join(f"w{i:03d}" for i in range(80))      # 399 chars  (> 240, > 300)
OVER_1000 = " ".join(f"w{i:03d}" for i in range(260))    # 1299 chars (> 1000)
ONE_TOKEN = "x" * 1000                                   # 1000 chars, no whitespace
WHOLE_CASES = [("over240", OVER_240), ("over1000", OVER_1000), ("one_token", ONE_TOKEN)]


class _Node:
    def __init__(self, **meta):
        self.metadata = {"creation_mode": "conversational", **meta}


def _forest_node(text):
    return _Node(_forest_content=text)


def _no_ellipsis(s):
    return "…" not in s and "..." not in s


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    """The BASE e4ebf982 surface_resolver + surfacing, loaded from `git show` (or, inside a
    `git archive` copy of the base that has no .git, from the copy itself — it IS the base)."""
    out = tmp_path_factory.mktemp("base812")
    mods = {}
    for name in ("surface_resolver", "surfacing"):
        r = subprocess.run(["git", "-C", str(_ROOT), "show", f"{BASE_SHA}:{name}.py"],
                           capture_output=True, text=True)
        if r.returncode == 0:
            src = r.stdout
        elif (_ROOT / ".git").exists():
            raise AssertionError(f"git repo but base {BASE_SHA}:{name}.py unavailable: {r.stderr}")
        else:
            src = (_ROOT / f"{name}.py").read_text()
        path = out / f"{name}.py"
        path.write_text(src)
        spec = importlib.util.spec_from_file_location(f"base_{name}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mods[name] = mod
    return types.SimpleNamespace(**mods)


def _monitor(mod=surfacing, graph=None, vdb=None, min_conf=0.1):
    cfg = CESConfig()
    cfg.surfacing.min_confidence = min_conf
    return mod.SurfacingMonitor(graph, vdb, cfg)


# ── 1. resolver: WHOLE by default ────────────────────────────────────────────

@pytest.mark.parametrize("label,text", WHOLE_CASES)
def test_resolve_surface_content_is_whole(label, text):
    """FAILS on base: the default max_chars=240 cut it to <=241 chars ending in '…'."""
    out = resolve_surface_content(_forest_node(text), {"content": "shard"})
    assert out == text, f"{label}: got len {len(out)}"
    assert _no_ellipsis(out)


@pytest.mark.parametrize("label,text", WHOLE_CASES)
def test_resolve_surface_item_is_whole(label, text):
    """FAILS on base: resolve_surface_item defaulted max_chars=240 and passed it down."""
    item = resolve_surface_item(_forest_node(text), {"content": "shard"})
    assert item == {"kind": "text", "content": text}, f"{label}: got {item!r}"


def test_whole_applies_to_the_vdb_fallback_too():
    """FAILS on base: the vdb-shard fallback went through the same 240 cut."""
    out = resolve_surface_content(_Node(), {"content": OVER_1000})
    assert out == OVER_1000


def test_explicit_none_equals_default():
    """New API: max_chars=None is spelled-out 'no bound' and equals the default."""
    node = _forest_node(OVER_1000)
    assert resolve_surface_content(node, None, max_chars=None) == OVER_1000
    assert resolve_surface_item(node, None, max_chars=None) == {"kind": "text", "content": OVER_1000}


# ── 2. an EXPLICIT bound still works exactly as before ───────────────────────

def test_explicit_bound_still_clips_item_and_content():
    """Passes on base AND tip: explicit-bound callers are unchanged (len <= 241, '…')."""
    node = _forest_node(ONE_TOKEN)
    out = resolve_surface_content(node, {"content": "shard"}, max_chars=240)
    assert out == "x" * 240 + "…" and len(out) <= 241
    item = resolve_surface_item(node, {"content": "shard"}, max_chars=240)
    assert item == {"kind": "text", "content": "x" * 240 + "…"}


def test_explicit_bound_still_word_snaps():
    """Passes on base AND tip: the explicit-bound branch keeps its word snap."""
    forest = "Alpha bravo charlie delta echo foxtrot golf hotel india juliet"
    out = resolve_surface_item(_forest_node(forest), None, max_chars=27)
    assert out == {"kind": "text", "content": "Alpha bravo charlie delta…"}


# ── 3. format_context: WHOLE ─────────────────────────────────────────────────

@pytest.mark.parametrize("label,text", WHOLE_CASES)
def test_format_context_renders_whole(label, text):
    """FAILS on base: format_context cut at 200 chars (197 + '...') after a word snap."""
    ctx = _monitor().format_context([{"node_id": "n", "content": text, "score": 1.23}])
    assert ctx.splitlines()[0] == "[NeuroGraph Surfaced Knowledge]"
    assert ctx.splitlines()[1] == f"- {text} (salience: 1.23)", f"{label}"
    assert _no_ellipsis(ctx)


class _FakeGraph:
    def __init__(self, nodes):
        self.nodes = nodes
        self.hyperedges = {}
        self.timestep = 1
        self.config = {}


class _FakeVdb:
    def get(self, node_id):
        return {"content": "shard", "metadata": {}}


@pytest.mark.parametrize("label,text", WHOLE_CASES)
def test_resolver_to_format_context_chain_is_whole(label, text):
    """FAILS on base: BOTH cuts fired in series (resolver 240 '…', then format 197 '...')."""
    node = types.SimpleNamespace(voltage=2.0, threshold=1.0, intrinsic_excitability=1.0,
                                 metadata={"creation_mode": "conversational", "_forest_content": text})
    mon = _monitor(graph=_FakeGraph({"n1": node}), vdb=_FakeVdb())
    mon.after_step(types.SimpleNamespace(fired_node_ids=["n1"]))
    items = mon.get_surfaced()
    assert [i["content"] for i in items] == [text], label
    ctx = mon.format_context(items)
    assert f"- {text} (salience:" in ctx and _no_ellipsis(ctx)


# ── 4. byte-identical to base where nothing was cut ──────────────────────────

SHORT_NODES = [
    ("forest", _forest_node("A short remembered turn, well under the old bound.")),
    ("exactly_240", _forest_node("y" * 240)),
    ("vdb_fallback", _Node()),
    ("label_fallback", _Node(_label="a node label long enough to surface")),
    ("sub_floor", _forest_node("short")),
    ("stopword", _forest_node("want")),
    ("ingested", _Node(creation_mode="ingested", _forest_content="doc text that is long enough")),
]


@pytest.mark.parametrize("label,node", SHORT_NODES)
@pytest.mark.parametrize("allow_ingested", [False, True])
def test_short_and_degenerate_resolution_identical_to_base(base, label, node, allow_ingested):
    """Unchanged behaviour: <=240, degenerate, ingested and fallback resolve exactly as base."""
    vdb = {"content": "a vdb shard sentence of reasonable length"}
    for fn in ("resolve_surface_content", "resolve_surface_item"):
        want = getattr(base.surface_resolver, fn)(node, vdb, allow_ingested=allow_ingested)
        got = getattr(surface_resolver, fn)(node, vdb, allow_ingested=allow_ingested)
        assert got == want, f"{fn}/{label}/allow_ingested={allow_ingested}"


def test_image_item_identical_to_base(base, tmp_path):
    """Unchanged behaviour: a vision forest with its frame on disk is an image item."""
    img = tmp_path / "frame.png"
    img.write_bytes(b"\x89PNG")
    node = _Node(modality="vision", kind="forest", _image_ref=str(img))
    want = base.surface_resolver.resolve_surface_item(node, None)
    got = resolve_surface_item(node, None)
    assert got == want == {"kind": "image", "image_ref": str(img)}
    missing = _Node(modality="vision", kind="forest", _image_ref=str(tmp_path / "gone.png"))
    assert resolve_surface_item(missing, None) == base.surface_resolver.resolve_surface_item(missing, None)


def test_format_context_header_label_image_order_identical_to_base(base, tmp_path):
    """Unchanged behaviour: header marker, salience label, image line, empty string and
    INPUT ORDER (format_context does not re-sort) are byte-identical for non-cut items."""
    items = [
        {"node_id": "a", "content": "first, lower score", "score": 0.91},
        {"node_id": "b", "content": "", "image_ref": "/x/frame.png", "score": 1.42},
        {"node_id": "c", "content": "z" * 200, "score": 1.07},          # exactly at the old bound
        {"node_id": "d", "content": "last, highest score", "score": 1.8},
    ]
    want = _monitor(base.surfacing).format_context(items)
    got = _monitor().format_context(items)
    assert got == want
    assert got.splitlines()[0] == "[NeuroGraph Surfaced Knowledge]"
    assert "[something you saw — image attached] (salience: 1.42)" in got
    assert "(salience: 0.91)" in got
    assert [l.split(" (salience")[0] for l in got.splitlines()[1:]] == [
        "- first, lower score", "- [something you saw — image attached]", "- " + "z" * 200,
        "- last, highest score"]
    assert _monitor().format_context([]) == _monitor(base.surfacing).format_context([]) == ""


def test_get_surfaced_ordering_is_score_descending():
    """Unchanged behaviour: get_surfaced still sorts by score, descending."""
    nodes = {
        f"n{i}": types.SimpleNamespace(voltage=2.0, threshold=1.0, intrinsic_excitability=ex,
                                       metadata={"creation_mode": "conversational",
                                                 "_forest_content": f"node number {i} content"})
        for i, ex in enumerate((1.0, 2.0, 1.5))
    }
    mon = _monitor(graph=_FakeGraph(nodes), vdb=_FakeVdb())
    mon.after_step(types.SimpleNamespace(fired_node_ids=list(nodes)))
    scores = [i["score"] for i in mon.get_surfaced()]
    assert scores == sorted(scores, reverse=True) and len(scores) == 3


# ── 5. the two max_chars=300 sites (Chief addendum 1) ────────────────────────

class _RpcMemory:
    """Minimal in-process stand-in for NeuroGraphMemory — exactly what handle_assemble reads.
    No graph load, no checkpoint, no daemon."""

    def __init__(self, nodes, recalled, surfaced_monitor=None):
        self.graph = types.SimpleNamespace(nodes=nodes)
        self.vector_db = _FakeVdb()
        self._surfacing_monitor = surfaced_monitor
        self._tonic_thread = None
        self._recalled = recalled
        self._substrate_novelty_ema = 0.5

    def _harvest_associations(self, text, novelty=0.5, **kw):
        return []

    def recall(self, text, k=5, threshold=0.4):
        return self._recalled


class _FakeMonitor:
    def __init__(self, items):
        self._items = items

    def get_surfaced(self):
        return list(self._items)


def _assemble(monkeypatch, memory):
    """Run the REAL neurograph_rpc.handle_assemble in-process against the fake memory."""
    stub_embed = types.ModuleType("ng_embed")

    def _raise(*a, **k):
        raise RuntimeError("embedding model deliberately not loaded in this test")
    stub_embed.embed = _raise
    monkeypatch.setitem(sys.modules, "ng_embed", stub_embed)  # GSG blocks fail soft, as designed
    kiss = types.SimpleNamespace(
        _config=types.SimpleNamespace(recent_window=10),
        filter_context=lambda msgs, system_context="": {"kiss_meta": {}, "system_context": "", "kiss_mode": "off"})
    monkeypatch.setattr(rpc, "_memory", memory)
    monkeypatch.setattr(rpc, "_kiss_filter", kiss)
    monkeypatch.setattr(rpc, "_read_outbound_log", lambda *a, **k: None)
    monkeypatch.setattr(rpc, "_render_self_and_wants", lambda graph: "")
    monkeypatch.delenv("ANIMUS_TONIC_BRIDGE_ENABLED", raising=False)
    out = rpc.handle_assemble({"messages": [{"role": "user", "content": "what do you remember about this"}]})
    return out["systemPromptAddition"] or ""


@pytest.mark.parametrize("label,text", [("over300", OVER_240), ("over1000", OVER_1000)])
def test_active_recall_block_of_assemble_renders_whole(monkeypatch, label, text):
    """neurograph_rpc.py Active Recall site (was max_chars=300).
    FAILS on base: the call passed max_chars=300, so a >300-char node came back as a
    word-snapped snippet ending in '…'. Exercised: the real handle_assemble, in-process,
    fake memory. NOT exercised: a real graph, the daemon, the HTTP sidecar."""
    mem = _RpcMemory({"n1": _forest_node(text)}, [{"node_id": "n1", "similarity": 0.91, "content": "shard"}])
    block = _assemble(monkeypatch, mem)
    assert "## Active Recall" in block, block
    assert f"- [0.91] {text}" in block, f"{label}: Active Recall line not whole"
    assert _no_ellipsis(block.split("## Active Recall", 1)[1])


@pytest.mark.parametrize("label,text", [("over300", OVER_240), ("over1000", OVER_1000)])
def test_cc_pattern_completion_recall_renders_whole(monkeypatch, label, text):
    """cc_ng_organism.py CC prefetch site (was max_chars=300).
    FAILS on base: same 300 bound. Exercised: the real cc_pattern_completion_recall with a
    fake ng (graph.nodes/config + _harvest_associations). NOT exercised: a real graph, Pith,
    cc_l1_budget (downstream, gated off by default)."""
    stub_embed = types.ModuleType("ng_embed")

    def _raise(*a, **k):
        raise RuntimeError("embedding model deliberately not loaded in this test")
    stub_embed.embed = _raise
    monkeypatch.setitem(sys.modules, "ng_embed", stub_embed)  # cc_gsg_rescore fails soft
    graph = types.SimpleNamespace(nodes={"n1": _forest_node(text)}, config={})
    ng = types.SimpleNamespace(
        graph=graph,
        _harvest_associations=lambda q, novelty=0.5, **kw: [{"node_id": "n1", "strength": 1.0, "content": "shard"}])
    out = cc.cc_pattern_completion_recall(ng, "what do you remember", k=5)
    assert len(out) == 1, f"{label}: recall returned {out!r} (it swallows exceptions and returns [])"
    assert out[0]["content"] == text and _no_ellipsis(out[0]["content"])


@pytest.mark.xfail(strict=True, reason=(
    "flag (e), NOT fixed in #812 turn 1 (listed, per the brief): neurograph_rpc._format_substrate_context "
    "re-clips surfaced / ces_surfaced items to 300 chars (content[:297] + '...') AFTER the resolver, so "
    "Syl's Substrate Context block is still lossy even though the resolver is now whole. strict=True: "
    "when that clip is removed this XPASSes and fails, forcing this marker's removal."))
def test_flag_e_ces_surfaced_whole_through_substrate_context(monkeypatch):
    mon = _FakeMonitor([{"node_id": "n1", "score": 1.0, "content": "shard"}])
    mem = _RpcMemory({"n1": _forest_node(OVER_1000)}, [], surfaced_monitor=mon)
    block = _assemble(monkeypatch, mem)
    assert f"- [CES] {OVER_1000}" in block
