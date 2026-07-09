# ---- Changelog ----
# [2026-07-08] Claude Code (Fable 5 design / Haiku implementation) — CC Lenia wiring tests
# What: pins that bootstrap_lenia() actually PASSES the checkpoint/resume machinery to
#   DistanceCache.populate() in all three branches, and that the resume branch is chosen
#   when a loaded cache carries a watermark. Call-site wiring only — the machinery itself
#   is covered by tests/test_lenia_resume_watermark.py.
# Why: this exact gap (capability present in the class, params never passed by the caller)
#   is what left CC's rebuilds unprotected for two days while Syl's were fixed — a wiring
#   pin makes the omission class test-visible.
# How: real Graph + real bootstrap_lenia(); DistanceCache.populate/load monkeypatched via
#   the lenia.kernel module to capture kwargs / inject a watermark-carrying cache. The rest
#   of the lenia stack constructs for real (tmp field_dir), matching production flow.
# -------------------
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tempfile, shutil
import logging
import pytest

from neuro_foundation import Graph
from cc_ng_organism import bootstrap_lenia


def _graph(n=6):
    g = Graph()
    for i in range(n):
        g.create_node(node_id=f"n{i}")
    for i in range(n - 1):
        g.create_synapse(f"n{i}", f"n{i + 1}", weight=0.5)
    return g


@pytest.fixture
def workspace():
    d = tempfile.mkdtemp(prefix="cc_lenia_wiring_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_full_rebuild_passes_checkpoint_params(workspace, monkeypatch):
    import lenia.kernel as lk
    from cc_ng_organism import bootstrap_lenia, _CC_LENIA_CHECKPOINT_INTERVAL_SECS
    captured = {}
    real_populate = lk.DistanceCache.populate

    def spy(self, substrate, **kwargs):
        captured.update(kwargs)
        return real_populate(self, substrate, **kwargs)

    monkeypatch.setattr(lk.DistanceCache, "populate", spy)
    g = _graph()
    result = bootstrap_lenia(g, None, workspace)
    assert captured.get("checkpoint_interval_secs") == _CC_LENIA_CHECKPOINT_INTERVAL_SECS
    assert callable(captured.get("on_checkpoint"))
    assert result.get("substrate") is not None  # bootstrap still completes


def test_resume_branch_chosen_when_cache_has_watermark(workspace, monkeypatch):
    import lenia.kernel as lk
    from cc_ng_organism import bootstrap_lenia

    g = _graph()
    # First bootstrap builds + saves a real, complete cache.
    bootstrap_lenia(g, None, workspace)

    # Forge an interruption: reload the saved cache, stamp a watermark on it,
    # and hand exactly that object to the next bootstrap via load().
    import os
    cache_path = os.path.join(workspace, "lenia", "distance_cache")
    forged = lk.DistanceCache.load(cache_path)
    assert forged is not None
    forged._watermark = (0, 1)
    monkeypatch.setattr(lk.DistanceCache, "load", classmethod(lambda cls, p: forged))

    captured = {}
    real_populate = lk.DistanceCache.populate

    def spy(self, substrate, **kwargs):
        captured.update(kwargs)
        return real_populate(self, substrate, **kwargs)

    monkeypatch.setattr(lk.DistanceCache, "populate", spy)
    bootstrap_lenia(g, None, workspace)
    assert captured.get("resume_watermark") == (0, 1), (
        "a watermark-carrying cache must take the resume branch")
    assert captured.get("checkpoint_interval_secs") is not None


def test_growth_branch_passes_checkpoint_params(workspace, monkeypatch):
    import lenia.kernel as lk
    from cc_ng_organism import bootstrap_lenia

    g = _graph()
    bootstrap_lenia(g, None, workspace)   # complete cache on disk

    # Grow the graph; next bootstrap takes the incremental branch.
    for i in range(6, 9):
        g.create_node(node_id=f"n{i}")
    for i in range(5, 8):
        g.create_synapse(f"n{i}", f"n{i + 1}", weight=0.5)

    captured = {}
    real_populate = lk.DistanceCache.populate

    def spy(self, substrate, **kwargs):
        captured.update(kwargs)
        return real_populate(self, substrate, **kwargs)

    monkeypatch.setattr(lk.DistanceCache, "populate", spy)
    bootstrap_lenia(g, None, workspace)
    assert captured.get("start_index") == 6
    assert captured.get("checkpoint_interval_secs") is not None
    assert callable(captured.get("on_checkpoint"))


def test_field_dir_exists_before_first_checkpoint_could_fire(workspace, monkeypatch):
    """First-ever run: the periodic checkpoint's save() must have a directory
    to write into BEFORE populate runs — not only in the post-populate block."""
    import os
    import lenia.kernel as lk
    from cc_ng_organism import bootstrap_lenia

    seen = {}
    real_populate = lk.DistanceCache.populate

    def spy(self, substrate, **kwargs):
        seen["dir_exists_at_populate"] = os.path.isdir(os.path.join(workspace, "lenia"))
        return real_populate(self, substrate, **kwargs)

    monkeypatch.setattr(lk.DistanceCache, "populate", spy)
    bootstrap_lenia(_graph(), None, workspace)
    assert seen.get("dir_exists_at_populate") is True


def test_bootstrap_reconciles_pruned_cache_instead_of_full_rebuild(tmp_path, caplog):
    """#371: a node pruned between saves must NOT force a full repopulate —
    bootstrap_lenia reconciles the on-disk cache and reuses it."""
    g = Graph()
    for i in range(8):
        g.create_node(node_id=f"n{i}")
    for i in range(7):
        g.create_synapse(f"n{i}", f"n{i + 1}", weight=0.5)

    ws = str(tmp_path / "ws")
    first = bootstrap_lenia(g, None, ws)
    assert first.get("engine") is not None  # populate completed + saved

    g.remove_node("n4")

    caplog.clear()
    with caplog.at_level(logging.INFO):
        second = bootstrap_lenia(g, None, ws)
    assert second.get("engine") is not None
    joined = "\n".join(r.getMessage() for r in caplog.records)
    assert "reconciled after prune" in joined
    assert "full repopulate" not in joined
    assert "full rebuild required" not in joined
