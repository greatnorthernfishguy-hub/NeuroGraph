# ---- Changelog ----
# [2026-07-07] Claude Code (Fable 5) — Regression: orphaned prime seeds must not kill the harvest
# What: Pins the 2026-07-07 openclaw_hook.py fix — _harvest_associations skips vdb
#   search hits whose graph node was orphan-pruned instead of handing them to
#   prime_and_propagate (where one dead ID raised KeyError and the whole harvest
#   silently returned []).
# Why: The failure was invisible (fail-soft + logger.debug). Constant for CC
#   (vdb >> live graph nodes), intermittent for Syl. This test makes any
#   reintroduction loud: an orphan as the TOP similarity hit must not prevent a
#   live, synaptically-relevant seed from surfacing.
# How: Real NeuroGraphMemory (no mocks — the vdb/graph membership interaction IS
#   what's under test). Orphan = vdb insert with no create_node. Live seed gets a
#   lowered threshold for deterministic firing (same technique as
#   test_cc_retrieval_enrichment's substrate-primacy test).
# -------------------
import sys
sys.path.insert(0, '/home/josh/NeuroGraph')
import tempfile, shutil
import pytest


@pytest.fixture
def cc_ng():
    from openclaw_hook import NeuroGraphMemory
    workspace = tempfile.mkdtemp(prefix='harvest_orphan_test_')
    ng = NeuroGraphMemory(workspace_dir=workspace,
                          config={"tonic": {"enabled": False}, "peer_bridge": {"enabled": False}})
    yield ng
    shutil.rmtree(workspace, ignore_errors=True)


def test_orphan_top_hit_does_not_kill_harvest(cc_ng):
    from ng_embed import embed
    query = "the lenia distance cache rebuild finished"
    # Orphan: vdb entry nearly identical to the query, NO graph node — will be
    # the top similarity hit, exactly the live failure shape.
    cc_ng.vector_db.insert(id="orphan", embedding=embed("the lenia distance cache rebuild"),
                           content="the lenia distance cache rebuild")
    assert "orphan" not in cc_ng.graph.nodes
    # Live seed: real node + vdb entry, deterministic firing.
    cc_ng.graph.create_node(node_id="live")
    cc_ng.graph.nodes["live"].threshold = 0.1
    cc_ng.vector_db.insert(id="live", embedding=embed("lenia cache rebuild status"),
                           content="lenia cache rebuild status")

    surfaced = cc_ng._harvest_associations(query)

    ids = [s["node_id"] for s in surfaced]
    assert "live" in ids, (
        "live seed must survive an orphaned top hit -- pre-fix, the orphan's "
        "KeyError silently emptied the whole harvest; got %r" % ids)
    assert "orphan" not in ids


def test_all_orphan_seeds_returns_empty_without_raising(cc_ng):
    from ng_embed import embed
    cc_ng.vector_db.insert(id="only_orphan", embedding=embed("completely orphaned memory"),
                           content="completely orphaned memory")
    assert cc_ng._harvest_associations("completely orphaned memory") == []
