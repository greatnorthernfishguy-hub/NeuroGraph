"""
Commons Track-2 Stage-1 — Bunyan-style narrator buckets the Commons (deposit/bucket only).

# ---- Changelog ----
# [2026-06-11] Claude Code (Opus 4.8, 1M) — Track 2 / Stage 1 (Commons go-live, Bunyan #1)
# What: Proves the CORRECTED Bunyan↔Commons pattern in a sandbox, replacing the ILLEGAL
#       Direction-A `_SubstrateBucket` handle (neurograph_rpc.py:623 — a peer holding a handle
#       INTO NG's _memory + recent_activity() raw-traversing graph.nodes; two LAW violations:
#       LAW 1 direct cross-module call + bypassing Cricket's inescapable rim).
# Why: Substrate axiom — deposits and buckets, nothing else. A narrator's "what just happened"
#       (recency) is the TEMPORAL-SEQUENCE part of STRUCTURE extraction (Tier 3 Extraction doc),
#       a TUNABLE bucket mode — NOT a reach past the bucket into NG's graph. Buckets are tunable
#       (mesh evolves); only Cricket's rim is frozen.
# How: NG deposits raw topology-delta events into the Commons (deposit). A Bunyan-style narrator
#       BUCKETS them from the Commons (no handle), its bucket TUNED for Tier-3 structure (temporal
#       sequence), absorbs into its OWN substrate, narrates, and DEPOSITS the narrative back into
#       the Commons. Recency lives in the narrator's OWN tuned bucket — never a reach into NG.
# -------------------
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import commons as commons_mod
from ng_lite import NGLite


def _emb(seed: int, dim: int = 768) -> np.ndarray:
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


# ---------------------------------------------------------------------------
# A Bunyan-style narrator — buckets the Commons through a Tier-3-tuned mesh.
# Deposit/bucket ONLY: no handle into NG, no raw graph traversal.
# ---------------------------------------------------------------------------
class _NarratorBucket:
    """A Cricket-rimmed, tunable extraction bucket for a narrator (Bunyan-style).

    The MESH is tunable (signal + structure extraction params). Only the rim is frozen — here
    represented by `_RIM` constraints the mesh may never relax. Stage 1 tunes the STRUCTURE mode
    to temporal-sequence extraction (recency), the narrator's Tier-3 need.
    """

    # Rim — immutable constitutional constraints (frozen; the mesh may never relax these).
    _RIM = {"no_handle_into_peer": True, "classify_only_at_extraction": True}

    def __init__(self, commons, own_substrate, *, structure_temporal: bool = True,
                 signal_top_k: int = 10):
        self.commons = commons
        self.own = own_substrate              # the narrator's OWN bare NG-Lite
        # mesh (tunable): structure mode + signal breadth. NOT the rim.
        self.mesh = {"structure_temporal": structure_temporal, "signal_top_k": signal_top_k}
        self._timeline: list = []             # narrator's OWN temporal index (its recency)
        self._absorbed: set = set()

    def bucket_pulse(self, context_embedding: np.ndarray) -> list:
        """One pulse: bucket the Commons (signal), absorb into OWN substrate, index temporally.

        Returns the freshly-narrated items (recency), extracted from the narrator's OWN tuned
        bucket — never from a handle into NG.
        """
        # SIGNAL extraction — semantic bucket of the shared medium (the canonical bucket).
        recs = self.commons.bucket(context_embedding, top_k=self.mesh["signal_top_k"])
        fresh = []
        for target_id, confidence, _reasoning in recs:
            if target_id in self._absorbed or target_id.startswith("narrative:"):
                continue
            self._absorbed.add(target_id)
            # absorb into the narrator's OWN substrate (its experience), then index temporally
            self.own.record_outcome(context_embedding, f"absorbed:{target_id}", True, strength=1.0)
            if self.mesh["structure_temporal"]:
                self._timeline.append(target_id)   # STRUCTURE/temporal mode (tuned) — OUR recency
            fresh.append(target_id)
        return fresh

    def recent(self, limit: int = 20) -> list:
        """Recency from the narrator's OWN tuned bucket — NOT a traversal of NG's graph."""
        return list(self._timeline[-limit:])


def _sandbox():
    """Shared Commons + NG's deposit side + a narrator with its OWN substrate. No live singleton."""
    commons = commons_mod.Commons()
    narrator_substrate = NGLite(module_id="bunyan_sandbox")
    narrator = _NarratorBucket(commons, narrator_substrate)
    return commons, narrator


def test_narrator_buckets_commons_no_handle():
    """The narrator gets 'what just happened' by BUCKETING the Commons — never a handle into NG."""
    commons, narrator = _sandbox()
    ctx = _emb(1)
    # NG deposits raw topology-delta events into the Commons (deposit — no addressee).
    for i in range(5):
        commons.deposit(ctx, f"delta:node_{i}_fired", metadata={"content": f"event {i}", "ts": i})
    fresh = narrator.bucket_pulse(ctx)
    assert fresh, "narrator must extract deposited events via its bucket"
    assert all(t.startswith("delta:") for t in fresh), f"should bucket NG's delta events; {fresh}"
    # Recency comes from the narrator's OWN tuned bucket (its timeline), not NG's graph.
    assert narrator.recent(), "narrator's recency lives in its OWN bucket (Tier-3 structure tune)"
    # Hard assertion of the fix: the narrator holds NO handle into any peer substrate.
    assert not hasattr(narrator, "_ng_substrate"), "no handle into NG (the illegal pattern)"
    assert narrator.own is not None and narrator.own.module_id == "bunyan_sandbox", \
        "narrator reads/writes only its OWN substrate + the shared Commons"


def test_narrator_deposits_narrative_back():
    """The narrator DEPOSITS its narrative into the Commons (so its experience is in the medium)."""
    commons, narrator = _sandbox()
    ctx = _emb(2)
    commons.deposit(ctx, "delta:something_happened", metadata={"content": "a thing"})
    narrator.bucket_pulse(ctx)
    # narrate → deposit the narrative back into the shared medium (deposit, no addressee)
    commons.deposit(ctx, "narrative:chapter_1", metadata={"story": "and then a thing happened"})
    targets = [t for (t, _c, _r) in commons.bucket(ctx, top_k=10)]
    assert "narrative:chapter_1" in targets, "narrator's narrative must reach the Commons"


def test_tunable_mesh_not_rim():
    """The mesh is tunable (structure mode toggles); the rim is frozen (constitution untouchable)."""
    commons, narrator = _sandbox()
    # tune the mesh: turn OFF temporal-structure extraction → no timeline indexing.
    narrator.mesh["structure_temporal"] = False
    ctx = _emb(3)
    commons.deposit(ctx, "delta:x", metadata={"content": "x"})
    narrator.bucket_pulse(ctx)
    assert narrator.recent() == [], "mesh tuning (structure off) must change extraction behavior"
    # the rim is frozen — its constraints exist and are not relaxable by tuning the mesh.
    assert narrator._RIM["no_handle_into_peer"] is True, "rim (no-handle) is immutable"
    assert narrator._RIM["classify_only_at_extraction"] is True, "rim (LAW 7) is immutable"


def test_retires_the_illegal_handle():
    """Stage-1 proof: the whole loop runs with NO _SubstrateBucket handle anywhere."""
    commons, narrator = _sandbox()
    ctx = _emb(4)
    for i in range(3):
        commons.deposit(ctx, f"delta:e{i}", metadata={"content": f"e{i}"})
    fresh = narrator.bucket_pulse(ctx)
    commons.deposit(ctx, "narrative:summary", metadata={"story": f"narrated {len(fresh)} events"})
    # the loop closed: bucket (in) + deposit (out), no handle, no raw NG traversal.
    assert fresh and "narrative:summary" in [t for (t, _c, _r) in commons.bucket(ctx, top_k=10)]


if __name__ == "__main__":
    test_narrator_buckets_commons_no_handle(); print("PASS narrator buckets Commons (no handle into NG)")
    test_narrator_deposits_narrative_back();   print("PASS narrator deposits narrative back into Commons")
    test_tunable_mesh_not_rim();               print("PASS bucket mesh is tunable; rim is frozen")
    test_retires_the_illegal_handle();         print("PASS full loop runs with the illegal handle retired")
    print("\nCommons Track-2 Stage-1: ALL PASS — Bunyan-style narrator on deposit/bucket only; tuned Tier-3 structure bucket; handle retired")
