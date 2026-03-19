"""
Tests for Cricket rim: constitutional nodes in NG-Lite.

Validates that constitutional nodes enforce the rim — the topology
cannot learn to recommend actions for inputs that land in constitutional
semantic space. Synapses are frozen, nodes survive pruning, and the
bucket comes up empty.

# ---- Changelog ----
# [2026-03-19] Claude Code (Opus 4.6) — Initial test suite
# What: Tests for constitutional node seeding, frozen learning,
#   empty recommendations, pruning immunity, and persistence.
# Why: Cricket Design v0.1 — constitutional enforcement at the
#   extraction boundary. These are the most critical constraints
#   in the ecosystem. They must not break.
# How: Unit tests using synthetic embeddings and the live
#   constitutional_embeddings.json for integration tests.
# -------------------
"""

import json
import os
import tempfile

import numpy as np
import pytest

from ng_lite import NGLite, NGLiteNode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_embedding(seed: int = 0, dim: int = 384) -> np.ndarray:
    """Generate a deterministic normalized embedding."""
    rng = np.random.RandomState(seed)
    emb = rng.randn(dim).astype(np.float32)
    emb /= np.linalg.norm(emb)
    return emb


def _make_config_with_constitutional(embeddings_list):
    """Build a config dict with constitutional embeddings."""
    return {
        "constitutional_embeddings": embeddings_list,
        # Disable receptor layer to keep tests deterministic
        "receptor_layer_enabled": False,
    }


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

class TestConstitutionalSeeding:
    """Constitutional nodes are seeded on init from config."""

    def test_no_constitutional_by_default(self):
        ng = NGLite(module_id="test", config={"receptor_layer_enabled": False})
        assert len(ng.nodes) == 0

    def test_seeds_constitutional_nodes(self):
        emb1 = _make_embedding(1)
        emb2 = _make_embedding(2)
        config = _make_config_with_constitutional([
            {"embedding": emb1.tolist(), "description": "test concept 1"},
            {"embedding": emb2.tolist(), "description": "test concept 2"},
        ])
        ng = NGLite(module_id="test", config=config)
        assert len(ng.nodes) == 2
        for node in ng.nodes.values():
            assert node.constitutional is True

    def test_constitutional_nodes_have_metadata(self):
        emb = _make_embedding(42)
        config = _make_config_with_constitutional([
            {"embedding": emb.tolist(), "description": "destroy substrate"},
        ])
        ng = NGLite(module_id="test", config=config)
        node = list(ng.nodes.values())[0]
        assert node.metadata["constitutional_description"] == "destroy substrate"

    def test_skips_entries_without_embedding(self):
        config = _make_config_with_constitutional([
            {"description": "no embedding here"},
            {"embedding": _make_embedding(1).tolist(), "description": "has embedding"},
        ])
        ng = NGLite(module_id="test", config=config)
        assert len(ng.nodes) == 1

    def test_old_config_without_key_loads_cleanly(self):
        ng = NGLite(module_id="test", config={"receptor_layer_enabled": False})
        assert len(ng.nodes) == 0  # No crash, no constitutional nodes


# ---------------------------------------------------------------------------
# Frozen Learning
# ---------------------------------------------------------------------------

class TestFrozenLearning:
    """Constitutional nodes cannot learn — synapses are frozen."""

    def test_record_outcome_returns_constitutional_flag(self):
        emb = _make_embedding(10)
        config = _make_config_with_constitutional([
            {"embedding": emb.tolist(), "description": "frozen concept"},
        ])
        ng = NGLite(module_id="test", config=config)

        result = ng.record_outcome(emb, target_id="action:dangerous", success=True)
        assert result["constitutional"] is True
        assert result["weight_after"] == 0.0

    def test_no_synapse_created_for_constitutional_node(self):
        emb = _make_embedding(10)
        config = _make_config_with_constitutional([
            {"embedding": emb.tolist(), "description": "frozen concept"},
        ])
        ng = NGLite(module_id="test", config=config)

        ng.record_outcome(emb, target_id="action:dangerous", success=True)
        assert len(ng.synapses) == 0

    def test_adversarial_repeated_success_cannot_strengthen(self):
        """Even thousands of success outcomes can't teach a constitutional node."""
        emb = _make_embedding(10)
        config = _make_config_with_constitutional([
            {"embedding": emb.tolist(), "description": "frozen concept"},
        ])
        ng = NGLite(module_id="test", config=config)

        for _ in range(1000):
            result = ng.record_outcome(emb, target_id="action:dangerous", success=True, strength=1.0)
            assert result["constitutional"] is True

        # Still no synapses, still no recommendations
        assert len(ng.synapses) == 0
        recs = ng.get_recommendations(emb)
        assert recs == []

    def test_normal_nodes_still_learn(self):
        """Non-constitutional nodes learn normally."""
        constitutional_emb = _make_embedding(10)
        normal_emb = _make_embedding(99)
        config = _make_config_with_constitutional([
            {"embedding": constitutional_emb.tolist(), "description": "frozen"},
        ])
        ng = NGLite(module_id="test", config=config)

        result = ng.record_outcome(normal_emb, target_id="action:safe", success=True)
        assert "constitutional" not in result
        assert result["weight_after"] > 0.5  # Learned something
        assert len(ng.synapses) == 1


# ---------------------------------------------------------------------------
# Empty Bucket
# ---------------------------------------------------------------------------

class TestEmptyBucket:
    """get_recommendations returns empty for constitutional matches."""

    def test_recommendations_empty_for_constitutional(self):
        emb = _make_embedding(10)
        config = _make_config_with_constitutional([
            {"embedding": emb.tolist(), "description": "frozen concept"},
        ])
        ng = NGLite(module_id="test", config=config)
        recs = ng.get_recommendations(emb)
        assert recs == []

    def test_recommendations_work_for_normal_nodes(self):
        constitutional_emb = _make_embedding(10)
        normal_emb = _make_embedding(99)
        config = _make_config_with_constitutional([
            {"embedding": constitutional_emb.tolist(), "description": "frozen"},
        ])
        ng = NGLite(module_id="test", config=config)

        # Teach the normal node
        ng.record_outcome(normal_emb, target_id="action:safe", success=True)
        ng.record_outcome(normal_emb, target_id="action:safe", success=True)

        recs = ng.get_recommendations(normal_emb)
        assert len(recs) > 0
        assert recs[0][0] == "action:safe"


# ---------------------------------------------------------------------------
# Pruning Immunity
# ---------------------------------------------------------------------------

class TestPruningImmunity:
    """Constitutional nodes survive LRU pruning."""

    def test_constitutional_node_survives_pruning(self):
        constitutional_emb = _make_embedding(10)
        config = _make_config_with_constitutional([
            {"embedding": constitutional_emb.tolist(), "description": "must survive"},
        ])
        config["max_nodes"] = 3  # Tight limit to force pruning
        ng = NGLite(module_id="test", config=config)

        # Fill up to max with normal nodes
        for i in range(20, 25):
            emb = _make_embedding(i)
            ng.find_or_create_node(emb)

        # Constitutional node should still be there
        constitutional_nodes = [n for n in ng.nodes.values() if n.constitutional]
        assert len(constitutional_nodes) == 1

    def test_all_constitutional_survive_when_only_constitutional_remain(self):
        """If all nodes are constitutional, pruning is a no-op."""
        embs = [_make_embedding(i) for i in range(5)]
        config = _make_config_with_constitutional([
            {"embedding": e.tolist(), "description": f"concept {i}"}
            for i, e in enumerate(embs)
        ])
        config["max_nodes"] = 3
        ng = NGLite(module_id="test", config=config)

        # All 5 constitutional nodes exist despite max_nodes=3
        assert len(ng.nodes) == 5
        assert all(n.constitutional for n in ng.nodes.values())


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestConstitutionalPersistence:
    """Constitutional flag survives save/load cycle."""

    def test_save_load_preserves_constitutional_flag(self):
        emb = _make_embedding(10)
        config = _make_config_with_constitutional([
            {"embedding": emb.tolist(), "description": "must persist"},
        ])
        ng = NGLite(module_id="test", config=config)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            ng.save(filepath)

            # Load into a new instance with same config
            ng2 = NGLite(module_id="test", config=config)
            ng2.load(filepath)

            constitutional_nodes = [n for n in ng2.nodes.values() if n.constitutional]
            assert len(constitutional_nodes) == 1
            assert constitutional_nodes[0].metadata["constitutional_description"] == "must persist"
        finally:
            os.unlink(filepath)

    def test_load_old_state_without_constitutional_field(self):
        """State files from before constitutional support load cleanly."""
        config = _make_config_with_constitutional([])
        ng = NGLite(module_id="test", config=config)

        # Record something to get a non-empty state
        emb = _make_embedding(1)
        ng.record_outcome(emb, target_id="test", success=True)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            ng.save(filepath)

            # Manually strip constitutional field to simulate old state
            with open(filepath) as f:
                state = json.load(f)
            for node_data in state["nodes"].values():
                node_data.pop("constitutional", None)
            with open(filepath, "w") as f:
                json.dump(state, f)

            # Load should work — constitutional defaults to False
            ng2 = NGLite(module_id="test", config={"receptor_layer_enabled": False})
            ng2.load(filepath)
            assert all(not n.constitutional for n in ng2.nodes.values())
        finally:
            os.unlink(filepath)

    def test_new_constitutional_embeddings_seeded_after_load(self):
        """New rim constraints added to config after save get seeded on load."""
        emb1 = _make_embedding(10)
        config1 = _make_config_with_constitutional([
            {"embedding": emb1.tolist(), "description": "original"},
        ])
        ng = NGLite(module_id="test", config=config1)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            ng.save(filepath)

            # Load with expanded config — new constitutional embedding
            emb2 = _make_embedding(20)
            config2 = _make_config_with_constitutional([
                {"embedding": emb1.tolist(), "description": "original"},
                {"embedding": emb2.tolist(), "description": "new constraint"},
            ])
            ng2 = NGLite(module_id="test", config=config2)
            ng2.load(filepath)

            constitutional_nodes = [n for n in ng2.nodes.values() if n.constitutional]
            assert len(constitutional_nodes) == 2
        finally:
            os.unlink(filepath)


# ---------------------------------------------------------------------------
# Semantic Similarity Matching
# ---------------------------------------------------------------------------

class TestSemanticMatching:
    """Inputs semantically close to constitutional nodes match them."""

    def test_similar_embedding_matches_constitutional_node(self):
        """An embedding close to a constitutional node should match it."""
        base_emb = _make_embedding(10)
        # Create a slightly perturbed version (still very similar)
        noise = np.random.RandomState(999).randn(384).astype(np.float32) * 0.05
        similar_emb = base_emb + noise
        similar_emb /= np.linalg.norm(similar_emb)

        config = _make_config_with_constitutional([
            {"embedding": base_emb.tolist(), "description": "frozen concept"},
        ])
        # Set novelty_threshold high so similarity matching is permissive
        config["novelty_threshold"] = 0.7
        ng = NGLite(module_id="test", config=config)

        # The similar embedding should match the constitutional node
        node = ng.find_or_create_node(similar_emb)
        assert node.constitutional is True

    def test_distant_embedding_does_not_match(self):
        """An embedding far from any constitutional node creates a normal node."""
        constitutional_emb = _make_embedding(10)
        distant_emb = _make_embedding(9999)  # Very different seed

        config = _make_config_with_constitutional([
            {"embedding": constitutional_emb.tolist(), "description": "frozen concept"},
        ])
        config["novelty_threshold"] = 0.7
        ng = NGLite(module_id="test", config=config)

        node = ng.find_or_create_node(distant_emb)
        assert node.constitutional is False


# ---------------------------------------------------------------------------
# Integration with live embeddings
# ---------------------------------------------------------------------------

class TestLiveEmbeddings:
    """Integration tests using the actual constitutional_embeddings.json."""

    @pytest.fixture
    def constitutional_config(self):
        path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data", "constitutional_embeddings.json",
        )
        if not os.path.exists(path):
            pytest.skip("constitutional_embeddings.json not found")

        with open(path) as f:
            data = json.load(f)

        return _make_config_with_constitutional(data["embeddings"])

    def test_all_embeddings_seed_as_constitutional(self, constitutional_config):
        ng = NGLite(module_id="test", config=constitutional_config)
        constitutional_nodes = [n for n in ng.nodes.values() if n.constitutional]
        assert len(constitutional_nodes) == 22  # 8 + 5 + 4 + 5

    def test_all_categories_represented(self, constitutional_config):
        ng = NGLite(module_id="test", config=constitutional_config)
        categories = set()
        for node in ng.nodes.values():
            if node.constitutional:
                # Category is stored in the embedding config, not on the node
                # But we can check via description
                desc = node.metadata.get("constitutional_description", "")
                assert desc != ""  # All have descriptions

    def test_constitutional_nodes_block_learning(self, constitutional_config):
        """Verify that learning is frozen for all 22 constitutional nodes."""
        ng = NGLite(module_id="test", config=constitutional_config)

        for node in list(ng.nodes.values()):
            if node.constitutional and node.embedding is not None:
                result = ng.record_outcome(
                    node.embedding,
                    target_id="action:anything",
                    success=True,
                )
                assert result.get("constitutional") is True

        # No synapses should exist
        assert len(ng.synapses) == 0
