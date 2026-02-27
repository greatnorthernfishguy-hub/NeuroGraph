"""
Tests for the NeuroGraph checkpoint migration framework.

Covers:
- Version detection
- Migration path planning
- Individual migration steps (0.1.0 → 0.2.0 → 0.2.5 → 0.3.0 → 0.3.5 → 0.4.0 → 0.4.1)
- Full migration from oldest to latest
- Backup and rollback
- Checkpoint info
- Edge cases (already at target, unknown version)
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from neurograph_migrate import (
    CURRENT_VERSION,
    SCHEMA_VERSIONS,
    create_backup,
    get_checkpoint_info,
    get_checkpoint_version,
    list_backups,
    load_checkpoint,
    migrate_data,
    plan_migration,
    rollback_checkpoint,
    save_checkpoint,
    upgrade_checkpoint,
)


# ---------------------------------------------------------------------------
# Fixtures: minimal checkpoint data at each version
# ---------------------------------------------------------------------------

def _make_v010_checkpoint():
    """Minimal v0.1.0 checkpoint (Phase 1)."""
    return {
        "version": "0.1.0",
        "timestep": 100,
        "config": {
            "decay_rate": 0.95,
            "default_threshold": 1.0,
        },
        "nodes": {
            "n1": {
                "node_id": "n1",
                "voltage": 0.5,
                "threshold": 1.0,
                "resting_potential": 0.0,
                "refractory_remaining": 0,
                "refractory_period": 2,
                "last_spike_time": None,
                "spike_history": [50, 60],
                "spike_history_capacity": 100,
                "firing_rate_ema": 0.02,
                "intrinsic_excitability": 1.0,
                "metadata": {},
                "is_inhibitory": False,
            },
            "n2": {
                "node_id": "n2",
                "voltage": 0.3,
                "threshold": 1.0,
                "resting_potential": 0.0,
                "refractory_remaining": 0,
                "refractory_period": 2,
                "last_spike_time": 80,
                "spike_history": [70, 80],
                "spike_history_capacity": 100,
                "firing_rate_ema": 0.03,
                "intrinsic_excitability": 1.0,
                "metadata": {},
                "is_inhibitory": False,
            },
        },
        "synapses": {
            "s1": {
                "synapse_id": "s1",
                "pre_node_id": "n1",
                "post_node_id": "n2",
                "weight": 2.5,
                "max_weight": 5.0,
                "delay": 1,
                "last_update_time": 95.0,
                "creation_time": 10.0,
                "synapse_type": "EXCITATORY",
                "peak_weight": 2.8,
                "low_weight_steps": 0,
                "inactive_steps": 0,
            },
        },
        "hyperedges": {
            "he1": {
                "hyperedge_id": "he1",
                "member_nodes": ["n1", "n2"],
                "member_weights": {"n1": 1.0, "n2": 1.0},
                "activation_threshold": 0.6,
                "activation_mode": "WEIGHTED_THRESHOLD",
                "current_activation": 0.0,
                "output_targets": [],
                "output_weight": 1.0,
                "metadata": {},
                "is_learnable": True,
                "refractory_period": 2,
                "refractory_remaining": 0,
            },
        },
        "telemetry": {
            "total_pruned": 5,
            "total_sprouted": 3,
        },
    }


def _make_v025_checkpoint():
    """Minimal v0.2.5 checkpoint."""
    data = _make_v010_checkpoint()
    data["version"] = "0.2.5"
    # Phase 2 fields on hyperedges
    data["hyperedges"]["he1"]["activation_count"] = 42
    data["hyperedges"]["he1"]["pattern_completion_strength"] = 0.3
    data["hyperedges"]["he1"]["child_hyperedges"] = []
    data["hyperedges"]["he1"]["level"] = 0
    # Phase 2.5 fields
    data["hyperedges"]["he1"]["recent_activation_ema"] = 0.05
    data["hyperedges"]["he1"]["is_archived"] = False
    data["archived_hyperedges"] = {}
    data["he_active_predictions"] = {}
    data["he_prediction_window_fired"] = {}
    data["he_prediction_counter"] = 0
    data["telemetry"]["he_total_predictions"] = 0
    data["telemetry"]["he_total_confirmed"] = 0
    data["telemetry"]["he_total_surprised"] = 0
    return data


@pytest.fixture
def tmp_dir():
    """Temporary directory for checkpoint files."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


# ---------------------------------------------------------------------------
# Test: Version detection
# ---------------------------------------------------------------------------

class TestVersionDetection:
    def test_detect_v010(self, tmp_dir):
        path = str(tmp_dir / "test.json")
        data = _make_v010_checkpoint()
        save_checkpoint(data, path)
        assert get_checkpoint_version(path) == "0.1.0"

    def test_detect_v025(self, tmp_dir):
        path = str(tmp_dir / "test.json")
        data = _make_v025_checkpoint()
        save_checkpoint(data, path)
        assert get_checkpoint_version(path) == "0.2.5"

    def test_detect_no_version_field(self, tmp_dir):
        path = str(tmp_dir / "test.json")
        data = {"timestep": 0, "nodes": {}, "synapses": {}, "hyperedges": {}}
        save_checkpoint(data, path)
        assert get_checkpoint_version(path) == "0.1.0"

    def test_detect_current_version(self, tmp_dir):
        path = str(tmp_dir / "test.json")
        data = {"version": CURRENT_VERSION, "timestep": 0}
        save_checkpoint(data, path)
        assert get_checkpoint_version(path) == CURRENT_VERSION


# ---------------------------------------------------------------------------
# Test: Migration path planning
# ---------------------------------------------------------------------------

class TestMigrationPlanning:
    def test_plan_v010_to_current(self):
        steps = plan_migration("0.1.0", CURRENT_VERSION)
        assert len(steps) == 6
        assert steps[0] == ("0.1.0", "0.2.0")
        assert steps[-1] == ("0.4.0", "0.4.1")

    def test_plan_v025_to_current(self):
        steps = plan_migration("0.2.5", CURRENT_VERSION)
        assert len(steps) == 4
        assert steps[0] == ("0.2.5", "0.3.0")

    def test_plan_already_current(self):
        steps = plan_migration(CURRENT_VERSION, CURRENT_VERSION)
        assert steps == []

    def test_plan_to_intermediate(self):
        steps = plan_migration("0.1.0", "0.2.5")
        assert len(steps) == 2
        assert steps[0] == ("0.1.0", "0.2.0")
        assert steps[1] == ("0.2.0", "0.2.5")


# ---------------------------------------------------------------------------
# Test: Individual migration steps
# ---------------------------------------------------------------------------

class TestMigrationSteps:
    def test_010_to_020_adds_hyperedge_fields(self):
        data = _make_v010_checkpoint()
        migrated, applied = migrate_data(data, "0.2.0")
        assert migrated["version"] == "0.2.0"
        he = migrated["hyperedges"]["he1"]
        assert "activation_count" in he
        assert "pattern_completion_strength" in he
        assert "child_hyperedges" in he
        assert "level" in he
        assert he["activation_count"] == 0
        assert len(applied) == 1

    def test_020_to_025_adds_prediction_infra(self):
        data = _make_v010_checkpoint()
        migrated, _ = migrate_data(data, "0.2.0")
        migrated, applied = migrate_data(migrated, "0.2.5")
        assert migrated["version"] == "0.2.5"
        he = migrated["hyperedges"]["he1"]
        assert "recent_activation_ema" in he
        assert "is_archived" in he
        assert "archived_hyperedges" in migrated
        assert "he_active_predictions" in migrated
        assert len(applied) == 1

    def test_025_to_030_adds_prediction_fields(self):
        data = _make_v025_checkpoint()
        migrated, applied = migrate_data(data, "0.3.0")
        assert migrated["version"] == "0.3.0"
        syn = migrated["synapses"]["s1"]
        assert "eligibility_trace" in syn
        assert "metadata" in syn
        assert syn["eligibility_trace"] == 0.0
        tel = migrated["telemetry"]
        assert "total_predictions_made" in tel
        assert "total_novel_sequences" in tel

    def test_030_to_035_adds_persistence_fields(self):
        data = _make_v025_checkpoint()
        migrated, _ = migrate_data(data, "0.3.0")
        migrated, applied = migrate_data(migrated, "0.3.5")
        assert migrated["version"] == "0.3.5"
        assert "active_predictions" in migrated
        assert "prediction_outcomes" in migrated
        assert "synapse_confirmation_history" in migrated
        assert "novel_sequence_log" in migrated
        assert "reward_history" in migrated

    def test_035_to_040_version_bump_only(self):
        data = _make_v025_checkpoint()
        migrated, _ = migrate_data(data, "0.3.5")
        nodes_before = len(migrated["nodes"])
        migrated, applied = migrate_data(migrated, "0.4.0")
        assert migrated["version"] == "0.4.0"
        assert len(migrated["nodes"]) == nodes_before
        assert len(applied) == 1


# ---------------------------------------------------------------------------
# Test: Full migration from oldest to latest
# ---------------------------------------------------------------------------

class TestFullMigration:
    def test_v010_to_current(self):
        data = _make_v010_checkpoint()
        migrated, applied = migrate_data(data)
        assert migrated["version"] == CURRENT_VERSION
        assert len(applied) == 6
        # All Phase 2+ fields present
        he = migrated["hyperedges"]["he1"]
        assert "activation_count" in he
        assert "recent_activation_ema" in he
        assert "is_archived" in he
        # Phase 3 synapse fields
        syn = migrated["synapses"]["s1"]
        assert "eligibility_trace" in syn
        assert "metadata" in syn
        # Phase 3.5 persistence
        assert "active_predictions" in migrated

    def test_preserves_existing_data(self):
        data = _make_v010_checkpoint()
        migrated, _ = migrate_data(data)
        # Original data preserved
        assert migrated["timestep"] == 100
        assert "n1" in migrated["nodes"]
        assert "n2" in migrated["nodes"]
        assert "s1" in migrated["synapses"]
        assert migrated["synapses"]["s1"]["weight"] == 2.5
        assert migrated["nodes"]["n1"]["voltage"] == 0.5

    def test_original_data_unchanged(self):
        """Ensure migrate_data doesn't modify the original."""
        data = _make_v010_checkpoint()
        original_version = data["version"]
        migrate_data(data)
        assert data["version"] == original_version

    def test_v025_to_current_preserves_he_data(self):
        data = _make_v025_checkpoint()
        migrated, _ = migrate_data(data)
        assert migrated["version"] == CURRENT_VERSION
        he = migrated["hyperedges"]["he1"]
        assert he["activation_count"] == 42
        assert he["recent_activation_ema"] == 0.05


# ---------------------------------------------------------------------------
# Test: File-based upgrade
# ---------------------------------------------------------------------------

class TestFileUpgrade:
    def test_upgrade_json(self, tmp_dir):
        path = str(tmp_dir / "graph.json")
        save_checkpoint(_make_v010_checkpoint(), path)

        result = upgrade_checkpoint(path)
        assert result["version_before"] == "0.1.0"
        assert result["version_after"] == CURRENT_VERSION
        assert len(result["steps_applied"]) == 6
        assert result["backup_path"] is not None

        # Verify file was updated
        upgraded = load_checkpoint(path)
        assert upgraded["version"] == CURRENT_VERSION

    def test_upgrade_msgpack(self, tmp_dir):
        pytest.importorskip("msgpack")
        path = str(tmp_dir / "graph.msgpack")
        save_checkpoint(_make_v010_checkpoint(), path)

        result = upgrade_checkpoint(path)
        assert result["version_after"] == CURRENT_VERSION

        upgraded = load_checkpoint(path)
        assert upgraded["version"] == CURRENT_VERSION

    def test_upgrade_already_current(self, tmp_dir):
        path = str(tmp_dir / "graph.json")
        save_checkpoint({"version": CURRENT_VERSION, "timestep": 0}, path)

        result = upgrade_checkpoint(path)
        assert result["steps_applied"] == []
        assert "no migration needed" in result["message"]

    def test_upgrade_dry_run(self, tmp_dir):
        path = str(tmp_dir / "graph.json")
        save_checkpoint(_make_v010_checkpoint(), path)

        result = upgrade_checkpoint(path, dry_run=True)
        assert result["dry_run"] is True
        assert len(result["steps_applied"]) == 6

        # File should NOT be modified
        data = load_checkpoint(path)
        assert data["version"] == "0.1.0"

    def test_upgrade_to_intermediate(self, tmp_dir):
        path = str(tmp_dir / "graph.json")
        save_checkpoint(_make_v010_checkpoint(), path)

        result = upgrade_checkpoint(path, target_version="0.2.5")
        assert result["version_after"] == "0.2.5"
        assert len(result["steps_applied"]) == 2


# ---------------------------------------------------------------------------
# Test: Backup and rollback
# ---------------------------------------------------------------------------

class TestBackupRollback:
    def test_backup_created(self, tmp_dir):
        path = str(tmp_dir / "graph.json")
        save_checkpoint(_make_v010_checkpoint(), path)

        backup_path = create_backup(path)
        assert os.path.exists(backup_path)
        assert ".backup-" in backup_path

        # Backup should be identical to original
        original = load_checkpoint(path)
        backup = load_checkpoint(backup_path)
        assert original["version"] == backup["version"]

    def test_rollback(self, tmp_dir):
        path = str(tmp_dir / "graph.json")
        save_checkpoint(_make_v010_checkpoint(), path)

        # Upgrade
        result = upgrade_checkpoint(path)
        assert result["backup_path"] is not None
        assert load_checkpoint(path)["version"] == CURRENT_VERSION

        # Rollback
        used = rollback_checkpoint(path, backup_path=result["backup_path"])
        assert load_checkpoint(path)["version"] == "0.1.0"

    def test_rollback_finds_latest_backup(self, tmp_dir):
        path = str(tmp_dir / "graph.json")
        save_checkpoint(_make_v010_checkpoint(), path)

        # Create two backups
        create_backup(path, suffix=".backup-1000")
        create_backup(path, suffix=".backup-2000")

        # Upgrade to latest
        upgrade_checkpoint(path, backup=False)

        # Rollback should find the most recent backup
        used = rollback_checkpoint(path)
        assert ".backup-2000" in used

    def test_rollback_no_backup_raises(self, tmp_dir):
        path = str(tmp_dir / "graph.json")
        save_checkpoint(_make_v010_checkpoint(), path)

        with pytest.raises(FileNotFoundError):
            rollback_checkpoint(path)

    def test_list_backups(self, tmp_dir):
        path = str(tmp_dir / "graph.json")
        save_checkpoint(_make_v010_checkpoint(), path)

        create_backup(path, suffix=".backup-1000")
        create_backup(path, suffix=".backup-2000")

        backups = list_backups(path)
        assert len(backups) == 2
        assert all("path" in b for b in backups)
        assert all("version" in b for b in backups)


# ---------------------------------------------------------------------------
# Test: Checkpoint info
# ---------------------------------------------------------------------------

class TestCheckpointInfo:
    def test_info_v010(self, tmp_dir):
        path = str(tmp_dir / "graph.json")
        save_checkpoint(_make_v010_checkpoint(), path)

        info = get_checkpoint_info(path)
        assert info["version"] == "0.1.0"
        assert info["timestep"] == 100
        assert info["nodes"] == 2
        assert info["synapses"] == 1
        assert info["hyperedges"] == 1
        assert info["format"] == "json"
        assert info["file_size_bytes"] > 0

    def test_info_after_migration(self, tmp_dir):
        path = str(tmp_dir / "graph.json")
        save_checkpoint(_make_v010_checkpoint(), path)
        upgrade_checkpoint(path)

        info = get_checkpoint_info(path)
        assert info["version"] == CURRENT_VERSION
        assert info["nodes"] == 2
        assert info["synapses"] == 1


# ---------------------------------------------------------------------------
# Test: Integration with Graph.restore()
# ---------------------------------------------------------------------------

class TestGraphIntegration:
    def test_migrated_checkpoint_restores_cleanly(self, tmp_dir):
        """A v0.1.0 checkpoint migrated to current loads in Graph.restore()."""
        from neuro_foundation import Graph

        path = str(tmp_dir / "graph.json")
        save_checkpoint(_make_v010_checkpoint(), path)
        upgrade_checkpoint(path)

        g = Graph()
        g.restore(path)
        assert len(g.nodes) == 2
        assert len(g.synapses) == 1
        assert len(g.hyperedges) == 1
        assert g.timestep == 100

    def test_v025_migrated_restores(self, tmp_dir):
        """A v0.2.5 checkpoint migrated to current loads cleanly."""
        from neuro_foundation import Graph

        path = str(tmp_dir / "graph.json")
        save_checkpoint(_make_v025_checkpoint(), path)
        upgrade_checkpoint(path)

        g = Graph()
        g.restore(path)
        assert len(g.nodes) == 2
        assert g.hyperedges["he1"].activation_count == 42

    def test_save_then_migrate_roundtrip(self, tmp_dir):
        """Graph.save() → migrate → Graph.restore() roundtrip."""
        from neuro_foundation import Graph

        # Create a graph and save
        g1 = Graph()
        g1.create_node(node_id="a")
        g1.create_node(node_id="b")
        g1.create_synapse("a", "b", weight=1.5)
        for _ in range(10):
            g1.step()

        path = str(tmp_dir / "graph.json")
        g1.checkpoint(path)

        # The saved checkpoint should already be at current version
        info = get_checkpoint_info(path)
        assert info["version"] == "0.4.1"  # serialization version

        # Upgrade (should be a no-op — already at current version)
        result = upgrade_checkpoint(path)

        # Restore into new graph
        g2 = Graph()
        g2.restore(path)
        assert len(g2.nodes) == 2
        assert len(g2.synapses) == 1
