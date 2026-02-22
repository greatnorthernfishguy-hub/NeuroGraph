"""
NeuroGraph Checkpoint Migration Framework

Provides versioned upgrade/downgrade of checkpoint files with automatic
backup and rollback capability. Detects checkpoint schema version and
applies migration steps sequentially to reach the target version.

Usage:
    # CLI
    neurograph upgrade --checkpoint /path/to/graph.msgpack
    neurograph upgrade --checkpoint /path/to/graph.msgpack --target 0.4.0

    # Programmatic
    from neurograph_migrate import upgrade_checkpoint, get_checkpoint_version
    version = get_checkpoint_version("/path/to/graph.msgpack")
    upgrade_checkpoint("/path/to/graph.msgpack")

Schema Version History:
    0.1.0  — Phase 1: Core Foundation (nodes, synapses, hyperedges)
    0.2.0  — Phase 2: Hypergraph Engine (activation_count, pattern_completion,
             child_hyperedges, level, member evolution)
    0.2.5  — Phase 2.5: Predictive Infrastructure (recent_activation_ema,
             is_archived, HE-level predictions, surprise events)
    0.3.0  — Phase 3: Predictive Coding (synapse-level predictions,
             eligibility_trace, synapse metadata, three-factor learning)
    0.3.5  — Phase 3.5: Prediction Persistence (active_predictions,
             prediction_outcomes, confirmation_history serialized)
    0.4.0  — Phase 4: Universal Ingestor (no checkpoint schema changes,
             version bump for tracking)
    0.4.1  — Phase 4.1: Consolidation Lifecycle (ConsolidationState enum,
             synapse salience, hyperedge creation_time, consolidation config)

Grok Review Changelog (v0.7.1):
    No code changes.  Grok's suggestions for neurograph_migrate.py evaluated:
    Rejected: 'Incomplete migrations — _migrate_0_3_0_to_0_3_5 adds fields
        but doesnt backfill' — The fields added (active_predictions,
        prediction_outcomes, confirmation_history, etc.) were transient
        before Phase 3.5: they existed only in memory and were NOT persisted
        to checkpoints.  There is no data to backfill from.  Initializing
        them as empty collections is the correct migration: the system
        starts with a clean prediction slate after upgrade, which is the
        same state it would have been in had the graph been saved under
        the old schema.
    Rejected: 'No validation after migration' — Graph._deserialize() already
        performs extensive validation on restore (expired predictions dropped,
        stale node references pruned, etc.).  Adding duplicate validation in
        the migration layer would create a maintenance burden without benefit.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import msgpack
except ImportError:
    msgpack = None


# Ordered list of all schema versions
SCHEMA_VERSIONS = ["0.1.0", "0.2.0", "0.2.5", "0.3.0", "0.3.5", "0.4.0", "0.4.1"]

# Current target version
CURRENT_VERSION = "0.4.1"


def _version_tuple(v: str) -> Tuple[int, ...]:
    """Convert version string to comparable tuple."""
    return tuple(int(x) for x in v.split("."))


def _version_index(v: str) -> int:
    """Return the index of a version in SCHEMA_VERSIONS, or -1 if not found."""
    try:
        return SCHEMA_VERSIONS.index(v)
    except ValueError:
        # If version not in list, find where it would go
        vt = _version_tuple(v)
        for i, sv in enumerate(SCHEMA_VERSIONS):
            if _version_tuple(sv) > vt:
                return i - 1 if i > 0 else 0
        return len(SCHEMA_VERSIONS) - 1


# ======================================================================
# Migration Functions: each takes data dict and returns modified data dict
# ======================================================================

def _migrate_0_1_0_to_0_2_0(data: Dict[str, Any]) -> Dict[str, Any]:
    """Phase 1 → Phase 2: Add hypergraph engine fields."""
    # Add Phase 2 fields to hyperedges
    for hid, hd in data.get("hyperedges", {}).items():
        hd.setdefault("activation_count", 0)
        hd.setdefault("pattern_completion_strength", 0.3)
        hd.setdefault("child_hyperedges", [])
        hd.setdefault("level", 0)

    # Add Phase 2 config defaults
    config = data.get("config", {})
    config.setdefault("he_pattern_completion_strength", 0.3)
    config.setdefault("he_member_weight_lr", 0.05)
    config.setdefault("he_threshold_lr", 0.01)
    config.setdefault("he_discovery_window", 10)
    config.setdefault("he_discovery_min_co_fires", 5)
    config.setdefault("he_discovery_min_nodes", 3)
    config.setdefault("he_consolidation_overlap", 0.8)
    config.setdefault("he_member_evolution_window", 50)
    config.setdefault("he_member_evolution_min_co_fires", 10)
    config.setdefault("he_member_evolution_initial_weight", 0.3)
    data["config"] = config

    data["version"] = "0.2.0"
    return data


def _migrate_0_2_0_to_0_2_5(data: Dict[str, Any]) -> Dict[str, Any]:
    """Phase 2 → Phase 2.5: Add predictive infrastructure fields."""
    # Add Phase 2.5 fields to hyperedges
    for hid, hd in data.get("hyperedges", {}).items():
        hd.setdefault("recent_activation_ema", 0.0)
        hd.setdefault("is_archived", False)

    # Add archived_hyperedges dict
    data.setdefault("archived_hyperedges", {})

    # Add Phase 2.5 config defaults
    config = data.get("config", {})
    config.setdefault("prediction_window", 5)
    config.setdefault("prediction_ema_alpha", 0.01)
    config.setdefault("he_experience_threshold", 100)
    data["config"] = config

    # Add Phase 2.5 telemetry
    tel = data.get("telemetry", {})
    tel.setdefault("he_total_predictions", 0)
    tel.setdefault("he_total_confirmed", 0)
    tel.setdefault("he_total_surprised", 0)
    data["telemetry"] = tel

    # Add HE-level prediction state (empty on migration)
    data.setdefault("he_active_predictions", {})
    data.setdefault("he_prediction_window_fired", {})
    data.setdefault("he_prediction_counter", 0)

    data["version"] = "0.2.5"
    return data


def _migrate_0_2_5_to_0_3_0(data: Dict[str, Any]) -> Dict[str, Any]:
    """Phase 2.5 → Phase 3: Add predictive coding fields."""
    # Add eligibility_trace and metadata to synapses
    for sid, sd in data.get("synapses", {}).items():
        sd.setdefault("eligibility_trace", 0.0)
        sd.setdefault("metadata", {})

    # Add Phase 3 config defaults
    config = data.get("config", {})
    config.setdefault("prediction_threshold", 3.0)
    config.setdefault("prediction_pre_charge_factor", 0.3)
    config.setdefault("prediction_window", 10)
    config.setdefault("prediction_chain_decay", 0.7)
    config.setdefault("prediction_max_chain_depth", 3)
    config.setdefault("prediction_confirm_bonus", 0.01)
    config.setdefault("prediction_error_penalty", 0.02)
    config.setdefault("prediction_max_active", 1000)
    config.setdefault("surprise_sprouting_weight", 0.1)
    config.setdefault("eligibility_trace_tau", 100)
    config.setdefault("three_factor_enabled", False)
    data["config"] = config

    # Add Phase 3 telemetry counters
    tel = data.get("telemetry", {})
    tel.setdefault("total_predictions_made", 0)
    tel.setdefault("total_predictions_confirmed", 0)
    tel.setdefault("total_predictions_errors", 0)
    tel.setdefault("total_novel_sequences", 0)
    tel.setdefault("total_rewards_injected", 0)
    data["telemetry"] = tel

    data["version"] = "0.3.0"
    return data


def _migrate_0_3_0_to_0_3_5(data: Dict[str, Any]) -> Dict[str, Any]:
    """Phase 3 → Phase 3.5: Add prediction persistence fields."""
    # These fields were transient before 0.3.5; initialize as empty
    data.setdefault("active_predictions", {})
    data.setdefault("prediction_outcomes", [])
    data.setdefault("synapse_confirmation_history", {})
    data.setdefault("novel_sequence_log", [])
    data.setdefault("reward_history", [])

    data["version"] = "0.3.5"
    return data


def _migrate_0_3_5_to_0_4_0(data: Dict[str, Any]) -> Dict[str, Any]:
    """Phase 3.5 → Phase 4: Version bump only (no schema changes)."""
    data["version"] = "0.4.0"
    return data


def _migrate_0_4_0_to_0_4_1(data: Dict[str, Any]) -> Dict[str, Any]:
    """Phase 4 → Phase 4.1: Consolidation Lifecycle.

    Adds ConsolidationState fields to hyperedges, salience to synapses,
    and consolidation config keys + telemetry fields.
    """
    # Synapse: add salience field (Amygdala Protocol armor)
    for sid, sd in data.get("synapses", {}).items():
        sd.setdefault("salience", 1.0)

    # Hyperedge: add consolidation_state and creation_time
    for hid, hd in data.get("hyperedges", {}).items():
        hd.setdefault("consolidation_state", "SPECULATIVE")
        hd.setdefault("creation_time", 0)

    # Archived hyperedges too
    for hid, hd in data.get("archived_hyperedges", {}).items():
        hd.setdefault("consolidation_state", "SPECULATIVE")
        hd.setdefault("creation_time", 0)

    # Config: add consolidation lifecycle keys
    config = data.get("config", {})
    config.setdefault("he_speculative_to_candidate_min_count", 10)
    config.setdefault("he_speculative_to_candidate_min_ema", 0.2)
    config.setdefault("he_candidate_to_consolidated_min_count", 100)
    config.setdefault("he_candidate_to_consolidated_min_age", 5000)
    config.setdefault("he_consolidation_eval_interval", 100)
    config.setdefault("he_cull_penalty_factor", 0.25)
    config.setdefault("he_consolidation_adapt_rate", 0.005)
    config.setdefault("he_salience_max", 5.0)
    config.setdefault("he_salience_decay_rate", 0.0002)

    data["version"] = "0.4.1"
    return data


# Migration registry: (from_version, to_version) → migration function
MIGRATIONS: List[Tuple[str, str, Callable]] = [
    ("0.1.0", "0.2.0", _migrate_0_1_0_to_0_2_0),
    ("0.2.0", "0.2.5", _migrate_0_2_0_to_0_2_5),
    ("0.2.5", "0.3.0", _migrate_0_2_5_to_0_3_0),
    ("0.3.0", "0.3.5", _migrate_0_3_0_to_0_3_5),
    ("0.3.5", "0.4.0", _migrate_0_3_5_to_0_4_0),
    ("0.4.0", "0.4.1", _migrate_0_4_0_to_0_4_1),
]


# ======================================================================
# Public API
# ======================================================================

def load_checkpoint(path: str) -> Dict[str, Any]:
    """Load a checkpoint file (msgpack or JSON)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if p.suffix == ".msgpack":
        if msgpack is None:
            raise ImportError("msgpack required for .msgpack files")
        with open(path, "rb") as f:
            return msgpack.unpack(f, raw=False)
    else:
        with open(path, "r") as f:
            return json.load(f)


def save_checkpoint(data: Dict[str, Any], path: str) -> None:
    """Save a checkpoint file (format determined by extension)."""
    p = Path(path)
    if p.suffix == ".msgpack":
        if msgpack is None:
            raise ImportError("msgpack required for .msgpack files")
        with open(path, "wb") as f:
            msgpack.pack(data, f, use_bin_type=True)
    else:
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)


def get_checkpoint_version(path: str) -> str:
    """Detect the schema version of a checkpoint file.

    Returns:
        Version string (e.g. "0.3.5"). Returns "0.1.0" if no version field.
    """
    data = load_checkpoint(path)
    return data.get("version", "0.1.0")


def get_checkpoint_info(path: str) -> Dict[str, Any]:
    """Get detailed information about a checkpoint.

    Returns:
        Dict with version, timestep, node/synapse/hyperedge counts, file size.
    """
    data = load_checkpoint(path)
    p = Path(path)
    return {
        "path": str(p.resolve()),
        "format": "msgpack" if p.suffix == ".msgpack" else "json",
        "file_size_bytes": p.stat().st_size,
        "version": data.get("version", "0.1.0"),
        "timestep": data.get("timestep", 0),
        "nodes": len(data.get("nodes", {})),
        "synapses": len(data.get("synapses", {})),
        "hyperedges": len(data.get("hyperedges", {})),
        "archived_hyperedges": len(data.get("archived_hyperedges", {})),
        "active_predictions": len(data.get("active_predictions", {})),
        "he_active_predictions": len(data.get("he_active_predictions", {})),
    }


def plan_migration(
    from_version: str, target_version: Optional[str] = None
) -> List[Tuple[str, str]]:
    """Determine the migration steps needed.

    Args:
        from_version: Current checkpoint version.
        target_version: Target version (default: CURRENT_VERSION).

    Returns:
        List of (from_ver, to_ver) tuples describing migration path.

    Raises:
        ValueError: If no migration path exists.
    """
    target = target_version or CURRENT_VERSION
    if _version_tuple(from_version) >= _version_tuple(target):
        return []  # Already at or beyond target

    steps = []
    current = from_version
    for mig_from, mig_to, _ in MIGRATIONS:
        if _version_tuple(current) < _version_tuple(mig_from):
            continue
        if current == mig_from:
            steps.append((mig_from, mig_to))
            current = mig_to
            if _version_tuple(current) >= _version_tuple(target):
                break

    if _version_tuple(current) < _version_tuple(target):
        raise ValueError(
            f"No migration path from {from_version} to {target}. "
            f"Reached {current}."
        )

    return steps


def create_backup(path: str, suffix: Optional[str] = None) -> str:
    """Create a backup copy of a checkpoint file.

    Args:
        path: Path to the checkpoint file.
        suffix: Custom suffix (default: timestamp-based).

    Returns:
        Path to the backup file.
    """
    p = Path(path)
    if suffix is None:
        suffix = f".backup-{int(time.time())}"
    backup_path = str(p) + suffix
    shutil.copy2(path, backup_path)
    return backup_path


def migrate_data(
    data: Dict[str, Any],
    target_version: Optional[str] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """Migrate checkpoint data in-memory.

    Args:
        data: Loaded checkpoint dict.
        target_version: Target version (default: CURRENT_VERSION).

    Returns:
        Tuple of (migrated_data, list_of_applied_migration_descriptions).
    """
    target = target_version or CURRENT_VERSION
    current = data.get("version", "0.1.0")
    applied = []

    if _version_tuple(current) >= _version_tuple(target):
        return data, []

    steps = plan_migration(current, target)
    result = copy.deepcopy(data)

    for mig_from, mig_to in steps:
        for reg_from, reg_to, func in MIGRATIONS:
            if reg_from == mig_from and reg_to == mig_to:
                result = func(result)
                desc = f"{mig_from} → {mig_to}"
                applied.append(desc)
                break

    return result, applied


def upgrade_checkpoint(
    path: str,
    target_version: Optional[str] = None,
    backup: bool = True,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Upgrade a checkpoint file to a newer schema version.

    Args:
        path: Path to checkpoint file.
        target_version: Target version (default: latest).
        backup: Whether to create a backup before upgrading.
        dry_run: If True, only report what would change without writing.

    Returns:
        Dict with migration results:
            - version_before: original version
            - version_after: new version
            - steps_applied: list of migration descriptions
            - backup_path: path to backup (if created)
            - dry_run: whether this was a dry run
    """
    data = load_checkpoint(path)
    version_before = data.get("version", "0.1.0")
    target = target_version or CURRENT_VERSION

    # Check if already at target
    if _version_tuple(version_before) >= _version_tuple(target):
        return {
            "version_before": version_before,
            "version_after": version_before,
            "steps_applied": [],
            "backup_path": None,
            "dry_run": dry_run,
            "message": f"Already at version {version_before}, no migration needed",
        }

    # Plan migration path
    steps = plan_migration(version_before, target)
    step_descriptions = [f"{f} → {t}" for f, t in steps]

    if dry_run:
        return {
            "version_before": version_before,
            "version_after": target,
            "steps_applied": step_descriptions,
            "backup_path": None,
            "dry_run": True,
            "message": f"Would migrate {version_before} → {target} in {len(steps)} steps",
        }

    # Create backup
    backup_path = None
    if backup:
        backup_path = create_backup(path)

    # Apply migrations
    migrated, applied = migrate_data(data, target)

    # Save migrated checkpoint
    save_checkpoint(migrated, path)

    return {
        "version_before": version_before,
        "version_after": migrated.get("version", target),
        "steps_applied": applied,
        "backup_path": backup_path,
        "dry_run": False,
        "message": f"Successfully migrated {version_before} → {migrated.get('version', target)}",
    }


def rollback_checkpoint(path: str, backup_path: Optional[str] = None) -> str:
    """Rollback a checkpoint to its backup.

    Args:
        path: Path to the current checkpoint.
        backup_path: Explicit backup path. If None, finds the most recent backup.

    Returns:
        The backup path that was restored.

    Raises:
        FileNotFoundError: If no backup is found.
    """
    if backup_path is None:
        # Find the most recent backup (sort by suffix timestamp, then mtime)
        p = Path(path)
        parent = p.parent
        stem = p.name
        backups = sorted(
            [f for f in parent.iterdir() if f.name.startswith(stem + ".backup-")],
            key=lambda f: f.name,
            reverse=True,
        )
        if not backups:
            raise FileNotFoundError(f"No backup found for {path}")
        backup_path = str(backups[0])

    if not Path(backup_path).exists():
        raise FileNotFoundError(f"Backup not found: {backup_path}")

    shutil.copy2(backup_path, path)
    return backup_path


def list_backups(path: str) -> List[Dict[str, Any]]:
    """List all backup files for a checkpoint.

    Returns:
        List of dicts with path, size, mtime, version for each backup.
    """
    p = Path(path)
    parent = p.parent
    stem = p.name
    backups = []

    for f in sorted(parent.iterdir()):
        if f.name.startswith(stem + ".backup-"):
            try:
                data = load_checkpoint(str(f))
                version = data.get("version", "0.1.0")
            except Exception:
                version = "unknown"
            backups.append({
                "path": str(f),
                "size_bytes": f.stat().st_size,
                "modified": f.stat().st_mtime,
                "version": version,
            })

    return backups
