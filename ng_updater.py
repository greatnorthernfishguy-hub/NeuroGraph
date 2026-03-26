"""
ng_updater.py — Automatic module update and vendored file sync.

Vendored into every E-T Systems module. Runs on module startup BEFORE
the main code loads. Pulls the latest code from the module's git remote
and syncs vendored files from the NeuroGraph canonical source.

Zero user interaction. Zero configuration for the common case (all
modules on the same host under ~/). Graceful on failure — if the
network is down, git fails, or NeuroGraph isn't reachable, the module
boots with its current code. Updates are opportunistic, not mandatory.

Usage (in a module's entry point, BEFORE other imports):

    from ng_updater import auto_update
    auto_update()  # pulls repo + syncs vendored files

    # Now import and run the module normally
    from my_hook import get_instance
    ...

Or from CLI:

    python3 ng_updater.py              # Update this module
    python3 ng_updater.py --ecosystem  # Update all registered modules
    python3 ng_updater.py --check      # Check only, don't pull
    python3 ng_updater.py --status     # Show update status

Vendored from NeuroGraph canonical source.
Canonical source: https://github.com/greatnorthernfishguy-hub/NeuroGraph

# ---- Changelog ----
# [2026-03-18] Claude (CC) — Initial creation
# What: Auto-update on startup + ecosystem-wide orchestration.
# Why: Smooth, friction-free update cycle for all tiers. Non-technical
#   users should never think about updates. Modules stay current
#   through normal operation (restart = update check).
# How: git pull on module's own repo, vendored file sync from
#   NeuroGraph canonical (if reachable), graceful failure on all
#   error paths. Ecosystem mode iterates peer registry.
# -------------------
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("ng_updater")

# Vendored files that must be synced from NeuroGraph canonical
VENDORED_FILES = [
    "ng_lite.py",
    "ng_peer_bridge.py",
    "ng_ecosystem.py",
    "ng_autonomic.py",
    "openclaw_adapter.py",
    "ng_updater.py",  # Keep ourselves current too
]

# Standard locations to search for NeuroGraph canonical
_CANONICAL_SEARCH_PATHS = [
    "~/NeuroGraph",
    "/home/josh/NeuroGraph",  # VPS standard
]


def _find_module_root() -> Optional[Path]:
    """Find the root of the current module by looking for et_module.json."""
    # Start from this file's location and search upward
    current = Path(__file__).resolve().parent
    for _ in range(5):  # Max 5 levels up
        if (current / "et_module.json").exists():
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None


def _load_module_manifest(module_root: Path) -> Optional[Dict[str, Any]]:
    """Load et_module.json from the module root."""
    manifest_path = module_root / "et_module.json"
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text())
    except Exception as exc:
        logger.debug("Failed to load et_module.json: %s", exc)
        return None


def _find_canonical_source() -> Optional[Path]:
    """Find the NeuroGraph canonical source directory.

    Search order:
    1. NEUROGRAPH_CANONICAL_PATH env var (explicit override)
    2. Standard locations on the filesystem
    """
    env_path = os.environ.get("NEUROGRAPH_CANONICAL_PATH")
    if env_path:
        p = Path(env_path).expanduser()
        if p.exists() and (p / "ng_lite.py").exists():
            return p

    for search_path in _CANONICAL_SEARCH_PATHS:
        p = Path(search_path).expanduser()
        if p.exists() and (p / "ng_lite.py").exists():
            return p

    return None


def _git_pull(repo_path: Path) -> Tuple[bool, str]:
    """Run git pull in the given repo. Returns (success, message)."""
    try:
        result = subprocess.run(
            ["git", "pull", "--ff-only"],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout.strip()
        if result.returncode == 0:
            if "Already up to date" in output:
                return True, "already current"
            return True, output.split("\n")[0]
        return False, result.stderr.strip()[:200]
    except subprocess.TimeoutExpired:
        return False, "git pull timed out (30s)"
    except FileNotFoundError:
        return False, "git not found"
    except Exception as exc:
        return False, str(exc)[:200]


def _git_has_remote_changes(repo_path: Path) -> Tuple[bool, str]:
    """Check if the remote has changes without pulling.

    Returns (has_changes, message).
    """
    try:
        # Fetch without merging
        fetch_result = subprocess.run(
            ["git", "fetch", "--dry-run"],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            timeout=15,
        )
        # If fetch --dry-run produces output, there are changes
        has_changes = bool(fetch_result.stderr.strip())
        if has_changes:
            return True, "remote has updates"
        return False, "up to date"
    except Exception as exc:
        return False, str(exc)[:200]


def _sync_vendored_files(
    module_root: Path,
    canonical_source: Path,
) -> List[str]:
    """Sync vendored files from NeuroGraph canonical to this module.

    Only copies files that differ. Returns list of updated filenames.
    """
    updated = []

    for filename in VENDORED_FILES:
        canonical_file = canonical_source / filename
        if not canonical_file.exists():
            continue

        # Find ALL copies of this vendored file in the module
        # (root + any subdirectory up to 2 levels deep)
        targets = list(module_root.glob(filename))
        targets.extend(module_root.glob(f"*/{filename}"))
        targets.extend(module_root.glob(f"*/*/{filename}"))
        # Filter out venv, .git, __pycache__
        targets = [
            t for t in targets
            if not any(skip in t.parts for skip in ("venv", ".git", "__pycache__"))
        ]

        if not targets:
            continue

        for target_file in targets:
            try:
                if canonical_file.read_bytes() == target_file.read_bytes():
                    continue  # Already identical

                shutil.copy2(str(canonical_file), str(target_file))
                rel_path = target_file.relative_to(module_root)
                updated.append(str(rel_path))
                logger.info("Synced vendored file: %s", rel_path)
            except Exception as exc:
                logger.debug("Failed to sync %s: %s", filename, exc)

    return updated


def check_and_update(
    module_root: Optional[Path] = None,
    pull: bool = True,
    sync_vendored: bool = True,
) -> Dict[str, Any]:
    """Check for updates and optionally apply them.

    This is the core update function. Called automatically by
    auto_update() on module startup, or manually via CLI.

    Args:
        module_root: Path to the module root (auto-detected if None).
        pull: Whether to git pull (False = check only).
        sync_vendored: Whether to sync vendored files from canonical.

    Returns:
        Dict with update results.
    """
    result: Dict[str, Any] = {
        "timestamp": time.time(),
        "module_id": None,
        "git_pull": None,
        "vendored_synced": [],
        "errors": [],
    }

    # Find module root
    if module_root is None:
        module_root = _find_module_root()
    if module_root is None:
        result["errors"].append("Could not find module root (no et_module.json)")
        return result

    # Load manifest
    manifest = _load_module_manifest(module_root)
    if manifest:
        result["module_id"] = manifest.get("module_id", "unknown")

    # Git pull
    if pull:
        success, msg = _git_pull(module_root)
        result["git_pull"] = {"success": success, "message": msg}
        if not success:
            result["errors"].append(f"git pull failed: {msg}")
    else:
        has_changes, msg = _git_has_remote_changes(module_root)
        result["git_pull"] = {
            "checked": True,
            "has_changes": has_changes,
            "message": msg,
        }

    # Sync vendored files
    if sync_vendored:
        canonical = _find_canonical_source()
        if canonical:
            # Don't sync NeuroGraph to itself
            if canonical.resolve() != module_root.resolve():
                updated = _sync_vendored_files(module_root, canonical)
                result["vendored_synced"] = updated
            else:
                result["vendored_synced"] = []  # We ARE the canonical source
        else:
            result["errors"].append(
                "NeuroGraph canonical not found — vendored files not synced. "
                "Set NEUROGRAPH_CANONICAL_PATH if NeuroGraph is in a non-standard location."
            )

    return result


def auto_update() -> None:
    """One-call auto-update for module startup.

    Call this BEFORE importing the module's main code. Pulls the
    latest code and syncs vendored files. Silent on success, logs
    warnings on failure. Never prevents the module from starting.

    Usage:
        from ng_updater import auto_update
        auto_update()
    """
    try:
        result = check_and_update(pull=True, sync_vendored=True)
        module_id = result.get("module_id", "unknown")

        # Log results
        git = result.get("git_pull", {})
        if git.get("success"):
            msg = git.get("message", "")
            if msg != "already current":
                logger.info("[%s] Updated: %s", module_id, msg)

        synced = result.get("vendored_synced", [])
        if synced:
            logger.info(
                "[%s] Vendored files synced: %s",
                module_id, ", ".join(synced),
            )

        for err in result.get("errors", []):
            logger.debug("[%s] Update note: %s", module_id, err)

    except Exception as exc:
        # Never prevent module startup due to update failure
        logger.debug("Auto-update failed (non-fatal): %s", exc)


def update_ecosystem() -> List[Dict[str, Any]]:
    """Update all registered modules in the ecosystem.

    Intended for Tier 3 orchestration — updates NeuroGraph first
    (canonical source), then iterates through the peer registry
    and updates each registered module.

    Returns:
        List of update results, one per module.
    """
    results = []

    # Step 1: Update NeuroGraph itself (canonical source must be current first)
    canonical = _find_canonical_source()
    if canonical:
        logger.info("Updating NeuroGraph canonical...")
        ng_result = check_and_update(
            module_root=canonical,
            pull=True,
            sync_vendored=False,  # We ARE the canonical source
        )
        ng_result["role"] = "canonical"
        results.append(ng_result)
    else:
        results.append({
            "module_id": "neurograph",
            "role": "canonical",
            "errors": ["NeuroGraph canonical not found"],
        })

    # Step 2: Find all registered modules via peer registry
    registry_path = Path.home() / ".et_modules" / "shared_learning" / "_peer_registry.json"
    modules_to_update: List[Tuple[str, Path]] = []

    if registry_path.exists():
        try:
            registry = json.loads(registry_path.read_text())
            for module_id, info in registry.get("modules", {}).items():
                # Find the module's install path
                module_path = _find_module_path(module_id)
                if module_path:
                    modules_to_update.append((module_id, module_path))
                else:
                    results.append({
                        "module_id": module_id,
                        "errors": [f"Could not find install path for {module_id}"],
                    })
        except Exception as exc:
            results.append({
                "module_id": "_registry",
                "errors": [f"Failed to read peer registry: {exc}"],
            })

    # Also scan for unregistered modules (cloned but never run)
    for scan_dir in [Path.home()]:
        for candidate in scan_dir.iterdir():
            if not candidate.is_dir():
                continue
            manifest_path = candidate / "et_module.json"
            if manifest_path.exists():
                try:
                    manifest = json.loads(manifest_path.read_text())
                    mid = manifest.get("module_id", "")
                    if mid and not any(m[0] == mid for m in modules_to_update):
                        # Skip NeuroGraph (already updated as canonical)
                        if candidate.resolve() != (canonical.resolve() if canonical else None):
                            modules_to_update.append((mid, candidate))
                except Exception:
                    pass

    # Step 3: Update each module
    for module_id, module_path in modules_to_update:
        logger.info("Updating %s at %s...", module_id, module_path)
        mod_result = check_and_update(
            module_root=module_path,
            pull=True,
            sync_vendored=True,
        )
        results.append(mod_result)

    logger.info(
        "Ecosystem update complete: %d modules processed", len(results),
    )
    return results


def _find_module_path(module_id: str) -> Optional[Path]:
    """Find a module's install path by its module_id.

    Searches common locations and et_module.json files.
    """
    # Common naming patterns
    candidates = [
        Path.home() / module_id,
        Path.home() / module_id.replace("_", "-"),
        Path.home() / module_id.replace("_", "-").title(),
    ]

    # Known module name → directory mappings
    known_mappings = {
        "inference_difference": "The-Inference-Difference",
        "healing_collective": "The-Healing-Collective",
        "trollguard": "TrollGuard",
        "elmer": "Elmer",
        "immunis": "Immunis",
        "neurograph": "NeuroGraph",
    }
    if module_id in known_mappings:
        candidates.insert(0, Path.home() / known_mappings[module_id])

    for candidate in candidates:
        if candidate.exists() and (candidate / "et_module.json").exists():
            return candidate

    return None


# ---------------------------------------------------------------------------
# CLI interface
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for manual updates."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="ng_updater",
        description="E-T Systems module updater",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--check", action="store_true",
        help="Check for updates without applying",
    )
    group.add_argument(
        "--ecosystem", action="store_true",
        help="Update all registered modules",
    )
    group.add_argument(
        "--status", action="store_true",
        help="Show current module status",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[ng_updater] %(message)s",
    )

    if args.ecosystem:
        results = update_ecosystem()
        for r in results:
            mid = r.get("module_id", "unknown")
            role = r.get("role", "")
            git = r.get("git_pull", {})
            synced = r.get("vendored_synced", [])
            errors = r.get("errors", [])

            status = "OK" if not errors else "WARN"
            git_msg = git.get("message", "") if git else ""
            sync_msg = f", synced: {', '.join(synced)}" if synced else ""

            label = f"{mid} ({role})" if role else mid
            print(f"  [{status}] {label}: {git_msg}{sync_msg}")
            for err in errors:
                print(f"       ⚠ {err}")

        print(f"\n{len(results)} modules processed")

    elif args.status:
        module_root = _find_module_root()
        if module_root is None:
            print("Not inside an E-T Systems module (no et_module.json found)")
            sys.exit(1)

        manifest = _load_module_manifest(module_root)
        canonical = _find_canonical_source()

        print(f"Module: {manifest.get('module_id', '?') if manifest else '?'}")
        print(f"Version: {manifest.get('version', '?') if manifest else '?'}")
        print(f"Root: {module_root}")
        print(f"Canonical source: {canonical or 'not found'}")

        # Check vendored file status
        if canonical and canonical.resolve() != module_root.resolve():
            print("\nVendored files:")
            for filename in VENDORED_FILES:
                can_file = canonical / filename
                if not can_file.exists():
                    continue
                # Find all copies
                targets = list(module_root.glob(filename))
                targets.extend(module_root.glob(f"*/{filename}"))
                targets.extend(module_root.glob(f"*/*/{filename}"))
                targets = [
                    t for t in targets
                    if not any(s in t.parts for s in ("venv", ".git", "__pycache__"))
                ]
                if not targets:
                    print(f"  {filename}: not vendored in this module")
                else:
                    for t in targets:
                        rel = t.relative_to(module_root)
                        if can_file.read_bytes() == t.read_bytes():
                            print(f"  {rel}: ✓ current")
                        else:
                            print(f"  {rel}: ✗ out of date")

    else:
        # Default: update this module
        result = check_and_update(
            pull=not args.check,
            sync_vendored=not args.check,
        )

        mid = result.get("module_id", "unknown")
        git = result.get("git_pull", {})
        synced = result.get("vendored_synced", [])
        errors = result.get("errors", [])

        if args.check:
            has_changes = git.get("has_changes", False)
            print(f"Module: {mid}")
            print(f"Updates available: {'yes' if has_changes else 'no'}")
        else:
            print(f"Module: {mid}")
            print(f"Git: {git.get('message', 'unknown')}")
            if synced:
                print(f"Vendored files updated: {', '.join(synced)}")
            else:
                print("Vendored files: all current")

        for err in errors:
            print(f"⚠ {err}")


if __name__ == "__main__":
    main()
