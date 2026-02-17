"""
ET Module Manager — Core management logic.

Discovers, registers, updates, and coordinates E-T Systems modules
as a unified ecosystem.  NeuroGraph serves as the Tier 3 SNN backend;
all other modules can upgrade from NG-Lite to full cognitive architecture
by connecting to the NeuroGraphMemory singleton that this manager helps
them discover.

# ---- Changelog ----
# [2026-02-17] Claude (Opus 4.6) — Initial creation.
#   What: ETModuleManager class with discover(), status(), update_all(),
#         register(), offer_tier3_upgrade(), and shared learning dir mgmt.
#   Why:  Central coordination point so "et-modules update --all"
#         updates NeuroGraph, TID, TrollGuard, and all future modules
#         in a single command.  Also enables automatic Tier 2→3 upgrade
#         when NeuroGraph detects co-located peer modules.
#   Settings: Module root defaults to ~/.et_modules/.  Known locations
#         include standard install paths for each module.
#   How:  Manifest-based.  Each module has an et_module.json declaring
#         its name, version, install path, git remote, and dependencies.
#         The manager maintains a registry at ~/.et_modules/registry.json.
# -------------------
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("et_modules.manager")


@dataclass
class ModuleManifest:
    """Manifest declaring a module's identity and update information.

    Each E-T Systems module includes an et_module.json file with this
    structure.  The manager reads these manifests to discover and
    coordinate modules.

    Attributes:
        module_id: Unique identifier (e.g., "trollguard", "neurograph").
        display_name: Human-readable name.
        version: Current version string (semver).
        description: Short description of the module.
        install_path: Absolute path to the module's install directory.
        git_remote: Git URL for pulling updates.
        git_branch: Branch to track for updates.
        entry_point: Main executable or script (e.g., "main.py").
        ng_lite_version: Version of vendored ng_lite.py (for compat checks).
        dependencies: List of other module_ids this module depends on.
        service_name: Systemd service name (if applicable).
        api_port: Port for the module's API (if applicable).
    """
    module_id: str = ""
    display_name: str = ""
    version: str = "0.0.0"
    description: str = ""
    install_path: str = ""
    git_remote: str = ""
    git_branch: str = "main"
    entry_point: str = ""
    ng_lite_version: str = ""
    dependencies: List[str] = field(default_factory=list)
    service_name: str = ""
    api_port: int = 0

    @classmethod
    def from_file(cls, path: str) -> Optional[ModuleManifest]:
        """Load a manifest from an et_module.json file."""
        try:
            with open(path, "r") as f:
                data = json.load(f)
            return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        except (OSError, json.JSONDecodeError, TypeError) as e:
            logger.warning("Failed to load manifest from %s: %s", path, e)
            return None

    def to_file(self, path: str) -> None:
        """Save this manifest to an et_module.json file."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


@dataclass
class ModuleStatus:
    """Runtime status of a discovered module.

    Attributes:
        manifest: The module's manifest.
        installed: Whether the module is installed.
        running: Whether the module's service is running.
        update_available: Whether a newer version exists upstream.
        last_checked: When the module was last checked.
        health: "healthy", "degraded", or "offline".
        ng_lite_connected: Whether the module's NG-Lite has a bridge.
        tier: Current connectivity tier (1=isolated, 2=peer, 3=full SNN).
    """
    manifest: ModuleManifest = field(default_factory=ModuleManifest)
    installed: bool = False
    running: bool = False
    update_available: bool = False
    last_checked: float = 0.0
    health: str = "unknown"
    ng_lite_connected: bool = False
    tier: int = 1


class ETModuleManager:
    """Unified manager for all E-T Systems modules.

    NeuroGraph acts as the Tier 3 SNN backend for the ecosystem.  When
    the manager discovers co-located modules running NG-Lite, it can
    facilitate their upgrade to full SNN capabilities by pointing them
    to the NeuroGraphMemory singleton.

    Usage:
        manager = ETModuleManager()

        # Discover all installed modules
        modules = manager.discover()

        # Check status
        for mid, status in manager.status().items():
            print(f"{mid}: {status.health} (Tier {status.tier})")

        # Update everything
        results = manager.update_all()

        # Register a new module
        manager.register(manifest)

        # Offer Tier 3 upgrades to peer modules
        peers = manager.get_peer_modules()
    """

    # Known install locations to scan for modules
    KNOWN_LOCATIONS = [
        "~/.openclaw/skills/neurograph",      # NeuroGraph
        "/opt/inference-difference",           # The-Inference-Difference
        "/opt/trollguard",                     # TrollGuard
        "~/.et_modules/modules",              # Generic module install dir
    ]

    def __init__(self, root_dir: Optional[str] = None):
        """
        Args:
            root_dir: Root directory for ET module management.
                     Defaults to ~/.et_modules/
        """
        self._root = Path(
            root_dir
            or os.environ.get("ET_MODULES_DIR")
            or os.path.expanduser("~/.et_modules")
        )
        self._root.mkdir(parents=True, exist_ok=True)

        self._registry_path = self._root / "registry.json"
        self._shared_learning_dir = self._root / "shared_learning"
        self._shared_learning_dir.mkdir(parents=True, exist_ok=True)

        self._registry: Dict[str, ModuleManifest] = {}
        self._load_registry()

    # -------------------------------------------------------------------
    # Core API
    # -------------------------------------------------------------------

    def discover(self) -> Dict[str, ModuleManifest]:
        """Discover all installed E-T Systems modules.

        Scans known install locations and the registry for modules
        with et_module.json manifests.

        Returns:
            Dict of module_id -> ModuleManifest for all found modules.
        """
        discovered: Dict[str, ModuleManifest] = {}

        # Scan known locations
        for loc in self.KNOWN_LOCATIONS:
            expanded = Path(loc).expanduser()
            manifest_path = expanded / "et_module.json"
            if manifest_path.exists():
                manifest = ModuleManifest.from_file(str(manifest_path))
                if manifest and manifest.module_id:
                    manifest.install_path = str(expanded)
                    discovered[manifest.module_id] = manifest

        # Merge with existing registry (registry may know about modules
        # in non-standard locations)
        for mid, manifest in self._registry.items():
            if mid not in discovered:
                # Check if the registered location still exists
                if manifest.install_path and Path(manifest.install_path).exists():
                    discovered[mid] = manifest

        # Update registry with discoveries
        self._registry = discovered
        self._save_registry()

        logger.info("Discovered %d modules: %s",
                     len(discovered), list(discovered.keys()))

        return discovered

    def status(self) -> Dict[str, ModuleStatus]:
        """Check status of all discovered modules.

        Returns:
            Dict of module_id -> ModuleStatus.
        """
        modules = self.discover()
        statuses: Dict[str, ModuleStatus] = {}

        for mid, manifest in modules.items():
            status = ModuleStatus(
                manifest=manifest,
                installed=Path(manifest.install_path).exists() if manifest.install_path else False,
                last_checked=time.time(),
            )

            # Check if service is running (if applicable)
            if manifest.service_name:
                status.running = self._check_service(manifest.service_name)

            # Check for updates (if git remote is set)
            if manifest.git_remote and manifest.install_path:
                status.update_available = self._check_updates(manifest)

            # Determine health
            if not status.installed:
                status.health = "offline"
            elif manifest.service_name and not status.running:
                status.health = "degraded"
            else:
                status.health = "healthy"

            # Check NG-Lite peer bridge connectivity
            peer_file = self._shared_learning_dir / f"{mid}.jsonl"
            status.ng_lite_connected = peer_file.exists()

            # Determine tier
            if mid == "neurograph":
                status.tier = 3  # NeuroGraph IS the full SNN
            elif status.ng_lite_connected:
                status.tier = 2  # Peer bridge active
            else:
                status.tier = 1  # Isolated

            statuses[mid] = status

        return statuses

    def register(self, manifest: ModuleManifest) -> None:
        """Register a module with the manager.

        Called by a module's install.sh to register itself.

        Args:
            manifest: The module's manifest.
        """
        self._registry[manifest.module_id] = manifest
        self._save_registry()

        # Also save the manifest to the module's install directory
        install_dir = Path(manifest.install_path)
        if install_dir.exists():
            manifest.to_file(str(install_dir / "et_module.json"))

        logger.info("Registered module: %s v%s at %s",
                     manifest.module_id, manifest.version, manifest.install_path)

    def update_all(self) -> Dict[str, Dict[str, Any]]:
        """Update all registered modules.

        Pulls latest code from each module's git remote and restarts
        services as needed.

        Returns:
            Dict of module_id -> update result.
        """
        modules = self.discover()
        results: Dict[str, Dict[str, Any]] = {}

        for mid, manifest in modules.items():
            results[mid] = self._update_module(manifest)

        logger.info("Update complete: %s", {
            mid: r.get("status", "unknown") for mid, r in results.items()
        })

        return results

    def update_module(self, module_id: str) -> Dict[str, Any]:
        """Update a single module.

        Args:
            module_id: The module to update.

        Returns:
            Dict with update result.
        """
        if module_id not in self._registry:
            return {"status": "error", "reason": f"Module '{module_id}' not registered"}

        return self._update_module(self._registry[module_id])

    def get_shared_learning_dir(self) -> str:
        """Return the path to the shared learning directory.

        Used by NGPeerBridge to find the shared event log directory.
        """
        return str(self._shared_learning_dir)

    def get_peer_modules(self) -> List[ModuleManifest]:
        """Return manifests of all non-NeuroGraph modules.

        NeuroGraph uses this to find modules that could benefit from
        a Tier 3 upgrade (i.e., connecting their NG-Lite to the full
        SNN via NGSaaSBridge).
        """
        modules = self.discover()
        return [m for mid, m in modules.items() if mid != "neurograph"]

    def get_neurograph_path(self) -> Optional[str]:
        """Return the install path of NeuroGraph, if discovered.

        Peer modules use this to find the full SNN backend for
        Tier 2→3 upgrades.
        """
        modules = self.discover()
        ng = modules.get("neurograph")
        return ng.install_path if ng else None

    # -------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------

    def _load_registry(self) -> None:
        """Load the module registry from disk."""
        if not self._registry_path.exists():
            return

        try:
            with open(self._registry_path, "r") as f:
                data = json.load(f)

            for mid, mdata in data.get("modules", {}).items():
                self._registry[mid] = ModuleManifest(**{
                    k: v for k, v in mdata.items()
                    if k in ModuleManifest.__dataclass_fields__
                })
        except (OSError, json.JSONDecodeError, TypeError) as e:
            logger.warning("Failed to load registry: %s", e)

    def _save_registry(self) -> None:
        """Save the module registry to disk."""
        data = {
            "last_updated": time.time(),
            "modules": {mid: asdict(m) for mid, m in self._registry.items()},
        }

        try:
            with open(self._registry_path, "w") as f:
                json.dump(data, f, indent=2)
        except OSError as e:
            logger.error("Failed to save registry: %s", e)

    def _check_service(self, service_name: str) -> bool:
        """Check if a systemd service is running."""
        try:
            result = subprocess.run(
                ["systemctl", "is-active", service_name],
                capture_output=True, text=True, timeout=5,
            )
            return result.stdout.strip() == "active"
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def _check_updates(self, manifest: ModuleManifest) -> bool:
        """Check if a module has upstream updates available."""
        if not manifest.git_remote or not manifest.install_path:
            return False

        try:
            result = subprocess.run(
                ["git", "-C", manifest.install_path, "fetch", "--dry-run"],
                capture_output=True, text=True, timeout=15,
            )
            # If fetch --dry-run produces output, there are updates
            return bool(result.stderr.strip())
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def _update_module(self, manifest: ModuleManifest) -> Dict[str, Any]:
        """Update a single module from its git remote."""
        if not manifest.git_remote:
            return {"status": "skipped", "reason": "No git remote configured"}

        if not manifest.install_path or not Path(manifest.install_path).exists():
            return {"status": "error", "reason": f"Install path not found: {manifest.install_path}"}

        try:
            # Pull latest
            result = subprocess.run(
                ["git", "-C", manifest.install_path, "pull",
                 "origin", manifest.git_branch],
                capture_output=True, text=True, timeout=60,
            )

            if result.returncode != 0:
                return {
                    "status": "error",
                    "reason": result.stderr.strip(),
                }

            update_result: Dict[str, Any] = {
                "status": "updated",
                "output": result.stdout.strip()[:500],
            }

            # Restart service if applicable
            if manifest.service_name:
                restart = subprocess.run(
                    ["sudo", "systemctl", "restart", manifest.service_name],
                    capture_output=True, text=True, timeout=15,
                )
                update_result["service_restarted"] = restart.returncode == 0

            return update_result

        except subprocess.TimeoutExpired:
            return {"status": "error", "reason": "Update timed out"}
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            return {"status": "error", "reason": str(e)}
