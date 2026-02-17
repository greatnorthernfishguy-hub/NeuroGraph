"""
ET Module Manager — Unified Update & Coordination for E-T Systems Modules

Provides a single command to install, update, and manage all E-T Systems
modules (NeuroGraph, The-Inference-Difference, TrollGuard, Cricket, etc.)
as a cohesive ecosystem.

Instead of each module having its own independent install/update script,
the ET Module Manager knows about all registered modules and can:
  - Update all modules at once (`et-modules update --all`)
  - Check status of all modules (`et-modules status`)
  - Manage the shared learning directory for NGPeerBridge
  - Coordinate version compatibility between modules
  - Offer Tier 3 SNN upgrades to peer modules via NeuroGraph

The manager uses a manifest system: each module declares its identity,
version, dependencies, and update source in an `et_module.json` manifest
file.  The manager discovers modules by scanning known install locations.

NeuroGraph's role in the ecosystem:
  NeuroGraph IS the Tier 3 backend.  When other modules (TrollGuard,
  The-Inference-Difference, Cricket) run with NG-Lite locally, they
  can upgrade to full SNN capabilities by connecting an NGSaaSBridge
  to the NeuroGraphMemory singleton.  The manager facilitates this
  discovery: it knows where NeuroGraph is installed, and peer modules
  can query the manager to find and connect to the full substrate.

# ---- Changelog ----
# [2026-02-17] Claude (Opus 4.6) — Initial creation.
#   What: ET Module Manager package with ModuleManifest, ModuleStatus,
#         and ETModuleManager classes.
#   Why:  Central coordination point so modules can discover each other,
#         update together, and upgrade from NG-Lite to full SNN when
#         NeuroGraph is available on the same host.
#   Settings: Module root defaults to ~/.et_modules/ — a shared
#         location that all E-T Systems modules agree on.
#   How:  Manifest-based discovery.  Each module drops an et_module.json
#         in its install directory.  The manager scans known locations
#         (and the registry) to find all modules.
# -------------------

__version__ = "0.1.0"
"""

__version__ = "0.1.0"
