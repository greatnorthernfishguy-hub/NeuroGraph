# ET Module Manager Integration — Prompt Primer for Claude Code Instances

## Context

You are working on an **E-T Systems module** that is part of a modular AI
ecosystem.  The other modules in this ecosystem include:

- **NeuroGraph** — Dynamic Spiking Neuro-Hypergraph foundation (the full SNN)
- **The-Inference-Difference** — Intelligent inference routing gateway
- **TrollGuard** — AI Agent security pipeline (4-layer defense with semantic air gap)
- **Cricket**, **Faux_Clawdbot**, and others (planned)

All modules are under: https://github.com/greatnorthernfishguy-hub/

## The Problem You're Solving

Right now each module has its own independent install/update script (`deploy.sh`
for NeuroGraph, `install.sh` for The-Inference-Difference).  With 6+ modules
planned, users should NOT have to update each module separately.  The modules
should **act as one when used together** — like the Apple ecosystem.

## What Has Been Built (in TrollGuard)

TrollGuard (on `main` branch) has implemented the foundational pieces.
Your job is to **integrate your module** with this system.  Here's what exists:

### 1. NGPeerBridge (Tier 2 Cross-Module Learning) — `ng_peer_bridge.py`

**What it does:** Connects co-located NG-Lite instances for shared learning
WITHOUT requiring the full NeuroGraph SNN.  When modules run on the same host,
they pool their pattern knowledge via a shared filesystem event log.

**How it works:**
- Each module writes JSONL learning events to `~/.et_modules/shared_learning/<module_id>.jsonl`
- Periodically, modules read each other's event files
- Cross-module embedding similarity is computed to absorb relevant patterns
- Falls back gracefully if no peers are present (Tier 1 standalone mode continues)

**The three tiers:**
- **Tier 1 (Standalone):** `ng_lite.py` alone — local Hebbian learning, zero deps
- **Tier 2 (Peer):** `NGPeerBridge` — shared learning between co-located modules
- **Tier 3 (SaaS):** `NGSaaSBridge` — full NeuroGraph SNN (already exists in NeuroGraph's `ng_bridge.py`)

**Integration pattern (copy this):**
```python
from ng_lite import NGLite
from ng_peer_bridge import NGPeerBridge

# Create NG-Lite with your module's ID
ng = NGLite(module_id="your_module_id")  # e.g. "inference_difference", "neurograph"

# Load persisted state if available
if Path("ng_lite_state.json").exists():
    ng.load("ng_lite_state.json")

# Connect the peer bridge for cross-module learning
bridge = NGPeerBridge(
    module_id="your_module_id",
    shared_dir="~/.et_modules/shared_learning",  # Same dir for ALL modules
    sync_interval=100,  # Sync with peers every 100 recorded outcomes
)
ng.connect_bridge(bridge)

# Now use ng.record_outcome(), ng.get_recommendations(), ng.detect_novelty()
# exactly as before — the bridge handles cross-module sharing transparently.
# If the bridge is unavailable, local learning continues uninterrupted.
```

**Key file to vendor:** Copy `ng_peer_bridge.py` from TrollGuard into your
module.  It imports from `ng_lite.py` (which your module should already vendor).

### 2. ET Module Manager — `et_modules/manager.py`

**What it does:** Discovers, registers, updates, and coordinates all E-T Systems
modules as a unified ecosystem.  One command updates everything.

**How it works:**
- Each module has an `et_module.json` manifest declaring its identity, version,
  install path, git remote, and dependencies
- The manager maintains a registry at `~/.et_modules/registry.json`
- `ETModuleManager.discover()` scans known locations + the registry
- `ETModuleManager.update_all()` git-pulls and restarts all registered modules
- `ETModuleManager.status()` reports health of all modules

**What your module needs to do:**

#### A. Create `et_module.json` in your project root:
```json
{
  "module_id": "your_module_id",
  "display_name": "Your Module Name",
  "version": "0.1.0",
  "description": "What your module does",
  "install_path": "",
  "git_remote": "https://github.com/greatnorthernfishguy-hub/YourRepo.git",
  "git_branch": "main",
  "entry_point": "main.py",
  "ng_lite_version": "1.0.0",
  "dependencies": [],
  "service_name": "",
  "api_port": 0
}
```

The `install_path` is populated at install time by the installer script.

#### B. Register with ET Module Manager during installation

In your `install.sh` (or `deploy.sh`), add this after deploying files:

```bash
ET_MODULES_DIR="${ET_MODULES_DIR:-$HOME/.et_modules}"
mkdir -p "$ET_MODULES_DIR/shared_learning"

python3 -c "
import json, time
registry_path = '$ET_MODULES_DIR/registry.json'
try:
    with open(registry_path, 'r') as f:
        registry = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    registry = {'modules': {}}

registry['modules']['your_module_id'] = {
    'module_id': 'your_module_id',
    'display_name': 'Your Module Name',
    'version': '0.1.0',
    'install_path': '$INSTALL_DIR',
    'registered_at': time.time(),
}
registry['last_updated'] = time.time()

with open(registry_path, 'w') as f:
    json.dump(registry, f, indent=2)
"
```

#### C. Vendor `ng_peer_bridge.py` alongside your existing `ng_lite.py`

Copy `ng_peer_bridge.py` from TrollGuard into your project root (same level
as your vendored `ng_lite.py`).

#### D. Connect the peer bridge in your module's initialization

Wherever your module currently creates an `NGLite` instance, add the peer
bridge connection.  For The-Inference-Difference, this would be in the app
startup.  For NeuroGraph, this would be in `openclaw_hook.py`.

### 3. Known Install Locations

The ET Module Manager scans these locations by default:

```python
KNOWN_LOCATIONS = [
    "~/NeuroGraph",                       # NeuroGraph (primary)
    "~/.openclaw/skills/neurograph",      # NeuroGraph (legacy)
    "~/The-Inference-Difference",         # The-Inference-Difference (primary)
    "/opt/inference-difference",           # The-Inference-Difference (legacy)
    "~/TrollGuard",                       # TrollGuard (primary)
    "/opt/trollguard",                     # TrollGuard (legacy)
    "~/.et_modules/modules",              # Generic module install dir
]
```

If your module installs somewhere else, make sure you register via the
registry mechanism (step B above) so the manager can find it.

### 4. Shared Learning Directory Structure

```
~/.et_modules/
├── registry.json                    # All registered modules
└── shared_learning/                 # NGPeerBridge event logs
    ├── trollguard.jsonl             # TrollGuard's learning events
    ├── inference_difference.jsonl   # TID's learning events
    ├── neurograph.jsonl             # NeuroGraph's learning events
    └── _peer_registry.json          # Active peer discovery
```

## Specific Instructions Per Module

### For The-Inference-Difference:

Your module already vendors `ng_lite.py` and uses it in the routing engine.
You need to:

1. Add `et_module.json` to the project root
2. Vendor `ng_peer_bridge.py` from TrollGuard
3. In your app initialization (wherever `NGLite(module_id="inference_difference")`
   is created), connect the `NGPeerBridge`
4. Update `install.sh` to:
   - Create `~/.et_modules/shared_learning/` directory
   - Register in `~/.et_modules/registry.json`
   - Copy `et_module.json` to install dir and set `install_path`
5. The routing engine's `report_outcome()` already calls `ng.record_outcome()`
   — no changes needed there.  The peer bridge intercepts transparently.

### For NeuroGraph:

NeuroGraph is the full SNN (Tier 3).  It's special because:
- It already HAS the `NGSaaSBridge` in `ng_bridge.py`
- When NeuroGraph is present, other modules should upgrade from Tier 2 (peer)
  to Tier 3 (full SNN) transparently

You need to:

1. Add `et_module.json` to the project root
2. Update `deploy.sh` to register with ET Module Manager
3. In `openclaw_hook.py`, have `NeuroGraphMemory` also write events to the
   shared learning directory so Tier 2 modules can benefit from the full
   graph's intelligence even if they haven't upgraded to Tier 3 yet
4. Consider adding a "bridge upgrade" mechanism: when NeuroGraph detects
   peer modules in `~/.et_modules/shared_learning/`, it can offer to
   upgrade their connections from Tier 2 (peer) to Tier 3 (full SNN)

## Design Principles

1. **Graceful degradation is mandatory.** If a peer is missing, the module
   works standalone.  If NeuroGraph is missing, peer learning continues.
   If everything is missing, local NG-Lite learning works fine.

2. **Living changelogs in the code.**  Every file should have an inline
   changelog block noting who, what, when, where, why, and the reasoning
   behind any settings chosen.  Format:
   ```python
   # ---- Changelog ----
   # [YYYY-MM-DD] Author — Short title.
   #   What: What was changed.
   #   Why:  Why it was changed.
   #   Settings: Any defaults chosen and why.
   #   How:  Implementation approach.
   # -------------------
   ```

3. **The module doesn't know or care which tier it's on.**  The `NGBridge`
   ABC is the universal interface.  Whether a module is talking to a peer
   bridge or the full NeuroGraph SNN, the API is identical.

4. **Shared directory is the coordination point.**  `~/.et_modules/` is the
   agreed-upon root.  All modules write there.  The ET Module Manager lives
   there.  The peer bridge events live there.

5. **Manifests are the discovery mechanism.**  `et_module.json` is the
   contract.  If a module has one, the manager can find it, update it,
   and coordinate it.

## Reference Implementation

The complete reference implementation is in TrollGuard on the `main` branch:
https://github.com/greatnorthernfishguy-hub/TrollGuard

Key files to study (all in the repo root or noted subdirectory):

- `ng_peer_bridge.py` — The Tier 2 peer bridge (vendor this into your module)
- `et_modules/manager.py` — The module manager
- `et_module.json` — The manifest format
- `install.sh` — Registration during installation
- `main.py` lines 59-82 — How `_init_ng_lite()` connects the peer bridge
- `config.yaml` lines 46-52 — NG-Lite + peer bridge configuration
- `api.py` — FastAPI REST API for runtime scanning
- `trollguard_hook.py` — OpenClaw integration hook (singleton pattern)
