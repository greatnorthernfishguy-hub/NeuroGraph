#!/usr/bin/env python3
# ---- Changelog ----
# [2026-04-20] CC (BLK-GUI-197) -- Registry fix helper script
# What: Fixes trollguard api_port (0->7438), darwin module_id/display_name (empty->'darwin'/'Darwin'),
#       adds neurograph entry if missing.
# Why:  Registry was seeded before these entries were complete or available.
# How:  Run once on VPS: python3 ~/NeuroGraph/apply_registry_update.py
# -------------------
"""VPS registry update helper. Run once: python3 ~/NeuroGraph/apply_registry_update.py"""
import json
import time
from pathlib import Path

registry_path = Path.home() / ".et_modules" / "registry.json"

if not registry_path.exists():
    print(f"ERROR: {registry_path} not found")
    raise SystemExit(1)

data = json.loads(registry_path.read_text())
mods = data.setdefault("modules", {})

if "trollguard" in mods:
    mods["trollguard"]["api_port"] = 7438
    print("Fixed: trollguard api_port -> 7438")

if "darwin" in mods:
    if not mods["darwin"].get("module_id"):
        mods["darwin"]["module_id"] = "darwin"
        print("Fixed: darwin module_id -> darwin")
    if not mods["darwin"].get("display_name"):
        mods["darwin"]["display_name"] = "Darwin"
        print("Fixed: darwin display_name -> Darwin")
    if not mods["darwin"].get("description"):
        mods["darwin"]["description"] = "Evolution Engine \u2014 evolutionary pressure for the NeuroGraph ecosystem."
        print("Fixed: darwin description")

if "neurograph" not in mods:
    mods["neurograph"] = {
        "module_id": "neurograph",
        "display_name": "NeuroGraph",
        "description": "The cortex, limbic system, and hippocampus \u2014 Syl's identity, memory, and continuity.",
        "version": "0.6.0",
        "install_path": str(Path.home() / "NeuroGraph"),
        "git_remote": "https://github.com/greatnorthernfishguy-hub/NeuroGraph.git",
        "service_name": "",
        "api_port": 8847,
    }
    print("Added: neurograph entry")
else:
    print("Skipped: neurograph entry already present")

data["last_updated"] = time.time()
registry_path.write_text(json.dumps(data, indent=2))
print("REGISTRY_OK")
