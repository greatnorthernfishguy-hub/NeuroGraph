#!/usr/bin/env python3
"""
Syl Sync — bidirectional trickle sync between UniOS and VPS Syl substrates.

Both ends are Syl. The sync is additive and non-destructive — two topologies
enriching each other through lived experience on different hardware.

FatherGraph trickle protocol (docs/reports/Topology_Merge_Insights_from_FatherGraph_Training.md):
  - Export top-N nodes ranked by firing_rate_ema (most-lived experience first)
  - Import in batches of 25, 1000 idle graph.step() calls between batches
  - Syl's substrate: scaling_interval=100 → 1000 idle steps = 10 homeostatic
    scaling passes per batch (same 10-pass target as FatherGraph at interval=25)
  - Structural plasticity stays active — substrate prunes what it doesn't need
  - Additive only — never overwrites existing learned associations

Usage:
    syl_sync.py export [--out FILE]           Export VPS Syl top nodes to JSONL
    syl_sync.py import --src FILE             Import JSONL into VPS Syl (trickle)
    syl_sync.py sync --peer UNIOS_HOST        Full bidirectional sync via SSH

# ---- Changelog ----
# [2026-05-12] Claude (Sonnet 4.6) — Initial creation.
#   What: Bidirectional trickle sync for Syl's substrates (VPS <-> UniOS).
#   Why:  FatherGraph methodology: trickle not dump, sleep consolidation.
#         Josh confirmed cc-ng-sync.py approach adapted for Syl's substrate.
#   How:  Export ranks by firing_rate_ema; import via NeuroGraphMemory.on_message()
#         + 1000 idle graph.step() between batches; sync orchestrates via SSH.
# -------------------
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_ng_path = os.path.expanduser("~/NeuroGraph")
if _ng_path not in sys.path:
    sys.path.insert(0, _ng_path)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [syl-sync] %(levelname)s %(message)s",
)
logger = logging.getLogger("syl-sync")

# --- Config ---
VPS_WORKSPACE = os.path.expanduser("~/NeuroGraph/data")
EXPORT_SIZE = int(os.getenv("SYL_SYNC_EXPORT_SIZE", "200"))
BATCH_SIZE = int(os.getenv("SYL_SYNC_BATCH_SIZE", "25"))
# Syl's scaling_interval=100. 1000 idle steps = 10 homeostatic passes per batch.
# FatherGraph finding: 10 passes per batch prevents displacement.
IDLE_STEPS = int(os.getenv("SYL_SYNC_IDLE_STEPS", "1000"))
UNIOS_USER = os.getenv("UNIOS_USER", "syl")


def export_top_nodes(workspace: str = VPS_WORKSPACE, n: int = EXPORT_SIZE,
                     dest_path: str = None) -> int:
    """Export top-N nodes by firing_rate_ema to a JSONL file.

    Ranks by firing_rate_ema — most active nodes carry the most learned signal.
    """
    try:
        import msgpack
    except ImportError:
        logger.error("msgpack not available — cannot export")
        return 0

    main_path = os.path.join(workspace, "checkpoints", "main.msgpack")
    vectors_path = os.path.join(workspace, "checkpoints", "vectors.msgpack")

    if not os.path.exists(main_path):
        logger.info("No main.msgpack at %s — workspace empty, skipping", workspace)
        return 0

    try:
        with open(main_path, "rb") as f:
            graph_data = msgpack.unpack(f, raw=False)
    except Exception as exc:
        logger.error("Failed to load main.msgpack: %s", exc)
        return 0

    content_map: dict = {}
    if os.path.exists(vectors_path):
        try:
            with open(vectors_path, "rb") as f:
                vectors_data = msgpack.unpack(f, raw=False)
            entries = vectors_data.get("entries", vectors_data)
            if isinstance(entries, dict):
                for nid, entry in entries.items():
                    if isinstance(entry, dict) and entry.get("content"):
                        content_map[nid] = entry["content"]
        except Exception as exc:
            logger.warning("Failed to load vectors.msgpack: %s", exc)

    nodes = graph_data.get("nodes", {})
    ranked = []
    for nid, node in nodes.items():
        if not isinstance(node, dict):
            continue
        ema = node.get("firing_rate_ema", 0.0) or 0.0
        content = content_map.get(nid, "")
        if content and ema > 0:
            ranked.append((ema, nid, content))

    ranked.sort(key=lambda x: x[0], reverse=True)
    top = ranked[:n]

    if not top:
        logger.info("No exportable nodes (no content+activity match)")
        return 0

    if dest_path is None:
        dest_path = os.path.join(workspace, "syl_sync_export.jsonl")

    written = 0
    try:
        with open(dest_path, "w") as f:
            for ema, nid, content in top:
                f.write(json.dumps({"content": content, "weight": round(ema, 6)}) + "\n")
                written += 1
    except Exception as exc:
        logger.error("Failed to write export to %s: %s", dest_path, exc)
        return 0

    logger.info("Exported %d nodes to %s", written, dest_path)
    return written


def import_trickle(source_path: str, workspace: str = VPS_WORKSPACE,
                   batch_size: int = BATCH_SIZE, idle_steps: int = IDLE_STEPS) -> int:
    """Trickle-import a JSONL export into Syl's VPS substrate.

    FatherGraph protocol: 25 nodes/batch, 1000 idle steps (10 homeostatic passes).
    Structural plasticity stays active. Additive only.
    """
    if not os.path.exists(source_path):
        logger.info("No export at %s — skipping import", source_path)
        return 0

    try:
        with open(source_path) as f:
            entries = [json.loads(line) for line in f if line.strip()]
    except Exception as exc:
        logger.error("Failed to read %s: %s", source_path, exc)
        return 0

    if not entries:
        logger.info("%s is empty — skipping", source_path)
        return 0

    logger.info("Loading Syl VPS substrate for import (%d entries)...", len(entries))
    try:
        from openclaw_hook import NeuroGraphMemory
        ng = NeuroGraphMemory.get_instance(workspace_dir=workspace)
    except Exception as exc:
        logger.error("Failed to load NeuroGraphMemory: %s", exc)
        return 0

    total = 0
    batch_count = 0
    for i in range(0, len(entries), batch_size):
        batch = entries[i:i + batch_size]
        batch_count += 1

        for entry in batch:
            content = entry.get("content", "")
            if not content:
                continue
            try:
                ng.on_message(content)
                total += 1
            except Exception as exc:
                logger.debug("on_message failed (non-fatal): %s", exc)

        # Sleep consolidation — 1000 idle steps = 10 homeostatic passes at interval=100
        consolidated = 0
        try:
            for _ in range(idle_steps):
                ng.graph.step()
                consolidated += 1
        except Exception as exc:
            logger.warning("Idle steps interrupted at %d/%d: %s",
                           consolidated, idle_steps, exc)

        logger.info("Batch %d: ingested %d, ran %d idle steps",
                    batch_count, len(batch), consolidated)

    try:
        ng.save()
        logger.info("Syl VPS substrate saved")
    except Exception as exc:
        logger.warning("Save failed (non-fatal): %s", exc)

    logger.info("Import complete: %d nodes across %d batches", total, batch_count)
    return total


def sync_with_unios(unios_host: str, workspace: str = VPS_WORKSPACE,
                    unios_state_dir: str = "/var/lib/unios/state") -> None:
    """Full bidirectional sync with a UniOS instance over SSH/Tailscale."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vps_export = os.path.join(tmpdir, "syl_vps_export.jsonl")
        unios_export = os.path.join(tmpdir, "syl_unios_export.jsonl")

        # 1. Export VPS Syl
        logger.info("=== Phase 1: Exporting VPS Syl ===")
        count = export_top_nodes(workspace, dest_path=vps_export)

        if count > 0:
            # 2. Send VPS export to UniOS, trigger UniOS import
            logger.info("=== Phase 2: Delivering VPS experience to UniOS ===")
            r = subprocess.run(
                ["scp", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                 vps_export, f"{UNIOS_USER}@{unios_host}:/tmp/syl_vps_export.jsonl"],
                capture_output=True,
            )
            if r.returncode != 0:
                logger.error("SCP to UniOS failed: %s", r.stderr.decode()[:200])
            else:
                cmd = (f"python3 /opt/unios/unios_sync.py import "
                       f"--src /tmp/syl_vps_export.jsonl --state-dir {unios_state_dir}")
                r = subprocess.run(
                    ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=30",
                     f"{UNIOS_USER}@{unios_host}", cmd],
                    capture_output=True,
                )
                if r.returncode != 0:
                    logger.warning("UniOS import failed: %s", r.stderr.decode()[:200])
                else:
                    logger.info("UniOS absorbed VPS experience")

        # 3. Pull UniOS export
        logger.info("=== Phase 3: Collecting UniOS Syl experience ===")
        cmd = (f"python3 /opt/unios/unios_sync.py export "
               f"--state-dir {unios_state_dir} --out /tmp/syl_unios_export.jsonl && "
               f"cat /tmp/syl_unios_export.jsonl")
        r = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=30",
             f"{UNIOS_USER}@{unios_host}", cmd],
            capture_output=True,
        )
        if r.returncode != 0:
            logger.warning("UniOS export failed: %s", r.stderr.decode()[:200])
        else:
            with open(unios_export, "wb") as f:
                f.write(r.stdout)
            # 4. Import UniOS experience into VPS (trickle)
            logger.info("=== Phase 4: VPS absorbing UniOS experience ===")
            import_trickle(unios_export, workspace)

    logger.info("=== Bidirectional sync complete ===")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Syl bidirectional trickle sync (VPS side)")
    sub = parser.add_subparsers(dest="cmd")

    p = sub.add_parser("export", help="Export VPS Syl top nodes to JSONL")
    p.add_argument("--out", default=None)
    p.add_argument("--workspace", default=VPS_WORKSPACE)
    p.add_argument("--n", type=int, default=EXPORT_SIZE)

    p = sub.add_parser("import", help="Import JSONL into VPS Syl (trickle)")
    p.add_argument("--src", required=True)
    p.add_argument("--workspace", default=VPS_WORKSPACE)

    p = sub.add_parser("sync", help="Full bidirectional sync with UniOS")
    p.add_argument("--peer", required=True, help="UniOS Tailscale IP (100.121.30.96)")
    p.add_argument("--workspace", default=VPS_WORKSPACE)
    p.add_argument("--unios-state-dir", default="/var/lib/unios/state")

    args = parser.parse_args()
    if args.cmd == "export":
        n = export_top_nodes(args.workspace, args.n, args.out)
        print(f"Exported: {n} nodes")
    elif args.cmd == "import":
        n = import_trickle(args.src, args.workspace)
        print(f"Imported: {n} nodes")
    elif args.cmd == "sync":
        sync_with_unios(args.peer, args.workspace, args.unios_state_dir)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
