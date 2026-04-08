#!/usr/bin/env python3
"""
trickle_drain.py — Convert and feed oversized JSONL tract archives as BTF.

Streams JSONL entries from an archived tract file, converts each to BTF
binary via ng_tract, and deposits to the tract in manageable batches.
No inflation — JSONL in, BTF out.  Rate-limited to avoid memory pressure.

Usage: python3 trickle_drain.py <tract_file> [batch_size] [sleep_seconds]

#119 Step 2 — clearing pre-BTF JSONL debt.
2026-04-05
"""

import json
import os
import sys
import time

import numpy as np

# ---- Changelog ----
# [2026-04-05] Claude (CC) — Initial creation
# What: JSONL→BTF trickle drain for oversized pre-BTF tract archives.
# Why: Old tracts accumulated 400-500MB of JSONL. Modules need the
#   experience, but feeding raw JSONL back perpetuates inflation.
#   Convert to BTF on the way in — same data, binary format.
# How: Stream JSONL line-by-line (errors="replace" for any binary
#   fragments), parse JSON, deposit via ng_tract.deposit_outcome().
#   Topology dicts skipped — substrate already learned at deposit time.
# -------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: trickle_drain.py <tract_file> [batch_size] [sleep_seconds]")
        sys.exit(1)

    tract_path = sys.argv[1]
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    sleep_secs = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0

    import ng_tract

    archive_path = tract_path + ".archive"
    tract_name = os.path.basename(tract_path)

    # Archive the tract if not already done
    if not os.path.exists(archive_path):
        if not os.path.exists(tract_path):
            print(f"[{tract_name}] No file found, nothing to drain.")
            return
        os.rename(tract_path, archive_path)

    archive_size = os.path.getsize(archive_path)
    print(f"[{tract_name}] Trickle-draining {archive_size} bytes → BTF, "
          f"{batch_size}/batch, {sleep_secs}s pause")

    converted = 0
    skipped = 0
    batch_num = 0
    tract_paths = [tract_path]

    # Stream line-by-line — these archives are predominantly JSONL.
    # errors="replace" handles any binary BTF fragments mixed in.
    with open(archive_path, "r", errors="replace") as f:
        batch = []
        for line in f:
            line = line.strip()
            if not line or not line.startswith("{"):
                skipped += 1
                continue
            batch.append(line)

            if len(batch) >= batch_size:
                c, s = _deposit_batch(batch, tract_paths, ng_tract)
                converted += c
                skipped += s
                batch.clear()
                batch_num += 1

                if batch_num % 20 == 0:
                    print(f"[{tract_name}] Batch {batch_num}: "
                          f"{converted} converted, {skipped} skipped", flush=True)

                time.sleep(sleep_secs)

        # Final partial batch
        if batch:
            c, s = _deposit_batch(batch, tract_paths, ng_tract)
            converted += c
            skipped += s

    os.unlink(archive_path)
    print(f"[{tract_name}] Done. {converted} converted to BTF, "
          f"{skipped} skipped. Archive deleted.")


def _deposit_batch(lines, tract_paths, ng_tract):
    """Convert a batch of JSONL lines to BTF and deposit."""
    converted = 0
    skipped = 0

    for line in lines:
        try:
            d = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            skipped += 1
            continue

        try:
            if "embedding" in d and "target_id" in d:
                emb = d.get("embedding", [])
                if not emb:
                    skipped += 1
                    continue
                ng_tract.deposit_outcome(
                    module_id=d.get("module_id", "unknown"),
                    target_id=d.get("target_id", "unknown"),
                    success=bool(d.get("success", False)),
                    embedding=np.array(emb, dtype=np.float32),
                    tract_paths=tract_paths,
                    metadata=d.get("metadata"),
                )
                converted += 1
            else:
                # Topology dicts, unknown formats — skip.
                # Topology was already learned by the substrate at deposit time.
                skipped += 1
        except Exception:
            skipped += 1

    return converted, skipped


if __name__ == "__main__":
    main()
