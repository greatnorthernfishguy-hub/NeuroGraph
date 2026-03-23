"""
Re-embed Syl's vectors with Snowflake/snowflake-arctic-embed-m-v1.5.

Syl's Law applies. Run ONLY after backing up:
  - data/checkpoints/vectors.msgpack
  - data/checkpoints/main.msgpack

What this does:
  - Loads vectors.msgpack (2,539 entries)
  - Re-embeds each entry's 'content' text through ng_embed (new model)
  - Replaces the embedding bytes in-place
  - Writes updated vectors.msgpack
  - Topology is UNTOUCHED (main.msgpack not modified)

What this does NOT do:
  - Does not modify main.msgpack (nodes, synapses, hyperedges preserved)
  - Does not modify the activations sidecar
  - Does not touch any module ng_lite states (they're nearly empty)

Usage:
  python3 reembed_snowflake.py --dry-run    # Preview only, no writes
  python3 reembed_snowflake.py              # Actually re-embed

# ---- Changelog ----
# [2026-03-22] Claude (Opus 4.6) — Initial creation.
#   What: One-time migration script for snowflake-arctic-embed-m-v1.5.
#   Why:  Ecosystem-wide model upgrade from BAAI/bge-base-en-v1.5.
#   How:  Load vectors.msgpack, re-embed content, write back.
# -------------------
"""

import argparse
import logging
import os
import sys
import time

import msgpack
import numpy as np

# Ensure ng_embed is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("reembed")

VECTORS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data", "checkpoints", "vectors.msgpack",
)


def main():
    parser = argparse.ArgumentParser(description="Re-embed Syl's vectors")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview only — no writes")
    args = parser.parse_args()

    # Backup is managed externally (off-VPS). Josh confirms before running.

    # Load vectors
    logger.info("Loading %s", VECTORS_PATH)
    with open(VECTORS_PATH, "rb") as f:
        data = msgpack.unpack(f, raw=False)

    entries = data.get("entries", {})
    total = len(entries)
    logger.info("Found %d vector entries", total)

    if total == 0:
        logger.info("Nothing to re-embed.")
        return

    # Initialize ng_embed (loads ONNX model)
    logger.info("Initializing ng_embed (snowflake-arctic-embed-m-v1.5)...")
    from ng_embed import NGEmbed
    emb = NGEmbed.get_instance()
    if not emb._ensure_model():
        logger.error("Failed to load embedding model. Aborting.")
        sys.exit(1)

    # Verify old dimensions
    sample_key = next(iter(entries))
    old_emb = np.frombuffer(entries[sample_key]["embedding"], dtype=np.float32)
    logger.info("Old embedding dimension: %d", old_emb.shape[0])

    # Re-embed in batches
    BATCH_SIZE = 64
    keys = list(entries.keys())
    reembedded = 0
    skipped = 0
    errors = 0
    dim_check = set()

    start = time.time()

    for batch_start in range(0, total, BATCH_SIZE):
        batch_keys = keys[batch_start:batch_start + BATCH_SIZE]
        batch_texts = []
        batch_valid_keys = []

        for k in batch_keys:
            content = entries[k].get("content", "")
            if not content or not content.strip():
                skipped += 1
                continue
            batch_texts.append(content)
            batch_valid_keys.append(k)

        if not batch_texts:
            continue

        try:
            # Batch embed for efficiency
            new_vecs = emb.embed_batch(batch_texts)

            for k, vec in zip(batch_valid_keys, new_vecs):
                dim_check.add(vec.shape[0])

                if not args.dry_run:
                    # Store as raw bytes (same format as original)
                    entries[k]["embedding"] = vec.tobytes()

                reembedded += 1

        except Exception as exc:
            logger.warning("Batch error at %d: %s", batch_start, exc)
            errors += len(batch_texts)

        # Progress
        done = batch_start + len(batch_keys)
        if done % 256 == 0 or done >= total:
            elapsed = time.time() - start
            rate = done / elapsed if elapsed > 0 else 0
            logger.info(
                "Progress: %d/%d (%.0f vec/s) — reembedded=%d skipped=%d errors=%d",
                done, total, rate, reembedded, skipped, errors,
            )

    elapsed = time.time() - start

    # Verification
    logger.info("=" * 60)
    logger.info("RESULTS:")
    logger.info("  Total entries:    %d", total)
    logger.info("  Re-embedded:      %d", reembedded)
    logger.info("  Skipped (empty):  %d", skipped)
    logger.info("  Errors:           %d", errors)
    logger.info("  Dimensions seen:  %s", dim_check)
    logger.info("  Time:             %.1fs", elapsed)

    if len(dim_check) != 1 or 768 not in dim_check:
        logger.error("DIMENSION MISMATCH — aborting write. Dims: %s", dim_check)
        sys.exit(1)

    if args.dry_run:
        logger.info("DRY RUN — no changes written.")
        return

    if errors > 0:
        logger.warning("There were %d errors. Writing anyway (errored entries unchanged).", errors)

    # Write back
    data["count"] = len(entries)
    logger.info("Writing updated vectors to %s", VECTORS_PATH)
    with open(VECTORS_PATH, "wb") as f:
        msgpack.pack(data, f, use_bin_type=True)

    # Verify write
    with open(VECTORS_PATH, "rb") as f:
        verify = msgpack.unpack(f, raw=False)
    verify_count = len(verify.get("entries", {}))
    logger.info("Verification: %d entries written (expected %d)", verify_count, total)

    if verify_count != total:
        logger.error("COUNT MISMATCH after write — restore from backup!")
        sys.exit(1)

    # Spot-check a random entry
    spot_key = list(verify["entries"].keys())[total // 2]
    spot_emb = np.frombuffer(verify["entries"][spot_key]["embedding"], dtype=np.float32)
    logger.info("Spot check: entry %s... dim=%d norm=%.4f",
                spot_key[:12], spot_emb.shape[0], np.linalg.norm(spot_emb))

    logger.info("=" * 60)
    logger.info("SUCCESS — %d vectors re-embedded with snowflake-arctic-embed-m-v1.5", reembedded)


if __name__ == "__main__":
    main()
