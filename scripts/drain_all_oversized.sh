#!/bin/bash
# Sequentially trickle-drain all oversized JSONL tracts to BTF.
# One at a time to keep memory pressure flat.
# 2026-04-05

TRACTS_DIR="/home/josh/.et_modules/tracts/neurograph"
DRAIN="/home/josh/NeuroGraph/scripts/trickle_drain.py"

for f in "$TRACTS_DIR"/*.tract; do
    [ -f "$f" ] || continue
    size=$(stat -c%s "$f" 2>/dev/null)
    # Skip small files (<10MB) — already BTF or recently created
    [ "$size" -lt 10000000 ] && continue
    echo "=== Starting: $(basename "$f") ($((size / 1048576))MB) ==="
    python3 "$DRAIN" "$f" 500 5
    echo "=== Finished: $(basename "$f") ==="
    echo ""
done
echo "All oversized tracts drained."
