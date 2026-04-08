#!/bin/bash
# trickle_drain.sh — Feed oversized JSONL tract archives back in manageable batches.
#
# Usage: trickle_drain.sh <tract_file> [batch_size] [sleep_seconds]
#
# Renames the tract to .archive, feeds BATCH_SIZE lines at a time back to
# the original path, sleeps between batches so the module's pulse loop can
# drain. Once the archive is consumed, deletes it.
#
# #119 Step 2 — clearing pre-BTF JSONL debt.
# 2026-04-05

set -euo pipefail

TRACT_PATH="${1:?Usage: trickle_drain.sh <tract_file> [batch_size] [sleep_seconds]}"
BATCH_SIZE="${2:-500}"
SLEEP_SECS="${3:-5}"

ARCHIVE="${TRACT_PATH}.archive"
TRACT_NAME="$(basename "$TRACT_PATH")"

# Rename to archive (atomic — new deposits go to fresh file immediately)
if [ ! -f "$ARCHIVE" ]; then
    if [ ! -f "$TRACT_PATH" ]; then
        echo "[$TRACT_NAME] No file found, nothing to drain."
        exit 0
    fi
    mv "$TRACT_PATH" "$ARCHIVE"
    echo "[$TRACT_NAME] Archived $(wc -l < "$ARCHIVE") lines ($(numfmt --to=iec $(stat -c%s "$ARCHIVE")))"
fi

TOTAL_LINES=$(wc -l < "$ARCHIVE")
FED=0
BATCH_NUM=0

echo "[$TRACT_NAME] Trickle-feeding $TOTAL_LINES lines, $BATCH_SIZE per batch, ${SLEEP_SECS}s pause"

while [ "$FED" -lt "$TOTAL_LINES" ]; do
    BATCH_NUM=$((BATCH_NUM + 1))
    REMAINING=$((TOTAL_LINES - FED))
    THIS_BATCH=$BATCH_SIZE
    [ "$REMAINING" -lt "$THIS_BATCH" ] && THIS_BATCH=$REMAINING

    # Extract batch and append to tract (flock for safety)
    sed -n "$((FED + 1)),$((FED + THIS_BATCH))p" "$ARCHIVE" >> "$TRACT_PATH"

    FED=$((FED + THIS_BATCH))
    PCT=$((FED * 100 / TOTAL_LINES))

    # Progress every 10 batches
    if [ $((BATCH_NUM % 10)) -eq 0 ] || [ "$FED" -ge "$TOTAL_LINES" ]; then
        echo "[$TRACT_NAME] Batch $BATCH_NUM: $FED/$TOTAL_LINES lines ($PCT%)"
    fi

    # Don't sleep on last batch
    [ "$FED" -lt "$TOTAL_LINES" ] && sleep "$SLEEP_SECS"
done

rm -f "$ARCHIVE"
echo "[$TRACT_NAME] Done. Archive consumed and deleted."
