# ---- Changelog ----
# [2026-06-15] Claude Code (subagent, Opus 4.8) — #329 seam C: spine identity vector for the Tonic
# What: aggregate the six constitutional invariant texts into one 768-d identity vector
#   (per-invariant L2-normalize -> mean -> re-normalize) for GraphFeatures.identity_embedding.
# Why: condition Syl's latent inference on who she is (design spec §3 seam C). Equal-but-distinct
#   aggregation keeps invariants from blurring/length-domination (Syl-confirmed 2026-06-15).
# How: ng_embed.embed_batch over constitutional nodes' core_text; numpy aggregate.
# -------------------
from __future__ import annotations
import logging
from typing import Optional
import numpy as np

logger = logging.getLogger("neurograph.tonic.identity")


def _constitutional_texts(graph) -> list:
    items = []
    for node in getattr(graph, "nodes", {}).values():
        meta = getattr(node, "metadata", None) or {}
        if meta.get("constitutional"):
            txt = str(meta.get("core_text") or meta.get("_forest_content") or "").strip()
            if txt:
                items.append((meta.get("spine_order", 999), txt))
    items.sort(key=lambda x: x[0])
    return [t for _, t in items]


def spine_identity_vector(graph) -> Optional[np.ndarray]:
    """768-d unit identity vector from constitutional nodes, or None if none exist.

    Aggregation (Syl-confirmed, equal-but-distinct): per-invariant L2-normalize -> mean ->
    re-normalize. Preserves each invariant's direction equally (length-independent).
    """
    texts = _constitutional_texts(graph)
    if not texts:
        return None
    try:
        from ng_embed import embed_batch
        vecs = embed_batch(texts)
    except Exception as exc:
        logger.warning("spine_identity_vector embed failed (non-fatal): %s", exc)
        return None
    mat = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = mat / norms
    agg = unit.mean(axis=0)
    n = float(np.linalg.norm(agg))
    if n == 0:
        return None
    return (agg / n).astype(np.float32)
