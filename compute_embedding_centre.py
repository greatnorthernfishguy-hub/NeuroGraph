"""compute_embedding_centre.py — compute (or verify) the embedding-space centre that ng_embed.py ships.

# ---- Changelog ----
# [2026-10-14] Claude Code (Opus 5.5, lane mean-centre) — new file (NG-only tool, not vendored)
# What: Computes the fixed mean-centring vector for the canonical embedding model (ng_embed.MODEL_NAME, 768-d) from a
#   vector store (a COPY of a SimpleVectorDB vectors.msgpack): the mean of the L2-normalised stored embeddings,
#   accumulated in float64 in sorted-key order, cast to float32. Prints the version line, sha256 and the base64 that
#   ng_embed.py embeds; `--verify` recomputes and compares with ng_embed.embedding_centre() (bit-exact).
# Why: Josh 2026-10-14 ("Mean center", punch list #1107): Snowflake vectors share a common direction (unrelated texts
#   sit at ~0.69 cosine), so NG-Lite's 0.30 similarity snap matched nearly everything. The centre is computed ONCE
#   and shipped as a versioned constant — never recomputed at runtime, because a drifting centre would silently
#   change every similarity in every module.
# How: msgpack store -> {"entries": {id: {"embedding": float32 bytes, ...}}}; refuses wrong-dim entries.
# -------------------

usage:
    python compute_embedding_centre.py VECTORS.msgpack            # print sha256 / stats / base64
    python compute_embedding_centre.py VECTORS.msgpack --verify   # recompute and compare with ng_embed's constant
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import sys

import msgpack
import numpy as np


def compute(path: str, dim: int) -> tuple:
    with open(path, "rb") as f:
        entries = msgpack.unpackb(f.read(), raw=False, strict_map_key=False)["entries"]
    acc = np.zeros(dim, dtype=np.float64)
    n = 0
    for k in sorted(entries):
        v = np.frombuffer(entries[k]["embedding"], dtype=np.float32)
        if v.shape[0] != dim:
            raise SystemExit(f"entry has {v.shape[0]} dims, expected {dim}: refusing a mixed store")
        nv = float(np.linalg.norm(v))
        if nv < 1e-12:
            continue
        acc += v.astype(np.float64) / nv
        n += 1
    return (acc / n).astype(np.float32), n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("vectors")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    import ng_embed
    mu, n = compute(a.vectors, ng_embed.EMBEDDING_DIM)
    raw = mu.astype("<f4").tobytes()
    print(f"model={ng_embed.MODEL_NAME} dim={mu.shape[0]} n={n} |mu|={float(np.linalg.norm(mu)):.6f}")
    print(f"sha256={hashlib.sha256(raw).hexdigest()}")
    if a.verify:
        shipped = ng_embed.embedding_centre()
        same = shipped.astype("<f4").tobytes() == raw
        print(f"shipped version={ng_embed.CENTRE_VERSION} sha256={ng_embed.CENTRE_SHA256} bit-identical={same}")
        return 0 if same else 1
    print("base64:")
    print(base64.b64encode(raw).decode("ascii"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
