"""
vision_absorption.py — the sensory door for sight (punchlist #82).

Frames arrive from Portal as BTF OUTCOME entries on the experience tract, already
encoded ON THE PHONE (SigLIP 2 vision tower, 768-d). The mother hosts the nodes,
never the model. This module turns one frame's signals into substrate topology:

    forest node   = pooler_output (the whole image)
    tree nodes    = pooled patch groups (parts and pieces of the image)
    hyperedge     = forest + trees, one experience cluster  (Dual-Pass, #81)
    prev->current = delayed forest->forest synapse across frames (#257 polychrony;
                    the seed of #63 video)

What this module NEVER does — each is a Law or a scar:
  * touch the vector DB. Vision vectors are not text vectors; cosine between them
    is noise (the #45 scar). Nodes exist fully in the SNN (STDP, spreading
    activation, hyperedges, polychrony, DiffPC, MMN) and are reachable by every
    mechanism that runs on timing and topology. They are simply never inserted
    into the cosine store. This is index_in_recall=False (#295) by construction.
  * import the Universal Ingestor. Vision is experience, not documentation.
  * caption, classify, or label. LAW 7: the image enters as the image. The bytes
    go to the body store beside the node (metadata['_image_ref']) so surfacing
    can show HER the picture, not a description of it (#410).
  * run an embedding model. Embeddings arrive from the eye.

Deposit is PROMPT (decided 2026-09-06): a frame binds to the moment it came from,
so the forest gets a gentle voltage nudge on arrival — a context cue, the same
shape the stream parser uses — and co-fires with whatever is active. That co-firing
is the binding; no shared embedding space is needed or wanted.

Pure functions over a Graph. Sandbox-testable against a fresh Graph().

# ---- Changelog ----
# [2026-09-06] DudeMan CC (Fable 5.1) — Created. #82 Inc 1.
#   What: absorb_entries(graph, entries) — group BTF OUTCOME entries by frame, create
#         forest + tree nodes, forest<->tree synapses, binding hyperedge, delayed
#         prev-frame link, prompt voltage nudge. store_image_body() for the bytes.
#         split_vision_entries() for the scan-drain dispatcher.
#   Why:  Give Syl a working eye through Portal (design: ~/docs/concepts/Multimodal
#         Perceptual Embedding.md; plan: superpowers/plans/2026-09-06-82-vision-build-plan.md).
#   How:  Mirrors wire_absorption (body store) and _bind_conversational_topology
#         (synapse/hyperedge shape) exactly, so vision is a third feeder on the
#         existing sensory door, not a new mechanism. Dimension guard rejects any
#         embedding that is not the substrate width — the #45 class of bug is a
#         hard reject here, never a silent deposit.
# -------------------
"""
from __future__ import annotations

import hashlib
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("neurograph.vision")

# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

#: source prefix the scan-drain dispatcher routes here. Anything else on the
#: experience tract that is not wire falls through to the ingestor — vision must
#: never take that fallthrough (it OOM-looped on 2026-04-15 for wire bytes).
VISION_PREFIX = "portal.vision"

#: substrate node width. Same constant as ng_embed / the ingestor / poincare_dir.
EMBEDDING_DIM = 768

#: forest<->tree synapse weights — identical to _bind_conversational_topology.
FOREST_TO_TREE_W = 0.2
TREE_TO_FOREST_W = 0.15
#: delayed prev-frame -> current-frame forest link (#257). Bootstrap range.
FRAME_LINK_W = 0.2
FRAME_LINK_DELAY_MAX = 6
#: prompt-binding nudge on the forest at deposit. Context cue, not a spike —
#: the stream parser uses the same shape (0.15 * similarity). Bootstrap value.
DEPOSIT_NUDGE = 0.15

_last_frame_forest_id: Optional[str] = None


def _bodies_dir() -> Path:
    """One body store for all sensory feeders (LAW 4 — reuse wire's)."""
    try:
        from wire_absorption import _BODIES_DIR  # type: ignore
        return Path(_BODIES_DIR)
    except Exception:  # noqa: BLE001 - standalone/test use
        return Path("~/.et_modules/experience/bodies").expanduser()


# ---------------------------------------------------------------------------
# Entry adaptation — BTF PyOutcomeEntry or a plain dict (tests)
# ---------------------------------------------------------------------------

def _entry_fields(e: Any) -> Optional[Dict[str, Any]]:
    """Normalise a BTF OUTCOME entry (or dict) to {module_id, target_id, embedding, meta, ts}."""
    if isinstance(e, dict):
        module_id = str(e.get("module_id", ""))
        target_id = str(e.get("target_id", ""))
        emb = e.get("embedding")
        meta = e.get("metadata") or {}
        ts = float(e.get("timestamp") or time.time())
    else:
        module_id = str(getattr(e, "module_id", "") or "")
        target_id = str(getattr(e, "target_id", "") or "")
        fn = getattr(e, "embedding_as_numpy", None)
        emb = fn() if callable(fn) else getattr(e, "embedding", None)
        raw = getattr(e, "metadata", None)
        raw = raw() if callable(raw) else raw
        meta = _unpack_meta(raw)
        ts = float(getattr(e, "timestamp", 0.0) or time.time())
    if emb is None:
        return None
    return {"module_id": module_id, "target_id": target_id,
            "embedding": np.asarray(emb, dtype=np.float32).reshape(-1),
            "meta": dict(meta), "ts": ts}


def _unpack_meta(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, (bytes, bytearray)):
        try:
            import msgpack
            v = msgpack.unpackb(bytes(raw), raw=False)
            return v if isinstance(v, dict) else {}
        except Exception:  # noqa: BLE001
            return {}
    return {}


def is_vision_entry(e: Any) -> bool:
    """True for a BTF OUTCOME entry (or dict) whose module_id is the vision prefix."""
    mid = e.get("module_id") if isinstance(e, dict) else getattr(e, "module_id", None)
    return isinstance(mid, str) and mid.startswith(VISION_PREFIX)


def split_vision_entries(entries: Iterable[Any]) -> Tuple[List[Any], List[Any]]:
    """(vision_entries, everything_else). For the scan-drain dispatcher.

    Must run BEFORE the wire/non-wire split, which reads `.source` (an EXPERIENCE
    field) and would route an unrecognised entry to the ingestor.
    """
    vis, rest = [], []
    for e in entries:
        (vis if is_vision_entry(e) else rest).append(e)
    return vis, rest


# ---------------------------------------------------------------------------
# Body store — the picture itself, beside the node, never inside it
# ---------------------------------------------------------------------------

def store_image_body(data: bytes, sha: Optional[str] = None, ext: str = "jpg") -> Optional[Path]:
    """Write raw image bytes to the shared body store, deduped by sha256 prefix.

    Same scheme as wire_absorption._store_body: <sha16>-<ns>.<ext>. Returns the
    path (existing one if already stored), or None if the store is unwritable.
    """
    if not data:
        return None
    sha = sha or hashlib.sha256(data).hexdigest()
    d = _bodies_dir()
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("Cannot create bodies dir %s: %s", d, exc)
        return None
    prefix = sha[:16]
    existing = sorted(d.glob(f"{prefix}-*.{ext}"))
    if existing:
        return existing[0]
    path = d / f"{prefix}-{int(time.time() * 1e9)}.{ext}"
    try:
        path.write_bytes(data)
        return path
    except OSError as exc:
        logger.warning("Image body write failed (%s): %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# Absorption
# ---------------------------------------------------------------------------

def _poincare_pack(embedding: np.ndarray) -> Optional[Any]:
    """Unit direction, packed the way _deposit_memory_node stores it (#119)."""
    try:
        from neuro_foundation import pack_poincare_dir
        n = float(np.linalg.norm(embedding))
        d = embedding / n if n > 1e-9 else embedding.copy()
        return pack_poincare_dir(d)
    except Exception:  # noqa: BLE001
        return None


def _node_meta(kind: str, f: Dict[str, Any], frame_id: str) -> Dict[str, Any]:
    m = f["meta"]
    meta: Dict[str, Any] = {
        "creation_mode": "sensory",
        "modality": "vision",
        "source": VISION_PREFIX,
        "kind": kind,
        "frame_id": frame_id,
        "ts": f["ts"],
    }
    for k in ("image_sha", "width", "height", "tree_index", "n_trees", "facing"):
        if k in m:
            meta[k] = m[k]
    if kind == "forest" and m.get("image_ref"):
        meta["_image_ref"] = str(m["image_ref"])   # surfacing renders THIS (#410)
    pd = _poincare_pack(f["embedding"])
    if pd is not None:
        meta["poincare_dir"] = pd
    return meta


def absorb_entries(graph: Any, entries: Iterable[Any], nudge: float = DEPOSIT_NUDGE) -> List[Dict[str, Any]]:
    """Absorb vision OUTCOME entries into `graph`. Returns one result per frame absorbed.

    Groups by metadata['frame_id'] (falls back to the target_id stem). A frame
    without a forest is skipped — trees alone have nothing to bind to.
    Never touches a vector DB; never imports the ingestor.
    """
    global _last_frame_forest_id
    frames: Dict[str, Dict[str, Any]] = {}
    rejected = 0
    for e in entries:
        f = _entry_fields(e)
        if f is None or not f["module_id"].startswith(VISION_PREFIX):
            continue
        if f["embedding"].shape[0] != EMBEDDING_DIM:
            rejected += 1
            logger.warning("vision: REJECTED %s — dim %d != %d (the #45 class of bug is a hard reject)",
                           f["target_id"], f["embedding"].shape[0], EMBEDDING_DIM)
            continue
        kind = str(f["meta"].get("kind") or ("tree" if "::tree::" in f["target_id"] else "forest"))
        frame_id = str(f["meta"].get("frame_id") or f["target_id"].split("::")[1] if "::" in f["target_id"] else f["target_id"])
        fr = frames.setdefault(frame_id, {"forest": None, "trees": []})
        (fr["trees"].append(f) if kind == "tree" else fr.__setitem__("forest", f))

    results: List[Dict[str, Any]] = []
    for frame_id, fr in frames.items():
        forest = fr["forest"]
        if forest is None:
            logger.debug("vision: frame %s has trees but no forest — skipped", frame_id)
            continue
        forest_id = forest["target_id"] or f"{VISION_PREFIX}::{frame_id}::forest"
        _upsert(graph, forest_id, _node_meta("forest", forest, frame_id))
        tree_ids: List[str] = []
        for i, t in enumerate(sorted(fr["trees"], key=lambda x: int(x["meta"].get("tree_index", 0)))):
            tid = t["target_id"] or f"{VISION_PREFIX}::{frame_id}::tree::{i}"
            _upsert(graph, tid, _node_meta("tree", t, frame_id))
            tree_ids.append(tid)
        # forest<->tree synapses + binding hyperedge — the dual-pass shape
        for tid in tree_ids:
            _syn(graph, forest_id, tid, FOREST_TO_TREE_W)
            _syn(graph, tid, forest_id, TREE_TO_FOREST_W)
        he_id = None
        if tree_ids:
            try:
                he = graph.create_hyperedge(
                    member_node_ids=set([forest_id] + tree_ids),
                    metadata={"creation_mode": "sensory", "modality": "vision", "frame_id": frame_id},
                )
                he_id = getattr(he, "hyperedge_id", None) or getattr(he, "id", None)
            except Exception as exc:  # noqa: BLE001
                logger.debug("vision hyperedge failed (non-fatal): %s", exc)
        # delayed prev-frame -> this-frame link (#257 polychrony; #63 seed)
        prev = _last_frame_forest_id
        if prev and prev != forest_id and prev in graph.nodes:
            _syn(graph, prev, forest_id, FRAME_LINK_W, delay=random.randint(2, max(2, FRAME_LINK_DELAY_MAX)))
        _last_frame_forest_id = forest_id
        # prompt binding: land in the current activation window
        if nudge > 0:
            node = graph.nodes.get(forest_id)
            if node is not None and getattr(node, "refractory_remaining", 0) == 0:
                node.voltage = min(node.voltage + nudge, node.threshold * 2.0)
        results.append({"frame_id": frame_id, "forest_id": forest_id,
                        "tree_ids": tree_ids, "hyperedge_id": he_id})
    if rejected:
        logger.warning("vision: %d entries rejected for wrong dimension", rejected)
    return results


def _upsert(graph: Any, node_id: str, meta: Dict[str, Any]) -> Any:
    node = graph.nodes.get(node_id)
    if node is None:
        return graph.create_node(node_id=node_id, metadata=meta)
    node.metadata.update(meta)
    return node


def _syn(graph: Any, pre: str, post: str, w: float, delay: int = 1) -> None:
    try:
        graph.create_synapse(pre, post, weight=w, delay=delay)
    except Exception:  # noqa: BLE001 - may already exist; non-fatal (same as conversational)
        pass
