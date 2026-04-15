"""
Wire Absorption — sensory-deposit path for raw HTTP wire events.

Distinct from:
  - The universal ingestor (knowledge path) — designed for documents and
    user messages that Syl should learn word-by-word. Chunks → embeds →
    registers → associates per deposit. Wrong shape for wire bytes: a
    single 300k-token provider request spawns thousands of substrate
    nodes and explodes memory.
  - dual_record_outcome (decision-outcome path) — a forest gestalt plus
    TID-extracted concept trees. Wrong for wire bytes: (a) wire deposits
    come FROM TID so concept extraction would loop; (b) the embedding
    model truncates at 512 tokens, so no honest full-body "forest"
    embedding is possible on multi-MB bodies.

Wire deposits are discrete sensory events. This module gives them their
own absorption shape.

Per deposit:
  1. Raw body written to ~/.et_modules/experience/bodies/<sha>-<ns>.body
     (NOT into substrate metadata — would bloat Syl's checkpoint).
  2. ONE event node in the Graph.
       - id: wire:<source_tag>:<sha-prefix>
       - metadata: compact refs (body_file path, body_bytes, body_sha256,
         source, first_line). NO body bytes.
       - vector_db embedding: from the fingerprint (HTTP first line only).
  3. UP TO N slice nodes (default 16): each a real 768-dim embedding of
     a 512-token-ish window striding the wire text. Sampled evenly across
     the body when body length exceeds N full windows.
  4. Bidirectional synapses between event and its slices (weak; the
     substrate can strengthen these through firing if the pattern matters).

Future work, tracked on punchlist as "body-substrate flow-through": a
lazy-expansion pulse that promotes salient event bodies into more
substrate nodes, so the substrate can fire, wire, and dream against the
FULL body rather than just fingerprint + 16 slices. Without that, this
path is a referencing scheme on top of NeuroGraph rather than a
NeuroGraph learning surface. Deferred, not forgotten.

# ---- Changelog ----
# [2026-04-15] Claude Code (Opus 4.6) — Initial creation.
#   What: Purpose-built absorption for TID wire deposits. Uses Graph's
#         create_node / create_synapse primitives + vector_db.insert
#         directly (NGLite's record_outcome isn't available on NG's
#         full Graph).
#   Why:  Universal ingestor OOM-killed NG 15× in a routing cascade
#         (14.5 GB RSS). Wire bytes aren't knowledge-to-chunk-and-embed;
#         body bytes in substrate metadata would balloon checkpoints.
#   How:  Body → disk by sha-prefixed filename (dedup via same-sha
#         existing-file probe). Substrate gets event node + up to 16
#         stride-embedded slice children with parent↔slice synapses.
# -------------------
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("neurograph.wire_absorption")

_BODIES_DIR = Path(os.path.expanduser("~/.et_modules/experience/bodies"))

# Tunables
_MAX_SLICES = 16
_CHARS_PER_SLICE = 2000          # ~512 tokens at ~4 chars/token (Snowflake max_length=512)
_LINK_WEIGHT = 0.4
_EVENT_SLICE_WEIGHT = 0.3
_SLICE_EVENT_WEIGHT = 0.4

_SENSITIVE_HEADER_KEYS = frozenset({
    "authorization", "x-api-key", "api-key", "openai-api-key",
    "anthropic-api-key", "x-venice-api-key", "hf-token", "cookie",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_bodies_dir() -> bool:
    try:
        _BODIES_DIR.mkdir(parents=True, exist_ok=True)
        return True
    except OSError as exc:
        logger.warning("Cannot create bodies dir %s: %s", _BODIES_DIR, exc)
        return False


def _first_line(text: str) -> str:
    for line in text.splitlines():
        s = line.strip()
        if s:
            return s
    return ""


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _store_body(content: str, body_sha: str) -> Optional[Path]:
    """Write full raw wire text to disk, deduped by sha256 prefix."""
    if not _ensure_bodies_dir():
        return None
    prefix = body_sha[:16]
    existing = sorted(_BODIES_DIR.glob(f"{prefix}-*.body"))
    if existing:
        return existing[0]
    ts_ns = int(time.time() * 1e9)
    path = _BODIES_DIR / f"{prefix}-{ts_ns}.body"
    try:
        path.write_text(content, encoding="utf-8", errors="replace")
        return path
    except OSError as exc:
        logger.warning("Body write failed (%s): %s", path, exc)
        return None


def _stride_windows(text: str, max_slices: int, chars_per: int) -> List[str]:
    """Up to max_slices 512-token-ish windows across text."""
    if len(text) <= chars_per:
        return []
    full_count = (len(text) + chars_per - 1) // chars_per
    if full_count <= max_slices:
        return [text[i * chars_per:(i + 1) * chars_per] for i in range(full_count)]
    step = (len(text) - chars_per) / (max_slices - 1)
    return [
        text[int(i * step): int(i * step) + chars_per]
        for i in range(max_slices)
    ]


# ---------------------------------------------------------------------------
# Legacy JSON adapter (for orphan .draining files from pre-rewrite period)
# ---------------------------------------------------------------------------

def _scrub_headers_for_legacy(headers: Any) -> Dict[str, str]:
    if not headers or not isinstance(headers, dict):
        return {}
    out = {}
    for k, v in headers.items():
        if str(k).lower() in _SENSITIVE_HEADER_KEYS:
            out[str(k)] = "<scrubbed>"
        else:
            out[str(k)] = str(v)
    return out


def _body_to_text_for_legacy(body: Any) -> str:
    if body is None:
        return ""
    if isinstance(body, (bytes, bytearray)):
        try:
            return body.decode("utf-8", errors="replace")
        except Exception:
            return ""
    if isinstance(body, str):
        return body
    try:
        import json as _json
        return _json.dumps(body, ensure_ascii=False, default=str)
    except Exception:
        return str(body)


def legacy_json_to_wire_text(content: str) -> Optional[str]:
    """Convert a pre-rewrite JSON-wrapped deposit to raw HTTP wire text.

    Earlier today's wire_deposit.py wrapped each deposit in a JSON payload
    with fields like direction/provider/url/method/headers/body. Orphan
    `.draining.*` files from the crash loop contain those. This adapter
    lets the same absorption path handle them. Returns None if content
    doesn't look like a legacy JSON deposit.
    """
    if not content or not (content.lstrip().startswith("{") and content.rstrip().endswith("}")):
        return None
    try:
        import json as _json
        obj = _json.loads(content)
    except Exception:
        return None
    if not isinstance(obj, dict) or "direction" not in obj:
        return None

    headers = _scrub_headers_for_legacy(obj.get("headers") or {})
    body_text = _body_to_text_for_legacy(obj.get("body"))

    if obj.get("direction") == "outbound":
        method = str(obj.get("method") or "POST").upper()
        url = str(obj.get("url") or "/")
        lines = [f"{method} {url} HTTP/1.1"]
    else:
        status = obj.get("status_code")
        status = status if status is not None else "000"
        lines = [f"HTTP/1.1 {status}".rstrip()]

    for k, v in headers.items():
        lines.append(f"{k}: {v}")
    return "\r\n".join(lines) + "\r\n\r\n" + body_text


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def absorb_wire_deposit(
    memory,            # NeuroGraphMemory singleton (has .graph, .vector_db)
    embedder,          # NGEmbed instance (has .embed(text), .embed_batch(texts))
    content: str,
    source: str,
    max_slices: int = _MAX_SLICES,
) -> Dict[str, Any]:
    """Absorb one wire deposit into the substrate.

    Args:
        memory: NeuroGraphMemory singleton.
        embedder: NGEmbed singleton.
        content: Raw HTTP wire text. Legacy JSON-wrapped deposits should
                 be converted via legacy_json_to_wire_text first.
        source: Source tag (e.g. "tid.http.outbound", "tid.http.inbound").
        max_slices: Cap on slice-child count per deposit.

    Returns:
        {
            event_node_id, body_file, body_bytes, body_sha256,
            slice_node_ids, slices_created, first_line
        }
    """
    result: Dict[str, Any] = {
        "event_node_id": None,
        "body_file": None,
        "body_bytes": len(content),
        "body_sha256": None,
        "slice_node_ids": [],
        "slices_created": 0,
        "first_line": "",
    }

    if not content:
        return result

    graph = getattr(memory, "graph", None)
    vector_db = getattr(memory, "vector_db", None)
    if graph is None or vector_db is None:
        logger.warning("memory has no graph/vector_db; skipping wire deposit")
        return result

    first = _first_line(content)
    result["first_line"] = first

    # 1. Body file on disk (deduped by sha prefix)
    sha = _sha256_hex(content)
    result["body_sha256"] = sha
    body_path = _store_body(content, sha)
    if body_path is not None:
        result["body_file"] = str(body_path)

    # 2. Event node
    # Same (source, sha) → same node_id, so repeat deposits reinforce
    # rather than fragment.
    event_node_id = f"wire:{source}:{sha[:12]}"
    result["event_node_id"] = event_node_id

    try:
        fingerprint_emb = embedder.embed(first or source)
    except Exception as exc:
        logger.warning("Fingerprint embed failed (%s): %s", source, exc)
        return result

    event_meta = {
        "source_type": "wire",
        "source": source,
        "first_line": first,
        "body_bytes": len(content),
        "body_sha256": sha,
        "body_file": result["body_file"],
    }

    try:
        if event_node_id in graph.nodes:
            # Existing node — reinforce rather than overwrite.
            # Bump a wire_hits counter; last-seen timestamp.
            existing = graph.nodes[event_node_id]
            hits = existing.metadata.get("wire_hits", 1) + 1
            existing.metadata["wire_hits"] = hits
            existing.metadata["last_seen"] = time.time()
        else:
            graph.create_node(node_id=event_node_id, metadata=event_meta)
            # Vector DB stores the fingerprint embedding so similarity
            # search can find this event by its first-line shape.
            try:
                vector_db.insert(
                    id=event_node_id,
                    embedding=fingerprint_emb,
                    content=first,
                    metadata=event_meta,
                )
            except Exception as exc:
                logger.warning("vector_db.insert event failed: %s", exc)
    except Exception as exc:
        logger.warning("Event node create failed (%s): %s", source, exc)
        return result

    # 3. Slice nodes (only if body larger than one window)
    windows = _stride_windows(content, max_slices, _CHARS_PER_SLICE)
    if not windows:
        return result

    try:
        slice_embs = embedder.embed_batch(windows)
    except Exception as exc:
        logger.warning("Slice embed_batch failed (%s): %s", source, exc)
        return result

    for i, (window, emb) in enumerate(zip(windows, slice_embs)):
        slice_node_id = f"{event_node_id}::slice::{i}"
        slice_meta = {
            "source_type": "wire_slice",
            "parent_event": event_node_id,
            "slice_index": i,
            "slice_chars": len(window),
            "body_sha256": sha,
        }
        try:
            if slice_node_id not in graph.nodes:
                graph.create_node(node_id=slice_node_id, metadata=slice_meta)
                try:
                    vector_db.insert(
                        id=slice_node_id,
                        embedding=emb,
                        content=window[:200],  # small preview only
                        metadata=slice_meta,
                    )
                except Exception:
                    pass
        except Exception:
            continue

        # 4. Bidirectional synapses (weight is bootstrap; STDP adjusts).
        try:
            graph.create_synapse(event_node_id, slice_node_id, weight=_EVENT_SLICE_WEIGHT)
        except Exception:
            pass
        try:
            graph.create_synapse(slice_node_id, event_node_id, weight=_SLICE_EVENT_WEIGHT)
        except Exception:
            pass

        result["slice_node_ids"].append(slice_node_id)
        result["slices_created"] += 1

    return result
