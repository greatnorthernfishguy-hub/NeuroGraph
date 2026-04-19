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
# [2026-04-19] Claude Code (Opus 4.7, 1M) — forest_embedding in batch result
#   What: batch_absorb_forests() now includes "forest_embedding": emb in
#     each result dict. Passed through _CONCEPT_QUEUE and into TriSyn
#     worker's handoff file.
#   Why:  TriSyn worker runs in a subprocess and would otherwise re-derive
#     the embedding from content first-line — which would diverge from
#     the actual forest-node embedding already written to vector_db.
#     Advisor flag 2026-04-18: carry what was created, don't re-derive.
#   How:  Single dict-entry addition. No behavioral change for existing
#     callers (extra key is ignored if not consumed).
# [2026-04-17] Claude Code (Sonnet 4.6) — Dual-pass retrofit (#150).
#   What: Replaced 16-stride-slice hack with proper dual_record_outcome.
#         Forest = event node fingerprint embedding (unchanged). Trees =
#         real concepts/keywords/key-phrases extracted by TID via the
#         ecosystem's existing _extract_concepts mechanism — same tool
#         every other module uses.
#   Why:  16 arbitrary 512-token strides produced redundancy (same content
#         re-sliced across deposits) and shallow understanding (stride
#         boundaries are arbitrary, not semantic). Dual-pass extracts
#         meaningful concepts the substrate can actually learn from.
#   How:  _PeerBridgeEco adapter wraps _memory._peer_bridge to match the
#         NGEcosystem.record_outcome interface dual_record_outcome expects.
#         Recursion guard: concept-extraction TID calls generate wire
#         deposits that would re-trigger dual_record_outcome → infinite
#         loop. Detected by _CONCEPT_SENTINEL string in wire body content.
#         Meta-deposits get forest-only recording (no TID call, loop breaks).
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

# Tunables — stride-slice constants retained for reference but no longer
# used after dual-pass retrofit (#150). Trees come from real concept
# extraction via TID, not arbitrary 512-token strides.
# _MAX_SLICES = 16  # REMOVED — dual-pass extracts up to 20 concepts
# _CHARS_PER_SLICE = 2000  # REMOVED — dual-pass uses first 2000 chars
# _LINK_WEIGHT = 0.4  # REMOVED — dual-pass uses 0.4 / 0.28 from ng_embed
# _EVENT_SLICE_WEIGHT = 0.3  # REMOVED
# _SLICE_EVENT_WEIGHT = 0.4  # REMOVED

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


# _stride_windows REMOVED — dual-pass retrofit (#150) replaced stride
# slicing with proper concept extraction via dual_record_outcome. Trees
# are real keywords/concepts/key-phrases extracted by TID, not arbitrary
# 512-token windows. The substrate learns routing patterns from meaningful
# concepts, not stride-boundary artifacts.


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
# Batch forest absorption (fast path)
# ---------------------------------------------------------------------------

# Batch size for drain pulse — how many entries to embed+record per tick.
# At 50: ~900ms embedding + fast record_outcome calls.  Leaves room in
# a 2s tick for other pulse work.  Clears a 2000-entry backlog in ~40 ticks
# (~80 seconds) instead of the old 1-per-tick throttle (67 minutes).
_DRAIN_BATCH_SIZE = 50


def batch_absorb_forests(
    memory,
    embedder,
    entries: List[Dict[str, str]],  # each: {"content": str, "source": str}
) -> List[Dict[str, Any]]:
    """Batch-absorb wire deposits as forest nodes only.

    Fast path: one embed_batch call for all fingerprints, then individual
    peer_bridge.record_outcome calls.  No TID calls, no concept extraction,
    no blocking network I/O.

    Returns a list of result dicts (one per entry) with event_node_id and
    content_preview populated — the concept pulse uses these to add trees
    later.
    """
    if not entries:
        return []

    graph = getattr(memory, "graph", None)
    vector_db = getattr(memory, "vector_db", None)
    peer_bridge = getattr(memory, "_peer_bridge", None)
    if graph is None or vector_db is None:
        return []

    # 1. Compute fingerprints (first non-empty line of each body)
    firsts = [_first_line(e["content"]) or e["source"] for e in entries]

    # 2. Batch embed all fingerprints in one call
    try:
        embeddings = embedder.embed_batch(firsts)
    except Exception as exc:
        logger.warning("Batch fingerprint embed failed: %s", exc)
        return []

    # 3. For each entry: store body, create/reinforce event node, record forest
    results = []
    for entry, first, emb in zip(entries, firsts, embeddings):
        content = entry["content"]
        source = entry["source"]
        sha = _sha256_hex(content)
        body_path = _store_body(content, sha)
        event_node_id = f"wire:{source}:{sha[:12]}"

        event_meta = {
            "source_type": "wire",
            "source": source,
            "first_line": first,
            "body_bytes": len(content),
            "body_sha256": sha,
            "body_file": str(body_path) if body_path else None,
        }

        # Create/reinforce event node in the graph
        try:
            if event_node_id in graph.nodes:
                existing = graph.nodes[event_node_id]
                hits = existing.metadata.get("wire_hits", 1) + 1
                existing.metadata["wire_hits"] = hits
                existing.metadata["last_seen"] = time.time()
            else:
                graph.create_node(node_id=event_node_id, metadata=event_meta)
                try:
                    vector_db.insert(
                        id=event_node_id, embedding=emb,
                        content=first, metadata=event_meta,
                    )
                except Exception:
                    pass
        except Exception:
            continue

        # Record forest outcome via peer bridge (River deposit)
        if peer_bridge is not None:
            try:
                peer_bridge.record_outcome(
                    embedding=emb, target_id=event_node_id,
                    success=True, module_id="neurograph",
                    metadata=event_meta,
                )
            except Exception:
                pass

        results.append({
            "event_node_id": event_node_id,
            "content_preview": content[:2000],
            "source": source,
            "body_file": str(body_path) if body_path else None,
            "body_bytes": len(content),
            "body_sha256": sha,
            # TriSyn subprocess worker reads this at spawn to avoid
            # re-deriving the embedding from content first-line (which
            # would diverge from the actual forest-node embedding).
            "forest_embedding": emb,
        })

    return results


# ---------------------------------------------------------------------------
# Concept extraction (slow path — deferred from drain pulse)
# ---------------------------------------------------------------------------

# Sentinel for recursion detection — concept-extraction TID calls
# produce wire deposits that get re-absorbed.  If the wire body
# contains this prompt fragment, it's a meta-deposit from our own
# concept extraction — skip trees, record forest only.
_CONCEPT_SENTINEL = "You extract concepts from text"


def absorb_trees_for_entry(
    memory,
    embedder,
    content_preview: str,
    source: str,
    event_node_id: str,
) -> Dict[str, Any]:
    """Add concept trees to an existing forest node.

    Slow path: calls _extract_concepts via TID (blocking network I/O),
    embeds concepts, records tree outcomes + cross-links.  Called from
    the concept pulse, NOT from the drain pulse.

    Returns tree extraction results or empty dict on failure.
    """
    result = {"tree_ids": [], "concepts": [], "trees_created": 0}

    if _CONCEPT_SENTINEL in content_preview:
        return result  # recursion guard — meta-deposit, no trees

    peer_bridge = getattr(memory, "_peer_bridge", None)
    if peer_bridge is None:
        return result

    try:
        from ng_embed import NGEmbed

        class _PeerBridgeEco:
            def __init__(self, bridge):
                self._bridge = bridge
            def record_outcome(self, embedding, target_id, success,
                               strength=1.0, metadata=None):
                meta = dict(metadata or {})
                if strength != 1.0:
                    meta["_strength"] = strength
                return self._bridge.record_outcome(
                    embedding, target_id, success, "neurograph", meta,
                )

        fingerprint_emb = embedder.embed(
            _first_line(content_preview) or source
        )
        eco_adapter = _PeerBridgeEco(peer_bridge)
        dp_result = NGEmbed.get_instance().dual_record_outcome(
            ecosystem=eco_adapter,
            content=content_preview,
            embedding=fingerprint_emb,
            target_id=event_node_id,
            success=True,
            strength=1.0,
            metadata={"source": source, "source_type": "wire"},
        )
        result["tree_ids"] = dp_result.get("tree_ids", [])
        result["concepts"] = dp_result.get("concepts", [])
        result["trees_created"] = len(result["tree_ids"])
    except Exception as exc:
        logger.warning("Tree extraction failed (%s): %s", source, exc)

    return result


# ---------------------------------------------------------------------------
# Legacy single-entry entry point (retained for non-batch callers)
# ---------------------------------------------------------------------------

def absorb_wire_deposit(
    memory,            # NeuroGraphMemory singleton (has .graph, .vector_db)
    embedder,          # NGEmbed instance (has .embed(text), .embed_batch(texts))
    content: str,
    source: str,
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

    # 3. Proper 2nd-pass embedding via dual_record_outcome (#150).
    #
    # Replaces the 16-stride-slice hack with the ecosystem's own dual-pass
    # mechanism: Forest (gestalt) + Trees (real concepts/keywords/key-phrases
    # extracted by TID). Same tool every other module uses — TrollGuard,
    # Elmer, Praxis, Agent Zero, Faux_Clawdbot all call dual_record_outcome.
    #
    # Recursion guard: _extract_concepts calls TID → TID wire_deposit
    # captures outbound → scan-drain absorbs → dual_record_outcome again.
    # Break: if the wire body contains the concept-extraction system prompt
    # ("You extract concepts from text"), this IS a meta-deposit from our
    # own tree-extraction path. Forest-only for those, no TID call.
    _CONCEPT_SENTINEL = "You extract concepts from text"

    peer_bridge = getattr(memory, '_peer_bridge', None)
    if peer_bridge is None:
        return result

    is_meta_deposit = _CONCEPT_SENTINEL in content[:2000]

    if is_meta_deposit:
        # Forest only — no tree extraction, breaks recursion.
        try:
            peer_bridge.record_outcome(
                embedding=fingerprint_emb,
                target_id=event_node_id,
                success=True,
                module_id="neurograph",
                metadata=event_meta,
            )
        except Exception:
            pass
    else:
        # Full dual-pass: Forest + Trees via TID concept extraction.
        # Adapter wraps peer_bridge to match NGEcosystem.record_outcome
        # signature that NGEmbed.dual_record_outcome expects.
        try:
            from ng_embed import NGEmbed

            class _PeerBridgeEco:
                """Adapt peer_bridge for dual_record_outcome's ecosystem API."""
                def __init__(self, bridge):
                    self._bridge = bridge
                def record_outcome(self, embedding, target_id, success,
                                   strength=1.0, metadata=None):
                    meta = dict(metadata or {})
                    if strength != 1.0:
                        meta["_strength"] = strength
                    return self._bridge.record_outcome(
                        embedding, target_id, success,
                        "neurograph", meta,
                    )

            eco_adapter = _PeerBridgeEco(peer_bridge)
            dp_result = NGEmbed.get_instance().dual_record_outcome(
                ecosystem=eco_adapter,
                content=content[:2000],   # first 2000 chars — same as all callers
                embedding=fingerprint_emb,
                target_id=event_node_id,
                success=True,
                strength=1.0,
                metadata=event_meta,
            )
            result["tree_ids"] = dp_result.get("tree_ids", [])
            result["concepts"] = dp_result.get("concepts", [])
            result["slices_created"] = len(result["tree_ids"])
            logger.info(
                "Wire dual-pass: %s → %d trees from %d concepts",
                source, len(result["tree_ids"]),
                len(result.get("concepts", [])),
            )
        except Exception as exc:
            # Dual-pass failed — forest already recorded above via
            # graph.create_node. Not fatal; just no trees this time.
            logger.warning("Wire dual-pass failed (%s): %s", source, exc)

    return result
