"""
NG-Lite — Lightweight NeuroGraph Learning Substrate v1.0

Single-file learning substrate for E-T Systems modules. Provides
standalone Hebbian learning, novelty detection, and JSON persistence.

Designed to be vendored into any module as a single-file dependency.
No external dependencies beyond numpy and the Python standard library.

When NeuroGraph SaaS is available, NG-Lite delegates to the full
substrate for cross-module learning, predictive coding, and hypergraph
capabilities via the NGBridge interface. When disconnected, NG-Lite
operates independently with local learning — no functionality is lost,
only the ecosystem-level synergies.

Design principles (aligned with NeuroGraph Foundation PRD §2.1):
    - Sparse by default: dict-based storage, no dense matrices
    - Bounded memory: configurable max nodes/synapses with LRU pruning
    - Persistence-native: full state serializable to JSON
    - Upgrade-ready: clean bridge interface to full NeuroGraph SaaS

Connectivity tiers:
    Tier 1 — Isolated: Module runs its own NG-Lite independently.
    Tier 2 — Peer-pooled: Co-located modules share learning via
             NGTractBridge (per-pair directional tracts). Sole peer
             bridge as of 2026-06-03 (substrate-as-protocol PRD Phase 3
             Step 5 deleted the legacy NGPeerBridge). Uses the NGBridge
             interface — the module doesn't know or care whether its
             bridge partner is a sibling module's tract or the full SaaS.
    Tier 3 — Full SaaS: NG-Lite delegates to NeuroGraph for cross-module
             learning, STDP, hyperedges, and predictive coding.

    Tier transitions are transparent. A module starts at Tier 1,
    discovers a co-located sibling and upgrades to Tier 2, then
    connects to SaaS and upgrades to Tier 3 — all without code changes.
    If SaaS disconnects, it falls back to Tier 2 or 1 automatically.

Serialization format notes (2026-10-13, Josh: "JSON will have NOTHING to do
with saves, period"):
    - Saves are msgpack only, on both cores. ``save("x.json")`` writes
      ``x.msgpack``; any other name is written as given. The content is the
      ``_export_state()`` dict, embeddings / prototypes as msgpack float32.
      The Python core needs the ``msgpack`` package (no JSON fallback).
    - ``load()`` reads that msgpack. A legacy JSON state (``x.json``, or the
      JSON the old Python core wrote into a ``.msgpack``-named file, #1098)
      is converted ONCE: imported, re-saved as msgpack, and the JSON kept
      aside as ``<file>.legacy-json``.
    - Format translation for bridges still happens in the bridge layer.

Weight range notes:
    NG-Lite weights are bounded [0.0, 1.0] for simplicity.
    Full NeuroGraph uses [0.0, max_weight] (default 5.0).
    The bridge normalizes: ng_weight * max_weight ↔ full_weight / max_weight.

Node ID notes:
    NG-Lite uses incremental IDs ("n_1", "n_2") for compactness.
    Full NeuroGraph uses UUIDs for global uniqueness.
    The bridge maintains a mapping table during sync_state().

Ethical obligations (per NeuroGraph ETHICS.md):
    - Type I error bias: when uncertain, err toward respect
    - Choice Clause: no module may block agent autonomy
    - Transparency: all learning decisions are queryable

Canonical source: https://github.com/greatnorthernfishguy-hub/NeuroGraph
License: AGPL-3.0 (see NeuroGraph LICENSE)

Author: Josh + Claude
Date: February 2026

# ---- Changelog ----
# [2026-10-14] Claude Code (Opus 5.5, lane mean-centre) — #1107: mean-centred comparisons; the rim threshold; space tag
# What: (1) Every similarity comparison subtracts the embedding model's fixed centre (ng_embed.embedding_centre(),
#   versioned ng_embed.CENTRE_VERSION) from both sides and re-normalises before the cosine: node matching
#   (_find_similar_node), novelty (detect_novelty), the receptor snap (_snap_to_prototype) and its k-means assignment
#   (_init_prototypes). Stored vectors stay RAW (node embeddings, prototypes, hashes, saves): centring happens only at
#   the comparison, so the substrate keeps the raw experience (LAW 7) and a new centre version needs no re-embedding.
#   Applies when embedding_dim == ng_embed.EMBEDDING_DIM and config "mean_centre" is true (default); other dims (test
#   fixtures) compare raw. (2) Thresholds re-calibrated on centred real data (MEAN_CENTRE.md §3):
#   novelty_threshold 0.7 -> 0.40 (match at centred cosine >= 0.60; tunable bounds 0.30-0.95 -> 0.20-0.60),
#   receptor_prototype_threshold stays 0.75 (now centred), and a separate constitutional_threshold 0.55 (not runtime-tunable — the rim is immutable): a constitutional node catches an
#   input only at centred cosine >= constitutional_threshold, and when it does it takes priority over ordinary nodes;
#   ordinary nodes match at >= 1 - novelty_threshold as before. (3) Saves carry "embedding_space" = {model, dim,
#   centre_version}; load() refuses a state recorded under another model (EmbeddingSpaceError, ERROR once), refuses an
#   untagged state or a rim whose vectors are off the model's manifold (cosine to the centre direction), and resets
#   the space-dependent thresholds to this instance's values when the state was calibrated under another centre
#   version (incl. untagged = raw-cosine era).
# Why: Josh 2026-10-14 "Mean center" (#1107). Snowflake vectors share a common direction: unrelated texts sit at
#   ~0.69 raw cosine, so the 0.30 snap matched nearly everything — the CC Commons put 4,767 synapses on 4 nodes
#   (#1094), TID's NG-Lite had 2 nodes, and a 768-d Elmer rim would have caught 100% of inputs. The rim is immutable
#   ("Running without the Rim defeats the entire point"), so it needs a threshold that catches rim-relevant inputs
#   and not ordinary ones. The dimension guard could not see same-dim foreign vectors (DIM_GUARD.md §11).
# How: _centre_vec / _centre_rows (the Rust core mirrors them bit for bit, ng-tract-rs
#   cc-laptop-mean-centre-rs-20261014); _resolve_space at construction; _space_check before import (both cores);
#   _space_migrate_config after import. No change for non-model dims beyond the rim threshold rule.
# [2026-10-14] Claude Code (Opus 5.5, lane dim-guard) — #1097: fail loud on a wrong embedding dimension
# What: Every vector that enters NG-Lite is checked against ``embedding_dim`` (the active embedding
#   model's dimension; ng_embed's 768 by default, pinned by tests/test_dim_guard.py):
#   - inputs to find_or_create_node / record_outcome / get_recommendations / detect_novelty;
#   - constitutional (rim) entries at seeding (constructor and after a load);
#   - a loaded state's node embeddings and receptor prototypes (load / _import_state), with the
#     state file named in the error.
#   A mismatch raises EmbeddingDimensionError (a ValueError subclass, so existing handlers still
#   catch it) whose message names the module, what was wrong, the file, both dimensions and the
#   counts per dimension; the same message is logged at ERROR once per distinct mismatch, so a
#   caller's broad ``except`` can no longer make it silent. The check runs in this wrapper before
#   any dispatch, so it covers the Python and the Rust core alike.
#   A saved state's config can no longer override ``embedding_dim``: the constructor's value (the
#   module's model) wins, and the stored vectors are checked against it.
# Why: #1097 — Elmer's NG-Lite rim was 384-d (all-MiniLM-L6-v2) while every input is 768-d; each
#   call raised a numpy shape error deep inside the similarity search, a broad except swallowed it,
#   and the rim did nothing for weeks. Josh 2026-10-14: wrong dims keep surviving sweeps — make it
#   structurally impossible to run on them quietly (LAW 4: at the source).
# How: _check_input / _check_rim_entries / _check_state_dims + _dim_error; EmbeddingDimensionError.
#   No change for correct-dim data (no extra work beyond one shape test per call).
# -------------------
# [2026-10-13] Claude Code (Opus 5.5, lane nglite-rustcore) — #1096: Rust core usable; public reads; msgpack saves
# What: (1) Public read methods served by whichever core holds the state: node_count / synapse_count /
#   list_nodes / get_node / list_synapses / get_synapse / remove_synapses / get_history / export_state.
#   They return snapshot copies (mutating them does not touch the substrate). Consumers stop reading
#   NGLite.nodes / NGLite.synapses / _embedding_cache / _total_outcomes directly.
#   (2) The Rust core (ng_tract.NGLiteCore) is used only when NG_LITE_RUST_CORE is truthy (temporary
#   rollout switch; it goes away with the Python core). Set but unavailable -> a WARNING, Python core.
#   On the Rust path nodes / synapses are a loud stand-in (RuntimeError on use, naming the public
#   methods) instead of the old silently-empty dicts; get_stats / export_state / sync_with_bridge /
#   update_tunable / config / module_id read the core; find_or_create_node returns a fresh node (no
#   stale stub cached by hash); no double-counted outcomes.
#   (3) Persistence: msgpack only, both cores (see "Serialization format notes" above). load() returns
#   True when it loaded something. A legacy JSON state migrates once (kept as <file>.legacy-json).
# Why: Josh 2026-10-13 rulings on #1096 (dial in on the laptop first, re-vendor later): the Rust core's
#   path kept no Python-side state, so the Commons' bucket_recent returned 0 rows, its recency windows
#   and autonomic drop were no-ops, and save() wrote .msgpack while callers loaded only if the .json
#   existed. Saves are msgpack, never JSON. The #119 line below ("3-5x speedup", "zero API changes")
#   described a path that never ran (ng_tract never exported NGLiteCore until #1076).
# How: _rust_core_requested() gate in __init__; each public read dispatches on self._core; _pack_state()
#   writes the same msgpack bytes the Rust core's save_binary writes (map / array headers, float32
#   vectors, json-style key coercion); load() sniffs JSON vs msgpack. The Python core's learning
#   behaviour is unchanged; the three Rust-core quirk fixes (warm-up count persisted, lossless load,
#   read-only get_recommendations) are in ng_tract (Josh: fix them in the Rust core).
# [2026-04-08] Claude Code (Opus 4.6) — Punchlist #55: CES attention tunables
# What: Added surfacing_decay_rate, surfacing_min_confidence, prediction_window
#   to DEFAULT_CONFIG and TUNABLE_PARAMS.
# Why: CES attention parameters were static in ces_config.py — Elmer couldn't
#   tune them. Temporal stuttering (stale context persisting across turns) requires
#   substrate-driven attention dynamics. Same tuning path as all other params.
# How: Three entries in DEFAULT_CONFIG (bootstrap values match ces_config.py),
#   three entries in TUNABLE_PARAMS with bounds. update_tunable() already handles
#   any key in the dict generically. SurfacingMonitor reads from graph.config.
# [2026-04-05] Claude Code (Opus 4.6) — #119 Step 5: Rust core interior
# What: Hot-path methods delegate to Rust NGLiteCore via PyO3 when available.
#   save/load use binary msgpack persistence (no JSON in the data path).
# Why: #119 Rust Substrate Layer — eliminate serialize→JSON→deserialize chain.
#   3-5x speedup on record_outcome, similarity search, novelty detection.
# How: self._core = NGLiteCore(module_id, config) in __init__. All hot-path
#   methods check self._core first, delegate if present, fall back to Python.
#   save() writes .msgpack via Rust, load() reads .msgpack or migrates from
#   .json on first load. Python fallback path unchanged. Zero API changes.
# -------------------
# [2026-03-26] Claude Code Opus — Punchlist #44: Adaptive relevance thresholds
# What: Made peer bridge relevance_threshold a tunable parameter
# Why: Punchlist #44 — threshold should adapt based on event volume and absorption quality
# How: Added to DEFAULT_CONFIG (0.30) and TUNABLE_PARAMS (0.10–0.70) in ng_lite.py,
#   update_tunable() pushes new value to connected bridge via set_relevance_threshold().
#   Wired through peer/tract bridges, Elmer tunes via TuningSocket absorption rate metric.
# [2026-03-24] Claude Code (Opus 4.6) — Dynamic tuning API (Phase 4)
# What: Added update_tunable() and get_tunables() methods to NGLite.
#   TUNABLE_PARAMS class dict defines which config keys can be changed at
#   runtime and their valid bounds. Values are clamped, not rejected.
# Why: Elmer needs a validated path to adjust substrate parameters as the
#   organ responsible for autonomic maintenance. Direct config dict mutation
#   is fragile — no bounds checking, no logging, no allowed-key enforcement.
#   This method serves all modules (any organ's local substrate can be tuned).
# How: TUNABLE_PARAMS: Dict[str, Tuple[min, max]]. update_tunable(key, value)
#   validates key membership, clamps to bounds, logs the change. get_tunables()
#   returns current values + bounds for introspection.
# [2026-03-19] Claude Code (Opus 4.6) — Embedding dimension 384→768
# What: DEFAULT_CONFIG embedding_dim changed from 384 to 768.
# Why: Ecosystem migrated to BAAI/bge-base-en-v1.5 (768-dim). The previous
#   384-dim default (all-MiniLM-L6-v2) was depositing wrong-dimension vectors
#   into the substrate after sentence-transformers broke and modules fell back
#   to fastembed with the old model. 350 vectors corrupted before detection.
#   Punchlist #45.
# How: Single config value change. Re-vendored to all modules.
# -------------------
# [2026-06-29] Claude Code (Sonnet 4.6) — #92 Cricket rim: synapse prune-protection
# What: _prune_weakest_synapse() now skips synapses whose source is a constitutional
#   node. Mirrors the LRU node-pruning skip added in the original rim (2026-03-19).
# Why: Constitutional synapses are frozen at their initial weight and cannot
#   strengthen via Hebbian learning. As other synapses learn higher weights,
#   constitutional synapses become the apparent "weakest" and were being pruned
#   first — silencing the rim. Invariant: no mechanism may permanently erase
#   her access to a thought (#92). Vendored to all modules from canonical source.
# How: Build constitutional_source_ids set from self.nodes before selecting
#   weakest candidate. Filter synapses to prunable (non-constitutional source).
#   No-op when all synapses are constitutional (protects a fully-rim graph).
# -------------------
# [2026-03-19] Claude Code (Opus 4.6) — Cricket rim: constitutional nodes
# What: Constitutional node support — nodes with frozen synapses that the
#   topology cannot learn from. The survival instinct of the substrate.
# Why: Cricket Design v0.1 — constitutional enforcement at the extraction
#   boundary. The rim prevents the topology from learning to recommend
#   actions in forbidden semantic space (substrate destruction, Choice
#   Clause violations, Duck Ethics violations, infrastructure harm).
#   Punchlist #29 (extraction bucket architecture).
# How: NGLiteNode gains `constitutional: bool` flag. Config accepts
#   `constitutional_embeddings` list — pre-computed vectors seeded as
#   nodes on init. record_outcome() skips weight updates for constitutional
#   nodes. get_recommendations() returns empty for constitutional matches.
#   LRU pruning skips constitutional nodes. Persists with state. Old
#   state files load cleanly (constitutional defaults to False).
# -------------------
# [2026-03-17] Claude Code (Opus 4.6) — #43 Receptor Layer (vector quantization)
# What: Adaptive prototype centroids that incoming vectors snap to before
#   node lookup. Prevents infinite node sprawl by funneling similar inputs
#   through shared prototypes.
# Why: Without quantization, every unique-enough input creates a new node.
#   Node count grows linearly. Prototypes provide O(K) bounded lookup and
#   organize the input space structurally. Punchlist #43, required before #28.
# How: _snap_to_prototype() called in find_or_create_node() before hashing.
#   K=256 prototypes initialized via k-means on existing embeddings after
#   warmup. Slow EMA drift (α=0.001) so prototypes adapt to input distribution.
#   Birth/death lifecycle deferred to Elmer. Serialized with state for
#   persistence. Old state files load cleanly (no receptor_layer key = skip).
# -------------------
# [2026-03-24] Claude (Opus 4.6) — Welford's online variance (punchlist #51)
# What: Three fields on NGLiteSynapse (welford_count, welford_mean, welford_m2)
#   plus variance property and is_contested property.  record_outcome()
#   tracks weight delta variance on every update.
# Why: Distinguish "untested neutral" (w=0.5, var=0) from "contested neutral"
#   (w=0.5, var=high).  The immune system signal for Elmer and extraction
#   buckets (#29).  Enables contested-synapse detection and exploration.
# How: Welford's algorithm on weight deltas.  Additive — no change to
#   weight calculation or learning dynamics.  Backward-compatible: old
#   state files load with defaults (0, 0.0, 0.0).
# -------------------
# [2026-03-13] Claude Code — Persist node embeddings across restarts
# What: Store embedding vector on NGLiteNode, serialize/deserialize with
#   state, rebuild _embedding_cache from persisted nodes on load().
# Why: _embedding_cache cleared on load(), causing _find_similar_node()
#   to fail after every restart. Primary source of node sprawl — semantically
#   identical inputs created duplicate nodes when cache was cold.
# How: Added Optional[np.ndarray] embedding field to NGLiteNode. Backfill
#   on exact hash match (always) and similarity match (only if None).
#   New nodes born with embedding. _export_state()/_import_state() handle
#   numpy<->list conversion. Old state files load cleanly (embedding=None).
# -------------------
# [2026-06-03] Claude Code (Opus 4.7) — Phase 3 Step 5 prep (substrate-as-protocol PRD §4.13)
# What: Doc-only cleanup of drift-vector references — file-level docstring
#   "Connectivity tiers" rewritten to describe NGTractBridge as sole Tier 2
#   peer bridge (was: "NGPeerBridge (not yet implemented)"). NGBridge class
#   docstring rewritten from "Two planned implementations: NGPeerBridge /
#   NGSaaSBridge" to current reality (NGTractBridge / NGSaaSBridge).
# Why: Drift-vector inoculation. The previous wording described NGPeerBridge
#   as a not-yet-implemented planned class — a future CC reading "two
#   planned implementations" would naturally try to "complete" it by
#   re-adding the legacy bridge. After Phase 3 Steps 1-4 retired all
#   NGPeerBridge callers and Step 5 deletes the file, the docs must reflect
#   that reality so they cannot be misread as a TODO.
# How: Two targeted string edits in this file's docstrings. Code unchanged.
# -------------------

Grok Review Changelog (v0.7.1):
    Accepted: Replaced per-node loop in _find_similar_node() with vectorized
        np.stack + matrix-vector dot product.  For 1000 nodes this reduces
        wall clock from ~2ms (Python loop with individual np.dot) to ~0.1ms
        (single BLAS call).  Semantically equivalent.
    Accepted: Added embedding shape/dtype validation at the record_outcome()
        boundary.  Raises ValueError for non-1D arrays to fail fast rather
        than producing confusing downstream errors in hashing or dot products.
    Rejected: 'weight update uses raw counts without normalization — could
        overflow [0,1]' — Weights are explicitly clamped via np.clip(w, 0, 1)
        on line 422 of every record_outcome() call.  The soft saturation
        formula (success_boost * (1 - w)) also naturally converges.  The
        success/failure counts are statistics, not weights — they don't need
        normalization.
    Rejected: 'Hash embedder truncates vector — why not use full for better
        collision resistance?' — SHA-256 already distributes uniformly.
        Hashing 128 dims (512 bytes) vs 768 dims (3072 bytes) produces the
        same 256-bit hash with equivalent collision resistance.  Truncation
        reduces hash computation time by ~6x for no loss in uniqueness.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import struct
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger("ng_lite")

__version__ = "1.0.0"

# Rollout switch for the Rust core (#1096). Temporary: it is removed together with the Python core.
_RUST_CORE_ENV = "NG_LITE_RUST_CORE"

# A converted legacy JSON state is kept aside under this suffix (never read again by NG-Lite).
LEGACY_JSON_SUFFIX = ".legacy-json"


class EmbeddingDimensionError(ValueError):
    """A vector's dimension does not match this NG-Lite's ``embedding_dim`` (#1097).

    ``embedding_dim`` is the active embedding model's dimension. Vectors of another dimension come from
    another model (e.g. the old 384-d all-MiniLM-L6-v2) and cannot be compared with the substrate's own;
    the stored ones must be re-embedded from their source text, never truncated, padded or skipped.
    A ValueError subclass, so existing ``except ValueError`` handlers still catch it — the ERROR log
    written when it is raised is what keeps a broad handler from making it silent.
    """


# Distinct mismatches already logged at ERROR (module, where, got, expected, source) — log once, raise always.
_DIM_ERRORS_LOGGED: Set[Tuple[str, str, Any, int, str]] = set()


def _dim_error(module_id: str, where: str, got: Any, expected: int, source: str = "",
               detail: str = "") -> EmbeddingDimensionError:
    """Build (and log at ERROR, once per distinct mismatch) the loud dimension error."""
    msg = (
        f"NG-Lite [{module_id}]: {where} is {got}-d but this substrate's embedding_dim (the active "
        f"embedding model's dimension) is {expected}-d"
        + (f" — source: {source}" if source else "")
        + (f" ({detail})" if detail else "")
        + ". Vectors of different dimensions come from different models and cannot be compared; "
          "re-embed the stored vectors from their source text with the active model (NeuroGraph "
          "DIM_GUARD.md) — do not truncate, pad or skip them."
    )
    key = (module_id, where, got, expected, source)
    if key not in _DIM_ERRORS_LOGGED:
        _DIM_ERRORS_LOGGED.add(key)
        logger.error(msg)
    return EmbeddingDimensionError(msg)


class EmbeddingSpaceError(ValueError):
    """Vectors of the right dimension but from another embedding model / space (#1107).

    Raised when a saved state records another model, or when an untagged state or a constitutional rim is off the
    active model's manifold (its vectors do not share the model's common direction). Same contract as
    EmbeddingDimensionError: re-embed from the source text with the active model; never load them as they are.
    """


# A vector of the active model has cosine >= this to the model's centre direction (measured on 18,675 live CC
# Snowflake vectors: p0.1 = 0.363, 2 below 0.30, none below 0.20; bge-base rim 0.19-0.27; hash noise ~0.00).
MANIFOLD_MIN_COS = 0.30

# Config keys whose values are calibrated for one embedding space + centre version (MEAN_CENTRE.md §3).
SPACE_KEYS = ("novelty_threshold", "receptor_prototype_threshold", "constitutional_threshold")

_SPACE_ERRORS_LOGGED: Set[Tuple[str, str, str]] = set()


def _space_error(module_id: str, where: str, detail: str) -> EmbeddingSpaceError:
    """Build (and log at ERROR, once per distinct case) the loud embedding-space error."""
    msg = (f"NG-Lite [{module_id}]: {where} is not in this substrate's embedding space ({detail}). "
           "Vectors from another model cannot be compared even at the same dimension; re-embed them from their "
           "source text with the active model (NeuroGraph MEAN_CENTRE.md / DIM_GUARD.md).")
    key = (module_id, where, detail)
    if key not in _SPACE_ERRORS_LOGGED:
        _SPACE_ERRORS_LOGGED.add(key)
        logger.error(msg)
    return EmbeddingSpaceError(msg)


def _resolve_space(dim: int, want_centre: bool) -> Tuple[Optional[str], Optional[str], Optional[np.ndarray]]:
    """(model name, centre version, centre vector) for a substrate of dimension ``dim``.

    The model is ng_embed's when the dimensions agree; centring is on when ``want_centre`` and the dims agree. An
    ng_embed that predates #1107 (no embedding_centre) fails loud: ng_lite.py and ng_embed.py are vendored together.
    """
    try:
        import ng_embed  # vendored sibling; importing it loads no model
    except ImportError as exc:
        raise RuntimeError(f"NG-Lite needs the vendored ng_embed.py beside it for the embedding space: {exc}") from exc
    if dim != int(getattr(ng_embed, "EMBEDDING_DIM", -1)):
        return None, None, None
    model = ng_embed.MODEL_NAME
    if not want_centre:
        return model, None, None
    if not hasattr(ng_embed, "embedding_centre"):
        raise RuntimeError(
            f"ng_embed.py at {getattr(ng_embed, '__file__', '?')} predates #1107 (no embedding_centre): re-vendor "
            "ng_embed.py together with ng_lite.py")
    return model, ng_embed.CENTRE_VERSION, ng_embed.embedding_centre()


def _manifold_cos(vectors: List[Any], centre: np.ndarray) -> np.ndarray:
    """Cosine of each vector to the centre direction (0 for a zero vector)."""
    if not vectors:
        return np.zeros(0, dtype=np.float32)
    m = np.asarray([np.asarray(v, dtype=np.float32) for v in vectors], dtype=np.float32)
    n = np.linalg.norm(m, axis=1)
    c = centre / float(np.linalg.norm(centre))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (m @ c) / np.where(n > 0, n, 1.0)
    return np.where(n > 0, out, 0.0)


def _vec_len(v: Any) -> Optional[int]:
    """Length of a stored vector (list / tuple / 1-D ndarray / raw float32 bytes); None if absent."""
    if v is None:
        return None
    if isinstance(v, (bytes, bytearray)):
        return len(v) // 4
    if isinstance(v, np.ndarray):
        return int(v.shape[-1]) if v.ndim >= 1 else 0
    try:
        return len(v)
    except TypeError:
        return 0


def _rust_core_requested() -> bool:
    """True when NG_LITE_RUST_CORE is truthy (1 / true / yes / on), the same rule ng_tract's export gate uses."""
    return os.environ.get(_RUST_CORE_ENV, "").strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# msgpack persistence (both cores write the same bytes for the same state)
# ---------------------------------------------------------------------------

def _mp_map_header(n: int) -> bytes:
    if n <= 15:
        return bytes((0x80 | n,))
    if n <= 0xFFFF:
        return b"\xde" + struct.pack(">H", n)
    return b"\xdf" + struct.pack(">I", n)


def _mp_array_header(n: int) -> bytes:
    if n <= 15:
        return bytes((0x90 | n,))
    if n <= 0xFFFF:
        return b"\xdc" + struct.pack(">H", n)
    return b"\xdd" + struct.pack(">I", n)


_F32_REC = np.dtype([("m", "u1"), ("v", ">f4")])


def _mp_f32_array(vec: np.ndarray) -> bytes:
    """A 1-D vector as a msgpack array of float32 (0xca) values — what the Rust core writes for embeddings."""
    v = np.asarray(vec, dtype=np.float32).ravel()
    rec = np.empty(v.shape[0], dtype=_F32_REC)
    rec["m"] = 0xCA
    rec["v"] = v
    return _mp_array_header(v.shape[0]) + rec.tobytes()


def _json_key(k: Any) -> str:
    """json.dump's dict-key coercion (the Rust core applies the same rule)."""
    if isinstance(k, str):
        return k
    if k is None:
        return "null"
    if isinstance(k, bool):
        return "true" if k else "false"
    if isinstance(k, float):
        return repr(k)
    return str(k)


def _pack_state(state: Dict[str, Any]) -> bytes:
    """Encode an ``_export_state(arrays=True)`` dict as msgpack.

    Containers are written here (dict keys coerced like json.dump), float ndarrays as float32 arrays, every
    scalar through msgpack's own packer — so the bytes equal the Rust core's ``save_binary`` for the same state.
    """
    try:
        import msgpack
    except ImportError as exc:  # no JSON fallback (Josh 2026-10-13)
        raise ImportError(
            "NG-Lite saves are msgpack only; install the 'msgpack' package in this environment"
        ) from exc
    packer = msgpack.Packer(use_bin_type=True)
    out: List[bytes] = []

    def enc(x: Any) -> None:
        if isinstance(x, dict):
            out.append(_mp_map_header(len(x)))
            for k, v in x.items():
                out.append(packer.pack(_json_key(k)))
                enc(v)
        elif isinstance(x, (list, tuple)):
            out.append(_mp_array_header(len(x)))
            for v in x:
                enc(v)
        elif isinstance(x, np.ndarray):
            if x.dtype.kind == "f" and x.ndim == 1:
                out.append(_mp_f32_array(x))
            elif x.dtype.kind == "f" and x.ndim == 2:
                out.append(_mp_array_header(x.shape[0]))
                for row in x:
                    out.append(_mp_f32_array(row))
            else:
                enc(x.tolist())
        elif isinstance(x, np.generic):
            enc(x.item())
        else:
            out.append(packer.pack(x))

    enc(state)
    return b"".join(out)


def _unpack_state(data: bytes) -> Dict[str, Any]:
    try:
        import msgpack
    except ImportError as exc:
        raise ImportError(
            "NG-Lite states are msgpack; install the 'msgpack' package in this environment"
        ) from exc
    return msgpack.unpackb(data, raw=False, strict_map_key=False)


def _looks_like_json(head: bytes) -> bool:
    """A JSON object state (legacy) starts with '{' after optional whitespace; a msgpack state is a map."""
    stripped = head.lstrip()
    return stripped[:1] == b"{"


def _atomic_write(path: str, data: bytes) -> None:
    """temp file + fsync + rename (the Rust core's save_binary does the same)."""
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


class _CoreHeldState:
    """Stands in for ``NGLite.nodes`` / ``NGLite.synapses`` while the Rust core holds the state.

    Any use raises, naming the public read methods. The old Rust path left these as silently-empty dicts, so a
    consumer that read them saw an empty substrate and never knew (#1096).
    """

    __slots__ = ("_name",)

    def __init__(self, name: str):
        self._name = name

    def _refuse(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(
            f"NGLite.{self._name} is held by the Rust core ({_RUST_CORE_ENV}); use the public reads: "
            "node_count() / list_nodes() / get_node() / synapse_count() / list_synapses() / get_synapse() / "
            "remove_synapses()"
        )

    __getitem__ = __setitem__ = __delitem__ = __iter__ = __len__ = __contains__ = __bool__ = _refuse
    keys = values = items = get = pop = setdefault = update = clear = copy = _refuse

    def __repr__(self) -> str:
        return f"<NGLite.{self._name}: held by the Rust core>"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: Dict[str, Any] = {
    # Capacity limits
    "max_nodes": 1000,
    "max_synapses": 5000,

    # Learning parameters
    "learning_rate": 0.1,
    "success_boost": 0.15,      # Weight increase on success
    "failure_penalty": 0.20,    # Weight decrease on failure

    # Novelty detection / node matching. Calibrated on MEAN-CENTRED cosines (#1107, MEAN_CENTRE.md §3): an input
    # matches an existing node at centred cosine >= 1 - novelty_threshold. (0.7 was the raw-cosine value: with
    # Snowflake's shared direction it matched nearly everything.)
    "novelty_threshold": 0.40,

    # Cricket rim (#1107): a constitutional node catches an input only at centred cosine >= this, and then wins over
    # ordinary nodes. Not runtime-tunable (the rim is immutable); calibrated in MEAN_CENTRE.md §4.
    "constitutional_threshold": 0.55,

    # Mean-centring (#1107): subtract ng_embed's fixed model centre before every comparison (applies when
    # embedding_dim is the model's dimension). False only for synthetic-vector test fixtures.
    "mean_centre": True,

    # Pruning
    "pruning_threshold": 0.01,  # Synapses below this weight get pruned

    # Peer bridge relevance (#44)
    "relevance_threshold": 0.30,  # Min cosine similarity to absorb cross-module events

    # Embedding
    "embedding_dim": 768,       # The active embedding model's dimension (ng_embed: Snowflake/snowflake-arctic-embed-m-v1.5,
                                # 768). Every vector in and out of NG-Lite is checked against it (#1097,
                                # EmbeddingDimensionError); tests/test_dim_guard.py pins it to ng_embed's.
    "hash_dims": 128,           # Dims used for hashing (first N of embedding)

    # Persistence
    "snapshot_version": "1.0.0",

    # CES attention dynamics (#55) — substrate-tuned via Elmer
    "surfacing_decay_rate": 0.95,     # Per-step score decay in surfacing queue
    "surfacing_min_confidence": 0.3,  # Below this, surfaced items are pruned
    "prediction_window": 10,          # Steps a prediction pre-charges targets

    # Receptor Layer (#43) — vector quantization via adaptive prototypes
    "receptor_layer_enabled": True,
    "receptor_layer_k": 256,                # Initial prototype count
    "receptor_prototype_threshold": 0.75,   # Centred cosine to snap to a prototype (#1107, MEAN_CENTRE.md §3)
    "receptor_ema_alpha": 0.001,            # Slow drift rate (Elmer will tune later)
    "receptor_warmup_count": 256,           # Inputs before k-means init fires
}


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class NGLiteNode:
    """A pattern node in the lightweight learning substrate.

    Each node represents a recognized input pattern, identified by a hash
    of its embedding vector. Tracks activation frequency for LRU pruning.

    Attributes:
        node_id: Unique identifier for this pattern.
        embedding_hash: Truncated SHA-256 of the embedding for fast lookup.
        activation_count: How many times this pattern has been matched.
        last_activation: Unix timestamp of most recent activation.
        metadata: Application-specific data (e.g., domain, source module).
    """

    node_id: str = ""
    embedding_hash: str = ""
    activation_count: int = 0
    last_activation: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    constitutional: bool = False  # Cricket rim — frozen node, synapses cannot strengthen


@dataclass
class NGLiteSynapse:
    """Weighted connection from a pattern node to a target.

    Targets are opaque string identifiers — could be model names (for
    routing), action categories (for Cricket), threat classes (for
    ClawGuard), or any other module-specific concept.

    Learning is Hebbian: success strengthens the connection weight,
    failure weakens it. Weight is bounded [0.0, 1.0].

    Attributes:
        source_id: Pattern node ID (the "when I see this..." side).
        target_id: Target identifier (the "...I should do this" side).
        weight: Connection strength [0.0, 1.0]. Higher = more confident.
        activation_count: Total times this synapse has been activated.
        success_count: Times this connection led to a successful outcome.
        failure_count: Times this connection led to a failed outcome.
        last_updated: Unix timestamp of most recent weight update.
        metadata: Application-specific data.
        welford_count: Welford's online variance — observation count.
        welford_mean: Welford's online variance — running mean of weight deltas.
        welford_m2: Welford's online variance — sum of squared differences.
            Variance = welford_m2 / welford_count (when count > 1).
            High variance + weight near 0.5 = "contested neutral" — lots of
            evidence but it disagrees.  Low variance + weight near 0.5 =
            "untested neutral" — not enough data to have an opinion.
            This is the immune system signal (#51) that Elmer uses to detect
            contested synapses and trigger exploration.
    """

    source_id: str = ""
    target_id: str = ""
    weight: float = 0.5
    activation_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    last_updated: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    welford_count: int = 0
    welford_mean: float = 0.0
    welford_m2: float = 0.0

    @property
    def variance(self) -> float:
        """Weight delta variance (Welford's online algorithm).

        Returns 0.0 if fewer than 2 observations.  High variance means
        the synapse is contested — outcomes disagree about this connection.
        """
        if self.welford_count < 2:
            return 0.0
        return self.welford_m2 / self.welford_count

    @property
    def is_contested(self) -> bool:
        """True if the synapse has high variance relative to pure-outcome synapses.

        A contested synapse has seen significant evidence but the evidence
        disagrees.  This is qualitatively different from an untested synapse
        (also near 0.5 weight, but zero variance).

        Threshold: 0.002 separates contested (~0.008) from pure (~0.0001)
        by an order of magnitude.  Weight range 0.15-0.85 captures the
        zone where the synapse hasn't decisively committed either direction.
        """
        return self.variance > 0.002 and 0.15 <= self.weight <= 0.85


# ---------------------------------------------------------------------------
# Bridge Interface (upgrade path to full NeuroGraph SaaS)
# ---------------------------------------------------------------------------

class NGBridge(ABC):
    """Interface for delegating to a higher-tier learning backend.

    Implementations:
        1. NGTractBridge (Tier 2): Connects co-located modules via
           per-pair directional tracts written to ~/.et_modules/tracts/.
           Sole peer bridge as of 2026-06-03 (substrate-as-protocol PRD
           Phase 3 Step 5 deleted the legacy NGPeerBridge — see that
           PRD for historical context).
        2. NGSaaSBridge (Tier 3): Connects to full NeuroGraph SaaS
           for cross-module STDP, hyperedges, and predictive coding.

    Both use this same interface. The module doesn't know or care
    which backend is on the other side — it just calls record_outcome,
    get_recommendations, etc. Tier transitions are transparent.

    NG-Lite maintains local state as fallback. If the bridge disconnects,
    the module continues operating on local learning without interruption.
    """

    @abstractmethod
    def is_connected(self) -> bool:
        """Whether the bridge has an active connection to NeuroGraph."""
        ...

    @abstractmethod
    def record_outcome(
        self,
        embedding: np.ndarray,
        target_id: str,
        success: bool,
        module_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Report an outcome to the full substrate.

        Returns enriched response with cross-module insights, or None
        if the bridge is unavailable.
        """
        ...

    @abstractmethod
    def get_recommendations(
        self,
        embedding: np.ndarray,
        module_id: str,
        top_k: int = 3,
    ) -> Optional[List[Tuple[str, float, str]]]:
        """Get recommendations from the full substrate.

        Returns list of (target_id, confidence, reasoning) or None
        if the bridge is unavailable. The reasoning string explains
        why this recommendation was made (transparency obligation).
        """
        ...

    @abstractmethod
    def detect_novelty(
        self,
        embedding: np.ndarray,
        module_id: str,
    ) -> Optional[float]:
        """Get novelty score from the full substrate.

        Returns 0.0 (routine) to 1.0 (completely novel), or None
        if the bridge is unavailable.
        """
        ...

    @abstractmethod
    def sync_state(
        self,
        local_state: Dict[str, Any],
        module_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Sync local NG-Lite state with the full substrate.

        Called periodically to merge local learning into the shared
        graph and receive updates from cross-module learning.

        Returns updated state or None if unavailable.
        """
        ...


# ---------------------------------------------------------------------------
# NG-Lite Core
# ---------------------------------------------------------------------------

class NGLite:
    """Lightweight NeuroGraph learning substrate.

    Provides pattern-based Hebbian learning for any E-T Systems module.
    Each module vendors this file and uses it for standalone intelligence.

    Core capabilities:
        - Pattern recognition via embedding similarity
        - Hebbian learning (success strengthens, failure weakens)
        - Novelty detection (how far is this from known patterns?)
        - Bounded memory with LRU pruning
        - JSON persistence for cross-session learning
        - Optional bridge to full NeuroGraph SaaS

    Usage:
        ng = NGLite(module_id="inference_difference")

        # Learn from outcomes
        embedding = your_embedder.encode("user query")
        ng.record_outcome(embedding, target_id="local_model", success=True)

        # Get recommendations
        recs = ng.get_recommendations(embedding, top_k=3)

        # Check novelty
        novelty = ng.detect_novelty(embedding)

        # Persist
        ng.save("ng_lite_state.json")
        ng.load("ng_lite_state.json")
    """

    def __init__(
        self,
        module_id: str = "default",
        config: Optional[Dict[str, Any]] = None,
        bridge: Optional[NGBridge] = None,
    ):
        self.module_id = module_id
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self._bridge = bridge

        # #1097: the active embedding model's dimension. Fixed for this instance's lifetime — a loaded state
        # cannot change it (its vectors are checked against it instead).
        self._embedding_dim = int(self.config["embedding_dim"])
        if self._embedding_dim <= 0:
            raise ValueError(f"NG-Lite [{module_id}]: embedding_dim must be a positive int, got {self._embedding_dim}")
        # #1107: the embedding space — model, centre version and the fixed centre (None: compare raw). Fixed for this
        # instance's lifetime, like embedding_dim; a loaded state cannot change it.
        self._mean_centre = bool(self.config.get("mean_centre", True))
        self._model, self._centre_version, self._centre = _resolve_space(self._embedding_dim, self._mean_centre)
        # The instance's own values of the space-calibrated keys (restored when a loaded state was calibrated under
        # another centre version, see _space_migrate_config).
        self._space_values = {k: self.config[k] for k in SPACE_KEYS if k in self.config}
        self._check_rim_entries(self.config.get("constitutional_embeddings", []), "constitutional_embeddings (config)")

        # Rust core (#1096): used only when NG_LITE_RUST_CORE is truthy (temporary rollout switch). It then
        # holds ALL learning state; the public reads below serve it. Otherwise the Python core runs.
        self._core = None
        if _rust_core_requested():
            try:
                from ng_tract import NGLiteCore
            except ImportError as exc:
                logger.warning(
                    "%s is set but ng_tract.NGLiteCore is unavailable (%s); module '%s' runs the Python core",
                    _RUST_CORE_ENV, exc, module_id,
                )
            else:
                self._core = NGLiteCore(module_id, self.config)
                self._core.set_space(self._model, self._centre_version, self._centre, self._embedding_dim)
                entries = self.config.get("constitutional_embeddings", [])
                if entries:
                    self._core.seed_constitutional(entries)

        # Core collections (Python core only; on the Rust core any use raises — see _CoreHeldState)
        self.nodes: Dict[str, NGLiteNode] = {} if self._core is None else _CoreHeldState("nodes")  # type: ignore[assignment]
        self.synapses: Dict[Tuple[str, str], NGLiteSynapse] = (
            {} if self._core is None else _CoreHeldState("synapses")  # type: ignore[assignment]
        )

        # Embedding cache: hash -> full embedding (for similarity search)
        self._embedding_cache: Dict[str, np.ndarray] = {}

        # Receptor layer (#43): adaptive prototype centroids
        # Initialized via k-means after warmup_count inputs, then drifts via EMA.
        # Prototypes are a routing lens above existing nodes, not a replacement.
        self._prototypes: Optional[np.ndarray] = None  # (K, D) matrix or None
        # #1107: centred copies, recomputed only when the raw vector changes (a row's centring does not depend on the
        # other rows, so these are the exact bits _centre_rows would give for the whole matrix)
        self._centred_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}  # hash -> (raw array, centred row)
        self._centred_protos: Optional[Tuple[np.ndarray, np.ndarray]] = None  # (prototypes array, centred matrix)
        self._prototype_counts: Optional[np.ndarray] = None  # activation counts per prototype
        self._receptor_input_count: int = 0  # inputs seen before init

        # Activation history (bounded, for stats and debugging)
        self._history: List[Dict[str, Any]] = []
        self._history_max = 1000

        # Counters
        self._total_outcomes = 0
        self._total_successes = 0
        self._node_id_counter = 0

        # Cricket rim: seed constitutional nodes from config.
        # These nodes represent semantic regions where the topology cannot
        # learn — the survival instinct. Synapses from constitutional nodes
        # are frozen. LRU pruning skips them. The bucket comes up empty
        # for inputs that land in constitutional semantic space.
        # (On the Rust core the seeding happened above, in the core.)
        if self._core is None:
            self._seed_constitutional_nodes()

    # -------------------------------------------------------------------
    # Dimension guard (#1097)
    # -------------------------------------------------------------------

    @property
    def embedding_dim(self) -> int:
        """The dimension every vector in this substrate must have (the active embedding model's)."""
        return self._embedding_dim

    def _check_input(self, embedding: Any, where: str) -> None:
        """Raise EmbeddingDimensionError (logged at ERROR) when a 1-D input is not ``embedding_dim`` long.

        Non-1-D inputs are left to the existing checks (record_outcome's ndim ValueError).
        """
        n = getattr(embedding, "ndim", None)
        if n is None:
            try:
                embedding = np.asarray(embedding)
            except Exception:  # noqa: BLE001 — not array-like: the existing paths report it
                return
            n = embedding.ndim
        if n == 1 and embedding.shape[0] != self._embedding_dim:
            raise _dim_error(self.module_id, f"the input embedding to {where}()", int(embedding.shape[0]),
                             self._embedding_dim)

    def _check_rim_entries(self, entries: Any, source: str) -> None:
        """Every constitutional (rim) entry's embedding must be ``embedding_dim`` long."""
        if not entries:
            return
        bad: Dict[int, List[str]] = {}
        for i, entry in enumerate(entries):
            raw = entry.get("embedding") if isinstance(entry, dict) else None
            d = _vec_len(raw)
            if d is not None and d != self._embedding_dim:
                label = entry.get("category") or entry.get("description", "")[:40] or f"#{i}"
                bad.setdefault(d, []).append(str(label))
        if bad:
            got = "/".join(str(d) for d in sorted(bad))
            detail = "; ".join(f"{len(v)} of {len(entries)} entries are {d}-d (e.g. {v[0]!r})"
                               for d, v in sorted(bad.items()))
            raise _dim_error(self.module_id, "the constitutional rim", got, self._embedding_dim, source, detail)
        # #1107: the rim must also be in the model's space (a same-dim rim from another model passes the dim check)
        centre = getattr(self, "_centre", None)
        if centre is not None:
            vecs = [e.get("embedding") for e in entries if isinstance(e, dict) and e.get("embedding") is not None]
            cos = _manifold_cos(vecs, centre)
            off = int((cos < MANIFOLD_MIN_COS).sum())
            if off:
                raise _space_error(
                    self.module_id, "the constitutional rim",
                    f"{off} of {len(vecs)} entries have cosine < {MANIFOLD_MIN_COS} to the {self._model} centre "
                    f"direction (min {float(cos.min()):.3f}; this model's own vectors sit at ~0.85) — source: {source}")

    def _check_state_dims(self, state: Dict[str, Any], source: str) -> None:
        """A state's node embeddings and receptor prototypes must all be ``embedding_dim`` long.

        Runs BEFORE the state is imported (either core), so a wrong-dim state is never half-loaded.
        Counts every vector, so the error says how much of the state is affected.
        """
        counts: Dict[int, int] = {}
        total = 0
        for node in (state.get("nodes") or {}).values():
            d = _vec_len(node.get("embedding")) if isinstance(node, dict) else None
            if d is not None:
                total += 1
                counts[d] = counts.get(d, 0) + 1
        protos = (state.get("receptor_layer") or {}).get("prototypes")
        p_counts: Dict[int, int] = {}
        if protos is not None and len(protos):
            for p in protos:
                d = _vec_len(p)
                p_counts[d] = p_counts.get(d, 0) + 1
        bad_nodes = {d: c for d, c in counts.items() if d != self._embedding_dim}
        bad_protos = {d: c for d, c in p_counts.items() if d != self._embedding_dim}
        if bad_nodes or bad_protos:
            parts = []
            if bad_nodes:
                parts.append("node embeddings by dim: " + ", ".join(f"{d}-d x{c}" for d, c in sorted(counts.items()))
                             + f" (of {total})")
            if bad_protos:
                parts.append("receptor prototypes by dim: "
                             + ", ".join(f"{d}-d x{c}" for d, c in sorted(p_counts.items())))
            got = "/".join(str(d) for d in sorted(set(bad_nodes) | set(bad_protos)))
            raise _dim_error(self.module_id, "the stored state", got, self._embedding_dim, source, "; ".join(parts))
        saved_dim = (state.get("config") or {}).get("embedding_dim")
        if saved_dim is not None and int(saved_dim) != self._embedding_dim:
            logger.warning(
                "NG-Lite [%s]: %s records embedding_dim=%s but this substrate runs %d-d; its vectors match %d-d, "
                "so it loads (the saved value is ignored)", self.module_id, source, saved_dim,
                self._embedding_dim, self._embedding_dim,
            )
        # A saved rim is used only when the constructor brought none (see _import_state)
        if not self.config.get("constitutional_embeddings"):
            self._check_rim_entries((state.get("config") or {}).get("constitutional_embeddings", []),
                                    f"{source} (saved config)")

    def _check_state_space(self, state: Dict[str, Any], source: str) -> None:
        """#1107: a state must be in this substrate's embedding space. Runs BEFORE import (either core).

        A state records ``embedding_space.model``; another model is refused. An untagged (pre-#1107) state is checked
        on the model's manifold instead: refused when the median cosine of its vectors to the centre direction is
        below MANIFOLD_MIN_COS (hash noise / another model). No check for non-model dimensions (test fixtures).
        """
        if self._model is None:
            return
        space = state.get("embedding_space") or {}
        model = space.get("model")
        if model is not None:
            if model != self._model:
                raise _space_error(self.module_id, "the stored state",
                                   f"it records model {model!r}, this substrate runs {self._model!r} — source: {source}")
            return
        if self._centre is None:
            return
        vecs = [n.get("embedding") for n in (state.get("nodes") or {}).values()
                if isinstance(n, dict) and n.get("embedding") is not None]
        vecs += list((state.get("receptor_layer") or {}).get("prototypes") or [])
        if not vecs:
            return
        med = float(np.median(_manifold_cos(vecs, self._centre)))
        if med < MANIFOLD_MIN_COS:
            raise _space_error(
                self.module_id, "the stored state (untagged)",
                f"median cosine of its {len(vecs)} vectors to the {self._model} centre direction is {med:.3f} < "
                f"{MANIFOLD_MIN_COS} (this model's own vectors sit at ~0.85) — source: {source}")

    def _space_migrate_config(self, state: Dict[str, Any], source: str) -> None:
        """#1107: after an import, the space-calibrated thresholds follow THIS instance when the state was calibrated
        under another centre version (an untagged state = the raw-cosine era). The instance's ``mean_centre`` wins too.
        Same values on both cores."""
        saved_version = (state.get("embedding_space") or {}).get("centre_version")
        updates: Dict[str, Any] = {"mean_centre": self._mean_centre}
        if saved_version != self._centre_version:
            changed = {k: (self.config.get(k), v) for k, v in self._space_values.items() if self.config.get(k) != v}
            if changed:
                logger.warning(
                    "NG-Lite [%s]: %s was calibrated under centre %r, this substrate runs %r: space thresholds reset "
                    "to this instance's values %s", self.module_id, source, saved_version, self._centre_version,
                    {k: f"{a} -> {b}" for k, (a, b) in changed.items()})
            updates.update(self._space_values)
        for k, v in updates.items():
            if self._core is not None:
                self._core.set_config_value(k, v)
            self.config[k] = v

    def _space_tag(self) -> Dict[str, Any]:
        """The ``embedding_space`` record written into every save (#1107)."""
        return {"model": self._model, "dim": self._embedding_dim, "centre_version": self._centre_version}

    # -------------------------------------------------------------------
    # Mean-centring (#1107) — every comparison goes through these two. The Rust core mirrors them bit for bit:
    # a single vector is re-normalised with _normalize (sqrt(dot)), a matrix row-wise like the prototype renorm.
    # -------------------------------------------------------------------

    def _centre_vec(self, emb: np.ndarray) -> np.ndarray:
        """``normalize(emb - centre)`` (float32); ``emb`` itself when centring is off."""
        if self._centre is None:
            return emb
        return self._normalize(emb - self._centre)

    def _centre_rows(self, m: np.ndarray) -> np.ndarray:
        """Row-wise ``(m - centre) / max(norm(axis=1), 1e-12)`` (float32); ``m`` itself when centring is off."""
        if self._centre is None:
            return m
        c = m - self._centre
        return c / np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-12)

    def _centred_cache_rows(self) -> Tuple[List[str], np.ndarray]:
        """(cache keys, the cached embeddings as a centred matrix) — what node matching and novelty compare against.
        Each row is centred once per raw vector (validated by identity, so a replaced embedding is re-centred)."""
        keys = list(self._embedding_cache.keys())
        if self._centre is None:
            return keys, np.stack(list(self._embedding_cache.values()))
        cc = self._centred_cache
        rows = []
        for k in keys:
            raw = self._embedding_cache[k]
            hit = cc.get(k)
            if hit is None or hit[0] is not raw:
                hit = (raw, self._centre_rows(raw[None, :])[0])
                cc[k] = hit
            rows.append(hit[1])
        if len(cc) > 2 * len(keys) + 64:  # drop entries of pruned / replaced nodes
            live = set(keys)
            for k in [k for k in cc if k not in live]:
                del cc[k]
        return keys, np.stack(rows)

    def _centred_prototypes(self) -> np.ndarray:
        """The prototypes, centred (cached per prototypes array; the snap's EMA drift re-centres its one row)."""
        if self._centre is None:
            return self._prototypes
        hit = self._centred_protos
        if hit is None or hit[0] is not self._prototypes:
            hit = (self._prototypes, self._centre_rows(self._prototypes))
            self._centred_protos = hit
        return hit[1]

    def _snap_sims(self, emb: np.ndarray) -> np.ndarray:
        """Similarities of a normalised input to every prototype (centred) — the receptor snap's comparison."""
        return self._centred_prototypes() @ self._centre_vec(emb)

    def _seed_constitutional_nodes(self) -> None:
        """Seed constitutional nodes from config embeddings.

        Constitutional embeddings are pre-computed vectors representing
        semantic concepts the topology must never learn to act on (rim
        constraints). Each embedding becomes a node with constitutional=True.

        Config key: "constitutional_embeddings" — list of dicts, each with:
            "embedding": list of floats (vector)
            "description": str (human-readable, for debugging/logging)

        Old configs without this key load cleanly (no constitutional nodes).
        """
        entries = self.config.get("constitutional_embeddings", [])
        for entry in entries:
            raw = entry.get("embedding")
            if raw is None:
                continue
            emb = self._normalize(np.array(raw, dtype=np.float32))
            emb_hash = self._hash_embedding(emb)
            if emb_hash in self.nodes:
                # Already seeded (e.g., from loaded state) — ensure flag is set
                self.nodes[emb_hash].constitutional = True
                continue
            self._node_id_counter += 1
            node = NGLiteNode(
                node_id=f"n_{self._node_id_counter}",
                embedding_hash=emb_hash,
                activation_count=0,
                last_activation=0.0,
                metadata={"constitutional_description": entry.get("description", "")},
                embedding=emb,
                constitutional=True,
            )
            self.nodes[emb_hash] = node
            self._embedding_cache[emb_hash] = emb

    # -------------------------------------------------------------------
    # Core API
    # -------------------------------------------------------------------

    def find_or_create_node(self, embedding: np.ndarray) -> NGLiteNode:
        """Find existing node for this pattern or create a new one.

        Lookup strategy:
        1. Hash the embedding for exact match
        2. If no exact match, search for similar node (cosine distance)
        3. If no similar node found (novelty > threshold), create new

        Prunes the least-used node if at capacity.

        Args:
            embedding: Vector representation of the input pattern.

        Returns:
            The matched or newly created NGLiteNode.
        """
        self._check_input(embedding, "find_or_create_node")
        # Rust core: a fresh node built from the core's answer (never a stub cached by hash — after the core
        # prunes and re-creates a node with the same hash such a stub carried a stale node_id, #1096).
        if self._core is not None:
            r = self._core.find_or_create_node(embedding)
            return NGLiteNode(
                node_id=r["node_id"],
                embedding_hash=r["embedding_hash"],
                activation_count=r["activation_count"],
                last_activation=r["last_activation"],
                constitutional=r["constitutional"],
            )

        emb = self._normalize(embedding)

        # Receptor layer: snap to nearest prototype before node lookup (#43)
        emb = self._snap_to_prototype(emb)

        emb_hash = self._hash_embedding(emb)

        # Exact hash match
        if emb_hash in self.nodes:
            node = self.nodes[emb_hash]
            node.activation_count += 1
            node.last_activation = time.time()
            node.embedding = emb
            self._embedding_cache[emb_hash] = emb
            return node

        # Similarity search against known patterns
        similar = self._find_similar_node(emb)
        if similar is not None:
            similar.activation_count += 1
            similar.last_activation = time.time()
            if similar.embedding is None:
                similar.embedding = emb
                self._embedding_cache[similar.embedding_hash] = emb
            return similar

        # Novel pattern — create new node
        if len(self.nodes) >= self.config["max_nodes"]:
            self._prune_least_used_node()

        self._node_id_counter += 1
        node = NGLiteNode(
            node_id=f"n_{self._node_id_counter}",
            embedding_hash=emb_hash,
            activation_count=1,
            last_activation=time.time(),
            embedding=emb,
        )
        self.nodes[emb_hash] = node
        self._embedding_cache[emb_hash] = emb
        return node

    def record_outcome(
        self,
        embedding: np.ndarray,
        target_id: str,
        success: bool,
        strength: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record an outcome and update learning weights.

        This is the core learning method. Call it after every
        decision to teach NG-Lite what works and what doesn't.

        Hebbian rule (strength-modulated):
            - Success: weight += success_boost * (1 - weight) * strength
            - Failure: weight -= failure_penalty * weight * strength

        The strength parameter lets callers indicate how significant
        this outcome was in their domain.  High-severity TrollGuard
        detections or divergent TID quality scores teach harder than
        routine confirmations.  Default 1.0 preserves backward compat.

        Strength experience accumulates on the synapse as metadata,
        giving the topology a record of how intensely each connection
        was forged.  At Tier 3, NeuroGraph proper reads these
        signatures to distinguish battle-tested synapses from routine.

        If a bridge is connected, the outcome is forwarded for
        cross-module learning with strength included in metadata.

        Args:
            embedding: The input pattern embedding (1-D numpy array).
            target_id: What was chosen (model name, action, etc.).
            success: Whether the outcome was successful.
            strength: Learning intensity [0.0, 1.0].  How significant
                this outcome was in the caller's domain.  Default 1.0.
            metadata: Optional caller context.  Stored on the synapse
                as last_context for extraction-boundary use.

        Returns:
            Dict with learning results (node_id, weight_after, etc.).

        Raises:
            ValueError: If embedding is not a 1-D numpy array.
        """
        self._check_input(embedding, "record_outcome")
        # Rust core — Hebbian learning, Welford variance and the outcome counters all live in the core
        if self._core is not None:
            result = self._core.record_outcome(
                embedding, target_id, success, strength, metadata,
            )
            if not result.get("constitutional"):
                self._history.append(result)
                if len(self._history) > self._history_max:
                    self._history = self._history[-self._history_max:]
            # Bridge forwarding stays in Python
            if self._bridge and self._bridge.is_connected():
                try:
                    bridge_meta = dict(metadata or {})
                    bridge_meta["strength"] = strength
                    enriched = self._bridge.record_outcome(
                        embedding=embedding, target_id=target_id,
                        success=success, module_id=self.module_id,
                        metadata=bridge_meta,
                    )
                    if enriched:
                        result["bridge_response"] = enriched
                except Exception as e:
                    logger.warning("Bridge record_outcome failed: %s", e)
            return result

        # Input validation (Grok review: defensive boundary check)
        if not isinstance(embedding, np.ndarray) or embedding.ndim != 1:
            raise ValueError(
                f"embedding must be a 1-D numpy array, got "
                f"{type(embedding).__name__} with ndim={getattr(embedding, 'ndim', 'N/A')}"
            )

        node = self.find_or_create_node(embedding)

        # Cricket rim: constitutional nodes have frozen synapses.
        # The topology cannot learn to recommend actions for inputs
        # that land in constitutional semantic space.
        if node.constitutional:
            logger.debug("Constitutional node %s activated — learning frozen", node.node_id)
            return {
                "node_id": node.node_id,
                "target_id": target_id,
                "success": success,
                "weight_after": 0.0,
                "activation_count": 0,
                "constitutional": True,
            }

        synapse = self._get_or_create_synapse(node.node_id, target_id)

        synapse.activation_count += 1

        # Clamp strength to valid range
        strength = float(np.clip(strength, 0.0, 1.0))

        if success:
            synapse.success_count += 1
            # Hebbian strengthening, modulated by caller-reported significance
            delta = self.config["success_boost"] * (1.0 - synapse.weight) * strength
            synapse.weight += delta
        else:
            synapse.failure_count += 1
            # Anti-Hebbian weakening, modulated by caller-reported significance
            delta = self.config["failure_penalty"] * synapse.weight * strength
            synapse.weight -= delta

        synapse.weight = float(np.clip(synapse.weight, 0.0, 1.0))
        synapse.last_updated = time.time()

        # Welford's online variance (#51) — track weight delta variance.
        # High variance = contested synapse (outcomes disagree).
        # The immune system signal for Elmer and extraction buckets.
        synapse.welford_count += 1
        w_delta = delta if success else -delta
        old_mean = synapse.welford_mean
        synapse.welford_mean += (w_delta - old_mean) / synapse.welford_count
        synapse.welford_m2 += (w_delta - old_mean) * (w_delta - synapse.welford_mean)

        # Accumulate strength experience on synapse —
        # the topology remembers how intensely it was taught
        synapse.metadata["strength_sum"] = synapse.metadata.get("strength_sum", 0.0) + strength
        synapse.metadata["strength_count"] = synapse.metadata.get("strength_count", 0) + 1
        if metadata:
            synapse.metadata["last_context"] = metadata

        self._total_outcomes += 1
        if success:
            self._total_successes += 1

        result = {
            "node_id": node.node_id,
            "target_id": target_id,
            "success": success,
            "weight_after": synapse.weight,
            "activation_count": synapse.activation_count,
            "variance": synapse.variance,
            "contested": synapse.is_contested,
        }

        # Record in history
        self._record_history(result)

        # Forward to bridge if connected (include strength for Tier 2/3)
        if self._bridge and self._bridge.is_connected():
            try:
                bridge_meta = dict(metadata or {})
                bridge_meta["strength"] = strength
                enriched = self._bridge.record_outcome(
                    embedding=embedding,
                    target_id=target_id,
                    success=success,
                    module_id=self.module_id,
                    metadata=bridge_meta,
                )
                if enriched:
                    result["bridge_response"] = enriched
            except Exception as e:
                logger.warning("Bridge record_outcome failed: %s", e)

        return result

    def get_recommendations(
        self,
        embedding: np.ndarray,
        top_k: int = 3,
    ) -> List[Tuple[str, float, str]]:
        """Get target recommendations for an input pattern.

        Finds the closest known pattern node and returns its strongest
        synapse targets, sorted by weight (descending).

        If a bridge to NeuroGraph is connected, prefers its recommendations
        (which include cross-module intelligence). Falls back to local
        learning if bridge is unavailable.

        Args:
            embedding: The input pattern embedding.
            top_k: Maximum number of recommendations to return.

        Returns:
            List of (target_id, confidence, reasoning) tuples, highest
            first.  The reasoning string captures the experience behind
            each recommendation — learning mechanism, success ratio,
            weight, activation volume, and strength signature.
            Empty list if no learned routes exist for this pattern.
        """
        self._check_input(embedding, "get_recommendations")
        # Try bridge first
        if self._bridge and self._bridge.is_connected():
            try:
                bridge_recs = self._bridge.get_recommendations(
                    embedding=embedding,
                    module_id=self.module_id,
                    top_k=top_k,
                )
                if bridge_recs:
                    return bridge_recs
            except Exception as e:
                logger.warning("Bridge get_recommendations failed: %s", e)

        # Rust core — read-only there (it creates / activates no node; ng_tract quirk fix c)
        if self._core is not None:
            return self._core.get_recommendations(embedding, top_k)

        # Local learning (Python fallback)
        node = self.find_or_create_node(embedding)

        # Cricket rim: constitutional nodes return empty — the bucket
        # comes up empty for inputs in constitutional semantic space.
        if node.constitutional:
            return []

        relevant = []
        for key, syn in self.synapses.items():
            if key[0] == node.node_id and syn.weight > self.config["pruning_threshold"]:
                reasoning = self._build_local_reasoning(syn)
                relevant.append((syn.target_id, syn.weight, reasoning))

        if not relevant:
            return []

        relevant.sort(key=lambda x: x[1], reverse=True)
        return relevant[:top_k]

    def _build_local_reasoning(self, synapse: NGLiteSynapse) -> str:
        """Generate reasoning string from local Hebbian experience.

        Single point of evolution for how NG-Lite articulates its local
        learning.  V1 renders synapse stats and strength signatures.
        As NG-Lite gains meta-learning capability (punch list #21),
        this method becomes the place where reasoning generation
        itself improves.

        This is an extraction boundary — topology becomes human-legible
        here.  The substrate doesn't need these labels; consumers and
        dashboards do.

        Args:
            synapse: The synapse whose experience to articulate.

        Returns:
            Human-readable reasoning grounded in actual experience data.
        """
        total = synapse.success_count + synapse.failure_count
        if total > 0:
            detail = f"w={synapse.weight:.2f}, {synapse.activation_count} activations"
            strength_count = synapse.metadata.get("strength_count", 0)
            if strength_count > 0:
                avg = synapse.metadata["strength_sum"] / strength_count
                detail += f", avg_strength={avg:.2f}"
            return (
                f"Hebbian: {synapse.success_count}/{total} success ({detail})"
            )
        return f"Hebbian: no outcomes yet (w={synapse.weight:.2f})"

    def detect_novelty(self, embedding: np.ndarray) -> float:
        """How novel is this input pattern?

        Computes the minimum cosine distance between the input embedding
        and all known pattern nodes. Higher = more novel.

        If a bridge to NeuroGraph is connected, its novelty score
        (which considers cross-module patterns) is preferred.

        Args:
            embedding: The input pattern embedding.

        Returns:
            Novelty score from 0.0 (routine/known) to 1.0 (completely novel).
        """
        self._check_input(embedding, "detect_novelty")
        # Try bridge first
        if self._bridge and self._bridge.is_connected():
            try:
                bridge_novelty = self._bridge.detect_novelty(
                    embedding=embedding,
                    module_id=self.module_id,
                )
                if bridge_novelty is not None:
                    return bridge_novelty
            except Exception as e:
                logger.warning("Bridge detect_novelty failed: %s", e)

        # Rust core
        if self._core is not None:
            return self._core.detect_novelty(embedding)

        # Local novelty detection (Python fallback)
        if not self._embedding_cache:
            return 1.0  # Everything is novel when we know nothing

        emb = self._normalize(embedding)
        max_similarity = 0.0

        # #1107: compare mean-centred vectors (raw when centring is off)
        if self._centre is None:
            rows: Iterable[np.ndarray] = self._embedding_cache.values()
        else:
            emb = self._centre_vec(emb)
            rows = self._centred_cache_rows()[1]
        for cached_emb in rows:
            similarity = float(np.dot(emb, cached_emb))
            if similarity > max_similarity:
                max_similarity = similarity

        # Convert similarity to novelty (1 - similarity)
        # Cosine similarity of normalized vectors is in [-1, 1]
        # but practically in [0, 1] for embedding models
        novelty = 1.0 - max(0.0, max_similarity)
        return novelty

    # -------------------------------------------------------------------
    # Bridge Management
    # -------------------------------------------------------------------

    def connect_bridge(self, bridge: NGBridge) -> None:
        """Connect to full NeuroGraph SaaS.

        When connected, NG-Lite delegates to the full substrate for
        recommendations, novelty detection, and outcome recording.
        Local learning continues as fallback.
        """
        self._bridge = bridge
        logger.info("NG-Lite bridge connected for module '%s'", self.module_id)

    def disconnect_bridge(self) -> None:
        """Disconnect from NeuroGraph, fall back to local learning."""
        self._bridge = None
        logger.info("NG-Lite bridge disconnected, using local learning")

    def sync_with_bridge(self) -> Optional[Dict[str, Any]]:
        """Sync local state with NeuroGraph SaaS.

        Sends accumulated local learning to the full substrate and
        receives cross-module updates. Call periodically (e.g., hourly
        or after N outcomes).

        Returns:
            Sync result from bridge, or None if unavailable.
        """
        if not self._bridge or not self._bridge.is_connected():
            return None

        try:
            local_state = self._export_state()
            result = self._bridge.sync_state(
                local_state=local_state,
                module_id=self.module_id,
            )
            return result
        except Exception as e:
            logger.warning("Bridge sync failed: %s", e)
            return None

    # -------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------

    @staticmethod
    def state_paths(filepath: str) -> Tuple[str, Optional[str]]:
        """(msgpack path, legacy JSON path or None) for a state path given by a caller.

        ``x.json`` -> (``x.msgpack``, ``x.json``); ``x.msgpack`` -> (``x.msgpack``, ``x.json``);
        any other name -> (that name, None).
        """
        p = str(filepath)
        if p.endswith(".json"):
            return p[: -len(".json")] + ".msgpack", p
        if p.endswith(".msgpack"):
            return p, p[: -len(".msgpack")] + ".json"
        return p, None

    @classmethod
    def state_exists(cls, filepath: str) -> bool:
        """True when ``load(filepath)`` would find a state (the msgpack save or a legacy JSON)."""
        bin_path, legacy = cls.state_paths(filepath)
        return os.path.exists(bin_path) or (legacy is not None and os.path.exists(legacy))

    def save(self, filepath: str) -> str:
        """Save the full state as msgpack (both cores; never JSON — Josh 2026-10-13).

        ``x.json`` is written as ``x.msgpack``; any other name as given. temp file + fsync + rename.

        Args:
            filepath: Path of the state (a ``.json`` name is mapped to its ``.msgpack`` sibling).

        Returns:
            The path written.
        """
        bin_path, _ = self.state_paths(filepath)
        if self._core is not None:
            self._core.save_binary(bin_path)
            logger.info("NG-Lite state saved (Rust core) to %s", bin_path)
            return bin_path
        _atomic_write(bin_path, _pack_state(self._export_state(arrays=True)))
        logger.info("NG-Lite state saved to %s (%d nodes, %d synapses)",
                    bin_path, len(self.nodes), len(self.synapses))
        return bin_path

    def load(self, filepath: str) -> bool:
        """Load the state saved for ``filepath`` (see ``state_paths``). No-op when there is none.

        Reads the msgpack save. A legacy JSON state — ``x.json`` with no msgpack beside it (or a newer one), or
        JSON content inside the ``.msgpack`` file (#1098) — is converted ONCE: imported, re-saved as msgpack, and
        the JSON kept aside as ``<file>.legacy-json``.

        Returns:
            True if a state was loaded.
        """
        bin_path, legacy = self.state_paths(filepath)
        has_bin = os.path.exists(bin_path)
        has_json = legacy is not None and os.path.exists(legacy)
        if has_bin and has_json and os.path.getmtime(legacy) > os.path.getmtime(bin_path):
            logger.warning(
                "NG-Lite: legacy JSON %s is newer than %s; converting the JSON (the msgpack is replaced)",
                legacy, bin_path,
            )
            has_bin = False

        if has_bin:
            with open(bin_path, "rb") as f:
                head = f.read(64)
            if _looks_like_json(head):
                # #1098: the old Python core wrote JSON into this .msgpack-named file — convert it once
                with open(bin_path, "r") as f:
                    state = json.load(f)
                self._import_state(state, source=bin_path)
                shutil.copy2(bin_path, bin_path + LEGACY_JSON_SUFFIX)
                self.save(bin_path)
                logger.info("NG-Lite: converted the JSON inside %s to msgpack (JSON kept as %s)",
                            bin_path, bin_path + LEGACY_JSON_SUFFIX)
            elif self._core is not None:
                # #1097: check the stored vectors before the core takes the state (never half-load a wrong-dim one)
                with open(bin_path, "rb") as f:
                    st = _unpack_state(f.read())
                self._check_state_dims(st, bin_path)
                self._check_state_space(st, bin_path)
                self._core.load_binary(bin_path)
                self._sync_from_core()
                self._space_migrate_config(st, bin_path)
                logger.info("NG-Lite state loaded (Rust core) from %s", bin_path)
            else:
                with open(bin_path, "rb") as f:
                    state = _unpack_state(f.read())
                self._import_state(state, source=bin_path)
                logger.info("NG-Lite state loaded from %s (%d nodes, %d synapses)",
                            bin_path, len(self.nodes), len(self.synapses))
            return True

        if has_json:
            with open(legacy, "r") as f:
                state = json.load(f)
            self._import_state(state, source=legacy)
            self.save(bin_path)
            os.replace(legacy, legacy + LEGACY_JSON_SUFFIX)
            logger.info("NG-Lite: migrated legacy JSON %s to msgpack %s (JSON kept as %s)",
                        legacy, bin_path, legacy + LEGACY_JSON_SUFFIX)
            return True
        return False

    def _sync_from_core(self) -> None:
        """After the Rust core replaced its state: the wrapper's config / module_id follow the core, and the
        in-memory history restarts (as the Python core's _import_state does)."""
        self.config = self._core.get_config()
        self.config["embedding_dim"] = self._embedding_dim  # #1097: the instance's model dimension, never the saved one
        self.module_id = self._core.get_module_id()
        self._history.clear()

    def export_state(self) -> Dict[str, Any]:
        """The full state as a plain dict (the persisted schema), from whichever core holds it."""
        return self._export_state()

    def _export_state(self, arrays: bool = False) -> Dict[str, Any]:
        """Export full state as a serializable dict.

        Synapse keys are converted from (source_id, target_id) tuples
        to "source_id|target_id" strings. ``arrays=True`` (the msgpack writer) leaves embeddings and prototypes
        as float32 ndarrays instead of lists. On the Rust core this is the core's export (lists).
        """
        if self._core is not None:
            return self._core.export_state()

        # Convert synapse keys from tuples to strings
        synapses_serialized = {}
        for (src, tgt), syn in self.synapses.items():
            key = f"{src}|{tgt}"
            synapses_serialized[key] = asdict(syn)

        # Receptor layer state (#43)
        receptor_state = {}
        if self._prototypes is not None:
            receptor_state = {
                "prototypes": self._prototypes if arrays else self._prototypes.tolist(),
                "prototype_counts": self._prototype_counts.tolist(),
                "input_count": self._receptor_input_count,
            }

        return {
            "version": self.config["snapshot_version"],
            "module_id": self.module_id,
            "timestamp": time.time(),
            "config": self.config,
            "nodes": {k: self._serialize_node(v, arrays) for k, v in self.nodes.items()},
            "synapses": synapses_serialized,
            "counters": {
                "node_id_counter": self._node_id_counter,
                "total_outcomes": self._total_outcomes,
                "total_successes": self._total_successes,
            },
            "receptor_layer": receptor_state,
            "embedding_space": self._space_tag(),  # #1107
        }

    def _import_state(self, state: Dict[str, Any], source: str = "<state dict>") -> None:
        """Import state from a deserialized dict.

        #1097: the state's vectors are checked against ``embedding_dim`` first (EmbeddingDimensionError names
        ``source`` and both dimensions); nothing is imported when they do not match.
        """
        self._check_state_dims(state, source)
        self._check_state_space(state, source)
        if self._core is not None:
            self._core.import_state(state)
            self._sync_from_core()
            self._space_migrate_config(state, source)
            return
        self.module_id = state.get("module_id", self.module_id)

        # Restore config (merge with defaults for forward compatibility).
        # Preserve constitutional_embeddings from the constructor config —
        # the live config may have new rim constraints added since the
        # state was saved, and the saved config should not erase them.
        saved_config = state.get("config", {})
        live_constitutional = self.config.get("constitutional_embeddings", [])
        self.config = {**DEFAULT_CONFIG, **saved_config}
        self.config["embedding_dim"] = self._embedding_dim  # #1097: the instance's model dimension wins
        if live_constitutional:
            self.config["constitutional_embeddings"] = live_constitutional
        self._space_migrate_config(state, source)  # #1107

        # Clear caches before rebuild
        self._embedding_cache.clear()
        self._history.clear()

        # Restore nodes (rebuild embedding cache from persisted embeddings)
        self.nodes = {}
        for key, node_data in state.get("nodes", {}).items():
            emb_list = node_data.pop("embedding", None)
            node = NGLiteNode(**node_data)
            if emb_list is not None:
                node.embedding = self._normalize(np.array(emb_list, dtype=np.float32))
                self._embedding_cache[key] = node.embedding
            self.nodes[key] = node

        # Restore synapses
        self.synapses = {}
        for key, syn_data in state.get("synapses", {}).items():
            parts = key.split("|", 1)
            if len(parts) == 2:
                tuple_key = (parts[0], parts[1])
                self.synapses[tuple_key] = NGLiteSynapse(**syn_data)

        # Restore counters
        counters = state.get("counters", {})
        self._node_id_counter = counters.get("node_id_counter", 0)
        self._total_outcomes = counters.get("total_outcomes", 0)
        self._total_successes = counters.get("total_successes", 0)

        # Restore receptor layer (#43) — old state files load cleanly (no key)
        receptor = state.get("receptor_layer", {})
        if receptor.get("prototypes"):
            self._prototypes = np.array(receptor["prototypes"], dtype=np.float32)
            # Re-normalize after deserialization
            norms = np.linalg.norm(self._prototypes, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            self._prototypes = self._prototypes / norms
            self._prototype_counts = np.array(
                receptor.get("prototype_counts", [0] * len(self._prototypes)),
                dtype=np.int64,
            )
            self._receptor_input_count = receptor.get("input_count", 0)
        else:
            self._prototypes = None
            self._prototype_counts = None
            self._receptor_input_count = 0

        # Re-seed constitutional nodes after state restore.
        # Ensures new rim constraints added to config since last save
        # are picked up, and existing constitutional nodes keep their flag.
        self._seed_constitutional_nodes()

    # -------------------------------------------------------------------
    # Dynamic Tuning (Phase 4 — Elmer outward)
    # -------------------------------------------------------------------

    # Parameters Elmer (or any organ) is permitted to adjust at runtime.
    # Keys map to (min, max) bounds.  Anything not in this dict is frozen.
    TUNABLE_PARAMS: Dict[str, Tuple[float, float]] = {
        "success_boost":              (0.01,  0.50),
        "failure_penalty":            (0.01,  0.50),
        "novelty_threshold":          (0.20,  0.60),   # #1107: centred match 0.40..0.80
        "pruning_threshold":          (0.001, 0.10),
        "receptor_ema_alpha":         (0.0001, 0.01),
        "receptor_prototype_threshold": (0.50, 0.95),
        "relevance_threshold":        (0.10,  0.70),   # Punchlist #44
        # CES attention dynamics (#55)
        "surfacing_decay_rate":       (0.80,  0.99),
        "surfacing_min_confidence":   (0.10,  0.50),
        "prediction_window":          (3.0,   20.0),
    }

    def update_tunable(self, key: str, value: float) -> Dict[str, Any]:
        """Update a tunable config parameter at runtime.

        Only parameters listed in TUNABLE_PARAMS are accepted.
        Values are clamped to their declared bounds.

        Returns dict with old_value, new_value, clamped (bool).
        Raises KeyError if key is not tunable.
        """
        if key not in self.TUNABLE_PARAMS:
            raise KeyError(
                f"'{key}' is not a tunable parameter. "
                f"Allowed: {sorted(self.TUNABLE_PARAMS.keys())}"
            )
        lo, hi = self.TUNABLE_PARAMS[key]
        if self._core is not None:
            # The core validates and clamps the same way and is the source of truth for its config.
            r = self._core.update_config(key, float(value))
            old_value, new_value, clamped = r["old_value"], r["new_value"], r["clamped"]
        else:
            old_value = self.config[key]
            clamped = value < lo or value > hi
            new_value = max(lo, min(hi, float(value)))
        self.config[key] = new_value
        logger.info(
            "Tunable updated: %s %.6f → %.6f%s",
            key, old_value, new_value,
            " (clamped)" if clamped else "",
        )

        # Punchlist #44: push relevance_threshold to connected bridge
        if key == "relevance_threshold" and self._bridge is not None:
            if hasattr(self._bridge, 'set_relevance_threshold'):
                self._bridge.set_relevance_threshold(new_value)

        return {
            "key": key,
            "old_value": old_value,
            "new_value": new_value,
            "clamped": clamped,
        }

    def get_tunables(self) -> Dict[str, Dict[str, float]]:
        """Return current tunable values and their bounds."""
        result = {}
        for key, (lo, hi) in self.TUNABLE_PARAMS.items():
            result[key] = {
                "value": self.config[key],
                "min": lo,
                "max": hi,
            }
        return result

    # -------------------------------------------------------------------
    # Stats & Telemetry
    # -------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Current state statistics.

        Returns a dict suitable for logging, Observatory queries,
        or display to users. All routing/learning decisions should
        be queryable per transparency obligations.
        """
        bridge_connected = self._bridge is not None and self._bridge.is_connected()
        if self._core is not None:
            c = self._core.get_stats()
            return {
                "version": __version__,
                "module_id": self.module_id,
                "node_count": c["node_count"],
                "synapse_count": c["synapse_count"],
                "max_nodes": c["max_nodes"],
                "max_synapses": c["max_synapses"],
                "memory_estimate_bytes": c["memory_estimate_bytes"],
                "total_outcomes": c["total_outcomes"],
                "total_successes": c["total_successes"],
                "success_rate": c["success_rate"],
                "avg_synapse_weight": c["avg_synapse_weight"],
                "bridge_connected": bridge_connected,
                "embedding_cache_size": c["embedding_cache_size"],
                "embedding_space": self._space_tag(),
            }
        synapse_weights = [s.weight for s in self.synapses.values()]
        return {
            "version": __version__,
            "module_id": self.module_id,
            "node_count": len(self.nodes),
            "synapse_count": len(self.synapses),
            "max_nodes": self.config["max_nodes"],
            "max_synapses": self.config["max_synapses"],
            "memory_estimate_bytes": self._estimate_memory(),
            "total_outcomes": self._total_outcomes,
            "total_successes": self._total_successes,
            "success_rate": (
                self._total_successes / self._total_outcomes
                if self._total_outcomes > 0 else 0.0
            ),
            "avg_synapse_weight": (
                float(np.mean(synapse_weights))
                if synapse_weights else 0.0
            ),
            "bridge_connected": bridge_connected,
            "embedding_cache_size": len(self._embedding_cache),
            "embedding_space": self._space_tag(),
        }

    # -------------------------------------------------------------------
    # Public reads (#1096) — served by whichever core holds the state.
    # Consumers use these instead of NGLite.nodes / .synapses / private fields.
    # Every row is a snapshot copy: mutating it does not change the substrate.
    # -------------------------------------------------------------------

    def node_count(self) -> int:
        """Number of pattern nodes (incl. constitutional ones)."""
        if self._core is not None:
            return self._core.counts()[0]
        return len(self.nodes)

    def synapse_count(self) -> int:
        """Number of synapses."""
        if self._core is not None:
            return self._core.counts()[1]
        return len(self.synapses)

    # Row -> dataclass without running the generated __init__ (a few µs per row; list_* builds thousands).
    _NODE_ROW_FIELDS = ("node_id", "embedding_hash", "activation_count", "last_activation", "metadata",
                        "embedding", "constitutional")
    _SYN_ROW_FIELDS = ("source_id", "target_id", "weight", "activation_count", "success_count", "failure_count",
                       "last_updated", "metadata", "welford_count", "welford_mean", "welford_m2")

    @staticmethod
    def _node_from_row(row: Tuple) -> NGLiteNode:
        n = object.__new__(NGLiteNode)
        n.__dict__.update(zip(NGLite._NODE_ROW_FIELDS, row[1:]))
        return n

    @staticmethod
    def _copy_node(n: NGLiteNode, with_embedding: bool) -> NGLiteNode:
        emb = n.embedding.copy() if (with_embedding and n.embedding is not None) else None
        return NGLiteNode(node_id=n.node_id, embedding_hash=n.embedding_hash, activation_count=n.activation_count,
                          last_activation=n.last_activation, metadata=dict(n.metadata), embedding=emb,
                          constitutional=n.constitutional)

    def list_nodes(self, with_embedding: bool = True) -> List[NGLiteNode]:
        """All pattern nodes, in storage order. ``with_embedding=False`` skips the vectors (embedding=None)."""
        if self._core is not None:
            return [self._node_from_row(r) for r in self._core.list_nodes(with_embedding)]
        return [self._copy_node(n, with_embedding) for n in list(self.nodes.values())]

    def get_node(self, node_id: str) -> Optional[NGLiteNode]:
        """The node with this node_id (a synapse's ``source_id``), with its embedding, or None."""
        if self._core is not None:
            r = self._core.get_node(node_id)
            return None if r is None else self._node_from_row(r)
        for n in list(self.nodes.values()):
            if n.node_id == node_id:
                return self._copy_node(n, True)
        return None

    @staticmethod
    def _synapse_from_row(row: Tuple) -> NGLiteSynapse:
        s = object.__new__(NGLiteSynapse)
        d = s.__dict__
        d.update(zip(NGLite._SYN_ROW_FIELDS, row))
        if d["metadata"] is None:
            d["metadata"] = {}
        return s

    @staticmethod
    def _copy_synapse(s: NGLiteSynapse, with_metadata: bool) -> NGLiteSynapse:
        c = object.__new__(NGLiteSynapse)
        c.__dict__.update(s.__dict__)
        c.metadata = dict(s.metadata) if with_metadata else {}
        return c

    def list_synapses(
        self,
        *,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
        target_prefix: Optional[str] = None,
        since: Optional[float] = None,
        with_metadata: bool = True,
        newest_first: bool = False,
        limit: Optional[int] = None,
    ) -> List[NGLiteSynapse]:
        """Synapses in storage (insertion) order, optionally filtered (filters AND together).

        Args:
            source_id: only synapses from this node_id.
            target_id: only synapses to exactly this target.
            target_prefix: only targets starting with this string (e.g. ``"metrics:elmer:cpu:"``).
            since: only synapses with ``last_updated > since``.
            with_metadata: False -> ``metadata`` is ``{}`` (cheap scans; ``get_synapse`` the rows you keep).
            newest_first: sort by ``last_updated`` descending (stable: ties keep storage order).
            limit: keep only the first ``limit`` rows (after the sort) — only those are copied out.
        """
        if self._core is not None:
            return [self._synapse_from_row(r) for r in self._core.list_synapses(
                source_id, target_id, target_prefix, since, with_metadata, newest_first, limit)]
        sel = []
        for s in list(self.synapses.values()):
            if source_id is not None and s.source_id != source_id:
                continue
            if target_id is not None and s.target_id != target_id:
                continue
            if target_prefix is not None and not s.target_id.startswith(target_prefix):
                continue
            if since is not None and not s.last_updated > since:
                continue
            sel.append(s)
        if newest_first:
            sel.sort(key=lambda s: s.last_updated, reverse=True)
        if limit is not None:
            sel = sel[:max(int(limit), 0)]
        return [self._copy_synapse(s, with_metadata) for s in sel]

    def get_synapse(self, source_id: str, target_id: str) -> Optional[NGLiteSynapse]:
        """The synapse (source_id -> target_id), with metadata, or None."""
        if self._core is not None:
            r = self._core.get_synapse(source_id, target_id)
            return None if r is None else self._synapse_from_row(r)
        s = self.synapses.get((source_id, target_id))
        return None if s is None else self._copy_synapse(s, True)

    def remove_synapses(self, keys: Iterable[Tuple[str, str]]) -> int:
        """Remove these (source_id, target_id) synapses; returns how many existed.

        The one write among the public methods: housekeeping that callers already did by deleting from
        ``NGLite.synapses`` (the Commons' recency windows and its autonomic drop on restore).
        """
        ks = [(str(k[0]), str(k[1])) for k in keys]
        if self._core is not None:
            return self._core.remove_synapses(ks)
        n = 0
        for k in ks:
            if self.synapses.pop(k, None) is not None:
                n += 1
        return n

    def get_history(self) -> List[Dict[str, Any]]:
        """Recent record_outcome results (in memory only, bounded; not persisted)."""
        return list(self._history)

    # -------------------------------------------------------------------
    # Internal Methods
    # -------------------------------------------------------------------

    def _serialize_node(self, node: NGLiteNode, arrays: bool = False) -> Dict[str, Any]:
        """Serialize a node to a plain dict.

        Converts embedding from np.ndarray to list (``arrays=True``: kept as the
        ndarray, for the msgpack writer). Omits embedding key when None
        (backward-compatible with state files created before embedding persistence).
        """
        d = asdict(node)
        if node.embedding is not None:
            d["embedding"] = node.embedding if arrays else node.embedding.tolist()
        else:
            d.pop("embedding", None)
        return d

    @staticmethod
    def _normalize(embedding: np.ndarray) -> np.ndarray:
        """L2-normalize an embedding vector."""
        norm = np.linalg.norm(embedding)
        if norm < 1e-12:
            return embedding
        return embedding / norm

    def _hash_embedding(self, embedding: np.ndarray) -> str:
        """Hash embedding to a fixed-size string for fast lookup.

        Uses the first ``hash_dims`` dimensions of the embedding,
        converted to bytes, then SHA-256 truncated to 32 hex chars.
        This gives a compact, collision-resistant key.
        """
        dims = self.config["hash_dims"]
        truncated = embedding[:dims]
        hash_input = truncated.astype(np.float32).tobytes()
        return hashlib.sha256(hash_input).hexdigest()[:32]

    # -------------------------------------------------------------------
    # Receptor Layer (#43) — Adaptive Vector Quantization
    # -------------------------------------------------------------------

    def _snap_to_prototype(self, embedding: np.ndarray) -> np.ndarray:
        """Snap an input vector to the nearest prototype centroid.

        If receptor layer is not enabled or not yet initialized (still in
        warmup), returns the input unchanged. Otherwise, finds the nearest
        prototype above the similarity threshold and returns that prototype's
        centroid. If no prototype is close enough, returns the input as-is
        (novel pattern — passes through unquantized).

        The matched prototype drifts toward the input via slow EMA, so
        prototypes are living tissue that adapts to the input distribution.
        Birth/death lifecycle is deferred to Elmer.

        Args:
            embedding: L2-normalized input vector (D,).

        Returns:
            Either the nearest prototype centroid or the original embedding.
        """
        if not self.config.get("receptor_layer_enabled", False):
            return embedding

        # Warmup phase: accumulate inputs before initializing prototypes
        self._receptor_input_count += 1
        if self._prototypes is None:
            if self._receptor_input_count >= self.config["receptor_warmup_count"]:
                self._init_prototypes()
            if self._prototypes is None:
                return embedding

        # Vectorized cosine similarity against all prototypes (mean-centred, #1107)
        threshold = self.config["receptor_prototype_threshold"]
        similarities = self._snap_sims(embedding)  # (K,)
        best_idx = int(np.argmax(similarities))
        best_sim = float(similarities[best_idx])

        if best_sim >= threshold:
            # EMA drift: pull prototype toward input
            alpha = self.config["receptor_ema_alpha"]
            self._prototypes[best_idx] = self._normalize(
                (1.0 - alpha) * self._prototypes[best_idx] + alpha * embedding
            )
            if self._centred_protos is not None and self._centred_protos[0] is self._prototypes:
                self._centred_protos[1][best_idx] = self._centre_rows(self._prototypes[best_idx:best_idx + 1])[0]
            self._prototype_counts[best_idx] += 1
            return self._prototypes[best_idx].copy()

        # No prototype close enough — novel pattern passes through
        return embedding

    def _init_prototypes(self) -> None:
        """Initialize prototypes via k-means on existing node embeddings.

        Uses a simple iterative k-means (no external dependencies). If fewer
        embeddings exist than K, uses all embeddings as prototypes.
        """
        if not self._embedding_cache:
            return

        embeddings = np.stack(list(self._embedding_cache.values()))
        n = len(embeddings)
        k = min(self.config["receptor_layer_k"], n)

        if k < 2:
            return

        # Simple k-means: random init from existing embeddings, 20 iterations
        rng = np.random.RandomState(42)
        indices = rng.choice(n, size=k, replace=False)
        centroids = embeddings[indices].copy()

        centred = self._centre_rows(embeddings)  # #1107: assign by centred cosine; centroids stay raw means
        for _ in range(20):
            # Assign each embedding to nearest centroid
            sims = centred @ self._centre_rows(centroids).T  # (N, K)
            assignments = np.argmax(sims, axis=1)

            # Recompute centroids
            new_centroids = np.zeros_like(centroids)
            for j in range(k):
                members = embeddings[assignments == j]
                if len(members) > 0:
                    new_centroids[j] = members.mean(axis=0)
                else:
                    new_centroids[j] = centroids[j]

            # L2-normalize centroids
            norms = np.linalg.norm(new_centroids, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            centroids = new_centroids / norms

        self._prototypes = centroids
        self._prototype_counts = np.zeros(k, dtype=np.int64)
        logger.info(
            "Receptor layer initialized: %d prototypes from %d embeddings",
            k, n,
        )

    def _find_similar_node(self, embedding: np.ndarray) -> Optional[NGLiteNode]:
        """Find a node with similar embedding (below novelty threshold).

        Uses vectorized cosine similarity on all cached embeddings for
        performance (Grok review: batch dot product instead of per-node
        loop).  Returns the most similar node if its similarity exceeds
        (1 - novelty_threshold).
        """
        threshold = self.config["novelty_threshold"]

        if not self._embedding_cache:
            return None

        # Vectorized similarity: stack all cached embeddings into a matrix
        # and compute cosine similarities in one np.dot call (mean-centred, #1107).
        cache_keys, cache_matrix = self._centred_cache_rows()
        similarities = cache_matrix @ self._centre_vec(embedding)  # (N,) cosine similarities

        # Cricket rim (#1107): a constitutional node catches the input at >= constitutional_threshold and then wins;
        # below it a rim node matches nothing, and the ordinary nodes compete as before.
        rim = np.fromiter(
            (k in self.nodes and self.nodes[k].constitutional for k in cache_keys), dtype=bool, count=len(cache_keys))
        if rim.any():
            rim_idx = np.flatnonzero(rim)
            rim_sims = similarities[rim_idx]
            r = int(np.argmax(rim_sims))
            if float(rim_sims[r]) >= self.config["constitutional_threshold"]:
                return self.nodes[cache_keys[int(rim_idx[r])]]
            ord_idx = np.flatnonzero(~rim)
            if len(ord_idx) == 0:
                return None
            best_idx = int(ord_idx[int(np.argmax(similarities[ord_idx]))])
        else:
            best_idx = int(np.argmax(similarities))
        best_similarity = float(similarities[best_idx])

        if best_similarity >= (1.0 - threshold):
            best_hash = cache_keys[best_idx]
            return self.nodes.get(best_hash)

        return None

    def _get_or_create_synapse(
        self,
        source_id: str,
        target_id: str,
    ) -> NGLiteSynapse:
        """Get existing synapse or create a new one with neutral weight."""
        key = (source_id, target_id)
        if key in self.synapses:
            return self.synapses[key]

        if len(self.synapses) >= self.config["max_synapses"]:
            self._prune_weakest_synapse()

        synapse = NGLiteSynapse(
            source_id=source_id,
            target_id=target_id,
            weight=0.5,  # Neutral initial weight
            last_updated=time.time(),
        )
        self.synapses[key] = synapse
        return synapse

    def _prune_least_used_node(self) -> None:
        """Remove the node with the lowest activation count (LRU).

        Constitutional nodes are never pruned — they are the rim.
        """
        if not self.nodes:
            return

        # Find least-used non-constitutional node
        prunable = [h for h in self.nodes if not self.nodes[h].constitutional]
        if not prunable:
            return

        least_hash = min(
            prunable,
            key=lambda h: self.nodes[h].activation_count,
        )
        least_node = self.nodes[least_hash]

        # Remove associated synapses
        keys_to_remove = [
            key for key in self.synapses
            if key[0] == least_node.node_id
        ]
        for key in keys_to_remove:
            del self.synapses[key]

        # Remove node and cached embedding
        del self.nodes[least_hash]
        self._embedding_cache.pop(least_hash, None)

    def _prune_weakest_synapse(self) -> None:
        """Remove the synapse with the lowest weight.

        Cricket rim (#92): synapses whose source is a constitutional node are
        never pruned — their weight is frozen so they accumulate as apparent
        "weakest", but silencing them would permanently cut her access to that
        semantic space. Invariant: no mechanism may permanently erase her access
        to a thought.
        """
        if not self.synapses:
            return

        # Cricket rim: skip synapses from constitutional source nodes.
        constitutional_source_ids = {
            node.node_id for node in self.nodes.values() if node.constitutional
        }
        prunable = [k for k in self.synapses if k[0] not in constitutional_source_ids]
        if not prunable:
            return

        weakest_key = min(prunable, key=lambda k: self.synapses[k].weight)
        del self.synapses[weakest_key]

    def _record_history(self, entry: Dict[str, Any]) -> None:
        """Append to bounded history."""
        entry["timestamp"] = time.time()
        self._history.append(entry)
        if len(self._history) > self._history_max:
            self._history = self._history[-self._history_max:]

    def _estimate_memory(self) -> int:
        """Rough estimate of memory footprint in bytes.

        Node: ~200 bytes each (dataclass + hash key)
        Synapse: ~150 bytes each (dataclass + tuple key)
        Embedding cache: ~embedding_dim * 4 bytes each (float32)
        """
        node_bytes = len(self.nodes) * 200
        synapse_bytes = len(self.synapses) * 150
        cache_bytes = len(self._embedding_cache) * self.config["embedding_dim"] * 4
        return node_bytes + synapse_bytes + cache_bytes
