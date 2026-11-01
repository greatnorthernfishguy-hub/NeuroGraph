"""
ng_embed — Centralized embedding service for the E-T Systems ecosystem.

Singleton embedding engine used by every module. Provides:
  1. Unified embedding via Snowflake/snowflake-arctic-embed-m-v1.5 (ONNX)
  2. Dual-pass embedding (forest + trees) via TID concept extraction
  3. Thread-safe singleton — one model instance per process
  4. Fail-closed: raise EmbeddingUnavailableError when the model is unavailable

This is a VENDORED file. Canonical source: ~/NeuroGraph/ng_embed.py
Do NOT modify vendored copies. Changes made here, re-vendored everywhere.

Model: Snowflake/snowflake-arctic-embed-m-v1.5
  - 768-dim, CLS pooling, standard BERT architecture
  - Query prefix: "Represent this sentence for searching relevant passages: "
  - Documents: no prefix
  - ONNX quantized (~110MB) via onnxruntime — no PyTorch dependency

Dual-pass (Punchlist #81 — Josh's invention):
  Pass 1 (Forest): Gestalt embedding of whole content. One node.
  Pass 2 (Trees): LLM extracts concepts via TID. Each concept embedded
  separately. Each tree linked to its forest via synapses. Cross-document
  tree links form naturally through similarity association.

# ---- Changelog ----
# [2026-10-23] Claude Code (Opus 5.5, laptop; #920 / #922, Josh "go ahead" 2026-10-23; P503: trying is allowed, NO re-vendor)
#   What: (1) NEW NGEmbed.prepare_dual_record(content): the pure COMPUTE half of dual_record_outcome (TID concept
#         extraction + the tree embeddings), writes nothing, takes no lock. (2) dual_record_outcome(..., prepared=)
#         keyword-only: with a prepare_dual_record() result it skips the extraction and the tree embedding and only
#         WRITES; prepared=None (default) is the old single call, unchanged. (3) _tid_extract_window remembers the
#         exception CLASS of the last failed TID call (self._last_failure_class; stats["dual_pass"]["last_failure_class"])
#         and the rate-limited R3 WARNING names it (class name only, never str(exc)).
#   Why:  #920/P497(4): the slow TID call ran under the CC daemon's _concurrent_lock (one call per 2000-char window,
#         up to tid_timeout=30 s each), so a slow TID made hooks and the Tonic wait. Compute outside, commit inside.
#         And the poison-turn diagnosis (2026-10-23) found the failing calls invisible: requests gives up at
#         tid_timeout, uvicorn then drops TID's access line for the disconnected client, and the class was only at debug.
#   How:  R3 is unchanged: a None extraction still raises DualPassIncompleteError before ANY write, on both paths.
#         Canonical change on the NG review branch cc-laptop-poison-920-20261023 only; NOT re-vendored anywhere.
# [2026-10-15] Claude Code (Opus 5.5, Executive; Josh "#1115 is good") — #1115: embeddings are batch-invariant
#   What: _onnx_embed_ids_batch runs each sequence through the ONNX session on its own (shape [1, len]).
#   Why:  the int8 dynamically-quantized model computes activation scales over the whole batch tensor, so the
#         same text came out at 0.977-0.992 cosine depending on its batch-mates and their padding. "Same text ->
#         same vector -> same node" did not hold across batches (Commons, dual-pass trees, windows, hash nodes).
#   How:  per-sequence session.run with the batch padding stripped (masked pad positions still widen the
#         quantizer's range); results equal the existing single-text path, so stored vectors, the
#         mean-centring vector and the thresholds stay valid. ~1.3x slower per batch on the laptop CPU.
# [2026-10-14] Claude Code (Opus 5.5, lane mean-centre) — #1107: the model's fixed mean-centring vector
# What: CENTRE_VERSION / CENTRE_SHA256 and embedding_centre(): a 768-d float32 vector shipped inside this file (base64,
#   sha256-checked on first use, returned read-only). It is the mean of 18,675 L2-normalised document-mode Snowflake
#   vectors (the CC vector store, 2026-10-14 copy: 3,026 whole deposits + 15,649 dual-pass concepts = the mix NG-Lite's
#   dual-pass receives); |mu| = 0.802. Reproduce / verify with compute_embedding_centre.py (NG-only tool).
# Why: Josh 2026-10-14 "Mean center" (punch list #1107). Snowflake vectors share a common direction: two unrelated
#   texts sit at ~0.69 cosine, so a raw-cosine threshold cannot separate related from unrelated. NG-Lite subtracts this
#   vector (and re-normalises) before every similarity comparison. It is a CONSTANT, versioned with the model: never
#   recomputed at runtime (a drifting centre would silently change every similarity in every module). A new model or a
#   new centre = a new CENTRE_VERSION, and NG-Lite states record the version they were calibrated under.
# How: Constants after MODEL_NAME / EMBEDDING_DIM; embedding_centre() decodes once and caches. No other behaviour change.
# [2026-10-14] Claude Code (Opus 5.5, lane dim-guard) — #1097: public MODEL_NAME / EMBEDDING_DIM
# What: Module constants MODEL_NAME and EMBEDDING_DIM, read from _DEFAULT_CONFIG (no behaviour change).
# Why: The active model's identity and dimension had no public name. cc_topology_export/merge look up
#   ng_embed.MODEL_NAME, which did not exist, so both sides read "unknown" and the cross-machine model check
#   never fired; NG-Lite and the vector store need the dimension to fail loud on wrong-dim vectors
#   (tests/test_dim_guard.py pins ng_lite.DEFAULT_CONFIG["embedding_dim"] to EMBEDDING_DIM).
# How: Two assignments after _DEFAULT_CONFIG.
# -------------------
# [2026-09-26] openrouter/deepseek/deepseek-v4.1-flash (OpenCode harness on T3 Code),
#   lane z2-ngembed-cap-roundrobin-20260926 — #666.
# What: _extract_concepts now builds its concept union by round-robin interleave
#   across windows (for rank i, each window contributes its i-th concept in
#   window order; dedup; first-seen order within a window) before applying the
#   unchanged max_concepts cap.
# Why:  the prior positional union let the head window fill the cap and silently
#   drop later windows' concepts (#666, Exec P267(2)).
# How:  collect the per-window lists first (still fail-fast None), then
#   round-robin; max_concepts stays 20, _DEFAULT_CONFIG untouched.
# -------------------
# [2026-09-26] openrouter/deepseek/deepseek-v4.1-flash (OpenCode harness on T3 Code),
#   lane z2-dualpass-reconcile-20260926 — P264(2) Task B, Z11 note.
# What: the NGEmbed.__init__ comment block under "# Dual-pass stats" is reworded
#   comment-only from "Forest-only warning rate-limit" to name the pass-2
#   extraction-failure warning rate-limit; it now states that a failed
#   extraction raises DualPassIncompleteError with no deposit.
# Why: Z11 review of the dual-pass branch flagged the stale forest-only wording
#   (there is no forest-only path); Chief ruling item 3. The
#   DualPassIncompleteError class docstring is left as is (accurate).
# How: one comment block reworded; date 2026-07-16 kept. No code change.
# -------------------
# [2026-09-25] Claude Code (kimi-k2.7-code) — Packet 214 V-1/V-2/D-2/D-3/LAW-5
#   corrective build (chief-p226-ruling-001).
#   What: (1) Call tokenizer.no_truncation() after every Tokenizer.from_pretrained
#         calls so the shipped tokenizer.json max_length does not re-impose
#         truncation; (2) build EmbedWindow.text from raw prefixed string char
#         offsets instead of decode-round-trip; (3) wrap _ensure_tokenizer's
#         from_pretrained in EmbeddingUnavailableError; (4) correct stale
#         dual-pass contract docs; (5) read NG_EMBED_TID_ENDPOINT from env;
#         (6) publish the fully-configured tokenizer BEFORE assigning to
#         self._tokenizer in both load paths, closing the race where a concurrent
#         caller could encode against a still-truncating tokenizer; (7) import
#         Tokenizer inside the D-2 try block; (8) revert pad_token to the
#         original string (the empty pad_token change was unexplained and broke
#         tokenizer-emitted padding warnings).
#   Why:  V-1: source-grep test "enable_truncation not in source" shipped while
#         the tokenizer's own config still truncated. V-2: decode round-trip
#         altered experience before deposit (LAW 7) and diverged local/remote
#         vectors. D-2/D-3: contract leaks and stale docs. LAW 5: env is source
#         of truth. Race/pad_token: identified by native law-enforcer review.
#   How:  no_truncation() and enable_padding() now run on a local tokenizer
#         variable before publishing to self._tokenizer. _ensure_tokenizer's from
#         tokenizers import moved inside try/except. pad_token restored in both
#         load paths. All other changes per Packet 214.
# -------------------
# [2026-09-22] Grok 4.6 — punchlist-001 B2: overlapping TID extraction windows
#   What: _extract_concepts no longer slices text[:2000]. Overlapping
#         max_content_for_extraction (2000) char windows cover the full
#         text; each window is one TID call (same per-call timeout);
#         results are unioned then capped at max_concepts. Any window
#         returning None fails the whole extraction (no deposit).
#   Why:  The 2000-char slice dropped the rest of the turn from Pass 2.
#         Forest embed()/embed_windows() path is unchanged.
#   How:  _char_extract_windows + per-window _tid_extract_window; overlap
#         extract_window_overlap (256). Short text (≤2000) is one call.
# [2026-09-21] Grok 4.6 — HF token is not a _hf_post parameter (R-2).
#   What: _hf_post reads the bearer via _get_hf_token() internally.
#         Token is no longer a function argument (traceback locals).
#   Why:  Lane 3 dumped a live HF token from pytest locals.
#         Law-enforcer R-2.
#   How:  Drop token arg; _hf_remote_call no longer threads it through.
# -------------------
# [2026-09-21] Grok 4.6 — CLS/SEP wrap every ONNX window (R-3).
#   What: Window the interior (between leading CLS and trailing SEP)
#         at 510 / overlap 64; wrap each slice as [CLS]+slice+[SEP]
#         before ONNX. Decode interior for window text. Missing
#         specials raise EmbeddingUnavailableError. Short path
#         (full encoding ≤512) unchanged — no double wrap.
#   Why:  Arctic is CLS-pooling; mid-sentence slices pooled from
#         position 0 are degenerate. Law-enforcer R-3.
#   How:  _cls_sep_ids from tokenizer.encode(""); _window_token_ids
#         strips specials, windows interior, returns wrapped ids.
# -------------------
# [2026-09-21] Grok 4.6 — dual_record_outcome extract-first (R3 atomicity).
#   What: Extract (and embed_batch) before any forest write. concepts is
#         None → best-effort signal_error, then DualPassIncompleteError;
#         no forest. Legitimate [] writes forest. Tree ids un-sliced.
#         Optional windows= echoed into result, not deposited.
#   Why:  Spec R3 — dual-pass is atomic or there is no deposit.
#   How:  Reorder dual_record_outcome; tree id f"{target_id}::tree::{concept}".
# -------------------
# [2026-09-21] Grok 4.6 — Ref-counted keep-warm pinger.
#   What: start_keepalive/stop_keepalive reference-counted; daemon thread
#         pings every 20s. No-op when not remote. Ping failures stay debug.
#   Why:  HF idle-eviction window is 30-60s; acceptance 8 requires a
#         concurrency test on the counter.
#   How:  Lock around the counter; 0→1 starts the thread; 1→0 signals stop.
# -------------------
# [2026-09-21] Grok 4.6 — Remote retry-3-then-raise + failed_embeds.jsonl.
#   What: 3 attempts, backoff 1s/3s/9s including after the last fail,
#         then one JSONL quarantine line, then raise. Quarantine write
#         failure must not mask the original error.
#   Why:  2026-07-07 remote shape; spec acceptance 6. Never hash.
#   How:  _hf_remote_call wraps _hf_post; _log_failed_embed appends
#         {cache_dir}/failed_embeds.jsonl.
# -------------------
# [2026-09-21] Grok 4.6 — NG_EMBED_REMOTE=hf gate + HF router primitive.
#   What: Opt-in remote feature-extraction via router.huggingface.co.
#         Invalid NG_EMBED_REMOTE values raise. Token from HF_TOKEN or
#         ~/.cache/huggingface/token. Body is {"inputs": ...} only.
#   Why:  Spec R4 + 2026-07-07 remote API shape. Canonical keeps the name.
#   How:  _ensure_model selects remote before any ONNX import; _hf_post
#         uses stdlib urllib.request; client-side prefix and optional L2.
# -------------------
# [2026-09-21] Grok 4.6 — Overlapping token windows + length-weighted pool.
#   What: Drop tokenizer truncation. Window at 512 / overlap 64. Pool
#         length-weighted, then L2-normalize the pooled long path only.
#         Short path (≤512) is byte-identical to the single-window primitive.
#   Why:  LAW 7 / spec R2 — truncation alters experience on the way in.
#         GSG poincare_dir needs a unit pooled forest vector (criterion 4).
#   How:  embed_windows + _window_token_ids + _pool_windows; embed()
#         returns .pooled; embed_batch windows then flattens ONNX batch.
# -------------------
# [2026-09-21] Grok 4.6 — Fail-closed embed: no hash fallback.
#   What: Raise EmbeddingUnavailableError when the model cannot load.
#         Remove the SHA-based fallback. DualPassIncompleteError declared
#         (unwired until Lane 2).
#   Why:  Spec R1 — no hash embedding anywhere; a failed embed is a
#         real failure. Plan Task 1.
#   How:  Public exception types; embed()/embed_batch() raise instead
#         of synthesizing a vector; empty batch still returns [] before
#         model load.
# -------------------
# [2026-03-22] Claude (Opus 4.6) — Initial creation.
#   What: Centralized embedding + dual-pass for entire ecosystem.
#   Why:  PRD §5 (Dual_Pass_Embedding_Implementation.md). Replaces 7+
#         identical _embed() functions. Prevents embedding dimension
#         mismatch incidents. Upgrades model from bge-base-en-v1.5 to
#         snowflake-arctic-embed-m-v1.5 (+1.89 retrieval MTEB).
#   How:  ONNX Runtime + tokenizers for embedding. TID for concept
#         extraction. Substrate-learnable gate for Pass 2 value.
# -------------------
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ng_ecosystem import NGEcosystem

logger = logging.getLogger("ng_embed")


class EmbeddingUnavailableError(Exception):
    """Raised when a real embedding cannot be produced.

    Public failure contract of embed()/embed_batch(). There is no hash
    fallback and no env var that re-enables one.
    """


class DualPassIncompleteError(Exception):
    """Raised when dual-pass cannot complete both forest and trees.

    Pass-2 extraction failure (TID down / timeout / malformed) must not
    leave a forest-only deposit. Distinct from EmbeddingUnavailableError.
    """


@dataclass(frozen=True)
class EmbedWindow:
    text: str
    embedding: np.ndarray  # shape (768,) float32, same contract as today's single embed
    token_count: int


@dataclass(frozen=True)
class WindowedEmbedding:
    pooled: np.ndarray     # short path: byte-identical to today's single embed
                           # long path: length-weighted mean of window vectors, then L2-normalized
    windows: tuple         # () if the input fit in one window; else one EmbedWindow per window
    token_count: int       # total tokenizer tokens of the (prefixed) input


# ---------------------------------------------------------------------------
# Configuration defaults — all values are bootstrap scaffolding
# ---------------------------------------------------------------------------

_WINDOW_TOKENS = 512
_WINDOW_OVERLAP = 64
_WINDOW_INTERIOR = _WINDOW_TOKENS - 2  # room for [CLS] + [SEP] wrap

_DEFAULT_CONFIG = {
    # Model
    "model_id": "Snowflake/snowflake-arctic-embed-m-v1.5",
    "onnx_filename": "onnx/model_quantized.onnx",
    "embedding_dim": 768,
    "pooling": "cls",
    "query_prefix": "Represent this sentence for searching relevant passages: ",
    "document_prefix": "",
    "cache_dir": str(Path.home() / ".cache" / "ng_embed"),

    # Dual-pass (Punchlist #81)
    "tid_endpoint": "http://127.0.0.1:7437/v1/chat/completions",
    "max_content_for_extraction": 2000,     # Chars per TID extraction window
    "extract_window_overlap": 256,          # Char overlap between TID windows
    "max_concepts": 20,                     # Cap extracted concepts
    "forest_to_tree_weight": 0.4,           # Bootstrap synapse weight
    "tree_to_forest_ratio": 0.7,            # tree→forest = forest_weight * ratio
    "tid_timeout": 30,                      # Seconds
    "tid_model": "auto",                    # TID routes to appropriate model
    "tid_temperature": 0.2,
    "tid_max_tokens": 500,
}

# The active embedding model and its output dimension — the single source of truth for every consumer that stores
# or compares vectors (#1097). A vector of any other dimension came from another model.
MODEL_NAME: str = _DEFAULT_CONFIG["model_id"]
EMBEDDING_DIM: int = int(_DEFAULT_CONFIG["embedding_dim"])

# The model's fixed mean-centring vector (#1107): the mean of L2-normalised document-mode vectors of this model over a
# reference corpus, subtracted (then re-normalised) by NG-Lite before every cosine comparison, because this model's
# vectors share a common direction (unrelated texts at ~0.69 raw cosine). Versioned with the model and NEVER recomputed
# at runtime. Provenance: CC vector store copy 2026-10-14 (sha256 90e3ccb4...), 18,675 vectors (3,026 whole deposits +
# 15,649 dual-pass concepts), float64 accumulation in sorted-key order, cast to float32; |mu| = 0.8023.
# compute_embedding_centre.py (NeuroGraph repo) reproduces it bit for bit.
CENTRE_VERSION: str = "snowflake-arctic-embed-m-v1.5/cc-store-20261014/v1"
CENTRE_SHA256: str = "99f0e568b670e1591e40f1a7a9d518dc42dcc6b342fbf2ec6e269964c9824518"
_CENTRE_B64 = (
    "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"
)
_centre: Optional[np.ndarray] = None


def embedding_centre() -> np.ndarray:
    """The model's mean-centring vector (EMBEDDING_DIM float32, read-only). Raises if the shipped bytes are damaged."""
    global _centre
    if _centre is None:
        import base64
        import hashlib

        raw = base64.b64decode("".join(_CENTRE_B64))
        if hashlib.sha256(raw).hexdigest() != CENTRE_SHA256 or len(raw) != 4 * EMBEDDING_DIM:
            raise RuntimeError(f"ng_embed: the shipped centre {CENTRE_VERSION} is damaged (sha256 mismatch)")
        c = np.frombuffer(raw, dtype="<f4").astype(np.float32)
        c.setflags(write=False)
        _centre = c
    return _centre

# Concept extraction prompt — not classification, not labeling.
# The LLM reads content and identifies distinct concepts within it.
# This is extraction at the ingestion boundary — the LLM is a tool
# that helps the substrate receive richer raw experience (Law 7).
_EXTRACTION_PROMPT = """Extract the key concepts, terms, and specific references from this text. Return them as a JSON array of short strings, each one a distinct concept or term mentioned in the text.

Focus on:
- Specific technical terms
- Named entities (people, tools, systems)
- Domain-specific concepts
- Action descriptions
- Relationships between things

Return ONLY a JSON array of strings. No explanation. Example: ["concept one", "concept two", "specific term"]

Text:
{content}"""


# ---------------------------------------------------------------------------
# NGEmbed — The singleton embedding service
# ---------------------------------------------------------------------------

class NGEmbed:
    """Centralized embedding engine for the E-T Systems ecosystem.

    Thread-safe singleton. One ONNX model instance per process, shared
    by all modules. Provides both single-pass embedding and dual-pass
    (forest + trees) via TID concept extraction.

    Usage:
        from ng_embed import embed, embed_batch

        vec = embed("some text")                    # 768-dim document embedding
        vec = embed("query text", is_query=True)    # With query prefix
        vec = embed("text", normalize=True)         # L2-normalized (Praxis)

        vecs = embed_batch(["text1", "text2"])      # Batch embedding
    """

    _instance: Optional["NGEmbed"] = None
    _lock = threading.Lock()

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = dict(_DEFAULT_CONFIG)
        if config:
            self._config.update(config)
        self._config["tid_endpoint"] = os.environ.get(
            "NG_EMBED_TID_ENDPOINT", self._config["tid_endpoint"]
        )

        self._session = None          # ONNX InferenceSession (lazy)
        self._tokenizer = None        # tokenizers.Tokenizer (lazy)
        self._model_loaded = False
        self._model_failed = False
        self._remote_mode = False
        self._model_lock = threading.Lock()
        self._keepalive_lock = threading.Lock()
        self._keepalive_refs = 0
        self._keepalive_thread: Optional[threading.Thread] = None
        self._keepalive_stop = threading.Event()
        self._keepalive_interval = 20.0

        # Dual-pass stats
        self._extractions = 0
        self._concepts_total = 0
        self._failures = 0
        # Pass-2 extraction-failure warning rate-limit (2026-07-16): when TID is
        # permanently absent (e.g. CC on a TID-less River), every deposit fails
        # pass-2 extraction — a failed extraction raises DualPassIncompleteError
        # with no deposit, so warn periodically with a suppressed-count instead
        # of per-deposit, so the signal survives without flooding. self._failures
        # stays the cumulative truth.
        self._last_extract_warn = 0.0
        self._failures_at_last_warn = 0
        self._last_failure_class = None   # [2026-10-23] #920: class name of the last failed TID call

    # -- Singleton -----------------------------------------------------------

    @classmethod
    def get_instance(cls, config: Optional[Dict[str, Any]] = None) -> "NGEmbed":
        """Thread-safe singleton factory."""
        if cls._instance is not None:
            return cls._instance
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(config)
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Destroy singleton (testing only)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance._session = None
                cls._instance._tokenizer = None
            cls._instance = None

    # -- Model loading -------------------------------------------------------

    def _ensure_model(self) -> bool:
        """Lazy-load ONNX model + tokenizer on first use."""
        if self._model_loaded:
            return True
        if self._model_failed:
            return False

        with self._model_lock:
            if self._model_loaded:
                return True
            if self._model_failed:
                return False

            remote = os.environ.get("NG_EMBED_REMOTE")
            if remote is not None:
                if remote != "hf":
                    raise EmbeddingUnavailableError(
                        f"unsupported NG_EMBED_REMOTE={remote!r}"
                    )
                self._remote_mode = True
                self._model_loaded = True
                logger.info(
                    "ng_embed: NG_EMBED_REMOTE=hf — using HF remote inference, "
                    "no local ONNX load"
                )
                return True

            try:
                import onnxruntime as ort
                from huggingface_hub import hf_hub_download
                from tokenizers import Tokenizer

                model_id = self._config["model_id"]
                cache_dir = self._config["cache_dir"]
                os.makedirs(cache_dir, exist_ok=True)

                # Download ONNX model
                onnx_path = hf_hub_download(
                    repo_id=model_id,
                    filename=self._config["onnx_filename"],
                    cache_dir=cache_dir,
                )

                # Load ONNX session (CPU, optimized)
                sess_opts = ort.SessionOptions()
                sess_opts.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                )
                sess_opts.intra_op_num_threads = max(1, os.cpu_count() // 2)
                self._session = ort.InferenceSession(
                    onnx_path,
                    sess_options=sess_opts,
                    providers=["CPUExecutionProvider"],
                )

                # Load tokenizer. The model's shipped tokenizer.json may
                # include truncation config; disable it explicitly (V-1).
                # Configure fully before publishing to self._tokenizer so a
                # concurrent caller never sees truncation-enabled state.
                _tokenizer = Tokenizer.from_pretrained(model_id)
                _tokenizer.no_truncation()
                _tokenizer.enable_padding(
                    pad_id=0, pad_token="[PAD]",
                )
                self._tokenizer = _tokenizer

                self._model_loaded = True
                logger.info(
                    "ng_embed: loaded %s (ONNX, %d-dim, CLS pooling)",
                    model_id, self._config["embedding_dim"],
                )
                return True

            except Exception as exc:
                logger.warning("ng_embed: model load failed: %s", exc)
                self._model_failed = True
                return False

    # -- Embedding -----------------------------------------------------------

    def embed(
        self,
        text: str,
        normalize: bool = False,
        is_query: bool = False,
    ) -> np.ndarray:
        """Embed text → 768-dim float32 numpy array.

        Args:
            text: Raw text to embed.
            normalize: L2-normalize output (True for Praxis compatibility).
            is_query: Prepend query prefix (for recall/search operations).

        Returns:
            768-dim float32 numpy array.
        """
        if not self._ensure_model():
            raise EmbeddingUnavailableError("embedding model unavailable")
        return self.embed_windows(text, normalize=normalize, is_query=is_query).pooled

    def embed_batch(
        self,
        texts: List[str],
        normalize: bool = False,
        is_query: bool = False,
    ) -> List[np.ndarray]:
        """Batch embedding for efficiency.

        Args:
            texts: List of texts to embed.
            normalize: L2-normalize outputs.
            is_query: Prepend query prefix to all texts.

        Returns:
            List of 768-dim float32 numpy arrays.
        """
        if not texts:
            return []
        if not self._ensure_model():
            raise EmbeddingUnavailableError("embedding model unavailable")
        self._ensure_tokenizer()

        prefixed = [self._apply_prefix(t, is_query) for t in texts]
        encodings = self._tokenizer.encode_batch(prefixed)
        results: List[Optional[np.ndarray]] = [None] * len(texts)
        short_idx = [i for i, enc in enumerate(encodings) if len(enc.ids) <= _WINDOW_TOKENS]
        long_idx = [i for i, enc in enumerate(encodings) if len(enc.ids) > _WINDOW_TOKENS]

        if short_idx:
            short_texts = [texts[i] for i in short_idx]
            if self._remote_mode:
                short_vecs = self._hf_remote_embed_batch(
                    short_texts, normalize=normalize, is_query=is_query,
                )
            else:
                short_vecs = self._onnx_embed_batch(
                    short_texts, normalize=normalize, is_query=is_query,
                )
            for i, vec in zip(short_idx, short_vecs):
                results[i] = vec

        if long_idx:
            window_jobs: List[tuple] = []
            for i in long_idx:
                ids = list(encodings[i].ids)
                prefixed_text = prefixed[i]
                enc = encodings[i]
                for wrapped, _interior_slice, weight, start, end in self._window_token_ids(ids):
                    char_start = enc.offsets[start][0]
                    char_end = enc.offsets[end - 1][1]
                    w_text = prefixed_text[char_start:char_end]
                    window_jobs.append((i, weight, wrapped, w_text))
            if self._remote_mode:
                vecs = self._hf_remote_embed_batch(
                    [job[3] for job in window_jobs],
                    normalize=False,
                    is_query=False,
                    skip_prefix=True,
                )
            else:
                vecs = self._onnx_embed_ids_batch(
                    [job[2] for job in window_jobs], normalize=False,
                )
            grouped: Dict[int, List[tuple]] = {}
            for (i, weight, _w_ids, _w_text), vec in zip(window_jobs, vecs):
                grouped.setdefault(i, []).append((weight, vec))
            for i, parts in grouped.items():
                results[i] = self._pool_windows(
                    [p[1] for p in parts],
                    [p[0] for p in parts],
                )

        return [vec for vec in results]  # type: ignore[misc]

    def embed_windows(
        self,
        text: str,
        normalize: bool = False,
        is_query: bool = False,
    ) -> WindowedEmbedding:
        """Windowed embed: one call if ≤512 tokens, else overlapping windows.

        Short path pooled vector is byte-identical to today's single-window
        primitive (including default normalize=False). Long path length-weighted
        mean-pools window vectors, then L2-normalizes the pooled result.
        """
        if not self._ensure_model():
            raise EmbeddingUnavailableError("embedding model unavailable")
        self._ensure_tokenizer()

        prefixed = self._apply_prefix(text, is_query)
        encoding = self._tokenizer.encode(prefixed)
        ids = list(encoding.ids)
        n = len(ids)

        if n <= _WINDOW_TOKENS:
            if self._remote_mode:
                vec = self._hf_remote_embed(
                    text, normalize=normalize, is_query=is_query,
                )
            else:
                vec = self._onnx_embed(text, normalize=normalize, is_query=is_query)
            return WindowedEmbedding(pooled=vec, windows=(), token_count=n)

        windows: List[EmbedWindow] = []
        embeddings: List[np.ndarray] = []
        weights: List[int] = []
        jobs = []
        for wrapped, _interior_slice, weight, start, end in self._window_token_ids(ids):
            char_start = encoding.offsets[start][0]
            char_end = encoding.offsets[end - 1][1]
            w_text = prefixed[char_start:char_end]
            jobs.append((w_text, wrapped, weight))

        if self._remote_mode:
            vecs = self._hf_remote_embed_batch(
                [j[0] for j in jobs],
                normalize=False,
                is_query=False,
                skip_prefix=True,
            )
        else:
            vecs = [
                self._onnx_embed(
                    w_text, normalize=False, is_query=False, _ids=w_ids,
                )
                for w_text, w_ids, _weight in jobs
            ]

        for (w_text, _w_ids, weight), vec in zip(jobs, vecs):
            windows.append(EmbedWindow(text=w_text, embedding=vec, token_count=weight))
            embeddings.append(vec)
            weights.append(weight)

        pooled = self._pool_windows(embeddings, weights)
        return WindowedEmbedding(
            pooled=pooled,
            windows=tuple(windows),
            token_count=n,
        )

    def _ensure_tokenizer(self) -> None:
        """Load tokenizer without ONNX (needed to window in remote mode).

        Raises EmbeddingUnavailableError on any load failure.
        """
        if self._tokenizer is not None:
            return
        try:
            from tokenizers import Tokenizer
            _tokenizer = Tokenizer.from_pretrained(self._config["model_id"])
            _tokenizer.no_truncation()
            _tokenizer.enable_padding(
                pad_id=0, pad_token="[PAD]",
            )
            self._tokenizer = _tokenizer
        except Exception as exc:
            raise EmbeddingUnavailableError(f"tokenizer load failed: {exc}") from exc

    def _apply_prefix(self, text: str, is_query: bool) -> str:
        if is_query:
            return self._config["query_prefix"] + text
        prefix = self._config["document_prefix"]
        return (prefix + text) if prefix else text

    def _cls_sep_ids(self) -> tuple:
        """Read CLS/SEP from the tokenizer. Do not hardcode 101/102."""
        encoding = self._tokenizer.encode("")
        specials = list(encoding.ids)
        if len(specials) < 2:
            raise EmbeddingUnavailableError(
                "tokenizer missing CLS/SEP specials"
            )
        return specials[0], specials[-1]

    def _window_token_ids(self, ids: Sequence[int]) -> List[tuple]:
        """Overlapping interior windows, each wrapped as [CLS]+slice+[SEP].

        Interior window length ≤ 510 so the wrapped sequence stays ≤ 512.
        Raises if the full encoding is missing a leading CLS or trailing SEP.

        Returns tuples of (wrapped_ids, interior_slice, weight, start, end)
        where start/end are ENCODING-relative indices (i.e., start points at
        the first token after CLS and end is one-past the last interior token),
        suitable for indexing encoding.offsets.
        """
        cls_id, sep_id = self._cls_sep_ids()
        if not ids or ids[0] != cls_id or ids[-1] != sep_id:
            raise EmbeddingUnavailableError(
                "encoding missing leading CLS or trailing SEP"
            )
        interior = list(ids[1:-1])
        n = len(interior)
        out: List[tuple] = []
        start = 0
        while start < n:
            end = min(start + _WINDOW_INTERIOR, n)
            slice_ids = interior[start:end]
            wrapped = [cls_id] + slice_ids + [sep_id]
            # +1 to make bounds encoding-relative (skip leading CLS).
            out.append((wrapped, slice_ids, end - start, start + 1, end + 1))
            if end >= n:
                break
            start += _WINDOW_INTERIOR - _WINDOW_OVERLAP
        return out

    def _pool_windows(
        self,
        vecs: Sequence[np.ndarray],
        weights: Sequence[int],
    ) -> np.ndarray:
        """Length-weighted mean, then L2-normalize (long path, unconditional)."""
        w = np.asarray(weights, dtype=np.float64)
        stacked = np.stack([np.asarray(v, dtype=np.float64) for v in vecs], axis=0)
        pooled = (stacked * w[:, None]).sum(axis=0) / w.sum()
        pooled32 = pooled.astype(np.float32)
        norm = float(np.linalg.norm(pooled32))
        if norm > 0:
            pooled32 = pooled32 / norm
        return pooled32

    def _onnx_embed(
        self,
        text: str,
        normalize: bool = False,
        is_query: bool = False,
        _ids: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        """Single-window embedding via ONNX Runtime. Raises if >512 tokens."""
        if _ids is not None:
            ids = list(_ids)
            attn = [1] * len(ids)
            return self._onnx_embed_ids(ids, attn, normalize=normalize)

        text = self._apply_prefix(text, is_query)
        encoding = self._tokenizer.encode(text)
        ids = list(encoding.ids)
        attn = list(encoding.attention_mask)
        return self._onnx_embed_ids(ids, attn, normalize=normalize)

    def _onnx_embed_ids(
        self,
        ids: Sequence[int],
        attention: Sequence[int],
        normalize: bool = False,
    ) -> np.ndarray:
        if len(ids) > _WINDOW_TOKENS:
            raise EmbeddingUnavailableError(
                "embedding window exceeds 512 tokens"
            )
        results = self._onnx_embed_ids_batch([list(ids)], normalize=normalize)
        return results[0]

    def _onnx_embed_ids_batch(
        self,
        ids_list: List[List[int]],
        normalize: bool = False,
        attention_list: Optional[List[List[int]]] = None,
    ) -> List[np.ndarray]:
        if not ids_list:
            return []
        if attention_list is None:
            attention_list = [[1] * len(ids) for ids in ids_list]
        for ids in ids_list:
            if len(ids) > _WINDOW_TOKENS:
                raise EmbeddingUnavailableError(
                    "embedding window exceeds 512 tokens"
                )
        # #1115: one sequence per session.run. The shipped model is dynamically int8-quantized
        # (onnx/model_quantized.onnx): activation scales are computed over the WHOLE input tensor, so
        # other texts and their padding in the same batch changed this text's vector (measured 0.977-0.992
        # cosine to its single-text vector). Running each sequence alone makes every result equal the
        # single-text embedding (same model, same vector space, same centre); cost ~1.3x on the laptop CPU.
        results = []
        for i, ids in enumerate(ids_list):
            attn = list(attention_list[i][:len(ids)])
            # Strip the tokenizer's batch padding too: padded positions (attention 0) still enter the
            # quantizer's activation range even when masked, so a text padded to a longer batch-mate's
            # length would not equal its own single-text embedding.
            length = len(attn)
            while length > 0 and attn[length - 1] == 0:
                length -= 1
            outputs = self._session.run(
                None,
                {
                    "input_ids": np.asarray([ids[:length]], dtype=np.int64),
                    "attention_mask": np.asarray([attn[:length]], dtype=np.int64),
                },
            )
            vec = outputs[1][0, :].astype(np.float32)
            if normalize:
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
            results.append(vec)
        return results

    def _onnx_embed_batch(
        self,
        texts: List[str],
        normalize: bool = False,
        is_query: bool = False,
    ) -> List[np.ndarray]:
        """Batch embedding via ONNX Runtime with padding. Short texts only."""
        if not texts:
            return []
        prefixed = [self._apply_prefix(t, is_query) for t in texts]
        encodings = self._tokenizer.encode_batch(prefixed)
        ids_list = [list(enc.ids) for enc in encodings]
        attn_list = [list(enc.attention_mask) for enc in encodings]
        return self._onnx_embed_ids_batch(
            ids_list, normalize=normalize, attention_list=attn_list,
        )

    def _hf_feature_extraction_url(self) -> str:
        model_id = self._config["model_id"]
        return (
            "https://router.huggingface.co/hf-inference/models/"
            f"{model_id}/pipeline/feature-extraction"
        )

    def _get_hf_token(self) -> str:
        env_tok = os.environ.get("HF_TOKEN")
        if env_tok:
            return env_tok.strip()
        path = Path.home() / ".cache" / "huggingface" / "token"
        try:
            file_tok = path.read_text().strip()
        except OSError:
            file_tok = ""
        if not file_tok:
            raise EmbeddingUnavailableError("HF token unavailable")
        return file_tok

    def _hf_post(
        self,
        url: str,
        payload: Dict[str, Any],
        timeout: int = 30,
    ) -> Any:
        import urllib.error
        import urllib.request

        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self._get_hf_token()}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise ConnectionError(f"HF HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            raise ConnectionError(str(exc.reason) if exc.reason else "url error") from exc

    def _log_failed_embed(
        self,
        text: str,
        is_query: bool,
        normalize: bool,
        error: BaseException,
        attempts: int,
    ) -> None:
        rec = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "text": text,
            "is_query": is_query,
            "normalize": normalize,
            "error": str(error),
            "attempts": attempts,
        }
        path = Path(self._config["cache_dir"]) / "failed_embeds.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _hf_remote_call(
        self,
        payload: Dict[str, Any],
        *,
        text: str,
        is_query: bool,
        normalize: bool,
        parse: Callable[[Any], Any],
    ) -> Any:
        url = self._hf_feature_extraction_url()
        last_exc: Optional[BaseException] = None
        for delay in (1, 3, 9):
            try:
                raw = self._hf_post(url, payload, timeout=30)
                return parse(raw)
            except EmbeddingUnavailableError:
                raise
            except Exception as exc:
                last_exc = exc
                time.sleep(delay)
        try:
            self._log_failed_embed(
                text=text,
                is_query=is_query,
                normalize=normalize,
                error=last_exc if last_exc is not None else RuntimeError("remote embed failed"),
                attempts=3,
            )
        except Exception:
            logger.debug("ng_embed: quarantine write failed")
        raise EmbeddingUnavailableError(
            f"remote embed failed: {last_exc}"
        ) from last_exc

    def _parse_hf_vector(self, raw: Any) -> np.ndarray:
        dim = int(self._config["embedding_dim"])
        if not isinstance(raw, list) or not raw:
            raise ValueError("malformed embedding response")
        if not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in raw):
            raise ValueError("malformed embedding response")
        if len(raw) != dim:
            raise ValueError("malformed embedding dimension")
        arr = np.asarray(raw, dtype=np.float32)
        if arr.shape != (dim,) or not np.all(np.isfinite(arr)):
            raise ValueError("malformed embedding response")
        return arr

    def _maybe_l2(self, vec: np.ndarray, normalize: bool) -> np.ndarray:
        if not normalize:
            return vec
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def _hf_remote_embed(
        self,
        text: str,
        normalize: bool = False,
        is_query: bool = False,
        _ids: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        payload_text = text if _ids is not None else self._apply_prefix(text, is_query)
        vec = self._hf_remote_call(
            {"inputs": payload_text},
            text=payload_text,
            is_query=is_query,
            normalize=normalize,
            parse=self._parse_hf_vector,
        )
        return self._maybe_l2(vec, normalize)

    def _hf_remote_embed_batch(
        self,
        texts: List[str],
        normalize: bool = False,
        is_query: bool = False,
        skip_prefix: bool = False,
    ) -> List[np.ndarray]:
        if not texts:
            return []
        payload = texts if skip_prefix else [self._apply_prefix(t, is_query) for t in texts]

        def _parse_batch(raw: Any) -> List[np.ndarray]:
            if not isinstance(raw, list) or len(raw) != len(texts):
                raise ValueError("malformed embedding batch")
            return [self._parse_hf_vector(item) for item in raw]

        vecs = self._hf_remote_call(
            {"inputs": payload},
            text=payload[0] if len(payload) == 1 else json.dumps(payload),
            is_query=is_query,
            normalize=normalize,
            parse=_parse_batch,
        )
        return [self._maybe_l2(v, normalize) for v in vecs]

    def start_keepalive(self) -> None:
        """Increment keepalive refcount; start the pinger on 0→1 if remote."""
        if not self._remote_mode:
            return
        with self._keepalive_lock:
            self._keepalive_refs += 1
            if self._keepalive_refs == 1:
                self._keepalive_stop.clear()
                thread = threading.Thread(
                    target=self._keepalive_loop,
                    name="ng_embed_keepalive",
                    daemon=True,
                )
                self._keepalive_thread = thread
                thread.start()

    def stop_keepalive(self) -> None:
        """Decrement keepalive refcount; signal stop on 1→0."""
        with self._keepalive_lock:
            if self._keepalive_refs <= 0:
                return
            self._keepalive_refs -= 1
            if self._keepalive_refs == 0:
                self._keepalive_stop.set()

    def _keepalive_loop(self) -> None:
        while not self._keepalive_stop.wait(self._keepalive_interval):
            try:
                self._hf_remote_embed("ping", normalize=False, is_query=False)
            except Exception:
                logger.debug("ng_embed: keepalive ping failed")

    # -- Dual-pass (Punchlist #81) -------------------------------------------

    def dual_record_outcome(
        self,
        ecosystem: "NGEcosystem",
        content: str,
        embedding: np.ndarray,
        target_id: str,
        success: bool,
        strength: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        windows: Optional[Sequence[EmbedWindow]] = None,
        prepared: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Dual-pass learning: forest embedding + tree concept extraction.

        Extract-first. `_extract_concepts` returning None is a pass-2
        failure: best-effort `signal_error`, then DualPassIncompleteError,
        with no forest write. Legitimate empty `[]` writes the forest
        only. Concepts are `embed_batch`'d before any write so an embed
        failure also leaves no deposit.

        Args:
            ecosystem: The module's NGEcosystem instance.
            content: Raw text content (for concept extraction).
            embedding: Pre-computed forest embedding (Pass 1).
            target_id: Opaque string for what was decided.
            success: Whether the outcome was successful.
            strength: Caller-reported significance [0.0, 1.0].
            metadata: Additional metadata dict.
            windows: Optional precomputed EmbedWindow sequence. Echoed
                into result["windows"] when non-empty; not deposited here.
            prepared: Optional result of prepare_dual_record(content) for
                THIS content (2026-10-23, #920/#922): the extraction and
                the tree embeddings were computed earlier (outside the
                caller's lock); this call only writes. A prepared
                extraction failure (concepts None) takes the SAME R3 path:
                WARNING + signal_error + DualPassIncompleteError, no write.
                None (default) = extract and embed here, as before.

        Returns:
            {
                "forest_result": dict,      # record_outcome result for forest
                "tree_ids": [str],           # Target IDs for tree nodes
                "concepts": [str],           # Extracted concept strings
                "pass2_attempted": bool,
                "extraction_failed": bool,   # always False on the return path
            }
        """
        # Extract before any write. None = TID broke (raise, no deposit).
        # [] = completed dual-pass with zero trees (write forest).
        if prepared is not None:
            concepts = prepared.get("concepts")
        else:
            concepts = self._extract_concepts(content)
        if concepts is None:
            # Warn + signal RATE-LIMITED (see _extraction_warn_due): when
            # TID is permanently absent every deposit fails, so per-deposit
            # warning floods the log. The raise is never rate-limited.
            _since = self._extraction_warn_due()
            if _since:
                logger.warning(
                    "dual_record_outcome[%s]: concept extraction FAILED — no deposit "
                    "(R3 atomicity); cumulative TID extraction failures=%d "
                    "(+%d since last warn — TID absent/unreachable); last_failure=%s",
                    target_id, self._failures, _since,
                    (prepared or {}).get("failure_class") or getattr(self, "_last_failure_class", None) or "-",
                )
                _signal = getattr(ecosystem, "signal_error", None)
                if callable(_signal):
                    try:
                        _signal(
                            RuntimeError("dual-pass concept extraction failed (no deposit)"),
                            {"target_id": target_id, "stage": "pass2_trees",
                             "extraction_failures": self._failures},
                        )
                    except Exception:  # noqa: BLE001 — signalling must never mask the raise
                        pass
            raise DualPassIncompleteError(
                f"pass-2 concept extraction failed for {target_id}; no deposit"
            )

        # Embed trees before any write so embed failure leaves no forest.
        if prepared is not None:
            tree_embeddings = list(prepared.get("tree_embeddings") or [])
            if len(tree_embeddings) != len(concepts):
                raise DualPassIncompleteError(
                    f"prepared dual pass for {target_id} has {len(tree_embeddings)} tree embeddings "
                    f"for {len(concepts)} concepts; no deposit"
                )
        else:
            tree_embeddings = self.embed_batch(concepts) if concepts else []

        # Workstream 2 (#274, 2026-05-31): use record_outcome_broadcast when
        # the ecosystem supports it; fall back to record_outcome for
        # consumers that pre-date the broadcast method (CommonsEco).
        if hasattr(ecosystem, "record_outcome_broadcast"):
            _record = ecosystem.record_outcome_broadcast
        else:
            _record = ecosystem.record_outcome

        forest_result = _record(
            embedding, target_id, success,
            strength=strength, metadata=metadata,
        )

        result = {
            "forest_result": forest_result,
            "tree_ids": [],
            "concepts": [],
            "pass2_attempted": True,
            "extraction_failed": False,
        }
        if windows:
            result["windows"] = [
                {"text": w.text, "embedding": w.embedding, "token_count": w.token_count}
                for w in windows
            ]

        if not concepts:
            return result  # legitimate empty — forest written, zero trees

        result["concepts"] = concepts

        for concept, tree_emb in zip(concepts, tree_embeddings):
            tree_meta = dict(metadata or {})
            tree_meta["_tree_concept"] = True
            tree_meta["_forest_target_id"] = target_id
            tree_meta["_concept"] = concept

            tree_target = f"{target_id}::tree::{concept}"
            tree_result = _record(
                tree_emb, tree_target, success,
                strength=strength * 0.8,  # Trees slightly softer than forest
                metadata=tree_meta,
            )

            if tree_result:
                result["tree_ids"].append(tree_target)

            # Forest→tree synapse creation happens in the substrate
            # through ng_lite's similarity-based association when the
            # tree embedding is close enough to the forest. The explicit
            # synapses below reinforce this connection at bootstrap weight.
            self._create_substrate_link(
                ecosystem, embedding, tree_emb,
                target_id, tree_target,
            )

        self._extractions += 1
        self._concepts_total += len(result["tree_ids"])

        logger.debug(
            "Dual-pass: forest=%s, %d trees from %d concepts",
            target_id[:32], len(result["tree_ids"]), len(concepts),
        )

        return result

    def prepare_dual_record(self, content: str) -> Dict[str, Any]:
        """The pure COMPUTE half of dual_record_outcome (2026-10-23, #920/#922).

        Runs the TID concept extraction and embeds the concepts, and nothing
        else: no ecosystem, no write, no lock. Pass the result to
        dual_record_outcome(..., prepared=) for the SAME content to write.

        Returns {"concepts": [str] | None, "tree_embeddings": [np.ndarray],
        "failure_class": str | None}. concepts None = the extraction FAILED
        (failure_class = the exception class of the failed TID call, e.g.
        "ReadTimeout", or "extraction_failed"); dual_record_outcome then
        raises DualPassIncompleteError without writing (R3). An embedding
        failure raises here (EmbeddingUnavailableError), as it would have
        inside dual_record_outcome, before any write.
        """
        self._last_failure_class = None
        concepts = self._extract_concepts(content)
        if concepts is None:
            return {"concepts": None, "tree_embeddings": [],
                    "failure_class": self._last_failure_class or "extraction_failed"}
        tree_embeddings = self.embed_batch(concepts) if concepts else []
        return {"concepts": concepts, "tree_embeddings": list(tree_embeddings), "failure_class": None}

    def _create_substrate_link(
        self,
        ecosystem: "NGEcosystem",
        forest_emb: np.ndarray,
        tree_emb: np.ndarray,
        forest_target: str,
        tree_target: str,
    ) -> None:
        """Create forest↔tree link in the substrate via record_outcome.

        Uses cross-recording: record the tree embedding against the forest
        target_id, and vice versa. This creates bidirectional associations
        in the substrate's Hebbian network.
        """
        weight = self._config["forest_to_tree_weight"]
        ratio = self._config["tree_to_forest_ratio"]

        # Forest→tree: "when I see this tree, recall the forest"
        try:
            ecosystem.record_outcome(
                tree_emb, forest_target, True,
                strength=weight,
                metadata={"_link": "dual_pass_tree_to_forest"},
            )
        except Exception:
            pass

        # Tree→forest: "when I see this forest, recall the tree"
        try:
            ecosystem.record_outcome(
                forest_emb, tree_target, True,
                strength=weight * ratio,
                metadata={"_link": "dual_pass_forest_to_tree"},
            )
        except Exception:
            pass

    def _char_extract_windows(self, text: str) -> List[str]:
        """Overlapping char windows covering the full text. Short text is one window."""
        size = int(self._config["max_content_for_extraction"])
        overlap = int(self._config.get("extract_window_overlap", 256))
        if size <= 0 or len(text) <= size:
            return [text]
        overlap = max(0, min(overlap, size - 1))
        step = max(1, size - overlap)
        windows: List[str] = []
        start = 0
        n = len(text)
        while start < n:
            end = min(start + size, n)
            windows.append(text[start:end])
            if end >= n:
                break
            start += step
        return windows

    def _tid_extract_window(self, content: str) -> Optional[List[str]]:
        """One TID call for one extraction window. None = this call failed."""
        import requests

        prompt = _EXTRACTION_PROMPT.format(content=content)
        try:
            resp = requests.post(
                self._config["tid_endpoint"],
                json={
                    "model": self._config["tid_model"],
                    "messages": [
                        {
                            "role": "system",
                            "content": "You extract concepts from text. "
                                       "Return only a JSON array of strings.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": self._config["tid_temperature"],
                    "max_tokens": self._config["tid_max_tokens"],
                },
                timeout=self._config["tid_timeout"],
            )
            resp.raise_for_status()
            response_text = (
                resp.json()["choices"][0]["message"]["content"].strip()
            )
            return self._parse_concepts(response_text)
        except Exception as exc:
            # Count every failure (self._failures is the cumulative truth, surfaced
            # in status). The user-facing WARNING is emitted rate-limited by the
            # caller (dual_record_outcome) so a permanently-absent TID doesn't flood
            # the log; the per-call exception detail stays at debug.
            self._failures += 1
            self._last_failure_class = type(exc).__name__
            logger.debug("Concept extraction failed (TID): %s", exc)
            return None

    def _extract_concepts(self, text: str) -> Optional[List[str]]:
        """Extract concepts from text via TID LLM call.

        Overlapping windows of `max_content_for_extraction` chars cover the
        full text. Each window is one TID call (same per-call timeout).
        Window results are unioned by round-robin interleave — for rank i,
        each window contributes its i-th concept in window order (dedup,
        first-seen order within a window) — then capped at `max_concepts`.
        A positional union let the head window fill the cap and silently
        drop later windows' concepts (#666).

        Returns the list of concept strings (possibly empty `[]` when TID
        legitimately found none), or **`None`** when any window call FAILED
        (TID down / timeout / malformed response). The None-vs-[] distinction
        lets the caller surface a real failure instead of silently treating a
        broken extraction as "no concepts" (no silent failures).
        """
        union: List[str] = []
        seen = set()
        per_window: List[List[str]] = []
        for window in self._char_extract_windows(text):
            concepts = self._tid_extract_window(window)
            if concepts is None:
                return None
            per_window.append(concepts)
        max_rank = max((len(concepts) for concepts in per_window), default=0)
        for i in range(max_rank):
            for concepts in per_window:
                if i < len(concepts):
                    concept = concepts[i]
                    if concept not in seen:
                        seen.add(concept)
                        union.append(concept)
        return union[: self._config["max_concepts"]]

    def _extraction_warn_due(self) -> int:
        """Rate-limit the pass-2 extraction failure warning. Returns the number of
        failures since the last emitted warning when a warning is due (>=1, truthy),
        else 0. Interval via CC_EXTRACT_WARN_INTERVAL_S (default 60s). Attrs are
        getattr-defaulted so it works regardless of construction path."""
        interval = float(os.environ.get("CC_EXTRACT_WARN_INTERVAL_S", "60"))
        now = time.monotonic()
        last = getattr(self, "_last_extract_warn", 0.0)
        if now - last >= interval:
            since = self._failures - getattr(self, "_failures_at_last_warn", 0)
            self._last_extract_warn = now
            self._failures_at_last_warn = self._failures
            return max(1, since)
        return 0

    @staticmethod
    def _parse_concepts(text: str) -> List[str]:
        """Parse a JSON array from LLM response, handling markdown fences."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        try:
            result = json.loads(text)
            if isinstance(result, list):
                return [str(c).strip() for c in result if str(c).strip()]
        except json.JSONDecodeError:
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                try:
                    result = json.loads(text[start:end])
                    if isinstance(result, list):
                        return [str(c).strip() for c in result if str(c).strip()]
                except json.JSONDecodeError:
                    pass

        return []

    # -- Stats ---------------------------------------------------------------

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "model_id": self._config["model_id"],
            "model_loaded": self._model_loaded,
            "embedding_dim": self._config["embedding_dim"],
            "pooling": self._config["pooling"],
            "dual_pass": {
                "extractions": self._extractions,
                "concepts_total": self._concepts_total,
                "failures": self._failures,
                "last_failure_class": getattr(self, "_last_failure_class", None),
                "avg_concepts": (
                    round(self._concepts_total / self._extractions, 1)
                    if self._extractions > 0 else 0
                ),
            },
        }


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def embed(
    text: str,
    normalize: bool = False,
    is_query: bool = False,
) -> np.ndarray:
    """Embed text → 768-dim float32 numpy array.

    Convenience wrapper around NGEmbed.get_instance().embed().
    """
    return NGEmbed.get_instance().embed(text, normalize=normalize, is_query=is_query)


def embed_batch(
    texts: List[str],
    normalize: bool = False,
    is_query: bool = False,
) -> List[np.ndarray]:
    """Batch embed texts → list of 768-dim float32 numpy arrays."""
    return NGEmbed.get_instance().embed_batch(
        texts, normalize=normalize, is_query=is_query,
    )
