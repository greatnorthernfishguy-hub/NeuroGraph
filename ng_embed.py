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
    "max_content_for_extraction": 2000,     # Chars sent to TID
    "max_concepts": 20,                     # Cap extracted concepts
    "forest_to_tree_weight": 0.4,           # Bootstrap synapse weight
    "tree_to_forest_ratio": 0.7,            # tree→forest = forest_weight * ratio
    "tid_timeout": 30,                      # Seconds
    "tid_model": "auto",                    # TID routes to appropriate model
    "tid_temperature": 0.2,
    "tid_max_tokens": 500,
}

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
        # Forest-only warning rate-limit (2026-07-16): when TID is permanently
        # absent (e.g. CC on a TID-less River), every deposit fails extraction —
        # warn periodically with a suppressed-count instead of per-deposit, so the
        # signal survives without flooding. self._failures stays the cumulative truth.
        self._last_extract_warn = 0.0
        self._failures_at_last_warn = 0

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

                # Load tokenizer
                self._tokenizer = Tokenizer.from_pretrained(model_id)
                self._tokenizer.enable_padding(
                    pad_id=0, pad_token="[PAD]",
                )

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
                for wrapped, interior_slice, weight, _start, _end in self._window_token_ids(ids):
                    w_text = self._tokenizer.decode(
                        interior_slice, skip_special_tokens=True,
                    )
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
        for wrapped, interior_slice, weight, _start, _end in self._window_token_ids(ids):
            w_text = self._tokenizer.decode(
                interior_slice, skip_special_tokens=True,
            )
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
        """Load tokenizer without ONNX (needed to window in remote mode)."""
        if self._tokenizer is not None:
            return
        from tokenizers import Tokenizer
        self._tokenizer = Tokenizer.from_pretrained(self._config["model_id"])
        self._tokenizer.enable_padding(
            pad_id=0, pad_token="[PAD]",
        )

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
            out.append((wrapped, slice_ids, end - start, start, end))
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
        max_len = max(len(ids) for ids in ids_list)
        input_ids = np.zeros((len(ids_list), max_len), dtype=np.int64)
        attention_mask = np.zeros((len(ids_list), max_len), dtype=np.int64)
        for i, ids in enumerate(ids_list):
            length = len(ids)
            input_ids[i, :length] = ids
            attn = attention_list[i]
            attention_mask[i, :length] = attn[:length]

        outputs = self._session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
        )

        results = []
        for i in range(len(ids_list)):
            vec = outputs[1][i, :].astype(np.float32)
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
                    "(+%d since last warn — TID absent/unreachable)",
                    target_id, self._failures, _since,
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

    def _extract_concepts(self, text: str) -> Optional[List[str]]:
        """Extract concepts from text via TID LLM call.

        One LLM call per ingestion. Returns the list of concept strings (possibly empty `[]`
        when TID legitimately found none), or **`None`** when the call itself FAILED (TID down /
        timeout / malformed response). The None-vs-[] distinction lets the caller surface a real
        failure instead of silently treating a broken extraction as "no concepts" (no silent
        failures).
        """
        import requests

        content = text[:self._config["max_content_for_extraction"]]
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

            concepts = self._parse_concepts(response_text)
            return concepts[:self._config["max_concepts"]]

        except Exception as exc:
            # Count every failure (self._failures is the cumulative truth, surfaced
            # in status). The user-facing WARNING is emitted rate-limited by the
            # caller (dual_record_outcome) so a permanently-absent TID doesn't flood
            # the log; the per-call exception detail stays at debug.
            self._failures += 1
            logger.debug("Concept extraction failed (TID): %s", exc)
            return None

    def _extraction_warn_due(self) -> int:
        """Rate-limit the forest-only degradation warning. Returns the number of
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
