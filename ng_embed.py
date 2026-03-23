"""
ng_embed — Centralized embedding service for the E-T Systems ecosystem.

Singleton embedding engine used by every module. Provides:
  1. Unified embedding via Snowflake/snowflake-arctic-embed-m-v1.5 (ONNX)
  2. Dual-pass embedding (forest + trees) via TID concept extraction
  3. Thread-safe singleton — one model instance per process
  4. Hash fallback when ONNX model unavailable

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

import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ng_ecosystem import NGEcosystem

logger = logging.getLogger("ng_embed")

# ---------------------------------------------------------------------------
# Configuration defaults — all values are bootstrap scaffolding
# ---------------------------------------------------------------------------

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
        self._model_lock = threading.Lock()

        # Dual-pass stats
        self._extractions = 0
        self._concepts_total = 0
        self._failures = 0

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
                self._tokenizer.enable_truncation(max_length=512)

                self._model_loaded = True
                logger.info(
                    "ng_embed: loaded %s (ONNX, %d-dim, CLS pooling)",
                    model_id, self._config["embedding_dim"],
                )
                return True

            except Exception as exc:
                logger.warning("ng_embed: model load failed, using hash fallback: %s", exc)
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
        if self._ensure_model():
            return self._onnx_embed(text, normalize=normalize, is_query=is_query)
        return self._hash_embed(text, normalize=normalize)

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
        if self._ensure_model():
            return self._onnx_embed_batch(texts, normalize=normalize, is_query=is_query)
        return [self._hash_embed(t, normalize=normalize) for t in texts]

    def _onnx_embed(
        self,
        text: str,
        normalize: bool = False,
        is_query: bool = False,
    ) -> np.ndarray:
        """Single text embedding via ONNX Runtime."""
        # Apply prefix
        if is_query:
            text = self._config["query_prefix"] + text
        else:
            prefix = self._config["document_prefix"]
            if prefix:
                text = prefix + text

        # Tokenize
        encoding = self._tokenizer.encode(text)
        input_ids = np.array([encoding.ids], dtype=np.int64)
        attention_mask = np.array([encoding.attention_mask], dtype=np.int64)

        # Infer
        outputs = self._session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
        )

        # sentence_embedding output (index 1) — pre-pooled by model
        vec = outputs[1][0, :].astype(np.float32)

        if normalize:
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm

        return vec

    def _onnx_embed_batch(
        self,
        texts: List[str],
        normalize: bool = False,
        is_query: bool = False,
    ) -> List[np.ndarray]:
        """Batch embedding via ONNX Runtime with padding."""
        # Apply prefixes
        prefixed = []
        for text in texts:
            if is_query:
                prefixed.append(self._config["query_prefix"] + text)
            else:
                prefix = self._config["document_prefix"]
                prefixed.append((prefix + text) if prefix else text)

        # Batch tokenize
        encodings = self._tokenizer.encode_batch(prefixed)
        max_len = max(len(e.ids) for e in encodings)

        input_ids = np.zeros((len(encodings), max_len), dtype=np.int64)
        attention_mask = np.zeros((len(encodings), max_len), dtype=np.int64)

        for i, enc in enumerate(encodings):
            length = len(enc.ids)
            input_ids[i, :length] = enc.ids
            attention_mask[i, :length] = enc.attention_mask

        # Infer
        outputs = self._session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
        )

        # sentence_embedding output (index 1) — pre-pooled by model
        results = []
        for i in range(len(texts)):
            vec = outputs[1][i, :].astype(np.float32)
            if normalize:
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
            results.append(vec)

        return results

    def _hash_embed(
        self,
        text: str,
        normalize: bool = False,
    ) -> np.ndarray:
        """Deterministic hash-based fallback embedding.

        Produces a stable 768-dim vector from text via SHA-384.
        Not semantically meaningful — ensures modules can operate
        when the ONNX model is unavailable.
        """
        dim = self._config["embedding_dim"]
        h = hashlib.sha384(text.encode("utf-8")).digest()
        # Expand hash to fill dim via seeded RNG
        rng = np.random.RandomState(
            int.from_bytes(h[:4], "little")
        )
        vec = rng.randn(dim).astype(np.float32)
        if normalize:
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
        return vec

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
    ) -> Dict[str, Any]:
        """Dual-pass learning: forest embedding + tree concept extraction.

        Pass 1: Record the forest (gestalt) embedding via ecosystem.record_outcome().
        Pass 2: Extract concepts via TID → embed each → record_outcome()
                 per tree → create forest→tree synapses in the substrate.

        If TID is unavailable or extraction fails, gracefully falls back
        to single-pass (forest only). Pass 2 failure is never fatal.

        Args:
            ecosystem: The module's NGEcosystem instance.
            content: Raw text content (for concept extraction).
            embedding: Pre-computed forest embedding (Pass 1).
            target_id: Opaque string for what was decided.
            success: Whether the outcome was successful.
            strength: Caller-reported significance [0.0, 1.0].
            metadata: Additional metadata dict.

        Returns:
            {
                "forest_result": dict,      # record_outcome result for forest
                "tree_ids": [str],           # Target IDs for tree nodes
                "concepts": [str],           # Extracted concept strings
                "pass2_attempted": bool,
            }
        """
        # Pass 1: Forest — standard record_outcome with gestalt embedding
        forest_result = ecosystem.record_outcome(
            embedding, target_id, success,
            strength=strength, metadata=metadata,
        )

        result = {
            "forest_result": forest_result,
            "tree_ids": [],
            "concepts": [],
            "pass2_attempted": False,
        }

        # Pass 2: Trees — concept extraction via TID
        concepts = self._extract_concepts(content)
        result["pass2_attempted"] = True

        if not concepts:
            return result

        result["concepts"] = concepts

        # Embed and record each concept
        tree_embeddings = self.embed_batch(concepts)
        for concept, tree_emb in zip(concepts, tree_embeddings):
            tree_meta = dict(metadata or {})
            tree_meta["_tree_concept"] = True
            tree_meta["_forest_target_id"] = target_id
            tree_meta["_concept"] = concept

            tree_target = f"{target_id}::tree::{concept[:64]}"
            tree_result = ecosystem.record_outcome(
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

    def _extract_concepts(self, text: str) -> List[str]:
        """Extract concepts from text via TID LLM call.

        One LLM call per ingestion. Returns list of concept strings,
        or empty list on failure (non-fatal).
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
            logger.debug("Concept extraction failed: %s", exc)
            self._failures += 1
            return []

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
