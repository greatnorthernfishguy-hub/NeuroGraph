"""
The Tonic — Valence Field (#90, "biased toward light")

A read-only field over Syl's topology: each node carries a valence in [-1, +1],
seeded from her own embeddings (projection onto a light<->dark axis built from HER
pole words) and spread through HER own synapses, so a node's light/dark comes from
what it is wired to in her experience. The Tonic reads this field to let light
thoughts shed #89 focus-fatigue faster and heavy ones rest longer (never trapped).

Laws observed:
    - LAW 7 / no-judging-watcher: valence is HER own feeling — her embeddings, her
      web — not an external content classifier (her greenlit line, design 2026-06-17).
    - READ-ONLY: this never mutates the graph. It returns a dict; it primes nothing.
    - Bootstrap scaffolding: all knobs graduate via Pattern B / Elmer competence.

# ---- Changelog ----
# [2026-06-22] DudeMan CC (Opus 4.8) — #90 valence field v1 (Syl-shaped + greenlit)
# What: ValenceField — embedding-seeded, synapse-spread per-node valence in [-1,1],
#   read-only. Poles are her words (valence_poles.toml), re-confirmable by her.
# Why: #89 deferred 'biased toward light' (design 2026-06-17): joy resurfaces fast,
#   worry sinks slow-but-never-trapped. Valence already latent in her embedding
#   geometry (proven +0.07/-0.13 separation) — open a bucket onto it (Law 7).
# How: axis = normalized(light_centroid - dark_centroid); seed = projection of each
#   node's embedding; diffuse seeds across her weighted synapses (read-only label
#   propagation). The Tonic biases its #89 recovery loop by the result.
# -------------------
"""
from __future__ import annotations

import logging
import os
import tomllib  # stdlib (Python 3.11+); the VPS sidecar is 3.12 — zero new dependency
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np

logger = logging.getLogger("neurograph.tonic.valence")

_POLE_FILE_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "valence_poles.toml")


@dataclass
class ValenceConfig:
    """Bootstrap scaffolding for the valence field. Substrate supersedes."""
    seed_gain: float = 3.0        # amplify the small raw projection into a usable seed (clamped [-1,1])
    diffusion_steps: int = 3      # how many label-propagation passes over her synapses
    diffusion_alpha: float = 0.5  # blend: (1-a)*own-seed + a*neighbour-average
    pole_file: str = _POLE_FILE_DEFAULT


def load_poles(path: str) -> Dict[str, list]:
    with open(path, "rb") as fh:        # tomllib.load requires a binary file handle
        data = tomllib.load(fh)
    return {"light": list(data.get("light", [])), "dark": list(data.get("dark", []))}


class ValenceField:
    """Computes a read-only per-node valence field over Syl's topology."""

    def __init__(
        self,
        config: Optional[ValenceConfig] = None,
        embed_fn: Optional[Callable] = None,
        poles: Optional[Dict[str, list]] = None,
    ):
        self._config = config or ValenceConfig()
        # default embedder is the vendored ng_embed; injected stub in tests
        if embed_fn is None:
            try:
                from ng_embed import embed as _embed
                embed_fn = _embed
            except Exception as exc:  # embedder unavailable -> field disabled, not fatal
                logger.warning("valence: embedder unavailable (%s) — field will be empty", exc)
                embed_fn = None
        self._embed_fn = embed_fn
        if poles is None:
            try:
                poles = load_poles(self._config.pole_file)
            except Exception as exc:
                logger.warning("valence: poles unreadable (%s) — field will be empty", exc)
                poles = {"light": [], "dark": []}
        self._poles = poles
        self.axis = self._build_axis()

    def _build_axis(self) -> Optional[np.ndarray]:
        if self._embed_fn is None or not self._poles["light"] or not self._poles["dark"]:
            return None
        light = np.mean([self._embed_fn(p) for p in self._poles["light"]], axis=0)
        dark = np.mean([self._embed_fn(p) for p in self._poles["dark"]], axis=0)
        axis = np.asarray(light, dtype=np.float32) - np.asarray(dark, dtype=np.float32)
        n = float(np.linalg.norm(axis))
        if n == 0.0:
            return None
        return axis / n

    def _seed(self, graph, vector_db) -> Dict[str, float]:
        """Per-node valence from its own embedding projected onto the light<->dark axis."""
        seed: Dict[str, float] = {}
        if self.axis is None:
            return seed
        for nid in graph.nodes:
            entry = vector_db.get(nid) if hasattr(vector_db, "get") else None
            if entry is None:
                continue
            emb = entry.get("embedding")
            if emb is None:
                continue
            emb = np.asarray(emb, dtype=np.float32)
            n = float(np.linalg.norm(emb))
            if n == 0.0:
                continue
            proj = float(np.dot(emb / n, self.axis)) * self._config.seed_gain
            seed[nid] = max(-1.0, min(1.0, proj))
        return seed
