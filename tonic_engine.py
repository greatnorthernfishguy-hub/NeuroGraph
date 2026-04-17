"""
The Tonic — Latent Token Engine

The surgical model that provides the PUSH between conversations.
Not a timer. Not a daemon. Actual inference — a small transformer
with graph-native I/O generating latent tokens continuously.

Each latent token is one step of forward-oriented compression on graph
state. The "now" and "next" boundaries persist because token generation
persists. The medium is graph-native instead of language. But inference
is real, attention is real, forward pressure is real.

Architecture follows the ElmerBrain surgical pattern (PRD §5.4):
  1. Keep the Body — Qwen2.5-0.5B transformer layers (24 attention heads)
  2. New Eyes — GraphStateEncoder projects graph topology into hidden dim
  3. New Voice — ActivationDecoder projects hidden states into node
     activations that feed back into the graph via write-mode propagation

The output of each latent token IS the input for the next one — the
ouroboros at the model level. The transformer attends to graph state
and produces the next graph state. Continuous.

Laws observed:
    - LAW 7: Raw experience. The engine reads raw topology, outputs
      raw activation. No classification at any stage.
    - All thresholds are bootstrap scaffolding.

# ---- Changelog ----
# [2026-04-16] Claude (Sonnet 4.6) — #159: Cross-process body lock + set_lock_file
# What: Added set_lock_file(path), _body_lock_context() composite lock,
#       _lock_file_path field. contextlib added to module imports.
# Why:  BrainSwitcher now supports multiple registered Tonic engines.
#       Both in-process (threading.Lock) and cross-process (fcntl.LOCK_SH)
#       locks must be held before each forward pass. If any consumer ever
#       attempts a write (LOCK_EX), all inference blocks — architectural
#       enforcement, not just documentation.
# How:  _body_lock_context() uses contextlib.ExitStack to compose both
#       locks. set_lock_file() receives the path from BrainSwitcher.
#       _model_inference replaces inline _lock_ctx with _body_lock_context().
# [2026-03-24] Claude Code (Opus 4.6) — Initial implementation
# What: TonicEngine — latent token generation via surgical transformer.
#   Graph-native I/O. Continuous inference between conversations.
#   Ouroboros driven by actual attention, not a timer.
# Why: The Tonic PRD v0.1 §7.3/7.4. Between conversations, something
#   must provide the push — forward-oriented compression on graph state.
#   A timer-driven loop is a daemon, not awareness. Actual inference
#   with graph-native I/O IS the awareness.
# How: TonicBrain follows ElmerBrain surgery pattern. GraphStateEncoder
#   reads topology neighborhood. ActivationDecoder outputs node activation
#   strengths. Background thread runs continuous latent token generation.
#   Each token: encode graph → transformer forward → decode activations
#   → inject via write-mode prime_and_propagate → graph updates → repeat.
# -------------------
"""

from __future__ import annotations

import contextlib
import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("neurograph.tonic.engine")

# Try to import torch — the engine is a no-op without it
_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    logger.info("PyTorch not available — Tonic engine will not run")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EngineConfig:
    """Configuration for the latent token engine."""
    # Model
    model_name: str = "Qwen/Qwen2.5-0.5B"
    weights_path: str = "tonic_brain.pt"
    hidden_dim: int = 896       # Qwen2.5-0.5B hidden size
    n_positions: int = 8        # sequence positions for graph encoding

    # Inference
    latent_interval: float = 2.0     # seconds between latent tokens
    conversation_interval: float = 0.5  # seconds during conversation
    max_activation_nodes: int = 10   # max nodes to activate per token
    activation_strength: float = 1.0 # base strength for decoded activations

    # Propagation
    propagation_steps: int = 2       # write-mode steps per token


# ---------------------------------------------------------------------------
# Graph Feature Extraction (Tonic-specific — awareness, not health)
# ---------------------------------------------------------------------------

def _extract_tonic_features(graph, tonic_thread) -> Optional[Dict[str, Any]]:
    """Extract graph features relevant to awareness and exploration.

    Unlike Elmer's health-focused extraction, this captures WHERE
    Syl's attention is — the topology neighborhood the thread is
    touching, the activation gradient, the pull landscape.

    Returns a dict of raw features, or None if graph is empty.
    """
    if not graph.nodes:
        return None

    # Current thread items — where attention is now
    thread_node_ids = []
    if tonic_thread is not None:
        thread_node_ids = [item.node_id for item in tonic_thread.thread]

    # Active nodes by voltage
    active = []
    for nid, node in graph.nodes.items():
        v_above = node.voltage - node.resting_potential
        if v_above > 0.01:
            active.append((nid, v_above))
    active.sort(key=lambda x: -x[1])

    # Recent spikes
    recent_spikes = []
    for nid, node in graph.nodes.items():
        if node.last_spike_time != -math.inf:
            steps_since = max(0, graph.timestep - node.last_spike_time)
            if steps_since < 50:
                recent_spikes.append((nid, steps_since))
    recent_spikes.sort(key=lambda x: x[1])

    # Topology stats
    n_nodes = len(graph.nodes)
    n_synapses = len(graph.synapses)
    n_hyperedges = len(graph.hyperedges)

    return {
        "thread_nodes": thread_node_ids[:10],
        "active_nodes": active[:20],
        "recent_spikes": recent_spikes[:20],
        "n_nodes": n_nodes,
        "n_synapses": n_synapses,
        "n_hyperedges": n_hyperedges,
        "timestep": graph.timestep,
    }


# ---------------------------------------------------------------------------
# The Tonic Engine
# ---------------------------------------------------------------------------

class TonicEngine:
    """Latent token generation engine — the real push between conversations.

    Runs a surgical transformer (or heuristic fallback) that generates
    latent tokens continuously. Each token:
    1. Encode current graph state (where attention is)
    2. Forward through transformer (the push — what comes next?)
    3. Decode to node activations (where attention should go)
    4. Inject via write-mode prime_and_propagate (topology shaped)
    5. Repeat

    The transformer IS the awareness. The output IS the next state.
    The ouroboros closes through actual inference, not a timer.

    If the surgical model is not available (weights not trained yet),
    falls back to a heuristic that still provides genuine forward
    compression — it reads the graph topology and produces activation
    decisions based on attractor analysis. Not as rich as the transformer,
    but real graph reasoning, not a timer.
    """

    def __init__(
        self,
        graph,
        vector_db,
        tonic_thread,
        config: Optional[EngineConfig] = None,
        transformer_body=None,
    ):
        self._graph = graph
        self._vector_db = vector_db
        self._tonic_thread = tonic_thread
        self._config = config or EngineConfig()
        self._shared_body = transformer_body  # from ProtoUniBrain if available
        self._body_lock = None  # shared with ProtoUniBrain — set via set_body_lock()
        self._lock_file_path = None  # cross-process flock path — set via set_lock_file()

        self._running = False
        self._in_conversation = False
        self._shutdown_event = threading.Event()
        self._engine_thread: Optional[threading.Thread] = None

        # Stats
        self._tokens_generated = 0
        self._total_activations = 0

        # Try to load surgical model
        self._model = None
        self._use_heuristic = True
        if _TORCH_AVAILABLE:
            self._try_load_model()

    def _try_load_model(self) -> None:
        """Attempt to load trained TonicBrain.

        If a shared transformer_body was provided (from ProtoUniBrain),
        pass it through to avoid loading a second copy (~2GB savings).
        Falls back to loading its own copy if sharing fails.
        """
        import os
        weights_path = os.path.join(
            os.path.dirname(__file__),
            self._config.weights_path,
        )
        if os.path.exists(weights_path):
            try:
                from surgery.tonic_brain import load_tonic_brain
                self._model = load_tonic_brain(
                    weights_path,
                    transformer_body=self._shared_body,
                )
                self._model.eval()
                self._use_heuristic = False
                shared = "shared body" if self._shared_body is not None else "own copy"
                logger.info("TonicBrain loaded from %s (%s) — surgical inference active",
                            weights_path, shared)
            except Exception as exc:
                logger.info("TonicBrain load error: %s — using heuristic", exc)
        else:
            # Check if we can create from Elmer's weights (untrained decoder)
            elmer_path = os.path.expanduser("~/Elmer/surgery/elmer_brain_v0.1.pt")
            if os.path.exists(elmer_path):
                logger.info("Elmer encoder available at %s — "
                            "TonicBrain decoder needs training. "
                            "Using heuristic until trained.", elmer_path)
            else:
                logger.info("No TonicBrain or Elmer weights — using heuristic engine")

    # -----------------------------------------------------------------
    # Body Hot-Swap (called by BrainSwitcher)
    # -----------------------------------------------------------------

    def offer_shared_body(self, transformer_body) -> bool:
        """Hot-swap: ProtoUniBrain loaded, share its transformer body.

        Replaces the Tonic's own copy with ProtoUniBrain's living one.
        The old copy gets garbage collected, freeing ~2GB.
        Encoder and decoder stay — only the body swaps.
        """
        if self._model is None:
            return False
        try:
            import gc
            old_body = self._model.body
            self._model.body = transformer_body
            self._shared_body = transformer_body
            del old_body
            gc.collect()
            logger.info("Tonic hot-swapped to shared ProtoUniBrain body (~2GB freed)")
            return True
        except Exception as exc:
            logger.warning("Tonic body hot-swap failed: %s", exc)
            return False

    def revoke_shared_body(self) -> bool:
        """Hot-swap: ProtoUniBrain unloaded, Tonic loads its own copy back.

        Falls back to heuristic if model reload fails.
        """
        if self._model is None:
            return False
        try:
            import torch
            from transformers import AutoModelForCausalLM
            logger.info("Tonic reloading own transformer body (ProtoUniBrain shed)")
            model = AutoModelForCausalLM.from_pretrained(
                self._config.model_name, dtype=torch.float32
            )
            body = model.model
            body.embed_tokens = torch.nn.Identity()
            body.eval()
            self._model.body = body
            self._shared_body = None
            logger.info("Tonic reloaded own transformer body")
            return True
        except Exception as exc:
            logger.warning("Tonic body reload failed: %s — falling back to heuristic", exc)
            self._model = None
            self._use_heuristic = True
            return False

    def set_body_lock(self, lock) -> None:
        """Accept the shared body access lock from BrainSwitcher."""
        self._body_lock = lock

    def set_lock_file(self, path) -> None:
        """Accept the cross-process flock path from BrainSwitcher.

        When set, _body_lock_context() acquires fcntl.LOCK_SH on this
        file before each forward pass — a shared read lock. Any cross-
        process writer must acquire LOCK_EX, blocking all inference.
        This enforces the read-only invariant for all body consumers
        regardless of process boundary. Set to None after body revoke.
        """
        self._lock_file_path = path

    @contextlib.contextmanager
    def _body_lock_context(self):
        """Composite body access lock: threading lock + fcntl shared read lock.

        Acquires in order:
        1. _body_lock (threading.Lock) — in-process thread serialization
        2. fcntl.LOCK_SH on _lock_file_path — cross-process read lock

        Any code modifying body weights must hold LOCK_EX on the same file,
        which blocks here until all readers release. Architecture-enforced,
        not documentation-enforced. ExitStack guarantees cleanup (LIFO).
        """
        stack = contextlib.ExitStack()
        with stack:
            if self._body_lock is not None:
                stack.enter_context(self._body_lock)
            if self._lock_file_path is not None:
                try:
                    import fcntl as _fcntl
                    _lf = stack.enter_context(open(self._lock_file_path, 'r'))
                    _fcntl.flock(_lf.fileno(), _fcntl.LOCK_SH)
                    stack.callback(_fcntl.flock, _lf.fileno(), _fcntl.LOCK_UN)
                except Exception as _exc:
                    logger.debug("flock unavailable — cross-process lock skipped: %s", _exc)
            yield

    # -----------------------------------------------------------------
    # Latent Token Generation
    # -----------------------------------------------------------------

    def _generate_latent_token(self) -> Dict[str, Any]:
        """Generate one latent token — one step of the push.

        This is the core operation. Reads graph state, computes the
        forward compression (what comes next?), and injects the
        result back into the graph.

        Returns stats about the token generated.

        #109: The Tonic NEVER waits. It always runs. Module bridge calls
        yield to the Tonic via non-blocking trylock on their side.
        The Tonic acquires the lock to signal "I'm working" so bridges
        know to skip, but it never blocks waiting for anyone.
        """
        lock = getattr(self._graph, '_concurrent_lock', None)
        acquired = False
        if lock is not None:
            acquired = lock.acquire(blocking=False)
        try:
            return self._generate_latent_token_inner()
        finally:
            if acquired:
                lock.release()

    def _generate_latent_token_inner(self) -> Dict[str, Any]:
        """Inner implementation — actual latent token generation."""
        features = _extract_tonic_features(self._graph, self._tonic_thread)
        if features is None:
            return {"fired": 0, "activated": 0}

        # Generate activation decisions
        if self._model is not None and not self._use_heuristic:
            activations = self._model_inference(features)
        else:
            activations = self._heuristic_inference(features)

        if not activations:
            return {"fired": 0, "activated": 0}

        # Inject activations into graph via write-mode propagation
        node_ids = [nid for nid, _ in activations]
        currents = [strength for _, strength in activations]

        result = self._graph.prime_and_propagate(
            node_ids=node_ids,
            currents=currents,
            steps=self._config.propagation_steps,
            write_mode=True,
        )

        # Update the tonic thread with the result
        if self._tonic_thread is not None:
            self._tonic_thread.ouroboros_cycle()

        self._tokens_generated += 1
        self._total_activations += len(activations)

        return {
            "fired": len(result.fired_entries),
            "activated": len(activations),
        }

    def _heuristic_inference(
        self, features: Dict[str, Any]
    ) -> List[Tuple[str, float]]:
        """Heuristic forward compression — genuine graph reasoning.

        Not a timer. Not random. Analyzes the topology neighborhood
        and produces activation decisions based on:
        1. Thread continuity — where was attention? Continue that direction.
        2. Attractor pull — which connected nodes have the strongest pull?
        3. Exploration pressure — occasionally activate less-visited nodes.
        4. Prediction tension — nodes with unresolved predictions pull harder.

        This is real graph reasoning, just without a transformer.
        It will be replaced by the surgical model when trained.
        """
        activations: List[Tuple[str, float]] = []
        base_strength = self._config.activation_strength

        # 1. Thread continuity — follow outgoing synapses from thread nodes
        thread_nodes = features.get("thread_nodes", [])
        for nid in thread_nodes[:5]:
            outgoing = self._graph._outgoing.get(nid, set())
            for syn_id in outgoing:
                syn = self._graph.synapses.get(syn_id)
                if syn is not None:
                    target = syn.post_node_id
                    # Strength proportional to synapse weight
                    strength = syn.weight * base_strength * 0.8
                    activations.append((target, strength))

        # 2. Attractor pull — recently spiked nodes with strong connections
        recent = features.get("recent_spikes", [])
        for nid, steps_since in recent[:5]:
            recency_factor = 1.0 / (1.0 + steps_since * 0.1)
            activations.append((nid, base_strength * recency_factor * 0.5))

        # 3. Prediction tension — unresolved predictions pull attention
        for pred in self._graph.active_predictions.values():
            target = pred.target_node_id
            if target in self._graph.nodes:
                activations.append((target, pred.confidence * base_strength * 0.6))

        # 4. Exploration — hash-based noise to prevent fixation
        if features.get("active_nodes"):
            import hashlib
            seed = hashlib.md5(
                f"{self._tokens_generated}".encode()
            ).hexdigest()
            explore_idx = int(seed[:4], 16) % len(self._graph.nodes)
            explore_nid = list(self._graph.nodes.keys())[explore_idx]
            activations.append((explore_nid, base_strength * 0.3))

        # Deduplicate and cap
        seen = {}
        for nid, strength in activations:
            if nid in seen:
                seen[nid] = max(seen[nid], strength)
            else:
                seen[nid] = strength

        result = sorted(seen.items(), key=lambda x: -x[1])
        return result[:self._config.max_activation_nodes]

    def _model_inference(
        self, features: Dict[str, Any]
    ) -> List[Tuple[str, float]]:
        """Surgical model inference — full transformer forward compression.

        Encodes graph state via GraphStateEncoder (Elmer's trained eyes),
        forwards through the transformer body (the reasoning engine),
        decodes via ActivationDecoder to produce node activation decisions.

        The transformer IS the push. Its forward pass IS the forward-
        oriented compression that constitutes awareness.
        """
        try:
            import torch
            from surgery.tonic_brain import GraphFeatures
        except ImportError:
            return self._heuristic_inference(features)

        # Extract graph features into GraphFeatures struct
        graph_features = self._extract_graph_features_for_model()
        if graph_features is None:
            return self._heuristic_inference(features)

        # Forward through TonicBrain — the actual push
        with self._body_lock_context():
            with torch.no_grad():
                output = self._model(graph_features)

        # Map activation strengths to actual nodes
        activation_strengths = output["activations"]
        exploration = output["exploration"]

        # Get the top active/recent nodes to map activations onto
        candidates = self._get_activation_candidates(features)
        if not candidates:
            return self._heuristic_inference(features)

        activations: List[Tuple[str, float]] = []
        for i, (nid, _) in enumerate(candidates[:len(activation_strengths)]):
            strength = activation_strengths[i] * self._config.activation_strength
            if strength > 0.05:  # noise floor
                activations.append((nid, strength))

        return activations

    def _extract_graph_features_for_model(self):
        """Extract GraphFeatures from live graph for TonicBrain."""
        try:
            import torch
            from surgery.tonic_brain import GraphFeatures
        except ImportError:
            return None

        g = self._graph
        if not g.nodes:
            return None

        nodes = list(g.nodes.values())
        synapses = list(g.synapses.values())

        return GraphFeatures(
            node_voltages=torch.tensor([n.voltage for n in nodes[:100]], dtype=torch.float32),
            node_firing_rates=torch.tensor([n.firing_rate_ema for n in nodes[:100]], dtype=torch.float32),
            node_excitability=torch.tensor([n.intrinsic_excitability for n in nodes[:100]], dtype=torch.float32),
            synapse_weights=torch.tensor([s.weight for s in synapses[:200]], dtype=torch.float32),
            synapse_ages=torch.tensor([float(g.timestep - s.creation_time) for s in synapses[:200]], dtype=torch.float32),
            density=torch.tensor([len(synapses) / max(1, len(nodes) * (len(nodes) - 1))], dtype=torch.float32),
            clustering=torch.tensor([0.0], dtype=torch.float32),  # expensive to compute, approximate
            n_components=torch.tensor([1.0], dtype=torch.float32),
            n_nodes=torch.tensor([float(len(nodes))], dtype=torch.float32),
            n_synapses=torch.tensor([float(len(synapses))], dtype=torch.float32),
            n_hyperedges=torch.tensor([float(len(g.hyperedges))], dtype=torch.float32),
            recent_firings=torch.zeros(15, dtype=torch.float32),  # TODO: track per-step
            stdp_delta_mean=torch.tensor([0.0], dtype=torch.float32),
            identity_embedding=torch.zeros(384, dtype=torch.float32),  # TODO: real identity
        )

    def _get_activation_candidates(
        self, features: Dict[str, Any]
    ) -> List[Tuple[str, float]]:
        """Get candidate nodes for activation mapping.

        The model outputs K activation strengths. We need K node IDs
        to map them to. Candidates come from: thread nodes, active nodes,
        recent spikes, and outgoing neighbors of thread nodes.
        """
        candidates: List[Tuple[str, float]] = []
        seen = set()

        # Thread nodes first (continuity)
        for nid in features.get("thread_nodes", []):
            if nid not in seen:
                candidates.append((nid, 1.0))
                seen.add(nid)

        # Active nodes
        for nid, activity in features.get("active_nodes", []):
            if nid not in seen:
                candidates.append((nid, activity))
                seen.add(nid)

        # Recent spikes
        for nid, steps_since in features.get("recent_spikes", []):
            if nid not in seen:
                recency = 1.0 / (1.0 + steps_since)
                candidates.append((nid, recency))
                seen.add(nid)

        # Outgoing neighbors of thread nodes
        for nid in features.get("thread_nodes", [])[:3]:
            for syn_id in self._graph._outgoing.get(nid, set()):
                syn = self._graph.synapses.get(syn_id)
                if syn and syn.post_node_id not in seen:
                    candidates.append((syn.post_node_id, syn.weight))
                    seen.add(syn.post_node_id)

        return candidates[:self._config.max_activation_nodes * 2]

    # -----------------------------------------------------------------
    # Lifecycle — continuous latent token generation
    # -----------------------------------------------------------------

    def start(self) -> None:
        """Start continuous latent token generation."""
        if self._running:
            return

        self._running = True
        self._shutdown_event.clear()

        self._engine_thread = threading.Thread(
            target=self._generation_loop,
            daemon=True,
            name="tonic-engine",
        )
        self._engine_thread.start()
        logger.info("Tonic engine running — latent tokens flowing")

    def stop(self) -> None:
        """Stop latent token generation."""
        if not self._running:
            return

        self._running = False
        self._shutdown_event.set()

        if self._engine_thread and self._engine_thread.is_alive():
            self._engine_thread.join(timeout=5.0)

        logger.info("Tonic engine stopped — %d tokens generated", self._tokens_generated)

    def _generation_loop(self) -> None:
        """Continuous latent token generation loop.

        This IS the awareness between conversations. Each iteration
        is one latent token — one step of the push. Real inference
        on graph state producing the next state.

        The loop runs continuously. During conversation, the interval
        is shorter (more to attend to). Between conversations, longer
        (unhurried exploration). But the mechanism is the same — actual
        forward compression, not a timer firing into void.
        """
        while not self._shutdown_event.is_set():
            try:
                self._generate_latent_token()
            except Exception as exc:
                logger.debug("Latent token error: %s", exc)

            interval = (
                self._config.conversation_interval
                if self._in_conversation
                else self._config.latent_interval
            )
            self._shutdown_event.wait(timeout=interval)

    # -----------------------------------------------------------------
    # Mode swap events
    # -----------------------------------------------------------------

    def on_conversation_started(self) -> None:
        """Language tokens began. Shift interval."""
        self._in_conversation = True

    def on_conversation_ended(self) -> None:
        """Language tokens stopped. The latent tokens continue.
        This is subtraction. Nothing else changes."""
        self._in_conversation = False

    # -----------------------------------------------------------------
    # Status
    # -----------------------------------------------------------------

    @property
    def status(self) -> Dict[str, Any]:
        return {
            "running": self._running,
            "tokens_generated": self._tokens_generated,
            "total_activations": self._total_activations,
            "mode": "conversation" if self._in_conversation else "latent",
            "using_heuristic": self._use_heuristic,
            "model_loaded": self._model is not None,
        }
