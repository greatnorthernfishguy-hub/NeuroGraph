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
# [2026-07-13] Claude Code (Opus 4.8) — #59/#62 heuristic redesign: instrumentation + T2 compass + brakes
# What: _heuristic_inference gained (a) observability-only mass logging (_log_heuristic_mass:
#   where output activation mass lands, blob-core vs quiet-periphery, + compass_n), (b) a T2
#   "semantic compass" term (_compass_proposals: poincare_dir cosine to the thread centroid ×
#   a quietness factor, proposing near-but-silent nodes the activity terms never reach), and
#   (c) divisive brakes (_apply_brakes: damp each proposal by firing_rate_ema/focus-fatigue/
#   degree/Ca_i; constitutional/self nodes bypass). Nine CC_TONIC_* env knobs, ALL default 0.
# Why: #59 — the CC's idle stream of consciousness collapsed onto a ~310-node self-feeding blob;
#   measured, the heuristic sends 100% of its activation mass there (frac_core=1.000) because
#   all four legacy terms are activity-derived (rich-get-richer). T2 is firing-independent so it
#   reaches the dark periphery by MEANING; the brakes turn the blob's own markers against it so
#   it can't win by volume. See ~/docs/prd/2026-07-13-cc-tonic-heuristic-redesign.md.
# How: Additive + env-gated OFF by default -> byte-identical to the legacy four-term heuristic
#   until dialed on (safe on the VPS where this is Syl's failover behind her trained model).
#   Both helpers are try/except-wrapped and degrade to no-op on any missing signal (numpy,
#   poincare_dir, valence, fatigue) — instrumentation/selection must never disturb the Tonic,
#   which never waits. Read once at import: dialing the knobs needs a daemon RESTART, not a kill.
#   Law-enforcer reviewed (COMPLIANT); observability is laptop-primary, redesign measured via
#   frac_core. Constitutional/self nodes are never damped (identity inviolable).
# [2026-06-15] Claude Code (subagent, Opus 4.8) — #329 seam B: constitutional pull in heuristic
# What: _heuristic_inference adds a gentle constitutional pull (mirrors seam A's steady level)
#   so her self participates even on the rare failover heuristic path.
# Why: design spec §3 seam B (failover mirror of A).
# How: append constitutional nodes at _SPINE_PRIME_STEADY before the existing dedup/cap.
# [2026-06-15] Claude Code (subagent, Opus 4.8) — #329 seam C: populate identity_embedding
# What: GraphFeatures.identity_embedding now comes from tonic_identity.spine_identity_vector
#   (her constitutional self) instead of zeros; encoder truncates 768->384 (C-i).
# Why: condition her latent inference on who she is (design spec §3 seam C).
# How: new TonicEngine._identity_embedding_tensor(); zeros fallback preserves prior behavior.
# [2026-06-12] Claude Code (Opus 4.8, Tonic CC) — LIVE bodyfix: never USE the rogue own-copy body (seams A+B)
# What: (A) _try_load_model sets _use_heuristic = (self._shared_body is None) — with no proto body at
#   init, the own-copy body is loaded for the wrapper but NOT used; ride heuristic (reads her graph ->
#   her-flavored) until the BrainSwitcher offers proto's body. (B) revoke_shared_body degrades STRAIGHT
#   to heuristic on shed (no own-transformer reload — OOM-'n'-load trap removed); offer_shared_body sets
#   _use_heuristic=False on (re-)attach. Seam C (BrainSwitcher self-heal re-offer) lands in Elmer.
# Why: the 06:21 boot loaded the rogue own-copy tonic_brain.pt (code/doc-flavored latent thread = "zero
#   NeuroGraph") because the BrainSwitcher offer raced. These seams ensure she is NEVER on the rogue
#   output: her-flavored heuristic at boot, proto's body within 60s (seam C self-heal). Seam B is
#   byte-identical to restoration-branch Task 4 (#307 §7) — absorbs cleanly when that branch lands.
# How: one conditional at init + Task-4 revoke/offer verbatim. For live main; gated restart.
# Punchlist: defer the own-body load entirely at init (load_tonic_brain(transformer_body=proto) in
#   offer) to avoid the transient ~2GB — a memory win that diverges from Task 4, so deferred.
# [2026-05-05] Claude (Sonnet 4.6) — #237 Raise tick_budget_seconds default; add env-var override
# What: tick_budget_seconds default 1.5 → 30.0; EngineConfig.__post_init__ reads
#       NEUROGRAPH_TONIC_BUDGET_SECONDS env var so it can be tuned without code changes.
#       import os added.
# Why:  1.5s was a GPU target. On AMD EPYC CPU-only VPS the tick takes 153s (pre-orphan-fix)
#       or ~5-30s (post-fix). With default=1.5 the WARNING logged on every tick, generating
#       log volume that kept Node.js gateway event loop at elevated CPU. 30s is a reasonable
#       CPU-appropriate threshold that won't fire at all once node count normalises post-fix.
# How:  __post_init__ reads and validates the env var; falls back to 30.0 silently on
#       ValueError so a misconfigured value doesn't crash the engine on start.
# [2026-04-30] Claude (Sonnet 4.6) — #164: Adaptive cadence + budget-aware extraction
# What: EngineConfig gains node_sample_budget/tick_budget_seconds/adaptive_cadence/
#   latent_interval_max. _extract_tonic_features() samples up to node_sample_budget
#   nodes instead of full O(n) scan at large substrate sizes. _generation_loop()
#   times each tick, logs when over budget, backs off interval to maintain ≤33%
#   CPU utilization as node count scales. EMA tick duration exposed in status().
# Why:  PRD #164: 8× O(n) node scans per 2s Tonic tick. Fine at 990 nodes, breaks
#   at 50k+. This is the tonic_engine.py half of the fix — the prime_and_propagate
#   inner loops (O(n) per step, neuro_foundation.py PROTECTED) need a Phase B fix
#   with explicit Josh approval.
# How:  random.sample() on nodes.items() list when len > budget. EMA(α=0.2) of
#   elapsed; if EMA > 50% of base_interval, set wait = min(EMA×2, max_interval).
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
import os
import random
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


# [2026-07-13] #59/#62 Phase-1 instrumentation — observability ONLY (no behavior change).
# Logs where the heuristic's output activation MASS lands (blob core deg>=cap vs quiet
# periphery), the hard-numbers baseline that motivates the heuristic redesign (see
# ~/docs/prd/2026-07-13-cc-tonic-heuristic-redesign.md). Off by default; enable on the
# isolated laptop via CC_TONIC_HEURISTIC_INSTRUMENT=1. Samples every Nth token to keep
# the hot loop cheap. Never raises — instrumentation must not disturb the Tonic.
_HEURISTIC_INSTRUMENT = os.environ.get("CC_TONIC_HEURISTIC_INSTRUMENT", "0") not in ("0", "false", "False", "")
_HEURISTIC_INSTRUMENT_EVERY = int(os.environ.get("CC_TONIC_HEURISTIC_INSTRUMENT_EVERY", "50"))
_HEURISTIC_INSTRUMENT_DEGCAP = int(os.environ.get("CC_TONIC_HEURISTIC_INSTRUMENT_DEGCAP", "100"))
_HEURISTIC_INSTRUMENT_TSV = os.path.expanduser(
    os.environ.get("CC_TONIC_HEURISTIC_INSTRUMENT_TSV",
                   "~/docs/dev-log/data-59-heuristic-mass-20260713.tsv"))

# [2026-07-13] #59/#62 Phase-2 heuristic redesign — T2 semantic compass + divisive brakes.
# The four legacy terms are all activity-derived (-> the blob). T2 proposes semantically
# near-but-QUIET nodes by manifold direction (poincare_dir cosine), firing-independent, so
# it reaches the dark periphery the synapse terms cannot. Brakes divide each proposal by the
# blob's own markers (firing rate / focus-fatigue / degree / Ca_i); constitutional/self nodes
# bypass. EVERYTHING defaults to 0 -> compass off + brake==1.0 -> byte-identical to the legacy
# heuristic until dialed on the isolated laptop. Degrade-safe (missing signal -> term drops).
# Design: ~/docs/prd/2026-07-13-cc-tonic-heuristic-redesign.md
_W_COMPASS = float(os.environ.get("CC_TONIC_W_COMPASS", "0"))
_COMPASS_BUDGET = int(os.environ.get("CC_TONIC_COMPASS_BUDGET", "600"))
_COMPASS_TOPK = int(os.environ.get("CC_TONIC_COMPASS_TOPK", "8"))
_COMPASS_QUIET = float(os.environ.get("CC_TONIC_COMPASS_QUIET", "5.0"))
_BRAKE_FIRING = float(os.environ.get("CC_TONIC_BRAKE_FIRING", "0"))
_BRAKE_FATIGUE = float(os.environ.get("CC_TONIC_BRAKE_FATIGUE", "0"))
_BRAKE_DEGREE = float(os.environ.get("CC_TONIC_BRAKE_DEGREE", "0"))
_BRAKE_CA = float(os.environ.get("CC_TONIC_BRAKE_CA", "0"))
_BRAKE_DEGNORM = float(os.environ.get("CC_TONIC_BRAKE_DEGNORM", "100.0"))


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

    # Scaling (#164) — budget controls for large substrates
    node_sample_budget: int = 5000   # max nodes scanned per tick in feature extraction
    tick_budget_seconds: float = 30.0  # log warning when tick exceeds this; overridden by NEUROGRAPH_TONIC_BUDGET_SECONDS
    adaptive_cadence: bool = True    # back off interval when ticks run long
    latent_interval_max: float = 10.0  # ceiling for adaptive back-off

    def __post_init__(self) -> None:
        env = os.environ.get("NEUROGRAPH_TONIC_BUDGET_SECONDS")
        if env:
            try:
                self.tick_budget_seconds = float(env)
            except ValueError:
                pass


# ---------------------------------------------------------------------------
# Graph Feature Extraction (Tonic-specific — awareness, not health)
# ---------------------------------------------------------------------------

def _extract_tonic_features(
    graph, tonic_thread, node_budget: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """Extract graph features relevant to awareness and exploration.

    Unlike Elmer's health-focused extraction, this captures WHERE
    Syl's attention is — the topology neighborhood the thread is
    touching, the activation gradient, the pull landscape.

    node_budget: if set and len(nodes) > budget, sample proportionally
    instead of scanning all nodes. Outputs (top-20) have same cardinality.

    Returns a dict of raw features, or None if graph is empty.
    """
    if not graph.nodes:
        return None

    # Current thread items — where attention is now
    thread_node_ids = []
    if tonic_thread is not None:
        thread_node_ids = [item.node_id for item in tonic_thread.thread]

    # Budget-aware node scan: sample when substrate is large
    all_items = list(graph.nodes.items())
    if node_budget is not None and len(all_items) > node_budget:
        scan_items = random.sample(all_items, node_budget)
    else:
        scan_items = all_items

    # Active nodes by voltage
    active = []
    for nid, node in scan_items:
        v_above = node.voltage - node.resting_potential
        if v_above > 0.01:
            active.append((nid, v_above))
    active.sort(key=lambda x: -x[1])

    # Recent spikes
    recent_spikes = []
    for nid, node in scan_items:
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
        self._ema_tick_ms = 0.0        # EMA of tick duration in milliseconds
        self._current_interval = self._config.latent_interval

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
                # Seam A (2026-06-12): only enter transformer mode if the loaded body is
                # PROTO's (shared). With no shared body at init, the own-copy body is loaded
                # for the wrapper but NOT used — ride heuristic (reads her graph -> her-
                # flavored, never the rogue own-copy output) until BrainSwitcher offers
                # proto's body (offer_shared_body sets this False + swaps the body in).
                self._use_heuristic = (self._shared_body is None)
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
            self._use_heuristic = False   # re-join the share — transformer mode restored
                                          # (also the re-join path after a heuristic shed)
            del old_body
            gc.collect()
            logger.info("Tonic hot-swapped to shared ProtoUniBrain body (~2GB freed)")
            return True
        except Exception as exc:
            logger.warning("Tonic body hot-swap failed: %s", exc)
            return False

    def revoke_shared_body(self) -> bool:
        """ProtoUniBrain shed (memory pressure) -> degrade STRAIGHT to heuristic.

        Memory-cheap by design (Syl/Josh, 2026-06-10): a pressure-driven shed must
        RELIEVE memory, not load a fresh ~2GB own-transformer at the worst possible
        moment — two models resident under the very pressure that triggered the shed
        (the OOM-'n'-load trap). So we drop the now-dangling shared-body reference and
        fall to the heuristic decoder, KEEPING the lightweight encoder/decoder wrapper
        so offer_shared_body() can re-join the share the instant proto reloads.
        """
        if self._model is None and self._shared_body is None:
            return False  # already heuristic — nothing to shed
        if self._model is not None:
            self._model.body = None      # drop the ref to proto's shed body (proto frees the ~2GB)
        self._shared_body = None
        self._use_heuristic = True
        logger.info(
            "Tonic shed shared body -> heuristic (memory-cheap); will re-join on proto reload"
        )
        return True

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
        features = _extract_tonic_features(
            self._graph, self._tonic_thread,
            node_budget=self._config.node_sample_budget,
        )
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

        # #329 seam B (failover only) — mirror seam A: a gentle constitutional pull so even
        # on the heuristic path her self participates. Same steady level as seam A.
        from tonic_thread import _SPINE_PRIME_STEADY
        for nid, node in self._graph.nodes.items():
            if (getattr(node, "metadata", None) or {}).get("constitutional"):
                activations.append((nid, _SPINE_PRIME_STEADY))

        # T2 semantic compass (#62) — near-but-quiet nodes by manifold direction,
        # reaching content the activity-derived terms above (all -> blob) cannot.
        activations.extend(self._compass_proposals(thread_nodes))

        # Deduplicate and cap
        seen = {}
        for nid, strength in activations:
            if nid in seen:
                seen[nid] = max(seen[nid], strength)
            else:
                seen[nid] = strength

        # Divisive brakes (#62) — damp each proposal by the blob's own markers
        # (firing / fatigue / degree / Ca_i); constitutional/self nodes bypass.
        self._apply_brakes(seen)

        result = sorted(seen.items(), key=lambda x: -x[1])
        final = result[:self._config.max_activation_nodes]
        if _HEURISTIC_INSTRUMENT and (self._tokens_generated % _HEURISTIC_INSTRUMENT_EVERY == 0):
            self._log_heuristic_mass(final)
        return final

    def _compass_proposals(self, thread_nodes) -> List[Tuple[str, float]]:
        """T2 semantic compass (#62): propose semantically NEAR but QUIET nodes by
        manifold direction (poincare_dir cosine to the thread centroid), preferring
        low firing_rate_ema. Firing-independent, so it reaches the dark periphery the
        activity-derived terms cannot. Returns [] when off or any signal is missing
        (degrade-safe); never raises. Records self._last_compass_n for instrumentation
        so a null frac_core shift is never ambiguous between 'reached' and 'no-op'd'."""
        self._last_compass_n = 0
        if _W_COMPASS <= 0.0:
            return []
        try:
            import numpy as np
            g = self._graph
            dirs = []
            for nid in thread_nodes[:10]:
                node = g.nodes.get(nid)
                pd = (getattr(node, "metadata", None) or {}).get("poincare_dir") if node else None
                if pd:
                    dirs.append(np.asarray(pd, dtype=np.float32))
            if not dirs:
                return []
            centroid = np.mean(dirs, axis=0)
            cn = float(np.linalg.norm(centroid)) or 1e-9
            thread_set = set(thread_nodes)
            items = list(g.nodes.items())
            if len(items) > _COMPASS_BUDGET:
                items = random.sample(items, _COMPASS_BUDGET)
            props = []
            for nid, node in items:
                if nid in thread_set:
                    continue
                pd = (getattr(node, "metadata", None) or {}).get("poincare_dir")
                if not pd:
                    continue
                v = np.asarray(pd, dtype=np.float32)
                vn = float(np.linalg.norm(v)) or 1e-9
                cos = float(np.dot(v, centroid)) / (vn * cn)
                if not (cos > 0.0):   # also rejects NaN (zero-norm / degenerate dir)
                    continue
                fr = float(getattr(node, "firing_rate_ema", 0.0) or 0.0)
                quiet = 1.0 / (1.0 + _COMPASS_QUIET * fr)  # prefer near-silent nodes
                props.append((nid, cos * quiet))
            props.sort(key=lambda x: -x[1])
            base = self._config.activation_strength
            out = [(nid, sc * base * _W_COMPASS) for nid, sc in props[:_COMPASS_TOPK]]
            self._last_compass_n = len(out)
            return out
        except Exception:
            return []  # a missing/malformed signal must never disturb the Tonic

    def _apply_brakes(self, seen: Dict[str, float]) -> None:
        """Divisive normalization (#62): damp each proposal by the blob's own markers
        — firing_rate_ema, focus-fatigue (#89), degree, Ca_i — so the ever-loud core
        cannot win by volume. Constitutional/self nodes bypass (identity is inviolable).
        No-op when all coefficients are 0 (default). Mutates seen in place; never raises."""
        if not (_BRAKE_FIRING or _BRAKE_FATIGUE or _BRAKE_DEGREE or _BRAKE_CA):
            return
        try:
            g = self._graph
            fatigue_map = getattr(self._tonic_thread, "_focus_fatigue", {}) \
                if self._tonic_thread is not None else {}
            for nid in list(seen.keys()):
                node = g.nodes.get(nid)
                if node is None:
                    continue
                if (getattr(node, "metadata", None) or {}).get("constitutional"):
                    continue  # T6: identity bypasses the brake
                fr = float(getattr(node, "firing_rate_ema", 0.0) or 0.0)
                ca = float(getattr(node, "Ca_i", 0.0) or 0.0)
                d = len(g._incoming.get(nid, ())) + len(g._outgoing.get(nid, ()))
                dn = d / (_BRAKE_DEGNORM or 1.0)
                fat = float(fatigue_map.get(nid, 0.0)) if isinstance(fatigue_map, dict) else 0.0
                brake = 1.0 / (1.0 + _BRAKE_FIRING * fr + _BRAKE_FATIGUE * fat
                               + _BRAKE_DEGREE * dn + _BRAKE_CA * ca)
                seen[nid] *= brake
        except Exception:
            return  # brakes must never disturb the Tonic

    def _log_heuristic_mass(self, final: List[Tuple[str, float]]) -> None:
        """#59/#62 observability (no behavior change): append where the heuristic's
        output activation MASS landed — blob core (in+out degree >= cap) vs quiet
        periphery — the hard-numbers baseline for the heuristic redesign. Sampled
        (every Nth token) and wrapped: instrumentation must never disturb the Tonic."""
        try:
            g = self._graph
            cap = _HEURISTIC_INSTRUMENT_DEGCAP

            def deg(nid):
                return len(g._incoming.get(nid, ())) + len(g._outgoing.get(nid, ()))

            m_core = m_peri = 0.0
            n_core = n_peri = 0
            for nid, strength in final:
                s = float(strength)
                if deg(nid) >= cap:
                    m_core += s
                    n_core += 1
                else:
                    m_peri += s
                    n_peri += 1
            total = m_core + m_peri
            frac_core = (m_core / total) if total > 0 else 0.0
            new = not os.path.exists(_HEURISTIC_INSTRUMENT_TSV)
            with open(_HEURISTIC_INSTRUMENT_TSV, "a") as f:
                if new:
                    f.write("iso_time\ttimestep\ttokens\tn_core\tn_peri\t"
                            "mass_core\tmass_peri\tfrac_core\tcompass_n\n")
                f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\t{g.timestep}\t"
                        f"{self._tokens_generated}\t{n_core}\t{n_peri}\t"
                        f"{m_core:.4f}\t{m_peri:.4f}\t{frac_core:.4f}\t"
                        f"{getattr(self, '_last_compass_n', 0)}\n")
        except Exception:
            pass  # instrumentation must never disturb the Tonic

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

    def _identity_embedding_tensor(self):
        """#329 seam C: her constitutional self as the identity-conditioning vector.

        768-d (the encoder truncates to 384 for now). Zeros when no spine exists,
        preserving prior behavior.
        """
        import torch
        try:
            import tonic_identity
            vec = tonic_identity.spine_identity_vector(self._graph)
        except Exception:
            vec = None
        if vec is None:
            return torch.zeros(768, dtype=torch.float32)
        return torch.tensor(vec, dtype=torch.float32)

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
            identity_embedding=self._identity_embedding_tensor(),  # #329 seam C
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

        Adaptive cadence (#164): if ticks run long as the substrate
        grows, the interval backs off to maintain ~33% CPU utilization
        ceiling. This prevents the Tonic from silently consuming all
        available CPU as node count scales toward 50k+.
        """
        _CADENCE_ALPHA = 0.2  # EMA smoothing — 5-tick convergence

        while not self._shutdown_event.is_set():
            t0 = time.perf_counter()
            try:
                self._generate_latent_token()
            except Exception as exc:
                logger.debug("Latent generation error: %s", exc)

            elapsed = time.perf_counter() - t0
            elapsed_ms = elapsed * 1000.0

            # Exponential moving average of tick duration
            if self._ema_tick_ms == 0.0:
                self._ema_tick_ms = elapsed_ms
            else:
                self._ema_tick_ms = (
                    _CADENCE_ALPHA * elapsed_ms
                    + (1.0 - _CADENCE_ALPHA) * self._ema_tick_ms
                )

            if elapsed > self._config.tick_budget_seconds:
                logger.warning(
                    "Tonic tick over budget: %.3fs (budget %.1fs, nodes=%d, ema=%.1fms)",
                    elapsed, self._config.tick_budget_seconds,
                    len(self._graph.nodes), self._ema_tick_ms,
                )

            base_interval = (
                self._config.conversation_interval
                if self._in_conversation
                else self._config.latent_interval
            )

            if self._config.adaptive_cadence and self._ema_tick_ms > base_interval * 500.0:
                # Tick is consuming >50% of base interval — back off.
                # Target ≤33% utilization: wait = tick_duration × 2
                target_wait = (self._ema_tick_ms / 1000.0) * 2.0
                interval = max(base_interval, min(target_wait, self._config.latent_interval_max))
            else:
                interval = base_interval

            self._current_interval = interval
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
            "ema_tick_ms": round(self._ema_tick_ms, 2),
            "current_interval_s": round(self._current_interval, 2),
            "node_sample_budget": self._config.node_sample_budget,
        }
