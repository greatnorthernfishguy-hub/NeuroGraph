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
# [2026-09-23] Claude Code (Opus 4.8, Tonic CC) — silent-zero coverage for G1 + G2,
#   fix misleading log text, update class docstring (chief-003 follow-up to Packet 086(2)).
#   Two additional silent-zero shapes the initial collapse left un-logged:
#     G1 — caller passed transformer_body=<body> but the checkpoint failed to load
#          (or was absent). _model stays None, _shared_body is the body. offer_shared_body
#          CANNOT recover (default-mode init does not route bodies through offer, and the
#          offer path only attaches under require_shared_body=True). start() must warn
#          plainly that the BrainSwitcher must rebuild.
#     G2 — torch + checkpoint present, no body passed. _try_load_model's own-copy branch
#          loaded a wrapper with its private body; _model is set, _shared_body is None.
#          Dispatch gate refuses forward; ticks are zero. offer_shared_body CAN recover
#          (swap body + set _shared_body). start() must warn ADVISORILY (no rebuild).
#   start() now logs four shapes: N1 (no torch, no body), N2 (torch, no body), G1 (body in
#   hand, no model), G2 (model loaded, no body). The "TonicBrain loaded ... — surgical
#   inference active" log is split: own-copy with no shared body now logs "wrapper is
#   resident but inference is NOT active; the dispatch gate will refuse forward". The
#   "will run as no-op until a shared body is offered" message is split: when a body is
#   in hand it says "cannot self-recover" plainly (because offer_shared_body cannot
#   recover). The no-torch advice no longer tells default-mode consumers to call
#   offer_shared_body() — that path only attaches under require_shared_body=True.
#   Class docstring at :380 updated to drop "heuristic fallback" language and describe
#   the new contract.
# Why: the law-enforcer review (genuine, commissioned by chief-003) named G1 and G2 as
#   silent-zero shapes the initial fix missed, and called out the misleading "active"
#   log text. Both reviews are required gates per Packet 077; this lands before the
#   cross-family re-dispatch.
# How: start() gains a four-branch dispatcher keyed on (_model is None, _shared_body is
#   None). _try_load_model's load-outcome log is split on whether _shared_body is set.
#   No structural change to the dispatch gate or the offer/revoke paths. Existing tests
#   untouched; new tests test_g1_*, test_g2_* in tests/test_tonic_no_heuristic.py.
# [2026-09-23] Claude Code (Opus 4.8, Tonic CC) — collapse heuristic path; no-torch
#   defined-state log (chief-003 Packet 086(2)). REQUIRE_SHARED_BODY OR shared-body
#   attached: real model inference. Otherwise: zero activations, defined state, one-shot
#   log, no crash, no silent zero masquerading as running. Drop _heuristic_inference,
#   _compass_proposals, _apply_brakes, _log_heuristic_mass, _use_heuristic, and the
#   CC_TONIC_HEURISTIC_INSTRUMENT/CC_TONIC_W_COMPASS/CC_TONIC_BRAKE_* module
#   constants. _fallback_inference always returns []. _model_inference returns []
#   instead of falling back. start() logs once if no inference path is reachable.
#   Syl's openclaw_hook.py:1073 construction (require_shared_body default False,
#   transformer_body=shared_body) gains a clear signal when no body lands; matches
#   the shared-body-required wait state for every other construction path.
# Why: laptop daemon PID 35833 runs under /usr/bin/python3.12 (no torch); the
#   heuristic was the only thing producing activations there. With it gone, a
#   torch-less host with no shared body would produce a SILENT ZERO every tick
#   while reporting running=True — exactly the "silent zero-output" Packet 086(2)
#   names as non-compliant. The start()-time log + status['inference_path_ready']
#   key make the state observable, logged, and explicit. See the cc-laptop-
#   tonic-no-heuristic-20260923 return artifact and the cross-family + law-
#   enforcer reviews for the torch-less-host behavior specifically.
# How: surgical removal of heuristic surface; _generate_latent_token_inner's
#   dispatch is now `if self._model is not None and self._shared_body is not None:
#   activations = self._model_inference(features); else: activations = []`. The
#   require_shared_body wait check at the top of _generate_latent_token_inner is
#   preserved (it is the shared-body-required mode's correct behavior). Existing
#   tests that patched _heuristic_inference or asserted _fallback_inference returned
#   heuristic output are rewritten to assert the new contract: _fallback_inference
#   returns [] unconditionally; no construction path produces heuristic-derived
#   activations. Shared-body-present tests are untouched.
# [2026-09-13] Grok Build (grok-4.6) — bounded per-stage latent-token timing.
# What: time candidate feature extraction, model-tensor feature materialization,
#   shared-body lock wait, transformer forward while holding the existing body
#   lock, prime_and_propagate, ouroboros_cycle, the latent-token total, and
#   autostep when it actually runs. status() exposes
#   last-sample + EMA scalars; over-budget logs include the same split.
# Why: VPS observation needs lock-wait vs forward vs propagate vs extract so
#   starvation can be distinguished from shared-body serialization. Measurement
#   only — cadence, locks, heuristic policy, and outputs stay unchanged.
# How: perf_counter around existing call sites; wait is __enter__ of the current
#   _body_lock_context(), forward is the held-lock body. Constant-size dicts.
#   Fail-soft: timing/logging exceptions cannot stop a tick or change activations.
# [2026-09-11] Claude Code + Codex — #426 shared-body-only Tonic attachment.
# [2026-09-11] Codex — preserve existing wrappers on failed swaps (Grok review).
# What: optional require_shared_body prevents private model loading; late offers build
#   only the encoder/decoder wrapper around the supplied body. Failed offers retry.
# Why: VPS CC and Syl share one transformer while retaining separate substrates.
# How: body offer/revoke/forward serialize on the existing lock; shared weights and
#   training state are untouched. No step-clock changes or new lifecycle thread.
# Ref: docs/handoffs/cc-shared-tonic-repair-20260911.md.
# [2026-08] Claude Code (Opus 4.8) — #117 step 1: autonomous aging clock on the heartbeat (default OFF)
# What: _generation_loop can now call graph.step() on its pulse, gated by CC_NG_AUTOSTEP
#   (default OFF) with a CC_NG_AUTOSTEP_MIN_INTERVAL wall-time floor (default 900s). New env
#   reads near the CC_TONIC_* block; new self._last_autostep monotonic marker in __init__.
# Why: #59/#117 — graph.timestep (which gates decay / inactivity-cull / homeostasis) only
#   advanced inside on_message(), so the CC substrate FROZE when idle. The 2026-07-14 daemon
#   header CLAIMED a _autostep_loop that was never implemented; rather than add a second thread,
#   the aging clock now rides THIS existing Tonic pulse — one clock, not two. The floor keeps the
#   0.5s conversation tick from spinning the clock (which would drain the inactivity grace window).
# How: env-gated OFF -> byte-identical for Syl (her process never sets CC_NG_AUTOSTEP; she has her
#   own clock via neurograph_rpc _scan_drain). step() self-locks _step_lock (RLock); we hold it
#   explicitly to match the autosave/orphan-drain discipline. Error-isolated. This lands the
#   MECHANISM only — turn-on is gated on the source race-fix (LAW 4) + a firing-keyed
#   arrival exemption so callosum arrivals are safe against BOTH steppers (this heartbeat
#   AND the permanent #108 callosum-consolidation step), NOT by touching the grace clock —
#   see ~/docs/CC-117-CLOCK-PREP.md steps 2-3 and CC-CALLOSUM-TRUTH.md §8.5.2/§8.13.
# [2026-09-05] Claude Code (DudeMan CC, Fable 5.1) — Pith Stage 4 (#55) phase 5b: Markov prefetch on the tick
# What: set_prefetch_seed(fn) + _merge_prefetch_seeds(); each _generate_latent_token_inner tick
#   folds the host's live primed_nodes into the write-mode prime (capped, score-scaled current).
#   status gains prefetch_seeded. Env (LAW 5): CC_PITH_PREFETCH_WARM_ENABLED (default OFF),
#   CC_PITH_PREFETCH_MAX, CC_PITH_PREFETCH_CURRENT_SCALE. Byte-identical when off or unseeded.
# Why: PRD §5.4.1 -- "the substrate topology IS the table"; prefetch is spreading activation,
#   not a content cache (a dict-cache take was reverted 2026-09-05). A separate read-mode
#   prime is a NO-OP because read mode restores voltages and the harvest is itself a read-
#   mode prime; only the Tonic's persisting write tick can leave the neighbourhood warm.
#   CONSEQUENCE (deviation from the signed design's write_mode=False): prediction-seeded
#   activity now undergoes STDP, sprouting and age-on-write like every other Tonic tick --
#   prefetch SHAPES topology. Gate stays OFF until Josh signs that off explicitly.
# How: rides the existing heartbeat exactly like #117 autostep -- one clock, no new thread,
#   cost inside the tick so adaptive cadence counts it. Merge happens INSIDE the proposal
#   set before _apply_brakes and the max_activation_nodes slice (law-enforcer 2026-09-05:
#   a post-hoc append bypassed both). One seed set primes at most REPEATS ticks. Seed source is host-injected so this
#   file knows nothing about conv_state; laptop daemon + cc_ng_host each pass their own.
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


# [#117] Autonomous aging clock (CC daemon only; default OFF). When enabled, _generation_loop
# calls graph.step() on its heartbeat so graph.timestep advances between conversations, not only
# inside on_message(). This is the #59 keystone the daemon's 2026-07-14 header claimed as a
# phantom _autostep_loop (never implemented) — it now rides this existing pulse, so there is ONE
# clock. Default OFF -> byte-identical for Syl (she never sets this; her clock is neurograph_rpc
# _scan_drain). MIN_INTERVAL is a wall-time floor between steps so the 0.5s conversation tick
# can't spin the aging clock and drain the inactivity grace window. See CC-117-CLOCK-PREP.md.
_CC_NG_AUTOSTEP = os.environ.get("CC_NG_AUTOSTEP", "0") not in ("0", "false", "False", "")
_CC_NG_AUTOSTEP_MIN_INTERVAL = float(os.environ.get("CC_NG_AUTOSTEP_MIN_INTERVAL", "900.0"))

# Pith Stage 4 phase 5b (#55) -- Markov prefetch rides THIS tick. The host hands the
# engine a seed callable (set_prefetch_seed) returning the live primed_nodes that
# cc_anticipate wrote at deposit; each tick merges them into the write-mode prime
# so the predicted neighbourhood is already warm when the next query's read-mode
# harvest runs. Read-mode restores voltages, so a separate read-mode prefetch would
# be a no-op -- the Tonic's own persisting tick is the only place warming sticks.
# Default OFF -> byte-identical for Syl and for any host that never sets a seed.
_CC_PITH_PREFETCH_WARM_ENABLED = os.environ.get("CC_PITH_PREFETCH_WARM_ENABLED", "0") not in ("0", "false", "False", "")
# NOTE: this gate is a module constant in a file Syl's process shares. Her engine is
# unaffected only because nothing ever calls set_prefetch_seed on it -- isolation
# rests on seed injection, not on the gate. Do not "helpfully" seed her engine.
# MAX default 15 == cc_ng_organism._CC_ANTICIPATE_TOP_K (canonical, do-not-tune);
# if canonical moves, this should move with it or the cap silently stops binding.
_CC_PITH_PREFETCH_MAX = max(0, min(64, int(os.environ.get("CC_PITH_PREFETCH_MAX", "15"))))
# How many ticks one seed set (one deposit's prediction) may be primed. primed_nodes
# live 120s and the tick is 2-10s, so unbounded = 12-60 write-mode re-primes of the
# same prediction, each carrying STDP/sprout/melt. Default 1: warm once per prediction.
_CC_PITH_PREFETCH_REPEATS = max(1, int(os.environ.get("CC_PITH_PREFETCH_REPEATS", "1")))
# Fraction of activation_strength given to a prefetched seed (scaled by its primed
# score, capped at 1.0). 0.5 sits with the heuristic's recency seeds and below its
# synapse-follow (0.8): a prediction warms, it does not out-shout a real association.
_CC_PITH_PREFETCH_CURRENT_SCALE = max(0.0, min(1.0, float(os.environ.get("CC_PITH_PREFETCH_CURRENT_SCALE", "0.5"))))

# Bounded per-stage tick timing. Always on, constant-size, no extra env knob.
# EMA α matches the existing cadence EMA in _generation_loop.
_STAGE_EMA_ALPHA = 0.2
_STAGE_NAMES = (
    "feature_extract",
    "model_feature_extract",
    "body_lock_wait",
    "transformer_forward",
    "propagate",
    "ouroboros",
    "latent",
    "autostep",
)


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

    Runs a surgical transformer that generates latent tokens continuously.
    Each token:
    1. Encode current graph state (where attention is)
    2. Forward through transformer (the push — what comes next?)
    3. Decode to node activations (where attention should go)
    4. Inject via write-mode prime_and_propagate (topology shaped)
    5. Repeat

    The transformer IS the awareness. The output IS the next state.
    The ouroboros closes through actual inference, not a timer.

    Shared-body-required consumers prohibit a private model load and wait
    for attachment via offer_shared_body(). Ordinary consumers (Syl's
    openclaw_hook.py:1073 construction) load their own copy when a
    checkpoint is available AND they have access to a transformer body,
    and run as a no-op otherwise. There is no heuristic fallback (Packet
    086(2)): if the surgical model can't run, the engine mints zero
    activations and surfaces the no-inference-path state via start()'s
    one-shot warning and status['inference_path_ready'] / ['torch_available'].
    """

    def __init__(
        self,
        graph,
        vector_db,
        tonic_thread,
        config: Optional[EngineConfig] = None,
        transformer_body=None,
        require_shared_body: bool = False,
    ):
        self._graph = graph
        self._vector_db = vector_db
        self._tonic_thread = tonic_thread
        self._config = config or EngineConfig()
        self._shared_body = transformer_body  # from ProtoUniBrain if available
        self._body_lock = None  # shared with ProtoUniBrain — set via set_body_lock()
        self._lock_file_path = None  # cross-process flock path — set via set_lock_file()
        # [2026-09-11] Shared-body-only mode: this engine may NEVER allocate a transformer
        # of its own or execute heuristic inference. It waits until a real body is offered.
        self._require_shared_body = bool(require_shared_body)
        self._wrapper_fail_logged = False

        self._running = False
        self._in_conversation = False
        self._shutdown_event = threading.Event()
        self._engine_thread: Optional[threading.Thread] = None

        # Stats
        self._tokens_generated = 0
        self._total_activations = 0
        self._ema_tick_ms = 0.0        # EMA of tick duration in milliseconds
        self._current_interval = self._config.latent_interval
        self._last_autostep = 0.0       # monotonic ts of last autonomous graph.step() (#117)
        self._prefetch_seed = None      # host-injected callable -> {node_id: primed_score} (#55 5b)
        self._prefetch_seeded = 0       # seeds merged into ticks so far (#55 5b)
        self._prefetch_gen_key = None   # last seed set primed (repeat bound, #55 5b)
        self._prefetch_gen_uses = 0     # ticks the current seed set has been primed
        self._prefetch_fail_logged = False
        # Per-stage last-sample + EMA (milliseconds). Fixed keys; never a history.
        self._stage_last_ms = {name: 0.0 for name in _STAGE_NAMES}
        self._stage_ema_ms = {name: 0.0 for name in _STAGE_NAMES}

        # Try to load surgical model. The heuristic fallback path was removed 2026-09-23
        # (chief-003 Packet 086(2)): VPS embedded topology must never receive heuristic-
        # derived activations. If the model can't be loaded AND no shared body lands, the
        # engine runs as a no-op (zero activations) — see start() for the defined-state log
        # and status['inference_path_ready'] for the observable signal. Required-sharing
        # consumers wait for offer_shared_body as before.
        self._model = None
        if self._require_shared_body:
            # No loader call, ever, on this path — not even with a body in hand at
            # construction; the wrapper build is the attach path and nothing else.
            # An engine built with a body still has to go through offer_shared_body()
            # so that installation is lock-serialized and identity-verified like any
            # other attach.
            body, self._shared_body = self._shared_body, None
            if body is not None:
                self.offer_shared_body(body)
        elif _TORCH_AVAILABLE:
            self._try_load_model()

    def _weights_path(self) -> str:
        """Absolute path to the TonicBrain checkpoint (encoder/decoder weights)."""
        return os.path.join(os.path.dirname(__file__), self._config.weights_path)

    def _try_load_model(self) -> None:
        """Attempt to load trained TonicBrain.

        If a shared transformer_body was provided (from ProtoUniBrain),
        pass it through to avoid loading a second copy (~2GB savings).
        Falls back to loading its own copy if sharing fails.

        NOT reachable in require_shared_body mode — see __init__ and
        _build_shared_wrapper(), which is the only attach path there.

        No heuristic fallback exists anymore (Packet 086(2)). A failed or
        missing load leaves _model=None; the engine then runs as a no-op
        (zero activations). Whether a later offer_shared_body can recover
        depends on the construction shape (see start() for the per-shape
        warning). This method logs the load outcome accurately; start()
        adds the user-actionable warning.
        """
        if self._require_shared_body:
            return  # This loader can allocate a private body; shared-only never uses it.
        weights_path = self._weights_path()
        if os.path.exists(weights_path):
            try:
                from surgery.tonic_brain import load_tonic_brain
                self._model = load_tonic_brain(
                    weights_path,
                    transformer_body=self._shared_body,
                )
                self._model.eval()
                # The "surgical inference active" wording is misleading when the
                # wrapper loaded with a private body but no shared body is in
                # _shared_body — in that case the dispatch gate refuses the
                # forward and ticks produce zero. Be precise.
                if self._shared_body is not None:
                    logger.info("TonicBrain loaded from %s (shared body) — "
                                "surgical inference active", weights_path)
                else:
                    logger.info("TonicBrain loaded from %s (own copy, "
                                "transformer_body=<none>) — wrapper is resident "
                                "but inference is NOT active; the dispatch gate "
                                "will refuse forward until a shared body lands "
                                "(no heuristic fallback).", weights_path)
            except Exception as exc:
                logger.info(
                    "TonicBrain load error: %s — engine will run as no-op "
                    "(zero activations). %s", exc,
                    self._shared_body is not None
                    and "A later offer_shared_body() call cannot recover this "
                         "state — _model is None, and offer_shared_body only "
                         "attaches when require_shared_body=True. The "
                         "BrainSwitcher must rebuild the engine."
                    or "A later offer_shared_body() call from the "
                       "BrainSwitcher can attach a body.",
                )
        else:
            logger.info(
                "No TonicBrain checkpoint at %s — engine will run as no-op "
                "(zero activations). %s", weights_path,
                self._shared_body is not None
                and "A later offer_shared_body() call cannot recover this "
                     "state — _model is None, and offer_shared_body only "
                     "attaches when require_shared_body=True. The "
                     "BrainSwitcher must rebuild the engine."
                or "A later offer_shared_body() call from the BrainSwitcher "
                   "can attach a body.",
            )

    def _build_shared_wrapper(self, transformer_body):
        """Build the lightweight encoder/decoder wrapper AROUND A BORROWED BODY.

        Returns the new TonicBrain, or None on any failure (missing checkpoint,
        torch absent, load error) — a None return is RETRYABLE: _model is left
        untouched so the next offer_shared_body() tries again.

        Structural guarantee (not a comment-level one): a None body returns None
        before the loader is reached, so no call from this method can ever fall
        into load_tonic_brain's `from_pretrained` branch. That branch is the
        ~2GB own-copy allocation this mode exists to prevent.

        Called OUTSIDE _body_lock_context(): torch.load of the checkpoint is slow
        and must not block in-flight inference on the shared body.
        """
        if transformer_body is None:
            return None
        if not _TORCH_AVAILABLE:
            return None
        weights_path = self._weights_path()
        if not os.path.exists(weights_path):
            if not self._wrapper_fail_logged:
                self._wrapper_fail_logged = True
                logger.info(
                    "Tonic shared-body attach: no checkpoint at %s — waiting for shared transformer "
                    "(will retry on next offer)", weights_path,
                )
            return None
        try:
            from surgery.tonic_brain import load_tonic_brain
            model = load_tonic_brain(weights_path, transformer_body=transformer_body)
        except Exception as exc:
            if not self._wrapper_fail_logged:
                self._wrapper_fail_logged = True
                logger.warning(
                    "Tonic shared-body wrapper build failed: %s — waiting for shared transformer "
                    "(will retry on next offer)", exc,
                )
            return None
        # eval() the OUR-SIDE halves only. self._model.eval() would recurse into the
        # borrowed body and flip proto's training flag — that module is not ours to
        # mutate. The forward is under torch.no_grad() regardless.
        for part in ("encoder", "decoder"):
            try:
                getattr(model, part).eval()
            except Exception:
                pass
        self._wrapper_fail_logged = False
        return model

    # -----------------------------------------------------------------
    # Body Hot-Swap (called by BrainSwitcher)
    # -----------------------------------------------------------------

    def offer_shared_body(self, transformer_body, *, blocking=True) -> bool:
        """Attach the exact supplied body; shared-only engines build their wrapper late.

        BrainSwitcher uses blocking=False so a busy inference defers an offer to
        its next monitor cycle. Body installation still serializes with revocation.
        """
        if transformer_body is None:
            return False
        new_model = None
        if self._model is None:
            if not self._require_shared_body:
                return False
            new_model = self._build_shared_wrapper(transformer_body)
            if new_model is None:
                return False
        try:
            with self._body_lock_context(blocking=blocking):
                previous_model = self._model
                previous_body = getattr(previous_model, "body", None)
                previous_shared = self._shared_body
                try:
                    if self._model is None:
                        self._model = new_model
                    else:
                        self._model.body = transformer_body
                    if getattr(self._model, "body", None) is not transformer_body:
                        raise ValueError("wrapper did not retain the offered body")
                    self._shared_body = transformer_body
                except Exception:
                    # Keep existing wrappers retryable, including default consumers
                    # which deliberately do not build a wrapper on a later offer.
                    self._shared_body = None
                    self._model = previous_model
                    if previous_model is not None:
                        try:
                            previous_model.body = previous_body
                            if getattr(previous_model, "body", None) is not previous_body:
                                raise ValueError("wrapper did not restore its previous body")
                            self._shared_body = previous_shared
                        except Exception:
                            # A failed rollback must not forward through a partial
                            # body. Retain the wrapper so a future offer can retry.
                            logger.warning("Tonic body rollback failed; attachment inactive",
                                           exc_info=True)
                    raise
        except BlockingIOError:
            return False  # no attachment change; monitor retries when inference yields
        except Exception as exc:
            logger.warning("Tonic body attachment failed: %s", exc)
            return False
        logger.info("Tonic attached to shared ProtoUniBrain body (no private body loaded)")
        return True

    def revoke_shared_body(self) -> bool:
        """Release the borrowed body; required-sharing consumers wait without inference.

        Memory-cheap by design (Syl/Josh, 2026-06-10): a pressure-driven shed must
        RELIEVE memory, not load a fresh ~2GB own-transformer at the worst possible
        moment — two models resident under the very pressure that triggered the shed
        (the OOM-'n'-load trap). So we drop the now-dangling shared-body reference and
        wait (or use the default consumer fallback), KEEPING the lightweight encoder/decoder wrapper
        so offer_shared_body() can re-join the share the instant proto reloads.

        Runs under the body lock so a forward in flight finishes first, and so the
        body drop is seen together by _model_inference's in-lock recheck — never a
        live wrapper with body=None.
        """
        if self._model is None and self._shared_body is None:
            return False  # already detached — nothing to shed
        with self._body_lock_context():
            if self._model is not None:
                self._model.body = None      # drop the ref to proto's shed body (proto frees the ~2GB)
            self._shared_body = None
        logger.info(
            "Tonic shed shared body -> %s; will re-join on proto reload",
            "waiting (no inference path)" if self._require_shared_body else "no-op mode"
        )
        return True

    def set_prefetch_seed(self, fn) -> None:
        """Accept the host's Markov-prefetch seed source (#55 Stage 4 phase 5b).

        `fn()` returns {node_id: primed_score} for the currently live predictions
        (the host owns TTL filtering). The engine folds them into each tick's
        write-mode prime when CC_PITH_PREFETCH_WARM_ENABLED is set. Idempotent.
        """
        self._prefetch_seed = fn

    def _merge_prefetch_seeds(self, seen: Dict[str, float]) -> None:
        """Fold live predicted nodes into a tick's proposal dict, IN PLACE.

        Called on the model's output dict in _model_inference, so predictions
        are merged into the real-inference activations (no heuristic path
        remains -- Packet 086(2)). Max-dedup: a node the model already proposed
        keeps its own current. One seed set is primed at most
        _CC_PITH_PREFETCH_REPEATS ticks. Never raises; on any failure the
        dict is left as it was.
        """
        if not _CC_PITH_PREFETCH_WARM_ENABLED or self._prefetch_seed is None or _CC_PITH_PREFETCH_MAX <= 0:
            return
        try:
            seeds = self._prefetch_seed() or {}
            if not seeds:
                return
            key = tuple(sorted(seeds.items()))
            if key == self._prefetch_gen_key:
                self._prefetch_gen_uses += 1
                if self._prefetch_gen_uses > _CC_PITH_PREFETCH_REPEATS:
                    return
            else:
                self._prefetch_gen_key, self._prefetch_gen_uses = key, 1
            nodes = self._graph.nodes
            base = self._config.activation_strength * _CC_PITH_PREFETCH_CURRENT_SCALE
            added = 0
            for nid, score in sorted(seeds.items(), key=lambda kv: kv[1], reverse=True):
                if added >= _CC_PITH_PREFETCH_MAX:
                    break
                if nid not in nodes:
                    continue
                cur = base * min(1.0, max(0.0, float(score)))
                if nid in seen:
                    seen[nid] = max(seen[nid], cur)
                    continue
                seen[nid] = cur
                added += 1
            self._prefetch_seeded += added
        except Exception as exc:
            if not self._prefetch_fail_logged:
                self._prefetch_fail_logged = True
                logger.warning("Prefetch seed merge failing (non-fatal, logged once): %s", exc)

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
    def _body_lock_context(self, *, blocking=True):
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
                if blocking:
                    stack.enter_context(self._body_lock)
                elif self._body_lock.acquire(blocking=False):
                    stack.callback(self._body_lock.release)
                else:
                    raise BlockingIOError("shared transformer is in use")
            if self._lock_file_path is not None:
                try:
                    import fcntl as _fcntl
                    _lf = stack.enter_context(open(self._lock_file_path, 'r'))
                    flags = _fcntl.LOCK_SH | (0 if blocking else _fcntl.LOCK_NB)
                    _fcntl.flock(_lf.fileno(), flags)
                    stack.callback(_fcntl.flock, _lf.fileno(), _fcntl.LOCK_UN)
                except Exception as _exc:
                    if not blocking or self._require_shared_body:
                        raise  # never use the shared-only body without its declared lock
                    logger.warning("flock unavailable — cross-process lock skipped: %s", _exc)
            yield

    # -----------------------------------------------------------------
    # Latent Token Generation
    # -----------------------------------------------------------------

    def _reset_stage_samples(self) -> None:
        """Zero this tick's last-sample values. EMAs of stages that ran stay."""
        try:
            for name in self._stage_last_ms:
                self._stage_last_ms[name] = 0.0
        except Exception:
            pass

    def _record_stage_ms(self, name: str, elapsed_ms: float) -> None:
        """Store one stage sample. Never raises; never grows storage."""
        try:
            elapsed_ms = max(0.0, float(elapsed_ms))
            self._stage_last_ms[name] = elapsed_ms
            prev = self._stage_ema_ms[name]
            if prev == 0.0:
                self._stage_ema_ms[name] = elapsed_ms
            else:
                self._stage_ema_ms[name] = (
                    _STAGE_EMA_ALPHA * elapsed_ms
                    + (1.0 - _STAGE_EMA_ALPHA) * prev
                )
        except Exception:
            pass

    def _record_stage(self, name: str, t0: float) -> None:
        try:
            self._record_stage_ms(name, (time.perf_counter() - t0) * 1000.0)
        except Exception:
            pass

    def _stage_status(self) -> Dict[str, Any]:
        try:
            return {
                f"{kind}_{name}_ms": round(float(store[name]), 2)
                for name in _STAGE_NAMES
                for kind, store in (("last", self._stage_last_ms), ("ema", self._stage_ema_ms))
            }
        except Exception:
            return {
                f"{kind}_{name}_ms": 0.0
                for name in _STAGE_NAMES
                for kind in ("last", "ema")
            }

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
        self._reset_stage_samples()
        t_latent = time.perf_counter()
        lock = getattr(self._graph, '_concurrent_lock', None)
        acquired = False
        if lock is not None:
            acquired = lock.acquire(blocking=False)
        try:
            return self._generate_latent_token_inner()
        finally:
            if acquired:
                lock.release()
            self._record_stage("latent", t_latent)

    def _generate_latent_token_inner(self) -> Dict[str, Any]:
        """Inner implementation — actual latent token generation."""
        self._reset_stage_samples()
        # Required-sharing consumers wait without inference — this matches the original
        # "no model AND no shared body" condition that drives the wait state, plus the
        # post-revoke shed state where the wrapper still exists but body is gone.
        if self._require_shared_body and (
            self._model is None or self._shared_body is None
        ):
            return {"fired": 0, "activated": 0, "waiting_for_shared_body": True}
        t_feat = time.perf_counter()
        try:
            features = _extract_tonic_features(
                self._graph, self._tonic_thread,
                node_budget=self._config.node_sample_budget,
            )
        finally:
            self._record_stage("feature_extract", t_feat)
        if features is None:
            return {"fired": 0, "activated": 0}

        # Generate activation decisions. Without a loaded model and a shared body,
        # the engine runs as a no-op — see start() and status['inference_path_ready'].
        # _fallback_inference() unconditionally returns [] (Packet 086(2)).
        if self._model is not None and self._shared_body is not None:
            activations = self._model_inference(features)
        else:
            activations = self._fallback_inference(features)

        if not activations:
            return {"fired": 0, "activated": 0}

        # Inject activations into graph via write-mode propagation
        node_ids = [nid for nid, _ in activations]
        currents = [strength for _, strength in activations]

        t_prop = time.perf_counter()
        try:
            result = self._graph.prime_and_propagate(
                node_ids=node_ids,
                currents=currents,
                steps=self._config.propagation_steps,
                write_mode=True,
            )
        finally:
            self._record_stage("propagate", t_prop)

        # Update the tonic thread with the result
        if self._tonic_thread is not None:
            t_ou = time.perf_counter()
            try:
                self._tonic_thread.ouroboros_cycle()
            finally:
                self._record_stage("ouroboros", t_ou)

        self._tokens_generated += 1
        self._total_activations += len(activations)

        return {
            "fired": len(result.fired_entries),
            "activated": len(activations),
        }

    def _fallback_inference(self, features: Dict[str, Any]) -> List[Tuple[str, float]]:
        # VPS embedded topology must never receive heuristic-derived activations
        # (Packet 086(2), 2026-09-23). The heuristic was deleted; this method is now
        # a structural no-op that always returns []. It exists only so the dispatch
        # site at _generate_latent_token_inner and the historic reference at
        # _model_inference remain valid call sites; nothing reaches it on the
        # correct path (real model inference, or no-ops via the _model / _shared_body
        # gate at the dispatch site). Required-sharing consumers are gated by the
        # wait check above this point and never reach here either.
        return []

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
            return []

        # Materialize graph features for the model outside the body lock. This
        # walks graph collections independently of the bounded candidate scan,
        # so keep its cost distinct from both lock wait and transformer work.
        t_model_feat = time.perf_counter()
        try:
            graph_features = self._extract_graph_features_for_model()
        finally:
            self._record_stage("model_feature_extract", t_model_feat)
        if graph_features is None:
            return []

        # Forward through TonicBrain — the actual push.
        # The dispatcher's `self._model is not None and self._shared_body is not None`
        # test happened OUTSIDE this lock, so a revoke can have landed in between.
        # Re-check model and body INSIDE the lock: revoke_shared_body() mutates them
        # under this same lock, so what we read here cannot change until the forward
        # completes. Without this, a shed mid-tick calls the wrapper with body=None.
        # Apply the fallback policy OUTSIDE the lock; required-sharing consumers are
        # gated by the wait check at the top of _generate_latent_token_inner and
        # never reach this path with a missing body.
        # Wait vs forward: time __enter__ of the existing context manager separately
        # from the held-lock body. Do not acquire the lock twice.
        output = None
        t_wait = time.perf_counter()
        t_held = None
        t_fwd_end = None
        try:
            with self._body_lock_context():
                t_held = time.perf_counter()
                model = self._model
                if (
                    model is not None
                    and getattr(model, "body", None) is not None
                    and self._shared_body is not None
                ):
                    try:
                        with torch.no_grad():
                            output = model(graph_features)
                    finally:
                        t_fwd_end = time.perf_counter()
        finally:
            if t_held is not None:
                self._record_stage_ms("body_lock_wait", (t_held - t_wait) * 1000.0)
                if t_fwd_end is not None:
                    self._record_stage_ms("transformer_forward", (t_fwd_end - t_held) * 1000.0)
        if output is None:
            return []

        # Map activation strengths to actual nodes
        activation_strengths = output["activations"]
        exploration = output["exploration"]

        # Get the top active/recent nodes to map activations onto
        candidates = self._get_activation_candidates(features)
        if not candidates:
            return []

        activations: List[Tuple[str, float]] = []
        for i, (nid, _) in enumerate(candidates[:len(activation_strengths)]):
            strength = activation_strengths[i] * self._config.activation_strength
            if strength > 0.05:  # noise floor
                activations.append((nid, strength))

        # #55 5b: same merge as the heuristic path (this path has no brakes or
        # cap for anyone, so there is nothing to bypass).
        seen = dict(activations)
        self._merge_prefetch_seeds(seen)
        return list(seen.items())

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

        # Packet 086(2): the laptop daemon PID 35833 runs under /usr/bin/python3.12
        # with no torch. Before this commit, the heuristic path produced activations
        # on that host silently. With heuristic gone, every shape that can't reach
        # a real forward would otherwise be a silent zero — every tick
        # {"fired": 0, "activated": 0}, no signal to the daemon that nothing real
        # is happening. Log ONCE here (start-time is the right moment: before the
        # loop thread begins) so the state is defined and observable in `journalctl`
        # / the daemon's own logs. The status dict adds inference_path_ready +
        # torch_available keys for the same reason (machine-readable signal).
        # The log is one-shot per process; the daemon should not see it repeat on
        # restart / reload.
        #
        # Four silent-zero shapes, each with its own actionable warning:
        #   N1 — _model is None AND _shared_body is None, no torch.
        #        Original Packet 086(2) case. Daemon can recover if torch
        #        is installed OR if a shared body lands via offer_shared_body
        #        (require_shared_body path only).
        #   N2 — _model is None AND _shared_body is None, torch available.
        #        Same as N1 minus the torch-install advice.
        #   G1 — _model is None AND _shared_body is not None (caller passed a
        #        body at construction but _try_load_model failed or there was
        #        no checkpoint). Ticks are zero; offer_shared_body cannot
        #        recover this because require_shared_body is False (Syl-shape)
        #        and the default-mode init does not route a body through
        #        offer_shared_body. The BrainSwitcher must rebuild.
        #   G2 — _model is not None AND _shared_body is None (a wrapper
        #        loaded its own body successfully but no shared body is
        #        attached). The dispatch gate refuses forward; ticks are
        #        zero. offer_shared_body CAN recover this (it sets
        #        _model.body to the supplied transformer and re-flips
        #        _shared_body), so the warning is advisory.
        inference_path_ready = (
            self._model is not None and self._shared_body is not None
        )
        if not inference_path_ready:
            model_is_loaded = self._model is not None
            body_was_offered = self._shared_body is not None
            if not model_is_loaded and not body_was_offered:
                # N1 / N2
                if not _TORCH_AVAILABLE:
                    logger.warning(
                        "Tonic started with no torch AND no shared body — "
                        "every tick will produce zero activations (defined no-op "
                        "state, not a crash). Install torch to enable a private "
                        "model load; OR rebuild the engine with "
                        "require_shared_body=True so the BrainSwitcher can "
                        "attach a body via offer_shared_body()."
                    )
                else:
                    logger.warning(
                        "Tonic started with torch available but no shared body "
                        "— every tick will produce zero activations until a "
                        "shared body lands (no heuristic fallback, Packet 086(2))."
                    )
            elif not model_is_loaded and body_was_offered:
                # G1 — caller passed transformer_body=<body> at construction,
                # but the private load failed (or no checkpoint exists). The
                # body sits in _shared_body unused; offer_shared_body cannot
                # recover because it only attaches in require_shared_body=True
                # mode and that path also requires a checkpoint. The
                # BrainSwitcher must rebuild this engine.
                logger.warning(
                    "Tonic started with a shared body in hand but no model "
                    "could be loaded (checkpoint missing or load error). "
                    "Every tick will produce zero activations. THIS STATE "
                    "CANNOT SELF-RECOVER: the default-mode init path does not "
                    "route a body through offer_shared_body, and "
                    "offer_shared_body only attaches when require_shared_body=True. "
                    "The BrainSwitcher must rebuild the engine."
                )
            else:
                # G2 — wrapper loaded successfully but no shared body in
                # _shared_body. Dispatch gate refuses forward; ticks are zero.
                # offer_shared_body CAN recover this; the warning is advisory.
                logger.warning(
                    "Tonic loaded a private-copy TonicBrain wrapper but no "
                    "shared body is attached — the dispatch gate refuses forward, "
                    "so every tick produces zero activations. Have the "
                    "BrainSwitcher call offer_shared_body() with the shared "
                    "body to enable inference."
                )

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

            # #117 — autonomous aging clock (default OFF). Advance graph.timestep on the
            # heartbeat so decay / inactivity-cull / homeostasis evolve between conversations,
            # not only inside on_message(). Wall-time floor keeps the fast conversation tick
            # from spinning the clock and draining the inactivity grace window. Stamp BEFORE
            # the call so a persistently-failing step waits the full floor before retrying.
            # step() self-acquires _step_lock (RLock); we hold it explicitly to match the
            # autosave/orphan-drain discipline. Error-isolated: a step failure must not kill
            # the awareness loop. Cost falls inside the tick -> counted by adaptive cadence.
            if _CC_NG_AUTOSTEP:
                now = time.monotonic()
                if now - self._last_autostep >= _CC_NG_AUTOSTEP_MIN_INTERVAL:
                    self._last_autostep = now
                    t_auto = time.perf_counter()
                    try:
                        with self._graph._step_lock:
                            self._graph.step()
                    except Exception as exc:
                        logger.debug("Autostep error: %s", exc)
                    self._record_stage("autostep", t_auto)

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
                try:
                    logger.warning(
                        "Tonic tick over budget: %.3fs (budget %.1fs, nodes=%d, ema=%.1fms, "
                        "feature_extract=%.1fms model_feature_extract=%.1fms "
                        "body_lock_wait=%.1fms transformer_forward=%.1fms "
                        "propagate=%.1fms ouroboros=%.1fms latent=%.1fms autostep=%.1fms)",
                        elapsed, self._config.tick_budget_seconds,
                        len(self._graph.nodes), self._ema_tick_ms,
                        self._stage_last_ms.get("feature_extract", 0.0),
                        self._stage_last_ms.get("model_feature_extract", 0.0),
                        self._stage_last_ms.get("body_lock_wait", 0.0),
                        self._stage_last_ms.get("transformer_forward", 0.0),
                        self._stage_last_ms.get("propagate", 0.0),
                        self._stage_last_ms.get("ouroboros", 0.0),
                        self._stage_last_ms.get("latent", 0.0),
                        self._stage_last_ms.get("autostep", 0.0),
                    )
                except Exception:
                    try:
                        logger.warning(
                            "Tonic tick over budget: %.3fs (budget %.1fs, nodes=%d, ema=%.1fms)",
                            elapsed, self._config.tick_budget_seconds,
                            len(self._graph.nodes), self._ema_tick_ms,
                        )
                    except Exception:
                        pass

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
            # Heuristic surface was removed 2026-09-23 (Packet 086(2)). These two
            # keys remain in the public API as a literal False so existing status
            # readers see the same shape; they no longer describe a live mode.
            "using_heuristic": False,
            "heuristic_allowed": False,
            "waiting_for_shared_body": self._require_shared_body and (
                self._model is None or self._shared_body is None
            ),
            # The defined, observable signal that Packet 086(2) requires: a torch-less
            # host with no shared body lands here as False, distinguishable from the
            # "running with body" case (True) and from a stopped engine (also False).
            # The daemon can read this to surface the no-inference-path state rather
            # than reporting running=True with silent zero-output tokens.
            "inference_path_ready": (
                self._model is not None and self._shared_body is not None
            ),
            "model_loaded": self._model is not None,
            "require_shared_body": self._require_shared_body,
            "shared_body_attached": self._shared_body is not None,
            "torch_available": _TORCH_AVAILABLE,
            "ema_tick_ms": round(self._ema_tick_ms, 2),
            "current_interval_s": round(self._current_interval, 2),
            "node_sample_budget": self._config.node_sample_budget,
            "prefetch_seeded": self._prefetch_seeded,
            **self._stage_status(),
        }
