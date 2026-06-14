"""
ng_salience_gate.py — VENDORED salience gate for module telemetry → the Commons.

# ---- Changelog ----
# [2026-06-14] Claude Code (Fable 5) — NEW VENDORED FILE (Josh-approved, LAW 2; #320 Part 2)
# What: Parameterized salience gate. Shared substrate toolkit (like ng_lite, ng_embed): each
#       module instantiates its OWN gate with its OWN salience signal. The gate shapes telemetry
#       deposits — GRANULAR when the module's own signal is salient (surprise/anomaly), a NOMINAL
#       SPAN SUMMARY (aggregate counts, never blind) otherwise, RUN-LENGTH for repeats — and
#       competence-governs its own threshold. It deposits RAW via commons.deposit().
# Why: >1 module needs salience-gated metric deposit (NG, QG, Darwin, THC, Immunis). Vendoring it
#       once beats hand-crafting the same gate N times (Josh, 2026-06-14: "vendoring simplifies
#       adding/changing on multiple modules at once").
# How: SalienceGate(source_id, salience_fn, ...). observe(metrics) gates + deposits raw. This is
#       the DEPOSITOR's own logic — NOT a Commons service. The Commons has exactly two operations
#       (deposit, bucket); nobody calls a smart verb (the Substrate Axiom: deposit raw + bucket).
#       Bounded at the SOURCE (OOM's First Law) — no per-step flood. LAW 7: telemetry, gated by the
#       module's OWN signal, not an imposed classification; the raw experience path stays raw.
# -------------------

VENDORED FILE — do not modify per-module. Change at the canonical NeuroGraph source + re-vendor.

Usage (each module owns its instance):

    from ng_salience_gate import SalienceGate

    def _surprise(m):
        t = m.get("predictions_confirmed", 0) + m.get("predictions_surprised", 0)
        return (m.get("predictions_surprised", 0) / t) if t else 0.0

    gate = SalienceGate("neurograph", _surprise,
                        agg_fields=("fired_nodes", "predictions_confirmed", "predictions_surprised"))
    gate.observe(metrics_dict)   # gates + deposits raw via commons.deposit(); fail-soft
"""

import hashlib
import logging
import time
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

logger = logging.getLogger("ng.salience_gate")


def _default_commons_provider():
    from commons import get_commons
    return get_commons()


def _default_embed(text: str):
    from ng_embed import embed
    return embed(text)


class SalienceGate:
    """Salience-gated telemetry deposit (the module's own deposit-shaping logic)."""

    def __init__(
        self,
        source_id: str,
        salience_fn: Callable[[Dict[str, Any]], float],
        *,
        agg_fields: Sequence[str] = (),
        signature_fn: Optional[Callable[[Dict[str, Any], float], Tuple]] = None,
        nominal_flush_every: int = 60,
        threshold_min: float = 0.30,   # competence 0 → verbose/sensitive (keep more when naive)
        threshold_max: float = 0.70,   # competence 1 → compress (summarize moderate noise once trusted)
        calm_below: float = 0.20,      # raw salience below this ⇒ calm ⇒ earn competence (slow)
        spike_above: float = 0.60,     # raw salience at/above this ⇒ turbulence ⇒ lose competence (fast)
        comp_gain: float = 0.02,       # asymmetric: trust gained slowly...
        comp_loss: float = 0.08,       # ...lost quickly (a spike re-sensitizes the gate)
        commons_provider: Optional[Callable[[], Any]] = None,
        embed_fn: Optional[Callable[[str], Any]] = None,
    ):
        self._source = source_id
        self._salience_fn = salience_fn
        self._agg_fields = tuple(agg_fields)
        self._signature_fn = signature_fn or (lambda m, s: (round(s, 1),))
        self._flush_every = nominal_flush_every
        self._t_min, self._t_max = threshold_min, threshold_max
        self._calm_below, self._spike_above = calm_below, spike_above
        self._gain, self._loss = comp_gain, comp_loss
        self._commons_provider = commons_provider or _default_commons_provider
        self._embed = embed_fn or _default_embed
        # state
        self._nominal_count = 0
        self._nominal_agg: Dict[str, Any] = {}
        self._run_sig = None
        self._run_count = 0
        self._competence = 0.0   # start safe + verbose; graduate toward compression
        self._seq = 0            # monotonic — guarantees unique target_ids (no same-ms collision)

    # ---- competence (measured on RAW salience — threshold-independent, no runaway) ----
    def _update_competence(self, salience: float) -> None:
        if salience >= self._spike_above:
            self._competence = max(0.0, self._competence - self._loss)   # fast loss
        elif salience < self._calm_below:
            self._competence = min(1.0, self._competence + self._gain)   # slow gain

    def _effective_threshold(self) -> float:
        return self._t_min + self._competence * (self._t_max - self._t_min)

    # ---- the gate ----
    def observe(self, metrics: Dict[str, Any]) -> None:
        try:
            commons = self._commons_provider()
        except Exception as exc:  # noqa: BLE001 — no Commons (early boot) is graceful
            logger.debug("[%s] Commons unavailable for salience gate: %s", self._source, exc)
            return
        if commons is None:
            return
        try:
            salience = float(self._salience_fn(metrics))
            self._update_competence(salience)
            if salience >= self._effective_threshold():
                self._on_anomaly(commons, metrics, salience)
            else:
                self._on_nominal(commons, metrics)
        except Exception as exc:  # noqa: BLE001 — the gate never breaks the caller's step
            logger.debug("[%s] salience gate failed: %s", self._source, exc)

    def _on_anomaly(self, commons, metrics, salience):
        self._flush_nominal(commons)                 # summarize the nominal span that preceded it
        sig = self._signature_fn(metrics, salience)
        if sig == self._run_sig:
            self._run_count += 1                     # repeat — accumulate; run summary when broken
            return
        self._flush_run(commons)                     # a different anomaly — close the prior run
        self._run_sig, self._run_count = sig, 1
        meta = {"kind": "metrics", "source": self._source, "metric_kind": f"{self._source}_anomaly",
                "salience": "anomaly", "signal": round(salience, 3)}
        meta.update({k: metrics.get(k, 0) for k in self._agg_fields})
        self._deposit(commons, "anomaly", meta)

    def _on_nominal(self, commons, metrics):
        self._flush_run(commons)                     # an anomaly run just ended → summarize it
        self._nominal_count += 1
        for k in self._agg_fields:
            self._nominal_agg[k] = self._nominal_agg.get(k, 0) + metrics.get(k, 0)
        if self._nominal_count >= self._flush_every:
            self._flush_nominal(commons)

    def _flush_nominal(self, commons):
        if self._nominal_count <= 0:
            return
        self._deposit(commons, "nominal", {
            "kind": "metrics", "source": self._source,
            "metric_kind": f"{self._source}_nominal_span", "salience": "nominal",
            "span_steps": self._nominal_count, "aggregate": dict(self._nominal_agg),
            "gate_competence": round(self._competence, 3),
            "gate_threshold": round(self._effective_threshold(), 3),
        })
        self._nominal_count, self._nominal_agg = 0, {}

    def _flush_run(self, commons):
        if self._run_count > 1:                      # the anomaly repeated → one run-length summary
            self._deposit(commons, "anomaly_run", {
                "kind": "metrics", "source": self._source,
                "metric_kind": f"{self._source}_anomaly_run", "salience": "anomaly_run",
                "signature": str(self._run_sig), "repeats": self._run_count,
            })
        self._run_sig, self._run_count = None, 0

    def _deposit(self, commons, kind, metadata):
        try:
            label = f"{self._source} metrics: {metadata.get('metric_kind', kind)}"
            emb = self._embed(label)                 # embedding addresses the metric KIND
            if emb is None:
                return
            self._seq += 1
            ts = time.time()
            tid = (f"metrics:{self._source}:{kind}:"
                   f"{hashlib.sha1(repr(metadata).encode()).hexdigest()[:8]}:{ts:.3f}:{self._seq}")
            commons.deposit(emb, tid, metadata=metadata)
            logger.info("Commons: deposited metrics (%s:%s)", self._source, kind)
        except Exception as exc:  # noqa: BLE001 — a deposit failure never breaks the caller's step
            logger.debug("[%s] metrics Commons deposit failed: %s", self._source, exc)
