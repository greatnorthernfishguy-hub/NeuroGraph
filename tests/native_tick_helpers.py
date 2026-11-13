# ---- Changelog ----
# [2026-11-13] DudeMan CC (Claude Fable 5.2, laptop) -- native tick piece 1 (punch #1221)
# What: miss(g, pred, recent_fired): one failed prediction through the engine's own pass, with a given fired window.
# Why:  _on_prediction_error / _surprise_exploration / _log_novel_sequence no longer exist as Python methods (their
#       bodies are ng_tract.PredictionBook's); the tests that called them directly now drive Graph._cleanup_predictions.
# How:  the spike record is set so that exactly `recent_fired` is in the window, the prediction is registered already
#       expired, the pass runs, the spike record is put back. Open predictions already in the graph would expire too if
#       they are due: callers use it on graphs with none.
# -------------------
from collections import deque


def miss(g, pred, recent_fired, error_penalty=None):
    saved = [(nid, vals, cap) for nid, vals, cap in g._pb.spike_items()]
    cfg_saved = g.config.get("prediction_error_penalty")
    g._pb.spike_clear()
    for nid, _vals, cap in saved:
        g._pb.spike_set(nid, [], cap)
    for nid in recent_fired:
        g._recent_spikes[nid] = deque([g.timestep], maxlen=20)
    pred.expires_at = g.timestep - 1
    if error_penalty is not None:
        g.config["prediction_error_penalty"] = error_penalty
    g.active_predictions[pred.prediction_id] = pred
    try:
        g._cleanup_predictions()
    finally:
        if error_penalty is not None:
            g.config["prediction_error_penalty"] = cfg_saved
        g._pb.spike_clear()
        for nid, vals, cap in saved:
            g._pb.spike_set(nid, vals, cap)
