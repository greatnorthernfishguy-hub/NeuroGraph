# ---- Changelog ----
# [2026-11-09] DudeMan CC (Claude Opus 5.5, laptop; #1210 / #1210b; test repair #1217, Josh 2026-11-09 "Yes, please")
# What: CREATE. #1210's one intended checkpoint change, applied to an OLDER engine's checkpoint for whole-run comparisons:
#       a novel-sequence event keeps the _NOVEL_SAMPLE (32) smallest firing ids + n_firing; an error prediction outcome
#       keeps the 32 smallest actual_firing_nodes + n_actual_firing (saved only when non-zero). Everything else is
#       compared as decoded msgpack, unchanged.
# Why:  #1210 (454c3b2) / #1210b (45a3dcd), 2026-11-04, live since. Tests that pin an older base (test_stability_s19 ->
#       15b1958, test_rust_hotpaths_onto_s4 -> 5246b63) failed on exactly these two fields from then on (measured
#       2026-11-09: the only differing fields), so they stopped checking anything. Same pattern as d15_intended.py:
#       "distinguish intentional change from regression", the change named in the test.
# How:  mirrors neuro_foundation.Graph._log_novel_sequence / _on_prediction_error (heapq.nsmallest). Base side only.
# -------------------
"""#1210 / #1210b: the named intended checkpoint change, applied to a base engine's checkpoint."""
import heapq

import msgpack

SAMPLE = 32   # neuro_foundation.Graph._NOVEL_SAMPLE


def as_1210(d):
    """A decoded checkpoint from an engine before #1210, as #1210 / #1210b would have saved it."""
    for ev in d.get("novel_sequence_log", []):
        fired = ev["firing_nodes"]
        ev["firing_nodes"], ev["n_firing"] = heapq.nsmallest(SAMPLE, fired), len(fired)
    for po in d.get("prediction_outcomes", []):
        if not po["confirmed"]:
            fired = po["actual_firing_nodes"]
            po["actual_firing_nodes"] = heapq.nsmallest(SAMPLE, fired)
            if fired:
                po["n_actual_firing"] = len(fired)
    return d


def _un(b):
    return msgpack.unpackb(b, strict_map_key=False, raw=False)


def same_ckpt(base_bytes, new_bytes):
    """True when the newer engine's checkpoint equals the base's under #1210 (bytes first: equal bytes need no decode)."""
    return base_bytes == new_bytes or as_1210(_un(base_bytes)) == _un(new_bytes)


def same_ckpts(base_list, new_list):
    return len(base_list) == len(new_list) and all(same_ckpt(a, b) for a, b in zip(base_list, new_list))
