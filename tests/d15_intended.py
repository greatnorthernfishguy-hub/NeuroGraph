# ---- Changelog ----
# [2026-10-06] Claude (lane sleep-p1) — CREATE: D15's one intended behaviour change, for the older whole-run tests
# What: apply_d15_rule(base_module) wraps the BASE module's Graph._remove_synapse_internal so that, after the base
#       removal, pre.pred_weights[post] is dropped when no other pre->post synapse is left — exactly what D15 adds to the
#       engine and nothing else. Used by the whole-run tests that compare the branch against an OLDER tip
#       (test_nodestore_p1 / p2a / p2b, test_rust_hotpaths_onto_s4), so they keep checking everything else bitwise.
# Why:  spec superpowers/specs/2026-10-06-sleep-phase-design.md D15, §8 P1 ("an intended change, named in the test");
#       "distinguish intentional change from regression". tests/test_sleep_p1.py proves engine == base + this rule, and
#       SLEEP_P1.md shows those files pass with D15 switched off.
# How:  patches ONLY the given (separately loaded) base module object; the branch module is never touched.
# -------------------
"""D15 (sleep-phase spec): the named intended change, applied to a base engine module for whole-run comparisons."""


def apply_d15_rule(base_module):
    G = base_module.Graph
    if getattr(G, "_d15_rule_applied", False):
        return
    orig = G._remove_synapse_internal

    def _remove_synapse_internal(self, synapse_id):
        s = self.synapses.get(synapse_id)
        pair = (s.pre_node_id, s.post_node_id) if s is not None else None
        orig(self, synapse_id)
        if pair is not None:
            n = self.nodes.get(pair[0])
            if n is not None and pair[1] in n.pred_weights and self._find_synapse(*pair) is None:
                del n.pred_weights[pair[1]]

    G._remove_synapse_internal = _remove_synapse_internal
    G._d15_rule_applied = True
