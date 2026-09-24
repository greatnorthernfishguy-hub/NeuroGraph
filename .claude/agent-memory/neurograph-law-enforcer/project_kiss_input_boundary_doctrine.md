---
name: kiss-input-boundary-doctrine
description: Sanctioned KISS application point is CC's input/deposit boundary as redundancy->reinforcement, not the outbound-resend layer; cc_ng_organism.py is a reference-quality substrate-native module
metadata:
  type: project
---

KISS (per docs/concepts/KISS.md "Current State", 2026-07-08) is sanctioned to be applied at CC's OWN substrate deposit boundary in `run_conversational_dual_pass` (cc_ng_organism.py), NOT at the outbound-resend layer (kiss_filter.py / Elmer's kiss.py / miniTID's disabled Rust port).

The sanctioned form is redundancy -> **reinforcement**, never dedup-drop: a near-duplicate turn (cosine >= threshold vs existing `{"cc":True,"creation_mode":"conversational"}` nodes) confirms the existing node (counter bump + probation acceleration + topology binding via `_cc_bind_conversational_topology`) instead of depositing a duplicate. This is LAW-7-honoring because the turn still touches the substrate.

**Why:** This is the biological/organism-native reading of KISS — Hebbian confirmation, not deduplication. It is the reference pattern for "Delta Gate at the input boundary."

**How to apply:** When reviewing any KISS or dedup work, verify it is at the deposit/input boundary AND takes the reinforcement (not drop) form. cc_ng_organism.py's redundancy gate is a strong reference implementation with high ethos alignment. Two watch-items in that gate: (1) the caller must honor `_cc_kiss_reinforce_node`'s documented False-return fallback (prune race -> LAW 7 drop risk); (2) the `_cc_is_synthetic_clutter` clutter-strip is the one drop-not-reinforce piece and sits in LAW 4 / LAW 7 tension (duplicates miniTID's upstream field-based skip; silent-drop false-positive risk).

Related: [[tid-peninsula-locked-form]].
