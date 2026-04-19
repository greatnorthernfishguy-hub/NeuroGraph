"""TriSyn concept-extraction helper (NG-internal).

The Trisynaptic Circuit: hippocampal-pathway-named organ that drains
NG's concept-extraction backlog by spawning isolated subprocess workers
under systemd-run. Named after the DG → CA3 → CA1 pathway — the brain's
canonical episodic-memory encoding circuit.

Design: ~/docs/inbox/trisynaptic-circuit-design-v0.1.md
"""

from trisynaptic.manager import TrisynapticManager

__all__ = ["TrisynapticManager"]
