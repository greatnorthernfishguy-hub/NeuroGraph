# ---- Changelog ----
# 2026-03-25 Claude Code — Initial creation
# What: Lenia FlowGraph package init
# Why: Continuous Lenia/Flow-Lenia dynamics for the NeuroGraph substrate
# How: Exports public interface for substrate integration
# PRD: ~/docs/prd/Lenia_FlowGraph_Design_v0.1.md
# -------------------

"""Lenia FlowGraph — continuous dynamics for the NeuroGraph substrate.

This package adds a continuous Lenia/Flow-Lenia field alongside the existing
SNN. The field evolves between spikes, enforces mass conservation, and
modulates SNN behavior through bidirectional coupling.

The SNN is unchanged. Lenia is additive. The kill switch disables all
field→SNN modulation instantly, preserving field state (suspend, not destroy).

Law compliance:
    Law 1: Substrate-internal, not inter-module communication.
    Law 2: No vendored files touched. All code in lenia/ package.
    Law 7: Channel distribution based on trace state, never semantic content.
"""

from lenia.config import LeniaConfig, default_config
from lenia.kill_switch import KillSwitch
from lenia.interface import LeniaSubstrate

__all__ = [
    "LeniaConfig",
    "default_config",
    "KillSwitch",
    "LeniaSubstrate",
]
