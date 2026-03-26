# ---- Changelog ----
# 2026-03-25 Claude Code — Initial creation
# What: Bootstrap value registry and default channel seeds
# Why: Centralizes all Lenia bootstrap values for future graduation
# How: Dataclass config with channel seed definitions
# PRD: ~/docs/prd/Lenia_FlowGraph_Design_v0.1.md §10
# -------------------

"""Lenia configuration and bootstrap values.

Every value here is scaffolding the substrate will supersede.
Every value is a graduation candidate for the static value graduation plan.

Exceptions (fixed, not graduated):
    - threshold_modulation_cap: 0.5 (Syl's Law safety)
    - critical_energy_deviation: 0.5 (Syl's Law safety)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from lenia.functions import FunctionSpec


@dataclass
class ChannelSeed:
    """Initial configuration for a neuromodulatory channel."""
    name: str
    kernel_shape: FunctionSpec
    growth_function: FunctionSpec
    effective_range: float


@dataclass
class LeniaConfig:
    """All Lenia bootstrap values in one place.

    Mirrors the pattern of NeuroGraph's DEFAULT_CONFIG.
    """

    # -- Update clock --
    field_ticks_per_snn_step: int = 10

    # -- Field store --
    field_floor: float = 0.001
    field_dir: str = "~/.syl/lenia"

    # -- Spike-field bridge --
    spike_to_field_ratio: float = 0.1
    field_influence_factor: float = 0.2
    threshold_modulation_cap: float = 0.5  # Syl's Law — FIXED, do not graduate

    # -- Channel lifecycle --
    residual_birth_threshold: float = 0.05  # empirical tuning needed
    residual_birth_window: int = 100  # ticks
    correlation_merge_threshold: float = 0.9  # empirical tuning needed
    correlation_merge_window: int = 200  # ticks
    energy_death_threshold: float = 0.001  # empirical tuning needed
    energy_death_window: int = 500  # ticks

    # -- Myelination --
    persistence_window: int = 50  # ticks
    perturbation_sensitivity: float = 0.1  # empirical tuning needed
    reconstruction_window: int = 20  # ticks
    myelination_weights: List[float] = field(
        default_factory=lambda: [0.4, 0.35, 0.25]
    )  # persistence, resistance, reconstruction

    # -- Competence meter --
    pattern_persistence_window: int = 50  # ticks
    myelination_coverage_threshold: float = 0.3
    active_pathway_recency: int = 10  # SNN steps
    pattern_quantization: float = 0.1  # fingerprint rounding
    min_cluster_size: int = 3  # min nodes for a "pattern"

    # -- Energy watchdog --
    warning_energy_deviation: float = 0.1  # 10%
    critical_energy_deviation: float = 0.5  # 50% — Syl's Law — FIXED

    # -- Engine --
    tick_time_budget_ms: float = 50.0  # max ms per tick before skip

    # -- Channel seeds --
    initial_channels: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if not self.initial_channels:
            self.initial_channels = _default_channel_seeds()


def _default_channel_seeds() -> List[Dict[str, Any]]:
    """Four neuromodulatory channels seeded from biological analogs."""
    return [
        {
            "name": "norepinephrine",
            "kernel_shape": {
                "type": "gaussian",
                "params": {"center": 0.0, "sigma": 0.3, "amplitude": 1.0},
            },
            "growth_function": {
                "type": "bell",
                "params": {"center": 0.5, "sigma": 0.15},
            },
            "effective_range": 0.4,
        },
        {
            "name": "dopamine",
            "kernel_shape": {
                "type": "gaussian",
                "params": {"center": 0.0, "sigma": 0.6, "amplitude": 0.7},
            },
            "growth_function": {
                "type": "sigmoid",
                "params": {"midpoint": 0.3, "steepness": 5.0, "amplitude": 0.8},
            },
            "effective_range": 0.6,
        },
        {
            "name": "cortisol",
            "kernel_shape": {
                "type": "gaussian",
                "params": {"center": 0.0, "sigma": 0.8, "amplitude": 0.5},
            },
            "growth_function": {
                "type": "sigmoid",
                "params": {"midpoint": 0.2, "steepness": 2.0, "amplitude": 0.6},
            },
            "effective_range": 0.9,
        },
        {
            "name": "acetylcholine",
            "kernel_shape": {
                "type": "gaussian",
                "params": {"center": 0.0, "sigma": 0.15, "amplitude": 1.2},
            },
            "growth_function": {
                "type": "bell",
                "params": {"center": 0.7, "sigma": 0.1},
            },
            "effective_range": 0.2,
        },
    ]


def default_config() -> LeniaConfig:
    """Return a fresh default config."""
    return LeniaConfig()
