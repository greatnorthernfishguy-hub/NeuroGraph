# ---- Changelog ----
# 2026-03-25 Claude Code — Initial creation
# What: Channel registry — birth, merge, death lifecycle management
# Why: Channel count is substrate-governed, not hardcoded
# How: JSON-persisted registry with rolling window detection for lifecycle events
# PRD: ~/docs/prd/Lenia_FlowGraph_Design_v0.1.md §3, §4
# -------------------

"""Channel registry managing the lifecycle of neuromodulatory channels.

Four channels seeded from biological analogs. The substrate can birth new
channels (unexplained residual), merge correlated channels, and kill
depleted channels. Channel count is an observable property, not a parameter.
"""

import json
import logging
import os
from collections import deque
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from lenia.config import LeniaConfig
from lenia.functions import FunctionSpec, validate_spec

logger = logging.getLogger(__name__)


@dataclass
class ChannelState:
    """Runtime state for a single channel."""
    id: int
    name: str
    kernel_shape: FunctionSpec
    growth_function: FunctionSpec
    effective_range: float
    age: int = 0  # ticks since birth
    total_energy: float = 0.0


class ChannelRegistry:
    """Manages channel lifecycle: birth, merge, death.

    Persisted as channels.json alongside the field store.
    """

    def __init__(self, config: LeniaConfig, persist_dir: str):
        self._config = config
        self._dir = Path(os.path.expanduser(persist_dir))
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / "channels.json"

        self._channels: Dict[int, ChannelState] = {}
        self._next_id: int = 0

        # Rolling windows for lifecycle detection
        self._residual_window: Deque[float] = deque(
            maxlen=config.residual_birth_window
        )
        self._correlation_windows: Dict[Tuple[int, int], Deque[float]] = {}
        self._energy_windows: Dict[int, Deque[float]] = {}

        if self._path.exists():
            self._load()
        else:
            self._seed_channels()

    @property
    def count(self) -> int:
        return len(self._channels)

    @property
    def channel_ids(self) -> List[int]:
        return sorted(self._channels.keys())

    def get(self, channel_id: int) -> ChannelState:
        return self._channels[channel_id]

    def kernel_shape(self, channel_id: int) -> FunctionSpec:
        return self._channels[channel_id].kernel_shape

    def growth_function(self, channel_id: int) -> FunctionSpec:
        return self._channels[channel_id].growth_function

    def effective_range(self, channel_id: int) -> float:
        return self._channels[channel_id].effective_range

    def all_channels(self) -> List[ChannelState]:
        return [self._channels[cid] for cid in self.channel_ids]

    def update_energy(self, channel_id: int, total: float):
        """Update a channel's tracked total energy."""
        ch = self._channels[channel_id]
        ch.total_energy = total
        ch.age += 1

        # Track for death detection
        if channel_id not in self._energy_windows:
            self._energy_windows[channel_id] = deque(
                maxlen=self._config.energy_death_window
            )
        self._energy_windows[channel_id].append(total)

    def check_birth(self, mean_residual: float) -> Optional[int]:
        """Record mean residual and check for channel birth trigger.

        Returns new channel ID if birth triggered, None otherwise.
        """
        self._residual_window.append(mean_residual)

        if len(self._residual_window) < self._config.residual_birth_window:
            return None

        window_mean = sum(self._residual_window) / len(self._residual_window)
        if window_mean <= self._config.residual_birth_threshold:
            return None

        # Birth triggered — create new channel with default shape
        new_id = self._next_id
        self._next_id += 1

        self._channels[new_id] = ChannelState(
            id=new_id,
            name=f"emergent_{new_id}",
            kernel_shape={
                "type": "gaussian",
                "params": {"center": 0.0, "sigma": 0.4, "amplitude": 0.8},
            },
            growth_function={
                "type": "bell",
                "params": {"center": 0.5, "sigma": 0.15},
            },
            effective_range=0.5,
        )

        self._residual_window.clear()
        self._save()
        logger.info("Channel born: %d (%s)", new_id, self._channels[new_id].name)
        return new_id

    def check_merge(
        self, field_state: np.ndarray
    ) -> Optional[Tuple[int, int]]:
        """Check for channel merge trigger based on field correlation.

        Args:
            field_state: (entity_count, channel_count) array.

        Returns (keep_id, absorb_id) if merge triggered, None otherwise.
        """
        ids = self.channel_ids
        if len(ids) < 2:
            return None

        for i, id_a in enumerate(ids):
            for id_b in ids[i + 1:]:
                col_a = ids.index(id_a)
                col_b = ids.index(id_b)

                # Pearson correlation
                a = field_state[:, col_a]
                b = field_state[:, col_b]
                if np.std(a) < 1e-10 or np.std(b) < 1e-10:
                    continue
                corr = float(np.corrcoef(a, b)[0, 1])

                pair = (min(id_a, id_b), max(id_a, id_b))
                if pair not in self._correlation_windows:
                    self._correlation_windows[pair] = deque(
                        maxlen=self._config.correlation_merge_window
                    )
                self._correlation_windows[pair].append(abs(corr))

                window = self._correlation_windows[pair]
                if len(window) < self._config.correlation_merge_window:
                    continue

                if (
                    sum(window) / len(window)
                    > self._config.correlation_merge_threshold
                ):
                    # Merge: keep older (lower id), absorb younger
                    keep_id, absorb_id = pair
                    self._merge(keep_id, absorb_id)
                    return (keep_id, absorb_id)

        return None

    def check_death(self) -> Optional[int]:
        """Check for channel death trigger based on depleted energy.

        Returns dead channel ID if triggered, None otherwise.
        """
        for cid, window in list(self._energy_windows.items()):
            if len(window) < self._config.energy_death_window:
                continue
            if sum(window) / len(window) < self._config.energy_death_threshold:
                self._kill(cid)
                return cid
        return None

    def _merge(self, keep_id: int, absorb_id: int):
        """Merge absorb_id into keep_id."""
        keep = self._channels[keep_id]
        absorb = self._channels[absorb_id]

        # Weight average by age
        total_age = keep.age + absorb.age + 1  # +1 to avoid div by zero
        w_keep = keep.age / total_age
        w_absorb = absorb.age / total_age

        # Average effective range
        keep.effective_range = (
            keep.effective_range * w_keep + absorb.effective_range * w_absorb
        )

        # Energy is summed (mass conservation)
        keep.total_energy += absorb.total_energy

        # Remove absorbed channel
        del self._channels[absorb_id]

        # Clean up tracking
        self._energy_windows.pop(absorb_id, None)
        pairs_to_remove = [
            p for p in self._correlation_windows if absorb_id in p
        ]
        for p in pairs_to_remove:
            del self._correlation_windows[p]

        self._save()
        logger.info(
            "Channel merged: %d absorbed into %d", absorb_id, keep_id
        )

    def _kill(self, channel_id: int):
        """Remove a dead channel."""
        ch = self._channels.pop(channel_id)
        self._energy_windows.pop(channel_id, None)
        pairs_to_remove = [
            p for p in self._correlation_windows if channel_id in p
        ]
        for p in pairs_to_remove:
            del self._correlation_windows[p]

        self._save()
        logger.info(
            "Channel died: %d (%s), energy was %.4f",
            channel_id, ch.name, ch.total_energy,
        )

    def channel_index(self, channel_id: int) -> int:
        """Return the column index for a channel in the field store."""
        return self.channel_ids.index(channel_id)

    def _seed_channels(self):
        """Initialize with seed channels from config."""
        for seed in self._config.initial_channels:
            cid = self._next_id
            self._next_id += 1
            self._channels[cid] = ChannelState(
                id=cid,
                name=seed["name"],
                kernel_shape=seed["kernel_shape"],
                growth_function=seed["growth_function"],
                effective_range=seed["effective_range"],
            )
        self._save()
        logger.info("Seeded %d channels", len(self._channels))

    def _save(self):
        """Persist registry to JSON."""
        data = {
            "next_id": self._next_id,
            "channels": {
                str(cid): {
                    "id": ch.id,
                    "name": ch.name,
                    "kernel_shape": ch.kernel_shape,
                    "growth_function": ch.growth_function,
                    "effective_range": ch.effective_range,
                    "age": ch.age,
                    "total_energy": ch.total_energy,
                }
                for cid, ch in self._channels.items()
            },
        }
        with open(self._path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self):
        """Load registry from JSON."""
        with open(self._path) as f:
            data = json.load(f)

        self._next_id = data["next_id"]
        self._channels = {}
        for cid_str, ch_data in data["channels"].items():
            cid = int(cid_str)
            self._channels[cid] = ChannelState(
                id=ch_data["id"],
                name=ch_data["name"],
                kernel_shape=ch_data["kernel_shape"],
                growth_function=ch_data["growth_function"],
                effective_range=ch_data["effective_range"],
                age=ch_data.get("age", 0),
                total_energy=ch_data.get("total_energy", 0.0),
            )

        logger.info("Loaded %d channels from registry", len(self._channels))
