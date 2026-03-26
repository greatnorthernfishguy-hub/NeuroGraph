# ---- Changelog ----
# 2026-03-25 Claude Code — Initial creation
# What: Mmap-backed double-buffered field store
# Why: Persistent, crash-recoverable field state for Lenia dynamics
# How: Memory-mapped binary file with header + two buffers + energy ledger
# PRD: ~/docs/prd/Lenia_FlowGraph_Design_v0.1.md §4
# -------------------

"""Mmap-backed field store with double buffering.

Layout:
    Header (64 bytes): magic, version, entity_count, channel_count,
                        tick_counter, buffer_selector, reserved
    Buffer A: entity_count * channel_count * 8 bytes (float64, entity-major)
    Buffer B: same size
    Energy ledger: entity_count * 2 * 8 bytes (energy_in, energy_out per entity)

Thread safety: double buffer. Update engine reads from current buffer,
writes to the other, then swaps atomically. External reads always hit
the non-writing buffer.
"""

import logging
import os
import struct
import shutil
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# File format constants
MAGIC = b"LNFD"
VERSION = 1
HEADER_SIZE = 64
HEADER_FMT = "<4sHIHqB"  # magic(4) version(2) entities(4) channels(2) ticks(8) selector(1)
HEADER_PACK_SIZE = struct.calcsize(HEADER_FMT)  # 21 bytes, padded to 64


class FieldStore:
    """Mmap-backed double-buffered field state.

    Each entity has a field state vector of N channels (float64).
    Two buffers enable lock-free reads during writes.
    """

    def __init__(self, path: str, entity_count: int, channel_count: int):
        """Initialize or load field store.

        Args:
            path: Directory for field.bin (e.g., ~/.syl/lenia)
            entity_count: Number of entities (nodes)
            channel_count: Number of active channels
        """
        self._dir = Path(os.path.expanduser(path))
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / "field.bin"

        self._entity_count = entity_count
        self._channel_count = channel_count
        self._tick_counter = 0
        self._buffer_selector = 0  # 0 = buffer A is current, 1 = buffer B

        # Buffer size in float64 elements
        self._buf_elements = entity_count * channel_count
        self._buf_bytes = self._buf_elements * 8

        # Energy ledger: [energy_in, energy_out] per entity
        self._ledger_elements = entity_count * 2
        self._ledger_bytes = self._ledger_elements * 8

        if self._path.exists():
            self._load()
        else:
            self._create()

    def _file_size(self) -> int:
        return HEADER_SIZE + 2 * self._buf_bytes + self._ledger_bytes

    def _create(self):
        """Create a fresh field store with all zeros."""
        total = self._file_size()
        with open(self._path, "wb") as f:
            # Header
            header = struct.pack(
                HEADER_FMT,
                MAGIC, VERSION,
                self._entity_count, self._channel_count,
                self._tick_counter, self._buffer_selector,
            )
            f.write(header)
            f.write(b"\x00" * (HEADER_SIZE - HEADER_PACK_SIZE))
            # Two buffers + ledger, all zeros
            f.write(b"\x00" * (2 * self._buf_bytes + self._ledger_bytes))

        self._map()
        logger.info(
            "Created field store: %d entities, %d channels, %d bytes",
            self._entity_count, self._channel_count, total,
        )

    def _load(self):
        """Load existing field store and resume."""
        with open(self._path, "rb") as f:
            raw = f.read(HEADER_PACK_SIZE)

        magic, version, entities, channels, ticks, selector = struct.unpack(
            HEADER_FMT, raw
        )

        if magic != MAGIC:
            raise ValueError(f"Bad field store magic: {magic!r}")
        if version != VERSION:
            raise ValueError(f"Unsupported field store version: {version}")

        self._entity_count = entities
        self._channel_count = channels
        self._tick_counter = ticks
        self._buffer_selector = selector

        self._buf_elements = entities * channels
        self._buf_bytes = self._buf_elements * 8
        self._ledger_elements = entities * 2
        self._ledger_bytes = self._ledger_elements * 8

        self._map()
        logger.info(
            "Loaded field store: %d entities, %d channels, tick %d, buffer %d",
            entities, channels, ticks, selector,
        )

    def _map(self):
        """Memory-map the file into numpy arrays."""
        # Map the full file as raw bytes, then create views
        self._mmap = np.memmap(
            str(self._path), dtype=np.uint8, mode="r+",
        )

        buf_a_start = HEADER_SIZE
        buf_b_start = buf_a_start + self._buf_bytes
        ledger_start = buf_b_start + self._buf_bytes

        # Create float64 views into the mmap
        self._buffer_a = np.ndarray(
            (self._entity_count, self._channel_count),
            dtype=np.float64,
            buffer=self._mmap,
            offset=buf_a_start,
        )
        self._buffer_b = np.ndarray(
            (self._entity_count, self._channel_count),
            dtype=np.float64,
            buffer=self._mmap,
            offset=buf_b_start,
        )
        self._ledger = np.ndarray(
            (self._entity_count, 2),
            dtype=np.float64,
            buffer=self._mmap,
            offset=ledger_start,
        )

    @property
    def entity_count(self) -> int:
        return self._entity_count

    @property
    def channel_count(self) -> int:
        return self._channel_count

    @property
    def tick_counter(self) -> int:
        return self._tick_counter

    def read_buffer(self) -> np.ndarray:
        """Return numpy view of the current (readable) buffer. Zero-copy."""
        if self._buffer_selector == 0:
            return self._buffer_a
        return self._buffer_b

    def write_buffer(self) -> np.ndarray:
        """Return numpy view of the write target buffer. Zero-copy."""
        if self._buffer_selector == 0:
            return self._buffer_b
        return self._buffer_a

    def read(self, entity_idx: int, channel: Optional[int] = None) -> np.ndarray:
        """Read field state for an entity.

        Args:
            entity_idx: Integer index of the entity.
            channel: If given, return single channel value. Otherwise all channels.
        """
        buf = self.read_buffer()
        if channel is not None:
            return float(buf[entity_idx, channel])
        return buf[entity_idx, :].copy()

    def inject(
        self,
        entity_idx: int,
        amount: float,
        channel_distribution: np.ndarray,
    ):
        """Inject energy into the current buffer at an entity.

        Args:
            entity_idx: Integer index.
            amount: Total energy to inject.
            channel_distribution: Array of shape (channel_count,) summing to ~1.0.
                Energy per channel = amount * channel_distribution[c].
        """
        buf = self.read_buffer()
        energy = amount * channel_distribution
        buf[entity_idx, :] += energy
        self._ledger[entity_idx, 0] += amount  # energy_in

    def swap(self):
        """Atomically swap which buffer is current.

        After swap, reads see the freshly written buffer.
        """
        self._buffer_selector = 1 - self._buffer_selector
        self._write_header()

    def tick(self):
        """Increment tick counter and persist to header."""
        self._tick_counter += 1
        self._write_header()

    def total_energy(self, channel: Optional[int] = None) -> float:
        """Total field energy across all entities, optionally for one channel."""
        buf = self.read_buffer()
        if channel is not None:
            return float(buf[:, channel].sum())
        return float(buf.sum())

    def reset_ledger(self):
        """Zero the energy in/out ledger for a new accounting period."""
        self._ledger[:] = 0.0

    def snapshot(self, name: str = "checkpoint"):
        """Copy current field state to a checkpoint file."""
        dest = self._dir / f"field_{name}.bin"
        shutil.copy2(self._path, dest)
        logger.info("Field snapshot saved: %s", dest)

    def restore(self, name: str = "checkpoint"):
        """Restore field state from a checkpoint file."""
        src = self._dir / f"field_{name}.bin"
        if not src.exists():
            raise FileNotFoundError(f"No checkpoint: {src}")
        # Unmap, copy, remap
        del self._mmap
        shutil.copy2(src, self._path)
        self._load()
        logger.info("Field restored from: %s", src)

    def resize_entities(self, new_count: int):
        """Resize for node additions/removals. Preserves existing data."""
        if new_count == self._entity_count:
            return

        old_buf = self.read_buffer().copy()
        old_ledger = self._ledger.copy()
        old_channels = self._channel_count

        # Unmap old file
        del self._mmap, self._buffer_a, self._buffer_b, self._ledger

        self._entity_count = new_count
        self._buf_elements = new_count * old_channels
        self._buf_bytes = self._buf_elements * 8
        self._ledger_elements = new_count * 2
        self._ledger_bytes = self._ledger_elements * 8

        # Create new file
        self._create()

        # Copy preserved data
        copy_entities = min(old_buf.shape[0], new_count)
        self._buffer_a[:copy_entities, :] = old_buf[:copy_entities, :]
        self._buffer_b[:copy_entities, :] = old_buf[:copy_entities, :]
        self._ledger[:copy_entities, :] = old_ledger[:copy_entities, :]

        logger.info("Field store resized: %d entities", new_count)

    def resize_channels(self, new_count: int):
        """Resize for channel birth/death. Preserves existing data."""
        if new_count == self._channel_count:
            return

        old_buf = self.read_buffer().copy()
        old_ledger = self._ledger.copy()
        old_entities = self._entity_count

        del self._mmap, self._buffer_a, self._buffer_b, self._ledger

        self._channel_count = new_count
        self._buf_elements = old_entities * new_count
        self._buf_bytes = self._buf_elements * 8

        self._create()

        copy_channels = min(old_buf.shape[1], new_count)
        self._buffer_a[:, :copy_channels] = old_buf[:, :copy_channels]
        self._buffer_b[:, :copy_channels] = old_buf[:, :copy_channels]
        self._ledger[:] = old_ledger

        logger.info("Field store resized: %d channels", new_count)

    def _write_header(self):
        """Persist header fields to the mmap file."""
        header = struct.pack(
            HEADER_FMT,
            MAGIC, VERSION,
            self._entity_count, self._channel_count,
            self._tick_counter, self._buffer_selector,
        )
        self._mmap[:HEADER_PACK_SIZE] = np.frombuffer(header, dtype=np.uint8)
        self._mmap.flush()

    def close(self):
        """Flush and unmap."""
        if hasattr(self, "_mmap") and self._mmap is not None:
            self._mmap.flush()
            del self._mmap
