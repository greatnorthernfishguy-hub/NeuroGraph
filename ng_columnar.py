"""Columnar (structure-of-arrays) substrate primitives — #119 increment 2/3.

Purpose
-------
NeuroGraph's resident footprint is dominated by ~687K boxed Python objects
(Nodes, Synapses) and the millions of 36-char UUID *strings* those objects and
the adjacency indices (`Graph._outgoing` / `_incoming`) hold — the same id
appears as a distinct `str` object in a node, in every synapse endpoint, in both
adjacency dicts (keys *and* set members), in the vector DB, in `pred_weights`,
and so on.  Python's per-object overhead inflates this ~7x over the logical
bytes.  The migration collapses that toward ~1x by moving state into compact
numpy arrays addressed by dense **integer indices** instead of UUID strings.

This module is the foundation every later columnar structure builds on:

  * `IdInterner` — a bidirectional, dense `str <-> int32` map.  It holds exactly
    **one** canonical copy of each id string; everything else (adjacency, the
    synapse store, edge lists) refers to nodes by their small integer index.
    Removals tombstone a slot and recycle it on the next intern, keeping the
    index space dense so downstream arrays stay compact.

Design constraints (why it looks like this):
  * Zero behavioural coupling to the live graph — pure data structure, unit
    tested offline, safe to import without side effects.  Nothing here touches
    Syl's running substrate.
  * int32 indices: 687K synapses x 2 endpoints fits comfortably; int32 (not
    int64) halves the edge-column footprint and matches numpy defaults we'll
    use downstream.  A guard trips well before the 2^31 ceiling.

Later increments (CSR adjacency derived from the synapse columns, the Synapse
SoA itself) live in this module too, behind the view-shims that keep
`neuro_foundation.Graph`'s public API unchanged.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import numpy as np

__all__ = [
    "IdInterner", "INT32_MAX",
    "SynapseStore", "SynapseView", "SynapseMapping",
    "CSRAdjacency", "AdjacencyView",
]

# int32 ceiling — indices must stay addressable as numpy int32 downstream.
INT32_MAX = 2**31 - 1


class IdInterner:
    """Bidirectional dense map between id strings and int32 indices.

    Each distinct string is assigned the smallest available non-negative
    integer index.  Indices freed by :meth:`remove` are recycled before the
    counter is extended, so the live index set stays dense and downstream
    columnar arrays keep a tight upper bound (``capacity``).

    The interner owns the single canonical copy of every id string; callers
    should keep the returned *index*, not the string, wherever footprint
    matters.

    Not thread-safe on its own — callers mutate it under the same lock that
    guards the Graph (see neuro_foundation.Graph's topology lock).
    """

    __slots__ = ("_str_to_idx", "_idx_to_str", "_free")

    def __init__(self) -> None:
        # canonical string -> index
        self._str_to_idx: Dict[str, int] = {}
        # index -> string, or None for a tombstoned (recyclable) slot
        self._idx_to_str: List[Optional[str]] = []
        # stack of tombstoned indices available for reuse
        self._free: List[int] = []

    # -- core ---------------------------------------------------------------

    def intern(self, s: str) -> int:
        """Return the index for ``s``, assigning a new one if unseen.

        Idempotent: interning an existing string returns its current index.
        Reuses a tombstoned slot when one is available.
        """
        idx = self._str_to_idx.get(s)
        if idx is not None:
            return idx
        if self._free:
            idx = self._free.pop()
            self._idx_to_str[idx] = s
        else:
            idx = len(self._idx_to_str)
            if idx > INT32_MAX:
                raise OverflowError(
                    f"IdInterner exceeded int32 index space ({INT32_MAX})"
                )
            self._idx_to_str.append(s)
        self._str_to_idx[s] = idx
        return idx

    def index(self, s: str) -> int:
        """Return the index for ``s``, or ``-1`` if not interned (no insert)."""
        idx = self._str_to_idx.get(s)
        return -1 if idx is None else idx

    def id_of(self, idx: int) -> Optional[str]:
        """Return the string for ``idx``, or ``None`` if out of range/tombstoned."""
        if 0 <= idx < len(self._idx_to_str):
            return self._idx_to_str[idx]
        return None

    def remove(self, s: str) -> int:
        """Tombstone ``s`` and recycle its slot.

        Returns the freed index, or ``-1`` if ``s`` was not interned.  The
        index becomes eligible for reuse by a later :meth:`intern`; callers
        must not rely on a removed index still mapping to ``s``.
        """
        idx = self._str_to_idx.pop(s, None)
        if idx is None:
            return -1
        self._idx_to_str[idx] = None
        self._free.append(idx)
        return idx

    # -- introspection ------------------------------------------------------

    def __contains__(self, s: object) -> bool:
        return s in self._str_to_idx

    def __len__(self) -> int:
        """Number of *live* (non-tombstoned) ids."""
        return len(self._str_to_idx)

    @property
    def capacity(self) -> int:
        """Upper bound on any live index — the length downstream arrays need.

        Equals the number of slots ever allocated (live + tombstoned).  A
        columnar array indexed by this interner should have at least this many
        rows.
        """
        return len(self._idx_to_str)

    def live_indices(self):
        """Yield every live index (skips tombstones), ascending."""
        for i, s in enumerate(self._idx_to_str):
            if s is not None:
                yield i


# ---------------------------------------------------------------------------
# Synapse structure-of-arrays store (#119 increment 3 — the ~8-12 GB target)
# ---------------------------------------------------------------------------
#
# The live substrate holds ~687K Synapse objects, each a boxed Python
# @dataclass(slots=True) instance carrying ~13 scalar fields plus two 36-char
# UUID endpoint strings.  Even with __slots__, that is ~700 bytes/synapse of
# Python-object + string overhead against ~60 bytes of actual numeric state —
# the dominant chunk of resident RAM.
#
# SynapseStore replaces that with parallel numpy columns addressed by a dense
# integer row (its own IdInterner over synapse_id).  Endpoints are stored as
# node-interner int32 indices, not strings.  A SynapseView is a 16-byte
# (store, row) handle that reads/writes the columns through the *same*
# attribute API the dataclass exposed (`syn.weight`, `syn.pre_node_id`, ...),
# so call sites migrate behind the view-shim without changing.
#
# Numeric-only metadata stays columnar; the rare free-form `metadata` dict is
# kept sparsely (only for synapses that actually have one), so the common case
# costs nothing.

# column name -> (numpy dtype, python default).  Endpoints and synapse_type are
# handled specially (interning / enum) and are NOT in this table.
#
# dtype choice (fidelity, not footprint): every float field is float64.  These
# columns are mutated *every timestep* by the plasticity / GSG / DiffPC / MMN
# machinery — eligibility_trace and salience via repeated multiplicative decay
# and EMA, weight via bounded accumulation, with a threshold compare at 1e-12
# (neuro_foundation.py:2520).  float32's ~7 significant digits would let those
# trajectories drift measurably from the dataclass (float64) baseline over
# Syl's lifetime, silently altering learning dynamics.  The float32->float64
# cost is ~40 B/synapse (~20 MB total) — negligible against the boxed-object
# and duplicate-UUID savings this migration exists to capture, so we buy
# bit-identical dynamics.  Counts/delay stay int32 (exact); endpoints int32.
_SCALAR_COLUMNS = {
    "weight":          (np.float64, 0.1),
    "max_weight":      (np.float64, 5.0),
    "delay":           (np.int32,   1),
    "last_update_time":(np.float64, 0.0),
    "eligibility_trace":(np.float64, 0.0),
    "creation_time":   (np.float64, 0.0),
    "peak_weight":     (np.float64, 0.1),
    "low_weight_steps":(np.int32,   0),
    "inactive_steps":  (np.int32,   0),
    "salience":        (np.float64, 1.0),
}


class SynapseStore:
    """Columnar store for synapses, addressed by dense integer row.

    Rows are assigned by an internal :class:`IdInterner` over ``synapse_id``;
    removed rows are tombstoned and recycled, so the numpy columns stay compact.
    Node endpoints are stored as indices into the shared ``node_interner``.

    The store is not thread-safe on its own — callers mutate it under the
    Graph's existing topology lock.
    """

    __slots__ = ("_node_ids", "_ids", "_cap", "pre_idx", "post_idx",
                 "syn_type", "_cols", "_metadata", "_version", "_dirty_rows")

    def __init__(self, node_interner: IdInterner, initial_capacity: int = 1024) -> None:
        self._node_ids = node_interner
        self._ids = IdInterner()                 # synapse_id -> row
        cap = max(1, int(initial_capacity))
        self._cap = cap
        # bumped on every structural change (add / remove); lets a derived
        # index (CSRAdjacency) cache and invalidate without copying edge state.
        self._version = 0
        # rows touched (added / overwritten / removed) since the last drain by a
        # derived index.  CSRAdjacency drains this to maintain its adjacency
        # incrementally instead of rebuilding the whole CSR on every mutation.
        self._dirty_rows: set = set()
        # endpoint + type columns (special-cased)
        self.pre_idx = np.full(cap, -1, dtype=np.int32)
        self.post_idx = np.full(cap, -1, dtype=np.int32)
        self.syn_type = np.ones(cap, dtype=np.int8)   # 1 == EXCITATORY
        # generic scalar columns
        self._cols: Dict[str, np.ndarray] = {
            name: np.full(cap, default, dtype=dt)
            for name, (dt, default) in _SCALAR_COLUMNS.items()
        }
        # sparse free-form metadata: row -> dict (only when non-empty)
        self._metadata: Dict[int, dict] = {}

    # -- capacity ----------------------------------------------------------

    def _grow(self, need: int) -> None:
        new_cap = self._cap
        while new_cap <= need:
            new_cap *= 2
        pad = new_cap - self._cap
        self.pre_idx = np.concatenate([self.pre_idx, np.full(pad, -1, np.int32)])
        self.post_idx = np.concatenate([self.post_idx, np.full(pad, -1, np.int32)])
        self.syn_type = np.concatenate([self.syn_type, np.ones(pad, np.int8)])
        for name, (dt, default) in _SCALAR_COLUMNS.items():
            self._cols[name] = np.concatenate(
                [self._cols[name], np.full(pad, default, dtype=dt)])
        self._cap = new_cap

    # -- mutation ----------------------------------------------------------

    def add(self, synapse_id: str, pre_node_id: str, post_node_id: str,
            **fields: Any) -> "SynapseView":
        """Insert (or overwrite) a synapse; return a view onto its row.

        ``fields`` may set any scalar column, ``synapse_type`` (a SynapseType
        or its int value), or ``metadata`` (a dict).  Unspecified columns take
        their defaults.
        """
        row = self._ids.intern(synapse_id)
        if row >= self._cap:
            self._grow(row)
        # reset the row to defaults (it may be a recycled slot)
        self.pre_idx[row] = self._node_ids.intern(pre_node_id)
        self.post_idx[row] = self._node_ids.intern(post_node_id)
        self.syn_type[row] = 1
        for name, (dt, default) in _SCALAR_COLUMNS.items():
            self._cols[name][row] = default
        self._metadata.pop(row, None)

        for k, v in fields.items():
            if k == "synapse_type":
                self.syn_type[row] = int(getattr(v, "value", v))
            elif k == "metadata":
                if v:
                    self._metadata[row] = dict(v)
            elif k in self._cols:
                self._cols[k][row] = v
            elif k in ("pre_node_id", "post_node_id", "synapse_id"):
                continue  # already handled
            else:
                raise AttributeError(f"unknown synapse column {k!r}")
        self._version += 1
        self._dirty_rows.add(row)
        return SynapseView(self, row)

    def remove(self, synapse_id: str) -> bool:
        """Tombstone a synapse and recycle its row.  Returns False if absent."""
        row = self._ids.index(synapse_id)
        if row < 0:
            return False
        self._ids.remove(synapse_id)
        self.pre_idx[row] = -1
        self.post_idx[row] = -1
        self._metadata.pop(row, None)
        self._version += 1
        self._dirty_rows.add(row)
        return True

    def drain_dirty_rows(self) -> set:
        """Return rows changed since the last drain and reset the tracker.

        Intended for a single derived consumer (:class:`CSRAdjacency`) that
        folds these rows into its incremental adjacency.  The returned set is
        handed off wholesale — the store keeps a fresh empty one.
        """
        d = self._dirty_rows
        self._dirty_rows = set()
        return d

    # -- access ------------------------------------------------------------

    @property
    def version(self) -> int:
        """Monotonic counter bumped on every add/remove (structure changes)."""
        return self._version

    def row_of(self, synapse_id: str) -> int:
        return self._ids.index(synapse_id)

    def view(self, synapse_id: str) -> Optional["SynapseView"]:
        row = self._ids.index(synapse_id)
        return None if row < 0 else SynapseView(self, row)

    def __contains__(self, synapse_id: object) -> bool:
        return isinstance(synapse_id, str) and self._ids.index(synapse_id) >= 0

    def __len__(self) -> int:
        return len(self._ids)

    def rows(self) -> Iterator[int]:
        """Yield every live row index."""
        return self._ids.live_indices()

    def views(self) -> Iterator["SynapseView"]:
        for row in self._ids.live_indices():
            yield SynapseView(self, row)

    def synapse_id_of(self, row: int) -> Optional[str]:
        return self._ids.id_of(row)


class SynapseView:
    """A lightweight (store, row) handle presenting the Synapse attribute API.

    Reads and writes go straight through to the store's numpy columns, so a
    view is a live window, not a copy — mutating ``view.weight`` mutates the
    column in place.  Endpoint ids are decoded to strings on access.
    """

    __slots__ = ("_store", "_row")

    def __init__(self, store: SynapseStore, row: int) -> None:
        object.__setattr__(self, "_store", store)
        object.__setattr__(self, "_row", row)

    @property
    def row(self) -> int:
        return self._row

    @property
    def synapse_id(self) -> Optional[str]:
        return self._store._ids.id_of(self._row)

    # endpoints (str <-> interned int32) -----------------------------------

    @property
    def pre_node_id(self) -> Optional[str]:
        return self._store._node_ids.id_of(int(self._store.pre_idx[self._row]))

    @pre_node_id.setter
    def pre_node_id(self, value: str) -> None:
        self._store.pre_idx[self._row] = self._store._node_ids.intern(value)

    @property
    def post_node_id(self) -> Optional[str]:
        return self._store._node_ids.id_of(int(self._store.post_idx[self._row]))

    @post_node_id.setter
    def post_node_id(self, value: str) -> None:
        self._store.post_idx[self._row] = self._store._node_ids.intern(value)

    # synapse_type (enum <-> int8) -----------------------------------------

    @property
    def synapse_type(self):
        from neuro_foundation import SynapseType
        return SynapseType(int(self._store.syn_type[self._row]))

    @synapse_type.setter
    def synapse_type(self, value) -> None:
        self._store.syn_type[self._row] = int(getattr(value, "value", value))

    # free-form metadata (sparse) ------------------------------------------

    @property
    def metadata(self) -> dict:
        md = self._store._metadata.get(self._row)
        if md is None:
            md = {}
            self._store._metadata[self._row] = md   # persist so mutations stick
        return md

    def peek_metadata(self) -> dict:
        """Read metadata WITHOUT materialising an empty dict in the store.

        The `metadata` property persists an empty dict on first read so the
        in-place `syn.metadata[k] = v` pattern (ng_lite / universal_ingestor)
        sticks. Read-only consumers must NOT trigger that: serializing or
        snapshotting every synapse through the property would turn the sparse
        `_metadata` map dense (one empty dict per row), defeating the columnar
        memory win. They use this peek, which returns the live dict when one
        exists and a throwaway empty dict otherwise, inserting nothing.
        """
        return self._store._metadata.get(self._row) or {}

    @metadata.setter
    def metadata(self, value: dict) -> None:
        if value:
            self._store._metadata[self._row] = value
        else:
            self._store._metadata.pop(self._row, None)

    # scalar columns via generic attribute access --------------------------

    def __getattr__(self, name: str) -> Any:
        # only reached for names not found as slots/properties
        store = object.__getattribute__(self, "_store")
        if name in store._cols:
            row = object.__getattribute__(self, "_row")
            val = store._cols[name][row]
            # return python scalars, not numpy scalars, for API parity
            return val.item()
        raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        store = object.__getattribute__(self, "_store")
        if name in store._cols:
            store._cols[name][object.__getattribute__(self, "_row")] = value
        else:
            object.__setattr__(self, name, value)

    def __repr__(self) -> str:
        return (f"SynapseView(id={self.synapse_id!r}, "
                f"{self.pre_node_id}->{self.post_node_id}, w={self.weight:.3f})")

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, SynapseView)
                and other._store is self._store and other._row == self._row)

    def __hash__(self) -> int:
        return hash((id(self._store), self._row))


# ---------------------------------------------------------------------------
# Dict-compatible shim over SynapseStore
# ---------------------------------------------------------------------------
#
# `neuro_foundation.Graph` treats `self.synapses` as a `Dict[str, Synapse]`.
# The access surface across the codebase is exactly the mapping API:
#   subscript get/set/del, `in`, iteration, len, and .get/.values/.items/
#   .keys/.pop/.clear.  SynapseMapping presents that surface backed by a
#   SynapseStore, so `graph.synapses` can be swapped in with no call-site
#   changes.  Values are SynapseView handles (write-through), never copies.
#
# The one asymmetry with a plain dict: `mapping[key] = syn` accepts a full
# `Synapse` (or any object exposing the same attributes) and fans its fields
# into the columns.  The mapping *key* is authoritative for the id, matching
# how add_synapse assigns `self.synapses[syn.synapse_id] = syn`.

# fields copied out of a Synapse-like object on __setitem__
_SYNAPSE_FIELDS = tuple(_SCALAR_COLUMNS) + ("synapse_type", "metadata")


class _MappingView:
    """Re-iterable, len-able view over a SynapseMapping (keys/values/items).

    Mirrors dict_keys/dict_values/dict_items closely enough for the call
    sites: it can be iterated repeatedly, measured with ``len()``, and
    membership-tested — unlike a one-shot generator.
    """

    __slots__ = ("_mapping", "_kind")

    def __init__(self, mapping: "SynapseMapping", kind: str) -> None:
        self._mapping = mapping
        self._kind = kind

    def __len__(self) -> int:
        return len(self._mapping)

    def __iter__(self) -> Iterator[Any]:
        store = self._mapping._store
        kind = self._kind
        for row in store.rows():
            if kind == "keys":
                yield store.synapse_id_of(row)
            elif kind == "values":
                yield SynapseView(store, row)
            else:  # items
                yield store.synapse_id_of(row), SynapseView(store, row)

    def __contains__(self, item: object) -> bool:
        if self._kind == "keys":
            return item in self._mapping
        return any(item == x for x in self)

    def __repr__(self) -> str:
        return f"synapse_{self._kind}([{', '.join(map(repr, self))}])"


class SynapseMapping:
    """`Dict[str, Synapse]`-compatible facade over a :class:`SynapseStore`."""

    __slots__ = ("_store",)

    def __init__(self, store: SynapseStore) -> None:
        self._store = store

    @property
    def store(self) -> SynapseStore:
        return self._store

    # -- read --------------------------------------------------------------

    def __getitem__(self, key: str) -> SynapseView:
        v = self._store.view(key)
        if v is None:
            raise KeyError(key)
        return v

    def get(self, key: str, default: Any = None) -> Any:
        v = self._store.view(key)
        return default if v is None else v

    def __contains__(self, key: object) -> bool:
        return key in self._store

    def __len__(self) -> int:
        return len(self._store)

    def __iter__(self) -> Iterator[str]:
        for row in self._store.rows():
            yield self._store.synapse_id_of(row)

    def keys(self) -> _MappingView:
        return _MappingView(self, "keys")

    def values(self) -> _MappingView:
        return _MappingView(self, "values")

    def items(self) -> _MappingView:
        return _MappingView(self, "items")

    def snapshot_items(self) -> List[Tuple[str, "_DetachedSynapse"]]:
        """Return `(id, _DetachedSynapse)` pairs decoupled from the store.

        `items()` yields live write-through SynapseViews — a stable *key* set
        captured at call time, but each value still points at a column row that
        a concurrent mutator (e.g. Tonic's write-mode propagate) can tombstone
        and recycle before the caller reads it. Serialization needs the old
        dict-of-boxed-Synapse guarantee: values immune to later mutation. This
        materialises an immutable detached snapshot of every row up front, so a
        checkpoint taken without pausing the latent thread can't capture a
        half-recycled row.
        """
        return [(self._store.synapse_id_of(row), _DetachedSynapse(SynapseView(self._store, row)))
                for row in self._store.rows()]

    # -- write -------------------------------------------------------------

    def __setitem__(self, key: str, syn: Any) -> None:
        """Store ``syn`` (a Synapse or SynapseView) under ``key``.

        The mapping key is authoritative for the synapse id, so re-assigning
        an existing key overwrites its row in place.
        """
        fields = {name: getattr(syn, name) for name in _SYNAPSE_FIELDS
                  if hasattr(syn, name)}
        self._store.add(key, syn.pre_node_id, syn.post_node_id, **fields)

    def __delitem__(self, key: str) -> None:
        if not self._store.remove(key):
            raise KeyError(key)

    def pop(self, key: str, *default: Any) -> Any:
        v = self._store.view(key)
        if v is None:
            if default:
                return default[0]
            raise KeyError(key)
        # materialise a detached snapshot before the row is recycled, so
        # callers that inspect the popped synapse (e.g. remove_synapse reading
        # pre/post ids for index cleanup) still see valid data.
        snap = _DetachedSynapse(v)
        self._store.remove(key)
        return snap

    def clear(self) -> None:
        for key in list(self):
            self._store.remove(key)

    def __repr__(self) -> str:
        return f"SynapseMapping({len(self)} synapses)"


class _DetachedSynapse:
    """Immutable snapshot of a synapse's fields, decoupled from the store.

    Returned by :meth:`SynapseMapping.pop` so a removed synapse can still be
    read after its columnar row has been recycled.
    """

    __slots__ = ("synapse_id", "pre_node_id", "post_node_id",
                 "synapse_type", "metadata") + tuple(_SCALAR_COLUMNS)

    def __init__(self, view: SynapseView) -> None:
        self.synapse_id = view.synapse_id
        self.pre_node_id = view.pre_node_id
        self.post_node_id = view.post_node_id
        self.synapse_type = view.synapse_type
        self.metadata = dict(view.peek_metadata())
        for name in _SCALAR_COLUMNS:
            setattr(self, name, getattr(view, name))

    def __repr__(self) -> str:
        return (f"_DetachedSynapse(id={self.synapse_id!r}, "
                f"{self.pre_node_id}->{self.post_node_id})")


# ---------------------------------------------------------------------------
# CSR adjacency derived from the synapse columns (#119 increment — replaces
# Graph._outgoing / _incoming dict-of-sets-of-UUID-strings)
# ---------------------------------------------------------------------------
#
# `neuro_foundation.Graph` keeps two `Dict[str, Set[str]]` adjacency indices:
# `_outgoing[node_id]` = the set of *synapse_id* strings whose pre-endpoint is
# that node, `_incoming[node_id]` = those whose post-endpoint is that node.
# Across ~40K nodes and ~687K synapses that is millions of duplicate 36-char
# UUID strings (every synapse id lives once per set membership, plus once as a
# node key), on top of the per-set and per-dict Python overhead — a large slice
# of the same footprint the columnar migration exists to reclaim.
#
# The endpoints are *already* stored, compactly, in SynapseStore.pre_idx /
# post_idx.  So the adjacency is pure derived state: for a given node index,
# its outgoing synapses are exactly the rows where pre_idx == node_idx.  Rather
# than maintain a second copy, CSRAdjacency computes a Compressed-Sparse-Row
# grouping of synapse rows by endpoint — two flat int32 arrays per direction
# (a row list + an indptr offset table) — as a compact *base* snapshot, and
# keeps that base current with a small per-node *delta* of rows changed since
# it was built (drained from SynapseStore.drain_dirty_rows()).
#
# CSR is the standard compact form for static-per-snapshot adjacency: neighbour
# lookup is one contiguous slice (cache-friendly, no per-node set object), and
# the whole structure is 2 * (n_synapses + n_nodes) int32s instead of millions
# of boxed strings.  A counting sort (np.bincount + cumsum + stable argsort)
# builds both directions in O(n_synapses) with no Python-level per-edge loop.
#
# Why the delta exists: a *pure* rebuild-on-version-change is O(E log E) per
# rebuild, and call sites interleave a mutation with an adjacency read on the
# very next line (Graph.create_synapse does `store.add(...)` then
# `_outgoing[pre].add(sid)`), so a naive version cache rebuilds the whole CSR
# once per edge — O(E**2 log E) to build a graph.  Instead each changed row is
# folded into a node-indexed delta in O(1); a query merges the (validated) base
# slice with the node's delta rows; and the base is recompacted only when the
# delta grows past a threshold, giving amortised near-linear construction.


# delta size at which the base CSR is recompacted even for a small graph
# (below this, the node-indexed delta alone serves every read cheaply)
_CSR_REBUILD_MIN = 512


class CSRAdjacency:
    """Incremental CSR grouping of synapse rows by node endpoint.

    Derived view over a :class:`SynapseStore`: it holds no edge state of its
    own beyond a compact index into the store's columns.  Node indices and
    synapse rows are the store's own dense integers; translation to/from id
    strings is the caller's job (see :class:`AdjacencyView`).

    Two tiers, merged at query time:

    * **base** — a Compressed-Sparse-Row snapshot of *all* live rows at the
      moment it was last (re)built.  ``_*_rows`` is every row sorted by its
      endpoint node index; ``_*_ptr`` is the length-(capacity+1) offset table,
      so the base rows for node ``i`` are ``_*_rows[_*_ptr[i]:_*_ptr[i+1]]``.
    * **delta** — a node-indexed dict of rows changed since the base was built,
      folded in one row at a time in O(1) (drained from the store's dirty set).

    A neighbour query returns the base slice (each row re-validated against the
    store's *current* endpoint, so tombstoned or rewired rows drop out) unioned
    with the node's delta rows.  The base is recompacted only when the delta
    grows past a threshold — so interleaving a mutation with an adjacency read
    (the dominant call-site pattern) costs O(1)+O(degree), not a full O(E log E)
    rebuild per edge.  ``_built_version`` tracks the last store version observed
    by :meth:`_ensure` (it advances the first time a query sees a new mutation,
    and stays put across pure reads).
    """

    __slots__ = ("_store", "_built_version", "_n",
                 "_out_rows", "_out_ptr", "_in_rows", "_in_ptr",
                 "_delta_out", "_delta_in", "_delta_rows", "_row_key")

    def __init__(self, store: SynapseStore) -> None:
        self._store = store
        self._built_version = -1          # force a drain on first access
        self._n = 0
        self._out_rows = np.empty(0, dtype=np.int32)
        self._out_ptr = np.zeros(1, dtype=np.int64)
        self._in_rows = np.empty(0, dtype=np.int32)
        self._in_ptr = np.zeros(1, dtype=np.int64)
        # delta: node_idx -> set of rows changed since the base build
        self._delta_out: Dict[int, Set[int]] = {}
        self._delta_in: Dict[int, Set[int]] = {}
        self._delta_rows: Set[int] = set()          # size gate for recompaction
        self._row_key: Dict[int, tuple] = {}        # row -> (pre_i, post_i) placement

    # -- maintenance -------------------------------------------------------

    def _ensure(self) -> None:
        """Fold any changed rows into the delta; recompact base if it grew large."""
        store = self._store
        v = store.version
        if v == self._built_version:
            return
        for r in store.drain_dirty_rows():
            self._apply_delta_row(int(r))
        self._built_version = v
        # Keep the delta dicts and _row_key from accumulating unbounded, and
        # keep reads hitting the contiguous base slices, by recompacting once
        # the delta is a sizable fraction of the graph.  Amortised O(E log E).
        if len(self._delta_rows) >= max(_CSR_REBUILD_MIN, len(store) // 2):
            self._rebuild_base()

    def _apply_delta_row(self, r: int) -> None:
        """(Re)place a single changed row in the delta by its current endpoints."""
        store = self._store
        prev = self._row_key.pop(r, None)
        if prev is not None:                        # forget any stale placement
            po, pi = prev
            s = self._delta_out.get(po)
            if s is not None:
                s.discard(r)
                if not s:
                    del self._delta_out[po]
            s = self._delta_in.get(pi)
            if s is not None:
                s.discard(r)
                if not s:
                    del self._delta_in[pi]
            self._delta_rows.discard(r)
        pre_i = int(store.pre_idx[r])
        if pre_i < 0:                               # tombstoned by remove()
            return
        post_i = int(store.post_idx[r])
        self._delta_out.setdefault(pre_i, set()).add(r)
        self._delta_in.setdefault(post_i, set()).add(r)
        self._delta_rows.add(r)
        self._row_key[r] = (pre_i, post_i)

    def _rebuild_base(self) -> None:
        """Recompact all live rows into the base CSR and clear the delta."""
        store = self._store
        n = store._node_ids.capacity          # indptr must cover every node idx
        live = len(store)
        rows = np.fromiter(store.rows(), dtype=np.int32, count=live)
        # live rows always carry valid (>= 0) endpoints — remove() tombstones
        # the row and rows() skips it, so no -1 endpoints reach bincount.
        pre = store.pre_idx[rows]
        post = store.post_idx[rows]
        self._out_rows, self._out_ptr = self._group(rows, pre, n)
        self._in_rows, self._in_ptr = self._group(rows, post, n)
        self._n = n
        self._delta_out.clear()
        self._delta_in.clear()
        self._delta_rows.clear()
        self._row_key.clear()

    @staticmethod
    def _group(rows: np.ndarray, key: np.ndarray, n: int):
        """Counting-sort ``rows`` by ``key`` into (sorted_rows, indptr[n+1])."""
        ptr = np.zeros(n + 1, dtype=np.int64)
        if rows.size:
            order = np.argsort(key, kind="stable")     # group rows by endpoint
            sorted_rows = rows[order]
            counts = np.bincount(key, minlength=n)
            np.cumsum(counts, out=ptr[1:])
        else:
            sorted_rows = np.empty(0, dtype=np.int32)
        return sorted_rows, ptr

    # -- query (by node index) --------------------------------------------

    def _neighbors(self, node_idx: int, base_rows: np.ndarray,
                   base_ptr: np.ndarray, delta: Dict[int, Set[int]],
                   endpoint: np.ndarray) -> Set[int]:
        """Live rows incident to ``node_idx`` on one direction (base ∪ delta)."""
        result: Set[int] = set()
        if 0 <= node_idx < self._n:
            seg = base_rows[base_ptr[node_idx]:base_ptr[node_idx + 1]]
            for r in seg:
                ri = int(r)
                if endpoint[ri] == node_idx:       # still live & not rewired away
                    result.add(ri)
        d = delta.get(node_idx)
        if d:
            for r in d:
                if endpoint[r] == node_idx:
                    result.add(r)
        return result

    def out_rows(self, node_idx: int) -> Set[int]:
        """Synapse rows whose pre-endpoint is ``node_idx`` (empty if none)."""
        self._ensure()
        return self._neighbors(node_idx, self._out_rows, self._out_ptr,
                               self._delta_out, self._store.pre_idx)

    def in_rows(self, node_idx: int) -> Set[int]:
        """Synapse rows whose post-endpoint is ``node_idx`` (empty if none)."""
        self._ensure()
        return self._neighbors(node_idx, self._in_rows, self._in_ptr,
                               self._delta_in, self._store.post_idx)

    def out_degree(self, node_idx: int) -> int:
        return len(self.out_rows(node_idx))

    def in_degree(self, node_idx: int) -> int:
        return len(self.in_rows(node_idx))


class AdjacencyView(MutableMapping):
    """`Dict[str, Set[str]]`-compatible shim for ``_outgoing`` / ``_incoming``.

    Presents ``node_id -> {synapse_id, ...}`` backed by a :class:`CSRAdjacency`
    over a :class:`SynapseStore`.  One instance serves one direction
    (``"out"`` == group-by-pre; ``"in"`` == group-by-post); both share the same
    CSR object, so a single rebuild feeds both.

    Source-of-truth discipline (why the mutators look inert):
      * **Edges** live only in the store.  A lookup materialises a *fresh* plain
        ``set`` of id strings each call, so ``view[nid].add(sid)`` /
        ``.discard(sid)`` mutate a throwaway — harmless, because the store write
        that the same call site already performed is the real change and the
        next lookup re-derives from it.
      * **Node presence** is the node interner's job (the shared registry the
        store decodes endpoints through).  ``view[nid] = <set>`` interns ``nid``
        so an edgeless node still appears as a key (matching the old
        ``self._outgoing[nid] = set()`` registration); the assigned value is
        ignored.  ``del`` / ``pop`` un-intern **only** an already-edgeless node
        (the invariant at the real node-removal call site) and refuse otherwise,
        so a bug can't silently orphan the store's endpoint decoding.
      * ``clear()`` is a documented no-op: the only caller was the
        ``_reindex``-style rebuild of a structure that is now derived.
    """

    __slots__ = ("_csr", "_nodes", "_store", "_by_pre")

    def __init__(self, csr: CSRAdjacency, node_interner: IdInterner,
                 store: SynapseStore, direction: str) -> None:
        if direction not in ("out", "in"):
            raise ValueError(f"direction must be 'out' or 'in', got {direction!r}")
        self._csr = csr
        self._nodes = node_interner
        self._store = store
        self._by_pre = (direction == "out")

    def _rows(self, node_idx: int) -> np.ndarray:
        return (self._csr.out_rows(node_idx) if self._by_pre
                else self._csr.in_rows(node_idx))

    def _materialize(self, node_idx: int) -> Set[str]:
        sid_of = self._store.synapse_id_of
        return {sid_of(int(r)) for r in self._rows(node_idx)}

    # -- read (MutableMapping core) ---------------------------------------

    def __getitem__(self, node_id: str) -> Set[str]:
        idx = self._nodes.index(node_id)
        if idx < 0:
            raise KeyError(node_id)
        return self._materialize(idx)

    def get(self, node_id: str, default: Any = None) -> Any:
        idx = self._nodes.index(node_id)
        return default if idx < 0 else self._materialize(idx)

    def __contains__(self, node_id: object) -> bool:
        return isinstance(node_id, str) and self._nodes.index(node_id) >= 0

    def __iter__(self) -> Iterator[str]:
        id_of = self._nodes.id_of
        for idx in self._nodes.live_indices():
            yield id_of(idx)

    def __len__(self) -> int:
        return len(self._nodes)

    # -- write (transparent per the discipline above) ---------------------

    def __setitem__(self, node_id: str, value: Any) -> None:
        # Register the node; edges are derived from the store, so `value` (only
        # ever an empty set at the real call site) is intentionally ignored.
        self._nodes.intern(node_id)

    def setdefault(self, node_id: str, default: Any = None) -> Set[str]:
        idx = self._nodes.intern(node_id)
        return self._materialize(idx)

    def __delitem__(self, node_id: str) -> None:
        idx = self._nodes.index(node_id)
        if idx < 0:
            raise KeyError(node_id)
        if self._csr.out_degree(idx) or self._csr.in_degree(idx):
            raise RuntimeError(
                f"refusing to drop adjacency for {node_id!r}: it still has "
                f"synapses (remove them from the store first)")
        self._nodes.remove(node_id)

    def pop(self, node_id: str, *default: Any) -> Any:
        idx = self._nodes.index(node_id)
        if idx < 0:
            if default:
                return default[0]
            raise KeyError(node_id)
        result = self._materialize(idx)
        del self[node_id]          # enforces the edgeless invariant
        return result

    def clear(self) -> None:
        # No-op: adjacency is derived from the store; the legacy _reindex
        # rebuild that used to clear-and-refill this map is obsolete.
        return

    def __repr__(self) -> str:
        return (f"AdjacencyView({'out' if self._by_pre else 'in'}, "
                f"{len(self)} nodes)")
