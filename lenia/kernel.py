# ---- Changelog ----
# [2026-07-08] Claude Code (Fable 5 design / Haiku implementation) — #371 removal-aware reconcile
# What: DistanceCache.reconcile_removals() + _translate_watermark(): compact the six
#   component matrices with an order-preserving keep-slice when cached entities were
#   pruned from the live graph, remap the dirty set, translate the resume watermark
#   (direct when both endpoints survive; else greatest surviving computed pair at or
#   below the cut — DOWN only, never up). Returns survivors for known_entity_order,
#   or None -> caller takes the legacy full-rebuild path.
# Why: #371 — any node prune between save and restart forced callers to discard the
#   ENTIRE cache (confirmed live on Syl 2026-07-08: ~1.79M pairs / 24.5% of a rebuild
#   lost at one restart; with continuous pruning her cache could never complete).
# How: monotone compaction preserves the canonical (max, min) pair order populate()
#   depends on, so the computed-region invariant survives reindexing; callers fall
#   through to the untouched watermark-resume/growth branches.
# [2026-07-08] Claude Code (Fable 5 design / Haiku implementation) — Resume watermark
# What: DistanceCache gains a resume watermark: populate() iterates connected pairs in a
#   canonical (max_idx, min_idx)-sorted order; each periodic checkpoint records the last
#   processed pair on the instance so save() persists it; load() restores it; populate(
#   resume_watermark=...) skips pairs at or before it. neurograph_rpc.py resumes an
#   interrupted rebuild instead of treating a partial checkpoint as complete.
# Why: the 2026-07-06 periodic checkpoints made progress SURVIVE interruption but not
#   RESUME — a partial cache loaded as if complete (silently hollow Lenia dynamics) or
#   the growth path extended only new entities, never finishing the old region. On a
#   rebuild measured in days (7.3M pairs, 2026-07-07) against a VPS that restarts more
#   often than that, that gap was the difference between insurance and decoration.
# How: sort connected_pairs by (p[1], p[0]) — deterministic under append-stable entity
#   indexing, and all new-entity pairs sort after all old-old pairs, so resume+growth
#   compose in one filter. Watermark = the pair VALUE (indices into a changed list are
#   meaningless). Cleared on completion so a finished save carries no watermark.
#   Known accepted blind spot: new old-old synapses created mid-interruption sort <=
#   watermark and are skipped — same class as the start_index path's existing blind spot.
# 2026-03-25 Claude Code — Initial creation
# What: Multi-metric kernel computer with sparse distance cache
# Why: Lenia kernels operate over composite distance, not single-metric
# How: Sparse cache per distance component, lazy invalidation, per-channel ranges
# PRD: ~/docs/prd/Lenia_FlowGraph_Design_v0.1.md §3, §4
# [2026-03-26] Claude Code (Opus 4.6) — Vectorized kernel + dual-pass embeddings
# What: Rewrote kernel compute to use vectorized numpy ops instead of Python
#   loops. Added dual-pass embedding (forest + tree) as separate distance
#   components. Distance vector is now 6 components, not 5.
# Why: O(n^2) Python loops can't handle 2,277 nodes at tick speed.
#   Dual-pass gives kernel two semantic scales — channels can weight
#   broad concept similarity differently from specific detail similarity.
# How: CSR sparse matrices for neighbor lookup, numpy broadcast for kernel
#   evaluation. DistanceCache builds adjacency lists on populate() for
#   fast per-channel neighbor gathering.
# [2026-07-05] CC (laptop) — Incremental populate + entity_ids persistence
# What: DistanceCache now saves/loads the entity_id ordering it was built
#   against, and populate() accepts start_index to compute only pairs
#   touching entities at or past that index instead of every connected
#   pair in the graph. populate() also marks itself populated before the
#   expensive loop, not after, so a mid-loop crash still leaves whatever
#   was computed savable instead of discarding it.
# Why: neurograph_rpc.py's bootstrap invalidated (and fully recomputed)
#   the whole cache on ANY entity_count drift — on Syl's live graph this
#   took up to ~8 hours, and every restart before this one raced the next
#   restart before ever reaching save(), so the same full rebuild kept
#   retriggering from scratch indefinitely. Root cause traced via
#   journalctl history (2026-07-05): one successful save (Jun 30 -> Jul
#   02, ~8hrs), then every subsequent restart re-populated from the same
#   stale save and got interrupted before saving again. Entity indices
#   are stable now only if NeuroGraphSubstrate is constructed with
#   known_entity_order (see graph_substrate.py) — this is the cache-side
#   half that makes an incremental extension possible at all.
# How: resize()'s existing preserve-old-submatrix behavior already did
#   the hard part; start_index just filters populate()'s connected_pairs
#   down to ones touching a new entity so the expensive distance_vector
#   loop only runs for the delta, not the whole graph.
# [2026-07-06] Claude Code (Sonnet 5) — Periodic in-loop checkpointing
# What: populate() accepts checkpoint_interval_secs + on_checkpoint. Every
#   1000 pairs, checks wall-clock elapsed since the last checkpoint and, if
#   over the interval, rebuilds CSR and calls the caller's save callback.
# Why: The 2026-07-05 fix made restarts cheaper (incremental vs full
#   rebuild), but did nothing for a run that's interrupted mid-populate —
#   the only save() call happened once, after the whole loop returned, which
#   a hard process kill never reaches. Confirmed live: the distance cache
#   file was still dated 2026-07-02 after multiple full-day restart cycles,
#   each one discarding 100% of that attempt's progress. Root-caused with
#   Josh 2026-07-06 while watching a stuck multi-hour populate() run.
# How: checkpoint_interval_secs=0 (default) is a no-op — existing callers
#   that don't pass it get unchanged behavior. neurograph_rpc.py's two
#   populate() call sites pass a 5-minute interval + a lambda that calls
#   the same cache.save(path) it already calls once at the end.
# -------------------

"""Vectorized multi-metric kernel computer with dual-pass embeddings.

Distance between nodes is composite (6 components):
    [topology, synaptic_weight, cofire, hyperedge,
     embedding_forest, embedding_tree]

Dual-pass: forest = broad concept similarity, tree = specific detail.
Each channel's kernel can weight these scales differently.

Vectorized: CSR sparse matrices + numpy broadcast. No Python loops
over entities in the hot path.
"""

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy import sparse

from lenia.channels import ChannelRegistry
from lenia.functions import evaluate
from lenia.interface import LeniaSubstrate

logger = logging.getLogger(__name__)

# Distance vector component indices
DIST_TOPOLOGY = 0
DIST_SYNAPTIC = 1
DIST_COFIRE = 2
DIST_HYPEREDGE = 3
DIST_EMBEDDING_FOREST = 4
DIST_EMBEDDING_TREE = 5
NUM_DIST_COMPONENTS = 6


class DistanceCache:
    """Sparse, lazily-invalidated distance cache.

    Stores one CSR sparse matrix per distance component.
    Provides fast neighbor lookup via sparse row slicing.
    """

    def __init__(self, entity_count: int, entity_ids: Optional[List[str]] = None):
        self._n = entity_count
        # The entity_id ordering this cache's rows/cols are keyed against.
        # Persisted so a restart can tell which entities are genuinely new
        # (append to the cache) vs. which are the same set it already has
        # distances for (see NeuroGraphSubstrate's known_entity_order).
        self._entity_ids: Optional[List[str]] = (
            list(entity_ids) if entity_ids is not None else None
        )
        # Build as lil for efficient updates, convert to CSR for fast reads
        self._components_lil: List[sparse.lil_matrix] = [
            sparse.lil_matrix((entity_count, entity_count), dtype=np.float64)
            for _ in range(NUM_DIST_COMPONENTS)
        ]
        self._components_csr: Optional[List[sparse.csr_matrix]] = None
        # Dirty entries: set of (row, col, component_idx)
        self._dirty: Set[Tuple[int, int, int]] = set()
        self._populated = False
        # Combined magnitude matrix for fast neighbor filtering
        self._magnitude_csr: Optional[sparse.csr_matrix] = None
        # Resume watermark: the last (i, j) pair processed by an interrupted
        # populate() run, in canonical (max, min) order. None = complete (or
        # never checkpointed mid-run). Persisted by save(), restored by load().
        self._watermark: Optional[Tuple[int, int]] = None

    @property
    def entity_count(self) -> int:
        return self._n

    @property
    def entity_ids(self) -> Optional[List[str]]:
        return self._entity_ids

    @property
    def populated(self) -> bool:
        return self._populated

    @property
    def watermark(self) -> Optional[Tuple[int, int]]:
        """Last pair processed by an interrupted populate(), or None if complete."""
        return self._watermark

    def save(self, path: str) -> None:
        """Save the distance cache to disk.

        Serializes all CSR component matrices + metadata as a single
        numpy .npz archive.  Restore skips the 7-minute populate()
        call entirely if the graph hasn't changed since save.
        """
        if not self._populated:
            return
        self._rebuild_csr()  # ensure CSR is current
        data = {
            "entity_count": np.array([self._n]),
            "num_components": np.array([NUM_DIST_COMPONENTS]),
        }
        if self._entity_ids is not None:
            # numpy infers a fixed-width unicode dtype for a list of plain
            # strings — no allow_pickle needed on either side.
            data["entity_ids"] = np.array(self._entity_ids)
        if self._watermark is not None:
            data["watermark"] = np.array(list(self._watermark))
        for c in range(NUM_DIST_COMPONENTS):
            csr = self._components_csr[c]
            data[f"c{c}_data"] = csr.data
            data[f"c{c}_indices"] = csr.indices
            data[f"c{c}_indptr"] = csr.indptr
            data[f"c{c}_shape"] = np.array(csr.shape)
        np.savez_compressed(path, **data)
        logger.info("Distance cache saved: %s (%d entities, %d components)",
                     path, self._n, NUM_DIST_COMPONENTS)

    @classmethod
    def load(cls, path: str) -> Optional["DistanceCache"]:
        """Restore a distance cache from disk.

        Returns None if the file doesn't exist or is incompatible.
        The caller should fall back to populate() on None.
        """
        import os
        actual = path if path.endswith(".npz") else path + ".npz"
        if not os.path.exists(actual):
            return None
        try:
            arch = np.load(actual, allow_pickle=False)
            n = int(arch["entity_count"][0])
            nc = int(arch["num_components"][0])
            if nc != NUM_DIST_COMPONENTS:
                logger.warning("Distance cache component mismatch (%d vs %d), repopulating",
                               nc, NUM_DIST_COMPONENTS)
                return None
            entity_ids = (
                arch["entity_ids"].tolist() if "entity_ids" in arch.files else None
            )
            cache = cls(n, entity_ids=entity_ids)
            cache._watermark = (
                (int(arch["watermark"][0]), int(arch["watermark"][1]))
                if "watermark" in arch.files else None
            )
            cache._components_csr = []
            for c in range(nc):
                csr = sparse.csr_matrix(
                    (arch[f"c{c}_data"], arch[f"c{c}_indices"], arch[f"c{c}_indptr"]),
                    shape=tuple(arch[f"c{c}_shape"]),
                )
                cache._components_csr.append(csr)
                cache._components_lil[c] = csr.tolil()
            cache._populated = True
            cache._rebuild_csr()  # rebuilds magnitude too
            logger.info("Distance cache restored: %s (%d entities)", actual, n)
            return cache
        except Exception as exc:
            logger.warning("Distance cache restore failed (%s): %s", actual, exc)
            return None

    def set_distance(
        self, i: int, j: int, component: int, value: float
    ):
        """Set a distance component for a pair."""
        self._components_lil[component][i, j] = value
        self._components_lil[component][j, i] = value
        self._components_csr = None  # invalidate CSR cache
        self._magnitude_csr = None

    def populate(self, substrate: LeniaSubstrate, start_index: int = 0,
                 checkpoint_interval_secs: float = 0.0,
                 on_checkpoint: Optional[Callable[[], None]] = None,
                 resume_watermark: Optional[Tuple[int, int]] = None):
        """Populate the cache from the substrate.

        Called once on startup. Subsequent updates use dirty flags.
        Uses adjacency-based approach: only compute distances for
        pairs that are synaptically connected or share hyperedges,
        plus a hop radius for topological neighbors.

        Args:
            start_index: 0 (default) rebuilds every connected pair from
                scratch — used when the cache is empty or entity identities
                have changed incompatibly (e.g. after a prune). >0 computes
                only pairs that touch an entity whose index is >= this —
                i.e. genuinely new entities appended since the last save.
                Existing entity-to-entity distances already in the cache
                are left untouched. Caller must already have called
                resize() to grow the matrices to the new entity_count.
            checkpoint_interval_secs: if > 0, calls `on_checkpoint()` at
                most once every this many wall-clock seconds of progress
                through the loop below. This loop can run for hours; before
                this parameter existed, the only save happened once, after
                the whole loop returned — a hard process kill mid-loop never
                reaches that point, so every restart discarded 100% of that
                run's progress and re-triggered the same multi-hour attempt
                from the same stale save. 0 (default) disables periodic
                checkpointing — unchanged behavior otherwise.
            on_checkpoint: callback invoked per the interval above —
                typically the caller's own `lambda: cache.save(path)`. Kept
                as a callback, not a hardcoded path, so this class stays
                I/O-path-agnostic; neurograph_rpc.py already owns the path.
            resume_watermark: an (i, j) pair previously recorded by an
                interrupted run's checkpoint (see DistanceCache.watermark).
                Pairs at or before it in canonical (max, min) order are
                skipped — they were already computed and saved. Because all
                pairs touching a newly appended entity sort AFTER every
                old-old pair, a resume on a grown graph covers both the
                unfinished old region and all new pairs. None (default) =
                no resume, unchanged behavior.
        """
        n = substrate.entity_count()
        if n == 0:
            return

        entities = substrate.entities()
        if start_index > 0:
            logger.info(
                "Populating distance cache incrementally for entities "
                "%d..%d (of %d total)...", start_index, n - 1, n,
            )
        else:
            logger.info("Populating distance cache for %d entities...", n)

        # Get all connected pairs from the substrate
        connected_pairs = set()

        # Synaptic connections (direct)
        graph = getattr(substrate, '_graph', None)
        if graph is not None:
            for syn in graph.synapses.values():
                try:
                    i = substrate.entity_index(syn.pre_node_id)
                    j = substrate.entity_index(syn.post_node_id)
                    connected_pairs.add((min(i, j), max(i, j)))
                except (KeyError, ValueError):
                    continue

            # Hyperedge co-membership
            for he in graph.hyperedges.values():
                member_ids = getattr(he, 'node_ids', [])
                if hasattr(he, 'member_nodes'):
                    member_ids = he.member_nodes
                indices = []
                for nid in member_ids:
                    try:
                        indices.append(substrate.entity_index(nid))
                    except (KeyError, ValueError):
                        continue
                for a_idx, a in enumerate(indices):
                    for b in indices[a_idx + 1:]:
                        connected_pairs.add((min(a, b), max(a, b)))

        # Also add 2-hop neighbors from existing connections
        adj = {i: set() for i in range(n)}
        for i, j in connected_pairs:
            adj[i].add(j)
            adj[j].add(i)
        two_hop_pairs = set()
        for node, neighbors in adj.items():
            for neighbor in neighbors:
                for second in adj.get(neighbor, set()):
                    if second != node:
                        two_hop_pairs.add((min(node, second), max(node, second)))
        connected_pairs |= two_hop_pairs

        if start_index > 0:
            # Incremental: only pairs touching a genuinely new entity.
            # Existing entity-to-entity distances are already in the cache
            # (preserved by resize()) and don't need recomputing.
            connected_pairs = {
                (i, j) for (i, j) in connected_pairs
                if i >= start_index or j >= start_index
            }

        # Canonical processing order: (max, min)-sorted. Deterministic under
        # append-stable entity indexing (a set's iteration order is not
        # reproducible across processes), and every pair touching a newly
        # appended entity sorts after all old-old pairs — which is what lets
        # a resume_watermark compose with graph growth in one filter.
        ordered_pairs = sorted(connected_pairs, key=lambda p: (p[1], p[0]))

        if resume_watermark is not None:
            _wm_key = (resume_watermark[1], resume_watermark[0])
            _before = len(ordered_pairs)
            ordered_pairs = [p for p in ordered_pairs if (p[1], p[0]) > _wm_key]
            logger.info(
                "Resuming distance-cache populate from watermark (%d, %d): "
                "%d of %d pairs already done, %d remaining",
                resume_watermark[0], resume_watermark[1],
                _before - len(ordered_pairs), _before, len(ordered_pairs),
            )

        logger.info("Computing distances for %d connected pairs", len(ordered_pairs))

        # Mark populated before the expensive loop, not after — a save()
        # following a mid-loop crash below (e.g. a concurrent graph
        # mutation this snapshot-based code doesn't fully protect against)
        # should still persist whatever was computed rather than
        # discarding it because _populated never flipped true.
        self._populated = True
        _last_checkpoint = time.monotonic()
        for idx, (i, j) in enumerate(ordered_pairs):
            eid_i = substrate.index_to_entity(i)
            eid_j = substrate.index_to_entity(j)
            dvec = substrate.distance_vector(eid_i, eid_j)
            for c in range(NUM_DIST_COMPONENTS):
                if abs(dvec[c]) > 1e-15:
                    self._components_lil[c][i, j] = dvec[c]
                    self._components_lil[c][j, i] = dvec[c]
            # Periodic checkpoint, only time-checked every 1000 pairs so
            # time.monotonic() itself doesn't add per-pair overhead across
            # a loop that can run into the millions of iterations.
            if (checkpoint_interval_secs > 0 and on_checkpoint is not None
                    and idx % 1000 == 0):
                _now = time.monotonic()
                if _now - _last_checkpoint >= checkpoint_interval_secs:
                    try:
                        # Record how far we got BEFORE saving, so the
                        # checkpoint carries its own resume point. (i, j)
                        # is complete at this line — the checkpoint sits
                        # after the distance computation above.
                        self._watermark = (i, j)
                        self._rebuild_csr()  # save() reads the CSR form
                        on_checkpoint()
                        _last_checkpoint = _now
                        logger.info(
                            "Lenia periodic checkpoint: %d/%d pairs processed",
                            idx + 1, len(ordered_pairs),
                        )
                    except Exception:
                        logger.exception(
                            "Periodic Lenia checkpoint failed (non-fatal) at "
                            "pair %d/%d", idx + 1, len(ordered_pairs),
                        )

        # Loop completed — the cache is whole; a finished save must not
        # carry a resume point.
        self._watermark = None
        self._rebuild_csr()
        logger.info("Distance cache populated: %d pairs", len(ordered_pairs))

    def _rebuild_csr(self):
        """Convert lil matrices to CSR for fast row access."""
        self._components_csr = [
            comp.tocsr() for comp in self._components_lil
        ]
        # Build magnitude matrix: sqrt(sum of squares across components)
        magnitude_sq = sparse.csr_matrix(
            (self._n, self._n), dtype=np.float64
        )
        for csr in self._components_csr:
            magnitude_sq = magnitude_sq + csr.multiply(csr)
        self._magnitude_csr = magnitude_sq.sqrt()

    def get_csr(self, component: int) -> sparse.csr_matrix:
        """Get CSR matrix for a component. Rebuilds if dirty."""
        if self._components_csr is None:
            self._rebuild_csr()
        return self._components_csr[component]

    def get_neighbors_sparse(
        self, entity_idx: int, max_range: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return neighbor indices and their distance magnitudes.

        Returns:
            (neighbor_indices, magnitudes) — both 1D numpy arrays.
            Only includes pairs within max_range.
        """
        if self._magnitude_csr is None:
            self._rebuild_csr()

        row = self._magnitude_csr.getrow(entity_idx)
        indices = row.indices
        magnitudes = row.data

        mask = magnitudes <= max_range
        return indices[mask], magnitudes[mask]

    def get_neighbor_field_values(
        self, entity_idx: int, max_range: float, component: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return neighbor indices and their distance values for one component.

        Filtered to neighbors within max_range of combined magnitude.
        """
        neighbor_idx, _ = self.get_neighbors_sparse(entity_idx, max_range)
        if len(neighbor_idx) == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

        csr = self.get_csr(component)
        row = csr.getrow(entity_idx)
        # Get values for the filtered neighbors
        values = np.array([
            row[0, j] if j in row.indices else 0.0
            for j in neighbor_idx
        ], dtype=np.float64)

        return neighbor_idx, values

    def mark_dirty(self, i: int, j: int, component: int):
        """Mark a distance component as needing recomputation."""
        self._dirty.add((i, j, component))
        self._dirty.add((j, i, component))

    def mark_dirty_entity(self, entity_idx: int, component: int):
        """Mark all pairs involving an entity as dirty for a component."""
        # Only mark existing connections, not all pairs
        if self._components_csr is not None:
            csr = self._components_csr[component]
            row = csr.getrow(entity_idx)
            for j in row.indices:
                self._dirty.add((entity_idx, j, component))
                self._dirty.add((j, entity_idx, component))
        else:
            for j in range(self._n):
                if j != entity_idx:
                    self._dirty.add((entity_idx, j, component))
                    self._dirty.add((j, entity_idx, component))

    def sweep_dirty(self, substrate: LeniaSubstrate):
        """Recompute all dirty entries from the substrate."""
        if not self._dirty:
            return

        entities = substrate.entities()
        for i, j, component in list(self._dirty):
            if i >= len(entities) or j >= len(entities):
                continue
            eid_i = substrate.index_to_entity(i)
            eid_j = substrate.index_to_entity(j)
            dvec = substrate.distance_vector(eid_i, eid_j)
            self._components_lil[component][i, j] = dvec[component]
            self._components_lil[component][j, i] = dvec[component]

        self._dirty.clear()
        self._rebuild_csr()

    def resize(self, new_count: int, new_entity_ids: Optional[List[str]] = None):
        """Resize for entity addition/removal.

        new_entity_ids, if given, replaces the cache's known entity-id
        ordering (e.g. the substrate's post-growth entities()) so a
        subsequent save() keeps entity_ids in sync with the new size.
        Existing rows/cols are preserved in place — callers doing an
        append-only growth (via NeuroGraphSubstrate's known_entity_order)
        can safely follow this with populate(start_index=old_count) to
        fill in only the new entities instead of recomputing everything.
        """
        new_components = []
        for comp in self._components_lil:
            new_mat = sparse.lil_matrix(
                (new_count, new_count), dtype=np.float64
            )
            copy_n = min(self._n, new_count)
            new_mat[:copy_n, :copy_n] = comp[:copy_n, :copy_n]
            new_components.append(new_mat)
        self._components_lil = new_components
        self._components_csr = None
        self._magnitude_csr = None
        self._n = new_count
        if new_entity_ids is not None:
            self._entity_ids = list(new_entity_ids)

    def reconcile_removals(self, live_ids: Set[str]) -> Optional[List[str]]:
        """Compact the cache in place after entity removals (#371).

        Drops every cached entity not in `live_ids` from the six component
        matrices (order-preserving keep-slice, O(nnz) — trivial next to a
        recompute), remaps the dirty set, and translates the resume
        watermark through the reindex so an interrupted rebuild can still
        resume after a prune. Before this method, ANY removal forced the
        callers to discard the whole cache — hours (CC) to days (Syl) of
        computed distances lost per restart-after-prune.

        Returns the surviving entity ordering (callers pass it to
        NeuroGraphSubstrate as known_entity_order), or None when the cache
        cannot be preserved and the caller must take the legacy
        full-rebuild path: no entity_ids on the cache (pre-2026-07
        format), nothing survives, or an interrupted rebuild's cut cannot
        be translated (no computed pair at or below it survives).

        Safe because compaction is monotone — surviving indices keep
        their relative order, so the canonical (max, min) pair ordering
        populate() processes in is preserved, and "everything at or below
        the cut is exactly the already-computed region" survives the
        reindex. Does not widen the documented connectivity-drift blind
        spot (see file header): the fallback cut only ever moves DOWN.
        """
        if not self._entity_ids:
            return None
        n_old = len(self._entity_ids)
        keep_idx = [i for i, eid in enumerate(self._entity_ids)
                    if eid in live_ids]
        if len(keep_idx) == n_old:
            return list(self._entity_ids)   # nothing removed — no-op
        if not keep_idx:
            return None                     # nothing survives

        keep_mask = np.zeros(n_old, dtype=bool)
        keep_mask[keep_idx] = True
        new_index = np.full(n_old, -1, dtype=np.int64)
        new_index[keep_idx] = np.arange(len(keep_idx), dtype=np.int64)

        # Translate the watermark BEFORE slicing — the fallback scan reads
        # the old-index nonzero coordinates.
        old_wm = self._watermark
        if old_wm is not None:
            new_wm = self._translate_watermark(old_wm, keep_mask, new_index)
            if new_wm is None:
                return None   # cut untranslatable — caller full-rebuilds
            self._watermark = new_wm

        keep = np.asarray(keep_idx, dtype=np.int64)
        self._components_lil = [
            comp.tocsr()[keep][:, keep].tolil()
            for comp in self._components_lil
        ]
        self._components_csr = None
        self._magnitude_csr = None
        self._dirty = {
            (int(new_index[i]), int(new_index[j]), c)
            for (i, j, c) in self._dirty
            if 0 <= i < n_old and 0 <= j < n_old
            and keep_mask[i] and keep_mask[j]
        }
        survivors = [self._entity_ids[i] for i in keep_idx]
        self._n = len(keep_idx)
        self._entity_ids = survivors
        logger.info(
            "Distance cache reconciled after prune: %d entities removed, "
            "%d survive%s",
            n_old - len(keep_idx), len(keep_idx),
            "" if old_wm is None
            else f"; resume watermark {old_wm} -> {self._watermark}",
        )
        return survivors

    def _translate_watermark(
        self,
        wm: Tuple[int, int],
        keep_mask: np.ndarray,
        new_index: np.ndarray,
    ) -> Optional[Tuple[int, int]]:
        """Map an interrupted run's (min, max) watermark pair through a
        removal reindex (#371).

        Both endpoints alive -> direct translation (monotone compaction
        preserves order). An endpoint dead -> the exact cut pair is gone;
        fall back to the greatest surviving COMPUTED pair at or below the
        cut in canonical (max, min) order, read from the matrices' nonzero
        coordinates. The fallback only moves the cut DOWN: pairs between
        the fallback cut and the true cut get recomputed on resume
        (idempotent, safe); moving the cut up would silently skip
        never-computed pairs. Returns None when no computed pair at or
        below the cut survives — the caller falls back to a full rebuild.

        Note the all-zero-vector corner: populate() only stores components
        with |value| > 1e-15, so a computed pair whose entire distance
        vector was ~0 has no nonzero coordinates and is invisible to the
        fallback scan. That can only pick a LOWER cut than the true one —
        the safe direction.
        """
        wi, wj = wm   # stored as (min, max) — see populate()'s loop
        if keep_mask[wi] and keep_mask[wj]:
            return (int(new_index[wi]), int(new_index[wj]))

        wm_hi, wm_lo = wj, wi   # canonical key = (max, min)
        best_key = None
        best_pair = None
        for comp in self._components_lil:
            coo = comp.tocsr().tocoo()
            r, c = coo.row, coo.col
            upper = r < c          # symmetric storage — visit (min, max) once
            r, c = r[upper], c[upper]
            surv = keep_mask[r] & keep_mask[c]
            r, c = r[surv], c[surv]
            le_cut = (c < wm_hi) | ((c == wm_hi) & (r <= wm_lo))
            r, c = r[le_cut], c[le_cut]
            if r.size == 0:
                continue
            order = np.lexsort((r, c))   # sort by (c, r) = (max, min)
            ri, ci = int(r[order[-1]]), int(c[order[-1]])
            if best_key is None or (ci, ri) > best_key:
                best_key = (ci, ri)
                best_pair = (ri, ci)
        if best_pair is None:
            return None
        return (int(new_index[best_pair[0]]), int(new_index[best_pair[1]]))

    # Legacy compatibility for tests
    def set_distance_compat(self, i, j, component, value):
        self.set_distance(i, j, component, value)

    def get_distance_vector(self, i: int, j: int) -> np.ndarray:
        """Return the full distance vector for a pair."""
        vec = np.zeros(NUM_DIST_COMPONENTS, dtype=np.float64)
        for c in range(NUM_DIST_COMPONENTS):
            vec[c] = self._components_lil[c][i, j]
        return vec

    def get_neighbors(
        self, entity_idx: int, max_range: float
    ) -> List[Tuple[int, np.ndarray]]:
        """Legacy: Return neighbors as list of (idx, distance_vector).

        Use get_neighbors_sparse for performance.
        """
        idx, mags = self.get_neighbors_sparse(entity_idx, max_range)
        return [(int(j), self.get_distance_vector(entity_idx, j)) for j in idx]

    @property
    def dirty_count(self) -> int:
        return len(self._dirty)


class KernelComputer:
    """Vectorized kernel-weighted influence computation.

    Uses sparse matrix operations instead of Python loops.
    For each channel, gathers neighbors within effective range,
    applies kernel function to distance magnitudes, multiplies by
    neighbor field state, and sums — all vectorized.
    """

    def __init__(
        self,
        cache: DistanceCache,
        registry: ChannelRegistry,
    ):
        self._cache = cache
        self._registry = registry

    def compute(
        self,
        field_state: np.ndarray,
        channel_id: int,
    ) -> np.ndarray:
        """Compute kernel-weighted influence for all entities on one channel.

        Vectorized: uses sparse neighbor lookup + numpy broadcast.
        Falls back to per-entity loop only for very sparse graphs.

        Args:
            field_state: (entity_count, channel_count) current field state.
            channel_id: Which channel to compute.

        Returns:
            (entity_count,) array of total influence per entity.
        """
        n = field_state.shape[0]
        channel_idx = self._registry.channel_index(channel_id)
        effective_range = self._registry.effective_range(channel_id)
        kernel_spec = self._registry.kernel_shape(channel_id)

        influences = np.zeros(n, dtype=np.float64)

        if not self._cache.populated and self._cache._components_csr is None:
            # No distances computed yet — nothing to do
            return influences

        # Get the magnitude matrix filtered by range
        mag_csr = self._cache._magnitude_csr
        if mag_csr is None:
            return influences

        # Channel's field state column
        channel_field = field_state[:, channel_idx]

        # Process all entities using sparse row iteration
        for i in range(n):
            row = mag_csr.getrow(i)
            if row.nnz == 0:
                continue

            # Filter by effective range
            mask = row.data <= effective_range
            if not mask.any():
                continue

            neighbor_idx = row.indices[mask]
            neighbor_dist = row.data[mask]

            # Vectorized kernel evaluation on distance magnitudes
            kernel_weights = evaluate(kernel_spec, neighbor_dist)

            # Vectorized: kernel_weights * neighbor field values
            neighbor_values = channel_field[neighbor_idx]
            influences[i] = np.dot(kernel_weights, neighbor_values)

        return influences

    def compute_all_channels(
        self, field_state: np.ndarray
    ) -> Dict[int, np.ndarray]:
        """Compute influences for all channels.

        Returns dict mapping channel_id → influence array.
        """
        result = {}
        for cid in self._registry.channel_ids:
            result[cid] = self.compute(field_state, cid)
        return result
