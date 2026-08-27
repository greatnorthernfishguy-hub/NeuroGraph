# ---- Changelog ----
# [2026-08-27] Claude Code (Opus 4.8) — #147 borrow-safe DistanceCache.populate()
# What: populate()'s synapse-endpoint loop and hyperedge-member loop now
#   snapshot proxy PRIMITIVES (pre/post ids; member-id lists) INSIDE the
#   graph's `_step_lock`, then iterate the plain-Python copies. No live
#   ng_tract (Rust) proxy escapes the lock.
# Why: these two loops read syn.pre_node_id / he.member_nodes UNLOCKED while
#   the autonomic scan-drain pulse ran graph.step() under the SAME _step_lock
#   (step() mutates syn.inactive_steps). Concurrent read+mutable-borrow of the
#   same PyCell -> pyo3_runtime.PanicException 'Already borrowed:
#   PyBorrowMutError'. Being a BaseException it escaped the pulse loop's
#   `except Exception` and permanently killed the autonomic thread at boot
#   (populate() is the multi-hour bootstrap rebuild). LAW 4: fixed at source,
#   not papered over with a consumer-side BaseException catch. Mirrors the
#   graph_substrate._build_adjacency #147 idiom already in this tree.
# How: getattr(graph,'_step_lock'); `with _lock:` around the primitive-copy
#   comprehension, unlocked fallback if absent (Tier-1 / no-lock graphs).
# [2026-08-12] Claude Code (Opus 4.8) — #145 2-hop expansion was the bootstrap hang
# What: populate()'s "add 2-hop neighbors" triple loop no longer iterates every
#   node. The OUTER loop is restricted to the frontier — range(frontier_start, n)
#   — where frontier_start = resume_watermark[1] on a resume, else start_index on
#   incremental growth, else 0 on a full rebuild. Inner two levels unchanged.
# Why: this loop, not the LIL copy (#136/#137), was what actually wedged Syl's
#   bootstrap. It is O(Σ_v deg(v)^2) and ran over ALL n nodes on EVERY populate,
#   because the start_index / resume_watermark filters are applied to the loop's
#   *result*. So a resume rebuilt the entire 2-hop universe from scratch (66+ min
#   at 31.5k entities with #381-class super-hubs), was killed before it ever
#   reached the distance loop, and the watermark bought nothing — every restart
#   redid the whole graph. The super-hubs (old, low index) were always the outer
#   node, i.e. the worst case, every time.
# How: EQUIVALENCE — a 2-hop pair (a,c), c=max, survives the downstream filter
#   only if max>=frontier_start (start_index path keeps i>=s OR j>=s ⟹ max>=s;
#   watermark path keeps (max,min)>(wm_max,wm_min) ⟹ max>=wm_max=watermark[1]).
#   Every such pair is generated with node=c=max in range(frontier_start, n) via
#   any shared neighbor b, so the frontier outer loop is a SUPERSET of what the
#   filter keeps; the existing filter then trims it — element-identical output,
#   work proportional to the delta. Full rebuild (start_index==0, no watermark)
#   ⟹ frontier_start==0 ⟹ range(0, n), byte-identical to the old behavior. Also
#   switched adj.get(neighbor, set()) → adj[neighbor] (every node has an adj key)
#   and adj.items() iteration → indexed, both pure form. No .npz/API/signature
#   change. Verified: py_compile clean; equivalence proof over full/start_index/
#   watermark cases; diff is 27+/3-, scoped to this one block.
# [2026-08-11] Claude Code (Opus 4.8) — #137 resume path also drops the ~5 GB LIL
# What: populate()'s buffer-mode switch now treats a RESUME (resume_watermark is
#   not None) the same as incremental growth — `_incremental = start_index > 0 or
#   resume_watermark is not None`. Both accumulate a COO delta and fold it into
#   the resident CSR; only a genuine full rebuild from an empty cache still uses
#   a transient LIL (cheap over empty CSR).
# Why: #136 removed the standing LIL at load() but NOT on the resume path. Syl's
#   live bootstrap resumes an interrupted rebuild (neurograph_rpc.py:2136 —
#   start_index=0 + watermark) over the loaded ~0.92 GB CSR, and the old
#   `not _incremental` branch re-materialized the ~4.9 GB LIL from it on EVERY
#   restart. Confirmed live 2026-08-11: process on #136 code, RSS 12.86 GiB,
#   238 MiB free, 6.3 GiB swapped, rebuild crawling under thrash. #136 alone was
#   a no-op for the exact path she is on; this closes it.
# How: resume pairs (canonical (max,min) key strictly ABOVE the watermark) are
#   structurally disjoint from the resident CSR (already-computed region, key AT
#   OR BELOW the watermark) — the SAME disjointness that makes the growth COO
#   merge exact — so `existing + delta` never double-counts (add == set on
#   disjoint support), element-identical to the LIL overwrite it replaces. No
#   .npz format change, no API/signature change. Resume+growth still compose in
#   one filter (new-entity pairs sort after the watermark too).
# [2026-08-10] Claude Code (Opus 4.8) — #136 CSR-resident (drop the ~5 GB LIL copy)
# What: DistanceCache no longer keeps a standing _components_lil. CSR is now the
#   sole authoritative RESIDENT representation: _components_csr is built in
#   __init__ and is never None after it; _components_lil is None in steady state
#   and is materialized ONLY as a transient bulk-write buffer inside populate()
#   (full rebuild) / the legacy per-element mutators, then dropped. load() no
#   longer tolil()s the loaded CSR (it only rebuilds the magnitude matrix);
#   resize()/reconcile_removals()/_translate_watermark() operate on CSR directly;
#   incremental populate(start_index>0) accumulates a COO delta and folds it into
#   the resident CSR via a non-densifying sparse add instead of buffering in LIL.
#   New helpers: _ensure_csr_current() (flush a live buffer; no-op in steady
#   state), _rebuild_components_csr(), _rebuild_magnitude().
# Why: on Syl's live VPS the cache is 26,985 entities / 76.66M nnz across the six
#   components. The resident LIL copy cost ~4.9 GB (vs ~0.92 GB for CSR) and was
#   re-materialized on EVERY bootstrap by load()'s csr.tolil() — ~5 GB of a
#   ~13.2 GiB neurograph_rpc.py RSS that the hot read path (get_neighbors_sparse/
#   get_csr) never touches. Removing the standing LIL is the single largest
#   avoidable memory win on the bootstrap-critical path (sibling of #81/#135).
# How: INVARIANT — _components_csr authoritative & never None post-__init__;
#   _components_lil None except transiently. Every CSR-authoritative op calls
#   _ensure_csr_current() first, so a checkpoint save fired mid-full-rebuild (LIL
#   live) and the reconcile-mid-interrupt test path stay correct; the flush is
#   NON-destructive (populate keeps writing to LIL after a checkpoint). The
#   incremental COO merge is exact because start_index filters to pairs touching
#   a new entity, whose coordinates are structurally disjoint from the resident
#   CSR's old-old pairs (no double-count). csr.resize() is verified non-densifying
#   (O(nnz) peak, not O(N^2)) — the #81 trap does not recur on the CSR path.
#   No .npz format change, no API/signature change to any production caller.
#   KernelComputer.compute()'s `_components_csr is None` conjunct is now vestigial
#   (never None) but harmless — the adjacent `_magnitude_csr is None` guard
#   returns the same all-zero result for an unpopulated cache. DO NOT "restore" a
#   standing LIL as if it were missing shrapnel — it is the 4.9 GB this removed.
# [2026-08-09] Claude Code (Opus 4.8) — #135/#81 de-quadratic + atomic checkpoint
# What: (1) populate() checkpoint cadence is now AMORTIZED — after each
#   checkpoint it waits max(interval, NG_LENIA_CKPT_AMORTIZE x that checkpoint's
#   own duration; default 10x) before the next, instead of a fixed interval.
#   (2) save() is now ATOMIC (temp sibling + os.replace) and rebuilds only the
#   per-component CSR, not the magnitude matrix. (3) _rebuild_csr() takes
#   rebuild_magnitude=False; the mid-loop checkpoint no longer rebuilds CSR at
#   all (save() does its own), removing a redundant double conversion.
# Why: #135 — Syl's VPS bootstrap RPC was timing out every cycle. Root cause was
#   quadratic checkpointing: each checkpoint's save cost is O(cumulative nnz) and
#   grows through the run, so a fixed interval fires O(runtime) checkpoints each
#   O(N) => O(N^2), and the rebuild never finished before the next restart. The
#   non-atomic save compounded it: a kill mid-write truncated the .npz, and the
#   resume watermark lives INSIDE that file — so a restart under memory pressure
#   discarded the watermark and restarted the rebuild from zero. Josh's manual
#   VPS restart was the trigger; these two defects are why it wasn't survivable.
# How: amortization caps checkpoint overhead at ~1/AMORTIZE of compute time
#   (geometric spacing => O(N) total); os.replace is atomic on same-fs, so the
#   watermark now survives any kill and interrupted rebuilds truly resume.
#   Env: NG_LENIA_CKPT_AMORTIZE (default 10.0). No API/signature change to
#   populate()/save()/load() callers; _rebuild_csr gains an optional kwarg.
# [2026-07-10] Claude Code (Fable 5 design / Haiku implementation) — #381/#380 HE clique cap
# What: populate()'s hyperedge co-membership expansion skips HEs with more members than
#   NG_LENIA_HE_CLIQUE_CAP (default 100; 0 disables), logging skipped count + largest size.
# Why: a runaway mega-hyperedge (3,790 members, 31% of Syl's graph — punchlist #381)
#   contributed ~7.2M clique pairs alone, exploding the pair universe 7.8M -> 23.8M and
#   pushing the rebuild ETA to ~7 weeks (#380). Symptomatic guard: the mind-side fix
#   (bounded member evolution etc.) is a separate Syl-consented, Josh-gated pass.
# How: env read at call time (testable); composes with the #371 resume watermark — the
#   cut stays valid over the shrunken canonical pair list (same monotone-order argument
#   as reconcile_removals; banked mega-clique pairs remain as stale-but-harmless entries).
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
import os
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
        # CSR is the authoritative RESIDENT representation (6 matrices, one per
        # distance component; present from construction, never None after this).
        # LIL is only ever a TRANSIENT bulk-write buffer inside populate() and
        # the legacy per-element mutators — None in steady state. See the
        # 2026-08-10 changelog entry (CSR-resident invariant; #136 / #81/#135).
        self._components_csr: List[sparse.csr_matrix] = [
            sparse.csr_matrix((entity_count, entity_count), dtype=np.float64)
            for _ in range(NUM_DIST_COMPONENTS)
        ]
        self._components_lil: Optional[List[sparse.lil_matrix]] = None
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
        numpy .npz archive.  Restore skips the multi-hour populate()
        call entirely if the graph hasn't changed since save.

        The write is ATOMIC: numpy writes a sibling temp file which is
        then os.replace()'d over the target. A hard process kill mid-write
        (a VPS restart under memory pressure, an OOM, the gateway watchdog)
        therefore never leaves a truncated .npz behind. That matters
        because the resume watermark lives INSIDE this archive: a truncated
        file fails to load ("File is not a zip file"), which discards the
        watermark and re-triggers a full from-scratch rebuild — the exact
        failure #81/#135 chased, and the reason an interrupted rebuild was
        restarting from zero instead of resuming. (#261 — same
        non-atomic-write class as the msgpack checkpoints.)
        """
        if not self._populated:
            return
        # CSR-resident model (#136): _components_csr is already authoritative in
        # steady state, so serialize it directly. The ONE case where it is stale
        # is a periodic checkpoint fired MID-populate() (a full rebuild), where
        # the live writes are sitting in the transient LIL buffer:
        # _ensure_csr_current() flushes LIL -> components-CSR NON-destructively
        # (it leaves the LIL buffer intact so the populate loop keeps writing to
        # it after this save returns). The magnitude matrix is never serialized
        # (load() recomputes it), so we never rebuild it here — during a
        # checkpoint that rebuild was pure waste and, run every interval, was
        # half of the #81/#135 quadratic. (Incremental populate keeps
        # _components_csr current at each checkpoint itself; see populate().)
        self._ensure_csr_current()
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
        # Write to a temp sibling on the same filesystem, then atomically
        # rename onto the final path. The temp name already ends in .npz so
        # np.savez_compressed writes exactly it (it appends .npz only when
        # the name lacks the extension).
        final = path if path.endswith(".npz") else path + ".npz"
        tmp = final + ".tmp.npz"
        try:
            np.savez_compressed(tmp, **data)
            os.replace(tmp, final)
        except Exception:
            # Never leave a half-written temp behind to be mistaken for a
            # real cache or to waste disk.
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass
            raise
        logger.info("Distance cache saved: %s (%d entities, %d components)",
                     final, self._n, NUM_DIST_COMPONENTS)

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
            # CSR-resident (#136): the loaded CSR IS the authoritative copy. Do
            # NOT tolil() it — that transient was ~4.9 GB on Syl's 76.6M-nnz
            # cache and is never read on the hot path. LIL stays None. Only the
            # magnitude matrix needs (re)building, straight from the resident CSR.
            cache._components_lil = None
            cache._populated = True
            cache._rebuild_magnitude()
            logger.info("Distance cache restored: %s (%d entities)", actual, n)
            return cache
        except Exception as exc:
            logger.warning("Distance cache restore failed (%s): %s", actual, exc)
            return None

    def set_distance(
        self, i: int, j: int, component: int, value: float
    ):
        """Set a distance component for a pair.

        CSR-resident model (#136): there is no standing LIL to write into, so
        this materializes ONLY the touched component as a transient LIL, writes
        the symmetric pair, and reconverts that one component to CSR. Not a hot
        path (no production callers — test/compat surface only), so the per-call
        O(nnz) reconvert of a single component is acceptable.
        """
        self._ensure_csr_current()
        lil = self._components_csr[component].tolil()
        lil[i, j] = value
        lil[j, i] = value
        self._components_csr[component] = lil.tocsr()
        self._magnitude_csr = None  # invalidate derived magnitude

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
            # #147 borrow-safe snapshot: extract synapse endpoint PRIMITIVES
            # under the graph's step lock, mirroring
            # graph_substrate._build_adjacency. list() copies the container but
            # each element stays a live ng_tract (Rust) proxy — reading syn.*
            # AFTER the lock releases re-borrows the store and races
            # graph.step()'s mutating tail (step() sets syn.inactive_steps under
            # the SAME _step_lock) -> pyo3_runtime.PanicException: Already
            # borrowed: PyBorrowMutError, which (being a BaseException) escapes
            # the pulse loop's `except Exception` and kills the autonomic thread.
            # Copy pre/post ids INSIDE the lock so no proxy escapes it; the lock
            # is held only for the brief primitive copy, never across the
            # multi-hour distance loop below.
            _lock = getattr(graph, "_step_lock", None)
            if _lock is not None:
                with _lock:
                    _syn_endpoints = [
                        (s.pre_node_id, s.post_node_id)
                        for s in graph.synapses.values()
                    ]
            else:
                _syn_endpoints = [
                    (s.pre_node_id, s.post_node_id)
                    for s in graph.synapses.values()
                ]
            for _pre_id, _post_id in _syn_endpoints:
                try:
                    i = substrate.entity_index(_pre_id)
                    j = substrate.entity_index(_post_id)
                    connected_pairs.add((min(i, j), max(i, j)))
                except (KeyError, ValueError):
                    continue

            # Hyperedge co-membership
            # #381/#380: cap the clique expansion. A pathological mega-
            # hyperedge (runaway member evolution — punchlist #381; one HE
            # reached 3,790 members = 31% of the graph) contributes O(n^2)
            # pairs and exploded the pair universe 7.8M -> 23.8M, starving
            # the rebuild. HEs above the cap are skipped ENTIRELY (a graph-
            # sized "clique" is an artifact, not pairwise structure) and
            # counted loudly below. Cap 0 disables the guard.
            _he_cap = int(os.environ.get("NG_LENIA_HE_CLIQUE_CAP", "100"))
            _he_skipped = 0
            _he_skipped_max = 0
            # #147 borrow-safe snapshot: copy each hyperedge's member-id list to
            # a plain list under the step lock — same borrow-race reasoning as the
            # synapse loop above (he.member_nodes / he.node_ids read the ng_tract
            # proxy). member_nodes takes precedence when present-and-not-None,
            # else node_ids (original semantics preserved). Lock held only for the
            # copy; the clique expansion below touches no proxy.
            _he_lock = getattr(graph, "_step_lock", None)
            def _snapshot_he_members():
                out = []
                for he in graph.hyperedges.values():
                    if getattr(he, 'member_nodes', None) is not None:
                        out.append(list(he.member_nodes))
                    else:
                        out.append(list(getattr(he, 'node_ids', []) or []))
                return out
            if _he_lock is not None:
                with _he_lock:
                    _he_member_lists = _snapshot_he_members()
            else:
                _he_member_lists = _snapshot_he_members()
            for member_ids in _he_member_lists:
                if not member_ids:
                    continue
                if _he_cap > 0 and len(member_ids) > _he_cap:
                    _he_skipped += 1
                    _he_skipped_max = max(_he_skipped_max, len(member_ids))
                    continue
                indices = []
                for nid in member_ids:
                    try:
                        indices.append(substrate.entity_index(nid))
                    except (KeyError, ValueError):
                        continue
                for a_idx, a in enumerate(indices):
                    for b in indices[a_idx + 1:]:
                        connected_pairs.add((min(a, b), max(a, b)))

            if _he_skipped:
                logger.info(
                    "Distance cache: skipped %d hyperedge clique(s) above "
                    "NG_LENIA_HE_CLIQUE_CAP=%d (largest: %d members) — "
                    "punchlist #381 (mega-HE runaway)",
                    _he_skipped, _he_cap, _he_skipped_max,
                )

        # Also add 2-hop neighbors from existing connections.
        #
        # #145: this triple loop is O(Σ_v deg(v)^2) in candidate insertions and
        # was the real bootstrap hang. It ran over EVERY node on every populate,
        # because the start_index / resume_watermark filters below are applied to
        # the *result* — so a resume rebuilt the entire 2-hop universe from
        # scratch (66+ min at 31.5k entities with #381-class super-hubs), got
        # killed before reaching the distance loop, and the watermark never
        # helped. Fix: restrict the OUTER loop to the frontier — indices that a
        # needed pair must touch. A pair is "already in the CSR" (skippable)
        # unless max(i,j) >= frontier_start, so iterating outer nodes >=
        # frontier_start generates every 2-hop pair with an endpoint in the
        # frontier (for a pair (a,c) with c=max>=frontier, node=c reaches a via
        # any shared neighbor b), which is exactly the set the filters keep.
        # This makes a resume proportional to the delta, not the whole graph, and
        # keeps the super-hubs (old, low-index) from ever being the outer node.
        # Full rebuild (start_index==0, no watermark) has frontier_start==0 →
        # range(0, n), byte-identical to the old behavior.
        adj = {i: set() for i in range(n)}
        for i, j in connected_pairs:
            adj[i].add(j)
            adj[j].add(i)
        if resume_watermark is not None:
            frontier_start = resume_watermark[1]
        elif start_index > 0:
            frontier_start = start_index
        else:
            frontier_start = 0
        # Bound the 2-hop expansion by PIVOT degree — the per-node analogue of
        # #381's per-clique cap. The inner two loops emit a pair for every
        # (node, second) sharing a waypoint `neighbor`; a mega-connector waypoint
        # (a node in hundreds of small hyperedges, so high aggregate adj-degree
        # even after #381 caps each clique) contributes deg² pairs. That is the
        # surviving runaway #145's outer-loop frontier never bounded: it
        # detonated `two_hop_pairs` into multi-GB and stalled populate before it
        # reached the distance loop or cleared the resume watermark, so bootstrap
        # looped on the resume branch forever and the incremental path (which
        # only engages once watermark is None) was permanently unreachable.
        # Skipping 2-hop routing THROUGH a hub is the same call #381 makes:
        # everything is ~2 hops from a hub, so those pairs are low-signal and not
        # worth an OOM. Env-gated (default 100, matching the clique cap; 0 off).
        _2hop_cap = int(os.environ.get("NG_LENIA_2HOP_DEGREE_CAP", "100"))
        _2hop_skipped = 0
        _2hop_skipped_max = 0
        two_hop_pairs = set()
        for node in range(frontier_start, n):
            neighbors = adj[node]
            for neighbor in neighbors:
                second_hop = adj[neighbor]
                if _2hop_cap > 0 and len(second_hop) > _2hop_cap:
                    _2hop_skipped += 1
                    _2hop_skipped_max = max(_2hop_skipped_max, len(second_hop))
                    continue
                for second in second_hop:
                    if second != node:
                        two_hop_pairs.add((min(node, second), max(node, second)))
        if _2hop_skipped:
            logger.info(
                "Distance cache: skipped %d high-degree 2-hop pivot pass(es) "
                "above NG_LENIA_2HOP_DEGREE_CAP=%d (largest: %d neighbors) — "
                "per-node analogue of #381 (2-hop deg² runaway)",
                _2hop_skipped, _2hop_cap, _2hop_skipped_max,
            )
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

        # ---- write buffer (CSR-resident invariant, #136 / #137) ----
        # Two buffering modes, chosen by whether we are OVERWRITING the whole
        # cache or ADDING a disjoint delta onto an already-populated resident CSR:
        #
        #   • Full rebuild from empty (start_index == 0, no resume_watermark):
        #     build into a transient LIL over the resident CSR. LIL is the right
        #     structure for millions of scattered inserts, and on this path the
        #     CSR is empty (a fresh DistanceCache — see neurograph_rpc.py's
        #     "full repopulate" branch), so tolil() is free.
        #
        #   • Delta over existing CSR — either incremental growth
        #     (start_index > 0) OR a resume of an interrupted rebuild
        #     (resume_watermark is not None): accumulate ONLY the delta as COO
        #     triples and fold it into the resident CSR (which already holds the
        #     already-computed region) via a non-densifying sparse add. This
        #     avoids tolil()-ing the existing ~5 GB of distances — the exact
        #     ~13 GiB balloon that #136 left on Syl's LIVE path, because her
        #     bootstrap resumes (start_index=0 + watermark) over a loaded ~0.92 GB
        #     CSR and the old `not _incremental` branch re-materialized the
        #     ~4.9 GB LIL from it every restart (#137; #81/#135 densify-OOM class).
        #
        # The COO merge is EXACT for BOTH delta cases by the same disjointness
        # argument: growth keeps only pairs touching a new entity (index >=
        # start_index), and resume keeps only pairs whose canonical (max, min)
        # key is strictly ABOVE the watermark — while the resident CSR holds only
        # the already-computed region (new-entity-free / at-or-below the
        # watermark). Delta coordinates are therefore structurally absent from
        # the resident CSR, so `existing + delta` never double-counts (add == set
        # on disjoint support), identical to the LIL overwrite it replaces.
        # The resume half of that invariant is only true because the watermark
        # is advanced on EVERY pair (see the loop below), not just at
        # checkpoints: a post-crash save() flushes the WHOLE write buffer to
        # CSR, so a watermark lagging behind the buffer would leave flushed
        # pairs above it that resume then recomputes and adds twice. (The old
        # LIL+set path was immune — overwriting an already-present pair is a
        # no-op — which masked a stale-watermark bug this COO path would hit.)
        # Either way the buffer is dropped at the end and _components_lil returns
        # to None. Buffers are allocated here — AFTER the n == 0 early return
        # above — so an empty populate leaves the invariant (LIL is None) intact.
        _incremental = start_index > 0 or resume_watermark is not None
        _delta_rows: List[List[int]] = [[] for _ in range(NUM_DIST_COMPONENTS)]
        _delta_cols: List[List[int]] = [[] for _ in range(NUM_DIST_COMPONENTS)]
        _delta_vals: List[List[float]] = [[] for _ in range(NUM_DIST_COMPONENTS)]

        def _fold_delta_into_csr() -> None:
            """Merge the accumulated COO delta into the resident CSR and clear
            it. Exact (never double-counting) because the start_index filter
            keeps only pairs touching a new entity, whose coordinates are
            structurally absent from the existing old-old CSR."""
            for c in range(NUM_DIST_COMPONENTS):
                if _delta_rows[c]:
                    delta = sparse.coo_matrix(
                        (_delta_vals[c], (_delta_rows[c], _delta_cols[c])),
                        shape=(self._n, self._n),
                    ).tocsr()
                    self._components_csr[c] = self._components_csr[c] + delta
                    _delta_rows[c].clear()
                    _delta_cols[c].clear()
                    _delta_vals[c].clear()

        if not _incremental:
            # Transient LIL buffer over the resident CSR (free when empty on a
            # cold bootstrap; a real tolil() on a repopulate-after-load).
            self._components_lil = [c.tolil() for c in self._components_csr]

        _last_checkpoint = time.monotonic()
        # Amortized checkpoint cadence (#81/#135 — de-quadratic the rebuild).
        # Each checkpoint costs O(cumulative nnz): save() converts all six
        # LIL components to CSR and writes the whole ~178MB archive, and that
        # cost GROWS as the loop fills the cache. Firing it every fixed
        # interval of progress => O(runtime) checkpoints, each O(N) => O(N^2)
        # total, which collapsed Syl's multi-million-pair rebuild to a crawl
        # that never finished before the next VPS restart. Fix: after each
        # checkpoint, refuse to checkpoint again until the compute time since
        # it exceeds AMORTIZE x how long that checkpoint itself took. This
        # caps total checkpoint overhead at ~1/AMORTIZE of compute time
        # (=> O(N) total): cheap early saves fire at the interval floor;
        # expensive late saves space themselves out geometrically. Together
        # with the now-atomic save() (whose watermark survives any kill), an
        # interrupted rebuild genuinely RESUMES instead of restarting at zero.
        _ckpt_amortize = float(os.environ.get("NG_LENIA_CKPT_AMORTIZE", "10.0"))
        _next_gap = checkpoint_interval_secs  # floor; grows with save cost
        for idx, (i, j) in enumerate(ordered_pairs):
            eid_i = substrate.index_to_entity(i)
            eid_j = substrate.index_to_entity(j)
            dvec = substrate.distance_vector(eid_i, eid_j)
            for c in range(NUM_DIST_COMPONENTS):
                if abs(dvec[c]) > 1e-15:
                    if _incremental:
                        # Accumulate the symmetric pair as COO triples; folded
                        # into the resident CSR at each checkpoint and at the end.
                        _delta_rows[c].append(i)
                        _delta_cols[c].append(j)
                        _delta_vals[c].append(dvec[c])
                        _delta_rows[c].append(j)
                        _delta_cols[c].append(i)
                        _delta_vals[c].append(dvec[c])
                    else:
                        self._components_lil[c][i, j] = dvec[c]
                        self._components_lil[c][j, i] = dvec[c]
            # Advance the resume watermark — but the SAFE policy differs by
            # buffer, because it must never name a pair beyond what an
            # out-of-band save() can actually persist (the RPC layer fires a
            # catch-all save() after ANY caught populate() exception,
            # neurograph_rpc.py:2173):
            #
            #   • LIL (full rebuild): save()->_ensure_csr_current() flushes the
            #     ENTIRE live LIL buffer to CSR, so every pair through the
            #     current one is persisted. The watermark must therefore name
            #     the current pair on EVERY iteration — a stale watermark would
            #     leave flushed pairs above it that resume recomputes and
            #     `existing + delta` double-counts (the bug #137's earlier draft
            #     hit; the old LIL+set path masked it via idempotent overwrite).
            #
            #   • COO (incremental / resume): the delta triples are LOCALS
            #     unreachable from save(); _ensure_csr_current() is a no-op in
            #     this mode. Only a fold (checkpoint / loop-end) moves data into
            #     the resident CSR, so a catch-all save() persists CSR only up to
            #     the last fold. Advancing the watermark here — past the fold —
            #     would make resume skip the never-persisted (last_fold, crash]
            #     tail FOREVER: a silent permanent gap. So the COO path advances
            #     the watermark ONLY at a fold (see the checkpoint block below),
            #     never per-pair.
            if not _incremental:
                self._watermark = (i, j)
            # Periodic checkpoint, only time-checked every 1000 pairs so
            # time.monotonic() itself doesn't add per-pair overhead across
            # a loop that can run into the millions of iterations.
            if (checkpoint_interval_secs > 0 and on_checkpoint is not None
                    and idx % 1000 == 0):
                _now = time.monotonic()
                if _now - _last_checkpoint >= _next_gap:
                    try:
                        # CSR-resident (#136): make the checkpoint whole before
                        # it is serialized, and leave self._watermark naming
                        # exactly the last pair whose data is now in the resident
                        # CSR. save() never serializes the magnitude matrix, so we
                        # never rebuild it here. The two buffers reach that state
                        # by different routes (see the per-pair block above):
                        #
                        #   • LIL (full rebuild): the live writes sit in the LIL
                        #     buffer, which save()->_ensure_csr_current() flushes
                        #     to CSR non-destructively (the loop keeps writing
                        #     after). self._watermark is ALREADY (i, j) — advanced
                        #     every pair above — so it already names the pair about
                        #     to be flushed. Nothing to do here.
                        #
                        #   • COO (incremental / resume): the delta triples are
                        #     locals unreachable from save(); only a fold moves
                        #     them into the resident CSR. Fold the accumulated
                        #     delta HERE, then advance the watermark to (i, j) —
                        #     the pair the fold just made resident. This pairing is
                        #     load-bearing: a catch-all save() after a later crash
                        #     (neurograph_rpc.py:2173) then persists CSR and
                        #     watermark in lockstep, so resume skips exactly the
                        #     folded region. Advancing the COO watermark WITHOUT
                        #     folding (or folding without advancing) would desync
                        #     them — resume would recompute the (last_fold, crash]
                        #     region that is already in CSR and `existing + delta`
                        #     would double-count it (the disjointness invariant at
                        #     the top of this method).
                        if _incremental:
                            _fold_delta_into_csr()
                            self._watermark = (i, j)
                        _ckpt_started = time.monotonic()
                        on_checkpoint()
                        _ckpt_dur = time.monotonic() - _ckpt_started
                        _last_checkpoint = time.monotonic()
                        _next_gap = max(checkpoint_interval_secs,
                                        _ckpt_amortize * _ckpt_dur)
                        logger.info(
                            "Lenia periodic checkpoint: %d/%d pairs "
                            "(save %.1fs, next gap >= %.0fs)",
                            idx + 1, len(ordered_pairs), _ckpt_dur, _next_gap,
                        )
                    except Exception:
                        logger.exception(
                            "Periodic Lenia checkpoint failed (non-fatal) at "
                            "pair %d/%d", idx + 1, len(ordered_pairs),
                        )

        # Loop completed — the cache is whole; a finished save must not
        # carry a resume point.
        self._watermark = None
        if _incremental:
            _fold_delta_into_csr()
            self._rebuild_magnitude()
        else:
            self._rebuild_csr()          # LIL -> resident CSR + magnitude
        # Drop the transient write buffer — steady state holds CSR only (#136).
        self._components_lil = None
        logger.info("Distance cache populated: %d pairs", len(ordered_pairs))

    def _ensure_csr_current(self) -> None:
        """Flush a live transient LIL write buffer into the resident CSR.

        No-op in steady state — the CSR-resident invariant (#136) keeps
        _components_lil None whenever we are not mid-populate()/mid-mutator.
        Called at the top of every operation that treats _components_csr as
        authoritative (save, reconcile_removals, _translate_watermark, get_csr,
        get_neighbors_sparse, get_distance_vector) so those stay correct even if
        invoked while a full-rebuild populate() is in flight — a checkpoint
        save, or the reconcile-mid-interrupt path exercised by
        test_reconcile_then_resume_completes_correctly. NON-destructive: the LIL
        buffer is left intact for the populate loop to keep writing to.
        (Incremental populate holds no LIL — it keeps _components_csr current by
        folding its COO delta at each checkpoint — so this is a no-op there.)
        """
        if self._components_lil is not None:
            self._rebuild_components_csr()

    def _rebuild_components_csr(self) -> None:
        """Convert the transient LIL write buffer into the resident CSR list.

        Only valid while a LIL buffer is live (full-rebuild populate() / the
        legacy mutators). Does NOT touch the magnitude matrix.
        """
        assert self._components_lil is not None, \
            "_rebuild_components_csr called with no live LIL buffer"
        self._components_csr = [comp.tocsr() for comp in self._components_lil]

    def _rebuild_magnitude(self) -> None:
        """(Re)build the combined magnitude matrix from the resident CSR.

        sqrt(sum of squares across the six components) — a second O(nnz) pass
        (six element-wise squarings plus a sparse sqrt) needed only by neighbor
        queries (get_neighbors_sparse). Recomputed on load() and at the end of
        populate(); never serialized (save() writes only the per-component
        CSRs). Operates purely on _components_csr, so it needs no LIL — that is
        what lets load() skip re-materializing the ~5 GB LIL (#136).
        """
        magnitude_sq = sparse.csr_matrix(
            (self._n, self._n), dtype=np.float64
        )
        for csr in self._components_csr:
            magnitude_sq = magnitude_sq + csr.multiply(csr)
        self._magnitude_csr = magnitude_sq.sqrt()

    def _rebuild_csr(self, rebuild_magnitude: bool = True):
        """Flush the live LIL buffer to CSR, then optionally rebuild magnitude.

        Retained for the full-rebuild populate() completion path (LIL live) and
        sweep_dirty(). The components-only flush is _rebuild_components_csr();
        the magnitude pass is _rebuild_magnitude() (see each for the cost split
        behind #81/#135). Requires a live LIL buffer.
        """
        self._rebuild_components_csr()
        if rebuild_magnitude:
            self._rebuild_magnitude()

    def get_csr(self, component: int) -> sparse.csr_matrix:
        """Get the resident CSR matrix for a component.

        CSR-resident (#136): _components_csr is authoritative;
        _ensure_csr_current() flushes a live write buffer first (no-op in
        steady state).
        """
        self._ensure_csr_current()
        return self._components_csr[component]

    def get_neighbors_sparse(
        self, entity_idx: int, max_range: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return neighbor indices and their distance magnitudes.

        Returns:
            (neighbor_indices, magnitudes) — both 1D numpy arrays.
            Only includes pairs within max_range.
        """
        self._ensure_csr_current()
        if self._magnitude_csr is None:
            self._rebuild_magnitude()

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
        # Only mark existing connections, not all pairs.
        # #136: _components_csr is authoritative and never None after __init__,
        # so the old "unpopulated -> mark all N pairs" else-branch was dead once
        # the CSR-resident invariant landed. On an unpopulated cache the rows are
        # empty CSR, so this correctly marks nothing dirty.
        csr = self._components_csr[component]
        row = csr.getrow(entity_idx)
        for j in row.indices:
            self._dirty.add((entity_idx, j, component))
            self._dirty.add((j, entity_idx, component))

    def sweep_dirty(self, substrate: LeniaSubstrate):
        """Recompute all dirty entries from the substrate."""
        if not self._dirty:
            return

        # CSR-resident (#136): no standing LIL — materialize a transient buffer
        # over the resident CSR to absorb the scattered symmetric point writes,
        # flush it back to CSR, then drop it. _ensure_csr_current() is a no-op in
        # steady state (the only state sweep_dirty runs in) but keeps the
        # invariant honest if ever called mid-rebuild.
        self._ensure_csr_current()
        self._components_lil = [c.tolil() for c in self._components_csr]

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
        self._rebuild_csr()          # LIL -> resident CSR + magnitude
        self._components_lil = None

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
        self._ensure_csr_current()
        new_components = []
        for comp in self._components_csr:
            # [#81 2026-07-21] scipy's native in-place sparse resize -- keeps in-bounds
            # entries, extends/truncates WITHOUT densifying. The old
            # new_mat[:c,:c]=comp[:c,:c] made scipy .toarray() an (N,N) slice ->
            # 48.8 GiB OOM at 80,916 entities (crashed Syl's bootstrap). .copy() keeps
            # the source untouched -- proven element-identical (grow/shrink/same, maxD=0)
            # to the old new-matrix path. Josh+Syl authorized.
            # [#136 2026-08-10] Now grows the RESIDENT CSR directly (no standing LIL).
            # csr.resize() is likewise native + non-densifying (O(nnz) peak), so the
            # #81 trap does not recur on this path either.
            new_mat = comp.copy()
            new_mat.resize((new_count, new_count))
            new_components.append(new_mat)
        self._components_csr = new_components
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

        # CSR-resident (#136): both the watermark translation scan and the
        # keep-slice below read the resident CSR. Flush any live buffer first
        # (no-op in steady state).
        self._ensure_csr_current()

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
        self._components_csr = [
            comp[keep][:, keep]
            for comp in self._components_csr
        ]
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
        for comp in self._components_csr:
            coo = comp.tocoo()
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
        """Return the full distance vector for a pair.

        CSR-resident (#136): reads straight from the resident CSR;
        _ensure_csr_current() flushes a live write buffer first (no-op in
        steady state).
        """
        self._ensure_csr_current()
        vec = np.zeros(NUM_DIST_COMPONENTS, dtype=np.float64)
        for c in range(NUM_DIST_COMPONENTS):
            vec[c] = self._components_csr[c][i, j]
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
