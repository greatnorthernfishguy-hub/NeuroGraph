# ---- Changelog ----
# [2026-07-28] Claude Code (Opus 5) — #83 save-guard: structural gate + EMA node reference
# What: SaveGate.permit() now consults evaluate_save_health() instead of a raw
#   live/on-disk node ratio. New: guard-state sidecar (<checkpoint>.guard_state.json)
#   holding an EMA of permitted node counts; gate order = absolute floor →
#   synapse/hyperedge structural gate → node gate referenced to the EMA →
#   legacy node-only fallback when no synapse counts are available. permit()
#   takes optional live_synapses/live_hyperedges; omitting them keeps the old
#   (stricter) behavior exactly.
# Why: the node-only ratio could not tell the #59 tonic melt from a real
#   collapse, so it refused legitimate saves. On the CC an orphan sweep drops
#   ~61% of nodes in ONE plasticity step at boot — a rate-of-change gate alone
#   would refuse the very event it exists to permit. The discriminator is
#   structural: sweeping isolated nodes cannot remove a synapse (degree 0 by
#   definition), so synapses ~unchanged PROVES the connected core survived and
#   the lost nodes were isolates. A true collapse takes nodes AND synapses AND
#   hyperedges together. Supersedes the rejected cc_gateway class-exemption
#   (exempting a real memory class blinds the guard to a genuine wipe).
# How: pure decision function (no I/O) so it is directly testable; EMA advances
#   only on PERMITTED saves, so a stuck-collapsed process can never walk the
#   reference down to meet itself; every permitted-but-abnormal melt logs
#   CRITICAL "MELT PERMITTED" with the counts — nothing passes silently
#   (LAW 3). All knobs env-tunable (LAW 5). One implementation at the shared
#   save path (LAW 4).
# [2026-07-09] Claude Code (Fable 5 design / Haiku implementation) — #373 checkpoint guardian
# What: SaveGate (refuse-to-clobber + provisional mode), atomic tmp+os.replace writes,
#   manifest sidecars, quarantine + generation ring (this task), hardlinked generation ring with GFS
#   retention. Consumed by openclaw_hook.NeuroGraphMemory.save()/__init__ (guarded
#   import — absence = today's behavior exactly).
# Why: #373 — the empty-writer clobber destroyed real state three times (VPS
#   2026-06-14, laptop 2026-06-26, laptop 2026-07-08 ~1800→4-6 nodes); both
#   checkpoint writers are non-atomic in-place writes (torn by mid-write power
#   death); recoveries kept mixing generations and losing offline passes for lack
#   of manifests. One implementation at the shared save path (LAW 4).
# How: stdlib-only file mechanics; env-tunable knobs (LAW 5); every refusal is
#   logger.error with the consequence spelled out and the state quarantined —
#   nothing silently dropped (LAW 3); generation ring hardlinks .msgpack only —
#   in-place-written .json sidecars are copied (hardlinks would alias every
#   generation to the live inode).
# -------------------
"""Checkpoint resilience for NeuroGraph instances (#373).

The invariant this module enforces: a process whose in-RAM state is
dramatically poorer than the on-disk checkpoint must never overwrite that
checkpoint on the routine autosave path. Refused writes are preserved in a
quarantine path and screamed about; they are never silently dropped.
"""

import json
import logging
import os
import shutil
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("checkpoint_guardian")


def _env_flag(name: str, default: str) -> bool:
    return os.environ.get(name, default) not in ("0", "false", "False", "")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


# ---- manifests ----

def manifest_path_for(checkpoint_path) -> Path:
    return Path(str(checkpoint_path) + ".manifest.json")


def write_manifest(checkpoint_path, stats: Dict[str, Any]) -> Path:
    """Atomic manifest sidecar describing what is on disk at checkpoint_path."""
    p = manifest_path_for(checkpoint_path)
    data = {
        "version": 1,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    data.update(stats)
    tmp = p.with_name(p.name + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, p)
    return p


def read_manifest(checkpoint_path) -> Optional[Dict[str, Any]]:
    p = manifest_path_for(checkpoint_path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception as exc:
        logger.warning("Guardian: unreadable manifest %s (%s)", p, exc)
        return None


# ---- save-health state (EMA / drawdown reference) ----
#
# The static "last healthy raw node count" reference cannot tell a by-design
# tonic melt (#59 CC_NG_TONIC_AGES=1: the orphan sweep sheds hundreds of
# synapse-less forest nodes in ONE plasticity step) from a catastrophic
# collapse. Both look like "nodes fell off a cliff". Rate alone cannot
# separate them either -- the legitimate melt is instantaneous.
#
# What separates them is STRUCTURE. An orphan melt sheds isolated nodes and
# leaves the connected core untouched (nodes -61%, synapses ~0%). A collapse
# -- failed restore, empty writer, wipe bug -- takes the synapses and
# hyperedges with it. So the primary gate is the synapse retention ratio, and
# the node count is judged against an EMA that TRACKS a sustained melt instead
# of a frozen historical peak that a melt can never climb back to.

def guard_state_path_for(checkpoint_path) -> Path:
    return Path(str(checkpoint_path) + ".guard_state.json")


def read_guard_state(checkpoint_path) -> Dict[str, Any]:
    p = guard_state_path_for(checkpoint_path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.warning("Guardian: unreadable guard state %s (%s)", p, exc)
        return {}


def write_guard_state(checkpoint_path, state: Dict[str, Any]) -> None:
    """Best-effort atomic write. Never raises -- a missing guard state degrades
    to the manifest reference, it must never block a save."""
    p = guard_state_path_for(checkpoint_path)
    try:
        tmp = p.with_name(p.name + f".tmp-{os.getpid()}")
        tmp.write_text(json.dumps(state, indent=2))
        os.replace(tmp, p)
    except Exception as exc:
        logger.debug("Guardian: guard-state write failed (non-fatal): %s", exc)


def update_node_ema(prev_ema: Optional[float], live_nodes: int,
                    alpha: Optional[float] = None) -> float:
    if alpha is None:
        alpha = _env_float("NG_GUARDIAN_EMA_ALPHA", 0.15)
    alpha = min(max(alpha, 0.0), 1.0)
    if prev_ema is None or prev_ema <= 0:
        return float(live_nodes)
    return (1.0 - alpha) * float(prev_ema) + alpha * float(live_nodes)


def evaluate_save_health(live_nodes: int,
                         ref_nodes: Optional[int],
                         live_synapses: Optional[int] = None,
                         ref_synapses: Optional[int] = None,
                         live_hyperedges: Optional[int] = None,
                         ref_hyperedges: Optional[int] = None,
                         ema_nodes: Optional[float] = None) -> Tuple[bool, str]:
    """Pure decision function: may this in-RAM state overwrite the primary?

    Returns (permit, reason). Ordered cheapest-and-most-certain first. Every
    permitted-but-abnormal outcome returns a reason the caller logs loudly --
    a melt is allowed through, never silently (LAW 3).
    """
    floor = _env_int("NG_GUARDIAN_GATE_MIN_NODES", 100)
    if ref_nodes is None or ref_nodes < floor:
        return True, "no substantial on-disk reference"

    # 1. Near-empty next to a real mind on disk is ALWAYS the clobber shape,
    #    whatever the ratios say. This is the 06-14 / 06-26 / 07-08 signature.
    abs_floor = _env_int("NG_GUARDIAN_ABS_FLOOR_NODES", 25)
    if live_nodes < abs_floor:
        return False, (
            f"live graph ({live_nodes} nodes) is below the absolute floor "
            f"({abs_floor}) while {ref_nodes} nodes are on disk"
        )

    # 2. Structural gate -- the discriminator. Synapses surviving means the
    #    connected core survived and the lost nodes were isolates (melt).
    syn_ratio = _env_float("NG_GUARDIAN_GATE_SYNAPSE_RATIO", 0.5)
    min_ref_syn = _env_int("NG_GUARDIAN_MIN_REF_SYNAPSES", 100)
    have_syn = live_synapses is not None and ref_synapses is not None and ref_synapses >= min_ref_syn
    if have_syn and live_synapses < syn_ratio * ref_synapses:
        return False, (
            f"structural collapse: synapses {ref_synapses} -> {live_synapses} "
            f"(below {syn_ratio:.0%} of the on-disk reference), "
            f"nodes {ref_nodes} -> {live_nodes}"
        )

    # Hyperedges are a corroborating signal only, and a noisy one at small n:
    # the CC substrate carries ~12 of them, where losing 8 to ordinary churn is
    # not evidence of anything. Require a population big enough for the ratio to
    # mean something before this gate is allowed to refuse a save.
    he_ratio = _env_float("NG_GUARDIAN_GATE_HYPEREDGE_RATIO", 0.34)
    min_ref_he = _env_int("NG_GUARDIAN_MIN_REF_HYPEREDGES", 50)
    if (live_hyperedges is not None and ref_hyperedges is not None
            and ref_hyperedges >= min_ref_he
            and live_hyperedges < he_ratio * ref_hyperedges):
        return False, (
            f"structural collapse: hyperedges {ref_hyperedges} -> "
            f"{live_hyperedges} (below {he_ratio:.0%} of the on-disk reference)"
        )

    # 3. Node gate, referenced to the EMA (which follows a sustained melt down)
    #    rather than a frozen peak. Only consulted when structure is intact or
    #    unknown.
    node_ratio = _env_float("NG_GUARDIAN_GATE_RATIO", 0.5)
    reference = float(ema_nodes) if ema_nodes and ema_nodes >= floor else float(ref_nodes)
    ref_label = "EMA" if (ema_nodes and ema_nodes >= floor) else "on-disk"
    node_collapsed = live_nodes < node_ratio * reference

    if have_syn:
        # Structure is verified intact: a node drop is a melt. Permit, loudly.
        if node_collapsed or live_nodes < node_ratio * ref_nodes:
            retained = (live_synapses / ref_synapses) if ref_synapses else 1.0
            return True, (
                f"MELT PERMITTED: nodes {ref_nodes} -> {live_nodes} "
                f"({1 - live_nodes / max(ref_nodes, 1):.0%} shed) but structure intact "
                f"(synapses {retained:.0%} retained) -- isolated-node sweep, not a collapse"
            )
        return True, "ok"

    # 4. No synapse information (older manifest / caller): fall back to the
    #    legacy node-only ratio so behaviour is never weaker than before.
    if node_collapsed:
        return False, (
            f"live graph ({live_nodes} nodes) is below {node_ratio:.0%} of the "
            f"{ref_label} reference ({reference:.0f} nodes), and no synapse "
            f"counts were available to rule out a structural collapse"
        )
    return True, "ok"


def best_effort_git_hash(repo_dir) -> Optional[str]:
    """Short hash of repo_dir's HEAD, or None. Never raises, never blocks long."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_dir), capture_output=True, text=True, timeout=2,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


# ---- atomic writes ----

def atomic_file_write(final_path: str, write_fn: Callable[[str], Any]) -> Any:
    """Run write_fn against a tmp path, then os.replace onto final_path.

    The tmp name PRESERVES the final suffix ("main.tmp-1234.msgpack") because
    both writers dispatch on extension: graph.checkpoint() refuses
    non-.msgpack paths loudly (#325) and SimpleVectorDB.save() silently
    switches to JSON (#356 trap). On write_fn failure the final file is
    untouched and the tmp is removed.
    """
    final = Path(final_path)
    tmp = final.with_name(f"{final.stem}.tmp-{os.getpid()}{final.suffix}")
    try:
        result = write_fn(str(tmp))
        os.replace(tmp, final)
        return result
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


# ---- the gate ----

class SaveGate:
    """Refuses primary-checkpoint overwrites from processes whose in-RAM state
    is dramatically poorer than what is on disk, and quarantines instead.

    Provisional mode: a boot whose restore FAILED (or was skipped as
    mid-write-unstable) while a checkpoint file exists is running with an
    empty graph next to a real mind on disk — the exact shape that destroyed
    state on 2026-06-14, 2026-06-26 and 2026-07-08. Such a process never
    writes the primary until an operator clears it (clear_provisional() or
    env NG_GUARDIAN_CLEAR_PROVISIONAL=1 at boot).
    """

    def __init__(self, checkpoint_path):
        self._checkpoint_path = str(checkpoint_path)
        self.provisional = False
        self.provisional_reason: Optional[str] = None
        self._reference_nodes: Optional[int] = None

    def record_restore(self, outcome: str, restored_nodes: int) -> None:
        """outcome: 'ok' | 'failed' | 'skipped_unstable' | 'no_file'."""
        manifest = read_manifest(self._checkpoint_path)
        disk_nodes = (manifest.get("guardian_nodes") or manifest.get("nodes")) if manifest else None
        if outcome == "ok":
            self._reference_nodes = max(restored_nodes, disk_nodes or 0)
            return
        if outcome == "no_file" or not os.path.exists(self._checkpoint_path):
            return  # fresh install — nothing on disk to protect
        if _env_flag("NG_GUARDIAN_CLEAR_PROVISIONAL", "0"):
            logger.error(
                "Guardian: restore outcome %r with existing checkpoint %s, but "
                "NG_GUARDIAN_CLEAR_PROVISIONAL is set — provisional mode "
                "OVERRIDDEN by operator.", outcome, self._checkpoint_path,
            )
            return
        self.provisional = True
        self.provisional_reason = (
            f"boot restore outcome={outcome} while {self._checkpoint_path} "
            f"exists (manifest nodes={disk_nodes})"
        )
        logger.error(
            "Guardian: PROVISIONAL MODE — %s. This process is running with an "
            "empty/partial graph next to a real checkpoint; primary writes "
            "will be QUARANTINED until an operator investigates "
            "(clear_provisional() or NG_GUARDIAN_CLEAR_PROVISIONAL=1).",
            self.provisional_reason,
        )

    def clear_provisional(self) -> None:
        if self.provisional:
            logger.error("Guardian: provisional mode CLEARED by operator for %s",
                         self._checkpoint_path)
        self.provisional = False
        self.provisional_reason = None

    def permit(self, live_nodes: int,
               live_synapses: Optional[int] = None,
               live_hyperedges: Optional[int] = None) -> Tuple[bool, str]:
        """May this in-RAM state overwrite the primary checkpoint?

        Pass live_synapses/live_hyperedges when available: they are what lets
        the gate tell a by-design isolated-node melt from a real collapse.
        Omitting them is safe but falls back to the stricter node-only ratio.

        Side effect by design: a PERMITTED save advances the node EMA, because
        permit() is called exactly once per save attempt and only a permitted
        attempt changes what is on disk. A refusal leaves the reference alone,
        so a stuck-collapsed process can never walk the reference down to meet
        itself.
        """
        if not _env_flag("NG_GUARDIAN_ENABLED", "1"):
            return True, "guardian disabled"
        if self.provisional:
            return False, f"provisional: {self.provisional_reason}"
        manifest = read_manifest(self._checkpoint_path)
        reference = (manifest.get("guardian_nodes") or manifest.get("nodes")) if manifest else self._reference_nodes
        ref_synapses = manifest.get("synapses") if manifest else None
        ref_hyperedges = manifest.get("hyperedges") if manifest else None

        state = read_guard_state(self._checkpoint_path)
        ema = state.get("ema_nodes")

        ok, reason = evaluate_save_health(
            live_nodes=live_nodes,
            ref_nodes=reference,
            live_synapses=live_synapses,
            ref_synapses=ref_synapses,
            live_hyperedges=live_hyperedges,
            ref_hyperedges=ref_hyperedges,
            ema_nodes=ema,
        )

        if ok:
            if reason.startswith("MELT PERMITTED"):
                logger.critical("Guardian: %s", reason)
            new_ema = update_node_ema(ema, live_nodes)
            write_guard_state(self._checkpoint_path, {
                "ema_nodes": round(new_ema, 3),
                "last_nodes": live_nodes,
                "last_synapses": live_synapses,
                "last_hyperedges": live_hyperedges,
                "last_permit_ts": time.time(),
                "updated": datetime.now(timezone.utc).isoformat(),
            })
        return ok, reason



# ---- quarantine ----

def quarantine_save(checkpoint_dir, name_prefix: str,
                    write_fn: Callable[[str], Any],
                    suffix: str = ".msgpack",
                    keep: Optional[int] = None) -> str:
    """Atomically write refused state into <dir>/quarantine/ and prune old ones.

    A provisional daemon autosaves every few minutes — without pruning this
    fills the disk (Syl's set is ~600MB). Newest `keep` survive.
    """
    qdir = Path(checkpoint_dir) / "quarantine"
    qdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # Find the maximum collision number among existing files with the same timestamp
    max_n = -1
    for f in qdir.glob(f"{name_prefix}.{stamp}*{suffix}"):
        name = f.name
        base = name.rsplit(".", 1)[0] if "." in name else name
        if "-" in base:
            collision_str = base.split("-")[-1]
            try:
                n = int(collision_str)
                max_n = max(max_n, n)
            except ValueError:
                pass
        else:
            max_n = max(max_n, 0)

    # Create target with next collision number
    n = max_n + 1
    if n == 0:
        target = qdir / f"{name_prefix}.{stamp}{suffix}"
    else:
        target = qdir / f"{name_prefix}.{stamp}-{n}{suffix}"

    atomic_file_write(str(target), write_fn)
    if keep is None:
        keep = _env_int("NG_GUARDIAN_QUARANTINE_KEEP", 3)

    def quarantine_sort_key(p):
        """Sort by mtime, with collision suffix as tiebreaker for same-second files."""
        mtime = p.stat().st_mtime
        # Extract collision number as tiebreaker
        name = p.name
        base = name.rsplit(".", 1)[0] if "." in name else name
        if "-" in base:
            collision_str = base.split("-")[-1]
            try:
                collision = int(collision_str)
            except ValueError:
                collision = float('inf')
        else:
            collision = 0
        return (mtime, collision)

    entries = sorted(qdir.glob(f"{name_prefix}.*{suffix}"),
                     key=quarantine_sort_key)
    for old in entries[:-keep] if keep > 0 else []:
        try:
            old.unlink()
        except OSError as exc:
            logger.warning("Guardian: quarantine prune failed for %s (%s)", old, exc)
    return str(target)


# ---- generation ring ----

def _parse_stamp(name: str) -> Optional[datetime]:
    base = name.split("-")[0] if "-" in name and name.count("-") == 1 else name
    try:
        return datetime.strptime(base, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def rotate_generations(checkpoint_dir, files: List[str],
                       recent: Optional[int] = None,
                       hourly: Optional[int] = None,
                       daily: Optional[int] = None,
                       now: Optional[datetime] = None) -> Optional[str]:
    """Hardlink the consistent SET of checkpoint files into generations/<stamp>/.

    Hardlinks are frozen snapshots at zero copy cost: the atomic-save path
    replaces the primary's inode (os.replace), so the generation keeps the
    old bytes. Falls back to copy2 where hardlinks aren't possible (EXDEV).
    Generations are SETS — main + vectors + sidecars together, never mixed
    across saves (mixed-generation restores breed vdb orphans).
    """
    gen_root = Path(checkpoint_dir) / "generations"
    gen_root.mkdir(parents=True, exist_ok=True)
    if now is None:
        now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    gen_dir = gen_root / stamp
    n = 0
    while gen_dir.exists():
        n += 1
        gen_dir = gen_root / f"{stamp}-{n}"
    gen_dir.mkdir()
    for f in files:
        src = Path(f)
        if not src.exists():
            continue
        dst = gen_dir / src.name
        # Hardlink only .msgpack members: their writers replace the inode
        # atomically (tmp + os.replace), so a linked generation is frozen.
        # Sidecar .json files are written IN PLACE by their (protected)
        # writers — a hardlink would alias every generation to the live
        # file's inode and silently rewrite them all on each save. They are
        # KB-scale; copy them for real.
        try:
            if src.suffix == ".msgpack":
                os.link(src, dst)
            else:
                shutil.copy2(src, dst)
        except OSError:
            try:
                shutil.copy2(src, dst)
            except OSError as exc:
                logger.warning("Guardian: generation copy failed for %s (%s)", src, exc)
    _prune_generations(gen_root, gen_dir.name,
                       recent if recent is not None else _env_int("NG_GUARDIAN_GEN_RECENT", 3),
                       hourly if hourly is not None else _env_int("NG_GUARDIAN_GEN_HOURLY", 6),
                       daily if daily is not None else _env_int("NG_GUARDIAN_GEN_DAILY", 7),
                       now)
    return str(gen_dir)


def _prune_generations(gen_root: Path, new_gen_name: str, recent: int, hourly: int, daily: int,
                       now: datetime) -> None:
    """GFS retention: newest `recent` unconditionally; then one per hour for
    `hourly` hours; then one per day for `daily` days. Dirs whose names don't
    parse as stamps are NEVER deleted (manual backups are sacred).
    The newly created generation (new_gen_name) is always kept and doesn't count
    toward the recent limit — it's preserved for backup consistency."""
    dirs = sorted([d for d in gen_root.iterdir() if d.is_dir()],
                  key=lambda d: d.name, reverse=True)  # newest first
    # Keep the newest `recent` directories, EXCLUDING the newly created one
    other_dirs = [d for d in dirs if d.name != new_gen_name]
    keep = {d.name for d in other_dirs[:recent]}
    keep.add(new_gen_name)  # Always keep the newly created generation
    hourly_seen: set = set()
    daily_seen: set = set()
    for d in dirs:
        ts = _parse_stamp(d.name)
        if ts is None:
            keep.add(d.name)
            continue
        age = now - ts
        if age <= timedelta(hours=hourly):
            hour_key = ts.strftime("%Y%m%d%H")
            if hour_key not in hourly_seen:
                hourly_seen.add(hour_key)
                keep.add(d.name)
        elif age <= timedelta(days=daily):
            day_key = ts.strftime("%Y%m%d")
            if day_key not in daily_seen:
                daily_seen.add(day_key)
                keep.add(d.name)
    for d in dirs:
        if d.name not in keep:
            shutil.rmtree(d, ignore_errors=True)
