#!/usr/bin/env python3
# ---- Changelog ----
# [2026-09-30] Claude Code (claude-sonnet-5-5, Z12 BUILD worker seat, dispatch #11228, TURN A) — the
#   118-want TEXT repair ONE-SHOT TOOL, built exactly to plan-004 [R4b] (NG branch
#   cc-laptop-want-text-repair-20260930, HEAD 5b539216756bb0f9d731bef929ac0a104ff81912).
# What: classify (plan 4.2) -> outcome table / per-reason histograms / marker-bearing list / residuals /
#   repair-list / scope-ids / PRE-node line / T6 would-mint set (all STAMPED with the pin tuple, P5 refuses a
#   mismatching stamp); the value-granular rewrite writer (TMP file only in Phase 1); the verifier V1-V19; the
#   frozen-list approvals gate (an unapproved id is REFUSED); the mapping + INVERSE mapping; gates P1-P9
#   (P10 is S4's, the tool emits its input); --apply, --josh-go and the RETIRED-receipt mechanism EXIST but
#   --apply is REFUSED unless every Phase-2 gate holds (it cannot: there is no Josh go in this build).
# Why: Exec P402/P406/P416/P423 + Chief-003 via the plan. Josh P382: DELETE NOTHING; H-1: no want text/id/flag/
#   synapse is altered except by the approved repair under row #801 - and Phase 1 (this build) only READS.
# How: ONE file (its own sha256 is the tool's identity for the retirement refusal, plan 6.8). It imports
#   `parse_wants` / `want_id_for_text` (the ONE shared pure function, LAW 3/4) from the PIN worktree ONLY, fail-
#   closed (plan 6.2); it never re-implements either. The checkpoint is read with the canonical readers
#   (Graph.restore + SimpleVectorDB.load = the analysis-001 loader, analyze_pair.py:84-86; nothing forked) and
#   written value-granularly (untouched values are RAW byte slices; only values that hold an old id are
#   re-encoded, each proven by pack(unpack(raw)) == raw, V13). The canonical Graph.restore of the output is
#   the verifier (V11). Nothing protected or vendored is edited; no checkpoint under Syl's directories, no
#   tract, no ~/.bashrc, no primary checkout is ever opened for write (hard refusals, plan 6.3).
# -------------------
"""ONE-SHOT NOTICE (2026-09-30) - this tool is written for ONE repair (row #801, the 118 wants) and is NOT a
reusable re-key path. RETIRED: (not yet - a RETIRED stamp with the date, the mapping sha256 and "no reuse; a
future re-key is protected-file work and goes to Josh then" is added after the apply, plan 6.8). It lives ONLY
in handoffs/z12-want-text-repair/oneshot-tool/ on its own feature branch, is never merged, and the directory
is removed by a commit before any merge of the handoffs to main.

NO RAW WANT TEXT or conversation excerpt is written to any pushed file, commit message or reply. Reports are
ids / lengths / hashes / counts only, written OFF-REPO under /home/josh/backups/z12-want-text-repair-<UTC>/;
excerpts exist only in that directory's review/ sub-directory (mode 0600), scrubbed.

Run (this host exports a PYTHONPATH that binds the primary checkout - always start clean):
    env -u PYTHONPATH -u NG_EMBED_REMOTE PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 python3 -B \\
        want_text_repair_oneshot.py --pin-root <PIN worktree> --target-dir <CC checkpoint dir> \\
        --daemon-script <daemon script> --scope-min-len 600 --step classify
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import datetime as _dt
import gc
import glob
import hashlib
import importlib
import json
import mmap
import os
import re
import shutil
import subprocess
import sys
import types
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

TOOL_NAME = "want_text_repair_oneshot"
TOOL_VERSION = "turnA-1"
SCRUB_VERSION = "scrub-1"

# --------------------------------------------------------------------------------------------------
# The FROZEN pin (plan-004 section 1). Any change here re-opens the recount and voids every approval.
# --------------------------------------------------------------------------------------------------
PIN: Dict[str, str] = {
    "code_commit": "ae798b94cb14740d200fc3f4fd8d36eef8b86c6a",
    "branch_head": "c7921b8436fb174c3f70fcf02827f16bb16deff0",
    "cc_ng_organism_sha256": "8ad0f69ed4a7849c98e97e0f3970bbe4a36c727ef880455e8ddecec14de5a8e2",
    "cc_ng_organism_blob": "a3aa8a0ddb6a89fe468a9a20beadc5e381624cab",
    "test_file_sha256": "04b1a494e4a5762b1e3f97d07e12c0937a50bf5baac09e5208b2bd9f82a46c53",
}
BASE_COMMIT = "e4ebf982b1989fd9066d610b94853bc68bf70d37"
TEST_FILE_RELPATH = "tests/test_cc_want_legitimacy_810.py"

# The 11 reasons of the FINAL function (plan section 2); checked equal to the loaded module's tuple.
REASONS: Tuple[str, ...] = (
    "in_fence", "in_code_span", "code_adjacent", "escaped", "quoted",
    "in_json_string", "in_url", "in_link_target",
    "opener_unclosed", "closer_without_opener", "empty_pair",
)
REGION_FORCE_REVIEW = ("in_url", "in_json_string", "in_link_target")

# Identity protected by name (plan V15). NOT in S, NOT in the mapping, NOT in any approval.
CHOICE_CLAUSE_IDS: Tuple[str, str] = ("cc:want::7bd0f5fdca6eb404", "cc:want::3eecfa18710e3b6b")
CONSTITUTIONAL_ID = "constitutional::rim::choice_clause"

# The ripple table this tool carries into every backup manifest (Exec P428 / le-029 C6): what else holds bytes of
# the pre-repair checkpoint, and what may NEVER be used to undo the repair.
RIPPLE_TABLE: Dict[str, str] = {
    "generations/": ("incidental, expiring (the daemon's rotation prunes it), never a rollback source; the live main.msgpack / "
                     "vectors.msgpack may be hard-linked into it (Exec P428, judged by inode) and keep the PRE-repair bytes there; "
                     "never listed or opened by this tool beyond stat/hash of the recorded partner paths"),
    "last_good/": "NOT a link partner (Exec P428); never a rollback source; never touched by this tool",
    "rollback source": "ONLY the tool's own named pre-apply backup (<run>/backup/ + backup-manifest-<UTC>.json), every sha256 verified",
}

# The checkpoint set (plan 6.3; real names per checkpoint_guardian.manifest_path_for/guard_state_path_for).
MAIN_NAME = "main.msgpack"
VECTORS_NAME = "vectors.msgpack"
SIDECAR_NAME = "main.msgpack.activations.json"
GUARD_NAME = "main.msgpack.guard_state.json"
MANIFEST_NAME = "main.msgpack.manifest.json"
COMMONS_NAME = "commons.msgpack"
SIX_FILES: Tuple[str, ...] = (MAIN_NAME, VECTORS_NAME, SIDECAR_NAME, GUARD_NAME, MANIFEST_NAME, COMMONS_NAME)

EXPECTED_WANTS = 182
EXPECTED_PROTECTED = 183
EXPECTED_SCOPE = 118

# Recorded locations of a one-shot (plan 6.3). These are RECORDED constants of a single repair, not
# configuration; the target is a required argument with no default and is cross-checked against them.
_HOME = os.path.expanduser("~")
SYL_CHECKPOINTS = os.path.realpath(os.path.join(_HOME, "NeuroGraph", "data", "checkpoints"))
PRIMARY_NEUROGRAPH = os.path.realpath(os.path.join(_HOME, "NeuroGraph"))
RECORDED_CC_CHECKPOINT_DIR = os.path.realpath(os.path.join(_HOME, ".claude", "plugins", "neurograph", "checkpoints"))
BACKUPS_ROOT = os.path.realpath(os.path.join(_HOME, "backups"))
RUN_DIR_PREFIX = "z12-want-text-repair-"

# The NG modules this tool loads. ALL must resolve inside the PIN worktree.
NG_MODULE_NAMES: Tuple[str, ...] = (
    "cc_ng_organism", "neuro_foundation", "universal_ingestor", "checkpoint_guardian",
    "neurograph_rpc", "ng_lite", "ng_embed", "ng_ecosystem", "ng_tract_bridge", "ng_autonomic",
    "openclaw_adapter", "surface_resolver", "surfacing", "cc_ng_host", "activation_persistence",
)


class Refusal(Exception):
    """A guard or gate refused (exit 2). Nothing was written."""


class Stop(Exception):
    """A verifier / census / safety check failed (exit 3). No live write."""


# --------------------------------------------------------------------------------------------------
# small utilities
# --------------------------------------------------------------------------------------------------

def utc_stamp() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git_blob_id(data: bytes) -> str:
    """The git blob id of `data` (computed, not shelled out: it is the check, not the answer)."""
    return hashlib.new("sha1", b"blob %d\0" % len(data) + data).hexdigest()


def sha16(text: str) -> str:
    """want_text_sha16 exactly as plan-scratch/protected_census.py:53-56 defines it."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _git(root: str, *args: str) -> str:
    p = subprocess.run(["git", "-C", root, *args], capture_output=True, text=True)
    if p.returncode != 0:
        raise Refusal("git %s failed in %s: %s" % (" ".join(args), root, p.stderr.strip()[:200]))
    return p.stdout.strip()


def _under(path: str, parent: str) -> bool:
    path, parent = os.path.realpath(path), os.path.realpath(parent)
    return path == parent or path.startswith(parent.rstrip(os.sep) + os.sep)


def canonical_json(obj: Any) -> bytes:
    return (json.dumps(obj, sort_keys=True, indent=1, ensure_ascii=True) + "\n").encode("utf-8")


# --------------------------------------------------------------------------------------------------
# P1 / plan 6.2: the pinned function, import isolation, fail-closed
# --------------------------------------------------------------------------------------------------

def pin_stamp() -> Dict[str, str]:
    """The stamp every count/report artifact carries (plan 1 / 4.4): file sha256, blob, branch head,
    test-file sha256, scrub version. The branch head is the FROZEN value; the P1 record carries the actual
    tree HEAD (the pin worktree is the code commit, whose docs-only descendant is the frozen head)."""
    return {
        "cc_ng_organism_sha256": PIN["cc_ng_organism_sha256"],
        "cc_ng_organism_blob": PIN["cc_ng_organism_blob"],
        "branch_head": PIN["branch_head"],
        "test_file_sha256": PIN["test_file_sha256"],
        "scrub_version": SCRUB_VERSION,
    }


def is_primary_checkout_path(path: str) -> bool:
    """True if `path` lies inside a repository whose git-dir is its own common dir (i.e. NOT a linked
    worktree), or inside ~/NeuroGraph outside any linked worktree. A path in no repository is False."""
    path = os.path.realpath(path)
    probe = path if os.path.isdir(path) else os.path.dirname(path)
    top = subprocess.run(["git", "-C", probe, "rev-parse", "--show-toplevel"], capture_output=True, text=True)
    if top.returncode != 0:
        return False
    gd = subprocess.run(["git", "-C", probe, "rev-parse", "--absolute-git-dir"], capture_output=True, text=True)
    cd = subprocess.run(["git", "-C", probe, "rev-parse", "--path-format=absolute", "--git-common-dir"],
                        capture_output=True, text=True)
    if gd.returncode != 0 or cd.returncode != 0:
        return True  # cannot prove it is a linked worktree: treat as primary (fail closed)
    return os.path.realpath(gd.stdout.strip()) == os.path.realpath(cd.stdout.strip())


def load_pinned(pin_root: str) -> types.SimpleNamespace:
    """P1 + plan 6.2. Verify the pinned files' identity FROM DISK before importing anything, put the pin tree
    at sys.path[0], import, then re-verify what was actually loaded. Any doubt is a Refusal."""
    root = os.path.realpath(pin_root)
    if not os.path.isdir(root):
        raise Refusal("P1: pin root %r is not a directory" % pin_root)
    org_file = os.path.join(root, "cc_ng_organism.py")
    test_file = os.path.join(root, *TEST_FILE_RELPATH.split("/"))
    for p in (org_file, test_file):
        if not os.path.isfile(p):
            raise Refusal("P1: %s missing in the pin tree" % os.path.basename(p))
    if is_primary_checkout_path(root):
        raise Refusal("P1: the pin root is a primary checkout, not a linked worktree")
    if _under(root, SYL_CHECKPOINTS):
        raise Refusal("P1: the pin root is under Syl's checkpoints")
    data = Path(org_file).read_bytes()
    got = {
        "cc_ng_organism_sha256": sha256_bytes(data),
        "cc_ng_organism_blob": git_blob_id(data),
        "test_file_sha256": sha256_file(test_file),
    }
    for k, v in got.items():
        if v != PIN[k]:
            raise Refusal("P1: %s is %s, the frozen pin is %s" % (k, v, PIN[k]))
    head = _git(root, "rev-parse", "HEAD")
    head_blob = _git(root, "rev-parse", "HEAD:cc_ng_organism.py")
    if head_blob != PIN["cc_ng_organism_blob"]:
        raise Refusal("P1: the blob of cc_ng_organism.py at the tree HEAD is %s, not the pin" % head_blob)
    anc = subprocess.run(["git", "-C", root, "merge-base", "--is-ancestor", PIN["code_commit"], head])
    if anc.returncode != 0:
        raise Refusal("P1: tree HEAD %s does not descend from the code commit %s" % (head, PIN["code_commit"]))

    # import isolation (plan 6.2): nothing NG preloaded from elsewhere; pin tree first on sys.path
    for name in NG_MODULE_NAMES:
        mod = sys.modules.get(name)
        f = getattr(mod, "__file__", None) if mod is not None else None
        if f and not _under(f, root):
            raise Refusal("P1/6.2: %s is already imported from %s, outside the pin tree" % (name, f))
    sys.path[:] = [root] + [p for p in sys.path if os.path.realpath(p or os.getcwd()) != root]
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    org = importlib.import_module("cc_ng_organism")
    nf = importlib.import_module("neuro_foundation")
    ui = importlib.import_module("universal_ingestor")
    cg = importlib.import_module("checkpoint_guardian")

    org_path = os.path.realpath(org.__file__)
    if org_path == os.path.join(PRIMARY_NEUROGRAPH, "cc_ng_organism.py"):
        raise Refusal("6.2: cc_ng_organism resolved to the PRIMARY checkout")
    if org_path != os.path.join(root, "cc_ng_organism.py"):
        raise Refusal("6.2: cc_ng_organism resolved to %s, not the pin tree" % org_path)
    if sha256_file(org_path) != PIN["cc_ng_organism_sha256"]:
        raise Refusal("6.2: the LOADED cc_ng_organism.py sha256 differs from the pin")
    ng_loaded: Dict[str, str] = {}
    for name, mod in sorted(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if not f:
            continue
        rf = os.path.realpath(f)
        if name in NG_MODULE_NAMES or _under(rf, PRIMARY_NEUROGRAPH) and not _under(rf, root):
            ng_loaded[name] = rf
            if not _under(rf, root):
                raise Refusal("6.2: NG module %s resolved outside the pin tree: %s" % (name, rf))
            tracked = _git(root, "rev-parse", "HEAD:" + os.path.relpath(rf, root))
            if git_blob_id(Path(rf).read_bytes()) != tracked:
                raise Refusal("6.2: NG module %s differs from the tree HEAD (dirty pin tree)" % name)
    if tuple(org.WANT_SKIP_REASONS) != REASONS or org.WANT_OPEN != "[WANT]" or org.WANT_CLOSE != "[/WANT]":
        raise Refusal("P1: WANT constants / reasons of the loaded module are not the frozen ones")
    for fn in ("parse_wants", "want_id_for_text", "surface_wants"):
        if not callable(getattr(org, fn, None)):
            raise Refusal("P1: loaded module has no %s" % fn)
    record = {
        "p1": {**got, "tree_head": head, "tree_head_blob": head_blob, "code_commit": PIN["code_commit"],
               "frozen_branch_head": PIN["branch_head"]},
        "isolation": {
            "sys_executable": sys.executable,
            "sys_path_head": [p for p in sys.path[:6]],
            "cc_ng_organism_file": org_path,
            "cc_ng_organism_sha256": sha256_file(org_path),
            "ng_modules_loaded": ng_loaded,
            "pythonpath_env": os.environ.get("PYTHONPATH"),
            "ng_embed_env_names": sorted(k for k in os.environ if k.startswith("NG_EMBED")),
            "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
        },
    }
    return types.SimpleNamespace(root=root, org=org, nf=nf, ui=ui, cg=cg, record=record)


def p379_lines(pinned: types.SimpleNamespace) -> List[str]:
    iso = pinned.record["isolation"]
    p1 = pinned.record["p1"]
    lines = ["P1 pin/stack head (frozen) %s ; actual pin-worktree HEAD %s ; both recorded" % (p1["frozen_branch_head"], p1["tree_head"]),
             "P1 cc_ng_organism.py sha256 asserted equal to the pin: %s" % p1["cc_ng_organism_sha256"],
             "P379 sys.executable %s" % iso["sys_executable"],
             "P379 sys.path[0:6] %s" % iso["sys_path_head"],
             "P379 cc_ng_organism.__file__ %s" % iso["cc_ng_organism_file"],
             "P379 cc_ng_organism sha256 %s" % iso["cc_ng_organism_sha256"],
             "P379 PYTHONPATH %s ; NG_EMBED_* names %s" % (iso["pythonpath_env"], iso["ng_embed_env_names"] or "none")]
    for n, f in sorted(iso["ng_modules_loaded"].items()):
        lines.append("P379 %-22s %s" % (n, f))
    return lines


def load_base_module(pinned: types.SimpleNamespace):
    """BASE e4ebf982's cc_ng_organism.py under a private name, loaded exactly as #810's golden tests load it
    (`git show`, exec under a private module name). A missing base is a Refusal, never a skip."""
    p = subprocess.run(["git", "-C", pinned.root, "show", "%s:cc_ng_organism.py" % BASE_COMMIT], capture_output=True)
    if p.returncode != 0:
        raise Refusal("cannot read base %s via git show: %s" % (BASE_COMMIT, p.stderr.decode(errors="replace")[:200]))
    name = "cc_ng_organism_base_e4ebf982"
    mod = types.ModuleType(name)
    mod.__file__ = os.path.join(pinned.root, "cc_ng_organism.py")
    sys.modules[name] = mod
    try:
        exec(compile(p.stdout.decode("utf-8"), "<base e4ebf982 cc_ng_organism.py>", "exec"), mod.__dict__)
    finally:
        sys.modules.pop(name, None)
    return mod


# --------------------------------------------------------------------------------------------------
# P7 target guard (plan 6.3) and the write guard (Phase 1 writes ONLY under the backups run dir)
# --------------------------------------------------------------------------------------------------

def daemon_checkpoint_dir(daemon_script: str) -> str:
    """The CHECKPOINT_DIR the daemon script assigns, read from its text (plan 6.3 cross-check). The script
    is never imported or run. Anything not of the recorded two-line shape is a Refusal (fail closed)."""
    try:
        text = Path(daemon_script).read_text(encoding="utf-8")
    except OSError as exc:
        raise Refusal("P7: cannot read the daemon script %r: %s" % (daemon_script, exc))
    ws = re.search(r"^CC_NG_WORKSPACE\s*=\s*os\.path\.expanduser\((['\"])(.+?)\1\)\s*$", text, re.M)
    cd = re.search(r"^CHECKPOINT_DIR\s*=\s*os\.path\.join\(CC_NG_WORKSPACE,\s*(['\"])(.+?)\1\)\s*$", text, re.M)
    if not ws or not cd:
        raise Refusal("P7: the daemon script does not assign CC_NG_WORKSPACE/CHECKPOINT_DIR in the recorded shape")
    return os.path.realpath(os.path.join(os.path.expanduser(ws.group(2)), cd.group(2)))


def guard_target(target_dir: str, daemon_script: str) -> Dict[str, Any]:
    """P7. HARD REFUSALS, no override: a realpath under Syl's checkpoints, under any primary checkout, not equal
    to the recorded CC checkpoint directory, or disagreeing with the daemon script's CHECKPOINT_DIR."""
    if not target_dir:
        raise Refusal("P7: --target-dir is REQUIRED (no default)")
    real = os.path.realpath(target_dir)
    if not os.path.isdir(real):
        raise Refusal("P7: target %r is not a directory" % target_dir)
    if _under(real, SYL_CHECKPOINTS):
        raise Refusal("P7: target is under Syl's checkpoints (%s)" % SYL_CHECKPOINTS)
    if is_primary_checkout_path(real):
        raise Refusal("P7: target is inside a primary checkout (not a linked worktree)")
    if real != os.path.realpath(RECORDED_CC_CHECKPOINT_DIR):
        raise Refusal("P7: target is not the recorded CC laptop checkpoint directory")
    dd = daemon_checkpoint_dir(daemon_script)
    if dd != real:
        raise Refusal("P7: target disagrees with the daemon script's CHECKPOINT_DIR")
    return {"target_arg": target_dir, "target_realpath": real, "recorded_cc_dir": RECORDED_CC_CHECKPOINT_DIR,
            "daemon_script_checkpoint_dir": dd, "daemon_script": os.path.realpath(daemon_script)}


def guard_out_path(path: str) -> str:
    """Phase 1 may write ONLY under <backups>/z12-want-text-repair-*/ . Returns the realpath or Refuses."""
    real = os.path.realpath(path)
    root = os.path.realpath(BACKUPS_ROOT)
    if not real.startswith(root + os.sep):
        raise Refusal("write refused: %s is not under %s" % (real, root))
    first = real[len(root) + 1:].split(os.sep, 1)[0]
    if not first.startswith(RUN_DIR_PREFIX):
        raise Refusal("write refused: %s is not under a %s* run directory" % (real, RUN_DIR_PREFIX))
    if _under(real, SYL_CHECKPOINTS) or _under(real, RECORDED_CC_CHECKPOINT_DIR):
        raise Refusal("write refused: %s is inside a checkpoint directory" % real)
    return real


def refuse_inplace_write(path: str) -> None:
    """A destination that already exists with st_nlink > 1 is never opened for write (Exec P428 / le-029 C6):
    writing through it would change EVERY name that shares the inode (e.g. a generation copy). The only writer to a
    live file is the atomic tmp + os.replace path, which gives the live name a NEW inode."""
    if os.path.lexists(path):
        n = os.stat(path).st_nlink
        if n > 1:
            raise Refusal("write refused: %s has %d hard links - never written in place (a write would change every name "
                          "that shares the inode)" % (path, n))


def out_write_bytes(path: str, data: bytes, mode: int = 0o644) -> str:
    real = guard_out_path(path)
    refuse_inplace_write(real)
    os.makedirs(os.path.dirname(real), exist_ok=True)
    fd = os.open(real, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    os.chmod(real, mode)
    return real


# --------------------------------------------------------------------------------------------------
# stamped artifacts (plan 4.4 / 6.1 P5): ids, lengths, hashes, counts only
# --------------------------------------------------------------------------------------------------

_TEXTY_KEYS = {"text", "want_text", "head", "content", "excerpt", "x_text", "t_text", "prose"}


def _assert_text_free(obj: Any, path: str = "") -> None:
    """A mechanical guard on the rule 'no raw want text in a pushed/count artifact': no text-named key and no
    long string value may enter a stamped report."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).startswith("_"):
                continue
            if str(k) in _TEXTY_KEYS:
                raise Stop("report guard: text-like key %r at %s" % (k, path))
            _assert_text_free(v, path + "/" + str(k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _assert_text_free(v, path + "[%d]" % i)
    elif isinstance(obj, str) and len(obj) > 200:
        raise Stop("report guard: a %d-character string at %s" % (len(obj), path))


def _strip_private(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_private(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [_strip_private(v) for v in obj]
    return obj


def stamped(obj: Dict[str, Any]) -> Dict[str, Any]:
    out = _strip_private(obj)
    out["pin_stamp"] = pin_stamp()
    return out


def write_artifact(path: str, obj: Dict[str, Any]) -> str:
    """Stamp, text-guard, write (off-repo, guarded path). Returns the sha256 of the bytes written."""
    body = stamped(obj)
    _assert_text_free(body)
    data = canonical_json(body)
    out_write_bytes(path, data)
    return sha256_bytes(data)


def check_stamp(obj: Dict[str, Any], what: str = "artifact") -> None:
    """P5: REFUSE an artifact whose pin stamp does not equal the frozen pin."""
    if not isinstance(obj, dict) or obj.get("pin_stamp") != pin_stamp():
        raise Refusal("P5: %s carries no stamp equal to the frozen pin" % what)


def load_artifact(path: str, expected_sha256: Optional[str] = None) -> Dict[str, Any]:
    data = Path(path).read_bytes()
    if expected_sha256 is not None and sha256_bytes(data) != expected_sha256:
        raise Refusal("P5: %s sha256 %s != expected %s" % (os.path.basename(path), sha256_bytes(data), expected_sha256))
    obj = json.loads(data.decode("utf-8"))
    check_stamp(obj, os.path.basename(path))
    return obj


# --------------------------------------------------------------------------------------------------
# excerpt scrub (plan 5.3): token-like strings / key=value credentials -> [REDACTED:<rule>]; secrets by NAME
# --------------------------------------------------------------------------------------------------

_SCRUB_RULES: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    ("private_key_block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S)),
    ("sk_token", re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}")),
    ("slack_token", re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}")),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("bearer_token", re.compile(r"\b[Bb]earer\s+[A-Za-z0-9._\-]{16,}")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}")),
    ("credential_kv", re.compile(r"(?i)\b([A-Za-z0-9_.\-]*(?:api[_\-]?key|token|secret|passw(?:or)?d)[A-Za-z0-9_.\-]*)\s*([:=])\s*[\"']?[^\s\"',;]{6,}")),
)


def scrub(text: str) -> Tuple[str, int]:
    n = 0
    for name, rx in _SCRUB_RULES:
        if name == "credential_kv":
            text, k = rx.subn(lambda m: "%s%s[REDACTED:%s]" % (m.group(1), m.group(2), name), text)
        else:
            text, k = rx.subn("[REDACTED:%s]" % name, text)
        n += k
    return text, n


# --------------------------------------------------------------------------------------------------
# the checkpoint as bytes: a streaming section walk shared by the WRITER and the VERIFIER
# --------------------------------------------------------------------------------------------------

def _mp():
    import msgpack  # third-party, pinned by the NG environment; imported late so --help works without it
    return msgpack


_DESCEND = ("nodes", "synapses", "hyperedges", "archived_hyperedges")
_HDR = "__map_header__"
_HEADER = "__file_header__"


def _unpacker(raw: bytes):
    up = _mp().Unpacker(raw=False, max_buffer_size=len(raw) + 1, strict_map_key=False)
    up.feed(raw)
    return up


def iter_sections(raw: bytes):
    """Yield (section, entry_key, key_start, val_start, val_end) covering `raw` contiguously, in order.
    * (_HEADER, None, 0, 0, end)             the outer map header
    * (K, _HDR, ks, ks, end)                 top-level key K (a big map) + its map header, copied raw
    * (K, ek, eks, evs, eve)                 one entry of a big map (key bytes eks:evs, value evs:eve)
    * (K, None, ks, vs, ve)                  a plain top-level key/value
    The same walk drives the writer and the verifier, so 'positions preserved' is structural."""
    up = _unpacker(raw)
    n = up.read_map_header()
    yield (_HEADER, None, 0, 0, up.tell())
    for _ in range(n):
        ks = up.tell()
        key = up.unpack()
        vs = up.tell()
        if key in _DESCEND:
            m = up.read_map_header()
            yield (key, _HDR, ks, ks, up.tell())
            for _ in range(m):
                eks = up.tell()
                ek = up.unpack()
                evs = up.tell()
                up.skip()
                yield (key, ek, eks, evs, up.tell())
        else:
            up.skip()
            yield (key, None, ks, vs, up.tell())


def decode(b: bytes) -> Any:
    return _mp().unpackb(b, raw=False, strict_map_key=False)


def encode(obj: Any) -> bytes:
    return _mp().Packer(use_bin_type=True).pack(obj)


def fidelity_ok(raw_slice: bytes) -> bool:
    """V13: pack(unpack(raw)) == raw for a value we are about to re-encode."""
    return encode(decode(raw_slice)) == raw_slice


def is_protected(meta: Dict[str, Any]) -> bool:
    """Mirror of Graph._is_identity_protected (neuro_foundation.py:3551-3572): the constitutional flag, or a
    provenance ending '_authored'. A test asserts parity with the canonical method (LAW 4: cited, not forked)."""
    return bool(meta.get("constitutional")) or str(meta.get("provenance") or "").endswith("_authored")


def find_ids(obj: Any, ids) -> int:
    """Count whole-string occurrences of any id in `ids` anywhere in a decoded value (keys AND values)."""
    n = 0
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and k in ids:
                n += 1
            n += find_ids(v, ids)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            n += find_ids(v, ids)
    elif isinstance(obj, str) and obj in ids:
        n += 1
    return n


# --------------------------------------------------------------------------------------------------
# the raw-occurrence census (plan 3.4): whole-string occurrences of each old id across the six files
# --------------------------------------------------------------------------------------------------

def _mp_str_header(n: int) -> bytes:
    if n < 32:
        return bytes([0xA0 | n])
    if n < 256:
        return b"\xd9" + bytes([n])
    if n < 65536:
        return b"\xda" + n.to_bytes(2, "big")
    return b"\xdb" + n.to_bytes(4, "big")


def census_msgpack_file(path: str, ids: Iterable[str]) -> Counter:
    """Whole msgpack-str occurrences of each id (str header + the exact bytes) - independent of structure."""
    ids = list(ids)
    if not ids or os.path.getsize(path) == 0:
        return Counter()
    pats = {(_mp_str_header(len(i.encode())) + i.encode()): i for i in ids}
    rx = re.compile(b"|".join(re.escape(p) for p in sorted(pats, key=len, reverse=True)))
    out: Counter = Counter()
    with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        for m in rx.finditer(mm):
            out[pats[m.group(0)]] += 1
    return out


def census_json_file(path: str, ids: Iterable[str]) -> Counter:
    """Whole JSON-string occurrences (the id in double quotes) in a text file."""
    ids = list(ids)
    out: Counter = Counter()
    if not ids:
        return out
    text = Path(path).read_text(encoding="utf-8")
    rx = re.compile("|".join(re.escape('"%s"' % i) for i in ids))
    for m in rx.finditer(text):
        out[m.group(0)[1:-1]] += 1
    return out


def census_set(dirpath: str, ids: Iterable[str]) -> Dict[str, Counter]:
    """Census over the six files of a checkpoint directory (msgpack or JSON per file)."""
    ids = sorted(ids)
    res: Dict[str, Counter] = {}
    for name in SIX_FILES:
        p = os.path.join(dirpath, name)
        if not os.path.isfile(p):
            raise Stop("census: %s is missing from %s" % (name, dirpath))
        res[name] = census_msgpack_file(p, ids) if name.endswith(".msgpack") else census_json_file(p, ids)
    return res


# --------------------------------------------------------------------------------------------------
# the site table S1-S13 and the site-aware remap (plan 3.2). Anything found OUTSIDE it is a STOP.
# --------------------------------------------------------------------------------------------------

def _sub(s: Any, m: Dict[str, str]) -> Any:
    return m.get(s, s) if isinstance(s, str) else s


def _remap_list(lst: Any, m: Dict[str, str], subs: Counter, site: str) -> Any:
    if not isinstance(lst, list):
        return lst
    out, hit = [], 0
    for x in lst:
        y = _sub(x, m)
        hit += (y is not x and y != x)
        out.append(y)
    if hit:
        subs[site] += hit
        return out
    return lst


def _remap_keys(d: Any, m: Dict[str, str], subs: Counter, site: str) -> Any:
    """Re-key a dict in place order; a key colliding after the remap is a STOP (never merged)."""
    if not isinstance(d, dict):
        return d
    if not any(isinstance(k, str) and k in m for k in d):
        return d
    out: Dict[Any, Any] = {}
    for k, v in d.items():
        nk = _sub(k, m)
        if nk in out:
            raise Stop("remap: key collision on %s after the id remap (site %s)" % (nk, site))
        if nk != k:
            subs[site] += 1
        out[nk] = v
    return out


def remap_node(key: str, node: Dict[str, Any], m: Dict[str, str], old_text: Dict[str, str],
               new_text: Dict[str, str], subs: Counter) -> Tuple[str, Dict[str, Any], bool]:
    """S1 (key + node_id + want_text of a repaired node) and S2 (pred_weights keys, every node)."""
    changed = False
    new_key = key
    if key in m:
        if node.get("node_id") != key:
            raise Stop("remap: node_id field of %s does not equal its key" % key)
        md = node.get("metadata")
        if not isinstance(md, dict) or md.get("want_text") != old_text[key]:
            raise Stop("remap: %s want_text on disk is not the text the classification saw (drift)" % key)
        node = dict(node)
        node["node_id"] = m[key]
        subs["S1_nodes"] += 2  # the map key and the node_id field: two whole-string occurrences
        md2 = dict(md)
        md2["want_text"] = new_text[key]
        node["metadata"] = md2
        new_key, changed = m[key], True
    pw = node.get("pred_weights")
    pw2 = _remap_keys(pw, m, subs, "S2_pred_weights")
    if pw2 is not pw:
        node = dict(node)
        node["pred_weights"] = pw2
        changed = True
    return new_key, node, changed


def remap_synapse(syn: Dict[str, Any], m: Dict[str, str], subs: Counter) -> Tuple[Dict[str, Any], bool]:
    changed, out = False, syn
    for f in ("pre_node_id", "post_node_id"):
        v = syn.get(f)
        if isinstance(v, str) and v in m:
            if out is syn:
                out = dict(syn)
            out[f] = m[v]
            subs["S3_synapse_endpoints"] += 1
            changed = True
    md = syn.get("metadata")
    if isinstance(md, dict) and isinstance(md.get("expected_target"), str) and md["expected_target"] in m:
        if out is syn:
            out = dict(syn)
        md2 = dict(md)
        md2["expected_target"] = m[md["expected_target"]]
        out["metadata"] = md2
        subs["S4_expected_target"] += 1
        changed = True
    return out, changed


def remap_hyperedge(he: Dict[str, Any], m: Dict[str, str], subs: Counter) -> Tuple[Dict[str, Any], bool]:
    out = he
    for f, fn in (("member_nodes", _remap_list), ("output_targets", _remap_list), ("member_weights", _remap_keys)):
        v = he.get(f)
        nv = fn(v, m, subs, "S5_hyperedges")
        if nv is not v:
            if out is he:
                out = dict(he)
            out[f] = nv
    return out, out is not he


def remap_misc(key: str, value: Any, m: Dict[str, str], subs: Counter) -> Tuple[Any, bool]:
    """S6-S12: the non-entry top-level structures that hold node ids."""
    if key == "active_predictions" and isinstance(value, dict):
        out = value
        for pid, pd in value.items():
            if not isinstance(pd, dict):
                continue
            npd = pd
            for f in ("source_node_id", "target_node_id"):
                if isinstance(pd.get(f), str) and pd[f] in m:
                    if npd is pd:
                        npd = dict(pd)
                    npd[f] = m[pd[f]]
                    subs["S6_active_predictions"] += 1
            if npd is not pd:
                if out is value:
                    out = dict(value)
                out[pid] = npd
        return out, out is not value
    if key == "prediction_outcomes" and isinstance(value, list):
        out, hit = [], False
        for po in value:
            npo = po
            if isinstance(po, dict):
                pred = po.get("prediction")
                if isinstance(pred, dict):
                    npred = pred
                    for f in ("source_node_id", "target_node_id"):
                        if isinstance(pred.get(f), str) and pred[f] in m:
                            if npred is pred:
                                npred = dict(pred)
                            npred[f] = m[pred[f]]
                            subs["S7_prediction_outcomes"] += 1
                    if npred is not pred:
                        npo = dict(npo)
                        npo["prediction"] = npred
                afn = po.get("actual_firing_nodes")
                nafn = _remap_list(afn, m, subs, "S7_prediction_outcomes")
                if nafn is not afn:
                    if npo is po:
                        npo = dict(po)
                    npo["actual_firing_nodes"] = nafn
            hit = hit or (npo is not po)
            out.append(npo)
        return (out, True) if hit else (value, False)
    if key == "he_active_predictions" and isinstance(value, dict):
        out = value
        for pid, psd in value.items():
            if not isinstance(psd, dict):
                continue
            npsd = psd
            for f in ("predicted_targets", "confirmed_targets"):
                nv = _remap_list(psd.get(f), m, subs, "S8_he_active_predictions")
                if nv is not psd.get(f):
                    if npsd is psd:
                        npsd = dict(psd)
                    npsd[f] = nv
            if npsd is not psd:
                if out is value:
                    out = dict(value)
                out[pid] = npsd
        return out, out is not value
    if key == "he_output_candidates" and isinstance(value, dict):
        out = value
        for hid, inner in value.items():
            ninner = _remap_keys(inner, m, subs, "S9_he_output_candidates")
            if ninner is not inner:
                if out is value:
                    out = dict(value)
                out[hid] = ninner
        return out, out is not value
    if key == "novel_sequence_log" and isinstance(value, list):
        out, hit = [], False
        for ev in value:
            nev = ev
            if isinstance(ev, dict):
                if isinstance(ev.get("source"), str) and ev["source"] in m:
                    nev = dict(ev)
                    nev["source"] = m[ev["source"]]
                    subs["S10_novel_sequence_log"] += 1
                fn = ev.get("firing_nodes")
                nfn = _remap_list(fn, m, subs, "S10_novel_sequence_log")
                if nfn is not fn:
                    if nev is ev:
                        nev = dict(ev)
                    nev["firing_nodes"] = nfn
            hit = hit or (nev is not ev)
            out.append(nev)
        return (out, True) if hit else (value, False)
    if key == "delay_buffer" and isinstance(value, dict):
        out = value
        for ts, entries in value.items():
            if not isinstance(entries, list):
                continue
            nent, ehit = [], False
            for e in entries:
                if isinstance(e, (list, tuple)) and e and isinstance(e[0], str) and e[0] in m:
                    e = [m[e[0]]] + list(e[1:])
                    subs["S11_delay_buffer"] += 1
                    ehit = True
                nent.append(e)
            if ehit:
                if out is value:
                    out = dict(value)
                out[ts] = nent
        return out, out is not value
    if key == "recent_spikes" and isinstance(value, dict):
        nv = _remap_keys(value, m, subs, "S12_recent_spikes")
        return nv, nv is not value
    return value, False


ID_BEARING_TOP = ("nodes", "synapses", "hyperedges", "archived_hyperedges", "active_predictions",
                  "prediction_outcomes", "he_active_predictions", "he_output_candidates",
                  "novel_sequence_log", "delay_buffer", "recent_spikes")


# --------------------------------------------------------------------------------------------------
# the writer: value-granular, streaming, TMP file only (plan 3.4). Raw slices unless an id is inside.
# --------------------------------------------------------------------------------------------------

def rewrite_main(raw: bytes, out_path: str, m: Dict[str, str], old_text: Dict[str, str],
                 new_text: Dict[str, str]) -> Dict[str, Any]:
    """Write the rewritten main checkpoint to `out_path`. `m` = old->new ids (may be empty: then the output is
    byte-identical to the input). Returns stats (substitutions per site, entries re-encoded). Raises Stop on
    a fidelity failure (V13) or an id found outside the site table (census). The caller guards `out_path`."""
    out_path = guard_out_path(out_path)
    refuse_inplace_write(out_path)
    try:
        return _rewrite_main_into(raw, out_path, m, old_text, new_text)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(out_path)                   # all-or-nothing: no partial output survives a STOP
        raise


def _rewrite_main_into(raw: bytes, out_path: str, m: Dict[str, str], old_text: Dict[str, str],
                       new_text: Dict[str, str]) -> Dict[str, Any]:
    subs: Counter = Counter()
    reencoded = Counter()
    ids = set(m)
    packer = _mp().Packer(use_bin_type=True)
    with open(out_path, "wb") as out:
        for section, ek, ks, vs, ve in iter_sections(raw):
            if section == _HEADER or ek is _HDR:
                out.write(raw[ks:ve] if section != _HEADER else raw[0:ve])
                continue
            if ek is not None:                       # an entry of a big map
                vbytes = raw[vs:ve]
                new_key, new_val, changed = ek, None, False
                if section == "nodes":
                    node = decode(vbytes)
                    new_key, new_val, changed = remap_node(ek, node, m, old_text, new_text, subs)
                    residual = find_ids(new_val, ids)
                elif section == "synapses":
                    syn = decode(vbytes)
                    new_val, changed = remap_synapse(syn, m, subs)
                    residual = find_ids(new_val, ids)
                else:
                    he = decode(vbytes)
                    new_val, changed = remap_hyperedge(he, m, subs)
                    residual = find_ids(new_val, ids)
                if residual:
                    raise Stop("census: %d old-id occurrence(s) outside the site table in %s[%r]" % (residual, section, ek))
                if not changed:
                    out.write(raw[ks:ve])
                    continue
                if not fidelity_ok(vbytes):
                    raise Stop("V13: encoder fidelity failed for %s[%r]" % (section, ek))
                reencoded[section] += 1
                out.write(packer.pack(new_key) if new_key != ek else raw[ks:vs])
                out.write(packer.pack(new_val))
                continue
            # a plain top-level key/value
            if section in ID_BEARING_TOP:
                value = decode(raw[vs:ve])
                new_val, changed = remap_misc(section, value, m, subs)
                residual = find_ids(new_val, ids)
                if residual:
                    raise Stop("census: %d old-id occurrence(s) outside the site table in top-level %s" % (residual, section))
                if changed:
                    if not fidelity_ok(raw[vs:ve]):
                        raise Stop("V13: encoder fidelity failed for top-level %s" % section)
                    reencoded[section] += 1
                    out.write(raw[ks:vs])
                    out.write(packer.pack(new_val))
                    continue
            elif ids and find_ids(decode(raw[vs:ve]), ids):
                raise Stop("census: old-id occurrence(s) in top-level %s, which is not in the site table" % section)
            out.write(raw[ks:ve])
    return {"substitutions": dict(subs), "substitutions_total": int(sum(subs.values())),
            "reencoded_entries": dict(reencoded)}


def rewrite_sidecar(text: str, m: Dict[str, str]) -> Tuple[str, Dict[str, Any]]:
    """S13: re-key `entries`. Fidelity: the file must equal json.dumps(json.loads(file)) (write_state uses a
    plain json.dump), else STOP. Every other field is carried."""
    obj = json.loads(text)
    if json.dumps(obj) != text:
        raise Stop("V13: the activation sidecar does not round-trip through json byte-for-byte")
    ids = set(m)
    entries = obj.get("entries")
    subs = 0
    if isinstance(entries, dict):
        new_entries: Dict[str, Any] = {}
        for k, v in entries.items():
            nk = m.get(k, k)
            if nk in new_entries:
                raise Stop("remap: sidecar key collision on %s" % nk)
            subs += (nk != k)
            new_entries[nk] = v
        obj["entries"] = new_entries
    if find_ids({k: v for k, v in obj.items() if k != "entries"}, ids) or find_ids(list(obj.get("entries", {}).values()), ids):
        raise Stop("census: old-id occurrence in the sidecar outside its entries keys")
    return json.dumps(obj), {"substitutions": subs}


# --------------------------------------------------------------------------------------------------
# in-memory fakes (the same shape #810's own golden tests use) - for the pinned surface_wants, T6, residuals
# --------------------------------------------------------------------------------------------------

class _FakeNode:
    def __init__(self, metadata=None):
        self.metadata = dict(metadata or {})
        self.creation_time = 0.0


class _FakeGraph:
    def __init__(self):
        import threading
        self.nodes: Dict[str, _FakeNode] = {}
        self.synapses: List[Tuple[str, str, float]] = []
        self._step_lock = threading.RLock()

    def create_node(self, node_id, metadata=None):
        n = _FakeNode(metadata)
        self.nodes[node_id] = n
        return n

    def create_synapse(self, a, b, weight=0.0):
        self.synapses.append((a, b, weight))


class _FakeVDB:
    def __init__(self, content):
        self.content = content


# --------------------------------------------------------------------------------------------------
# classification (plan 4.2). The rule is the pinned function's; no proxy decides a node.
# --------------------------------------------------------------------------------------------------

def text_class(t: str) -> str:
    """A = backtick-led + marker, B = backtick-led no marker, C = not backtick-led + marker, D = neither
    (plan 4.4, definitions of analysis-scratch/want_probe.py)."""
    bt = t.lstrip().startswith("`")
    mk = ("[WANT]" in t) or ("[/WANT]" in t)
    return "A" if bt and mk else "B" if bt else "C" if mk else "D"


def derive_scope(nodes_meta: Dict[str, Dict[str, Any]], min_len: int) -> List[str]:
    """The rule-derived scope (a cross-check / the Phase-1 derivation): cc_authored wants over `min_len`."""
    out = []
    for nid, md in nodes_meta.items():
        wt = md.get("want_text")
        if md.get("kind") == "want" and md.get("provenance") == "cc_authored" and isinstance(wt, str) and len(wt) > min_len:
            out.append(nid)
    return sorted(out)


class Classifier:
    def __init__(self, org, nodes_meta: Dict[str, Dict[str, Any]], content: Dict[str, str]):
        self.org, self.nodes, self.content = org, nodes_meta, content
        self._cache: Dict[str, Any] = {}

    def parse(self, src: str):
        if src not in self._cache:
            self._cache[src] = self.org.parse_wants(self.content[src])
        return self._cache[src]

    def classify_node(self, nid: str) -> Dict[str, Any]:
        org = self.org
        WO, WC = org.WANT_OPEN, org.WANT_CLOSE
        meta = self.nodes[nid]
        T = meta.get("want_text")
        rec: Dict[str, Any] = {"id": nid, "old_len": len(T), "old_sha16": sha16(T), "class": text_class(T),
                               "outcome": None, "disposition": "unchanged", "detail": "", "flags": {}, "_t": T}
        src = meta.get("source_node")
        smeta = self.nodes.get(src) if isinstance(src, str) else None
        C = self.content.get(src) if smeta is not None else None
        conds = {
            "source_exists": smeta is not None,
            "source_conversational": bool(smeta) and smeta.get("creation_mode") == "conversational",
            "source_not_want": bool(smeta) and smeta.get("kind") != "want",
            "source_not_protected": bool(smeta) and not is_protected(smeta),
            "content_present": isinstance(C, str) and bool(C),
            "content_has_marker": isinstance(C, str) and "WANT]" in C,
        }
        if not all(conds.values()):
            rec.update(outcome="SOURCE_MISSING", detail=",".join(k for k, v in conds.items() if not v))
            return rec
        rec["source_node"] = src
        # A1 anchor
        if C.count(T) != 1:
            rec.update(outcome="ANCHOR_FAILED", detail="text_count=%d" % C.count(T))
            return rec
        p = C.index(T)
        head = C[:p].rstrip()
        if not head.endswith(WO):
            rec.update(outcome="ANCHOR_FAILED", detail="no_outer_opener")
            return rec
        i = len(head) - len(WO)
        j = p + len(T)
        j += len(C[j:]) - len(C[j:].lstrip())
        if not C.startswith(WC, j):
            rec.update(outcome="ANCHOR_FAILED", detail="no_paired_closer")
            return rec
        closer_end = j + len(WC)
        wp = self.parse(src)
        w = next((x for x in wp.wants if x.close_end == closer_end), None)
        span_reasons = Counter(sk.reason for sk in wp.skipped if i <= sk.start < closer_end)
        outer_skip = next((sk.reason for sk in wp.skipped if sk.start == i and sk.marker == WO), None)
        fired = [r for r in REGION_FORCE_REVIEW if span_reasons.get(r)]
        rec["reasons"] = {r: span_reasons[r] for r in REASONS if span_reasons.get(r)}
        rec["flags"] = {"t_has_marker": (WO in T) or (WC in T), "region_fired": fired,
                        "forces_hand_review": bool(fired), "outer_opener_skip_reason": outer_skip}
        rec["_i"], rec["_j"], rec["_closer_end"] = i, j, closer_end
        rec["closer_rel"] = j - i
        if w is None:
            overlap = [x for x in wp.wants if x.open_start < closer_end and x.close_end > i]
            if overlap:
                rec.update(outcome="OVERLAP", detail="overlap_other_pair")
                rec["overlap_wants"] = [{"want_id": x.want_id, "len": len(x.text), "sha16": sha16(x.text),
                                          "opens_before_span": x.open_start < i, "closes_after_span": x.close_end > closer_end}
                                         for x in overlap]
                rec["flags"]["overlap_marker_bearing"] = any((WO in x.text) or (WC in x.text) for x in overlap)
            else:
                rec.update(outcome="NONE", detail="every_marker_in_span_skipped")
            return rec
        rec["_w_open"] = w.open_start
        rec["inner_open_rel"] = w.open_start - i
        if w.open_start == i and w.text == T:
            if org.want_id_for_text(T) != nid:
                rec.update(outcome="ID_MISMATCH", detail="id_not_want_id_for_text")
            else:
                rec.update(outcome="GENUINE", detail="old_span_is_a_legitimate_want")
            return rec
        if w.open_start > i:
            rec["outcome"] = "SEPARATE"
            X = w.text
            inner = C[w.open_start + len(WO): w.close_end - len(WC)].strip()
            post = {
                "x_equals_offsets_slice": X == inner,
                "x_in_t": X in T,
                "t_endswith_x": T.endswith(X),
                "x_nonempty": bool(X),
                "id_equals_function": org.want_id_for_text(X) == w.want_id,
            }
            failed = [k for k, v in post.items() if not v]
            rec["removed_prefix_len"] = w.open_start - (i + len(WO))
            rec["flags"]["x_has_marker"] = (WO in X) or (WC in X)
            rec["flags"]["marker_bearing"] = rec["flags"]["x_has_marker"] or rec["flags"]["t_has_marker"]
            if failed:
                rec.update(disposition="assert_failed", detail=",".join(failed))
                return rec
            rec.update(disposition="candidate", detail="nested_pair_is_the_want", new_id=w.want_id,
                       new_len=len(X), new_sha16=sha16(X), _x=X)
            return rec
        rec.update(outcome="ANOMALY", detail=("pair_spans_beyond_old_span" if w.open_start < i else "function_disagrees_with_anchor"))
        return rec


def apply_collision_rule(records: List[Dict[str, Any]], existing_ids) -> List[Dict[str, Any]]:
    """Plan 3.1: a repair whose new_id already exists as ANY node in the pre-repair graph, or that two repairs
    produce, is DROPPED with both parties left unchanged and LISTED - never merged. Returns the dropped ones."""
    cands = [r for r in records if r["disposition"] == "candidate"]
    n_new = Counter(r["new_id"] for r in cands)
    dropped = []
    for r in cands:
        if r["new_id"] in existing_ids:
            r.update(disposition="collision_dropped", detail="collision:new_id_exists_in_graph")
        elif n_new[r["new_id"]] > 1:
            r.update(disposition="collision_dropped", detail="collision:new_id_produced_by_two_repairs")
        else:
            continue
        dropped.append(r)
    return dropped


def build_mapping(records: List[Dict[str, Any]], only_ids=None) -> Dict[str, str]:
    m: Dict[str, str] = {}
    for r in records:
        if r["disposition"] != "candidate":
            continue
        if only_ids is not None and r["id"] not in only_ids:
            continue
        m[r["id"]] = r["new_id"]
    if len(set(m.values())) != len(m):
        raise Stop("mapping is not injective")
    return m


def is_choice_clause_marked(md: Dict[str, Any]) -> bool:
    """The metadata flag the deny-check reads (le-029 C8): a truthy `constitutional`, a truthy `choice_clause`, or a
    tag / tags / kind / category mentioning choice_clause. (The plan names no other tag: this is my reading of
    'a Choice-Clause tag', flagged in the return.)"""
    if md.get("constitutional") or md.get("choice_clause"):
        return True
    for k in ("tag", "tags", "kind", "category"):
        v = md.get(k)
        for x in (v if isinstance(v, (list, tuple, set)) else [v]):
            if isinstance(x, str) and "choice_clause" in x.lower().replace(" ", "_").replace("-", "_"):
                return True
    return False


def deny_check(scope_ids, mapping: Dict[str, str], approval_ids=(), nodes_meta: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    """V15 deny-check: neither Choice Clause id, nor the constitutional id, is in S, in the mapping (old or new)
    or in any approval entry - AND no member of S (nor any id it maps from) carries the constitutional / Choice
    Clause flag in its metadata under some other id. The collision rule drops and lists; this is the proof it held."""
    protected = set(CHOICE_CLAUSE_IDS) | {CONSTITUTIONAL_ID}
    flagged = []
    if nodes_meta is not None:
        flagged = sorted(i for i in set(scope_ids) | set(mapping) if is_choice_clause_marked(nodes_meta.get(i, {})))
        if flagged:
            raise Stop("V15 deny-check: %d member(s) of S / the mapping carry the constitutional or Choice Clause flag "
                       "under another id (first: %s)" % (len(flagged), flagged[0]))
    bad = {
        "in_scope": sorted(protected & set(scope_ids)),
        "in_mapping_old": sorted(protected & set(mapping)),
        "in_mapping_new": sorted(protected & set(mapping.values())),
        "in_approvals": sorted(protected & set(approval_ids)),
    }
    if any(bad.values()):
        raise Stop("V15 deny-check: a Choice Clause / constitutional id is in S, the mapping or an approval: %s" % bad)
    return {"denied_ids": sorted(protected), "clean": True}


# --------------------------------------------------------------------------------------------------
# reports: histograms (a), marker-bearing minted list (b), residual classes (c), mention shapes (d)
# --------------------------------------------------------------------------------------------------

def conversational_marker_nodes(nodes_meta, content) -> List[str]:
    return sorted(nid for nid, md in nodes_meta.items()
                  if md.get("creation_mode") == "conversational" and isinstance(content.get(nid), str) and "WANT]" in content[nid])


def reason_histograms(cl: Classifier, s_sources: List[str], marker_nodes: List[str]) -> Dict[str, Any]:
    def hist(nodes):
        h = {r: 0 for r in REASONS}
        fired: Dict[str, List[str]] = {r: [] for r in REGION_FORCE_REVIEW}
        for nid in nodes:
            seen = set()
            for sk in cl.parse(nid).skipped:
                h[sk.reason] += 1
                seen.add(sk.reason)
            for r in REGION_FORCE_REVIEW:
                if r in seen:
                    fired[r].append(nid)
        return {"nodes": len(nodes), "by_reason": h, "hand_review_nodes": {r: sorted(v) for r, v in fired.items()},
                "hand_review_counts": {r: len(v) for r, v in fired.items()}}
    return {"s_sources": hist(s_sources), "all_conversational_marker_nodes": hist(marker_nodes)}


def marker_bearing_minted(cl: Classifier, marker_nodes: List[str], s_sources: set) -> List[Dict[str, Any]]:
    WO, WC = cl.org.WANT_OPEN, cl.org.WANT_CLOSE
    out = []
    for nid in marker_nodes:
        for w in cl.parse(nid).wants:
            if WO in w.text or WC in w.text:
                out.append({"source_node": nid, "want_id": w.want_id, "len": len(w.text), "sha16": sha16(w.text),
                            "open_start": w.open_start, "close_end": w.close_end,
                            "in_s_source": nid in s_sources, "to_frozen_list_review": True})
    return out


_SHAPE_RX = {
    "single_quoted_json_ish": re.compile(r"'\s*$"),
    "reference_definition_line": re.compile(r"^\s{0,3}\[[^\]\n]+\]:\s"),
    "www_mailto_data": re.compile(r"(?:www\.|mailto:|data:)\S*$"),
    "yaml_toml_string_line": re.compile(r"^\s*[\w.\-]+\s*[:=]\s*[\"'].*$"),
    "blockquote_line": re.compile(r"^\s{0,3}>"),
    "indented_code_line": re.compile(r"^(?: {4}|\t)"),
}


def mention_shape_flags(C: str, open_pos: int, close_end: int) -> List[str]:
    """ADVISORY only (never a decision input): which named 'still MINTS a mention' shapes (FINAL PIN (d)) the
    surrounding text resembles - for the frozen-list review. Booleans by name; no text leaves this function."""
    flags: List[str] = []
    ls = C.rfind("\n", 0, open_pos) + 1
    line_head = C[ls:open_pos]
    line = C[ls:(C.find("\n", close_end) if C.find("\n", close_end) != -1 else len(C))]
    if _SHAPE_RX["single_quoted_json_ish"].search(line_head):
        flags.append("single_quoted_json_ish")
    if _SHAPE_RX["reference_definition_line"].search(line):
        flags.append("scheme_less_reference_definition")
    if _SHAPE_RX["www_mailto_data"].search(line_head):
        flags.append("www_mailto_data")
    if _SHAPE_RX["yaml_toml_string_line"].search(line):
        flags.append("yaml_toml_string")
    if _SHAPE_RX["blockquote_line"].search(line):
        flags.append("blockquote")
    if _SHAPE_RX["indented_code_line"].search(line):
        flags.append("indented_code")
    before = C[max(0, open_pos - 200):open_pos]
    after = C[close_end:close_end + 200]
    if before.rfind("<!--") > before.rfind("-->") or before.rfind("<code>") > before.rfind("</code>"):
        flags.append("html_comment_or_code_tag")
    if before.endswith(("*", "_")) or after.startswith(("*", "_")):
        flags.append("emphasis_around_tokens")
    if before.rfind("[") > before.rfind("]") and "](" in C[close_end:close_end + 40]:
        flags.append("link_text_contains_pair")
    if re.search(r"\w\[[^\]\n]*\]\([^)\n]*$", line_head):
        flags.append("real_link_glued_to_word_f4b")
    return flags


def residual_classes(pinned, base_mod, nodes_meta, content, marker_nodes, cl: Classifier) -> Dict[str, Any]:
    """FINAL PIN (c): what the BASE function mints over the same content but the PINNED function does not (wants
    over 600 characters excluded by design in base), grouped by the pinned function's skip reason at that span,
    with an 'unclassified' bucket. LISTED with counts, NOT fixed."""
    org = pinned.org
    WO, WC = org.WANT_OPEN, org.WANT_CLOSE
    g = _FakeGraph()
    vdb_content: Dict[str, str] = {}
    for nid in marker_nodes:
        g.nodes[nid] = _FakeNode({"creation_mode": "conversational"})
        vdb_content[nid] = content[nid]
    base_mod.surface_wants(g, _FakeVDB(vdb_content))
    groups: Dict[str, Dict[str, Any]] = {}
    total = 0
    for wid, node in g.nodes.items():
        md = node.metadata
        if md.get("kind") != "want":
            continue
        src, text = md.get("source_node"), md.get("want_text")
        if not isinstance(src, str) or not isinstance(text, str):
            continue
        pinned_ids = {w.want_id for w in cl.parse(src).wants}
        if wid in pinned_ids:
            continue
        total += 1
        C = content[src]
        p = C.find(text)
        key = "unclassified"
        if p != -1:
            head = C[:p].rstrip()
            i = len(head) - len(WO) if head.endswith(WO) else p
            cend = C.find(WC, p + len(text))
            cend = (cend + len(WC)) if cend != -1 else p + len(text)
            reasons = sorted({sk.reason for sk in cl.parse(src).skipped if i <= sk.start < cend})
            key = "+".join(reasons) if reasons else "unclassified"
        grp = groups.setdefault(key, {"count": 0, "want_ids": []})
        grp["count"] += 1
        grp["want_ids"].append(wid)
    for grp in groups.values():
        grp["want_ids"].sort()
    return {"base_minted_pinned_drops_total": total, "by_pinned_reason": dict(sorted(groups.items()))}


# --------------------------------------------------------------------------------------------------
# T6: the real pinned surface_wants on a stub graph, before and after (plan 7 T6 / gate P10's input)
# --------------------------------------------------------------------------------------------------

def _minted_by_surface_wants(pinned, nodes_meta: Dict[str, Dict[str, Any]], content: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    g = _FakeGraph()
    for nid, md in nodes_meta.items():
        g.nodes[nid] = _FakeNode(md)          # metadata dict is copied by the fake; the source is never mutated
    before = set(g.nodes)
    pinned.org.surface_wants(g, _FakeVDB(content))
    return {nid: dict(g.nodes[nid].metadata) for nid in g.nodes if nid not in before}


def t6_replay(pinned, meta_before, meta_after, content, mapping: Dict[str, str], cl_after: Classifier) -> Dict[str, Any]:
    """minted(after) is a subset of minted(before); minted(before) - minted(after) is a subset of {new ids}.
    A non-empty minted(after) is NOT a repair defect: it is the would-mint set for gate P10 (S4's gate)."""
    WO, WC = pinned.org.WANT_OPEN, pinned.org.WANT_CLOSE
    mb = _minted_by_surface_wants(pinned, meta_before, content)
    ma = _minted_by_surface_wants(pinned, meta_after, content)
    new_ids = set(mapping.values())
    ok_subset = set(ma) <= set(mb)
    ok_diff = (set(mb) - set(ma)) <= new_ids
    would = []
    for wid in sorted(ma):
        md = ma[wid]
        text = md.get("want_text", "")
        src = md.get("source_node")
        span_fired, node_fired = [], []
        if isinstance(src, str) and src in content:
            wp = cl_after.parse(src)
            w = next((x for x in wp.wants if x.want_id == wid), None)
            node_fired = [r for r in REGION_FORCE_REVIEW if any(sk.reason == r for sk in wp.skipped)]
            if w is not None:
                span_fired = [r for r in REGION_FORCE_REVIEW
                              if any(sk.reason == r and w.open_start <= sk.start < w.close_end for sk in wp.skipped)]
        would.append({"want_id": wid, "len": len(text), "sha16": sha16(text), "source_node": src,
                      "marker_bearing": (WO in text) or (WC in text), "region_fired_in_span": span_fired,
                      "region_fired_in_node": node_fired})
    if not (ok_subset and ok_diff):
        raise Stop("T6: minted(after) subset of minted(before)=%s ; minted(before)-minted(after) subset of new ids=%s"
                   % (ok_subset, ok_diff))
    return {"minted_before": len(mb), "minted_after": len(ma), "minted_after_subset_of_before": ok_subset,
            "before_minus_after_subset_of_new_ids": ok_diff, "would_mint_count": len(would),
            "would_mint_marker_bearing": sum(1 for w in would if w["marker_bearing"]),
            "would_mint": would}


# --------------------------------------------------------------------------------------------------
# review artifacts (plan 5.1): excerpt anchors inside excerpt_sha256; excerpts off-repo, scrubbed, 0600
# --------------------------------------------------------------------------------------------------

EXCERPT_WINDOW = 80


def excerpt_anchors(org, C: str, rec: Dict[str, Any]) -> Dict[str, Any]:
    WO, WC = org.WANT_OPEN, org.WANT_CLOSE
    W = EXCERPT_WINDOW
    i, wo, j = rec["_i"], rec["_w_open"], rec["_j"]
    redactions = 0

    def around(pos, ln):
        nonlocal redactions
        s, k = scrub(C[max(0, pos - W): pos + ln + W])
        redactions += k
        return s

    outer, inner, closer = around(i, len(WO)), around(wo, len(WO)), around(j, len(WC))
    x, k = scrub(rec["_x"])
    redactions += k
    return {"outer_opener": outer, "inner_opener": inner, "closer": closer, "x_full": x,
            "removed_prefix_len": rec["removed_prefix_len"], "scrub_version": SCRUB_VERSION,
            "_redactions": redactions}


def excerpt_sha256(anchors: Dict[str, Any]) -> str:
    body = {k: v for k, v in anchors.items() if not k.startswith("_")}
    return sha256_bytes(json.dumps(body, sort_keys=True, ensure_ascii=True).encode("utf-8"))


def attach_excerpt_hashes(org, content: Dict[str, str], records: List[Dict[str, Any]]) -> int:
    """Compute excerpt_sha256 + the scrubbed anchors for every candidate (from the LIVE source at run time)."""
    total = 0
    for r in records:
        if r["disposition"] != "candidate":
            continue
        a = excerpt_anchors(org, content[r["source_node"]], r)
        r["_anchors"] = a
        r["excerpt_sha256"] = excerpt_sha256(a)
        r["flags"]["scrub_redactions"] = a["_redactions"]
        total += a["_redactions"]
    return total


def write_review_files(run_dir: str, utc: str, org, content, records) -> Dict[str, str]:
    """The ONLY files that carry excerpts: <run>/review/*.md, mode 0600. Returns {relpath: sha256}."""
    rdir = os.path.join(run_dir, "review")
    hdr = "PIN-STAMP: %s\nOFF-REPO / TEXT-BEARING / mode 0600 - never push, never paste (plan 5.3)\n\n" % json.dumps(pin_stamp(), sort_keys=True)
    lines = [hdr, "# review excerpts - SEPARATE candidates (the frozen-list review, per id)\n"]
    for r in sorted((x for x in records if x["disposition"] == "candidate"), key=lambda x: x["id"]):
        a = r["_anchors"]
        lines += ["## %s -> %s  (class %s, old_len %d, new_len %d)\n" % (r["id"], r["new_id"], r["class"], r["old_len"], r["new_len"]),
                  "excerpt_sha256: %s ; removed-prefix length: %d ; flags: %s\n" % (r["excerpt_sha256"], a["removed_prefix_len"], json.dumps(r["flags"], sort_keys=True)),
                  "### (1) outer opener +/-%d\n%s\n### (2) inner opener +/-%d\n%s\n### (3) closer +/-%d\n%s\n### (4) X in full\n%s\n" % (
                      EXCERPT_WINDOW, a["outer_opener"], EXCERPT_WINDOW, a["inner_opener"], EXCERPT_WINDOW, a["closer"], a["x_full"])]
    p1 = os.path.join(rdir, "review-excerpts-%s.md" % utc)
    out_write_bytes(p1, "\n".join(lines).encode("utf-8"), 0o600)
    left = [hdr, "# LEFT list - nodes left UNCHANGED (never guessed; delete nothing)\n"]
    WO, WC = org.WANT_OPEN, org.WANT_CLOSE
    for r in sorted((x for x in records if x["disposition"] != "candidate"), key=lambda x: x["id"]):
        left.append("## %s outcome=%s disposition=%s detail=%s reasons=%s\n" % (r["id"], r["outcome"], r["disposition"], r["detail"], json.dumps(r.get("reasons", {}), sort_keys=True)))
        src, i = r.get("source_node"), r.get("_i")
        if src and i is not None and src in content:
            seg, _ = scrub(content[src][max(0, i - EXCERPT_WINDOW): i + len(WO) + EXCERPT_WINDOW])
            left.append("outer opener +/-%d: %s\n" % (EXCERPT_WINDOW, seg))
    p2 = os.path.join(rdir, "left-list-%s.md" % utc)
    out_write_bytes(p2, "\n".join(left).encode("utf-8"), 0o600)
    return {"review/" + os.path.basename(p1): sha256_file(p1), "review/" + os.path.basename(p2): sha256_file(p2)}


# --------------------------------------------------------------------------------------------------
# approvals (plan 5.2): a node is written only if EVERYTHING holds; anything else is REFUSED per id
# --------------------------------------------------------------------------------------------------

PROVISIONAL_PACKET = "PROVISIONAL-SELF-DRY-RUN-NOT-AN-APPROVAL"


def approvals_body_for(records: List[Dict[str, Any]], repair_list_sha256: str, scope_ids_sha256: str,
                       packet: str, decision: str = "approved") -> Dict[str, Any]:
    return {
        "packet": packet, "function_pin": dict(PIN), "repair_list_sha256": repair_list_sha256,
        "scope_ids_sha256": scope_ids_sha256, "scrub_version": SCRUB_VERSION,
        "poincare_dir_carried": "acknowledged",
        "entries": [{"id": r["id"], "decision": decision, "excerpt_sha256": r["excerpt_sha256"],
                     "x_sha16": r["new_sha16"], "t_sha16": r["old_sha16"]}
                    for r in sorted(records, key=lambda x: x["id"]) if r["disposition"] == "candidate"],
    }


def provisional_body_for(records: List[Dict[str, Any]], repair_list_sha256: str, scope_ids_sha256: str) -> Dict[str, Any]:
    """The DRY-RUN packet (le-029 C5): decision `provisional` (not in approved/struck) and a top-level
    `provisional: true`. ONLY stage_rewrite accepts it (provisional_ok); load_approvals - the only door to
    Phase 2 - refuses it, whatever else is edited."""
    body = approvals_body_for(records, repair_list_sha256, scope_ids_sha256, PROVISIONAL_PACKET, decision="provisional")
    body["provisional"] = True
    return body


def load_approvals(path: str, expected_file_sha256: str, repair_list_sha256: str, scope_ids_sha256: str) -> Dict[str, Any]:
    if not expected_file_sha256:
        raise Refusal("approvals: --approvals-sha256 (the hash relayed from the Executive's packet) is REQUIRED")
    obj = load_artifact(path, expected_file_sha256)          # file sha256 == relayed value; stamp == frozen pin (P5)
    problems = []
    if obj.get("function_pin") != dict(PIN):
        problems.append("function_pin")
    if obj.get("scrub_version") != SCRUB_VERSION:
        problems.append("scrub_version")
    if obj.get("repair_list_sha256") != repair_list_sha256:
        problems.append("repair_list_sha256")
    if obj.get("scope_ids_sha256") != scope_ids_sha256:
        problems.append("scope_ids_sha256")
    if obj.get("poincare_dir_carried") != "acknowledged":
        problems.append("poincare_dir_carried")
    if not isinstance(obj.get("packet"), str) or not obj["packet"]:
        problems.append("packet")
    if obj.get("packet") == PROVISIONAL_PACKET or obj.get("provisional"):
        problems.append("provisional_packet_is_not_an_approval")
    ents = obj.get("entries")
    if not isinstance(ents, list) or len({e.get("id") for e in ents if isinstance(e, dict)}) != len(ents):
        problems.append("entries")
    else:
        for e in ents:
            if e.get("decision") not in ("approved", "struck"):
                problems.append("decision:%s" % e.get("id"))
    if problems:
        raise Refusal("approval_mismatch: %s" % ",".join(problems))
    return obj


def approved_decisions(approvals: Dict[str, Any], provisional_ok: bool) -> Tuple[str, ...]:
    """The decisions that count as 'approved'. `provisional` counts ONLY for the dry-run rewrite path."""
    if provisional_ok and approvals.get("provisional") is True and approvals.get("packet") == PROVISIONAL_PACKET:
        return ("approved", "provisional")
    return ("approved",)


def gate_write_set(records: List[Dict[str, Any]], approvals: Dict[str, Any], *, provisional_ok: bool = False) -> Tuple[List[str], List[Dict[str, str]]]:
    """(write_ids, refusals). `struck` = leave the node EXACTLY as-is (Exec P423-C1). Every recompute is from
    the run's own live data; an entry's hashes must equal them."""
    by_id = {e["id"]: e for e in approvals["entries"]}
    ok_dec = approved_decisions(approvals, provisional_ok)
    write, refused = [], []
    for r in sorted(records, key=lambda x: x["id"]):
        if r["disposition"] != "candidate":
            continue
        e = by_id.get(r["id"])
        if e is None or e["decision"] not in ok_dec:
            refused.append({"id": r["id"], "reason": "not_approved"})
        elif (e.get("excerpt_sha256") != r["excerpt_sha256"] or e.get("x_sha16") != r["new_sha16"]
              or e.get("t_sha16") != r["old_sha16"]):
            refused.append({"id": r["id"], "reason": "approval_mismatch"})
        else:
            write.append(r["id"])
    return write, refused


# --------------------------------------------------------------------------------------------------
# the analysis stage: the analysis-001 loader (canonical readers) -> classification -> reports
# --------------------------------------------------------------------------------------------------

def load_pair(dirpath: str, pinned):
    """THE analysis-001 loader (analysis-scratch/analyze_pair.py:84-86): Graph().restore + SimpleVectorDB().load,
    the two canonical readers. Nothing is forked; the checkpoint is only READ."""
    g = pinned.nf.Graph()
    g.restore(os.path.join(dirpath, MAIN_NAME))
    vdb = pinned.ui.SimpleVectorDB()
    vdb.load(os.path.join(dirpath, VECTORS_NAME))
    return g, vdb


def incident_figures(g, ids) -> Dict[str, Tuple[int, int, int]]:
    """V11 'before' figures: (#outgoing, #incoming, #hyperedges) per id, from the canonical restore."""
    return {i: (len(g._outgoing.get(i, ())), len(g._incoming.get(i, ())), len(g._node_hyperedges.get(i, ())))
            for i in ids if i in g.nodes}


def synapse_stats(raw_main: bytes, want_ids: set, scope_ids: set, mapped: set) -> Dict[str, Any]:
    """One streaming pass over the synapses: how many touch S / S<->any-want / rim<->S, and the PRE-node report
    line (plan 4.2-7): is either Choice Clause want, or the rim, a PRE-node of a synapse to a mapped id."""
    total = touch_s = s_any_want = rim_s = rim_mapped = 0
    pre_of_mapped: Counter = Counter()
    watch = set(CHOICE_CLAUSE_IDS) | {CONSTITUTIONAL_ID}
    for section, ek, ks, vs, ve in iter_sections(raw_main):
        if section != "synapses" or ek is None or ek is _HDR:
            continue
        syn = decode(raw_main[vs:ve])
        pre, post = syn.get("pre_node_id"), syn.get("post_node_id")
        total += 1
        if pre in scope_ids or post in scope_ids:
            touch_s += 1
            other = post if pre in scope_ids else pre
            if other in want_ids:
                s_any_want += 1
        if CONSTITUTIONAL_ID in (pre, post):
            other = post if pre == CONSTITUTIONAL_ID else pre
            if other in scope_ids:
                rim_s += 1
            if other in mapped:
                rim_mapped += 1
        if pre in watch and post in mapped:
            pre_of_mapped[pre] += 1
    line = {}
    for x in list(CHOICE_CLAUSE_IDS) + [CONSTITUTIONAL_ID]:
        line[x] = {"is_pre_node_of_synapse_to_mapped_id": pre_of_mapped[x] > 0, "count": int(pre_of_mapped[x])}
    return {"synapses_total": total, "touch_scope": touch_s, "scope_to_any_want": s_any_want,
            "rim_to_scope": rim_s, "rim_to_mapped": rim_mapped, "pre_node_report": line}


def analyze(pinned, dirpath: str, *, scope_min_len: int, frozen_scope: Optional[List[str]] = None,
            base_mod=None, full_reports: bool = True) -> Dict[str, Any]:
    """Load the pair with the canonical readers and classify every node of S (plan 4.2). `frozen_scope`
    (Phase 2) replaces the rule-derived scope; the rule stays a reported cross-check."""
    org = pinned.org
    g, vdb = load_pair(dirpath, pinned)
    nodes_meta = {nid: n.metadata for nid, n in g.nodes.items()}
    existing_ids = set(g.nodes)
    content = vdb.content
    vdb.embeddings = {}                      # free the vectors; only content is needed from here
    derived = derive_scope(nodes_meta, scope_min_len)
    scope = list(frozen_scope) if frozen_scope is not None else derived
    missing = [i for i in scope if i not in nodes_meta or not isinstance(nodes_meta[i].get("want_text"), str)]
    if missing:
        raise Stop("scope: %d id(s) of S are not want nodes in the graph (first: %s)" % (len(missing), missing[0]))
    cl = Classifier(org, nodes_meta, content)
    records = [cl.classify_node(nid) for nid in scope]
    dropped = apply_collision_rule(records, existing_ids)
    attach_excerpt_hashes(org, content, records)
    for r in records:
        if r["disposition"] == "candidate":
            r["flags"]["collision_candidate"] = False
            r["flags"]["mention_shapes"] = mention_shape_flags(content[r["source_node"]], r["_w_open"], r["_closer_end"])
    cand_old = [r["id"] for r in records if r["disposition"] == "candidate"]
    watch = list(CHOICE_CLAUSE_IDS) + [CONSTITUTIONAL_ID]
    before_fig = incident_figures(g, set(cand_old) | set(watch))
    render_before = len(org.render_wants(g).encode("utf-8"))
    counts = {"nodes": len(g.nodes),
              "wants": sum(1 for md in nodes_meta.values() if md.get("kind") == "want"),
              "protected": sum(1 for md in nodes_meta.values() if is_protected(md))}
    del g, vdb
    gc.collect()
    raw_main = Path(os.path.join(dirpath, MAIN_NAME)).read_bytes()
    want_ids = {nid for nid, md in nodes_meta.items() if md.get("kind") == "want"}
    syn = synapse_stats(raw_main, want_ids, set(scope), set(cand_old))
    del raw_main
    A: Dict[str, Any] = {
        "dir": dirpath, "records": records, "dropped": dropped, "scope": scope, "scope_derived": derived,
        "scope_min_len": scope_min_len, "nodes_meta": nodes_meta, "content": content, "existing_ids": existing_ids,
        "cl": cl, "before_figures": before_fig, "render_len_before": render_before, "counts": counts, "syn": syn,
    }
    if full_reports:
        s_sources = sorted({r["source_node"] for r in records if r.get("source_node")})
        marker_nodes = conversational_marker_nodes(nodes_meta, content)
        A["s_sources"], A["marker_nodes"] = s_sources, marker_nodes
        A["histograms"] = reason_histograms(cl, s_sources, marker_nodes)
        mb = marker_bearing_minted(cl, marker_nodes, set(s_sources))
        for e in mb:
            e["shape_flags"] = mention_shape_flags(content[e["source_node"]], e["open_start"], e["close_end"])
        A["marker_bearing"] = mb
        A["residuals"] = residual_classes(pinned, base_mod or load_base_module(pinned), nodes_meta, content, marker_nodes, cl)
    return A


def outcome_table(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_outcome = Counter(r["outcome"] for r in records)
    by_disp = Counter(r["disposition"] for r in records)
    by_class = Counter("%s/%s" % (r["class"], r["outcome"]) for r in records)
    return {"scope_size": len(records), "by_outcome": dict(sorted(by_outcome.items())),
            "by_disposition": dict(sorted(by_disp.items())), "by_class_outcome": dict(sorted(by_class.items())),
            "accounting_sum": int(sum(by_outcome.values())),
            "assert_failed_ids": sorted(r["id"] for r in records if r["disposition"] == "assert_failed"),
            "collision_dropped": sorted(({"id": r["id"], "detail": r["detail"]} for r in records if r["disposition"] == "collision_dropped"),
                                        key=lambda x: x["id"]),
            "unchanged_listed": sorted(({"id": r["id"], "outcome": r["outcome"], "detail": r["detail"],
                                         "reasons": r.get("reasons", {})} for r in records
                                        if r["disposition"] == "unchanged" and r["outcome"] != "GENUINE"), key=lambda x: x["id"])}


def repair_list_obj(A: Dict[str, Any]) -> Dict[str, Any]:
    cands = []
    for r in sorted(A["records"], key=lambda x: x["id"]):
        if r["disposition"] != "candidate":
            continue
        cands.append({"id": r["id"], "outcome": r["outcome"], "class": r["class"], "old_len": r["old_len"],
                      "new_len": r["new_len"], "old_sha16": r["old_sha16"], "new_sha16": r["new_sha16"],
                      "excerpt_sha256": r["excerpt_sha256"], "flags": r["flags"], "source_node": r["source_node"],
                      "inner_open_rel": r["inner_open_rel"], "closer_rel": r["closer_rel"],
                      "removed_prefix_len": r["removed_prefix_len"]})
    return {"kind": "repair-list", "candidates": cands, "count": len(cands),
            "dropped": sorted(({"id": r["id"], "disposition": r["disposition"], "detail": r["detail"]}
                               for r in A["records"] if r["disposition"] in ("collision_dropped", "assert_failed")),
                              key=lambda x: x["id"])}


def scope_ids_obj(A: Dict[str, Any], expected: int) -> Dict[str, Any]:
    ids = sorted(A["scope"])
    return {"kind": "scope-ids", "ids": ids, "count": len(ids), "expected_count": expected,
            "matches_expected_count": len(ids) == expected, "scope_min_len_cross_check": A["scope_min_len"],
            "rule_derived_count": len(A["scope_derived"]), "rule_derived_equals_list": sorted(A["scope_derived"]) == ids}


def id_map_objs(mapping: Dict[str, str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    pairs = sorted(mapping.items())
    fwd = {"kind": "id-map", "count": len(pairs), "pairs": [[o, n] for o, n in pairs]}
    inv = {"kind": "id-map-inverse", "count": len(pairs), "pairs": [[n, o] for o, n in pairs]}
    return fwd, inv


def build_reports(A: Dict[str, Any], expected_scope: int) -> Dict[str, Dict[str, Any]]:
    return {
        "outcome-table": {"kind": "outcome-table", "counts": A["counts"], **outcome_table(A["records"])},
        "histograms": {"kind": "reason-histograms", **A["histograms"]},
        "marker-bearing-minted": {"kind": "marker-bearing-minted", "count": len(A["marker_bearing"]), "entries": A["marker_bearing"]},
        "residual-classes": {"kind": "residual-classes", **A["residuals"]},
        "repair-list": repair_list_obj(A),
        "scope-ids": scope_ids_obj(A, expected_scope),
        "pre-node-report": {"kind": "pre-node-report", **A["syn"]},
    }


# --------------------------------------------------------------------------------------------------
# the verifier V1-V19: mechanical; a failing check means NO write
# --------------------------------------------------------------------------------------------------

def diff_modulo(a: Any, b: Any, m: Dict[str, str]) -> Tuple[List[Tuple], Counter]:
    """The independent diff walker (V7): the paths where a and b differ EXCEPT a string in a that equals what the
    mapping sends it to in b (dict keys included, key order preserved). Returns (diffs, substitutions-by-top-field)."""
    diffs: List[Tuple] = []
    subs: Counter = Counter()

    def rec(x, y, path):
        if len(diffs) > 50:
            return
        top = path[0] if path else "<root>"
        if isinstance(x, dict) and isinstance(y, dict):
            xk, yk = list(x.keys()), list(y.keys())
            if [m.get(k, k) if isinstance(k, str) else k for k in xk] != yk:
                diffs.append(path + ("<keys>",))
                return
            for k1, k2 in zip(xk, yk):
                if k1 != k2:
                    subs[top if path else str(k2)] += 1
                rec(x[k1], y[k2], path + (k2,))
        elif isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
            if len(x) != len(y):
                diffs.append(path + ("<len>",))
                return
            for n, (u, v) in enumerate(zip(x, y)):
                rec(u, v, path + (n,))
        elif isinstance(x, str) and isinstance(y, str):
            if x != y:
                if m.get(x) == y:
                    subs[top] += 1
                else:
                    diffs.append(path)
        else:
            same = type(x) is type(y) and (x == y or (isinstance(x, float) and x != x and y != y))
            if not same:
                diffs.append(path)

    rec(a, b, ())
    return diffs, subs


class Verifier:
    """V1-V19 over (input bytes, output bytes, the plan). Every check records ok + a small detail (ids/counts)."""

    def __init__(self, pinned, A: Dict[str, Any], plan: Dict[str, Any], expect: Dict[str, int]):
        self.pinned, self.A, self.plan, self.expect = pinned, A, plan, expect
        self.results: List[Dict[str, Any]] = []
        self.walk: Dict[str, Any] = {}

    def check(self, name: str, ok: bool, detail: Any = None) -> bool:
        self.results.append({"check": name, "ok": bool(ok), "detail": detail if detail is not None else ""})
        return bool(ok)

    # ---- one lockstep pass over input and output: V1-V7, V15-V17 -----------------------------------
    def lockstep(self, raw_a: bytes, raw_b: bytes) -> None:
        import itertools
        m, org = self.plan["mapping"], self.pinned.org
        new_ids = set(m.values())
        W: Dict[str, Any] = {
            "nodes_a": 0, "nodes_b": 0, "wants_a": 0, "wants_b": 0, "prot_a": 0, "prot_b": 0,
            "structure_ok": True, "key_order_bad": [], "v3_bad": [], "v7_bad": [], "v15_bad": [],
            "cc_present": [], "syn_bad": [], "syn_a": 0, "syn_b": 0, "inc_a": Counter(), "inc_b": Counter(),
            "rim_bad": [], "rim_incident": 0, "rim_incident_mapped": 0, "rim_changed": 0, "he_bad": [],
            "misc_bad": [], "other_top_bad": [], "walker_subs": 0, "written": set(), "b_node_keys": set(),
            "v2_bad": [], "text_bad": [], "a_node_keys": set(),
        }
        prot_names = set(CHOICE_CLAUSE_IDS) | {CONSTITUTIONAL_ID}
        for a, b in itertools.zip_longest(iter_sections(raw_a), iter_sections(raw_b)):
            if a is None or b is None or a[0] != b[0]:
                W["structure_ok"] = False
                break
            sec, eka, ksa, vsa, vea = a
            _, ekb, ksb, vsb, veb = b
            if sec == _HEADER or eka is _HDR:
                if raw_a[ksa:vea] != raw_b[ksb:veb]:
                    W["structure_ok"] = False
                continue
            if eka is None:                                        # plain top-level key
                A_, B_ = raw_a[ksa:vea], raw_b[ksb:veb]
                if A_ == B_:
                    continue
                if sec in ID_BEARING_TOP:
                    d, s = diff_modulo(decode(raw_a[vsa:vea]), decode(raw_b[vsb:veb]), m)
                    W["walker_subs"] += int(sum(s.values()))
                    if d:
                        W["misc_bad"].append(sec)
                else:
                    W["other_top_bad"].append(sec)
                continue
            expect_key = m.get(eka, eka)
            if ekb != expect_key:
                W["key_order_bad"].append("%s:%s" % (sec, eka))
                continue
            da, db = decode(raw_a[vsa:vea]), decode(raw_b[vsb:veb])
            if sec == "nodes":
                mda, mdb = (da.get("metadata") or {}), (db.get("metadata") or {})
                W["nodes_a"] += 1
                W["nodes_b"] += 1
                W["wants_a"] += mda.get("kind") == "want"
                W["wants_b"] += mdb.get("kind") == "want"
                W["prot_a"] += is_protected(mda)
                W["prot_b"] += is_protected(mdb)
                W["b_node_keys"].add(ekb)
                W["a_node_keys"].add(eka)
                if ekb != eka:
                    W["walker_subs"] += 1                            # the map key itself
                diffs, s = diff_modulo(da, db, m)
                W["walker_subs"] += int(sum(s.values()))
                if eka in prot_names:
                    W["cc_present"].append(eka)
                if eka in m:
                    W["written"].add(eka)
                    ok3 = (diffs == [("metadata", "want_text")] and db.get("node_id") == m[eka]
                           and mdb.get("want_text") == self.plan["new_text"][eka]
                           and mda.get("poincare_dir") == mdb.get("poincare_dir")
                           and set(s) <= {"node_id", "pred_weights"})
                    if not ok3:
                        W["v3_bad"].append(eka)
                    if org.want_id_for_text(mdb.get("want_text", "")) != db.get("node_id"):
                        W["v2_bad"].append(eka)
                else:
                    if diffs or not set(s) <= {"pred_weights"}:
                        if mda.get("kind") == "want" or eka in prot_names:
                            W["v15_bad"].append(eka)
                        else:
                            W["v7_bad"].append(eka)
            elif sec == "synapses":
                W["syn_a"] += 1
                W["syn_b"] += 1
                pa, qa, pb, qb = da.get("pre_node_id"), da.get("post_node_id"), db.get("pre_node_id"), db.get("post_node_id")
                for x in (pa, qa):
                    if x in m:
                        W["inc_a"][x] += 1
                for x in (pb, qb):
                    if x in new_ids:
                        W["inc_b"][x] += 1
                diffs, s = diff_modulo(da, db, m)
                W["walker_subs"] += int(sum(s.values()))
                if diffs or not set(s) <= {"pre_node_id", "post_node_id", "metadata"}:
                    W["syn_bad"].append(eka)
                if CONSTITUTIONAL_ID in (pa, qa):
                    W["rim_incident"] += 1
                    other = qa if pa == CONSTITUTIONAL_ID else pa
                    rim_ok = (pb if pa == CONSTITUTIONAL_ID else qb) == CONSTITUTIONAL_ID
                    if other in m:
                        W["rim_incident_mapped"] += 1
                    if da != db:
                        W["rim_changed"] += 1
                    if diffs or not rim_ok:
                        W["rim_bad"].append(eka)
            else:                                                    # hyperedges / archived_hyperedges
                diffs, s = diff_modulo(da, db, m)
                W["walker_subs"] += int(sum(s.values()))
                if diffs:
                    W["he_bad"].append("%s:%s" % (sec, eka))
        self.walk = W

    def run(self, raw_a: bytes, out_main: str, sidecar_a: str, sidecar_b: str, *, copy_dir: str,
            copy_hashes: Dict[str, str], census_in: Dict[str, Counter], writer_stats: Dict[str, Any],
            sidecar_stats: Dict[str, Any], out_graph=None, before_sha_main: str = "") -> List[Dict[str, Any]]:
        A, plan, org = self.A, self.plan, self.pinned.org
        m, inv = plan["mapping"], plan["inverse"]
        raw_b = Path(out_main).read_bytes()
        self.lockstep(raw_a, raw_b)
        W = self.walk
        # V1 counts
        self.check("V1", W["nodes_a"] == W["nodes_b"] and W["wants_a"] == W["wants_b"] == self.expect["wants"]
                   and W["prot_a"] == W["prot_b"] == self.expect["protected"],
                   {"nodes": [W["nodes_a"], W["nodes_b"]], "wants": [W["wants_a"], W["wants_b"], self.expect["wants"]],
                    "protected": [W["prot_a"], W["prot_b"], self.expect["protected"]]})
        # V2 ids
        acct = int(sum(Counter(r["outcome"] for r in A["records"]).values()))
        new_present = all(n in W["b_node_keys"] for n in m.values())
        old_absent = all(o not in W["b_node_keys"] for o in m)
        self.check("V2", not W["v2_bad"] and new_present and old_absent and len(set(m.values())) == len(m)
                   and W["structure_ok"] and not W["key_order_bad"] and acct == len(A["scope"]),
                   {"id_not_following_text": W["v2_bad"][:5], "new_present": new_present, "old_absent": old_absent,
                    "injective": len(set(m.values())) == len(m), "positions_preserved": not W["key_order_bad"],
                    "accounting": [acct, len(A["scope"])]})
        self.check("V3", not W["v3_bad"], {"bad": W["v3_bad"][:5]})
        self.check("V4", not W["syn_bad"] and W["syn_a"] == W["syn_b"]
                   and all(W["inc_a"][o] == W["inc_b"][n] for o, n in m.items()),
                   {"synapses": [W["syn_a"], W["syn_b"]], "bad": W["syn_bad"][:5],
                    "incident_mismatch": [o for o, n in m.items() if W["inc_a"][o] != W["inc_b"][n]][:5]})
        self.check("V5", not W["he_bad"], {"bad": W["he_bad"][:5]})
        self.check("V6", not W["misc_bad"] and not W["other_top_bad"] and W["structure_ok"],
                   {"misc_bad": W["misc_bad"], "other_top_bad": W["other_top_bad"]})
        self.check("V7", not W["v7_bad"], {"bad": W["v7_bad"][:5]})
        # V8 sidecar + other files
        sa, sb = json.loads(sidecar_a), json.loads(sidecar_b)
        ea, eb = sa.get("entries", {}), sb.get("entries", {})
        sidecar_ok = (list(eb) == [m.get(k, k) for k in ea] and all(ea[k] == eb[m.get(k, k)] for k in ea)
                      and all(sa.get(k) == sb.get(k) for k in sa if k != "entries") and set(sa) == set(sb))
        others_ok = all(sha256_file(os.path.join(copy_dir, n)) == copy_hashes[n]
                        for n in (VECTORS_NAME, GUARD_NAME, MANIFEST_NAME, COMMONS_NAME))
        self.check("V8", sidecar_ok and others_ok, {"sidecar_entries": [len(ea), len(eb)], "other_files_unchanged": others_ok})
        # V9 verbatim and untruncated
        v9_bad = []
        for r in A["records"]:
            if r["id"] in m:
                X, T = r["_x"], r["_t"]
                C = A["content"][r["source_node"]]
                ok = (X in T and T.endswith(X) and X == C[r["_w_open"] + len(org.WANT_OPEN): r["_closer_end"] - len(org.WANT_CLOSE)].strip()
                      and T in C)
                if not ok:
                    v9_bad.append(r["id"])
        self.check("V9", not v9_bad, {"bad": v9_bad[:5], "no_length_assertion": True})
        # V10 census
        after_main = census_msgpack_file(out_main, list(m))
        n_in_main = int(sum(census_in[MAIN_NAME].values()))
        n_in_side = int(sum(census_in[SIDECAR_NAME].values()))
        outside = {n: int(sum(c.values())) for n, c in census_in.items() if n not in (MAIN_NAME, SIDECAR_NAME) and sum(c.values())}
        self.check("V10", not sum(after_main.values()) and not outside
                   and n_in_main == writer_stats["substitutions_total"] == W["walker_subs"]
                   and n_in_side == sidecar_stats["substitutions"],
                   {"census_in_main": n_in_main, "writer_substitutions": writer_stats["substitutions_total"],
                    "walker_substitutions": W["walker_subs"], "census_in_sidecar": n_in_side,
                    "sidecar_substitutions": sidecar_stats["substitutions"], "old_ids_left_in_output": int(sum(after_main.values())),
                    "occurrences_outside_S1_S13": outside})
        # V11 canonical restore
        g2 = out_graph
        v11 = {"restored": g2 is not None}
        ok11 = g2 is not None
        if g2 is not None:
            bad_fig, dangling = [], 0
            for o, n in m.items():
                fb = A["before_figures"].get(o)
                fa = (len(g2._outgoing.get(n, ())), len(g2._incoming.get(n, ())), len(g2._node_hyperedges.get(n, ())))
                if fb != fa:
                    bad_fig.append(o)
            for nid in g2.nodes:
                for sid in g2._outgoing.get(nid, ()):
                    s = g2.synapses.get(sid)
                    if s is None or s.pre_node_id != nid or s.post_node_id not in g2.nodes:
                        dangling += 1
            for he in g2.hyperedges.values():
                dangling += sum(1 for x in list(he.member_nodes) + list(he.output_targets) if x not in g2.nodes)
            render_after = len(org.render_wants(g2).encode("utf-8"))
            ok11 = not bad_fig and dangling == 0 and len(g2.nodes) == W["nodes_b"]
            v11.update({"figure_mismatch": bad_fig[:5], "dangling": dangling, "nodes": len(g2.nodes),
                        "render_wants_bytes_before_after": [A["render_len_before"], render_after]})
        self.check("V11", ok11, v11)
        # V12 idempotence (parse-derived id-follows-text on the source; old ids gone; identity rewrite is a no-op)
        cl = A["cl"]
        rer_bad = []
        for r in A["records"]:
            if r["id"] in m:
                wp = cl.parse(r["source_node"])
                if not any(w.want_id == m[r["id"]] and w.text == r["_x"] for w in wp.wants):
                    rer_bad.append(r["id"])
        ident_tmp = out_main + ".idem"
        rewrite_main(raw_b, ident_tmp, {}, {}, {})
        idem = Path(ident_tmp).read_bytes() == raw_b
        os.unlink(ident_tmp)
        self.check("V12", not rer_bad and idem, {"not_rederived": rer_bad[:5], "identity_rewrite_is_noop": idem})
        # V13 fidelity (each re-encoded value was proven inside the writer; the sidecar round-trips)
        self.check("V13", True, {"enforced_by": "writer-enforced: rewrite_main raises Stop on any pack(unpack(raw)) != raw and the "
                                 "sidecar must round-trip through json; this line records the writer's counts, it is not a second pass",
                                 "reencoded_entries": writer_stats["reencoded_entries"], "sidecar_round_trip": True})
        # V14 shared-function proof
        src = Path(__file__).read_text(encoding="utf-8")
        shared = (("def " + "parse_wants") not in src and ("def " + "want_id_for_text") not in src
                  and callable(org.parse_wants) and org.parse_wants.__module__ == "cc_ng_organism")
        self.check("V14", shared, {"p1": self.pinned.record["p1"], "isolation_file": self.pinned.record["isolation"]["cc_ng_organism_file"]})
        # V15 Choice Clause / constitutional / every unrepaired want, plus the deny-check
        try:
            dc = deny_check(A["scope"], m, plan["approved_ids"], nodes_meta=A["nodes_meta"])
        except Stop as exc:
            dc = {"clean": False, "error": str(exc)[:120]}
        prot = set(CHOICE_CLAUSE_IDS) | {CONSTITUTIONAL_ID}                     # le-029 C7: PRESENT in the input AND the output
        present_in, present_out = sorted(prot & W["a_node_keys"]), sorted(prot & W["b_node_keys"])
        self.check("V15", not W["v15_bad"] and bool(dc.get("clean")) and set(present_in) == prot and set(present_out) == prot,
                   {"bad": W["v15_bad"][:5], "present_in_input": present_in, "present_in_output": present_out,
                    "missing": sorted(prot - set(present_in) - set(present_out)), "deny_check": dc})
        # V16 rim
        rim_present = CONSTITUTIONAL_ID in W["a_node_keys"] and CONSTITUTIONAL_ID in W["b_node_keys"]
        self.check("V16", rim_present and not W["rim_bad"] and W["rim_changed"] == W["rim_incident_mapped"],
                   {"rim_present_in_input_and_output": rim_present,
                    "rim_incident": W["rim_incident"], "rim_incident_with_mapped_want": W["rim_incident_mapped"],
                    "rim_changed": W["rim_changed"], "rim_to_scope_total": A["syn"]["rim_to_scope"], "bad": W["rim_bad"][:5]})
        # V17 approvals
        self.check("V17", W["written"] == set(plan["write_ids"]) and set(plan["write_ids"]) <= set(plan["approved_ids"]),
                   {"written": len(W["written"]), "approved_and_passed": len(plan["write_ids"]),
                    "refused": plan["refused"][:10]})
        # V18 rollback artefacts: inverse o mapping == identity, both files exist and hash-verify
        art = plan.get("artifacts", {})
        v18 = bool(art) and all(os.path.isfile(p) and sha256_file(p) == h for p, h in art.values())   # the hash check is UNCONDITIONAL (c026-C3)
        self.check("V18", v18 and all(inv[m[o]] == o for o in m) and len(inv) == len(m), {"artifacts": sorted(art)})
        # V19 all-or-nothing evaluation
        self.check("V19", plan.get("evaluated") == len(A["scope"]) and plan.get("assert_failed_listed") ==
                   sum(1 for r in A["records"] if r["disposition"] == "assert_failed"),
                   {"evaluated": plan.get("evaluated"), "scope": len(A["scope"]), "assert_failed_listed": plan.get("assert_failed_listed")})
        return self.results

    def failed(self) -> List[str]:
        return [r["check"] for r in self.results if not r["ok"]]


# --------------------------------------------------------------------------------------------------
# P3: the behavioural fingerprint. Recorded outputs of the PINNED function over a fixed SYNTHETIC battery
# (the closer-after-backtick repro of plan 4.1 first). A parser that regresses any row REFUSES.
# --------------------------------------------------------------------------------------------------

_BT = chr(96)
FINGERPRINT: Tuple[Tuple[str, str, Tuple[str, ...], Tuple[Tuple[int, int], ...], Tuple[Tuple[str, int, str], ...]], ...] = (
    ("closer_after_backtick", "[WANT]check " + _BT + "foo()" + _BT + "[/WANT]",
     ("check " + _BT + "foo()" + _BT,), ((0, 26),), ()),
    ("plain_pair", "before [WANT]do the thing[/WANT] after", ("do the thing",), ((7, 32),), ()),
    ("nested_live_pair", "[WANT]swallowed prose [WANT]the real intent[/WANT]", ("the real intent",), ((22, 50),),
     (("[WANT]", 0, "opener_unclosed"),)),
    ("mention_in_inline_code", "see " + _BT + "[WANT]documented[/WANT]" + _BT + " in the spec", (), (),
     (("[WANT]", 5, "in_code_span"), ("[/WANT]", 21, "in_code_span"))),
    ("mention_in_fence", "text\n" + _BT * 3 + "\n[WANT]fenced[/WANT]\n" + _BT * 3 + "\nmore", (), (),
     (("[WANT]", 9, "in_fence"), ("[/WANT]", 21, "in_fence"))),
    ("quoted_token_hug", "use ('[WANT]') and ('[/WANT]') as markers", (), (),
     (("[WANT]", 6, "quoted"), ("[/WANT]", 21, "closer_without_opener"))),
    ("escaped_opener", "write \\[WANT]not a want[/WANT] here", (), (),
     (("[WANT]", 7, "escaped"), ("[/WANT]", 23, "closer_without_opener"))),
    ("json_string_literal", 'Use {"note": "[WANT]typed in json[/WANT]"} for it', (), (),
     (("[WANT]", 14, "in_json_string"), ("[/WANT]", 33, "in_json_string"))),
    ("url_glued_opener", "see https://x.org/[WANT]z[/WANT] ok", (), (),
     (("[WANT]", 18, "in_url"), ("[/WANT]", 25, "closer_without_opener"))),
    ("mention_pair_inside_real_want", "[WANT]see https://x.org/[WANT]z[/WANT] ok[/WANT]", ("see https://x.org/[WANT]z[/WANT] ok",),
     ((0, 48),), (("[WANT]", 24, "in_url"), ("[/WANT]", 31, "in_url"))),
    ("unclosed_opener", "a [WANT]never closed", (), (), (("[WANT]", 2, "opener_unclosed"),)),
    ("stray_closer", "a stray [/WANT] closer", (), (), (("[/WANT]", 8, "closer_without_opener"),)),
    ("empty_pair", "[WANT][/WANT]", (), (), (("[WANT]", 0, "empty_pair"), ("[/WANT]", 6, "empty_pair"))),
    ("two_wants", "[WANT]first[/WANT] and [WANT]second[/WANT]", ("first", "second"), ((0, 18), (23, 42)), ()),
    ("long_genuine", "[WANT]" + "w" * 700 + "[/WANT]", ("w" * 700,), ((0, 713),), ()),
    ("prose_comma_quote", 'In 2026, "[WANT]revisit the exit policy[/WANT]", I wrote.', ("revisit the exit policy",), ((10, 46),), ()),
)


def gate_p3(pinned, *, run_pinned_tests: bool = False, parse_fn=None) -> Dict[str, Any]:
    """P3. (a) the in-process fingerprint battery; (b) optionally the pinned test file itself, run as a
    subprocess from the pin tree with a clean environment (zero failures, zero errors, zero skips)."""
    fn = parse_fn or pinned.org.parse_wants
    bad = []
    for name, s, texts, spans, skips in FINGERPRINT:
        wp = fn(s)
        got = (tuple(w.text for w in wp.wants), tuple((w.open_start, w.close_end) for w in wp.wants),
               tuple((k.marker, k.start, k.reason) for k in wp.skipped))
        if got != (texts, spans, skips):
            bad.append(name)
    out: Dict[str, Any] = {"gate": "P3", "battery_rows": len(FINGERPRINT), "mismatched_rows": bad, "ok": not bad}
    if run_pinned_tests and not bad:
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "NG_EMBED_REMOTE")}
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        p = subprocess.run([sys.executable, "-B", "-m", "pytest", TEST_FILE_RELPATH, "-q", "-p", "no:cacheprovider"],
                           cwd=pinned.root, env=env, capture_output=True, text=True)
        tail = (p.stdout or "").strip().splitlines()[-1:] or [""]
        out["pinned_tests"] = {"returncode": p.returncode, "summary": tail[0][:200]}
        out["ok"] = p.returncode == 0 and " failed" not in tail[0] and " error" not in tail[0] and " skipped" not in tail[0]
    return out


# --------------------------------------------------------------------------------------------------
# mechanical probes for P4 (daemon down, plan 6.4) and P6 (peer hold H1-H3, plan 6.5). Real by default;
# every one is a read-only query. Tests inject fakes - nothing here is ever run by a Phase-1 step.
# --------------------------------------------------------------------------------------------------

DAEMON_UNIT = "cc-ng-daemon.service"
RECOVER_TIMER = "cc-ng-daemon-recover.timer"
LEG2_TIMER = "cc-callosum-leg2.timer"
LEG2_SERVICE = "cc-callosum-leg2.service"
DAEMON_PROC_PATTERNS = ("cc-ng-service.py", "cc-ng-daemon.py", "neurograph_rpc.py")
SYNC_PROC_PATTERNS = ("cc-ng-sync.py leg2-tick", "cc_topology_merge", "cc-ng-sync.py")


class ProbeError(Exception):
    """A host query could not be answered: the truth is UNKNOWN. A gate must read this as 'not satisfied' -
    never as 'down', 'off', 'inactive' or 'no cron line' (le-029 C2 / checker-026 c026-C7)."""


_ACTIVE_WORDS = ("active", "activating", "reloading", "deactivating", "refreshing", "maintenance")
_DOWN_WORDS = ("inactive", "failed")
_ENABLED_WORDS = ("enabled", "enabled-runtime", "static", "linked", "linked-runtime", "alias", "indirect", "generated", "transient")
_OFF_WORDS = ("disabled", "masked", "masked-runtime")
PROBE_TIMEOUT_S = 20


class Probes:
    """Read-only host queries that FAIL CLOSED. `systemctl --user` counts as 'down' only for an exact
    inactive/failed answer (rc 3/4) from a REACHABLE bus; an empty answer, any other word, a non-zero rc that does
    not fit, an unreachable bus, a timeout or a missing binary raises ProbeError. `crontab -l`: rc 0 = the text,
    rc 1 with 'no crontab for' = empty, anything else raises. The /proc scans skip an entry only if it VANISHED
    mid-scan or belongs to another user (counted in foreign_unreadable); an unreadable entry of our own uid raises.
    Tests inject fakes - nothing here is ever run by a Phase-1 step."""

    def __init__(self):
        self.foreign_unreadable = 0

    def _run(self, argv):
        try:
            return subprocess.run(argv, capture_output=True, text=True, timeout=PROBE_TIMEOUT_S)
        except (OSError, subprocess.SubprocessError) as exc:            # FileNotFoundError, PermissionError, TimeoutExpired ...
            raise ProbeError("%s: %s" % (argv[0], exc.__class__.__name__))

    def _sysctl(self, *a):
        p = self._run(["systemctl", "--user", *a])
        if "Failed to connect to bus" in (p.stderr or ""):
            raise ProbeError("systemctl --user: the user bus is unreachable")
        return p

    def unit_active(self, unit: str) -> bool:
        p = self._sysctl("is-active", unit)
        word = (p.stdout or "").strip()
        if word in _DOWN_WORDS and p.returncode in (3, 4):
            return False
        if word in _ACTIVE_WORDS:
            return True
        raise ProbeError("systemctl is-active %s: rc %s, answer %r is neither up nor down" % (unit, p.returncode, word[:40]))

    def unit_enabled(self, unit: str) -> bool:
        p = self._sysctl("is-enabled", unit)
        word = (p.stdout or "").strip()
        if word in _ENABLED_WORDS:
            return True
        if word in _OFF_WORDS:
            return False
        raise ProbeError("systemctl is-enabled %s: rc %s, answer %r is neither on nor off" % (unit, p.returncode, word[:40]))

    def _foreign(self, d: str) -> Optional[bool]:
        try:
            return os.stat(d).st_uid != os.getuid()
        except (FileNotFoundError, ProcessLookupError):
            return None                                                    # the process vanished

    def processes_matching(self, patterns) -> List[int]:
        pids = []
        me = os.getpid()
        for d in glob.glob("/proc/[0-9]*"):
            try:
                pid = int(os.path.basename(d))
            except ValueError:
                continue
            if pid == me:
                continue
            try:
                cmd = Path(d, "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
            except (FileNotFoundError, ProcessLookupError, NotADirectoryError):
                continue                                                    # vanished mid-scan
            except OSError as exc:
                foreign = self._foreign(d)
                if foreign is None:
                    continue
                if foreign:
                    self.foreign_unreadable += 1
                    continue
                raise ProbeError("cannot read %s/cmdline (our own uid): %s" % (d, exc.__class__.__name__))
            if any(pat in cmd for pat in patterns) and "want_text_repair_oneshot" not in cmd:
                pids.append(pid)
        return sorted(pids)

    def files_held_open(self, paths) -> List[Tuple[int, str]]:
        want = {os.path.realpath(p) for p in paths}
        held = []
        for d in glob.glob("/proc/[0-9]*"):
            try:
                pid = int(os.path.basename(d))
            except ValueError:
                continue
            if pid == os.getpid():
                continue
            try:
                names = os.listdir(os.path.join(d, "fd"))
            except (FileNotFoundError, ProcessLookupError, NotADirectoryError):
                continue
            except OSError as exc:
                foreign = self._foreign(d)
                if foreign is None:
                    continue
                if foreign:
                    self.foreign_unreadable += 1
                    continue
                raise ProbeError("cannot list %s/fd (our own uid): %s" % (d, exc.__class__.__name__))
            for fd in names:
                try:
                    t = os.path.realpath(os.readlink(os.path.join(d, "fd", fd)))
                except OSError:
                    continue                                                # that descriptor closed while we looked
                if t in want:
                    held.append((pid, t))
        return held

    def pid_alive(self, pid: int) -> bool:
        return os.path.isdir("/proc/%d" % pid)

    def crontab_text(self) -> str:
        p = self._run(["crontab", "-l"])
        if p.returncode == 0:
            return p.stdout
        if p.returncode == 1 and "no crontab for" in (p.stderr or ""):
            return ""
        raise ProbeError("crontab -l: rc %s, %r" % (p.returncode, (p.stderr or "")[:60]))

    def env_get(self, name: str) -> Optional[str]:
        return os.environ.get(name)

    def stat_snapshot(self, dirpath: str) -> Dict[str, List[int]]:
        """names, sizes, mtimes, inodes - by stat only; the conduit files are NEVER opened (H3). An unreadable
        conduit is an error, not an empty conduit."""
        snap: Dict[str, List[int]] = {}
        if not dirpath or not os.path.isdir(dirpath):
            return snap

        def boom(exc):
            raise ProbeError("cannot walk the conduit: %s" % exc.__class__.__name__)
        for root, _dirs, files in os.walk(dirpath, onerror=boom):
            for n in sorted(files):
                p = os.path.join(root, n)
                try:
                    st = os.stat(p)
                except FileNotFoundError:
                    continue
                except OSError as exc:
                    raise ProbeError("cannot stat a conduit file: %s" % exc.__class__.__name__)
                snap[os.path.relpath(p, dirpath)] = [st.st_size, int(st.st_mtime_ns), st.st_ino]
        return snap


def _leg(checks: Dict[str, bool], errors: Dict[str, str], name: str, fn: Callable[[], Any]) -> None:
    try:
        checks[name] = bool(fn())
    except ProbeError as exc:
        checks[name] = False                                                # unknown is NOT satisfied
        errors[name] = str(exc)[:160]


def stat_ident(path: str) -> Dict[str, int]:
    """(st_dev, st_ino, st_nlink) - the inode evidence of Exec P428. A file identity is the (dev, ino) pair; a link
    count says only how many names an inode has, never which files share it."""
    st = os.stat(path)
    return {"st_dev": st.st_dev, "st_ino": st.st_ino, "st_nlink": st.st_nlink}


def _files_equal_backup(target_dir: str, manifest_files: Dict[str, Dict[str, Any]]) -> bool:
    for n in SIX_FILES:
        p = os.path.join(target_dir, n)
        st = os.stat(p)
        m = manifest_files.get(n, {})
        if not (m.get("sha256") == sha256_file(p) and m.get("size") == st.st_size and m.get("mtime_ns") == st.st_mtime_ns
                and m.get("st_dev") == st.st_dev and m.get("st_ino") == st.st_ino):
            return False
    return True


def gate_p4(probes, target_dir: str, *, code_placed_at: Optional[float], daemon_log: Optional[str],
            manifest_files: Optional[Dict[str, Dict[str, Any]]] = None, require_files_equal: bool = True) -> Dict[str, Any]:
    """P4 - 'daemon down' is a MECHANICAL check (plan 6.4), all of it, recorded. EVERY leg is always present in
    `checks`: a missing --code-placed-at, an absent or unreadable daemon.log, an unanswerable probe, or a missing
    manifest is a leg that is False - never a leg that is left out (le-029 C1/C2). `require_files_equal=False`
    (the rollback step only) records the equality leg under `skipped`, with the reason, instead of dropping it."""
    files = [os.path.join(target_dir, n) for n in SIX_FILES]
    checks: Dict[str, bool] = {}
    errors: Dict[str, str] = {}
    skipped: Dict[str, str] = {}
    _leg(checks, errors, "unit_inactive", lambda: not probes.unit_active(DAEMON_UNIT))
    _leg(checks, errors, "recover_timer_inactive", lambda: not probes.unit_active(RECOVER_TIMER))
    _leg(checks, errors, "no_daemon_process", lambda: probes.processes_matching(DAEMON_PROC_PATTERNS) == [])
    _leg(checks, errors, "no_process_holds_the_six_files", lambda: probes.files_held_open(files) == [])

    def pid_leg():
        pidf = os.path.join(target_dir, "..", "daemon.pid")
        if not os.path.isfile(pidf):
            return True
        try:
            return not probes.pid_alive(int(Path(pidf).read_text().strip()))
        except ValueError:
            return False
    _leg(checks, errors, "daemon_pid_file_names_a_dead_pid", pid_leg)
    if require_files_equal:
        checks["six_files_equal_the_start_of_phase2_backup"] = manifest_files is not None and _files_equal_backup(target_dir, manifest_files)
    else:
        skipped["six_files_equal_the_start_of_phase2_backup"] = "rollback: replaced by the identity check against the backup and the post-apply receipt"

    def pulse_ok() -> bool:
        if code_placed_at is None or not daemon_log:
            return False
        try:
            with open(daemon_log, "rb") as f:
                f.read(1)                                                   # a log that cannot be read is not evidence
            return bool(checks.get("unit_inactive")) and os.stat(daemon_log).st_mtime <= float(code_placed_at)
        except (OSError, ValueError):
            return False
    checks["no_pulse_since_code_placement"] = pulse_ok()
    return {"gate": "P4", "checks": checks, "probe_errors": errors, "skipped_legs": skipped, "ok": all(checks.values())}


def hold_snapshot(probes, conduit_dir: str) -> Dict[str, Any]:
    """P6 H1-H3, recorded (plan 6.5). H1: the callosum cron line and the Leg 1 flag. H2: no Leg 2 timer/process.
    H3: the conduit compared by stat only. A probe that cannot answer is recorded in `probe_errors` (and fails the gate)."""
    errs: Dict[str, str] = {}

    def q(name, fn, default):
        try:
            return fn()
        except ProbeError as exc:
            errs[name] = str(exc)[:160]
            return default
    cron = q("h1_crontab", probes.crontab_text, "")
    live_cron = [ln for ln in cron.splitlines() if "callosum" in ln.lower() and not ln.lstrip().startswith("#")]
    return {
        "h1_crontab_sha256": sha256_bytes(cron.encode()),
        "h1_callosum_cron_line_firing": bool(live_cron),
        "h1_leg1_flag_env_is_1": probes.env_get("CC_CALLOSUM_LEG1_ENABLED") == "1",
        "h2_leg2_timer_enabled": q("h2_leg2_timer_enabled", lambda: probes.unit_enabled(LEG2_TIMER), False),
        "h2_leg2_timer_active": q("h2_leg2_timer_active", lambda: probes.unit_active(LEG2_TIMER), False),
        "h2_leg2_service_active": q("h2_leg2_service_active", lambda: probes.unit_active(LEG2_SERVICE), False),
        "h2_sync_or_merge_process": q("h2_sync_or_merge_process", lambda: probes.processes_matching(SYNC_PROC_PATTERNS), []),
        "h3_conduit_dir": conduit_dir,
        "h3_conduit_stat": q("h3_conduit_stat", lambda: probes.stat_snapshot(conduit_dir), {}),
        "probe_errors": errs,
    }


def gate_p6(probes, conduit_dir: str, start: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    now = hold_snapshot(probes, conduit_dir)
    bad = [k for k in ("h1_callosum_cron_line_firing", "h1_leg1_flag_env_is_1", "h2_leg2_timer_enabled",
                       "h2_leg2_timer_active", "h2_leg2_service_active") if now[k]]
    if now["h2_sync_or_merge_process"]:
        bad.append("h2_sync_or_merge_process")
    bad += ["probe_error:%s" % k for k in now["probe_errors"]]
    if not conduit_dir or _under(conduit_dir, SYL_CHECKPOINTS) or _under(conduit_dir, os.path.join(_HOME, "NeuroGraph", "data")):
        bad.append("h3_conduit_path_not_recorded_or_under_syls_directories")
    if start is not None:
        if now["h1_crontab_sha256"] != start["h1_crontab_sha256"]:
            bad.append("h1_crontab_changed_since_start")
        if now["h3_conduit_stat"] != start["h3_conduit_stat"]:
            bad.append("h3_conduit_stat_changed_since_start")
    return {"gate": "P6", "snapshot": now, "violations": bad, "ok": not bad}


def gate_p2(daemon_organism_file: str, pin_root: str, tool_root: Optional[str] = None) -> Dict[str, Any]:
    """P2 (le-029 C3). The sha256 of the file the daemon imports must equal the pin - AND the path must be the
    unit's own import root: a file that resolves inside the PIN worktree or this tool's worktree satisfies the hash
    vacuously and is REFUSED. The file's realpath / inode / mtime are recorded. The Chief names the actual import
    root (Q9) in the go request."""
    tool_root = tool_root or _tool_worktree_root()
    out: Dict[str, Any] = {"gate": "P2", "path": daemon_organism_file, "ok": False}
    if not daemon_organism_file or not os.path.isfile(daemon_organism_file):
        out["reason"] = "not a readable file"
        return out
    real = os.path.realpath(daemon_organism_file)
    st = os.stat(real)
    out.update(realpath=real, st_ino=st.st_ino, st_dev=st.st_dev, mtime_ns=st.st_mtime_ns, sha256=sha256_file(real))
    if _under(real, pin_root) or _under(real, tool_root):
        out["reason"] = "resolves inside the pin worktree or the tool worktree - name the unit's actual import root"
        return out
    out["ok"] = out["sha256"] == PIN["cc_ng_organism_sha256"]
    if not out["ok"]:
        out["reason"] = "sha256 is not the pin"
    return out


def _tool_worktree_root() -> str:
    here = os.path.dirname(os.path.realpath(__file__))
    p = subprocess.run(["git", "-C", here, "rev-parse", "--show-toplevel"], capture_output=True, text=True)
    return os.path.realpath(p.stdout.strip()) if p.returncode == 0 and p.stdout.strip() else os.path.realpath(os.path.join(here, "..", "..", ".."))


# --------------------------------------------------------------------------------------------------
# the retirement mechanism (plan 6.8, LAW 3): the tool is written for ONE repair
# --------------------------------------------------------------------------------------------------

def tool_sha256() -> str:
    return sha256_file(os.path.realpath(__file__))


def find_retired_receipts(target_real: str) -> List[str]:
    """Refuse if ANY RETIRED-*.receipt exists for the same checkpoint directory, or if the tool's own sha256
    equals the sha256 recorded in one. An unreadable receipt blocks (fail closed)."""
    hits = []
    me = tool_sha256()
    for p in sorted(glob.glob(os.path.join(BACKUPS_ROOT, RUN_DIR_PREFIX + "*", "RETIRED-*.receipt"))):
        try:
            rec = json.loads(Path(p).read_text())
        except (OSError, ValueError):
            hits.append(p)
            continue
        if rec.get("checkpoint_dir") == target_real or rec.get("tool_sha256") == me:
            hits.append(p)
    return hits


def refuse_if_retired(target_real: str) -> None:
    hits = find_retired_receipts(target_real)
    if hits:
        raise Refusal("6.8: a RETIRED receipt exists (%s); the tool is one-shot - no reuse; a future re-key is "
                      "protected-file work and goes to Josh" % os.path.basename(hits[0]))


def write_retired_receipt(run_dir: str, target_real: str, mapping_sha256: str, utc: str) -> str:
    body = {"kind": "RETIRED", "retired_at_utc": utc, "checkpoint_dir": target_real, "tool_sha256": tool_sha256(),
            "mapping_sha256": mapping_sha256, "notice": "no reuse; a future re-key is protected-file work and goes to Josh then"}
    path = os.path.join(run_dir, "RETIRED-%s.receipt" % utc)
    write_artifact(path, body)
    return path


# --------------------------------------------------------------------------------------------------
# the run: copy step, the shared build/verify core, Phase 1 stages
# --------------------------------------------------------------------------------------------------

def new_run_dir() -> Tuple[str, str]:
    """A fresh <backups>/z12-want-text-repair-<UTC>/ ; a second run in the same second gets a -N suffix."""
    utc = utc_stamp()
    for n in range(1, 100):
        tag = utc if n == 1 else "%s-%d" % (utc, n)
        path = guard_out_path(os.path.join(BACKUPS_ROOT, RUN_DIR_PREFIX + tag))
        try:
            os.makedirs(path, mode=0o700, exist_ok=False)
            return path, tag
        except FileExistsError:
            continue
    raise Stop("could not create a fresh run directory")


def copy_six(src_dir: str, dst_dir: str) -> Dict[str, Any]:
    """READ the six files from src (never written), copy each to a NEW inode under the run dir, sha256 before
    and after the copy and of the copy: all three must agree or the run STOPS."""
    def stat_row(p):
        st = os.stat(p)
        return {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "inode": st.st_ino,
                "st_dev": st.st_dev, "st_ino": st.st_ino, "st_nlink": st.st_nlink}
    before = {}
    for n in SIX_FILES:
        p = os.path.join(src_dir, n)
        if not os.path.isfile(p):
            raise Stop("copy: %s is missing from the target directory" % n)
        before[n] = {"sha256": sha256_file(p), **stat_row(p)}
    guard_out_path(dst_dir)
    os.makedirs(dst_dir, mode=0o700, exist_ok=True)
    copies = {}
    for n in SIX_FILES:
        d = guard_out_path(os.path.join(dst_dir, n))
        refuse_inplace_write(d)
        shutil.copyfile(os.path.join(src_dir, n), d)       # a new file, never a link
        if os.stat(d).st_ino == before[n]["inode"]:
            raise Stop("copy: %s shares an inode with the source" % n)
        copies[n] = sha256_file(d)
    after = {n: sha256_file(os.path.join(src_dir, n)) for n in SIX_FILES}
    for n in SIX_FILES:
        if not (before[n]["sha256"] == copies[n] == after[n]):
            raise Stop("copy: sha256 disagree for %s (before/copy/after)" % n)
    return {"source_dir": src_dir, "files": before, "copy_sha256": copies, "source_sha256_after": after,
            "source_unchanged": True}


def artifact_sha256(obj: Dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(stamped(obj)))


def _write_reports(run_dir: str, reports: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    return {n: write_artifact(os.path.join(run_dir, "reports", n + ".json"), o) for n, o in reports.items()}


def build_outputs(pinned, A: Dict[str, Any], approvals: Dict[str, Any], *, in_dir: str, out_dir: str, run_dir: str,
                  utc: str, in_hashes: Dict[str, str], provisional_ok: bool = False) -> Dict[str, Any]:
    """The build core shared by Phase 1 (rewrite to TMP) and Phase 2 (stage). Every node of S has ALREADY been
    evaluated (A) - no byte is written before that (V19). Writes the rewritten pair, the mapping + INVERSE
    mapping and the post-apply receipt draft; returns everything the verifier needs."""
    records = A["records"]
    write_ids, refused = gate_write_set(records, approvals, provisional_ok=provisional_ok)
    ok_dec = approved_decisions(approvals, provisional_ok)
    approved_ids = sorted(e["id"] for e in approvals["entries"] if e["decision"] in ok_dec)
    mapping = build_mapping(records, only_ids=set(write_ids))
    deny_check(A["scope"], mapping, [e["id"] for e in approvals["entries"]], nodes_meta=A["nodes_meta"])
    by_id = {r["id"]: r for r in records}
    old_text = {o: by_id[o]["_t"] for o in mapping}
    new_text = {o: by_id[o]["_x"] for o in mapping}
    inverse = {n: o for o, n in mapping.items()}
    census_in = census_set(in_dir, list(mapping))
    outside = {n: int(sum(c.values())) for n, c in census_in.items() if n not in (MAIN_NAME, SIDECAR_NAME) and sum(c.values())}
    if outside:
        raise Stop("census: old-id occurrences in files outside S1-S13: %s" % outside)
    raw = Path(os.path.join(in_dir, MAIN_NAME)).read_bytes()
    sidecar_in = Path(os.path.join(in_dir, SIDECAR_NAME)).read_text(encoding="utf-8")
    os.makedirs(guard_out_path(out_dir), mode=0o700, exist_ok=True)
    out_main, out_side = os.path.join(out_dir, MAIN_NAME), os.path.join(out_dir, SIDECAR_NAME)
    wstats = rewrite_main(raw, out_main, mapping, old_text, new_text)
    stext, sstats = rewrite_sidecar(sidecar_in, mapping)
    out_write_bytes(out_side, stext.encode("utf-8"))
    g2 = pinned.nf.Graph()
    g2.restore(out_main)                                   # the canonical restore IS the verifier (V11)
    meta_after = {nid: n.metadata for nid, n in g2.nodes.items()}
    protected_after = sorted(nid for nid, md in meta_after.items() if is_protected(md))
    fwd, invo = id_map_objs(mapping)
    fwd_sha = write_artifact(os.path.join(run_dir, "id-map-%s.json" % utc), fwd)
    inv_sha = write_artifact(os.path.join(run_dir, "id-map-inverse-%s.json" % utc), invo)
    out_hashes = {n: (sha256_file(out_main) if n == MAIN_NAME else sha256_file(out_side) if n == SIDECAR_NAME else in_hashes[n])
                  for n in SIX_FILES}
    receipt_obj = {"kind": "post-apply-receipt-DRAFT", "six_file_sha256_after": out_hashes,
                   "protected_id_set": protected_after, "protected_count": len(protected_after),
                   "mapping_sha256": fwd_sha, "inverse_mapping_sha256": inv_sha, "tool_sha256": tool_sha256(),
                   "function_pin": dict(PIN), "approvals_packet": approvals.get("packet")}
    rec_sha = write_artifact(os.path.join(run_dir, "post-apply-receipt-%s.json" % utc), receipt_obj)
    artifacts = {"id-map": (os.path.join(run_dir, "id-map-%s.json" % utc), fwd_sha),
                 "id-map-inverse": (os.path.join(run_dir, "id-map-inverse-%s.json" % utc), inv_sha),
                 "post-apply-receipt": (os.path.join(run_dir, "post-apply-receipt-%s.json" % utc), rec_sha)}
    plan = {"mapping": mapping, "inverse": inverse, "old_text": old_text, "new_text": new_text, "write_ids": write_ids,
            "approved_ids": approved_ids, "refused": refused, "packet": approvals.get("packet"),
            "evaluated": len(records), "assert_failed_listed": len(outcome_table(records)["assert_failed_ids"]),
            "artifacts": artifacts}
    return {"plan": plan, "raw": raw, "sidecar_in": sidecar_in, "sidecar_out": stext, "out_main": out_main,
            "out_sidecar": out_side, "writer_stats": wstats, "sidecar_stats": sstats, "census_in": census_in,
            "g2": g2, "meta_after": meta_after, "in_dir": in_dir, "in_hashes": in_hashes, "out_hashes": out_hashes,
            "id_map_sha256": fwd_sha}


def run_verifier(pinned, A: Dict[str, Any], ctx: Dict[str, Any], expect: Dict[str, int], *, out_main: Optional[str] = None,
                 sidecar_out: Optional[str] = None, out_graph: Any = "ctx", plan: Optional[Dict[str, Any]] = None) -> "Verifier":
    """V1-V19 over a built pair. The overrides exist so a negative test can hand the verifier a TAMPERED output."""
    V = Verifier(pinned, A, plan or ctx["plan"], expect)
    V.run(ctx["raw"], out_main or ctx["out_main"], ctx["sidecar_in"], sidecar_out if sidecar_out is not None else ctx["sidecar_out"],
          copy_dir=ctx["in_dir"], copy_hashes=ctx["in_hashes"], census_in=ctx["census_in"], writer_stats=ctx["writer_stats"],
          sidecar_stats=ctx["sidecar_stats"], out_graph=ctx["g2"] if out_graph == "ctx" else out_graph)
    return V


def prepare_outputs(pinned, A: Dict[str, Any], approvals: Dict[str, Any], *, in_dir: str, out_dir: str, run_dir: str,
                    utc: str, expect: Dict[str, int], in_hashes: Dict[str, str], provisional_ok: bool = False) -> Dict[str, Any]:
    """Build + verify V1-V19 + the T6 replay. Returns the plan, results, paths and stats."""
    ctx = build_outputs(pinned, A, approvals, in_dir=in_dir, out_dir=out_dir, run_dir=run_dir, utc=utc, in_hashes=in_hashes,
                        provisional_ok=provisional_ok)
    V = run_verifier(pinned, A, ctx, expect)
    cl_after = Classifier(pinned.org, ctx["meta_after"], A["content"])
    t6 = t6_replay(pinned, A["nodes_meta"], ctx["meta_after"], A["content"], ctx["plan"]["mapping"], cl_after)
    ctx.pop("g2")
    gc.collect()
    return {"plan": ctx["plan"], "results": V.results, "failed": V.failed(), "out_main": ctx["out_main"],
            "out_sidecar": ctx["out_sidecar"], "writer_stats": ctx["writer_stats"], "sidecar_stats": ctx["sidecar_stats"],
            "t6": t6, "walk_counts": {"rim_incident": V.walk["rim_incident"], "rim_incident_mapped": V.walk["rim_incident_mapped"]},
            "out_hashes": ctx["out_hashes"], "id_map_sha256": ctx["id_map_sha256"]}


def _expect(args) -> Dict[str, int]:
    return {"wants": args.expect_wants, "protected": args.expect_protected, "scope": args.expect_scope}


def _run_record(run_dir, utc, args, pinned, target_info, copy, extra) -> Dict[str, Any]:
    body = {"kind": "run-record", "utc": utc, "step": args.step, "tool": {"name": TOOL_NAME, "version": TOOL_VERSION, "sha256": tool_sha256()},
            "target": target_info, "copy": {"source_dir": copy["source_dir"], "copy_sha256": copy["copy_sha256"],
                                            "source_unchanged": copy["source_unchanged"]},
            "p1": pinned.record["p1"], "isolation": pinned.record["isolation"], **extra}
    return body


def stage_classify(args, pinned, target_info) -> Dict[str, Any]:
    run_dir, utc = new_run_dir()
    copy_dir = os.path.join(run_dir, "copy")
    copy = copy_six(target_info["target_realpath"], copy_dir)
    write_artifact(os.path.join(run_dir, "copy-hashes.json"), {"kind": "copy-hashes", "files": copy["files"], "copy_sha256": copy["copy_sha256"]})
    A = analyze(pinned, copy_dir, scope_min_len=args.scope_min_len)
    deny_check(A["scope"], build_mapping(A["records"]), (), nodes_meta=A["nodes_meta"])
    reports = build_reports(A, args.expect_scope)
    shas = _write_reports(run_dir, reports)
    cand_map, _ = id_map_objs(build_mapping(A["records"]))
    # the name Phase 2 expects: freeze reports/repair-list.json, reports/scope-ids.json and reports/id-map.json UNEDITED
    shas["id-map"] = write_artifact(os.path.join(run_dir, "reports", "id-map.json"), cand_map)
    shas.update(write_review_files(run_dir, utc, pinned.org, A["content"], A["records"]))
    ot = reports["outcome-table"]
    rec = _run_record(run_dir, utc, args, pinned, target_info, copy, {"artifacts_sha256": shas, "counts": A["counts"],
                      "scope_size": len(A["scope"]), "scope_matches_expected": len(A["scope"]) == args.expect_scope})
    write_artifact(os.path.join(run_dir, "run-record.json"), rec)
    return {"run_dir": run_dir, "utc": utc, "outcomes": ot["by_outcome"], "dispositions": ot["by_disposition"],
            "scope_size": len(A["scope"]), "candidates": reports["repair-list"]["count"],
            "repair_list_sha256": shas["repair-list"], "scope_ids_sha256": shas["scope-ids"], "artifacts_sha256": shas}


def _load_run(run_dir: str) -> Tuple[str, Dict[str, Any]]:
    run_dir = guard_out_path(run_dir)
    rec = load_artifact(os.path.join(run_dir, "run-record.json"))
    return run_dir, rec


def stage_rewrite(args, pinned, target_info) -> Dict[str, Any]:
    """Phase 1 step 2 (on the COPY): re-derive the classification, prove it is the saved one (P5), gate every
    id by the approvals (or, for the dry run only, a PROVISIONAL self-approval that Phase 2 refuses), rewrite to
    TMP files, run V1-V19 and the T6 replay, write the stamped reports. Nothing live is touched."""
    if not args.run_dir:
        raise Refusal("--step rewrite needs --run-dir (the classify run)")
    run_dir, rec = _load_run(args.run_dir)
    copy_dir = os.path.join(run_dir, "copy")
    chash = load_artifact(os.path.join(run_dir, "copy-hashes.json"))["copy_sha256"]
    for n in SIX_FILES:
        if sha256_file(os.path.join(copy_dir, n)) != chash[n]:
            raise Stop("the run's copy of %s no longer matches its recorded sha256" % n)
    scope = load_artifact(os.path.join(run_dir, "reports", "scope-ids.json"))["ids"]
    A = analyze(pinned, copy_dir, scope_min_len=args.scope_min_len, frozen_scope=scope, full_reports=False)
    rl_sha, sc_sha = artifact_sha256(repair_list_obj(A)), artifact_sha256(scope_ids_obj(A, args.expect_scope))
    if rl_sha != rec["artifacts_sha256"]["repair-list"] or sc_sha != rec["artifacts_sha256"]["scope-ids"]:
        raise Stop("P5: the re-derived classification is not the saved one (repair-list/scope-ids sha256 differ)")
    utc = utc_stamp()
    if args.provisional_approve_all:
        prov = provisional_body_for(A["records"], rl_sha, sc_sha)
        write_artifact(os.path.join(run_dir, "reports", "PROVISIONAL-approvals-%s.json" % utc), prov)
        approvals = stamped(prov)
    else:
        approvals = load_approvals(args.approvals, args.approvals_sha256, rl_sha, sc_sha)
    out = prepare_outputs(pinned, A, approvals, in_dir=copy_dir, out_dir=os.path.join(run_dir, "rewrite-tmp"), run_dir=run_dir,
                          utc=utc, expect=_expect(args), in_hashes=chash, provisional_ok=bool(args.provisional_approve_all))
    report = {"kind": "verify-report", "approvals": {"packet": approvals.get("packet"),
              "provisional": approvals.get("packet") == PROVISIONAL_PACKET, "refused": out["plan"]["refused"]},
              "written_ids": out["plan"]["write_ids"], "checks": out["results"], "failed": out["failed"],
              "writer": out["writer_stats"], "sidecar": out["sidecar_stats"], "out_hashes": out["out_hashes"],
              "p1": pinned.record["p1"], "isolation": pinned.record["isolation"], "id_map_sha256": out["id_map_sha256"]}
    sha = write_artifact(os.path.join(run_dir, "reports", "verify-report-%s.json" % utc), report)
    t6sha = write_artifact(os.path.join(run_dir, "reports", "t6-would-mint-%s.json" % utc), {"kind": "t6-would-mint", **out["t6"]})
    if out["failed"]:
        raise Stop("verifier FAILED: %s (report %s)" % (",".join(out["failed"]), sha))
    return {"run_dir": run_dir, "written": len(out["plan"]["write_ids"]), "refused": len(out["plan"]["refused"]),
            "checks_ok": len(out["results"]), "verify_report_sha256": sha, "t6_sha256": t6sha,
            "would_mint_count": out["t6"]["would_mint_count"]}


# --------------------------------------------------------------------------------------------------
# Phase 2 (LIVE). Built and tested on SYNTHETIC directories only. In this build it is REFUSED without Josh's go.
# --------------------------------------------------------------------------------------------------

def _parse_ts(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return _dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def _daemon_log_for(target_real: str) -> str:
    return os.path.join(os.path.dirname(target_real), "daemon.log")


def _require_phase2_inputs(args) -> float:
    """C1 + C4, shared by phase2-backup, --apply and rollback: --code-placed-at is REQUIRED and must parse (a
    readable daemon.log is required by the P4 gate itself); the --expect-* flags must equal the module constants
    (118/182/183) - Phase 1 may take other values for a COPY, Phase 2 never."""
    got, pinned_c = (args.expect_wants, args.expect_protected, args.expect_scope), (EXPECTED_WANTS, EXPECTED_PROTECTED, EXPECTED_SCOPE)
    if got != pinned_c:
        raise Refusal("--expect-wants/--expect-protected/--expect-scope must equal the plan's pinned constants %s at Phase 2 (got %s)"
                      % ("/".join(map(str, pinned_c)), "/".join(map(str, got))))
    if not args.code_placed_at:
        raise Refusal("--code-placed-at is REQUIRED at Phase 2 (the 'no pulse since the code was placed' leg of P4 is never skipped)")
    try:
        return _parse_ts(args.code_placed_at)
    except ValueError:
        raise Refusal("--code-placed-at %r is not an epoch or an ISO time" % args.code_placed_at)


def _partner_specs(args, tdir: str) -> List[Tuple[str, str]]:
    """--generation-partner NAME=PATH: the two KNOWN partner paths, validated without listing anything. A path must
    lie under <target>/generations/ and be a regular file; the directory itself is never listed or opened."""
    root = os.path.join(tdir, "generations")
    out = []
    for spec in (getattr(args, "generation_partner", None) or []):
        name, sep, path = spec.partition("=")
        if not sep or name not in SIX_FILES or not path:
            raise Refusal("--generation-partner wants NAME=PATH with NAME one of the six checkpoint files (got %r)" % spec)
        real = os.path.realpath(path)
        if real == os.path.realpath(root) or not _under(real, root):
            raise Refusal("--generation-partner %s must lie under %s (a recorded path is stat/hashed only; the directory is never listed)"
                          % (name, root))
        if not os.path.isfile(real):
            raise Refusal("--generation-partner %s: %s is not a regular file" % (name, path))
        out.append((name, path))
    return out


def _partner_record(name: str, path: str, live: Dict[str, int]) -> Dict[str, Any]:
    """Read-only evidence about one generation partner: its inode identity and sha256. `same_file_as_live` is decided by
    the (st_dev, st_ino) pair - never by a link count."""
    if not os.path.isfile(path):
        return {"name": name, "path": path, "present": False}
    st = os.stat(path)
    return {"name": name, "path": path, "present": True, "st_dev": st.st_dev, "st_ino": st.st_ino, "st_nlink": st.st_nlink,
            "size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": sha256_file(path),
            "same_file_as_live": (st.st_dev, st.st_ino) == (live["st_dev"], live["st_ino"])}


def _hold_start_of(hold: Dict[str, Any]) -> Dict[str, Any]:
    return {k: hold[k] for k in hold if k.startswith("h")}


def _p4_failures(p4: Dict[str, Any], p6: Dict[str, Any]) -> str:
    return "%s %s %s" % ([k for k, v in p4["checks"].items() if not v], sorted(p4["probe_errors"]), p6["violations"])


def stage_phase2_backup(args, pinned, probes, target_info) -> Dict[str, Any]:
    """The START-OF-PHASE-2 BACKUP (plan 6.3/6.6): only with the daemon mechanically down and the peer hold in
    place; copies the six files to a new run directory, re-hashes, re-reads independently, writes
    backup-manifest-<UTC>.json and returns its sha256 - the value Josh's go must quote. The manifest carries the
    BEFORE inode evidence (st_dev/st_ino/st_nlink of the six live files), the read-only generation-partner records
    and the ripple table (Exec P428)."""
    tdir = target_info["target_realpath"]
    placed = _require_phase2_inputs(args)
    partners = _partner_specs(args, tdir)
    refuse_if_retired(tdir)
    p4 = gate_p4(probes, tdir, code_placed_at=placed, daemon_log=_daemon_log_for(tdir), require_files_equal=False)
    p6 = gate_p6(probes, args.conduit_dir or "")
    if not (p4["ok"] and p6["ok"]):
        raise Refusal("phase2-backup: P4/P6 not satisfied: %s" % _p4_failures(p4, p6))
    run_dir, utc = new_run_dir()
    copy = copy_six(tdir, os.path.join(run_dir, "backup"))
    reread = {n: sha256_file(os.path.join(run_dir, "backup", n)) for n in SIX_FILES}
    if reread != {n: copy["files"][n]["sha256"] for n in SIX_FILES}:
        raise Stop("backup: the independent re-read differs from the source hashes")
    manifest = {"kind": "backup-manifest", "utc": utc, "target_realpath": tdir,
                "files": {n: {"sha256": copy["files"][n]["sha256"], "size": copy["files"][n]["size"],
                              "mtime_ns": copy["files"][n]["mtime_ns"], "st_dev": copy["files"][n]["st_dev"],
                              "st_ino": copy["files"][n]["st_ino"], "st_nlink": copy["files"][n]["st_nlink"]} for n in SIX_FILES},
                "generation_partners": [_partner_record(n, pth, copy["files"][n]) for n, pth in partners],
                "ripple": RIPPLE_TABLE, "independent_reread_sha256": reread}
    msha = write_artifact(os.path.join(run_dir, "backup-manifest-%s.json" % utc), manifest)
    write_artifact(os.path.join(run_dir, "hold-start.json"), {"kind": "hold-start", **p6["snapshot"]})
    return {"run_dir": run_dir, "backup_manifest_sha256": msha, "p4": p4["checks"], "p6_ok": p6["ok"],
            "generation_partners_recorded": len(partners)}


def _one_manifest(run_dir: str, quoted_sha256: str) -> Tuple[str, str, Dict[str, Any]]:
    mpaths = sorted(glob.glob(os.path.join(run_dir, "backup-manifest-*.json")))
    if len(mpaths) != 1:
        raise Refusal("P8: the run directory must hold exactly one backup-manifest")
    msha = sha256_file(mpaths[0])
    if quoted_sha256 != msha:
        raise Refusal("P8: Josh's go quotes %s, the backup manifest is %s" % (quoted_sha256, msha))
    return mpaths[0], msha, load_artifact(mpaths[0])


def _assert_inodes_after(before: Dict[str, Dict[str, Any]], after: Dict[str, Dict[str, int]], rewritten) -> None:
    """The replace must have behaved as Exec P428 says: every REWRITTEN live file is a NEW inode on the same device
    with link count 1; every file that was not rewritten is still the same inode."""
    for n in SIX_FILES:
        b, a = before[n], after[n]
        if n in rewritten:
            if a["st_dev"] != b["st_dev"] or a["st_ino"] == b["st_ino"] or a["st_nlink"] != 1:
                raise Stop("post-write inode check FAILED for %s: expected a NEW inode with link count 1 (before %s, after %s)" % (n, b, a))
        elif (a["st_dev"], a["st_ino"]) != (b["st_dev"], b["st_ino"]):
            raise Stop("post-write inode check FAILED for %s: an untouched file changed inode" % n)


def stage_apply(args, pinned, probes, target_info) -> Dict[str, Any]:
    """--apply. Every gate P1-P9 must hold, Josh's go must quote the backup manifest, the tool re-runs on the LIVE
    bytes (never copies Phase-1 output over live), any deviation from the approved list is a STOP for the whole
    run, the staged pair is verified V1-V19 before the H1-H3 + daemon-down re-check immediately before the
    os.replace, and the RETIRED receipt is written strictly AFTER both files verify."""
    if not args.josh_go:
        raise Refusal("--apply is REFUSED: Phase 2 needs Josh's go (--josh-go <reference> quoting the backup-manifest sha256)")
    if not (args.run_dir and args.frozen_dir and args.approvals and args.approvals_sha256 and args.daemon_organism_file
            and args.josh_go_manifest_sha256 and args.conduit_dir):
        raise Refusal("--apply needs --run-dir --frozen-dir --approvals --approvals-sha256 --daemon-organism-file "
                      "--josh-go-manifest-sha256 --conduit-dir")
    tdir = target_info["target_realpath"]
    if tdir != os.path.realpath(RECORDED_CC_CHECKPOINT_DIR):
        raise Refusal("P7: the live write target must be the recorded CC checkpoint directory")
    placed = _require_phase2_inputs(args)
    refuse_if_retired(tdir)
    run_dir = guard_out_path(args.run_dir)
    mpath, msha, manifest = _one_manifest(run_dir, args.josh_go_manifest_sha256)
    hold_start = load_artifact(os.path.join(run_dir, "hold-start.json"))
    log = _daemon_log_for(tdir)
    gates = []
    gates.append({"gate": "P1", "ok": True, "p1": pinned.record["p1"]})
    gates.append(gate_p2(args.daemon_organism_file, pinned.root))
    gates.append(gate_p3(pinned, run_pinned_tests=True))
    gates.append(gate_p4(probes, tdir, manifest_files=manifest["files"], code_placed_at=placed, daemon_log=log))
    gates.append(gate_p6(probes, args.conduit_dir, _hold_start_of(hold_start)))
    gates.append({"gate": "P8", "ok": True, "go": args.josh_go, "manifest_sha256": msha})
    failed_gates = [g["gate"] for g in gates if not g["ok"]]
    if failed_gates:
        raise Refusal("Phase 2 gate(s) failed: %s" % ",".join(failed_gates))
    frozen_rl = load_artifact(os.path.join(args.frozen_dir, "repair-list.json"))
    frozen_sc = load_artifact(os.path.join(args.frozen_dir, "scope-ids.json"))
    frozen_map = load_artifact(os.path.join(args.frozen_dir, "id-map.json"))
    rl_sha0, sc_sha0 = sha256_file(os.path.join(args.frozen_dir, "repair-list.json")), sha256_file(os.path.join(args.frozen_dir, "scope-ids.json"))
    approvals = load_approvals(args.approvals, args.approvals_sha256, rl_sha0, sc_sha0)        # P5
    scope = frozen_sc["ids"]
    if len(scope) != args.expect_scope:
        raise Stop("P9: the frozen scope has %d ids, not %d" % (len(scope), args.expect_scope))
    A = analyze(pinned, tdir, scope_min_len=args.scope_min_len, frozen_scope=scope, full_reports=False)   # the LIVE bytes
    if sorted(A["scope_derived"]) != sorted(scope):
        raise Stop("P9: the live set of wants meeting the scope rule differs from the frozen list")
    if artifact_sha256(repair_list_obj(A)) != rl_sha0 or artifact_sha256(scope_ids_obj(A, args.expect_scope)) != sc_sha0:
        raise Stop("P5: the live classification is not the frozen one (repair-list / scope-ids differ)")
    utc = utc_stamp()
    in_hashes = {n: manifest["files"][n]["sha256"] for n in SIX_FILES}
    out = prepare_outputs(pinned, A, approvals, in_dir=tdir, out_dir=os.path.join(run_dir, "stage"), run_dir=run_dir, utc=utc,
                          expect=_expect(args), in_hashes=in_hashes)
    if out["failed"]:
        raise Stop("verifier FAILED on the staged pair: %s" % ",".join(out["failed"]))
    approved_and_candidate = sorted(r["id"] for r in A["records"] if r["disposition"] == "candidate"
                                    and r["id"] in set(out["plan"]["approved_ids"]))
    if out["plan"]["refused"] or sorted(out["plan"]["write_ids"]) != approved_and_candidate or \
            sorted(map(list, out["plan"]["mapping"].items())) != sorted(frozen_map["pairs"]):
        raise Stop("any deviation from the approved list is a STOP at Phase 2 (never a silent drop)")

    def recheck():
        p4 = gate_p4(probes, tdir, manifest_files=manifest["files"], code_placed_at=placed, daemon_log=log)
        p6 = gate_p6(probes, args.conduit_dir, _hold_start_of(hold_start))
        if not (p4["ok"] and p6["ok"]):
            raise Refusal("immediately before os.replace: P4/P6 no longer hold: %s" % _p4_failures(p4, p6))

    def writer(staged: str, want: str, is_main: bool):
        def fn(tmp: str):
            shutil.copyfile(staged, tmp)
            if sha256_file(tmp) != want:
                raise Stop("staged file changed while copying to the live tmp")
            if is_main:
                recheck()
        return fn

    if os.path.realpath(tdir) != os.path.realpath(RECORDED_CC_CHECKPOINT_DIR):
        raise Refusal("P7: refusing the live write - the target is not the recorded CC checkpoint directory")
    rewritten = (MAIN_NAME, SIDECAR_NAME)
    pinned.cg.atomic_file_write(os.path.join(tdir, MAIN_NAME), writer(out["out_main"], out["out_hashes"][MAIN_NAME], True))
    pinned.cg.atomic_file_write(os.path.join(tdir, SIDECAR_NAME), writer(out["out_sidecar"], out["out_hashes"][SIDECAR_NAME], False))
    live_ok = (sha256_file(os.path.join(tdir, MAIN_NAME)) == out["out_hashes"][MAIN_NAME]
               and sha256_file(os.path.join(tdir, SIDECAR_NAME)) == out["out_hashes"][SIDECAR_NAME]
               and not sum(census_msgpack_file(os.path.join(tdir, MAIN_NAME), list(out["plan"]["mapping"])).values()))
    if not live_ok:
        raise Stop("post-apply verification of the live files FAILED - roll back with --step rollback from the backup manifest")
    before_i = {n: {k: manifest["files"][n][k] for k in ("st_dev", "st_ino", "st_nlink")} for n in SIX_FILES}
    after_i = {n: stat_ident(os.path.join(tdir, n)) for n in SIX_FILES}
    _assert_inodes_after(before_i, after_i, rewritten)
    partners_after = [_partner_record(pr["name"], pr["path"], after_i[pr["name"]]) for pr in manifest.get("generation_partners", [])]
    final = write_artifact(os.path.join(run_dir, "post-apply-receipt-FINAL-%s.json" % utc), {
        "kind": "post-apply-receipt", "six_file_sha256_after": out["out_hashes"], "mapping_sha256": out["id_map_sha256"],
        "live_inodes_before": before_i, "live_inodes_after": after_i, "generation_partners_after": partners_after,
        "ripple": RIPPLE_TABLE, "p2": gates[1],
        "josh_go": args.josh_go, "backup_manifest_sha256": msha, "approvals_packet": approvals.get("packet"),
        "gates": [g.get("gate") for g in gates], "tool_sha256": tool_sha256()})
    retired = write_retired_receipt(run_dir, tdir, out["id_map_sha256"], utc)     # strictly AFTER both files verified
    return {"applied": len(out["plan"]["write_ids"]), "post_apply_receipt_sha256": final, "retired_receipt": os.path.basename(retired),
            "note": "the daemon is NOT started; the S4 start waits on gate P10 (the T6 would-mint set signed per id)"}


def stage_rollback(args, pinned, probes, target_info) -> Dict[str, Any]:
    """--step rollback (le-029 C6, decided: ADDED). Restores the six files from the tool's OWN named pre-apply
    backup (<run>/backup/, every sha256 verified against backup-manifest-<UTC>.json BEFORE any write) - never from a
    generation directory or last_good/. Gated like the apply: Josh's go quoting the manifest sha256, P4 (daemon down,
    the equality leg replaced by the identity check below), P6, --code-placed-at and the pinned --expect-*; refused
    unless EVERY live file is either the backup's bytes or the post-apply receipt's bytes (an apply that died between
    the two replaces, or a finished apply); it cannot undo anything after S4 (that needs the P391 export first, and its
    identity check refuses it). Writes only through atomic tmp + os.replace; the RETIRED receipt is left in place; the
    host stays DOWN until Chief's resume gate and the Executive's parser ruling (plan 6.7)."""
    if not args.josh_go:
        raise Refusal("--step rollback is REFUSED: it needs Josh's go (--josh-go <reference> quoting the backup-manifest sha256)")
    if not (args.run_dir and args.josh_go_manifest_sha256 and args.conduit_dir):
        raise Refusal("--step rollback needs --run-dir --josh-go-manifest-sha256 --conduit-dir")
    tdir = target_info["target_realpath"]
    if tdir != os.path.realpath(RECORDED_CC_CHECKPOINT_DIR):
        raise Refusal("P7: the live write target must be the recorded CC checkpoint directory")
    placed = _require_phase2_inputs(args)
    run_dir = guard_out_path(args.run_dir)
    mpath, msha, manifest = _one_manifest(run_dir, args.josh_go_manifest_sha256)
    hold_start = load_artifact(os.path.join(run_dir, "hold-start.json"))
    log = _daemon_log_for(tdir)
    p4 = gate_p4(probes, tdir, code_placed_at=placed, daemon_log=log, require_files_equal=False)
    p6 = gate_p6(probes, args.conduit_dir, _hold_start_of(hold_start))
    if not (p4["ok"] and p6["ok"]):
        raise Refusal("rollback: gate(s) P4/P6 not satisfied: %s" % _p4_failures(p4, p6))
    finals = sorted(glob.glob(os.path.join(run_dir, "post-apply-receipt-FINAL-*.json")))
    drafts = sorted(glob.glob(os.path.join(run_dir, "post-apply-receipt-[0-9]*.json")))
    rpaths = finals or drafts
    if len(rpaths) != 1:
        raise Refusal("rollback: the run directory must hold exactly one post-apply receipt (final, or the pre-replace draft); found %d" % len(rpaths))
    after_sha = load_artifact(rpaths[0])["six_file_sha256_after"]
    backup_sha = {n: manifest["files"][n]["sha256"] for n in SIX_FILES}
    live_sha = {n: sha256_file(os.path.join(tdir, n)) for n in SIX_FILES}
    for n in SIX_FILES:
        if live_sha[n] not in (backup_sha[n], after_sha[n]):
            raise Refusal("rollback: identity - live %s matches neither the pre-apply backup nor the post-apply receipt "
                          "(something else wrote it; a post-S4 restore needs the P391 export first)" % n)
    backup_dir = os.path.join(run_dir, "backup")
    for n in SIX_FILES:                                                    # EVERY sha256 verified before ANY write
        bp = guard_out_path(os.path.join(backup_dir, n))
        if not os.path.isfile(bp) or sha256_file(bp) != backup_sha[n]:
            raise Refusal("rollback: the backup copy of %s does not match its sha256 in the manifest" % n)
    to_restore = [n for n in SIX_FILES if live_sha[n] != backup_sha[n]]
    before_i = {n: stat_ident(os.path.join(tdir, n)) for n in SIX_FILES}

    def recheck():
        q4 = gate_p4(probes, tdir, code_placed_at=placed, daemon_log=log, require_files_equal=False)
        q6 = gate_p6(probes, args.conduit_dir, _hold_start_of(hold_start))
        if not (q4["ok"] and q6["ok"]):
            raise Refusal("immediately before os.replace: P4/P6 no longer hold: %s" % _p4_failures(q4, q6))

    def writer(src: str, want: str):
        def fn(tmp: str):
            shutil.copyfile(src, tmp)
            if sha256_file(tmp) != want:
                raise Stop("the backup copy changed while copying to the live tmp")
            recheck()
        return fn

    if os.path.realpath(tdir) != os.path.realpath(RECORDED_CC_CHECKPOINT_DIR):
        raise Refusal("P7: refusing the live write - the target is not the recorded CC checkpoint directory")
    for n in to_restore:
        pinned.cg.atomic_file_write(os.path.join(tdir, n), writer(os.path.join(backup_dir, n), backup_sha[n]))
    if {n: sha256_file(os.path.join(tdir, n)) for n in SIX_FILES} != backup_sha:
        raise Stop("post-rollback verification FAILED: a live file is not the backup's bytes")
    after_i = {n: stat_ident(os.path.join(tdir, n)) for n in SIX_FILES}
    for n in SIX_FILES:
        if n in to_restore:
            if after_i[n]["st_ino"] == before_i[n]["st_ino"] or after_i[n]["st_nlink"] != 1:
                raise Stop("post-rollback inode check FAILED for %s: expected a NEW inode with link count 1" % n)
        elif after_i[n]["st_ino"] != before_i[n]["st_ino"]:
            raise Stop("post-rollback inode check FAILED for %s: an untouched file changed inode" % n)
    utc = utc_stamp()
    rsha = write_artifact(os.path.join(run_dir, "rollback-receipt-%s.json" % utc), {
        "kind": "rollback-receipt", "restored": sorted(to_restore), "source_dir": os.path.realpath(backup_dir),
        "six_file_sha256_after": backup_sha, "live_inodes_before": before_i, "live_inodes_after": after_i,
        "never_a_source": "generations/ and last_good/ (incidental, expiring, never a rollback source)",
        "host_stays_down": True, "host_down_until": "Chief's post-restore resume gate (P398) AND the Executive's ruling on the parser",
        "josh_go": args.josh_go, "backup_manifest_sha256": msha, "post_apply_receipt": os.path.basename(rpaths[0]),
        "ripple": RIPPLE_TABLE, "tool_sha256": tool_sha256()})
    return {"restored": sorted(to_restore), "rollback_receipt_sha256": rsha, "host_stays_down": True,
            "note": "the daemon is NOT started; the host stays DOWN until the resume gate and the parser ruling"}


# --------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=TOOL_NAME, description=__doc__.split("\n")[0])
    ap.add_argument("--pin-root", required=True, help="the read-only PIN worktree (ae798b94) the parser is imported from")
    ap.add_argument("--target-dir", required=True, help="the CC checkpoint directory (REQUIRED, no default; read-only in Phase 1)")
    ap.add_argument("--daemon-script", required=True, help="cc-ng-daemon.py, read as text to cross-check CHECKPOINT_DIR")
    ap.add_argument("--scope-min-len", type=int, required=True, help="reported cross-check only (600); the enumerated list selects")
    ap.add_argument("--step", choices=("classify", "rewrite", "phase2-backup", "rollback"), default="classify")
    ap.add_argument("--run-dir")
    ap.add_argument("--approvals")
    ap.add_argument("--approvals-sha256", help="the hash relayed from the Executive's packet text")
    ap.add_argument("--provisional-approve-all", action="store_true",
                    help="DRY-RUN ONLY: exercise the rewrite+verifier on the COPY before approvals exist; Phase 2 refuses it")
    ap.add_argument("--expect-wants", type=int, default=EXPECTED_WANTS)
    ap.add_argument("--expect-protected", type=int, default=EXPECTED_PROTECTED)
    ap.add_argument("--expect-scope", type=int, default=EXPECTED_SCOPE)
    ap.add_argument("--apply", action="store_true", help="Phase 2 live write - REFUSED without Josh's go and every gate")
    ap.add_argument("--josh-go", help="Josh's go reference; it must quote the backup-manifest sha256")
    ap.add_argument("--josh-go-manifest-sha256")
    ap.add_argument("--frozen-dir", help="a directory holding UNEDITED copies of exactly three files of the classify run: "
                    "reports/repair-list.json, reports/scope-ids.json and reports/id-map.json (keep those names)")
    ap.add_argument("--daemon-organism-file", help="P2: the cc_ng_organism.py in the daemon unit's ACTUAL import root, named by the "
                    "Chief in the go request (Q9); a path inside the pin worktree or this tool's worktree is refused")
    ap.add_argument("--generation-partner", action="append", default=[], metavar="NAME=PATH",
                    help="phase2-backup: a KNOWN generation-directory partner of a live file, e.g. main.msgpack=<target>/generations/<stamp>/main.msgpack; "
                    "stat/hashed read-only and recorded by inode, never listed or used as a rollback source")
    ap.add_argument("--conduit-dir", help="the ng_topology conduit directory (H3; stat only)")
    ap.add_argument("--code-placed-at", help="when the pinned code was placed for the daemon (epoch or ISO)")
    return ap


def main(argv: Optional[List[str]] = None, probes: Optional[Probes] = None) -> int:
    args = build_parser().parse_args(argv)
    probes = probes or Probes()
    try:
        if (args.apply or args.step == "rollback") and not args.josh_go:   # the first refusal: no Josh go, nothing else is even loaded
            raise Refusal("%s is REFUSED: it needs Josh's go (--josh-go <reference> quoting the backup-manifest sha256)"
                          % ("--apply" if args.apply else "--step rollback"))
        pinned = load_pinned(args.pin_root)
        print("\n".join(p379_lines(pinned)))
        target_info = guard_target(args.target_dir, args.daemon_script)
        if args.apply:
            res = stage_apply(args, pinned, probes, target_info)
        elif args.step == "classify":
            res = stage_classify(args, pinned, target_info)
        elif args.step == "rewrite":
            res = stage_rewrite(args, pinned, target_info)
        elif args.step == "rollback":
            res = stage_rollback(args, pinned, probes, target_info)
        else:
            res = stage_phase2_backup(args, pinned, probes, target_info)
        _assert_text_free(res)
        print(json.dumps(res, sort_keys=True))
        return 0
    except Refusal as exc:
        print("REFUSED: %s" % exc, file=sys.stderr)
        return 2
    except Stop as exc:
        print("STOP: %s" % exc, file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
