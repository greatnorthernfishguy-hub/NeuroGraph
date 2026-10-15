"""
dim_reembed.py — rebuild wrong-dimension vectors from their OWN stored text with the active model (#1097).

Copies only: every subcommand reads an input file and writes a NEW output file; it never writes in place and
refuses to overwrite its input. Installing the output over a live file is a separate, Josh-approved step
(DIM_GUARD.md §Install).

Subcommands
-----------
  rim IN.json OUT.json
      A constitutional rim (``{"model", "rim_categories", "embeddings": [{"category", "description",
      "embedding"}]}``). Each entry's ``description`` is its source text (verified for Elmer's rim: the stored
      MiniLM vectors are exactly MiniLM(description), DIM_GUARD.md §3). Re-embeds every description with
      ng_embed (document mode, L2-normalized), keeps category / description / order, sets ``model`` to
      ng_embed.MODEL_NAME and records the replaced model under ``reembedded_from``.

  nglite-state IN OUT --rim RIM.json [--drop-unrecoverable]
      An NG-Lite state (msgpack, legacy JSON, or JSON inside a .msgpack). Wrong-dim constitutional (rim) nodes
      are dropped and re-seeded from RIM.json (already at the active dim — run ``rim`` first); their
      synapses (frozen, normally none) go with them. Wrong-dim receptor prototypes are dropped (they are
      derived centroids; the receptor layer re-warms from new inputs). Any OTHER wrong-dim node has no source
      text in an NG-Lite state, so it cannot be re-embedded: the tool stops and reports the count unless
      ``--drop-unrecoverable`` is given (then those nodes and their synapses are dropped and listed by count).
      The output is written through canonical NGLite (msgpack) and re-loaded with the dimension guard on.

  check FILE [FILE ...]
      Report the stored vector dimensions of NG-Lite states / rims / vectors.msgpack files against
      ng_embed.EMBEDDING_DIM (exit 1 if any file has a wrong-dim vector). Read-only.

For a NeuroGraph vector store (vectors.msgpack: every entry keeps its ``content``) use reembed_snowflake.py,
which already re-embeds each entry from its content (LAW 3: not duplicated here).

# ---- Changelog ----
# [2026-10-14] Claude Code (Opus 5.5, lane dim-guard) — Initial creation (#1097)
#   What: rim / nglite-state / check subcommands; copies only.
#   Why:  Elmer's NG-Lite rim (data/constitutional_embeddings.json + ~/.et_modules/elmer/ng_lite_state.msgpack)
#         is 384-d all-MiniLM-L6-v2 while every input is 768-d Snowflake; with the dimension guard such a
#         state is refused at load, so it must be rebuilt from its own text, never padded or truncated.
#   How:  ng_embed.embed_batch for the text; canonical NGLite for the state round trip.
# -------------------
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _refuse_same(a: str, b: str) -> None:
    if os.path.abspath(a) == os.path.abspath(b):
        sys.exit(f"refusing to write over the input ({a}); write a copy and install it separately")
    if os.path.exists(b):
        sys.exit(f"refusing to overwrite an existing output ({b})")


def _read_state(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        data = f.read()
    if data[:1] in (b"{", b"["):
        return json.loads(data.decode("utf-8"))
    import msgpack
    return msgpack.unpackb(data, raw=False, strict_map_key=False)


def _vlen(v: Any) -> int:
    if isinstance(v, (bytes, bytearray)):
        return len(v) // 4
    return len(v)


# --------------------------------------------------------------------------- rim

def cmd_rim(args: argparse.Namespace) -> int:
    import ng_embed
    _refuse_same(args.inp, args.out)
    with open(args.inp) as f:
        rim = json.load(f)
    entries = rim.get("embeddings", [])
    texts = [e.get("description", "") for e in entries]
    missing = [i for i, t in enumerate(texts) if not t.strip()]
    if missing:
        sys.exit(f"{len(missing)} rim entries have no description (no source text): indices {missing[:10]} — "
                 "not re-embedding (nothing is invented)")
    old_dims = sorted({_vlen(e["embedding"]) for e in entries if e.get("embedding") is not None})
    vecs = ng_embed.embed_batch(texts, normalize=True)
    out = copy.deepcopy(rim)
    out["reembedded_from"] = {"model": rim.get("model"), "dims": old_dims}
    out["model"] = ng_embed.MODEL_NAME
    out["embedding_dim"] = ng_embed.EMBEDDING_DIM
    for e, v in zip(out["embeddings"], vecs):
        v = np.asarray(v, dtype=np.float32)
        if v.shape != (ng_embed.EMBEDDING_DIM,) or not np.all(np.isfinite(v)):
            sys.exit("ng_embed returned a malformed vector; nothing written")
        e["embedding"] = [float(x) for x in v]
    tmp = args.out + ".partial"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, args.out)
    print(f"rim: {len(entries)} entries re-embedded {old_dims} -> {ng_embed.EMBEDDING_DIM} "
          f"({rim.get('model')} -> {ng_embed.MODEL_NAME}); wrote {args.out}")
    return 0


# --------------------------------------------------------------------------- NG-Lite state

def cmd_nglite_state(args: argparse.Namespace) -> int:
    import ng_embed
    from ng_lite import NGLite
    _refuse_same(args.inp, args.out)
    D = ng_embed.EMBEDDING_DIM
    with open(args.rim) as f:
        rim_entries = json.load(f).get("embeddings", [])
    bad_rim = [e for e in rim_entries if _vlen(e.get("embedding", [])) != D]
    if bad_rim:
        sys.exit(f"--rim has {len(bad_rim)} entries that are not {D}-d; run `dim_reembed.py rim` first")

    state = _read_state(args.inp)
    nodes = state.get("nodes", {}) or {}
    drop_keys: List[str] = []
    unrecoverable: List[str] = []
    for key, nd in nodes.items():
        emb = nd.get("embedding")
        if emb is None or _vlen(emb) == D:
            continue
        if nd.get("constitutional"):
            drop_keys.append(key)          # re-seeded from the new rim below
        else:
            unrecoverable.append(key)      # an NG-Lite node keeps no text: cannot be re-embedded
    if unrecoverable and not args.drop_unrecoverable:
        sys.exit(f"{len(unrecoverable)} non-rim nodes are wrong-dim and an NG-Lite state keeps no source text "
                 "for them (cannot re-embed, nothing invented). Re-run with --drop-unrecoverable to drop them "
                 "and their synapses, or keep the module on hold.")
    drop_keys += unrecoverable
    dropped_ids = {nodes[k]["node_id"] for k in drop_keys}
    for k in drop_keys:
        del nodes[k]
    syn = state.get("synapses", {}) or {}
    dropped_syn = [k for k, s in syn.items() if s.get("source_id") in dropped_ids]
    for k in dropped_syn:
        del syn[k]
    rl = state.get("receptor_layer") or {}
    protos_dropped = 0
    if rl.get("prototypes") and any(_vlen(p) != D for p in rl["prototypes"]):
        protos_dropped = len(rl["prototypes"])
        state["receptor_layer"] = {}
    cfg = dict(state.get("config", {}) or {})
    cfg["embedding_dim"] = D
    cfg["constitutional_embeddings"] = rim_entries
    state["config"] = cfg

    ng = NGLite(module_id=state.get("module_id", "default"), config={"embedding_dim": D,
                                                                     "constitutional_embeddings": rim_entries})
    ng._import_state(state, source=args.inp)          # guarded: raises if anything wrong-dim is left
    written = ng.save(args.out)
    # Re-load the written copy with the guard on (a fresh instance, the module's rim in its constructor)
    check = NGLite(module_id=state.get("module_id", "default"), config={"constitutional_embeddings": rim_entries})
    assert check.load(written)
    n_const = sum(1 for n in check.list_nodes(with_embedding=False) if n.constitutional)
    print(f"nglite-state: {args.inp}: dropped {len(drop_keys) - len(unrecoverable)} wrong-dim rim nodes "
          f"(re-seeded {n_const} from {args.rim}), {len(unrecoverable)} unrecoverable wrong-dim nodes, "
          f"{len(dropped_syn)} synapses, {protos_dropped} prototypes; kept {check.node_count()} nodes / "
          f"{check.synapse_count()} synapses; wrote {written} (re-loaded with the dimension guard: OK)")
    return 0


# --------------------------------------------------------------------------- check

def _dims_of(path: str) -> Dict[str, Dict[int, int]]:
    out: Dict[str, Dict[int, int]] = {}

    def add(k: str, d: int) -> None:
        out.setdefault(k, {})
        out[k][d] = out[k].get(d, 0) + 1

    st = _read_state(path)
    if isinstance(st, list):                                                                     # a list store
        for e in st:
            if isinstance(e, dict) and e.get("embedding") is not None:
                add("entries", _vlen(e["embedding"]))
        return out
    if isinstance(st, dict) and "embeddings" in st and isinstance(st["embeddings"], list):      # a rim
        for e in st["embeddings"]:
            add("rim", _vlen(e["embedding"]))
        return out
    if isinstance(st, dict) and "entries" in st:                                                 # vectors.msgpack
        for e in st["entries"].values():
            add("vectors", _vlen(e["embedding"]))
        return out
    for nd in (st.get("nodes") or {}).values():                                                  # NG-Lite state
        if isinstance(nd, dict) and nd.get("embedding") is not None:
            add("nodes", _vlen(nd["embedding"]))
    for p in (st.get("receptor_layer") or {}).get("prototypes") or []:
        add("prototypes", _vlen(p))
    for e in (st.get("config") or {}).get("constitutional_embeddings") or []:
        add("config.rim", _vlen(e["embedding"]))
    return out


def cmd_check(args: argparse.Namespace) -> int:
    import ng_embed
    D = ng_embed.EMBEDDING_DIM
    bad = 0
    for p in args.files:
        dims = _dims_of(p)
        wrong = {k: v for k, v in dims.items() if any(d != D for d in v)}
        bad += bool(wrong)
        print(f"{'WRONG' if wrong else 'ok   '} {p}: " + "; ".join(
            f"{k} " + ", ".join(f"{d}-d x{c}" for d, c in sorted(v.items())) for k, v in dims.items()))
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("rim"); a.add_argument("inp"); a.add_argument("out")
    b = sub.add_parser("nglite-state"); b.add_argument("inp"); b.add_argument("out")
    b.add_argument("--rim", required=True); b.add_argument("--drop-unrecoverable", action="store_true")
    c = sub.add_parser("check"); c.add_argument("files", nargs="+")
    args = ap.parse_args()
    return {"rim": cmd_rim, "nglite-state": cmd_nglite_state, "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
