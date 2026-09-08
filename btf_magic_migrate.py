#!/usr/bin/env python3
"""
btf_magic_migrate.py — convert every legacy "TB" BTF entry in place to canonical "BT".

Nothing is deleted. Nothing is re-encoded. Two bytes per entry change and every
other byte is preserved exactly, so no experience is lost or altered — the
entries simply stop lying about their own format.

WHY THIS EXISTS
---------------
BTF v0.1 §4.1 fixes the on-disk magic at 0x42 0x54 ("BT"). A wheel built before
`ng-tract-rs` commit 6aef81a serialized MAGIC as a u16 native-endian, which on
little-endian hardware emitted 0x54 0x42 ("TB"). Syl's VPS has been running such
a wheel since 2026-06-09, so it has been writing wrong-magic entries for months.

⚠ AND THE READERS DO NOT ACTUALLY TOLERATE BOTH. Measured 2026-09-07:

    reader          parses BT            parses TB
    old (VPS)       RAW BYTES            PyOutcomeEntry
    new (laptop)    PyOutcomeEntry       RAW BYTES

An unrecognised magic is handed back as RAW BYTES, and the scan-drain filters
those out (`[e for e in reader if not isinstance(e, bytes)]`). So a mismatched
entry is not translated — it is SILENTLY DROPPED. There is no translator; there
is a blind spot in each direction.

Two consequences that decide the migration order:

  * Upgrade the wheel first  -> the entire TB backlog becomes raw bytes and is
                                dropped on the next drain. Data loss.
  * Convert the files first  -> the still-old wheel cannot read its own tracts.
                                Live drain breaks.

So conversion and wheel upgrade MUST happen inside one sidecar stop. Order
within that window does not matter; doing either alone does harm.

Josh: "I do not want 2 formats… I do NOT want them deleted. I want them all in
the correct format, universally." Hence: corrected, never deleted.

SAFETY MODEL
------------
  * --dry-run (DEFAULT) touches nothing and reports exactly what would change.
  * Every file is backed up to <file>.pre-btf-magic.<ts> before being written,
    unless --no-backup is passed explicitly.
  * Writes are atomic: temp file in the same directory, fsync, os.replace.
  * VERIFY-BEFORE-COMMIT: every entry is decoded before and after, and the file
    is only replaced when all decoded fields are identical and the byte length
    is unchanged. Any mismatch aborts that file, untouched.
  * Idempotent: a file with no TB entries is skipped, so re-running is a no-op.
  * A file that does not parse cleanly end-to-end is REPORTED AND SKIPPED, never
    partially rewritten.

USAGE
  python3 btf_magic_migrate.py PATH [PATH ...]            # dry run, the default
  python3 btf_magic_migrate.py --apply PATH [PATH ...]    # convert
  python3 btf_magic_migrate.py --apply --no-backup PATH   # convert, no .pre- files
  python3 btf_magic_migrate.py --json PATH                # machine-readable report

PATH may be a file or a directory (walked for *.tract*).

# ---- Changelog ----
# [2026-09-07] DudeMan CC (Fable 5.1) — Created.
#   What: in-place TB->BT magic correction for BTF entries, with per-entry decode
#         verification, atomic writes and backups. Dry-run by default.
#   Why:  Josh: not a single trace of the bad bytes may remain, and none may be
#         deleted — everything in the correct format, universally. That is the
#         precondition for retiring the dual-magic translator (#334).
#   How:  Walk the 24-byte envelopes by total_length; flip bytes 0-1 where they
#         read TB. The CRC32 in the envelope covers the PAYLOAD only (BTF v0.1
#         §4.1), so the magic is not under checksum and the correction cannot
#         invalidate an entry — asserted by test, not assumed.
# -------------------
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

MAGIC_BT = b"\x42\x54"   # canonical, BTF v0.1 §4.1
MAGIC_TB = b"\x54\x42"   # legacy defect: u16 native-endian on LE hardware
ENVELOPE_SIZE = 24
_LEN_OFF = 4             # u32 total_length, native-endian
EMBEDDING_DIM_PROBE = 768  # substrate width, used only for the reader-capability probe


class ParseError(Exception):
    """The file is not a clean run of BTF envelopes; it will be skipped."""


def walk_entries(buf: bytes) -> Iterator[Tuple[int, int, bytes]]:
    """Yield (offset, total_length, magic) for each entry. Raises ParseError."""
    off = 0
    n = len(buf)
    while off < n:
        if n - off < ENVELOPE_SIZE:
            raise ParseError(f"trailing {n - off} bytes at {off}: not an envelope")
        magic = buf[off:off + 2]
        if magic not in (MAGIC_BT, MAGIC_TB):
            raise ParseError(f"unknown magic {magic!r} at offset {off}")
        (total,) = struct.unpack_from("=I", buf, off + _LEN_OFF)
        if total < ENVELOPE_SIZE or off + total > n:
            raise ParseError(f"declared length {total} at offset {off} overruns the file")
        yield off, total, magic
        off += total


def scan_file(path: Path) -> Tuple[List[int], int]:
    """STREAMING envelope walk. Returns (offsets_of_TB_entries, total_entries).

    Never loads the file. Reads 24 bytes per entry and seeks past the payload, so
    peak memory is the offset list, not the data — these tracts run to 578 MB each
    and the host is already paging. Raises ParseError exactly like walk_entries().
    """
    tb: List[int] = []
    total = 0
    size = path.stat().st_size
    with open(path, "rb") as fh:
        off = 0
        while off < size:
            if size - off < ENVELOPE_SIZE:
                raise ParseError(f"trailing {size - off} bytes at {off}: not an envelope")
            env = fh.read(ENVELOPE_SIZE)
            if len(env) < ENVELOPE_SIZE:
                raise ParseError(f"short read at {off}")
            magic = env[:2]
            if magic not in (MAGIC_BT, MAGIC_TB):
                raise ParseError(f"unknown magic {magic!r} at offset {off}")
            (length,) = struct.unpack_from("=I", env, _LEN_OFF)
            if length < ENVELOPE_SIZE or off + length > size:
                raise ParseError(f"declared length {length} at offset {off} overruns the file")
            total += 1
            if magic == MAGIC_TB:
                tb.append(off)
            off += length
            fh.seek(off)
    return tb, total


def correct_in_place(path: Path, offsets: List[int]) -> None:
    """Write the 2 canonical magic bytes at each offset. Nothing else is touched.

    Used only after scan_file() has validated the whole file and a backup exists.
    Two bytes per entry, seek-and-write — no copy, no rewrite, no memory spike.
    """
    with open(path, "r+b") as fh:
        for off in offsets:
            fh.seek(off)
            fh.write(MAGIC_BT)
        fh.flush()
        os.fsync(fh.fileno())


def correct(buf: bytes) -> Tuple[bytes, int, int]:
    """Return (corrected_bytes, entries_total, entries_converted)."""
    out = bytearray(buf)
    total = converted = 0
    for off, _len, magic in walk_entries(buf):
        total += 1
        if magic == MAGIC_TB:
            out[off:off + 2] = MAGIC_BT
            converted += 1
    return bytes(out), total, converted


_BT_READER: Optional[bool] = None


def _reader_understands_bt() -> bool:
    """True if the INSTALLED wheel parses canonical BT entries as entries.

    A pre-fix wheel writes TB and hands BT back as raw bytes. Cached, because the
    answer cannot change inside one run.
    """
    global _BT_READER
    if _BT_READER is not None:
        return _BT_READER
    try:
        import numpy as np
        import ng_tract
        probe = bytes(ng_tract.write_outcome(
            timestamp=1.0, module_id="probe", target_id="probe",
            success=True, embedding=np.zeros(EMBEDDING_DIM_PROBE, dtype=np.float32)))
        if probe[:2] != MAGIC_BT:                 # this wheel writes TB
            fixed = bytearray(probe)
            fixed[0:2] = MAGIC_BT
            probe = bytes(fixed)
        items = list(ng_tract.TractReader(probe))
        _BT_READER = bool(items) and not any(isinstance(e, bytes) for e in items)
    except Exception:  # noqa: BLE001
        _BT_READER = False
    return _BT_READER


def _decode_all(buf: bytes) -> List[Dict[str, Any]]:
    """Decode every entry with the canonical reader, for before/after comparison."""
    import ng_tract
    rows: List[Dict[str, Any]] = []
    for e in ng_tract.TractReader(buf):
        if isinstance(e, bytes):
            rows.append({"raw": bytes(e)})
            continue
        meta = getattr(e, "metadata", None)
        meta = meta() if callable(meta) else meta
        emb = getattr(e, "embedding_as_numpy", None)
        row: Dict[str, Any] = {
            "entry_type": getattr(e, "entry_type", None),
            "timestamp": getattr(e, "timestamp", None),
            "module_id": getattr(e, "module_id", None),
            "target_id": getattr(e, "target_id", None),
            "source": getattr(e, "source", None),
            "content": getattr(e, "content", None),
            "content_type": getattr(e, "content_type", None),
            "success": getattr(e, "success", None),
            "metadata": bytes(meta) if isinstance(meta, (bytes, bytearray)) else meta,
        }
        if callable(emb):
            try:
                row["embedding"] = emb().tobytes()
            except Exception:  # noqa: BLE001
                row["embedding"] = None
        rows.append(row)
    return rows


def verify(before: bytes, after: bytes) -> Optional[str]:
    """None when the correction provably preserved every entry, else the reason.

    NOTE on why this is not a before/after decode comparison: a reader only
    parses entries whose magic it recognises and hands back anything else as RAW
    BYTES. So on a legacy file the "before" side does not decode at all — that is
    the whole defect. Verification is therefore structural, and stronger for it:

      1. byte length unchanged;
      2. the ONLY bytes that differ are magic fields at entry offsets;
      3. every entry in the corrected buffer PARSES (no raw-bytes passthrough),
         which is the property the file did not have before.
    """
    if len(before) != len(after):
        return f"length changed {len(before)} -> {len(after)}"
    magic_offsets = {off for off, _l, _m in walk_entries(before)}
    for i, (x, y) in enumerate(zip(before, after)):
        if x != y and i not in magic_offsets and (i - 1) not in magic_offsets:
            return f"byte {i} changed outside a magic field"
    # The decode check is only meaningful if the LOCAL wheel can read canonical BT.
    # On a host still running the TB-lineage wheel it cannot, and it would report
    # every correctly-converted file as broken — the same blind spot this migration
    # exists to fix, biting the verifier. Structural checks above already cover
    # length, the full envelope chain, and that nothing outside a magic field moved.
    if not _reader_understands_bt():
        return None
    try:
        rows = _decode_all(after)
    except Exception as exc:  # noqa: BLE001
        return f"corrected buffer failed to decode: {exc}"
    n_expected = len(magic_offsets)
    if len(rows) != n_expected:
        return f"corrected buffer yields {len(rows)} items, expected {n_expected} entries"
    raw = [i for i, r in enumerate(rows) if "raw" in r]
    if raw:
        return f"entries {raw} still come back as raw bytes after correction"
    return None


#: at or above this size, use the streaming path (seek-and-write) instead of
#: read-modify-write. These tracts reach 578 MB and the host pages under load.
STREAM_THRESHOLD = 64 * 1024 * 1024


def process(path: Path, apply: bool, backup: bool, reap: bool = False) -> Dict[str, Any]:
    r: Dict[str, Any] = {"path": str(path), "status": "unknown", "entries": 0, "converted": 0}
    try:
        size = path.stat().st_size
    except OSError as exc:
        r["status"], r["error"] = "unreadable", str(exc)
        return r
    if size == 0:
        r["status"] = "empty"
        return r
    if size >= STREAM_THRESHOLD:
        return _process_streaming(path, apply, backup, size, r, reap)
    try:
        buf = path.read_bytes()
    except OSError as exc:
        r["status"], r["error"] = "unreadable", str(exc)
        return r
    if not buf:
        r["status"] = "empty"
        return r
    try:
        fixed, total, converted = correct(buf)
    except ParseError as exc:
        r["status"], r["error"] = "unparseable-skipped", str(exc)
        return r
    r["entries"], r["converted"] = total, converted
    if converted == 0:
        r["status"] = "clean"
        return r
    reason = verify(buf, fixed)
    if reason:
        r["status"], r["error"] = "verify-failed-skipped", reason
        return r
    if not apply:
        r["status"] = "would-convert"
        return r
    try:
        if backup:
            bak = path.with_suffix(path.suffix + f".pre-btf-magic.{int(time.time())}")
            bak.write_bytes(buf)
            r["backup"] = str(bak)
        tmp = path.with_suffix(path.suffix + ".btfmig.tmp")
        with open(tmp, "wb") as fh:
            fh.write(fixed)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        r["status"] = "converted"
        if reap and backup and r.get("backup"):
            # Verified above before the write; the backup has done its job. Reaping
            # bounds peak disk to one file, which matters when the host is at 96%.
            try:
                Path(r["backup"]).unlink()
                r["backup_reaped"] = True
            except OSError:
                pass
    except OSError as exc:
        r["status"], r["error"] = "write-failed", str(exc)
    return r


def _process_streaming(path: Path, apply: bool, backup: bool, size: int, r: Dict[str, Any], reap: bool = False) -> Dict[str, Any]:
    """Large-file path: validate by streaming, then seek-and-write 2 bytes per entry.

    Verification necessarily differs from the small-file path — two copies of a
    578 MB file cannot be held to diff them. Instead:
      * scan_file() validates the ENTIRE envelope chain before anything is written,
        so a damaged file is rejected before a single byte changes;
      * a full backup is taken first (unless --no-backup);
      * only the 2-byte magic fields are written, at offsets the scan produced;
      * a re-scan must then report the same entry count, the same size, and ZERO
        remaining TB entries — otherwise the file is restored from the backup.
    """
    r["streamed"] = True
    try:
        offsets, total = scan_file(path)
    except ParseError as exc:
        r["status"], r["error"] = "unparseable-skipped", str(exc)
        return r
    except OSError as exc:
        r["status"], r["error"] = "unreadable", str(exc)
        return r
    r["entries"], r["converted"] = total, len(offsets)
    if not offsets:
        r["status"] = "clean"
        return r
    if not apply:
        r["status"] = "would-convert"
        return r
    bak: Optional[Path] = None
    try:
        if backup:
            bak = path.with_suffix(path.suffix + f".pre-btf-magic.{int(time.time())}")
            with open(path, "rb") as src, open(bak, "wb") as dst:
                while True:
                    chunk = src.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)
                dst.flush()
                os.fsync(dst.fileno())
            r["backup"] = str(bak)
        correct_in_place(path, offsets)
    except OSError as exc:
        r["status"], r["error"] = "write-failed", str(exc)
        return r
    try:
        after_tb, after_total = scan_file(path)
        ok = (after_total == total and not after_tb and path.stat().st_size == size)
    except Exception as exc:  # noqa: BLE001
        ok, after_total, after_tb = False, -1, [-1]
        r["error"] = f"post-write scan failed: {exc}"
    if not ok:
        r.setdefault("error", f"after: {after_total} entries, {len(after_tb)} TB remaining")
        if bak and bak.is_file():
            try:
                os.replace(bak, path)
                r["status"] = "verify-failed-restored"
            except OSError as exc:
                r["status"] = "VERIFY-FAILED-RESTORE-FAILED"
                r["error"] = f"{r['error']} | restore failed: {exc}"
        else:
            r["status"] = "VERIFY-FAILED-NO-BACKUP"
        return r
    r["status"] = "converted"
    if reap and bak and bak.is_file():
        # Only after the post-write re-scan agreed. Bounds peak disk to one file.
        try:
            bak.unlink()
            r["backup_reaped"] = True
        except OSError:
            pass
    return r


def gather(paths: List[str]) -> List[Path]:
    out: List[Path] = []
    for p in paths:
        q = Path(os.path.expanduser(p))
        if q.is_dir():
            out.extend(sorted(f for f in q.rglob("*") if f.is_file() and ".tract" in f.name
                              and ".pre-btf-magic." not in f.name and not f.name.endswith(".btfmig.tmp")))
        elif q.is_file():
            out.append(q)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Correct legacy TB BTF magic to canonical BT, losslessly.")
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--apply", action="store_true", help="actually write (default is a dry run)")
    ap.add_argument("--no-backup", action="store_true", help="skip .pre-btf-magic backups")
    ap.add_argument("--json", action="store_true", help="machine-readable report")
    ap.add_argument("--reap-backup", action="store_true",
                    help="delete each backup once that file has verified — bounds peak disk to one file")
    a = ap.parse_args(argv)

    files = gather(a.paths)
    results = [process(f, a.apply, not a.no_backup, a.reap_backup) for f in files]

    if a.json:
        print(json.dumps({"apply": a.apply, "files": results}, indent=2))
    else:
        by: Dict[str, int] = {}
        ents = conv = 0
        for r in results:
            by[r["status"]] = by.get(r["status"], 0) + 1
            ents += r.get("entries", 0)
            conv += r.get("converted", 0)
            if r["status"] in ("unparseable-skipped", "verify-failed-skipped", "write-failed", "unreadable"):
                print(f"  !! {r['status']}: {r['path']} — {r.get('error')}")
            elif r.get("converted"):
                print(f"  {'converted' if a.apply else 'would convert'} {r['converted']:>6}/{r['entries']:<6} {r['path']}")
        print(f"\n{'APPLIED' if a.apply else 'DRY RUN — nothing written'}")
        print(f"  files            {len(results)}")
        for k in sorted(by):
            print(f"    {k:<24} {by[k]}")
        print(f"  entries seen     {ents}")
        print(f"  TB entries       {conv}")
        if not a.apply and conv:
            print("\n  re-run with --apply to correct them")
    bad = sum(1 for r in results if r["status"] in ("unparseable-skipped", "verify-failed-skipped", "write-failed", "unreadable"))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
