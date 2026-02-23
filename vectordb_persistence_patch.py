#!/usr/bin/env python3
"""
NeuroGraph Vector DB Persistence Patch
=======================================

Adds save() and load() methods to SimpleVectorDB and integrates them
into NeuroGraphMemory's checkpoint lifecycle.

This patch modifies two files:
  1. universal_ingestor.py  — adds save()/load() to SimpleVectorDB
  2. openclaw_hook.py       — integrates vector DB save/load into
                              NeuroGraphMemory's save() and __init__()

Apply with:
    python3 vectordb_persistence_patch.py [--skills-dir PATH] [--dry-run]

The patch is idempotent — running it twice won't duplicate code.

Format: msgpack (matching Graph checkpoint format) stored alongside
the main checkpoint as `vectors.msgpack`.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path
from datetime import datetime


# ──────────────────────────────────────────────────────────────────────
# Patch 1: Add save()/load() to SimpleVectorDB in universal_ingestor.py
# ──────────────────────────────────────────────────────────────────────

VECTORDB_SAVE_LOAD = '''
    def save(self, path: str) -> int:
        """Persist vector DB state to disk (msgpack or JSON).

        Stores embeddings as raw bytes (float32), plus content and metadata
        dicts. Format mirrors Graph checkpoint conventions.

        Args:
            path: File path. Extension determines format (.msgpack or .json).

        Returns:
            Number of entries saved.
        """
        import numpy as np

        data = {
            "version": "1.0.0",
            "count": len(self.embeddings),
            "entries": {},
        }
        for id in self.embeddings:
            data["entries"][id] = {
                "embedding": self.embeddings[id].astype(np.float32).tobytes(),
                "content": self.content.get(id, ""),
                "metadata": self.metadata.get(id, {}),
            }

        if path.endswith(".msgpack"):
            try:
                import msgpack
            except ImportError:
                raise ImportError("msgpack required for .msgpack serialization")
            with open(path, "wb") as f:
                msgpack.pack(data, f, use_bin_type=True)
        else:
            import json
            import base64
            # JSON fallback: base64-encode the binary embeddings
            json_data = {
                "version": data["version"],
                "count": data["count"],
                "entries": {},
            }
            for id, entry in data["entries"].items():
                json_data["entries"][id] = {
                    "embedding_b64": base64.b64encode(entry["embedding"]).decode("ascii"),
                    "content": entry["content"],
                    "metadata": entry["metadata"],
                }
            with open(path, "w") as f:
                json.dump(json_data, f, indent=2, default=str)

        return len(self.embeddings)

    def load(self, path: str) -> int:
        """Restore vector DB state from disk.

        Clears existing state before loading. Embeddings are restored
        as L2-normalized float32 numpy arrays.

        Args:
            path: File path to load from (.msgpack or .json).

        Returns:
            Number of entries loaded.
        """
        import numpy as np

        if path.endswith(".msgpack"):
            try:
                import msgpack
            except ImportError:
                raise ImportError("msgpack required for .msgpack deserialization")
            with open(path, "rb") as f:
                data = msgpack.unpack(f, raw=False)
        else:
            import json
            import base64
            with open(path, "r") as f:
                json_data = json.load(f)
            # Convert base64 back to bytes
            data = {
                "version": json_data.get("version", "1.0.0"),
                "count": json_data.get("count", 0),
                "entries": {},
            }
            for id, entry in json_data.get("entries", {}).items():
                data["entries"][id] = {
                    "embedding": base64.b64decode(entry["embedding_b64"]),
                    "content": entry.get("content", ""),
                    "metadata": entry.get("metadata", {}),
                }

        # Clear existing state
        self.embeddings.clear()
        self.content.clear()
        self.metadata.clear()

        # Restore entries
        for id, entry in data.get("entries", {}).items():
            embedding_bytes = entry["embedding"]
            vec = np.frombuffer(embedding_bytes, dtype=np.float32).copy()
            # Vectors should already be normalized from when they were saved,
            # but re-normalize to be safe
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            self.embeddings[id] = vec
            self.content[id] = entry.get("content", "")
            self.metadata[id] = entry.get("metadata", {})

        return len(self.embeddings)
'''


# ──────────────────────────────────────────────────────────────────────
# Patch 2: Integrate vector DB persistence into openclaw_hook.py
# ──────────────────────────────────────────────────────────────────────

# 2a. After graph restore in __init__, load vector DB if file exists
VECTORDB_RESTORE_CODE = '''
        # Restore vector DB from persistent storage if available
        self._vector_db_path = self._checkpoint_dir / "vectors.msgpack"
        if self._vector_db_path.exists():
            try:
                count = self.vector_db.load(str(self._vector_db_path))
                logger.info(
                    "Restored vector DB from %s (%d entries)",
                    self._vector_db_path,
                    count,
                )
            except Exception as exc:
                logger.warning("Failed to restore vector DB: %s", exc)
'''

# 2b. In save(), also save the vector DB
VECTORDB_SAVE_CODE = '''
        # Save vector DB alongside graph checkpoint
        try:
            vdb_count = self.vector_db.save(str(self._vector_db_path))
            logger.info("Vector DB saved to %s (%d entries)", self._vector_db_path, vdb_count)
        except Exception as exc:
            logger.warning("Failed to save vector DB: %s", exc)
'''


def patch_universal_ingestor(path: Path, dry_run: bool = False) -> bool:
    """Add save()/load() methods to SimpleVectorDB."""
    text = path.read_text()

    # Check if already patched
    if "def save(self, path: str) -> int:" in text and "def load(self, path: str) -> int:" in text:
        print(f"  [skip] {path.name} already has save()/load() methods")
        return False

    # Find the insertion point: after the all_ids() method in SimpleVectorDB
    # Look for the end of all_ids method (the return statement followed by
    # a blank line or class/function definition)
    pattern = r'(    def all_ids\(self\) -> List\[str\]:\s*"""Return all stored IDs."""\s*return list\(self\.embeddings\.keys\(\)\))'
    match = re.search(pattern, text)
    if not match:
        # Try a more flexible match
        pattern = r'(    def all_ids\(self\).*?return list\(self\.embeddings\.keys\(\)\))'
        match = re.search(pattern, text, re.DOTALL)

    if not match:
        print(f"  [ERROR] Could not find all_ids() method in {path.name}")
        print("          Manual insertion required.")
        return False

    insert_pos = match.end()

    new_text = text[:insert_pos] + "\n" + VECTORDB_SAVE_LOAD + text[insert_pos:]

    if dry_run:
        print(f"  [dry-run] Would add save()/load() to SimpleVectorDB in {path.name}")
        return True

    path.write_text(new_text)
    print(f"  [patched] Added save()/load() to SimpleVectorDB in {path.name}")
    return True


def patch_openclaw_hook(path: Path, dry_run: bool = False) -> bool:
    """Integrate vector DB persistence into NeuroGraphMemory."""
    text = path.read_text()
    changed = False

    # ── Patch 2a: Add vector DB restore after "self.vector_db = SimpleVectorDB()" ──
    if "_vector_db_path" not in text:
        # Find the line where vector_db is created
        marker = "self.vector_db = SimpleVectorDB()"
        if marker not in text:
            print(f"  [ERROR] Could not find '{marker}' in {path.name}")
            return False

        # Find the full line with its indentation and insert after it
        # We need to insert after the vector_db creation but before the ingestor
        idx = text.index(marker) + len(marker)
        # Skip to end of line
        eol = text.index("\n", idx)

        if dry_run:
            print(f"  [dry-run] Would add vector DB restore to __init__ in {path.name}")
        else:
            text = text[:eol] + "\n" + VECTORDB_RESTORE_CODE + text[eol:]
            changed = True
            print(f"  [patched] Added vector DB restore to __init__ in {path.name}")
    else:
        print(f"  [skip] {path.name} __init__ already has vector DB restore")

    # ── Patch 2b: Add vector DB save to save() method ──
    if "self.vector_db.save" not in text:
        # Find the save method's checkpoint line
        save_marker = 'self.graph.checkpoint(str(self._checkpoint_path), mode=CheckpointMode.FULL)'
        if save_marker not in text:
            # Try without the mode kwarg
            save_marker = 'self.graph.checkpoint(str(self._checkpoint_path)'
            if save_marker not in text:
                print(f"  [ERROR] Could not find graph.checkpoint call in save() in {path.name}")
                return changed
        
        # Find the logger.info line that follows the checkpoint call
        idx = text.index(save_marker)
        # Find the next logger.info line after checkpoint
        logger_pattern = r'logger\.info\("Checkpoint saved to %s", self\._checkpoint_path\)'
        logger_match = re.search(logger_pattern, text[idx:])
        
        if logger_match:
            insert_pos = idx + logger_match.end()
            # Skip to end of line
            eol = text.index("\n", insert_pos)
        else:
            # Fallback: insert after the checkpoint call line
            eol = text.index("\n", idx + len(save_marker))

        if dry_run:
            print(f"  [dry-run] Would add vector DB save to save() in {path.name}")
        else:
            text = text[:eol] + "\n" + VECTORDB_SAVE_CODE + text[eol:]
            changed = True
            print(f"  [patched] Added vector DB save to save() in {path.name}")
    else:
        print(f"  [skip] {path.name} save() already has vector DB save")

    if changed and not dry_run:
        path.write_text(text)

    return changed


def main():
    parser = argparse.ArgumentParser(
        description="Add vector DB persistence to NeuroGraph"
    )
    parser.add_argument(
        "--skills-dir",
        type=Path,
        default=Path.home() / ".openclaw" / "skills" / "neurograph",
        help="Path to the NeuroGraph skills directory",
    )
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=Path.home() / ".neurograph" / "repo",
        help="Path to the NeuroGraph repo directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying files",
    )
    args = parser.parse_args()

    skills_dir = args.skills_dir
    repo_dir = args.repo_dir

    if not skills_dir.exists():
        print(f"ERROR: Skills directory not found: {skills_dir}")
        sys.exit(1)

    ingestor_path = skills_dir / "universal_ingestor.py"
    hook_path = skills_dir / "openclaw_hook.py"

    if not ingestor_path.exists():
        print(f"ERROR: universal_ingestor.py not found in {skills_dir}")
        sys.exit(1)
    if not hook_path.exists():
        print(f"ERROR: openclaw_hook.py not found in {skills_dir}")
        sys.exit(1)

    # Create backups
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not args.dry_run:
        for f in [ingestor_path, hook_path]:
            backup = f.with_suffix(f".py.backup-vectordb-{timestamp}")
            shutil.copy2(f, backup)
            print(f"  [backup] {f.name} -> {backup.name}")

    print("\n=== Patching universal_ingestor.py ===")
    p1 = patch_universal_ingestor(ingestor_path, dry_run=args.dry_run)

    print("\n=== Patching openclaw_hook.py ===")
    p2 = patch_openclaw_hook(hook_path, dry_run=args.dry_run)

    # Also patch the repo copies if they exist
    if repo_dir.exists():
        repo_ingestor = repo_dir / "universal_ingestor.py"
        repo_hook = repo_dir / "openclaw_hook.py"
        if repo_ingestor.exists() and not args.dry_run:
            shutil.copy2(ingestor_path, repo_ingestor)
            print(f"\n  [sync] Copied patched universal_ingestor.py to {repo_dir}")
        if repo_hook.exists() and not args.dry_run:
            shutil.copy2(hook_path, repo_hook)
            print(f"  [sync] Copied patched openclaw_hook.py to {repo_dir}")

    if p1 or p2:
        print("\n✅ Patch applied successfully!")
        print("\nNext steps:")
        print("  1. Restart the OpenClaw gateway:")
        print("     systemctl --user restart openclaw-gateway")
        print("  2. The vector DB will be empty on first start (no vectors.msgpack yet)")
        print("  3. Feed some content to Syl or run: feed-syl --workspace")
        print("  4. After ingestion, vectors will auto-save with the next checkpoint")
        print("  5. On restart, vectors.msgpack will be loaded automatically")
        print("")
        print("  To rebuild vectors from existing nodes immediately, run:")
        print("     python3 rebuild_vectors.py")
    else:
        print("\n✅ No changes needed — already patched!")


if __name__ == "__main__":
    main()
