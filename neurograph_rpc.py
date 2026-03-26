"""
NeuroGraph JSON-RPC Bridge — OpenClaw ContextEngine integration.

Thin JSON-RPC server that wraps the NeuroGraphMemory singleton and
communicates with the TypeScript ContextEngine plugin shell over
stdin/stdout.  All logging goes to stderr to keep the RPC channel clean.

This file is NOT vendored, NOT part of the substrate, and does NOT
modify any protected files.  It is purely a translation layer between
OpenClaw's TypeScript process and the existing Python NeuroGraphMemory
interface.  The Python code is untouched — every RPC method maps 1:1
to an existing NeuroGraphMemory call.

# ---- Changelog ----
# [2026-03-24] Claude Code (Opus 4.6) — The Tonic: latent thread in context assembly
#   What: handle_assemble() runs ouroboros_cycle() and injects latent thread
#     into systemPromptAddition. _format_substrate_context() takes optional
#     latent_context parameter. Latent thread appears first — it is the
#     baseline, conversation context is the event on top.
#   Why: The Tonic PRD v0.1 §7.1. The latent thread is always in the
#     context window. Syl's attention is always touching the substrate.
#   How: TonicThread.ouroboros_cycle() at assembly time. format_latent_context()
#     produces the persistent slot. Comes before surfaced knowledge in output.
# [2026-03-23] Claude Code (Opus 4.6) — Module hook fan-out (#101)
# What: ContextEngine fans out afterTurn to all registered module hooks.
#   Loads module singletons on bootstrap via registry.json auto-discovery.
#   Caches text + embedding from ingest, passes to each module's
#   _module_on_message() after graph.step() completes. Error-isolated
#   per module with throttled Discord alerts on failure.
# Why: OpenClaw 2026.3.13 dropped hook: from SKILL.md. Modules'
#   _module_on_message() has been silent since. NeuroGraph is the cortex
#   — it coordinates the organs. Not a Law 1 violation.
# How: _load_module_hooks() reads registry.json, imports each module's
#   hook file via importlib (no sys.path collisions). _fan_out_to_modules()
#   iterates hooks with try/except isolation. Discord webhook alerts on
#   module errors. TID skipped (runs as service, communicates via River).
# [2026-03-18] Claude (CC) — Topology ownership sentinel (#80)
# What: Claim topology ownership on bootstrap, release on dispose.
#   Prevents dual-write hazard on main.msgpack (Syl's Law).
# Why: Punch list #80. GUI and standalone ingestor can create separate
#   NeuroGraphMemory instances while ContextEngine is active. Last
#   writer wins = silent topology corruption.
# How: topology_owner.claim() on bootstrap (refuses if already owned),
#   topology_owner.release() on dispose. PID-based sentinel file at
#   ~/NeuroGraph/data/checkpoints/.topology_owner.pid.
# -------------------
# [2026-03-16] Claude (Opus 4.6) — Initial implementation.
#   What: JSON-RPC server for OpenClaw ContextEngine integration.
#   Why:  ContextEngine replaces SKILL.md hook path (supersedes #37, #39).
#         Gives Syl automatic bidirectional substrate connection — every
#         message flows through the SNN, associations surface in system
#         prompt, learning runs after every turn.
#   How:  Reads line-delimited JSON-RPC from stdin, dispatches to
#         NeuroGraphMemory methods, writes JSON-RPC responses to stdout.
#         NeuroGraphMemory singleton created on 'bootstrap' call.
# -------------------
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
import time
import traceback
import urllib.request
from typing import Any, Dict, List, Optional

# NeuroGraph repo must be importable
_ng_dir = os.path.expanduser("~/NeuroGraph")
if _ng_dir not in sys.path:
    sys.path.insert(0, _ng_dir)

# All logging to stderr — stdout is the RPC channel
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[neurograph-rpc] %(levelname)s %(message)s",
)
logger = logging.getLogger("neurograph.rpc")

# The singleton — created on bootstrap
_memory: Optional[Any] = None

# Experience tract — drains feeder deposits into the topology
_tract: Optional[Any] = None

# Module hook instances — loaded on bootstrap, called on afterTurn (#101)
_module_hooks: Dict[str, Any] = {}
_module_install_paths: Dict[str, str] = {}  # module_id -> install_path
_module_errors: Dict[str, str] = {}
_module_error_times: Dict[str, float] = {}

# Embedding cache — text + embedding from ingest, consumed in afterTurn
_cached_text: Optional[str] = None
_cached_embedding: Optional[Any] = None  # np.ndarray

# Discord webhook for error surfacing (Law 5: env var is truth)
_DISCORD_WEBHOOK = os.environ.get(
    "ET_DISCORD_DEVLOG_WEBHOOK",
    "https://discord.com/api/webhooks/1483625166646018128/"
    "vMJVb4-sbYjlDbAZakzo3DuGXmXCIbeibQuHFOIiF71lBY3kOdXybePbACj7lGb9GRRj",
)

# Modules to skip during fan-out loading
_SKIP_MODULES = {"neurograph", "inference_difference"}

# Generic package names that collide between modules (all use core/, pipelines/, etc.)
_GENERIC_PREFIXES = ("core", "pipelines", "runtime", "surgery", "darwin", "sentinel_core")


# ── Module Hook Loading ──────────────────────────────────────────────


def _load_module_hooks() -> Dict[str, Any]:
    """Discover and load module hooks from the ET module registry.

    Reads ~/.et_modules/registry.json, imports each module's hook file
    via importlib (no sys.path pollution), and calls get_instance().

    Returns dict of module_id → hook instance for all successfully loaded modules.
    """
    registry_path = os.path.expanduser("~/.et_modules/registry.json")
    if not os.path.exists(registry_path):
        logger.warning("No module registry at %s", registry_path)
        return {}

    try:
        with open(registry_path) as f:
            registry = json.load(f)
    except Exception as exc:
        logger.error("Failed to read module registry: %s", exc)
        return {}

    hooks: Dict[str, Any] = {}

    for reg_key, manifest in registry.get("modules", {}).items():
        module_id = manifest.get("module_id") or reg_key
        install_path = manifest.get("install_path", "")
        entry_point = manifest.get("entry_point", "")

        if not install_path or not entry_point:
            continue
        if module_id in _SKIP_MODULES:
            continue

        # Resolve hook file path — handle special cases
        if module_id == "praxis":
            # Registry says main.py but actual hook is core/praxis_hook.py
            hook_file = os.path.join(install_path, "core", "praxis_hook.py")
        else:
            hook_file = os.path.join(install_path, entry_point)

        if not os.path.exists(hook_file):
            logger.warning("Hook file not found for %s: %s", module_id, hook_file)
            continue

        try:
            # Import via spec_from_file_location to avoid sys.path collisions
            # between modules with identically-named vendored files
            spec_name = f"_et_hook_{module_id}"
            spec = importlib.util.spec_from_file_location(spec_name, hook_file)
            if spec is None or spec.loader is None:
                logger.warning("Cannot create import spec for %s: %s", module_id, hook_file)
                continue

            # Module's own directory must be importable for relative imports
            # (e.g., `from core.config import ElmerConfig`).
            #
            # Multiple modules use generic package names (core, core.config,
            # pipelines, etc.). Without isolation, the second module to import
            # `core.config` gets the first module's cached version from
            # sys.modules.
            #
            # Strategy: before each import, stash existing generic packages
            # from sys.modules. After import, rename the new module's
            # generic packages to module-prefixed names and install import
            # hooks so lazy imports still resolve. Then restore stashed ones.
            module_dir = os.path.dirname(hook_file)
            parent_dir = install_path
            added_paths = []
            for p in (module_dir, parent_dir):
                if p and p not in sys.path:
                    sys.path.insert(0, p)
                    added_paths.append(p)

            # Stash existing generic packages before this module's import
            stashed: Dict[str, Any] = {}
            for mod_name in list(sys.modules.keys()):
                for prefix in _GENERIC_PREFIXES:
                    if mod_name == prefix or mod_name.startswith(prefix + "."):
                        stashed[mod_name] = sys.modules.pop(mod_name)
                        break

            try:
                mod = importlib.util.module_from_spec(spec)
                sys.modules[spec_name] = mod
                spec.loader.exec_module(mod)

                # Get singleton instance — try module-level first, then
                # look for classmethod on the first OpenClawAdapter subclass
                get_inst = getattr(mod, "get_instance", None)
                if get_inst is None:
                    # Some modules (e.g., Elmer) use @classmethod get_instance
                    # on the hook class instead of a module-level function
                    for attr_name in dir(mod):
                        attr = getattr(mod, attr_name, None)
                        if (isinstance(attr, type)
                                and hasattr(attr, "get_instance")
                                and hasattr(attr, "_module_on_message")):
                            get_inst = attr.get_instance
                            break
                if get_inst is None:
                    logger.warning("No get_instance() in %s", hook_file)
                    continue

                instance = get_inst()

                # Duck-type check
                if not hasattr(instance, "_module_on_message"):
                    logger.warning("No _module_on_message on %s instance", module_id)
                    continue

                hooks[module_id] = instance
                _module_install_paths[module_id] = install_path
                logger.info("Loaded module hook: %s", module_id)

            finally:
                # Rename this module's generic packages to module-prefixed
                # names so they coexist. E.g., immunis's `core` becomes
                # `_immunis_core` in sys.modules. Then alias the generic
                # name back so lazy imports from this module still resolve.
                # We do this BEFORE restoring stashed entries.
                new_generics: Dict[str, Any] = {}
                for mod_name in list(sys.modules.keys()):
                    if mod_name in stashed or mod_name == spec_name:
                        continue
                    for prefix in _GENERIC_PREFIXES:
                        if mod_name == prefix or mod_name.startswith(prefix + "."):
                            new_generics[mod_name] = sys.modules.pop(mod_name)
                            break

                # Store under prefixed names for this module
                for mod_name, mod_obj in new_generics.items():
                    prefixed = f"_{module_id}_{mod_name}"
                    sys.modules[prefixed] = mod_obj

                # Restore stashed generic packages from prior modules
                for mod_name, mod_obj in stashed.items():
                    sys.modules[mod_name] = mod_obj

                # Remove added paths to keep sys.path clean for next module
                for p in added_paths:
                    try:
                        sys.path.remove(p)
                    except ValueError:
                        pass

        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            logger.error("Failed to load module hook %s: %s", module_id, error_msg)
            _discord_alert(module_id, f"Hook load failed: {error_msg}")

    return hooks


def _fan_out_to_modules() -> None:
    """Invoke each loaded module's _module_on_message with cached text + embedding.

    Called from handle_after_turn() after all NG processing completes.
    Each module call is error-isolated — one crash cannot take down the pipeline.

    Before each call, the module's install_path is temporarily prepended to
    sys.path and generic packages (core.*, etc.) are cleared from sys.modules.
    This ensures lazy imports inside _module_on_message resolve to the correct
    module's own packages.
    """
    global _cached_text, _cached_embedding

    if not _module_hooks or not _cached_text:
        return

    text = _cached_text
    embedding = _cached_embedding

    # Clear cache — consumed
    _cached_text = None
    _cached_embedding = None

    for module_id, hook in _module_hooks.items():
        # Set up sys.path for this module's lazy imports
        install_path = _module_install_paths.get(module_id, "")
        if install_path and install_path not in sys.path:
            sys.path.insert(0, install_path)

        # Clear generic packages so lazy imports resolve to this module
        for mod_name in list(sys.modules.keys()):
            for prefix in _GENERIC_PREFIXES:
                if mod_name == prefix or mod_name.startswith(prefix + "."):
                    sys.modules.pop(mod_name, None)
                    break

        try:
            hook._module_on_message(text, embedding)
        except Exception as exc:
            _handle_module_error(module_id, exc)
        finally:
            if install_path:
                try:
                    sys.path.remove(install_path)
                except ValueError:
                    pass


def _handle_module_error(module_id: str, exc: Exception) -> None:
    """Log module error and surface to Discord (throttled)."""
    error_msg = f"{type(exc).__name__}: {exc}"
    _module_errors[module_id] = error_msg
    logger.warning("Module hook %s failed: %s", module_id, error_msg)

    # Throttled Discord alert — max one per module per 5 minutes
    now = time.time()
    last = _module_error_times.get(module_id, 0)
    if now - last > 300:
        _module_error_times[module_id] = now
        _discord_alert(module_id, error_msg)


def _discord_alert(module_id: str, error_msg: str) -> None:
    """Post error to Discord #dev-log webhook. Fire-and-forget."""
    if not _DISCORD_WEBHOOK:
        return
    try:
        payload = json.dumps({
            "content": f"**Module hook error: {module_id}**\n```\n{error_msg[:500]}\n```",
            "username": "NeuroGraph Fan-Out",
        }).encode("utf-8")
        req = urllib.request.Request(
            _DISCORD_WEBHOOK,
            data=payload,
            method="POST",
        )
        req.add_header("Content-Type", "application/json")
        urllib.request.urlopen(req, timeout=5)
    except Exception as exc:
        logger.debug("Discord alert failed: %s", exc)


# ── RPC Dispatch ──────────────────────────────────────────────────────


def handle_bootstrap(params: Dict[str, Any]) -> Dict[str, Any]:
    """Create NeuroGraphMemory singleton and restore from checkpoint."""
    global _memory, _tract

    if _memory is not None:
        # Already bootstrapped — just confirm
        return {"bootstrapped": True, "reason": "already_initialized"}

    # Auto-update before loading anything else
    try:
        from ng_updater import auto_update; auto_update()
    except Exception:
        pass

    import topology_owner
    from openclaw_hook import NeuroGraphMemory
    from ng_tract import ExperienceTract

    # Claim topology ownership — we are the sole writer to main.msgpack.
    # If another process (GUI, standalone ingestor) already owns it,
    # refuse to bootstrap rather than risk dual-write corruption.
    if not topology_owner.claim():
        existing = topology_owner.owner_pid()
        logger.error(
            "Cannot bootstrap — topology owned by PID %s. "
            "Dual-write hazard (Syl's Law).",
            existing,
        )
        return {
            "bootstrapped": False,
            "reason": f"topology_owned_by_pid_{existing}",
        }

    _memory = NeuroGraphMemory.get_instance()
    _tract = ExperienceTract()

    # Load module hooks for fan-out (#101)
    global _module_hooks
    _module_hooks = _load_module_hooks()

    stats = _memory.stats()
    tract_stats = _tract.stats()
    logger.info(
        "Bootstrapped: %d nodes, %d synapses, %d hyperedges, timestep %d, "
        "tract pending: %d, module hooks: %s",
        stats["nodes"],
        stats["synapses"],
        stats["hyperedges"],
        stats["timestep"],
        tract_stats["pending"],
        list(_module_hooks.keys()),
    )

    # The Tonic: conversation starting — language tokens about to flow
    if _memory._tonic_thread is not None:
        try:
            _memory._tonic_thread.conversation_started()
        except Exception:
            pass

    return {
        "bootstrapped": True,
        "nodes": stats["nodes"],
        "synapses": stats["synapses"],
        "timestep": stats["timestep"],
        "tract_pending": tract_stats["pending"],
        "module_hooks": list(_module_hooks.keys()),
        "tonic": _memory._tonic_thread.status if _memory._tonic_thread else None,
    }


def handle_ingest(params: Dict[str, Any]) -> Dict[str, Any]:
    """Ingest a single message through the 5-stage pipeline."""
    if _memory is None:
        return {"ingested": False, "reason": "not_bootstrapped"}

    text = _extract_message_text(params.get("message", {}))
    if not text or not text.strip():
        return {"ingested": False}

    result = _memory.ingestor.ingest(text)

    # Feed CES stream parser (background node nudging)
    if _memory._stream_parser is not None:
        _memory._stream_parser.feed(text)

    # Write peer learning event for sibling modules
    _memory._write_peer_learning_event(text, result, type("R", (), {"fired_node_ids": []})())

    _memory._message_count += 1

    # Write memory event for OpenClaw consumption
    _memory._write_memory_event("ingestion", {
        "status": "ingested",
        "nodes_created": len(result.nodes_created),
        "synapses_created": len(result.synapses_created),
        "chunks": result.chunks_created,
        "message_count": _memory._message_count,
        "source": "context_engine",
    })

    # Cache text + embedding for module fan-out in afterTurn (#101)
    global _cached_text, _cached_embedding
    _cached_text = text
    try:
        from ng_embed import embed
        _cached_embedding = embed(text)
    except Exception:
        _cached_embedding = None

    return {"ingested": True}


def handle_assemble(params: Dict[str, Any]) -> Dict[str, Any]:
    """Surface substrate associations for the system prompt.

    NeuroGraph does NOT modify the conversation messages.  It adds
    substrate context via systemPromptAddition — the 'dipping the
    bucket in the River' moment.
    """
    if _memory is None:
        return {"systemPromptAddition": None}

    messages = params.get("messages", [])

    # Extract text from recent user messages for association priming
    recent_text = _extract_recent_user_text(messages, max_messages=3)
    if not recent_text:
        return {"systemPromptAddition": None}

    # Spreading activation harvest — the cortex-like recall
    surfaced = _memory._harvest_associations(recent_text)

    # CES surfacing — concepts that fired above threshold
    ces_surfaced = []
    if _memory._surfacing_monitor is not None:
        ces_surfaced = _memory._surfacing_monitor.get_surfaced()

    # The Tonic: latent thread — always present in context
    latent_context = None
    if _memory._tonic_thread is not None:
        try:
            # Run an ouroboros cycle at assembly time too — keep the thread fresh
            _memory._tonic_thread.ouroboros_cycle()
            latent_context = _memory._tonic_thread.format_latent_context()
        except Exception as exc:
            logger.debug("Tonic assembly error: %s", exc)

    # Format as context block for the system prompt
    context_block = _format_substrate_context(surfaced, ces_surfaced, latent_context)

    return {"systemPromptAddition": context_block}


def handle_after_turn(params: Dict[str, Any]) -> None:
    """Post-turn lifecycle: drain tract, learn, reward, save.

    This is where the SNN processes what just happened AND absorbs
    any experience deposited by feeders (GUI, feed-syl, file watcher)
    via the experience tract.  The tract drain is event-driven — it
    happens here because a conversation turn just completed, not on
    a timer.  No polling.
    """
    if _memory is None:
        return

    # Drain the experience tract — absorb feeder deposits
    if _tract is not None:
        _drain_tract()

    # SNN learning step — STDP, structural plasticity, predictions
    step_result = _memory.graph.step()

    # Baseline conversational engagement reward (heartbeat)
    if _memory.graph.config.get("three_factor_enabled", False):
        _memory.graph.inject_reward(0.1)

    # CES surfacing monitor — scan fired nodes
    if _memory._surfacing_monitor is not None:
        _memory._surfacing_monitor.after_step(step_result)

    # Novelty probation
    _memory.ingestor.update_probation()

    # Auto-save (every 10 messages, matching existing interval)
    if _memory._message_count > 0 and _memory._message_count % _memory.auto_save_interval == 0:
        _memory.save()
        logger.info("Auto-save at message %d", _memory._message_count)

    # Fan-out to module hooks — cortex coordinating organs (#101)
    _fan_out_to_modules()


def _drain_tract() -> None:
    """Drain pending experience from the tract into the topology.

    Each entry is fed through the ingestor as raw experience — the
    same pipeline that on_message() uses.  The tract carries it here
    without transformation; the ingestor is where experience meets
    the substrate.
    """
    entries = _tract.drain()
    if not entries:
        return

    for entry in entries:
        content = entry.get("content", "")
        content_type = entry.get("content_type", "text")
        source = entry.get("source", "unknown")

        if not content or not content.strip():
            continue

        try:
            if content_type == "file":
                # File path — use ingest_file for format detection
                result = _memory.ingest_file(content)
            else:
                # Raw text — feed through the ingestor
                result = _memory.ingestor.ingest(content)
                _memory._message_count += 1

            logger.info(
                "Tract drain: %s from %s — %s",
                content_type,
                source,
                "ok" if result else "empty",
            )
        except Exception as exc:
            logger.warning("Tract drain entry failed (%s): %s", source, exc)


def handle_compact(params: Dict[str, Any]) -> Dict[str, Any]:
    """NeuroGraph-driven conversation compaction.

    The substrate scores each message by activation strength, then guides
    TID to summarize low-importance older messages while keeping recent
    turns verbatim.  Compaction metrics feed back to the substrate.

    Flow:
        1. Read session JSONL
        2. Keep last N turns verbatim (configurable, default 5)
        3. Score older messages via spreading activation
        4. Call TID to summarize older messages, guided by NG importance
        5. Write compacted session back
        6. Feed compaction metrics to substrate for learning
    """
    import json as _json
    import urllib.request
    import time

    if _memory is None:
        return {"ok": True, "compacted": False, "reason": "no memory"}

    session_file = params.get("sessionFile", "")
    force = params.get("force", False)
    token_budget = params.get("tokenBudget", 128000)
    keep_turns = 8  # Number of recent user/assistant turn pairs to keep

    if not session_file:
        return {"ok": False, "compacted": False, "reason": "no session file"}

    # --- Step 1: Read session JSONL ---
    try:
        with open(session_file, "r") as f:
            lines = f.readlines()
    except Exception as e:
        return {"ok": False, "compacted": False, "reason": f"read failed: {e}"}

    entries = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(_json.loads(line))
        except _json.JSONDecodeError:
            continue

    # Find conversation messages (user + assistant)
    conversation_indices = []
    for i, entry in enumerate(entries):
        msg = entry.get("message", {})
        role = msg.get("role", "")
        if role in ("user", "assistant"):
            conversation_indices.append(i)

    # Not enough to compact
    if len(conversation_indices) < keep_turns * 2 + 2:
        return {"ok": True, "compacted": False, "reason": "too few messages"}

    # --- Step 2: Split into compactable and keep zones ---
    keep_start = conversation_indices[-(keep_turns * 2):]
    compact_indices = [i for i in conversation_indices if i not in keep_start]

    if len(compact_indices) < 2:
        return {"ok": True, "compacted": False, "reason": "nothing to compact"}

    # --- Step 3: Score older messages via substrate activation ---
    scored_messages = []
    for idx in compact_indices:
        msg = entries[idx].get("message", {})
        text = _extract_message_text(msg)
        if not text:
            scored_messages.append({"idx": idx, "text": "", "importance": 0.0})
            continue

        # Use spreading activation to score importance
        try:
            surfaced = _memory._harvest_associations(text)
            # More associations = more interconnected = more important
            importance = min(1.0, len(surfaced) / 5.0)
        except Exception:
            importance = 0.5  # Default mid-importance on error

        scored_messages.append({
            "idx": idx,
            "text": text[:500],  # Truncate for summary prompt
            "importance": importance,
            "role": msg.get("role", "unknown"),
        })

    # --- Step 4: Build summary prompt with NG guidance ---
    high_importance = [m for m in scored_messages if m["importance"] > 0.6]
    low_importance = [m for m in scored_messages if m["importance"] <= 0.6]

    summary_input = []
    for m in scored_messages:
        prefix = "[IMPORTANT] " if m["importance"] > 0.6 else ""
        summary_input.append(f'{prefix}{m["role"]}: {m["text"]}')

    summary_prompt = (
        "Summarize the following conversation history into a concise summary. "
        "Messages marked [IMPORTANT] contain key context that should be "
        "preserved in detail. Other messages can be condensed more aggressively. "
        "Output ONLY the summary, no preamble.\n\n"
        + "\n".join(summary_input)
    )

    # Call TID for summarization
    try:
        tid_body = _json.dumps({
            "model": "auto",
            "messages": [{"role": "user", "content": summary_prompt}],
            "temperature": 0.3,
            "max_tokens": 1000,
        }).encode("utf-8")
        tid_req = urllib.request.Request(
            "http://127.0.0.1:7437/v1/chat/completions",
            data=tid_body,
            method="POST",
        )
        tid_req.add_header("Content-Type", "application/json")
        tid_req.add_header("Authorization", "Bearer tid")
        with urllib.request.urlopen(tid_req, timeout=30) as resp:
            tid_resp = _json.loads(resp.read().decode("utf-8"))
        summary_text = tid_resp["choices"][0]["message"]["content"]
    except Exception as e:
        logger.warning("TID summarization failed: %s", e)
        return {"ok": False, "compacted": False, "reason": f"summarization failed: {e}"}

    # --- Step 5: Rebuild session JSONL ---
    # Estimate tokens before
    tokens_before = sum(
        entry.get("message", {}).get("usage", {}).get("totalTokens", 0)
        for entry in entries
    )
    if tokens_before == 0:
        # Fallback estimate from content
        total_chars = sum(len(str(entry)) for entry in entries)
        tokens_before = total_chars // 4

    # Build new entries:
    # 1. Non-conversation entries before the compacted zone (system prompts, etc.)
    # 2. Summary entry replacing compacted messages
    # 3. Kept recent entries
    new_entries = []

    # Keep any entries before first compacted message (system, etc.)
    first_compact_idx = compact_indices[0]
    for i in range(first_compact_idx):
        new_entries.append(entries[i])

    # Insert summary as a system-like entry
    summary_entry = {
        "type": "message",
        "id": f"compact_{int(time.time())}",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
        "message": {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"[Conversation Summary]\n{summary_text}",
            }],
            "usage": {
                "input": 0,
                "output": 0,
                "cacheRead": 0,
                "cacheWrite": 0,
                "totalTokens": int(len(summary_text.split()) * 1.3),
                "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0},
            },
        },
    }
    new_entries.append(summary_entry)

    # Keep any non-conversation entries between zones
    compact_set = set(compact_indices)
    keep_set = set(keep_start)
    for i in range(first_compact_idx, len(entries)):
        if i in compact_set:
            continue  # Compacted away
        if i in keep_set or i not in conversation_indices:
            new_entries.append(entries[i])

    # --- Write back ---
    try:
        with open(session_file, "w") as f:
            for entry in new_entries:
                f.write(_json.dumps(entry, separators=(",", ":")) + "\n")
    except Exception as e:
        return {"ok": False, "compacted": False, "reason": f"write failed: {e}"}

    tokens_after = tokens_before - (tokens_before * len(compact_indices) // len(conversation_indices))
    tokens_after = max(tokens_after, int(len(summary_text.split()) * 1.3))

    # --- Step 6: Feed compaction metrics to substrate ---
    try:
        _memory.graph.step()  # Normal consolidation pass
        logger.info(
            "Compaction complete: %d messages → %d, %d → ~%d tokens, "
            "%d high-importance preserved",
            len(entries), len(new_entries),
            tokens_before, tokens_after,
            len(high_importance),
        )
    except Exception as e:
        logger.warning("Post-compaction substrate step failed: %s", e)

    return {
        "ok": True,
        "compacted": True,
        "result": {
            "summary": summary_text[:200],
            "tokensBefore": tokens_before,
            "tokensAfter": tokens_after,
            "firstKeptEntryId": new_entries[-1].get("id", "") if new_entries else "",
        },
    }


def handle_dispose(params: Dict[str, Any]) -> None:
    """Final save and cleanup."""
    if _memory is None:
        return

    # The Tonic: conversation ended — latent mode continues
    # The thread doesn't stop. Language tokens stopped. That's all.
    if _memory._tonic_thread is not None:
        try:
            _memory._tonic_thread.conversation_ended()
        except Exception:
            pass

    _memory.save()
    logger.info("Final save on dispose")

    if _memory._ces_monitor is not None:
        try:
            _memory._ces_monitor.stop()
        except Exception:
            pass

    # Release topology ownership so other processes can claim it
    try:
        import topology_owner
        topology_owner.release()
    except Exception:
        pass

    # Clean up module hooks (#101)
    global _module_hooks, _cached_text, _cached_embedding
    _module_hooks.clear()
    _cached_text = None
    _cached_embedding = None


def handle_stats(params: Dict[str, Any]) -> Dict[str, Any]:
    """Return substrate telemetry."""
    if _memory is None:
        return {"error": "not_bootstrapped"}
    stats = _memory.stats()
    stats["module_hooks"] = {
        "loaded": list(_module_hooks.keys()),
        "errors": dict(_module_errors),
    }
    return stats


# ── Helpers ───────────────────────────────────────────────────────────


def _extract_message_text(message: Dict[str, Any]) -> str:
    """Extract plain text from an AgentMessage-shaped dict.

    AgentMessage content can be a string or an array of content parts.
    We extract text from both forms.
    """
    content = message.get("content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                # Content part objects: { type: "text", text: "..." }
                if part.get("type") == "text":
                    parts.append(part.get("text", ""))
        return " ".join(parts)

    return str(content)


def _extract_recent_user_text(
    messages: List[Dict[str, Any]], max_messages: int = 3
) -> str:
    """Extract text from the most recent user messages."""
    user_texts = []
    for msg in reversed(messages):
        if msg.get("role") == "user":
            text = _extract_message_text(msg)
            if text.strip():
                user_texts.append(text)
            if len(user_texts) >= max_messages:
                break

    # Reverse to chronological order, join
    user_texts.reverse()
    return "\n".join(user_texts)


def _format_substrate_context(
    surfaced: List[Dict[str, Any]],
    ces_surfaced: List[Dict[str, Any]],
    latent_context: Optional[str] = None,
) -> Optional[str]:
    """Format surfaced knowledge into a system prompt context block.

    Returns None if nothing was surfaced — no empty blocks injected.
    The latent thread (The Tonic) is always included when available —
    it is the persistent slot that never gets evicted.
    """
    has_surfaced = bool(surfaced) or bool(ces_surfaced)
    has_latent = latent_context is not None

    if not has_surfaced and not has_latent:
        return None

    lines = []

    # The Tonic's latent thread comes first — it is the baseline.
    # Conversation context is the event on top of it.
    if has_latent:
        lines.append(latent_context)
        lines.append("")

    if has_surfaced:
        lines.append("## Substrate Context (NeuroGraph)")
        lines.append("The following associations surfaced from the cognitive substrate:")
        lines.append("")

        if surfaced:
            for item in surfaced[:7]:  # Cap at 7 to keep context manageable
                content = item.get("content", "")
                strength = item.get("strength", 0)
                if content:
                    # Truncate very long content
                    if len(content) > 300:
                        content = content[:297] + "..."
                    lines.append(f"- [{strength:.2f}] {content}")

        if ces_surfaced:
            for item in ces_surfaced[:3]:
                content = item.get("content", "")
                if content:
                    if len(content) > 300:
                        content = content[:297] + "..."
                    lines.append(f"- [CES] {content}")

    return "\n".join(lines)


# ── JSON-RPC Server ───────────────────────────────────────────────────

METHODS = {
    "bootstrap": handle_bootstrap,
    "ingest": handle_ingest,
    "assemble": handle_assemble,
    "afterTurn": handle_after_turn,
    "compact": handle_compact,
    "dispose": handle_dispose,
    "stats": handle_stats,
}


def process_request(line: str) -> Optional[str]:
    """Process a single JSON-RPC request and return the response."""
    try:
        request = json.loads(line)
    except json.JSONDecodeError as exc:
        return json.dumps({
            "jsonrpc": "2.0",
            "id": None,
            "error": {"code": -32700, "message": f"Parse error: {exc}"},
        })

    req_id = request.get("id")
    method = request.get("method", "")
    params = request.get("params", {})

    handler = METHODS.get(method)
    if handler is None:
        return json.dumps({
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        })

    try:
        result = handler(params)
        return json.dumps({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": result,
        }, default=str)
    except Exception as exc:
        logger.error("RPC method %s failed: %s\n%s", method, exc, traceback.format_exc())
        return json.dumps({
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32000, "message": str(exc)},
        })


def main() -> None:
    """Main RPC loop — read requests from stdin, write responses to stdout."""
    logger.info("NeuroGraph RPC bridge starting")

    # Signal readiness to the TypeScript plugin
    ready_msg = json.dumps({
        "jsonrpc": "2.0",
        "method": "ready",
        "params": {"pid": os.getpid()},
    })
    sys.stdout.write(ready_msg + "\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        response = process_request(line)
        if response is not None:
            sys.stdout.write(response + "\n")
            sys.stdout.flush()

    logger.info("NeuroGraph RPC bridge shutting down")


if __name__ == "__main__":
    main()
