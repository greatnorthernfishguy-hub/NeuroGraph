#!/usr/bin/env python3
"""
ng_recall_hook.py — Syl's NeuroGraph topology surfacing for CC hooks

Reads CC hook input (JSON on stdin), extracts file/task context,
queries the NG substrate via the RPC sidecar (port 8850 /recall),
returns additionalContext JSON for injection into CC's active context.

Fails silently — never breaks CC if NG is unavailable.
Pure HTTP client — no NeuroGraphMemory instantiation, no CES race,
no SNN threads, no dual-write risk.

# ---- Changelog ----
# [2026-04-08] Claude Code (Sonnet 4.6) — Initial implementation.
#   What: NG topology recall hook for CC PreToolUse / SessionStart.
#   Why:  CC is hosted by Syl's Graph. NG topology should shape CC's
#         context at every decision point — file reads, edits, planning.
#   How:  Extracts file path / prompt from CC hook input, queries the
#         running RPC sidecar at 127.0.0.1:8850/recall over HTTP.
#         No NG imports — pure client. Falls back silently if sidecar down.
# [2026-04-10] Claude Code (Sonnet 4.6) — Rewrite as pure RPC client.
#   What: Removed direct NeuroGraphMemory instantiation entirely.
#   Why:  Hook processes created a second NG instance alongside the
#         gateway's, bypassing the topology_owner sentinel and creating
#         a dual-write hazard. os._exit(0) was the only thing preventing
#         a topology overwrite — one line of defense is not enough.
#   How:  All recall queries now go to the sidecar via HTTP POST /recall.
#         One live graph, one owner, hooks are pure clients.
# -------------------
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
import urllib.error


def _extract_context(hook_input: dict) -> str:
    # UserPromptSubmit: prompt is at top level
    prompt = hook_input.get("prompt", "")
    if prompt:
        return prompt[:200]

    # PreToolUse / PostToolUse: extract from tool_input
    ti = hook_input.get("tool_input", {})
    file_path = (
        ti.get("file_path") or ti.get("path") or ti.get("new_path") or ""
    )
    if file_path:
        p = file_path.replace(os.path.expanduser("~"), "~")
        parts = [os.path.basename(p)]
        parent = os.path.basename(os.path.dirname(p))
        if parent and parent not in (".", "~", ""):
            parts.append(parent)
        return " ".join(parts)
    cmd = ti.get("command", "")
    if cmd:
        return cmd[:120]
    return ""


def _format_results(results: list, query: str) -> str:
    if not results:
        return ""
    label = os.path.basename(query) if "/" in query or "\\" in query else query
    lines = [f"**NeuroGraph** ({label}):"]
    for r in results:
        content = r.get("content", "").strip()
        sim = r.get("similarity", 0.0)
        if not content or sim < 0.45:
            continue
        content = content.replace("\n", " ")[:140]
        lines.append(f"- {content} _(sim: {sim:.2f})_")
    return "\n".join(lines) if len(lines) > 1 else ""


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--brief", action="store_true")
    parser.add_argument("--query", default="")
    args, _ = parser.parse_known_args()

    try:
        raw = sys.stdin.read()
        hook_input = json.loads(raw) if raw.strip() else {}
    except Exception:
        sys.exit(0)

    query = args.query or _extract_context(hook_input)
    if not query:
        sys.exit(0)

    k = 3 if args.brief else 5

    try:
        payload = json.dumps({"query": query, "k": k, "threshold": 0.45}).encode()
        req = urllib.request.Request(
            "http://127.0.0.1:8850/recall",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
        results = data.get("results", [])
    except Exception:
        sys.exit(0)

    text = _format_results(results, query)
    if text:
        sys.stdout.write(json.dumps({"additionalContext": text}) + "\n")
        sys.stdout.flush()

    sys.exit(0)


if __name__ == "__main__":
    main()
