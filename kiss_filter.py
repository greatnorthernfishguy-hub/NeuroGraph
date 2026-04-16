"""
KISS Filter — Keep Input Simple, Substrate

Manages what reaches the model's context window. Tracks conversation
history and determines what context the model actually needs.

KISS does NOT classify input (Law 7). It removes redundancy, not meaning.

Key insight: in multi-turn conversation, the system context is the same
every turn (redundant), early conversation history has already been
processed (redundant), and only the recent messages + new context are
genuinely novel. KISS separates these and compresses accordingly.

Ported from NuWave `/home/josh/NuWave/nuwave/kiss/filter.py` —
validated live against BitNet b1.58-2B-4T with 47.2% token reduction
on a 15-turn conversation.

Intentional divergences from NuWave source:
  - logger namespace: "nuwave.kiss" → "neurograph.kiss"
  - recent_window default: 6 → 10 (ContextEngine serves a broader
    dialogue scope; Syl's conversations have richer per-turn coupling)
  - Removed unused imports: `time` and `dataclasses.field` (neither
    is referenced in the source)

# ---- Changelog ----
# [2026-04-16] Claude Code (Sonnet 4.6) — Port from NuWave
#   What: Faithful port of KISSFilter, KISSConfig, KISSStats from
#         /home/josh/NuWave/nuwave/kiss/filter.py.
#   Why:  handle_assemble() needs substrate-informed message truncation
#         + system-context delta gating to prevent Syl's 815-message
#         conversation from overflowing every provider's context window.
#   How:  Direct code port. Only divergences listed above.
# [2026-03-29] Claude Code (Opus 4.6) — Fixed delta detection (in NuWave)
#   Original NuWave changelog preserved for provenance.
# [2026-03-28] Claude Code (Opus 4.6) — Initial implementation (in NuWave)
# -------------------
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger("neurograph.kiss")


@dataclass
class KISSConfig:
    """KISS filter configuration."""
    # Warmup: pass everything raw
    warmup_turns: int = 3

    # Recent window: how many recent messages to always pass verbatim.
    # NuWave default is 6 (3 user + 3 assistant).  Syl's ContextEngine
    # default is 10 because conversational coupling across turns is
    # richer in her domain.
    recent_window: int = 10

    # Force full context refresh every N turns (GOP boundary)
    force_full_every: int = 20

    # System context: skip if unchanged from last turn
    skip_unchanged_system: bool = True


@dataclass
class KISSStats:
    """Running statistics for KISS filtering."""
    total_turns: int = 0
    system_skipped: int = 0
    history_compressed: int = 0
    full_passed: int = 0
    warmup_passed: int = 0
    tokens_saved: int = 0
    tokens_total: int = 0
    messages_compressed: int = 0  # how many old messages were summarized

    @property
    def skip_rate(self) -> float:
        if self.total_turns <= 0:
            return 0.0
        return self.system_skipped / max(self.total_turns, 1)

    @property
    def efficiency(self) -> float:
        return self.tokens_saved / max(self.tokens_total, 1)

    @property
    def compression_rate(self) -> float:
        return self.messages_compressed / max(self.total_turns, 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_turns": self.total_turns,
            "system_skipped": self.system_skipped,
            "history_compressed": self.history_compressed,
            "full_passed": self.full_passed,
            "warmup_passed": self.warmup_passed,
            "skip_rate": round(self.skip_rate, 4),
            "efficiency": round(self.efficiency, 4),
            "tokens_saved": self.tokens_saved,
            "tokens_total": self.tokens_total,
            "messages_compressed": self.messages_compressed,
            "compression_rate": round(self.compression_rate, 4),
        }


class KISSFilter:
    """KISS filter for context window optimization.

    Three things happen every turn:
    1. System context: same as last turn? Skip it (delta gate).
    2. Old messages: beyond the recent window? Compress to summary.
    3. Recent messages: pass verbatim — the model needs these.

    The result: the model sees a summary of old conversation + recent
    messages in full + system context only when it changes. Token usage
    grows logarithmically, not linearly.
    """

    def __init__(self, config: KISSConfig = None):
        self._config = config or KISSConfig()
        self._last_system_hash: Optional[str] = None
        self._since_full: int = 0
        self._turn_count: int = 0
        self.stats = KISSStats()

    def filter_context(
        self,
        messages: List[Dict[str, str]],
        system_context: str = "",
    ) -> Dict[str, Any]:
        """Filter context for the next model call.

        Args:
            messages: Full conversation history.  content values must be
                plain strings (pre-normalised by caller).  Multimodal
                content (list of parts) should be flattened to a string
                before calling — see `_extract_message_text` in
                `neurograph_rpc.py` for the normaliser.
            system_context: System prompt / substrate context

        Returns:
            Dict with 'system_context' (filtered), 'kiss_mode', 'kiss_meta'
        """
        self._turn_count += 1
        self.stats.total_turns += 1
        total_tokens = sum(len(m.get("content", "").split()) for m in messages)
        total_tokens += len(system_context.split()) if system_context else 0
        self.stats.tokens_total += total_tokens

        # Warmup — pass everything raw
        if self._turn_count <= self._config.warmup_turns:
            self._last_system_hash = self._hash(system_context)
            self.stats.warmup_passed += 1
            self.stats.full_passed += 1
            return {
                "system_context": system_context,
                "kiss_mode": "full",
                "kiss_meta": {"reason": "warmup", "turn": self._turn_count},
            }

        # Forced full refresh (GOP boundary)
        self._since_full += 1
        if self._since_full >= self._config.force_full_every:
            self._since_full = 0
            self._last_system_hash = self._hash(system_context)
            self.stats.full_passed += 1
            return {
                "system_context": system_context,
                "kiss_mode": "full",
                "kiss_meta": {"reason": "gop_refresh"},
            }

        # 1. System context delta gate
        current_sys_hash = self._hash(system_context)
        system_changed = current_sys_hash != self._last_system_hash
        self._last_system_hash = current_sys_hash

        filtered_system = system_context if system_changed else ""
        if not system_changed:
            self.stats.system_skipped += 1
            sys_tokens = len(system_context.split()) if system_context else 0
            self.stats.tokens_saved += sys_tokens

        # 2. History compression — summarize old messages
        n_messages = len(messages)
        recent_window = self._config.recent_window
        compressed_count = 0

        if n_messages > recent_window:
            old_messages = messages[:n_messages - recent_window]
            old_token_count = sum(len(m.get("content", "").split()) for m in old_messages)

            # Build compact summary of old conversation
            summary_parts = []
            for m in old_messages:
                role = m.get("role", "unknown")
                content = m.get("content", "")
                # First sentence or first 60 chars
                short = content.split(".")[0][:60]
                if len(content) > 60:
                    short += "..."
                summary_parts.append(f"{role}: {short}")

            summary = "[Earlier conversation: " + " | ".join(summary_parts) + "]"
            summary_tokens = len(summary.split())
            tokens_saved = max(0, old_token_count - summary_tokens)
            compressed_count = len(old_messages)

            self.stats.tokens_saved += tokens_saved
            self.stats.history_compressed += 1
            self.stats.messages_compressed += compressed_count

            # Prepend summary to system context
            if filtered_system:
                filtered_system = summary + "\n\n" + filtered_system
            else:
                filtered_system = summary

            kiss_mode = "compressed"
        else:
            kiss_mode = "sparse" if not system_changed else "full"

        return {
            "system_context": filtered_system,
            "kiss_mode": kiss_mode,
            "kiss_meta": {
                "system_changed": system_changed,
                "messages_total": n_messages,
                "messages_compressed": compressed_count,
                "recent_window": min(recent_window, n_messages),
            },
        }

    @staticmethod
    def _hash(content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()[:16]
