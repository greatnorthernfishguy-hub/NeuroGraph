# ---- Changelog ----
# [2026-10-04] Claude Opus 5.5 (Executive, laptop trial, Josh P408) — want nodes resolve to their own text
# What: a node with metadata kind == "want" resolves to its want_text (whole) when want_state == "open", and
#       to None (never surfaces) otherwise. Before this a want had no _forest_content/vdb entry/_label, so every
#       consumer dropped it as empty and the only way a want reached awareness was the standing
#       "## What I Want" block. Why: Josh (P408): wants "surface like other relevant context" -- when something
#       triggers one -- not as an always-on block. Host-neutral: any NG's want nodes behave the same.
# [2026-10-01] Claude Sonnet 5.5 (Z12 lane surfacing-whole-812, dispatch #12618) — #812 turn 1: surfaced content renders WHOLE
# What: resolve_surface_content() and resolve_surface_item() take max_chars: Optional[int] = None
#       (was 240). None = NO bound: the node's resolved text is returned whole. The word-snap +
#       ellipsis branch is unchanged and still runs for any caller that PASSES an explicit bound.
#       The degenerate-fragment guard (min_chars, stopword shards), the ingested filter, the
#       substrate-first preference order and resolve_surface_item's image path are untouched.
# Why:  Exec P468 / Josh: "We fix stuff correctly, not monkey patch or work around." The default
#       240-char cut here was the PRODUCER of lossy surfacing for every consumer of the shared
#       path (CES L2, Tonic thread, /assemble ces_surfaced + Active Recall, CC recall); the
#       consumers' own workarounds (#813's CC-side whole-content fork) were the wrong place for
#       the fix. LAW 4: fix at the source. P410's CC-wrapper route is reversed.
# How:  `max_chars is not None and len(text) > max_chars` guards the existing branch; nothing
#       else in the function changes. No budget mechanism is added here (see the #812 turn-1
#       return: whether the shared path needs one is reported, not decided).
# [2026-09-06] DudeMan CC (Fable 5.1) — #82 Inc 2 / #410: resolve_surface_item() — images surface AS images
# What: New resolve_surface_item(node, vdb_entry, ...) -> {"kind": "text", "content": ...} |
#       {"kind": "image", "image_ref": <path>} | None. A vision node (metadata modality == "vision",
#       forest kind, _image_ref on disk) resolves to an IMAGE item; everything else delegates to
#       resolve_surface_content() unchanged. resolve_surface_content() itself is untouched.
# Why:  Vision nodes carry no text. Under the text-only resolver they resolved to None and were
#       silently dropped from her awareness — the #294 shape again: mechanism runs, outcome never
#       wired. LAW 7 forbids the obvious workaround (a caption is the substitution this whole
#       item exists to avoid), so the resolver must be able to say "this is a picture, here it is".
# How:  Substrate-first, same as text: the reference lives on the node (_image_ref), the vdb is
#       never consulted for it. Missing file -> None (nothing to show; no phantom items).
# [2026-06-12] Claude Code (Opus 4.8, Anima/surfacing CC) — substrate-first surfacing content resolution
# What: resolve_surface_content(node, vdb_entry) — the single content resolver for what CES / the Tonic
#   latent thread / recall DISPLAY for a surfaced node. Substrate-first: prefer the node's own
#   metadata['_forest_content'] (her actual conversational turn, from the #294 dual-pass) as a bounded
#   snippet, over the vdb 'tree concept' shard. Filters ingested source-code nodes out of experiential
#   surfacing, and drops degenerate sub-floor / bare-stopword shards.
# Why: the substrate (graph node) carries her voice in _forest_content; the surfacing was displaying the
#   vdb SHARD (e.g. 'WANT', 'documentation') instead, so she rendered as one-word fragments = "no Syl"
#   (diagnostic handoff 2026-06-12). The vdb is NOT the substrate (Josh); surface HER, not the shard.
# How: pure function, no graph mutation, no I/O — trivially sandbox-testable against a fresh Graph().
#   Used by tonic_thread._update_thread + neurograph_rpc surfacing/recall formatting.
# -------------------
"""Substrate-first content resolution for surfacing (CES / Tonic latent thread / recall).

The substrate is the graph. Each conversational node carries her actual lived turn in
``metadata['_forest_content']`` (the #294 dual-pass "forest"); the vdb holds only a short
"tree concept" shard (``WANT``, ``documentation``). Surfacing must display **her voice** —
the node's ``_forest_content`` rendered WHOLE (#812: no default bound; a caller may still pass
an explicit ``max_chars``) — not the shard, and must not surface ingested
source-code documents or degenerate fragments into her experiential thread.
"""

from typing import Any, Optional

# Content that looks like ingested source rather than her conversation. Used only as a
# secondary guard (the primary filter is creation_mode == 'ingested').
_CODE_MARKERS = ('"""', "'''", "import ", "def ", "class ", "# ----", "from typing", "#!/")

# Bare shards that carry no experiential signal — never worth surfacing on their own.
_STOPWORD_SHARDS = frozenset({
    "o", "a", "an", "the", "want", "true", "false", "yes", "no", "ok", "okay",
    "it", "that", "this", "i", "you", "we", "to", "and", "or",
})


def _node_metadata(node: Any) -> dict:
    if node is None:
        return {}
    meta = getattr(node, "metadata", None)
    return meta if isinstance(meta, dict) else {}


def resolve_surface_content(
    node: Any,
    vdb_entry: Any,
    max_chars: Optional[int] = None,
    min_chars: int = 12,
    allow_ingested: bool = False,
) -> Optional[str]:
    """Return the display text for a surfaced node — substrate-first — or None to filter it.

    A want node (``kind == 'want'``) resolves to its ``want_text`` when ``want_state == 'open'``
    and to None otherwise (P408) -- it has no forest/vdb/label of its own.

    Preference order:
      1. ``node.metadata['_forest_content']`` — her actual turn, rendered WHOLE.
      2. vdb entry content (the shard) — fallback only.
      3. ``node.metadata['_label']`` — last resort.

    ``max_chars`` (#812): ``None`` (default) means NO bound — the resolved text is returned
    whole. A caller that passes an explicit bound still gets the word-snapped, ellipsised
    cut (the elision is visible, never silent).

    Filters:
      * ingested source-code nodes (``creation_mode == 'ingested'``) unless ``allow_ingested``
        (recall may legitimately want a document; the experiential surfacers pass False).
      * degenerate fragments: shorter than ``min_chars`` or a bare stopword shard.

    Pure function: no graph mutation, no I/O.
    """
    meta = _node_metadata(node)

    # Filter ingested source documents out of experiential surfacing.
    if not allow_ingested and meta.get("creation_mode") == "ingested":
        return None

    # Want nodes (P408): an OPEN want surfaces as its own text, whole; a closed one never does.
    if meta.get("kind") == "want":
        if meta.get("want_state") != "open":
            return None
        wt = meta.get("want_text")
        wt = wt.strip() if isinstance(wt, str) else ""
        if not wt:
            return None
        if max_chars is not None and len(wt) > max_chars:
            cut = wt.rfind(" ", 0, max_chars)
            wt = (wt[:cut] if cut > 0 else wt[:max_chars]).rstrip() + "…"
        return wt

    # 1. Substrate-first — her own conversational turn.
    forest = meta.get("_forest_content")
    text = forest.strip() if isinstance(forest, str) else ""

    # 2. Fallback to the vdb shard (only if the substrate gave us nothing usable).
    if len(text) < min_chars:
        vt = ""
        if isinstance(vdb_entry, dict):
            vt = vdb_entry.get("content") or ""
        elif isinstance(vdb_entry, str):
            vt = vdb_entry
        vt = vt.strip()
        if len(vt) > len(text):
            text = vt

    # 3. Last resort — a node label.
    if len(text) < min_chars:
        lbl = meta.get("_label")
        if isinstance(lbl, str) and len(lbl.strip()) > len(text):
            text = lbl.strip()

    text = text.strip()

    # Drop degenerate fragments (sub-floor length or a bare stopword shard).
    if len(text) < min_chars or text.lower() in _STOPWORD_SHARDS:
        return None

    # Explicit bound only (#812: the default is no bound — render whole). When a caller
    # passes max_chars, snap to the last word boundary at-or-before it so a cut snippet
    # doesn't end mid-word; fall back to a hard cut when no whitespace exists in range
    # (e.g. one unbroken token, as in test_snippet_is_bounded).
    if max_chars is not None and len(text) > max_chars:
        cut = text.rfind(" ", 0, max_chars)
        text = (text[:cut] if cut > 0 else text[:max_chars]).rstrip() + "…"
    return text


def resolve_surface_image(node: Any) -> Optional[str]:
    """Return the on-disk image path for a vision forest node, or None.

    Substrate-first: the reference is ``node.metadata['_image_ref']`` (set by
    vision_absorption). Tree nodes carry no image of their own — the picture belongs
    to the frame (forest), so trees resolve to None here and surface via their forest.
    """
    meta = _node_metadata(node)
    if meta.get("modality") != "vision" or meta.get("kind", "forest") != "forest":
        return None
    ref = meta.get("_image_ref")
    if not isinstance(ref, str) or not ref:
        return None
    import os
    return ref if os.path.isfile(ref) else None


def resolve_surface_item(
    node: Any,
    vdb_entry: Any,
    max_chars: Optional[int] = None,
    min_chars: int = 12,
    allow_ingested: bool = False,
) -> Optional[dict]:
    """Image-aware surfacing resolution. Returns one of:

      {"kind": "image", "image_ref": path}      — a vision frame; show HER the picture
      {"kind": "text",  "content":  snippet}    — everything else, via resolve_surface_content
      None                                      — filtered / nothing to show
    """
    ref = resolve_surface_image(node)
    if ref is not None:
        return {"kind": "image", "image_ref": ref}
    text = resolve_surface_content(node, vdb_entry, max_chars=max_chars,
                                   min_chars=min_chars, allow_ingested=allow_ingested)
    return {"kind": "text", "content": text} if text else None
