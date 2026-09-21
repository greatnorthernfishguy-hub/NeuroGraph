<!--
# ---- Changelog ----
# [2026-09-21] Cursor Grok 4.6 — operator record for landed fail-closed / HF remote embed
# What: new docs/NG_EMBED_OPERATOR.md (docs only)
# Why: 2026-09-20/21 ng_embed landings are on main at 1d6c659 and never appear
#   in CHANGELOG, USER_GUIDE, PUNCHLIST, README, or docs/
# How: verified against ng_embed.py, universal_ingestor.py, ng_commons_eco.py,
#   tests/test_ng_embed_dualpass.py, tests/test_dual_pass_extraction_visibility.py
# -------------------
-->

# NGEmbed operator record

The fail-closed / Hugging Face remote embed stack already landed on `main`
(`2c47dcd` … `1d6c659`, 2026-09-20/21). Tip: `1d6c659` — HF token is not a
`_hf_post` parameter. Missing mentions in CHANGELOG / USER_GUIDE / PUNCHLIST /
README are a docs gap, not a missing feature.

Canonical source: `ng_embed.py` (vendored). Tests:
`tests/test_ng_embed_dualpass.py`,
`tests/test_dual_pass_extraction_visibility.py`,
`tests/test_ng_commons_eco.py`.

This file is the embed-side record only. Anima intra-turn window chains
(`c53b903`) are a later docs PR.

---

## Fail-closed

`embed()` / `embed_batch()` raise `EmbeddingUnavailableError` when the model
cannot load or a real vector cannot be produced. Empty `embed_batch([])`
returns `[]` without loading.

There is no hash fallback in `ng_embed.py`. There is no env var that
re-enables one (`NG_EMBED_ALLOW_HASH_FALLBACK` is gone).

`DualPassIncompleteError` is a different failure: pass-2 extraction broke, so
there is no deposit at all.

---

## Default vs `NG_EMBED_REMOTE=hf`

Unset `NG_EMBED_REMOTE` → local ONNX Runtime,
`Snowflake/snowflake-arctic-embed-m-v1.5`, 768-dim, CLS pooling, CPU.

`NG_EMBED_REMOTE=hf` → Hugging Face Inference via
`https://router.huggingface.co/hf-inference/models/<model_id>/pipeline/feature-extraction`.
Remote is chosen in `_ensure_model` before any ONNX import. The tokenizer still
loads (windowing needs it). Request body is `{"inputs": ...}` only. Query prefix
and optional L2 stay client-side.

Any other value, including empty string, raises `EmbeddingUnavailableError`.

---

## Auth and token hygiene

Token comes from `HF_TOKEN`, else `~/.cache/huggingface/token`.
`HUGGING_FACE_HUB_TOKEN` is unused. Missing token raises
`EmbeddingUnavailableError("HF token unavailable")`.

`1d6c659`: `_hf_post` reads the bearer via `_get_hf_token()` internally. The
token is not a function argument, so it does not sit in that frame's traceback
locals.

Residual: `urllib.request.Request` still gets an `Authorization: Bearer …`
header. If that `Request` object appears in a traceback, the token can still
show.

---

## Remote retry and quarantine

Three attempts, backoff `1s` / `3s` / `9s` (including after the last fail),
then one JSONL line, then `EmbeddingUnavailableError`. A quarantine write
failure does not mask the original error.

Default path: `~/.cache/ng_embed/failed_embeds.jsonl` (`cache_dir`). Each line
includes `timestamp`, `text`, `is_query`, `normalize`, `error`, `attempts`.
`text` is the payload that was sent — that can be turn text.

---

## Keep-warm (unwired in production)

`start_keepalive()` / `stop_keepalive()` are reference-counted. No-op unless
remote. `0→1` starts a daemon thread (`ng_embed_keepalive`) that pings `"ping"`
every 20s. `1→0` signals stop. Ping failures stay at debug.

Nothing in production calls `start_keepalive`. Only the unit tests do. HF idle
eviction (30–60s) is therefore still in play unless an operator starts it.

---

## Overlapping windows and CLS/SEP wrap

Tokenizer truncation is gone.

- **Short path** (full encoding ≤ 512 tokens): one embed, including default
  `normalize=False`. Byte-identical to the single-window primitive. No extra
  CLS/SEP wrap.
- **Long path**: strip leading CLS and trailing SEP, window the interior at
  510 tokens with overlap 64, wrap each slice as `[CLS] + slice + [SEP]`
  (Arctic CLS-pools from position 0). Length-weighted mean, then L2-normalize
  the pooled vector. Per-window vectors are not L2-normalized unless the
  caller asked.

Missing CLS/SEP specials, or an encoding without leading CLS / trailing SEP,
raises `EmbeddingUnavailableError`. A window that would exceed 512 tokens
also raises.

`embed()` returns the pooled vector. `embed_windows()` also returns the
per-window structs.

---

## Dual-pass is atomic

`NGEmbed.dual_record_outcome` extracts concepts **before** any forest write,
then `embed_batch`s trees **before** any write.

- TID / extraction failure (`None`) → best-effort `signal_error` (rate-limited
  warning), then `DualPassIncompleteError`. No forest.
- Legitimate empty extraction (`[]`) → forest only, zero trees. That is a
  completed dual-pass, not a fallback.
- Optional `windows=` is echoed on the result dict and is not deposited here.

`ng_commons_eco.dual_record_outcome` no longer catches engine failures and
must not leave a forest-only Commons deposit (`238983c`).

`ng_ecosystem.dual_record_outcome` already delegated to `NGEmbed`; its
docstring still says "falls back to forest only" — stale comment, not
behavior.

---

## Operator-visible leftovers (not fixed here)

1. **Ingestor hash fallback.** `universal_ingestor.EmbeddingEngine._encode_batch`
   still `except Exception` around `ng_embed.embed_batch` and synthesizes
   SHA-256 hash vectors via `_hash_embed`. Fail-closed `NGEmbed` and the
   ingest path contradict each other: a failed embed can still become a
   deposited hash vector.
2. **Keepalive unwired.** The API exists; production never starts it.
3. **`feed-syl` ghost.** A `feed-syl` executable is still in the repo root.
   It is leftover, not a live operator interface. Do not treat it as the
   current ingest CLI.
