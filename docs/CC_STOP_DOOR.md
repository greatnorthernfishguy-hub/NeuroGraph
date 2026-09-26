<!--
# ---- Changelog ----
# [2026-09-26] Cursor Grok 4.7 — operator record for the landed CC Stop door
# What: new docs/CC_STOP_DOOR.md (docs only)
# Why: P240/P242 Stop-door and one-step-per-turn work landed on main with no
#   operator note, so a later nightly audit can treat the door as missing
# How: verified against main cc_ng_host.py, cc_ng_organism.py, and the Stop
#   and deposit-step tests; no runtime edit
# -------------------
-->

# CC Stop door — one step per turn

Operator note. Verified on `main` at `a430d9e4d27fcb0e32803824c255dcc843d1abc9` (2026-09-26). The Stop-door commits below are already ancestors of that tip. This file records them so a later audit does not treat the door as missing. Docs only.

## What landed

| Commit | What it did |
| --- | --- |
| `d2b5c636c4e0` | `z2-stop-door-restore-001`: restore the Stop door (P240(1)) |
| `b0ff14cb8649` | `z2-one-step-per-turn-001`: `_deposit` gains `step`; only the Stop door passes `step=True` (P240(2)) |
| `c1142646a1d8` | `z2-remove-deposit-step-flag-001`: remove `CC_NG_DEPOSIT_STEP`; the Stop door steps once with no flag; drains never step (P240(3)/P242) |

Test follow-ups on the same history: `3bc6aa8aa859`, `ba9e2c41232b`, `8c975e722253`, `97f21d4a14d9`.

## How the doors work

`cc_ng_host._DISPATCH` maps the event to the handler:

```python
"Stop": _handle_stop,
```

`_handle_stop` reads `last_assistant_message`. When that value is a non-empty string, it deposits the text on a daemon thread and asks that deposit to step:

```python
threading.Thread(target=_deposit, args=(msg,), kwargs={"step": True}, daemon=True).start()
```

The handler returns `{"ok": True}`. It does not return a context key. An empty string, whitespace, or a non-string returns `{"ok": True}` and does not start a deposit.

`_deposit(text, step=False)` runs `cc_ng_organism.run_conversational_dual_pass` under `graph._concurrent_lock`. The step call sits on that same lock, and only the `step=True` path takes it:

```python
if step:
    cc_ng_organism.cc_deposit_step(ng.graph, ingested)
```

`cc_deposit_step` takes `graph._step_lock`, calls `graph.step()` once, injects the flat `0.1` reward when the dual pass landed and `three_factor_enabled` is on, then runs `discover_hyperedges` on that step's fired set. Its docstring names this door: the Stop-side `_deposit(step=True)` is the only caller, once per turn, including when the dual pass failed.

These paths leave `step` at the default, or never call `_deposit`, and they do not call `cc_deposit_step`:

- `UserPromptSubmit` (`_handle_user_prompt_submit`) starts `_deposit` with the prompt and no `step` argument.
- Pith failure (`_deposit_pith_failure`) starts `_deposit` with no `step` argument.
- `PostToolUse` writes tool text through `_deposit_tool_experience` (Commons only).
- `drain_ingest_tract` and `drain_gateway_conduit` apply records and never step.

## What operators must configure

The handler is in the host. It stays inert until a Claude Code Stop hook for the host entrypoint is registered in settings. The P240(1) changelog names that entrypoint `cc-ng-hook.py`. This note does not include a settings snippet.

The `Stop` entry already in this repo's `.claude/settings.json` runs `.claude/hooks/stop_uncommitted_guard.sh`. That guard checks uncommitted work. It does not deliver `last_assistant_message` to `_handle_stop`.

## What not to re-enable

`CC_NG_DEPOSIT_STEP` is gone. `cc_ng_host.py` and `cc_ng_organism.py` have no live `os.environ` / `getenv` read of that name. `cc_ng_organism` has no `_CC_NG_DEPOSIT_STEP` attribute; `test_flag_attribute_is_gone` asserts that. Stepping on this door is unconditional: `_deposit` calls `cc_deposit_step` whenever `step` is true, with no flag in front of the call.

Leave the flag out of config and out of new code. Leave `UserPromptSubmit`, tool deposits, pith-failure deposits, and both drains off `cc_deposit_step`.

The same commits leave `CC_NG_AUTOSTEP` untouched. That name still gates the Tonic autostep in `tonic_engine.py` (default off). It is a separate switch. Turning it on is not how this door steps.

## Pointers

Code: `cc_ng_host.py` (`_handle_stop`, `_deposit`, `_DISPATCH`) and `cc_ng_organism.py` (`cc_deposit_step`).

Tests:

- `tests/test_cc_host_stop_door.py` — Stop deposits `last_assistant_message` through the real `_deposit` with `kwargs == {"step": True}`, and never returns context.
- `tests/test_cc_deposit_step.py` — `test_flag_attribute_is_gone`, `test_host_prompt_side_does_not_step`, `test_one_turn_steps_exactly_once`, `test_post_tool_use_never_steps`, `test_drain_ingest_tract_never_steps`, `test_drain_gateway_conduit_never_steps`.
