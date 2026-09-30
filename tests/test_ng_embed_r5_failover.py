# tests/test_ng_embed_r5_failover.py
# ---- Changelog ----
# [2026-09-30] Claude Sonnet 5.5 (T3 harness), lane
#   z11-r5-embed-failover-20260929 — R5 C4 (worker-004): fail-back (#766).
# What: tests for failover that lasts only while local is down, for BOTH episode
#   kinds (load failure, inference failure) with a controllable clock (nothing
#   sleeps): fail back when local recovers with one WARNING each way including
#   duration and served-count; still-down probes exactly once per interval (two
#   calls at a boundary make one probe) with one log line each and no traceback,
#   and a failed load probe leaves state untouched; explicit NG_EMBED_REMOTE=hf
#   never probes or fails back; NG_EMBED_REPROBE_SECS=0 disables (and the log
#   says so), default when unset, non-number falls back with one warning; fail
#   back then fail over again logs again, in order; require_local while failed
#   over never calls remote and may trigger a due probe; one prober at a time
#   (simulated with the lock); a bug-class probe defect is raised; the probe is
#   local-only (socket connect and DNS raise; load probe uses local_files_only;
#   inference probe sends exactly the fixed probe text); keepalive stops pinging
#   after a fail-back.
# Why:  Josh's #766 ruling. The probe must never leak caller text or touch the
#   network, and a failed probe must be visible but not spammy.
# How:  _Scenario builds either episode kind with a switchable "local is up"
#   flag, patched ng_embed._now, and the C3 remote spy. Each test sets/deletes
#   NG_EMBED_REMOTE and NG_EMBED_REPROBE_SECS itself and asserts them.
# -------------------
# [2026-09-30] Claude Sonnet 5.5 (T3 harness), lane
#   z11-r5-embed-failover-20260929 — R5 C3 (worker-003, delta-LE finding F1).
# What: require_local is honored by the inference path. With
#   require_local=True, embed/embed_batch/embed_windows raise
#   EmbeddingUnavailableError on an environment-class local inference failure
#   with ZERO remote calls (_hf_post, _hf_remote_call and urllib.request.urlopen
#   are all counted), _remote_mode unchanged, _model_loaded kept; a bug-class
#   defect is still raised as the original object; a call arriving when
#   _remote_mode is already True (NG_EMBED_REMOTE=hf, or a prior failover)
#   refuses; a mid-call flip of _remote_mode never routes it to the remote arm;
#   default and explicit require_local=False still fail over bit-identical.
#   Tool level: reembed_snowflake.main() with local inference failing on batch
#   2 of 3 exits 1, makes no remote call, never tries batch 3 and leaves the
#   vectors file byte-identical, with and without --dry-run.
# Why:  The #763 fix only covered model load; reembed_snowflake still called the
#   default embed_batch, which would have re-sent a failed batch to the HF API.
# How:  These tests do not stub _ensure_model, so the real guard is under test.
#   Each sets/deletes NG_EMBED_REMOTE itself and asserts it before the call.
# -------------------
# [2026-09-30] Claude Sonnet 5.5 (T3 harness), lane
#   z11-r5-embed-failover-20260929 — R5 correction pass (worker-002), C2 +
#   #765 + #763 (Executive Packet 376).
# What: C2: failover WARNING carries the traceback, logs once, and
#   _model_failed is True after an inference failover. #765: environment-class
#   failures (RuntimeError, OSError, MemoryError, bare Exception, real
#   onnxruntime RuntimeException/Fail, hub LocalEntryNotFoundError,
#   ImportError) still fail over bit-identical; bug-class defects
#   (TypeError, AttributeError, IndexError, KeyError, NameError, ValueError,
#   AssertionError, NotImplementedError) are raised and logged at ERROR, at
#   all four inference sites and at the load site, and never reach the
#   remote, the quarantine file or the failover state; a real IndexError from
#   a one-output session is covered. #763: require_local False on load
#   failure / remote selected / already failed over (no failover, no remote
#   call), True on a healthy load, still raises a bug-class load defect, and
#   reembed_snowflake.main() aborts with vectors untouched. Added a
#   module-level assertion that ng_embed came from this repo.
# Why:  Packet 376 widened the correction to #763 and #765. The assertion is
#   there because importing neurograph_rpc (the guard plugin does, on the
#   first test) puts the MAIN checkout at the head of sys.path, so a lazy
#   import can silently resolve to code that is not this branch.
# How:  reembed_snowflake is loaded by explicit path for the same reason
#   (my first version imported the main checkout's old copy and the test
#   correctly failed). Each test sets/deletes NG_EMBED_REMOTE itself and
#   asserts it before the call under test.
# -------------------
# [2026-09-30] Claude Sonnet 5.5 (T3 harness), lane
#   z11-r5-embed-failover-20260929 — R5 correction pass (worker-002), C1.
# What: Added test_inference_failover_site_matches_explicit_remote and
#   test_inference_failover_site_double_failure_quarantines_and_raises,
#   each parametrized over the three inference-failover sites the first
#   three tests do not reach: embed_batch short branch, embed_batch long
#   branch, embed_windows long path. Added _ClsSepTok import, a call counter
#   on _BoomSession, and helpers (_explicit_remote_*, _failover_instance).
# Why:  Cross-family review (ACCEPT-WITH-CORRECTIONS, HIGH): all first-delivery
#   tests call embed(), which reaches only the embed_windows short site, so the
#   other three wraps could be deleted with every test still green.
# How:  The fake HF router returns a vector derived from the exact text sent,
#   so a failover that sent different text (prefix, window slice) cannot match
#   the explicit NG_EMBED_REMOTE=hf ground truth. Double-failure cases compare
#   the failed_embeds.jsonl record to the one explicit remote mode writes for
#   the same input (identical bar timestamp). Each case sets/deletes
#   NG_EMBED_REMOTE itself and asserts it before the call under test. Negative
#   control: deleting each wrap in a scratch clone fails exactly that site's
#   tests (see returns/worker-002.md).
# -------------------
# [2026-09-29] Claude Sonnet 5 (T3 harness), lane
#   z11-r5-embed-failover-20260929 — Executive Packet 139 R5, Executive
#   Packet 353 build guard #732 follow-up.
# What: New test file for acceptance 6a. Three tests: (1) a forced local
#   model-load failure with NG_EMBED_REMOTE unset fails over to the HF
#   remote path and produces a vector bit-for-bit identical to explicit
#   NG_EMBED_REMOTE=hf for the same input; (2) same assertion for a forced
#   local inference failure (session.run raises after a successful load);
#   (3) local inference fails AND the remote call is also forced to fail —
#   asserts the input lands in failed_embeds.jsonl and
#   EmbeddingUnavailableError is raised. Every test (and the shared
#   _remote_mode_vector helper each of the first two calls) explicitly
#   monkeypatch.delenv/setenv's NG_EMBED_REMOTE itself and asserts the
#   value immediately before the call under test, rather than relying on
#   the module's autouse env-reset fixture alone — required by Packet 353
#   build guard #732 (the T3 worker shell inherits NG_EMBED_REMOTE=hf, so
#   a test that only relied on ambient/ fixture state could silently
#   exercise the wrong code path without any test-local evidence of it).
# Why:  R5 spec (2026-09-20-ng-embed-dual-pass-nontruncating-design.md)
#   acceptance 6a required this exact behavior; ng_embed.py had no test
#   covering either failure origin failing over. Packet 353 #732 added
#   the explicit-env-per-test requirement after dispatch.
# How:  huggingface_hub.hf_hub_download is monkeypatched to raise for the
#   load-failure case (the real package is installed, so this simulates a
#   local load failure — e.g. no usable hardware — without needing a fake
#   package). For the inference-failure case, a fake ONNX session whose
#   .run() raises is installed directly on the instance after stubbing
#   _ensure_model. Both cases pre-set emb._tokenizer to the existing
#   fixture _FakeTok so _ensure_tokenizer() short-circuits and no real
#   network tokenizer download is attempted. _hf_post is monkeypatched to
#   a deterministic fake so the remote vector is reproducible and
#   comparable byte-for-byte across the failover instance and an explicit
#   NG_EMBED_REMOTE=hf instance.
# -------------------
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ng_embed import NGEmbed, EmbeddingUnavailableError
import ng_embed as ng_embed_mod

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Importing neurograph_rpc (the guard plugin does, on the first test) prepends the
# MAIN checkout to sys.path; fail loudly if ng_embed did not come from this repo.
assert os.path.dirname(os.path.abspath(ng_embed_mod.__file__)) == _REPO_ROOT, (
    f"ng_embed resolved outside this repo: {ng_embed_mod.__file__}"
)

from test_ng_embed_dualpass import _FakeTok, _ClsSepTok  # reuse existing tokenizer fixtures


@pytest.fixture(autouse=True)
def _reset_singleton_and_env(monkeypatch):
    # Belt-and-suspenders module-wide reset. Packet 353 #732 additionally
    # requires each test function below to also explicitly manage
    # NG_EMBED_REMOTE itself (not rely on this fixture alone) — see each
    # test's own monkeypatch.delenv/setenv + assert calls.
    monkeypatch.delenv("NG_EMBED_REMOTE", raising=False)
    monkeypatch.delenv("NG_EMBED_ALLOW_HASH_FALLBACK", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("NG_EMBED_TID_ENDPOINT", raising=False)
    monkeypatch.delenv("NG_EMBED_REPROBE_SECS", raising=False)
    NGEmbed.reset_instance()
    yield
    NGEmbed.reset_instance()


class _BoomSession:
    """Stand-in for an ONNX InferenceSession whose .run() blows up —
    corrupt session / ORT runtime error / OOM, per the R5 assignment."""

    def __init__(self):
        self.calls = 0

    def run(self, *a, **k):
        self.calls += 1
        raise RuntimeError("simulated ORT runtime error")


def _remote_mode_vector(monkeypatch, tmp_path, text, raw_vector):
    """Build a fresh instance in explicit NG_EMBED_REMOTE=hf mode and embed
    `text`, returning the vector. Used as the ground truth to compare a
    failover vector against (acceptance 6a: bit-for-bit identical).

    Env (Packet 353 #732): NG_EMBED_REMOTE explicitly set to "hf" here via
    monkeypatch.setenv (not inherited), HF_TOKEN explicitly set to
    "tok-test" via monkeypatch.setenv. Before returning, NG_EMBED_REMOTE is
    explicitly deleted again via monkeypatch.delenv so the caller resumes
    with a known-unset value rather than whatever this helper leaves behind.
    """
    monkeypatch.delenv("NG_EMBED_REMOTE", raising=False)
    monkeypatch.setenv("NG_EMBED_REMOTE", "hf")
    monkeypatch.setenv("HF_TOKEN", "tok-test")
    assert os.environ.get("NG_EMBED_REMOTE") == "hf"
    emb = NGEmbed()
    emb._config["cache_dir"] = str(tmp_path)
    emb._tokenizer = _FakeTok()
    assert emb._ensure_model() is True
    assert emb._session is None  # explicit remote mode never touches ONNX

    def fake_post(url, payload, timeout=30):
        return list(raw_vector)

    monkeypatch.setattr(emb, "_hf_post", fake_post)
    vec = emb.embed(text)
    NGEmbed.reset_instance()
    monkeypatch.delenv("NG_EMBED_REMOTE", raising=False)
    assert os.environ.get("NG_EMBED_REMOTE") is None
    return vec


def test_local_load_failure_fails_over_to_remote_and_matches_remote_mode(
    monkeypatch, tmp_path,
):
    """Acceptance 6a (load-failure half): NG_EMBED_REMOTE unset, a forced
    local model/tokenizer load failure embeds through the same-model API
    automatically, producing a vector bit-for-bit identical to explicit
    NG_EMBED_REMOTE=hf mode for the same input.

    Env (Packet 353 #732): NG_EMBED_REMOTE is explicitly deleted by this
    test (monkeypatch.delenv) both before the ground-truth remote-mode call
    (inside _remote_mode_vector, which itself sets it to "hf" and restores
    the delenv before returning) and again, explicitly, immediately before
    the failover call under test — asserted unset via
    `assert os.environ.get("NG_EMBED_REMOTE") is None` right before
    `emb.embed(text)`. HF_TOKEN is explicitly set to "tok-test" (unused in
    practice since _hf_post is replaced directly, but set for clarity/
    parity with the ground-truth call).
    """
    import huggingface_hub

    monkeypatch.delenv("NG_EMBED_REMOTE", raising=False)

    text = "hello world, this is R5"
    raw_vector = [0.42] * 768

    # Ground truth: explicit remote mode for the same input, same fake response.
    expected = _remote_mode_vector(monkeypatch, tmp_path, text, raw_vector)

    # Force the local ONNX model download/load to raise.
    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda **kw: (_ for _ in ()).throw(
            RuntimeError("simulated: local hardware cannot load the model")
        ),
    )
    monkeypatch.delenv("NG_EMBED_REMOTE", raising=False)
    monkeypatch.setenv("HF_TOKEN", "tok-test")

    emb = NGEmbed()
    emb._config["cache_dir"] = str(tmp_path)
    # Pre-set the tokenizer so _ensure_tokenizer() (called unconditionally by
    # embed_windows) short-circuits instead of attempting a real network
    # tokenizer download after the ONNX load path has already failed.
    emb._tokenizer = _FakeTok()

    def fake_post(url, payload, timeout=30):
        return list(raw_vector)

    monkeypatch.setattr(emb, "_hf_post", fake_post)

    assert os.environ.get("NG_EMBED_REMOTE") is None

    vec = emb.embed(text)

    assert emb._model_failed is True
    assert emb._remote_mode is True
    assert np.array_equal(vec, expected), (
        "failed-over vector must be bit-for-bit identical to the explicit "
        "NG_EMBED_REMOTE=hf vector for the same input"
    )


def test_local_inference_failure_fails_over_to_remote_and_matches_remote_mode(
    monkeypatch, tmp_path,
):
    """Acceptance 6a (inference-failure half): NG_EMBED_REMOTE unset, the
    model loads fine but a later ONNX inference call raises. The call fails
    over to the same-model API automatically, producing a vector
    bit-for-bit identical to explicit NG_EMBED_REMOTE=hf mode.

    Env (Packet 353 #732): NG_EMBED_REMOTE explicitly deleted by this test
    (monkeypatch.delenv) both before the ground-truth remote-mode call and
    again, explicitly, before the failover call under test — asserted
    unset via `assert os.environ.get("NG_EMBED_REMOTE") is None`. HF_TOKEN
    is explicitly set to "tok-test" (unused in practice since _hf_post is
    replaced directly).
    """
    monkeypatch.delenv("NG_EMBED_REMOTE", raising=False)

    text = "hello world, this is R5"
    raw_vector = [0.77] * 768

    expected = _remote_mode_vector(monkeypatch, tmp_path, text, raw_vector)

    monkeypatch.delenv("NG_EMBED_REMOTE", raising=False)
    assert os.environ.get("NG_EMBED_REMOTE") is None
    monkeypatch.setenv("HF_TOKEN", "tok-test")

    emb = NGEmbed()
    emb._config["cache_dir"] = str(tmp_path)
    monkeypatch.setattr(emb, "_ensure_model", lambda: True)
    emb._model_loaded = True
    emb._tokenizer = _FakeTok()
    emb._session = _BoomSession()  # model "loaded", inference is broken

    def fake_post(url, payload, timeout=30):
        return list(raw_vector)

    monkeypatch.setattr(emb, "_hf_post", fake_post)

    vec = emb.embed(text)

    assert emb._remote_mode is True
    assert np.array_equal(vec, expected), (
        "failed-over vector must be bit-for-bit identical to the explicit "
        "NG_EMBED_REMOTE=hf vector for the same input"
    )


def test_local_and_remote_both_fail_quarantines_and_raises(monkeypatch, tmp_path):
    """Acceptance 6a (double-failure half): local inference fails AND the
    remote call is also forced to fail (after its existing 3x retry) —
    the input lands in failed_embeds.jsonl and EmbeddingUnavailableError
    is raised.

    Env (Packet 353 #732): NG_EMBED_REMOTE explicitly deleted by this test
    (monkeypatch.delenv) and asserted unset immediately before the call
    under test. HF_TOKEN is explicitly set to "tok-test" (unused in
    practice since _hf_post is replaced directly).
    """
    monkeypatch.delenv("NG_EMBED_REMOTE", raising=False)

    text = "hello world, this is R5"

    assert os.environ.get("NG_EMBED_REMOTE") is None
    monkeypatch.setenv("HF_TOKEN", "tok-test")
    monkeypatch.setattr(ng_embed_mod.time, "sleep", lambda s: None)

    emb = NGEmbed()
    emb._config["cache_dir"] = str(tmp_path)
    monkeypatch.setattr(emb, "_ensure_model", lambda: True)
    emb._model_loaded = True
    emb._tokenizer = _FakeTok()
    emb._session = _BoomSession()

    attempts = {"n": 0}

    def boom_post(url, payload, timeout=30):
        attempts["n"] += 1
        raise ConnectionError("remote also down")

    monkeypatch.setattr(emb, "_hf_post", boom_post)

    with pytest.raises(EmbeddingUnavailableError):
        emb.embed(text)

    assert attempts["n"] == 3
    assert emb._remote_mode is True  # local inference failure did engage failover

    q = tmp_path / "failed_embeds.jsonl"
    assert q.is_file()
    lines = q.read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["attempts"] == 3
    assert "error" in rec



# ---------------------------------------------------------------------------
# C1 (worker-002 correction): the three inference-failover sites that the
# tests above do not reach. Each parametrized case drives ONE site:
#   embed_batch short branch, embed_batch long branch, embed_windows long path.
# A single long text is used for the long cases and short texts only for the
# short case: once any site fails over, _remote_mode flips and later branches
# in the same call take the explicit-remote arm, so mixing would leave a wrap
# unexercised.
# ---------------------------------------------------------------------------

# _ClsSepTok is one token per char + CLS/SEP: 600 chars -> 602 tokens -> 2 windows.
_LONG_TEXT = "".join(chr(97 + i % 26) for i in range(600))


def _text_vec(text):
    seed = sum(ord(c) for c in text)
    return [((seed + i) % 251) / 251.0 for i in range(768)]


def _text_dependent_post(url, payload, timeout=30):
    """Fake HF router whose vector depends on the exact text sent, so a failover
    that sent different text (wrong prefix, wrong window slice) cannot match."""
    inputs = payload["inputs"]
    if isinstance(inputs, list):
        return [_text_vec(t) for t in inputs]
    return _text_vec(inputs)


def _call_embed_batch_short(emb):
    return emb.embed_batch(["alpha beta", "gamma delta"], normalize=True, is_query=True)


def _call_embed_batch_long(emb):
    return emb.embed_batch([_LONG_TEXT])


def _call_embed_windows_long(emb):
    we = emb.embed_windows(_LONG_TEXT)
    assert len(we.windows) == 2
    return [we.pooled] + [w.embedding for w in we.windows]


_SITES = [
    pytest.param(_FakeTok, _call_embed_batch_short, id="embed_batch_short"),
    pytest.param(_ClsSepTok, _call_embed_batch_long, id="embed_batch_long"),
    pytest.param(_ClsSepTok, _call_embed_windows_long, id="embed_windows_long"),
]


def _explicit_remote_instance(monkeypatch, cache_dir, tokenizer_cls):
    """Fresh instance in explicit NG_EMBED_REMOTE=hf mode (env set here, not inherited)."""
    monkeypatch.delenv("NG_EMBED_REMOTE", raising=False)
    monkeypatch.setenv("NG_EMBED_REMOTE", "hf")
    monkeypatch.setenv("HF_TOKEN", "tok-test")
    assert os.environ.get("NG_EMBED_REMOTE") == "hf"
    emb = NGEmbed()
    emb._config["cache_dir"] = str(cache_dir)
    emb._tokenizer = tokenizer_cls()
    assert emb._ensure_model() is True
    assert emb._remote_mode is True and emb._session is None
    return emb


def _explicit_remote_result(monkeypatch, cache_dir, tokenizer_cls, call):
    emb = _explicit_remote_instance(monkeypatch, cache_dir, tokenizer_cls)
    monkeypatch.setattr(emb, "_hf_post", _text_dependent_post)
    result = call(emb)
    NGEmbed.reset_instance()
    monkeypatch.delenv("NG_EMBED_REMOTE", raising=False)
    assert os.environ.get("NG_EMBED_REMOTE") is None
    return result


def _boom_post_factory(counter):
    def boom_post(url, payload, timeout=30):
        counter["n"] += 1
        raise ConnectionError("remote also down")
    return boom_post


def _read_single_record(cache_dir):
    q = cache_dir / "failed_embeds.jsonl"
    assert q.is_file()
    lines = q.read_text().strip().splitlines()
    assert len(lines) == 1
    return json.loads(lines[0])


def _explicit_remote_failure_record(monkeypatch, cache_dir, tokenizer_cls, call):
    emb = _explicit_remote_instance(monkeypatch, cache_dir, tokenizer_cls)
    counter = {"n": 0}
    monkeypatch.setattr(emb, "_hf_post", _boom_post_factory(counter))
    with pytest.raises(EmbeddingUnavailableError):
        call(emb)
    assert counter["n"] == 3
    rec = _read_single_record(cache_dir)
    NGEmbed.reset_instance()
    monkeypatch.delenv("NG_EMBED_REMOTE", raising=False)
    assert os.environ.get("NG_EMBED_REMOTE") is None
    return rec


def _failover_instance(monkeypatch, cache_dir, tokenizer_cls, session=None):
    """NG_EMBED_REMOTE unset (asserted); ONNX 'loaded' but session.run always raises."""
    monkeypatch.delenv("NG_EMBED_REMOTE", raising=False)
    assert os.environ.get("NG_EMBED_REMOTE") is None
    monkeypatch.setenv("HF_TOKEN", "tok-test")
    emb = NGEmbed()
    emb._config["cache_dir"] = str(cache_dir)
    monkeypatch.setattr(emb, "_ensure_model", lambda: True)
    emb._model_loaded = True
    emb._tokenizer = tokenizer_cls()
    session = session if session is not None else _BoomSession()
    emb._session = session
    return emb, session


@pytest.mark.parametrize("tokenizer_cls, call", _SITES)
def test_inference_failover_site_matches_explicit_remote(
    tokenizer_cls, call, monkeypatch, tmp_path,
):
    """Acceptance 6a at the sites the first three tests do not reach: local
    inference raises, the vectors are bit-for-bit those of explicit
    NG_EMBED_REMOTE=hf for the same input.

    Env (Packet 353 #732): NG_EMBED_REMOTE deleted then set to "hf" only inside
    _explicit_remote_instance (ground truth), deleted again and asserted unset
    before the failover call under test; HF_TOKEN set to "tok-test" (unused,
    _hf_post is replaced). No other NG_EMBED_* variable is read by this path.
    """
    monkeypatch.delenv("NG_EMBED_REMOTE", raising=False)
    expected = _explicit_remote_result(monkeypatch, tmp_path / "explicit", tokenizer_cls, call)

    emb, session = _failover_instance(monkeypatch, tmp_path / "failover", tokenizer_cls)
    monkeypatch.setattr(emb, "_hf_post", _text_dependent_post)
    assert os.environ.get("NG_EMBED_REMOTE") is None
    got = call(emb)

    assert session.calls >= 1, "local inference must have been attempted first"
    assert emb._remote_mode is True
    assert len(got) == len(expected)
    for g, e in zip(got, expected):
        assert g.dtype == e.dtype
        assert np.array_equal(g, e), (
            "failed-over vector must be bit-for-bit identical to the explicit "
            "NG_EMBED_REMOTE=hf vector for the same input"
        )


@pytest.mark.parametrize("tokenizer_cls, call", _SITES)
def test_inference_failover_site_double_failure_quarantines_and_raises(
    tokenizer_cls, call, monkeypatch, tmp_path,
):
    """Local inference fails AND the remote call fails 3x: EmbeddingUnavailableError,
    one failed_embeds.jsonl record with attempts 3, and that record is identical
    (bar the timestamp) to what explicit remote mode quarantines for the same input.

    Env (Packet 353 #732): NG_EMBED_REMOTE deleted/set/deleted exactly as in the
    test above and asserted unset before the call under test; HF_TOKEN set to
    "tok-test" (unused, _hf_post is replaced); time.sleep patched.
    """
    monkeypatch.delenv("NG_EMBED_REMOTE", raising=False)
    monkeypatch.setattr(ng_embed_mod.time, "sleep", lambda s: None)
    explicit_rec = _explicit_remote_failure_record(
        monkeypatch, tmp_path / "explicit", tokenizer_cls, call,
    )

    emb, session = _failover_instance(monkeypatch, tmp_path / "failover", tokenizer_cls)
    counter = {"n": 0}
    monkeypatch.setattr(emb, "_hf_post", _boom_post_factory(counter))
    assert os.environ.get("NG_EMBED_REMOTE") is None
    with pytest.raises(EmbeddingUnavailableError):
        call(emb)

    assert session.calls >= 1, "local inference must have been attempted first"
    assert counter["n"] == 3
    assert emb._remote_mode is True
    rec = _read_single_record(tmp_path / "failover")
    assert rec["attempts"] == 3
    assert "error" in rec

    def strip(r):
        return {k: v for k, v in r.items() if k != "timestamp"}

    assert strip(rec) == strip(explicit_rec), (
        "both failure origins must quarantine an identical record"
    )


# ---------------------------------------------------------------------------
# C2 / #765 / #763 (worker-002 correction). Every test below sets or deletes
# NG_EMBED_REMOTE itself and asserts it before the call under test; the only
# NG_EMBED_* name any of them uses is NG_EMBED_REMOTE (plus the autouse
# fixture's deletes of NG_EMBED_ALLOW_HASH_FALLBACK / NG_EMBED_TID_ENDPOINT).
# ---------------------------------------------------------------------------

import logging


class _RaisingSession:
    def __init__(self, exc):
        self.exc = exc
        self.calls = 0

    def run(self, *a, **k):
        self.calls += 1
        raise self.exc


class _OneOutputSession:
    """Returns one output where the real model returns two, so the real code's
    outputs[1] raises IndexError: a genuine defect, not a simulated exception."""

    def __init__(self):
        self.calls = 0

    def run(self, output_names, feed):
        self.calls += 1
        return [np.zeros((feed["input_ids"].shape[0], 768), dtype=np.float32)]


def _call_embed_short(emb):
    return [emb.embed("hello world")]


_SITES_ALL = _SITES + [pytest.param(_FakeTok, _call_embed_short, id="embed_windows_short")]

_BUG_CLASSES = [
    TypeError, AttributeError, IndexError, KeyError, NameError,
    ValueError, AssertionError, NotImplementedError,
]


def _ort_exc(name):
    st = pytest.importorskip("onnxruntime.capi.onnxruntime_pybind11_state")
    return getattr(st, name)(f"simulated ORT {name}")


def _log_records(caplog, level):
    return [r for r in caplog.records if r.name == "ng_embed" and r.levelno == level]


def _failover_warnings(caplog):
    return [r for r in _log_records(caplog, logging.WARNING) if "failing over to HF remote" in r.getMessage()]


def _counting_post(calls):
    def post(url, payload, timeout=30):
        calls.append(1)
        return _text_dependent_post(url, payload, timeout)
    return post


def test_failover_warning_carries_traceback_and_sets_model_failed(monkeypatch, tmp_path, caplog):
    """C2: the failover WARNING has the triggering traceback, is logged once,
    and _model_failed is True after an INFERENCE failover (not only a load one).

    Env: NG_EMBED_REMOTE deleted and asserted unset; HF_TOKEN set (unused)."""
    caplog.set_level(logging.DEBUG, logger="ng_embed")
    emb, session = _failover_instance(monkeypatch, tmp_path, _FakeTok)
    monkeypatch.setattr(emb, "_hf_post", _text_dependent_post)
    assert os.environ.get("NG_EMBED_REMOTE") is None
    assert emb._model_failed is False

    emb.embed("hello world")
    emb.embed("hello world again")

    assert emb._model_failed is True
    warns = _failover_warnings(caplog)
    assert len(warns) == 1, "one WARNING per process, not per call"
    rec = warns[0]
    assert rec.exc_info and rec.exc_info[0] is RuntimeError
    assert "simulated ORT runtime error" in str(rec.exc_info[1])
    assert "Traceback (most recent call last)" in logging.Formatter().format(rec)
    assert not _log_records(caplog, logging.ERROR)


@pytest.mark.parametrize("make_exc", [
    pytest.param(lambda: RuntimeError("runtime"), id="RuntimeError"),
    pytest.param(lambda: OSError("os"), id="OSError"),
    pytest.param(lambda: MemoryError(), id="MemoryError"),
    pytest.param(lambda: Exception("bare"), id="bare_Exception"),
    pytest.param(lambda: _ort_exc("RuntimeException"), id="ort_RuntimeException"),
    pytest.param(lambda: _ort_exc("Fail"), id="ort_Fail"),
])
def test_765_environment_failures_still_fail_over_bit_identical(make_exc, monkeypatch, tmp_path, caplog):
    """#765 direction 1: environment/runtime-class inference failures (incl. real
    onnxruntime classes, which derive from bare Exception) still fail over to a
    vector bit-for-bit equal to explicit NG_EMBED_REMOTE=hf, with no ERROR log.

    Env: NG_EMBED_REMOTE set to hf only in the ground-truth helper, then deleted
    and asserted unset; HF_TOKEN set (unused)."""
    caplog.set_level(logging.DEBUG, logger="ng_embed")
    monkeypatch.delenv("NG_EMBED_REMOTE", raising=False)
    expected = _explicit_remote_result(monkeypatch, tmp_path / "explicit", _FakeTok, _call_embed_short)
    exc = make_exc()
    emb, session = _failover_instance(monkeypatch, tmp_path / "failover", _FakeTok, _RaisingSession(exc))
    monkeypatch.setattr(emb, "_hf_post", _text_dependent_post)
    assert os.environ.get("NG_EMBED_REMOTE") is None

    got = _call_embed_short(emb)

    assert session.calls == 1
    assert np.array_equal(got[0], expected[0])
    warns = _failover_warnings(caplog)
    assert len(warns) == 1 and warns[0].exc_info[1] is exc
    assert not _log_records(caplog, logging.ERROR)


@pytest.mark.parametrize("bug_cls", _BUG_CLASSES, ids=lambda c: c.__name__)
def test_765_bug_class_defect_is_raised_logged_and_never_reaches_remote(bug_cls, monkeypatch, tmp_path, caplog):
    """#765 direction 2: a bug-class defect from local inference is raised as the
    original exception, logged at ERROR with traceback, and NEVER reaches the
    remote API, the quarantine file, or the failover state.

    Env: NG_EMBED_REMOTE deleted and asserted unset; HF_TOKEN set (unused)."""
    caplog.set_level(logging.DEBUG, logger="ng_embed")
    exc = bug_cls("simulated defect")
    emb, session = _failover_instance(monkeypatch, tmp_path, _FakeTok, _RaisingSession(exc))
    posts = []
    monkeypatch.setattr(emb, "_hf_post", _counting_post(posts))
    assert os.environ.get("NG_EMBED_REMOTE") is None

    with pytest.raises(bug_cls) as ei:
        emb.embed("hello world")

    assert ei.value is exc
    assert posts == [], "a defect must not be masked behind a remote vector"
    assert emb._remote_mode is False and emb._model_failed is False
    assert not (tmp_path / "failed_embeds.jsonl").exists()
    assert not _failover_warnings(caplog)
    errs = _log_records(caplog, logging.ERROR)
    assert len(errs) == 1 and errs[0].exc_info[1] is exc


@pytest.mark.parametrize("tokenizer_cls, call", _SITES_ALL)
def test_765_bug_class_raised_at_every_inference_site(tokenizer_cls, call, monkeypatch, tmp_path, caplog):
    """#765: the bug-class policy holds at all four inference-failover sites,
    not only the one embed() reaches.

    Env: NG_EMBED_REMOTE deleted and asserted unset; HF_TOKEN set (unused)."""
    caplog.set_level(logging.DEBUG, logger="ng_embed")
    exc = TypeError("simulated defect")
    emb, session = _failover_instance(monkeypatch, tmp_path, tokenizer_cls, _RaisingSession(exc))
    posts = []
    monkeypatch.setattr(emb, "_hf_post", _counting_post(posts))
    assert os.environ.get("NG_EMBED_REMOTE") is None

    with pytest.raises(TypeError) as ei:
        call(emb)

    assert ei.value is exc and session.calls == 1
    assert posts == [] and emb._remote_mode is False
    assert len(_log_records(caplog, logging.ERROR)) == 1


def test_765_real_indexerror_from_model_output_shape_is_raised(monkeypatch, tmp_path, caplog):
    """#765 with a real defect, not a simulated exception: a session returning
    one output makes the code's own outputs[1] raise IndexError.

    Env: NG_EMBED_REMOTE deleted and asserted unset; HF_TOKEN set (unused)."""
    caplog.set_level(logging.DEBUG, logger="ng_embed")
    emb, session = _failover_instance(monkeypatch, tmp_path, _FakeTok, _OneOutputSession())
    posts = []
    monkeypatch.setattr(emb, "_hf_post", _counting_post(posts))
    assert os.environ.get("NG_EMBED_REMOTE") is None

    with pytest.raises(IndexError):
        emb.embed("hello world")

    assert session.calls == 1 and posts == [] and emb._remote_mode is False
    errs = _log_records(caplog, logging.ERROR)
    assert len(errs) == 1 and errs[0].exc_info[0] is IndexError


def _load_failure_instance(monkeypatch, cache_dir, exc):
    """NG_EMBED_REMOTE unset (asserted); the local model download/load raises exc."""
    import huggingface_hub

    monkeypatch.delenv("NG_EMBED_REMOTE", raising=False)
    assert os.environ.get("NG_EMBED_REMOTE") is None
    monkeypatch.setenv("HF_TOKEN", "tok-test")

    def boom(**kw):
        raise exc

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", boom)
    emb = NGEmbed()
    emb._config["cache_dir"] = str(cache_dir)
    emb._tokenizer = _FakeTok()
    return emb


def _hub_exc(name):
    errors = pytest.importorskip("huggingface_hub.errors")
    return getattr(errors, name)(f"simulated hub {name}")


@pytest.mark.parametrize("make_exc", [
    pytest.param(lambda: RuntimeError("runtime"), id="RuntimeError"),
    pytest.param(lambda: _hub_exc("LocalEntryNotFoundError"), id="hub_LocalEntryNotFoundError"),
    pytest.param(lambda: Exception("bare"), id="bare_Exception"),
    pytest.param(lambda: ImportError("no onnxruntime"), id="ImportError"),
])
def test_765_load_environment_failures_still_fail_over(make_exc, monkeypatch, tmp_path, caplog):
    """#765 direction 1 at the load site: environment-class load failures still
    fail over, bit-identical to explicit NG_EMBED_REMOTE=hf, with a traceback.

    Env: NG_EMBED_REMOTE set to hf only in the ground-truth helper, then deleted
    and asserted unset; HF_TOKEN set (unused)."""
    caplog.set_level(logging.DEBUG, logger="ng_embed")
    monkeypatch.delenv("NG_EMBED_REMOTE", raising=False)
    expected = _explicit_remote_result(monkeypatch, tmp_path / "explicit", _FakeTok, _call_embed_short)
    exc = make_exc()
    emb = _load_failure_instance(monkeypatch, tmp_path / "failover", exc)
    monkeypatch.setattr(emb, "_hf_post", _text_dependent_post)

    got = _call_embed_short(emb)

    assert np.array_equal(got[0], expected[0])
    assert emb._remote_mode is True and emb._model_failed is True
    warns = _failover_warnings(caplog)
    assert len(warns) == 1 and warns[0].exc_info[1] is exc
    assert not _log_records(caplog, logging.ERROR)


@pytest.mark.parametrize("make_exc", [
    pytest.param(lambda: TypeError("defect"), id="TypeError"),
    pytest.param(lambda: _hub_exc("HFValidationError"), id="hub_HFValidationError_is_ValueError"),
])
def test_765_load_bug_class_defect_raises_instead_of_failing_over(make_exc, monkeypatch, tmp_path, caplog):
    """#765 direction 2 at the load site: a defect/misconfiguration during load
    (TypeError; HF's HFValidationError for a bad repo id, a ValueError) raises,
    is logged at ERROR, and does not put the process into remote mode.

    Env: NG_EMBED_REMOTE deleted and asserted unset; HF_TOKEN set (unused)."""
    caplog.set_level(logging.DEBUG, logger="ng_embed")
    exc = make_exc()
    emb = _load_failure_instance(monkeypatch, tmp_path, exc)
    posts = []
    monkeypatch.setattr(emb, "_hf_post", _counting_post(posts))

    with pytest.raises(type(exc)) as ei:
        emb.embed("hello world")

    assert ei.value is exc and posts == []
    assert emb._remote_mode is False and emb._model_loaded is False
    errs = _log_records(caplog, logging.ERROR)
    assert len(errs) == 1 and errs[0].exc_info[1] is exc


# ---- #763: _ensure_model(require_local=...) --------------------------------

def test_763_require_local_false_on_load_failure_without_failover_or_network(monkeypatch, tmp_path, caplog):
    """A caller that needs LOCAL gets False when the load fails: no failover, no
    remote call (neither _hf_post nor urllib), and the default path is unharmed.

    Env: NG_EMBED_REMOTE deleted and asserted unset; HF_TOKEN set (unused)."""
    import urllib.request

    caplog.set_level(logging.DEBUG, logger="ng_embed")
    emb = _load_failure_instance(monkeypatch, tmp_path, _hub_exc("LocalEntryNotFoundError"))
    posts = []
    monkeypatch.setattr(emb, "_hf_post", _counting_post(posts))
    urlopens = []
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: urlopens.append(1))
    assert os.environ.get("NG_EMBED_REMOTE") is None

    assert emb._ensure_model(require_local=True) is False

    assert emb._remote_mode is False and emb._model_loaded is False
    assert emb._model_failed is True
    assert posts == [] and urlopens == []
    assert not _failover_warnings(caplog)

    monkeypatch.setattr(emb, "_hf_post", _text_dependent_post)
    expected = _explicit_remote_result(monkeypatch, tmp_path / "explicit", _FakeTok, _call_embed_short)
    monkeypatch.delenv("NG_EMBED_REMOTE", raising=False)
    got = emb.embed("hello world")
    assert emb._remote_mode is True
    assert np.array_equal(got, expected[0]), "default callers keep the R5 failover"


def test_763_require_local_true_when_local_loads_and_stays_local(monkeypatch, tmp_path):
    """require_local=True is True on a healthy local load and never enters remote mode.

    Env: NG_EMBED_REMOTE deleted and asserted unset; HF_TOKEN set (unused).
    The download, ORT session and tokenizer are stubbed; nothing is loaded."""
    import huggingface_hub
    import onnxruntime
    import tokenizers

    class _Tok:
        def no_truncation(self):
            return None

        def enable_padding(self, **k):
            return None

    class _TokFactory:
        @staticmethod
        def from_pretrained(model_id):
            return _Tok()

    class _Sess:
        def __init__(self, *a, **k):
            pass

    monkeypatch.delenv("NG_EMBED_REMOTE", raising=False)
    assert os.environ.get("NG_EMBED_REMOTE") is None
    monkeypatch.setenv("HF_TOKEN", "tok-test")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **kw: "stub.onnx")
    monkeypatch.setattr(onnxruntime, "InferenceSession", _Sess)
    monkeypatch.setattr(tokenizers, "Tokenizer", _TokFactory)
    emb = NGEmbed()
    emb._config["cache_dir"] = str(tmp_path)

    assert emb._ensure_model(require_local=True) is True
    assert emb._ensure_model(require_local=True) is True
    assert emb._remote_mode is False and emb._model_failed is False
    assert isinstance(emb._session, _Sess)


def test_763_require_local_false_when_remote_selected_or_already_failed_over(monkeypatch, tmp_path):
    """require_local=True is False when NG_EMBED_REMOTE=hf selects remote (no local
    load attempted) and after a prior failover; the default call stays True.

    Env: first half NG_EMBED_REMOTE set to hf then deleted; second half unset
    (asserted); HF_TOKEN set (unused)."""
    import huggingface_hub

    loads = []
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **kw: loads.append(1) or (_ for _ in ()).throw(OSError("no")))

    monkeypatch.delenv("NG_EMBED_REMOTE", raising=False)
    monkeypatch.setenv("NG_EMBED_REMOTE", "hf")
    monkeypatch.setenv("HF_TOKEN", "tok-test")
    assert os.environ.get("NG_EMBED_REMOTE") == "hf"
    emb = NGEmbed()
    assert emb._ensure_model(require_local=True) is False
    assert loads == [] and emb._remote_mode is True
    assert emb._ensure_model() is True

    NGEmbed.reset_instance()
    monkeypatch.delenv("NG_EMBED_REMOTE", raising=False)
    assert os.environ.get("NG_EMBED_REMOTE") is None
    emb2 = NGEmbed()
    emb2._config["cache_dir"] = str(tmp_path)
    assert emb2._ensure_model() is True
    assert emb2._remote_mode is True and loads == [1]
    assert emb2._ensure_model(require_local=True) is False
    assert loads == [1], "no second load attempt once failed over"


def test_763_require_local_still_raises_a_bug_class_load_defect(monkeypatch, tmp_path):
    """A bug-class defect during a require_local load raises; it is not reported
    as 'local unavailable'.

    Env: NG_EMBED_REMOTE deleted and asserted unset; HF_TOKEN set (unused)."""
    exc = TypeError("defect")
    emb = _load_failure_instance(monkeypatch, tmp_path, exc)

    with pytest.raises(TypeError) as ei:
        emb._ensure_model(require_local=True)

    assert ei.value is exc and emb._model_loaded is False and emb._remote_mode is False


def test_763_reembed_snowflake_aborts_when_local_unavailable_and_never_goes_remote(monkeypatch, tmp_path):
    """reembed_snowflake.main() must abort (exit 1) when the local model is
    unavailable, with no failover, no remote call, and vectors untouched. Input
    is a temp vectors file; --dry-run is passed so a regression cannot write.

    Env: NG_EMBED_REMOTE deleted and asserted unset; HF_TOKEN set (unused).
    Only the local model download is made to fail; that is what 'local is
    unavailable' means, and it is not a remote embedding call."""
    import urllib.request

    import importlib.util

    import msgpack

    # Load by explicit path: a plain import can resolve to the main checkout's copy.
    rs_path = os.path.join(_REPO_ROOT, "reembed_snowflake.py")
    spec = importlib.util.spec_from_file_location("reembed_snowflake_under_test", rs_path)
    rs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rs)
    assert rs.__file__ == rs_path
    vec_path = tmp_path / "vectors.msgpack"
    vec_path.write_bytes(msgpack.packb({"entries": {"k": {
        "content": "hello", "embedding": np.zeros(4, dtype=np.float32).tobytes(),
    }}}))
    before = vec_path.read_bytes()
    monkeypatch.setattr(rs, "VECTORS_PATH", str(vec_path))
    assert rs.VECTORS_PATH == str(vec_path), "must not point at real vectors"
    monkeypatch.setattr(sys, "argv", ["reembed_snowflake.py", "--dry-run"])
    monkeypatch.setattr(ng_embed_mod.time, "sleep", lambda s: None)

    emb = _load_failure_instance(monkeypatch, tmp_path / "cache", _hub_exc("LocalEntryNotFoundError"))
    NGEmbed._instance = emb
    posts = []
    monkeypatch.setattr(emb, "_hf_post", _counting_post(posts))
    urlopens = []
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: urlopens.append(1))

    with pytest.raises(SystemExit) as ei:
        rs.main()

    assert ei.value.code == 1
    assert posts == [] and urlopens == []
    assert emb._remote_mode is False
    assert vec_path.read_bytes() == before


# ---------------------------------------------------------------------------
# C3 (worker-003, delta-LE finding F1): require_local is honored by the
# inference path, not only at load. Every test below sets/deletes
# NG_EMBED_REMOTE itself and asserts it before the call under test; the only
# NG_EMBED_* name used is NG_EMBED_REMOTE (plus the autouse fixture's deletes
# of NG_EMBED_ALLOW_HASH_FALLBACK / NG_EMBED_TID_ENDPOINT). These tests do NOT
# stub _ensure_model: the real require_local guard is what is under test.
# ---------------------------------------------------------------------------

class _OkSession:
    """Working stand-in returning the real model's two outputs (the second is the
    per-row vector). Optionally raises on the Nth run() call."""

    def __init__(self, fail_on_call=None, exc=None):
        self.calls = 0
        self.fail_on_call = fail_on_call
        self.exc = exc

    def run(self, output_names, feed):
        self.calls += 1
        if self.fail_on_call is not None and self.calls == self.fail_on_call:
            raise self.exc
        n = feed["input_ids"].shape[0]
        return [np.zeros((n, 768), dtype=np.float32), np.full((n, 768), 0.5, dtype=np.float32)]


def _local_ready_instance(monkeypatch, cache_dir, tokenizer_cls, session):
    """Local model 'loaded' with the REAL _ensure_model; NG_EMBED_REMOTE unset (asserted)."""
    monkeypatch.delenv("NG_EMBED_REMOTE", raising=False)
    assert os.environ.get("NG_EMBED_REMOTE") is None
    monkeypatch.setenv("HF_TOKEN", "tok-test")
    emb = NGEmbed()
    emb._config["cache_dir"] = str(cache_dir)
    emb._model_loaded = True
    emb._tokenizer = tokenizer_cls()
    emb._session = session
    return emb


def _spy_remote(monkeypatch, emb):
    """Count every way a remote call could happen; any nonzero count is a failure."""
    import urllib.request

    seen = {"post": 0, "call": 0, "urlopen": 0}

    def post(url, payload, timeout=30):
        seen["post"] += 1
        return _text_dependent_post(url, payload, timeout)

    orig_call = emb._hf_remote_call

    def call(*a, **k):
        seen["call"] += 1
        return orig_call(*a, **k)

    def urlopen(*a, **k):
        seen["urlopen"] += 1
        raise AssertionError("network call attempted")

    monkeypatch.setattr(emb, "_hf_post", post)
    monkeypatch.setattr(emb, "_hf_remote_call", call)
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    return seen


_NO_REMOTE = {"post": 0, "call": 0, "urlopen": 0}


def _rl_batch_short(emb):
    return emb.embed_batch(["alpha beta", "gamma delta"], normalize=True, is_query=True, require_local=True)


def _rl_batch_long(emb):
    return emb.embed_batch([_LONG_TEXT], require_local=True)


def _rl_windows_long(emb):
    return [emb.embed_windows(_LONG_TEXT, require_local=True).pooled]


def _rl_windows_short(emb):
    return [emb.embed_windows("hello world", require_local=True).pooled]


def _rl_embed(emb):
    return [emb.embed("hello world", require_local=True)]


_RL_SITES = [
    pytest.param(_FakeTok, _rl_batch_short, id="embed_batch_short"),
    pytest.param(_ClsSepTok, _rl_batch_long, id="embed_batch_long"),
    pytest.param(_ClsSepTok, _rl_windows_long, id="embed_windows_long"),
    pytest.param(_FakeTok, _rl_windows_short, id="embed_windows_short"),
    pytest.param(_FakeTok, _rl_embed, id="embed"),
]


@pytest.mark.parametrize("tokenizer_cls, call", _RL_SITES)
def test_c3_require_local_inference_failure_raises_without_any_remote_call(
    tokenizer_cls, call, monkeypatch, tmp_path, caplog,
):
    """An environment-class local inference failure with require_local=True raises
    EmbeddingUnavailableError and makes NO remote call of any kind, does not flip
    _remote_mode, keeps _model_loaded, and logs the traceback once.

    Env: NG_EMBED_REMOTE deleted and asserted unset; HF_TOKEN set (unused)."""
    caplog.set_level(logging.DEBUG, logger="ng_embed")
    session = _BoomSession()
    emb = _local_ready_instance(monkeypatch, tmp_path, tokenizer_cls, session)
    remote = _spy_remote(monkeypatch, emb)
    assert os.environ.get("NG_EMBED_REMOTE") is None

    with pytest.raises(EmbeddingUnavailableError) as ei:
        call(emb)

    assert isinstance(ei.value.__cause__, RuntimeError)
    assert session.calls >= 1, "local inference must have been attempted"
    assert remote == _NO_REMOTE
    assert emb._remote_mode is False and emb._model_loaded is True and emb._model_failed is True
    assert not _failover_warnings(caplog)
    warns = [r for r in _log_records(caplog, logging.WARNING) if "not failing over" in r.getMessage()]
    assert len(warns) == 1 and warns[0].exc_info[0] is RuntimeError
    assert not (tmp_path / "failed_embeds.jsonl").exists()


@pytest.mark.parametrize("tokenizer_cls, call", _RL_SITES)
def test_c3_require_local_bug_class_defect_still_raised_as_original(
    tokenizer_cls, call, monkeypatch, tmp_path, caplog,
):
    """A bug-class defect with require_local=True is raised as the original
    exception object (not wrapped, not masked), logged at ERROR, no remote call.

    Env: NG_EMBED_REMOTE deleted and asserted unset; HF_TOKEN set (unused)."""
    caplog.set_level(logging.DEBUG, logger="ng_embed")
    exc = TypeError("simulated defect")
    emb = _local_ready_instance(monkeypatch, tmp_path, tokenizer_cls, _RaisingSession(exc))
    remote = _spy_remote(monkeypatch, emb)
    assert os.environ.get("NG_EMBED_REMOTE") is None

    with pytest.raises(TypeError) as ei:
        call(emb)

    assert ei.value is exc
    assert remote == _NO_REMOTE and emb._remote_mode is False
    errs = _log_records(caplog, logging.ERROR)
    assert len(errs) == 1 and errs[0].exc_info[1] is exc


@pytest.mark.parametrize("state", ["explicit_hf", "after_default_failover"])
@pytest.mark.parametrize("tokenizer_cls, call", [
    pytest.param(_FakeTok, _rl_batch_short, id="embed_batch"),
    pytest.param(_FakeTok, _rl_windows_short, id="embed_windows"),
    pytest.param(_FakeTok, _rl_embed, id="embed"),
])
def test_c3_require_local_refuses_when_already_in_remote_mode(
    state, tokenizer_cls, call, monkeypatch, tmp_path,
):
    """If _remote_mode is already True (NG_EMBED_REMOTE=hf, or an earlier
    default-caller failover in the same process) a require_local call REFUSES
    with EmbeddingUnavailableError and makes no further remote call.

    Env: explicit_hf sets NG_EMBED_REMOTE=hf (asserted); after_default_failover
    deletes it (asserted unset); HF_TOKEN set (unused)."""
    if state == "explicit_hf":
        monkeypatch.delenv("NG_EMBED_REMOTE", raising=False)
        monkeypatch.setenv("NG_EMBED_REMOTE", "hf")
        monkeypatch.setenv("HF_TOKEN", "tok-test")
        assert os.environ.get("NG_EMBED_REMOTE") == "hf"
        emb = NGEmbed()
        emb._config["cache_dir"] = str(tmp_path)
        emb._tokenizer = tokenizer_cls()
        remote = _spy_remote(monkeypatch, emb)
    else:
        emb = _local_ready_instance(monkeypatch, tmp_path, tokenizer_cls, _BoomSession())
        remote = _spy_remote(monkeypatch, emb)
        assert os.environ.get("NG_EMBED_REMOTE") is None
        emb.embed_batch(["warm up"])
        assert emb._remote_mode is True and remote["post"] == 1

    before = dict(remote)
    with pytest.raises(EmbeddingUnavailableError):
        call(emb)

    assert remote == before, "a refused require_local call must not add any remote call"


# embed() is excluded here: it guards, then embed_windows() guards again, so a flip
# between the two makes the second guard refuse (safe, zero remote calls, and
# covered by the already-in-remote-mode test). embed_windows_short covers its path.
_RL_RACE_SITES = [p for p in _RL_SITES if p.id != "embed"]


@pytest.mark.parametrize("tokenizer_cls, call", _RL_RACE_SITES)
def test_c3_require_local_never_takes_the_remote_arm_even_if_remote_mode_flips_mid_call(
    tokenizer_cls, call, monkeypatch, tmp_path,
):
    """Race guard: another thread's default-caller failover can set _remote_mode
    after this call passed the guard. A require_local call must still use the
    local session, never the remote arm.

    Env: NG_EMBED_REMOTE deleted and asserted unset; HF_TOKEN set (unused)."""
    session = _OkSession()
    emb = _local_ready_instance(monkeypatch, tmp_path, tokenizer_cls, session)
    remote = _spy_remote(monkeypatch, emb)
    real_guard = emb._require_model

    def guard_then_flip(require_local=False):
        real_guard(require_local)
        emb._remote_mode = True

    monkeypatch.setattr(emb, "_require_model", guard_then_flip)
    assert os.environ.get("NG_EMBED_REMOTE") is None

    out = call(emb)

    assert session.calls >= 1
    assert remote == _NO_REMOTE
    assert all(v.shape == (768,) for v in out)


def test_c3_default_and_explicit_false_still_fail_over_bit_identical(monkeypatch, tmp_path):
    """Regression pin: without require_local (omitted, or False) embed_batch keeps
    the R5 failover, bit-for-bit equal to explicit NG_EMBED_REMOTE=hf.

    Env: NG_EMBED_REMOTE set to hf only in the ground-truth helper, then deleted
    and asserted unset; HF_TOKEN set (unused)."""
    monkeypatch.delenv("NG_EMBED_REMOTE", raising=False)
    texts = ["alpha beta", "gamma delta"]
    expected = _explicit_remote_result(
        monkeypatch, tmp_path / "explicit", _FakeTok, lambda e: e.embed_batch(texts),
    )
    for style, kwargs in (("omitted", {}), ("explicit False", {"require_local": False})):
        emb = _local_ready_instance(monkeypatch, tmp_path / style.replace(" ", "_"), _FakeTok, _BoomSession())
        remote = _spy_remote(monkeypatch, emb)
        assert os.environ.get("NG_EMBED_REMOTE") is None
        got = emb.embed_batch(texts, **kwargs)
        assert remote["post"] == 1 and emb._remote_mode is True, style
        assert all(np.array_equal(g, e) for g, e in zip(got, expected)), style


def _load_reembed_module():
    import importlib.util

    rs_path = os.path.join(_REPO_ROOT, "reembed_snowflake.py")
    spec = importlib.util.spec_from_file_location("reembed_snowflake_under_test", rs_path)
    rs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rs)
    assert rs.__file__ == rs_path
    return rs


@pytest.mark.parametrize("dry_run", [True, False], ids=["dry_run", "write_path_enabled"])
def test_c3_reembed_aborts_when_local_inference_fails_mid_run(dry_run, monkeypatch, tmp_path):
    """Tool level: local inference succeeds for batch 1 and raises on batch 2 of 3.
    main() must exit non-zero, make ZERO remote calls, never attempt batch 3, and
    leave the vectors file byte-identical: the only write is at the very end, so
    nothing from batch 1 (in memory only), 2 or 3 reaches disk. The write_path_enabled
    variant omits --dry-run so a regression that swallowed the error and carried
    on would change the bytes; VECTORS_PATH is a temp file (asserted).

    Env: NG_EMBED_REMOTE deleted and asserted unset; HF_TOKEN set (unused)."""
    import msgpack

    rs = _load_reembed_module()
    entries = {
        f"e{i:03d}": {"content": f"content number {i}", "embedding": np.zeros(4, dtype=np.float32).tobytes()}
        for i in range(130)
    }
    vec_path = tmp_path / "vectors.msgpack"
    vec_path.write_bytes(msgpack.packb({"entries": entries}))
    before = vec_path.read_bytes()
    monkeypatch.setattr(rs, "VECTORS_PATH", str(vec_path))
    assert rs.VECTORS_PATH == str(vec_path), "must not point at real vectors"
    monkeypatch.setattr(sys, "argv", ["reembed_snowflake.py"] + (["--dry-run"] if dry_run else []))
    monkeypatch.setattr(ng_embed_mod.time, "sleep", lambda s: None)

    session = _OkSession(fail_on_call=2, exc=RuntimeError("simulated ORT failure on batch 2"))
    emb = _local_ready_instance(monkeypatch, tmp_path / "cache", _FakeTok, session)
    NGEmbed._instance = emb
    remote = _spy_remote(monkeypatch, emb)
    assert os.environ.get("NG_EMBED_REMOTE") is None

    with pytest.raises(SystemExit) as ei:
        rs.main()

    assert ei.value.code == 1
    assert session.calls == 2, "batch 1 ok, batch 2 failed, batch 3 never attempted"
    assert remote == _NO_REMOTE and emb._remote_mode is False
    assert vec_path.read_bytes() == before


# ---------------------------------------------------------------------------
# C4 (worker-004, Josh's #766 ruling): failover lasts only while local is down.
# The code re-probes LOCAL ONLY (no network, no caller text) and fails back,
# logging loudly both ways. All timing goes through ng_embed._now, patched with
# a controllable clock, so nothing here sleeps. NG_EMBED_* names used:
# NG_EMBED_REMOTE and NG_EMBED_REPROBE_SECS (each test sets or deletes both
# itself and asserts them before the call under test).
# ---------------------------------------------------------------------------

import socket

_TEXT = "hello world"
_LOCAL_VEC = np.full(768, 0.5, dtype=np.float32)


def _remote_vec(text):
    return np.asarray(_text_vec(text), dtype=np.float32)


class _Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


class _StubTok(_FakeTok):
    """_FakeTok plus the no_truncation() that ng_embed calls on a loaded tokenizer."""

    def no_truncation(self):
        return None


class _SwitchSession:
    """Local ONNX session that fails while scenario.up is False; records every feed."""

    def __init__(self, scenario):
        self.scn = scenario
        self.calls = 0
        self.feeds = []

    def run(self, output_names, feed):
        self.calls += 1
        self.feeds.append(feed["input_ids"].copy())
        if self.scn.probe_exc is not None:
            raise self.scn.probe_exc
        if not self.scn.up:
            raise RuntimeError("simulated: local is down")
        n = feed["input_ids"].shape[0]
        return [np.zeros((n, 768), dtype=np.float32), np.full((n, 768), 0.5, dtype=np.float32)]


class _Scenario:
    """A failover episode of either kind ('load' or 'inference') with a switchable
    'local is up' flag, a controllable clock, and the remote spy installed."""

    def __init__(self, kind, monkeypatch, tmp_path, interval="60"):
        self.kind, self.up, self.probe_exc, self.dl_calls = kind, False, None, []
        self.clock = _Clock()
        monkeypatch.setattr(ng_embed_mod, "_now", self.clock)
        monkeypatch.delenv("NG_EMBED_REMOTE", raising=False)
        if interval is None:
            monkeypatch.delenv("NG_EMBED_REPROBE_SECS", raising=False)
            assert os.environ.get("NG_EMBED_REPROBE_SECS") is None
        else:
            monkeypatch.setenv("NG_EMBED_REPROBE_SECS", interval)
            assert os.environ.get("NG_EMBED_REPROBE_SECS") == interval
        assert os.environ.get("NG_EMBED_REMOTE") is None
        monkeypatch.setenv("HF_TOKEN", "tok-test")
        self.session = _SwitchSession(self)
        if kind == "inference":
            self.emb = _local_ready_instance(monkeypatch, tmp_path, _FakeTok, self.session)
        else:
            self._install_world(monkeypatch)
            self.emb = NGEmbed()
            self.emb._config["cache_dir"] = str(tmp_path)
        self.remote = _spy_remote(monkeypatch, self.emb)

    def _install_world(self, monkeypatch):
        import huggingface_hub
        import onnxruntime
        import tokenizers

        def fake_download(**kw):
            self.dl_calls.append(dict(kw))
            if kw.get("local_files_only") and self.probe_exc is not None:
                raise self.probe_exc
            if not self.up:
                raise OSError("simulated: model files unavailable")
            return "stub-" + kw["filename"].replace("/", "_")

        class _TokFactory:
            @staticmethod
            def from_pretrained(model_id):
                return _StubTok()

            @staticmethod
            def from_file(path):
                return _StubTok()

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
        monkeypatch.setattr(onnxruntime, "InferenceSession", lambda *a, **k: self.session)
        monkeypatch.setattr(tokenizers, "Tokenizer", _TokFactory)

    def start_episode(self):
        vec = self.emb.embed(_TEXT)
        assert np.array_equal(vec, _remote_vec(_TEXT)), "the failing call is served remotely"
        assert self.emb._remote_mode is True and self.emb._failover_since == self.clock.t
        assert self.emb._failover_kind == self.kind
        return vec

    def probes(self):
        if self.kind == "inference":
            probe_ids = np.arange(len(ng_embed_mod._PROBE_TEXT))
            return sum(1 for f in self.session.feeds
                       if f.shape == (1, len(probe_ids)) and np.array_equal(f[0], probe_ids))
        onnx = ng_embed_mod._DEFAULT_CONFIG["onnx_filename"]
        return sum(1 for c in self.dl_calls if c.get("local_files_only") and c["filename"] == onnx)


def _msgs(caplog, level, needle):
    return [r for r in _log_records(caplog, level) if needle in r.getMessage()]


_KINDS = [pytest.param("load", id="load_episode"), pytest.param("inference", id="inference_episode")]


@pytest.mark.parametrize("kind", _KINDS)
def test_c4_fails_back_when_local_recovers_and_logs_both_ways(kind, monkeypatch, tmp_path, caplog):
    """After the interval a default call re-probes local, fails BACK, and is served
    locally; loud WARNING with duration and served-count; probes stay local-only.

    Env: NG_EMBED_REMOTE deleted (asserted), NG_EMBED_REPROBE_SECS=60 (asserted)."""
    caplog.set_level(logging.DEBUG, logger="ng_embed")
    scn = _Scenario(kind, monkeypatch, tmp_path)
    emb = scn.emb
    scn.start_episode()
    scn.clock.advance(10)
    assert np.array_equal(emb.embed(_TEXT), _remote_vec(_TEXT))
    scn.clock.advance(10)
    assert np.array_equal(emb.embed(_TEXT), _remote_vec(_TEXT))
    assert scn.probes() == 0 and emb._remote_mode is True, "no probe inside the interval"

    scn.up = True
    scn.clock.advance(41)
    posts_before = scn.remote["post"]
    got = emb.embed(_TEXT)

    assert np.array_equal(got, _LOCAL_VEC), "served by the recovered local model"
    assert scn.remote["post"] == posts_before, "the probe and the local call made no remote call"
    assert emb._remote_mode is False and emb._failover_since is None
    assert emb._model_failed is True, "documented meaning: local has failed at least once"
    assert emb._session is scn.session
    assert scn.probes() == 1
    assert len(_failover_warnings(caplog)) == 1
    back = _msgs(caplog, logging.WARNING, "failing BACK")
    assert len(back) == 1
    assert "after 61.0s" in back[0].getMessage()
    assert "served remotely during the episode: 3 calls, 3 vectors" in back[0].getMessage()
    if kind == "load":
        onnx_probe = [c for c in scn.dl_calls if c.get("local_files_only")]
        assert {c["filename"] for c in onnx_probe} == {ng_embed_mod._DEFAULT_CONFIG["onnx_filename"], "tokenizer.json"}


@pytest.mark.parametrize("kind", _KINDS)
def test_c4_still_down_probes_exactly_once_per_interval_and_logs_once_each(kind, monkeypatch, tmp_path, caplog):
    """Local stays down: many calls inside an interval trigger no probe, exactly one
    probe fires at each interval boundary (even for two calls in the same instant),
    each failed probe logs once with no traceback, and a failed load probe leaves the
    state exactly as it was.

    Env: NG_EMBED_REMOTE deleted (asserted), NG_EMBED_REPROBE_SECS=60 (asserted)."""
    caplog.set_level(logging.DEBUG, logger="ng_embed")
    scn = _Scenario(kind, monkeypatch, tmp_path)
    emb = scn.emb
    scn.start_episode()

    for _ in range(50):
        scn.clock.advance(1)
        emb.embed(_TEXT)
    assert scn.probes() == 0, "no probe before the interval elapses"

    snapshot = lambda: (emb._session, emb._tokenizer, emb._remote_mode, emb._model_loaded,
                        emb._failover_since, emb._failover_kind)
    before = snapshot()
    scn.clock.advance(10)
    emb.embed(_TEXT)
    emb.embed(_TEXT)
    assert scn.probes() == 1, "two calls at the boundary make one probe"
    assert snapshot() == before, "a failed probe leaves the state exactly as it was"

    for _ in range(59):
        scn.clock.advance(1)
        emb.embed(_TEXT)
    assert scn.probes() == 1
    scn.clock.advance(1)
    emb.embed(_TEXT)
    assert scn.probes() == 2

    failed = _msgs(caplog, logging.WARNING, "re-probe failed")
    assert len(failed) == 2, "one log line per failed probe, no more"
    assert all(r.exc_info is None for r in failed), "no traceback storm"
    assert "RuntimeError" in failed[0].getMessage() or "OSError" in failed[0].getMessage()
    assert not _msgs(caplog, logging.WARNING, "failing BACK")
    assert emb._remote_mode is True


def test_c4_explicit_remote_hf_never_probes_or_fails_back(monkeypatch, tmp_path, caplog):
    """NG_EMBED_REMOTE=hf is an operator choice: no episode, no probe, no fail-back,
    however much time passes, and the episode counters stay untouched.

    Env: NG_EMBED_REMOTE=hf (asserted), NG_EMBED_REPROBE_SECS=1 (asserted)."""
    caplog.set_level(logging.DEBUG, logger="ng_embed")
    clock = _Clock()
    monkeypatch.setattr(ng_embed_mod, "_now", clock)
    monkeypatch.delenv("NG_EMBED_REMOTE", raising=False)
    monkeypatch.setenv("NG_EMBED_REMOTE", "hf")
    monkeypatch.setenv("NG_EMBED_REPROBE_SECS", "1")
    monkeypatch.setenv("HF_TOKEN", "tok-test")
    assert os.environ.get("NG_EMBED_REMOTE") == "hf" and os.environ.get("NG_EMBED_REPROBE_SECS") == "1"
    emb = NGEmbed()
    emb._config["cache_dir"] = str(tmp_path)
    emb._tokenizer = _FakeTok()
    remote = _spy_remote(monkeypatch, emb)
    probes = []
    monkeypatch.setattr(emb, "_load_local", lambda *a, **k: probes.append("load") or (_ for _ in ()).throw(AssertionError("probed")))
    monkeypatch.setattr(emb, "_onnx_embed", lambda *a, **k: probes.append("infer") or (_ for _ in ()).throw(AssertionError("probed")))

    for _ in range(5):
        clock.advance(1000)
        assert np.array_equal(emb.embed(_TEXT), _remote_vec(_TEXT))
    emb._maybe_fail_back()

    assert probes == []
    assert emb._remote_mode is True and emb._failover_since is None
    assert emb._failover_remote_calls == 0 and remote["post"] == 5
    assert not _msgs(caplog, logging.WARNING, "failing BACK")


@pytest.mark.parametrize("kind", _KINDS)
def test_c4_reprobe_secs_zero_disables_and_says_so(kind, monkeypatch, tmp_path, caplog):
    """NG_EMBED_REPROBE_SECS=0 never probes, and the failover WARNING says re-probing is disabled.

    Env: NG_EMBED_REMOTE deleted (asserted), NG_EMBED_REPROBE_SECS=0 (asserted)."""
    caplog.set_level(logging.DEBUG, logger="ng_embed")
    scn = _Scenario(kind, monkeypatch, tmp_path, interval="0")
    scn.start_episode()
    scn.up = True
    scn.clock.advance(10 ** 7)
    assert np.array_equal(scn.emb.embed(_TEXT), _remote_vec(_TEXT))
    assert scn.probes() == 0 and scn.emb._remote_mode is True
    warns = _failover_warnings(caplog)
    assert len(warns) == 1 and "DISABLED" in warns[0].getMessage()


def test_c4_reprobe_secs_default_and_invalid_value(monkeypatch, tmp_path, caplog):
    """Unset uses the in-code default; a non-numeric value falls back to it with ONE warning.

    Env: NG_EMBED_REMOTE deleted (asserted); NG_EMBED_REPROBE_SECS unset (asserted), then "abc" (asserted)."""
    caplog.set_level(logging.DEBUG, logger="ng_embed")
    default = ng_embed_mod._REPROBE_SECS_DEFAULT
    assert default > 0
    scn = _Scenario("inference", monkeypatch, tmp_path, interval=None)
    scn.start_episode()
    assert scn.emb._reprobe_interval() == default
    scn.clock.advance(default - 1)
    scn.emb.embed(_TEXT)
    assert scn.probes() == 0
    scn.clock.advance(1)
    scn.emb.embed(_TEXT)
    assert scn.probes() == 1

    monkeypatch.setenv("NG_EMBED_REPROBE_SECS", "abc")
    assert os.environ.get("NG_EMBED_REPROBE_SECS") == "abc"
    for _ in range(3):
        assert scn.emb._reprobe_interval() == default
    assert len(_msgs(caplog, logging.WARNING, "is not a number")) == 1


def test_c4_fail_back_then_fail_over_again_logs_again(monkeypatch, tmp_path, caplog):
    """The failover WARNING is once per EPISODE, not once per process: fail over,
    fail back, fail over again, and both failovers and the fail-back are logged, in order.

    Env: NG_EMBED_REMOTE deleted (asserted), NG_EMBED_REPROBE_SECS=60 (asserted)."""
    caplog.set_level(logging.DEBUG, logger="ng_embed")
    scn = _Scenario("inference", monkeypatch, tmp_path)
    emb = scn.emb
    scn.start_episode()
    scn.up = True
    scn.clock.advance(60)
    assert np.array_equal(emb.embed(_TEXT), _LOCAL_VEC)
    assert emb._remote_mode is False

    scn.up = False
    scn.clock.advance(5)
    assert np.array_equal(emb.embed(_TEXT), _remote_vec(_TEXT))
    assert emb._remote_mode is True and emb._failover_since == scn.clock.t
    assert emb._failover_remote_calls == 1, "episode counters restart"

    order = [
        "BACK" if "failing BACK" in r.getMessage() else "OVER"
        for r in _log_records(caplog, logging.WARNING)
        if "failing BACK" in r.getMessage() or "failing over to HF remote" in r.getMessage()
    ]
    assert order == ["OVER", "BACK", "OVER"]


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("state", ["not_due", "due_still_down", "due_recovered"])
def test_c4_require_local_while_failed_over_may_probe_but_never_calls_remote(
    kind, state, monkeypatch, tmp_path,
):
    """A require_local call while failed over never makes a network call. It may
    trigger a DUE local probe: proceeds locally if that recovered local, otherwise
    refuses as before. It does not probe when no probe is due.

    Env: NG_EMBED_REMOTE deleted (asserted), NG_EMBED_REPROBE_SECS=60 (asserted)."""
    scn = _Scenario(kind, monkeypatch, tmp_path)
    emb = scn.emb
    scn.start_episode()
    remote_before = dict(scn.remote)
    if state != "not_due":
        scn.clock.advance(60)
    scn.up = state == "due_recovered"

    if state == "due_recovered":
        got = emb.embed_batch(["alpha beta"], require_local=True)
        assert np.array_equal(got[0], _LOCAL_VEC) and emb._remote_mode is False
    else:
        with pytest.raises(EmbeddingUnavailableError):
            emb.embed_batch(["alpha beta"], require_local=True)
        assert emb._remote_mode is True

    assert scn.remote == remote_before, "require_local must never add a remote call"
    assert scn.probes() == (0 if state == "not_due" else 1)


@pytest.mark.parametrize("kind", _KINDS)
def test_c4_only_one_thread_probes_and_others_keep_using_remote_without_waiting(kind, monkeypatch, tmp_path):
    """While another thread holds the probe lock (mid-probe), a due call neither
    probes nor blocks: it is served remotely. Once released, the next call probes.
    Simulated with the lock itself; no real race is claimed.

    Env: NG_EMBED_REMOTE deleted (asserted), NG_EMBED_REPROBE_SECS=60 (asserted)."""
    scn = _Scenario(kind, monkeypatch, tmp_path)
    emb = scn.emb
    scn.start_episode()
    scn.up = True
    scn.clock.advance(60)

    assert emb._probe_lock.acquire(blocking=False)
    try:
        assert np.array_equal(emb.embed(_TEXT), _remote_vec(_TEXT)), "served remotely, not blocked"
        assert scn.probes() == 0 and emb._remote_mode is True
    finally:
        emb._probe_lock.release()

    assert np.array_equal(emb.embed(_TEXT), _LOCAL_VEC), "the skipped call did not consume the interval"
    assert scn.probes() == 1


@pytest.mark.parametrize("kind", _KINDS)
def test_c4_bug_class_defect_in_a_probe_is_raised_not_swallowed_as_still_down(kind, monkeypatch, tmp_path, caplog):
    """A bug-class exception during a probe is raised (ERROR + traceback, as #765),
    leaves the episode untouched, releases the probe lock, and does not storm: the
    interval was consumed, so the next call inside it just uses remote.

    Env: NG_EMBED_REMOTE deleted (asserted), NG_EMBED_REPROBE_SECS=60 (asserted)."""
    caplog.set_level(logging.DEBUG, logger="ng_embed")
    scn = _Scenario(kind, monkeypatch, tmp_path)
    emb = scn.emb
    scn.start_episode()
    since = emb._failover_since
    exc = TypeError("simulated probe defect")
    scn.probe_exc = exc
    scn.clock.advance(60)

    with pytest.raises(TypeError) as ei:
        emb.embed(_TEXT)

    assert ei.value is exc
    assert emb._remote_mode is True and emb._failover_since == since
    errs = _log_records(caplog, logging.ERROR)
    assert len(errs) == 1 and errs[0].exc_info[1] is exc
    assert not _msgs(caplog, logging.WARNING, "re-probe failed")
    assert emb._probe_lock.acquire(blocking=False)
    emb._probe_lock.release()
    probes_after = scn.probes()
    scn.clock.advance(1)
    assert np.array_equal(emb.embed(_TEXT), _remote_vec(_TEXT))
    assert scn.probes() == probes_after


@pytest.mark.parametrize("kind", _KINDS)
def test_c4_probe_is_local_only_and_sends_only_the_fixed_probe_text(kind, monkeypatch, tmp_path):
    """The probe makes no remote call of any kind, no network attempt at all (socket
    connect and DNS raise), and sends only the fixed probe text, never caller text.
    Called directly (no caller in flight), so nothing else can explain a remote call.

    Env: NG_EMBED_REMOTE deleted (asserted), NG_EMBED_REPROBE_SECS=60 (asserted)."""
    scn = _Scenario(kind, monkeypatch, tmp_path)
    emb = scn.emb
    scn.start_episode()
    scn.up = True
    scn.clock.advance(60)
    remote_before = dict(scn.remote)
    attempts = []

    def blocked(*a, **k):
        attempts.append(1)
        raise AssertionError("network attempted by the probe")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)

    emb._maybe_fail_back()

    assert emb._remote_mode is False, "the probe itself performed the fail-back"
    assert scn.remote == remote_before and attempts == []
    assert len(ng_embed_mod._PROBE_TEXT) != len(_TEXT)
    if kind == "inference":
        probe_feed = scn.session.feeds[-1]
        assert probe_feed.shape == (1, len(ng_embed_mod._PROBE_TEXT))
        assert np.array_equal(probe_feed[0], np.arange(len(ng_embed_mod._PROBE_TEXT)))
        assert scn.probes() == 1
    else:
        probe_dl = [c for c in scn.dl_calls if c.get("local_files_only")]
        assert len(probe_dl) == 2 and all(c["local_files_only"] is True for c in probe_dl)
        assert not scn.session.feeds, "a load probe runs no inference at all"


class _Stopper:
    """Stand-in for the keepalive stop Event: wait() answers from a script, so the
    loop can be driven synchronously with no thread and no sleep."""

    def __init__(self, answers):
        self.answers = list(answers)

    def wait(self, interval):
        return self.answers.pop(0)


@pytest.mark.parametrize("remote_mode", [False, True], ids=["failed_back_to_local", "on_remote"])
def test_c4_keepalive_pings_only_while_on_remote(remote_mode, monkeypatch, tmp_path):
    """After a fail-back the keepalive loop must stop pinging the HF endpoint (the
    process is local again) and resume if it fails over again.

    Env: NG_EMBED_REMOTE deleted (asserted); NG_EMBED_REPROBE_SECS deleted (asserted)."""
    monkeypatch.delenv("NG_EMBED_REMOTE", raising=False)
    monkeypatch.delenv("NG_EMBED_REPROBE_SECS", raising=False)
    assert os.environ.get("NG_EMBED_REMOTE") is None
    assert os.environ.get("NG_EMBED_REPROBE_SECS") is None
    emb = NGEmbed()
    emb._remote_mode = remote_mode
    pings = []
    monkeypatch.setattr(emb, "_hf_remote_embed", lambda *a, **k: pings.append(a))
    emb._keepalive_stop = _Stopper([False, False, True])

    emb._keepalive_loop()

    assert len(pings) == (2 if remote_mode else 0)
