# tests/test_ng_embed_r5_failover.py
# ---- Changelog ----
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
