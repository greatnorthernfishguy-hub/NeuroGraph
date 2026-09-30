# tests/test_ng_embed_r5_failover.py
# ---- Changelog ----
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


def _failover_instance(monkeypatch, cache_dir, tokenizer_cls):
    """NG_EMBED_REMOTE unset (asserted); ONNX 'loaded' but session.run always raises."""
    monkeypatch.delenv("NG_EMBED_REMOTE", raising=False)
    assert os.environ.get("NG_EMBED_REMOTE") is None
    monkeypatch.setenv("HF_TOKEN", "tok-test")
    emb = NGEmbed()
    emb._config["cache_dir"] = str(cache_dir)
    monkeypatch.setattr(emb, "_ensure_model", lambda: True)
    emb._model_loaded = True
    emb._tokenizer = tokenizer_cls()
    session = _BoomSession()
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
