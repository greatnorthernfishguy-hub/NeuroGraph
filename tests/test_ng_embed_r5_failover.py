# tests/test_ng_embed_r5_failover.py
# ---- Changelog ----
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

from test_ng_embed_dualpass import _FakeTok  # reuse existing one-token-per-char fixture


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

    def run(self, *a, **k):
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

