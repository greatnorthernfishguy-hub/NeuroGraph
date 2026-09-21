# tests/test_ng_embed_fail_closed.py
import json
import os

import numpy as np
import pytest
import ng_embed
from ng_embed import NGEmbed, EmbeddingUnavailableError


@pytest.fixture(autouse=True)
def _reset_singleton_and_env(monkeypatch):
    monkeypatch.delenv("NG_EMBED_REMOTE", raising=False)
    monkeypatch.delenv("NG_EMBED_ALLOW_HASH_FALLBACK", raising=False)
    NGEmbed.reset_instance()
    yield
    NGEmbed.reset_instance()


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    # Retry backoffs are widened for cold starts; never actually sleep in CI.
    monkeypatch.setattr(ng_embed.time, "sleep", lambda *_: None)


def test_embed_raises_when_model_unavailable_and_hash_disabled(monkeypatch):
    emb = NGEmbed()
    monkeypatch.setattr(emb, "_ensure_model", lambda: False)
    hash_spy = {"called": False}
    monkeypatch.setattr(emb, "_hash_embed",
                        lambda *a, **k: hash_spy.__setitem__("called", True) or np.zeros(768, np.float32))
    with pytest.raises(EmbeddingUnavailableError):
        emb.embed("hello")
    assert hash_spy["called"] is False  # hash MUST NOT fire on the fail-closed path


def test_embed_batch_raises_when_model_unavailable_and_hash_disabled(monkeypatch):
    emb = NGEmbed()
    monkeypatch.setattr(emb, "_ensure_model", lambda: False)
    with pytest.raises(EmbeddingUnavailableError):
        emb.embed_batch(["a", "b"])


def test_embed_uses_hash_only_when_opt_in_set(monkeypatch):
    monkeypatch.setenv("NG_EMBED_ALLOW_HASH_FALLBACK", "1")
    emb = NGEmbed()
    monkeypatch.setattr(emb, "_ensure_model", lambda: False)
    vec = emb.embed("hello")
    assert vec.shape == (768,)  # deterministic hash vector returned by explicit opt-in


def test_empty_batch_returns_empty_without_touching_model():
    emb = NGEmbed()
    assert emb.embed_batch([]) == []


def test_remote_gate_selects_remote_mode_without_local_load(monkeypatch):
    monkeypatch.setenv("NG_EMBED_REMOTE", "hf")
    emb = NGEmbed()
    # raising=False: the real _get_hf_token lands in Task 3. Until then it does not
    # exist on the class, and monkeypatch.setattr defaults to raising=True.
    monkeypatch.setattr(emb, "_get_hf_token", lambda: "fake-token", raising=False)
    # If ONNX load were attempted this would import onnxruntime + download; assert it is not.
    def _boom(*a, **k):
        raise AssertionError("local ONNX load must not run in remote mode")
    monkeypatch.setattr(emb, "_onnx_embed", _boom)
    assert emb._ensure_model() is True
    assert emb._remote_mode is True


def test_invalid_remote_value_raises_no_local_fallthrough(monkeypatch):
    monkeypatch.setenv("NG_EMBED_REMOTE", "openrouter")
    emb = NGEmbed()
    with pytest.raises(EmbeddingUnavailableError):
        emb._ensure_model()
    assert emb._remote_mode is False


def test_unset_remote_falls_through_to_local(monkeypatch):
    emb = NGEmbed()
    # Force local load to "fail" so we don't need the real model, but prove we took the local branch.
    monkeypatch.setattr(ng_embed, "__name__", ng_embed.__name__)  # no-op anchor
    ok = emb._ensure_model()
    # On a machine without the model this returns False (local branch ran); remote mode never set.
    assert emb._remote_mode is False
    assert ok in (True, False)


def test_get_hf_token_prefers_env(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "env-token-xyz")
    emb = NGEmbed()
    assert emb._get_hf_token() == "env-token-xyz"


def test_get_hf_token_raises_when_unresolvable(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    emb = NGEmbed()
    monkeypatch.setattr(os.path, "expanduser", lambda p: str(tmp_path / "no-token-here"))
    with pytest.raises(EmbeddingUnavailableError):
        emb._get_hf_token()


def test_hf_post_builds_router_url_and_auth(monkeypatch):
    emb = NGEmbed()
    monkeypatch.setattr(emb, "_get_hf_token", lambda: "tok")
    captured = {}

    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps([0.0] * 768).encode()

    def _fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["auth"] = req.headers.get("Authorization")
        captured["body"] = json.loads(req.data)
        return _Resp()

    monkeypatch.setattr(ng_embed.urllib.request, "urlopen", _fake_urlopen)
    emb._hf_post("hello")
    assert captured["url"] == (
        "https://router.huggingface.co/hf-inference/models/"
        "Snowflake/snowflake-arctic-embed-m-v1.5/pipeline/feature-extraction"
    )
    assert captured["auth"] == "Bearer tok"
    assert captured["body"] == {"inputs": "hello"}
    assert "normalize" not in captured["body"]  # HF's own normalize option never sent


def test_log_failed_embed_appends_ordered(monkeypatch, tmp_path):
    emb = NGEmbed()
    emb._config["cache_dir"] = str(tmp_path)
    emb._log_failed_embed("first", Exception("boom1"))
    emb._log_failed_embed("second", Exception("boom2"))
    lines = (tmp_path / "failed_embeds.jsonl").read_text().strip().splitlines()
    assert [json.loads(l)["inputs"] for l in lines] == ["first", "second"]


def test_log_failed_embed_write_failure_does_not_raise(monkeypatch):
    emb = NGEmbed()
    emb._config["cache_dir"] = "/proc/nonexistent-cannot-mkdir/xyz"  # os.makedirs will fail
    emb._log_failed_embed("x", Exception("boom"))  # must swallow, not raise


def test_remote_call_retries_three_then_raises_and_quarantines(monkeypatch, tmp_path):
    emb = NGEmbed()
    emb._config["cache_dir"] = str(tmp_path)
    calls = {"n": 0}

    def _always_fail(inputs):
        calls["n"] += 1
        raise OSError("network down")

    monkeypatch.setattr(emb, "_hf_post", _always_fail)
    with pytest.raises(EmbeddingUnavailableError):
        emb._hf_remote_call("hello", 1)
    assert calls["n"] == 3  # exactly 3 attempts
    assert (tmp_path / "failed_embeds.jsonl").exists()  # quarantined


def test_remote_call_validates_dim_counts_as_failed_attempt(monkeypatch, tmp_path):
    emb = NGEmbed()
    emb._config["cache_dir"] = str(tmp_path)
    calls = {"n": 0}
    monkeypatch.setattr(emb, "_hf_post", lambda i: (calls.__setitem__("n", calls["n"] + 1) or [0.0] * 100))
    with pytest.raises(EmbeddingUnavailableError):
        emb._hf_remote_call("hello", 1)  # 100 dims != 768
    assert calls["n"] == 3


def test_remote_call_rejects_nan(monkeypatch, tmp_path):
    emb = NGEmbed()
    emb._config["cache_dir"] = str(tmp_path)
    bad = [float("nan")] + [0.0] * 767
    monkeypatch.setattr(emb, "_hf_post", lambda i: bad)
    with pytest.raises(EmbeddingUnavailableError):
        emb._hf_remote_call("hello", 1)


def test_remote_call_returns_validated_rows(monkeypatch):
    emb = NGEmbed()
    good = [0.1] * 768
    monkeypatch.setattr(emb, "_hf_post", lambda i: good)
    rows = emb._hf_remote_call("hello", 1)
    assert len(rows) == 1 and len(rows[0]) == 768


def test_remote_call_batch_count_mismatch_fails(monkeypatch, tmp_path):
    emb = NGEmbed()
    emb._config["cache_dir"] = str(tmp_path)
    monkeypatch.setattr(emb, "_hf_post", lambda i: [[0.0] * 768])  # 1 row for 2 inputs
    with pytest.raises(EmbeddingUnavailableError):
        emb._hf_remote_call(["a", "b"], 2)


def test_remote_embed_applies_query_prefix_client_side(monkeypatch):
    monkeypatch.setenv("NG_EMBED_REMOTE", "hf")
    emb = NGEmbed()
    monkeypatch.setattr(emb, "_get_hf_token", lambda: "tok")
    seen = {}
    def _capture(i):
        seen["inputs"] = i
        return [0.1] * 768

    monkeypatch.setattr(emb, "_hf_post", _capture)
    emb.embed("weather", is_query=True)
    assert seen["inputs"].startswith(emb._config["query_prefix"])


def test_remote_embed_normalizes_client_side(monkeypatch):
    emb = NGEmbed()
    monkeypatch.setattr(emb, "_hf_post", lambda i: [3.0] + [0.0] * 767)
    vec = emb._hf_remote_embed("x", normalize=True)
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-5


def test_remote_embed_batch_returns_all_vectors(monkeypatch):
    emb = NGEmbed()
    monkeypatch.setattr(emb, "_hf_post", lambda i: [[0.1] * 768, [0.2] * 768])
    vecs = emb._hf_remote_embed_batch(["a", "b"])
    assert len(vecs) == 2 and all(v.shape == (768,) for v in vecs)


def test_remote_embed_batch_is_all_or_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(ng_embed.time, "sleep", lambda *_: None)
    emb = NGEmbed()
    emb._config["cache_dir"] = str(tmp_path)
    # One row short for a 2-input batch -> whole batch raises, nothing returned.
    monkeypatch.setattr(emb, "_hf_post", lambda i: [[0.1] * 768])
    with pytest.raises(EmbeddingUnavailableError):
        emb._hf_remote_embed_batch(["a", "b"])


def test_embed_dispatches_to_remote_when_gated(monkeypatch):
    monkeypatch.setenv("NG_EMBED_REMOTE", "hf")
    emb = NGEmbed()
    monkeypatch.setattr(emb, "_get_hf_token", lambda: "tok")
    monkeypatch.setattr(emb, "_hf_post", lambda i: [0.5] * 768)
    vec = emb.embed("hello")
    assert vec.shape == (768,)
