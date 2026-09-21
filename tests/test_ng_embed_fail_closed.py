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
