# tests/test_ng_embed_dualpass.py
import os
import sys
import inspect
import threading
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ng_embed import NGEmbed, EmbeddingUnavailableError, embed, embed_batch
import ng_embed as ng_embed_mod


@pytest.fixture(autouse=True)
def _reset_singleton_and_env(monkeypatch):
    monkeypatch.delenv("NG_EMBED_REMOTE", raising=False)
    monkeypatch.delenv("NG_EMBED_ALLOW_HASH_FALLBACK", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    NGEmbed.reset_instance()
    yield
    NGEmbed.reset_instance()


def test_hash_embed_symbol_does_not_exist():
    assert not hasattr(NGEmbed, "_hash_embed")
    src = inspect.getsource(ng_embed_mod)
    assert "_hash_embed" not in src
    assert "sha384" not in src.lower()
    assert "NG_EMBED_ALLOW_HASH_FALLBACK" not in src


def test_embed_raises_when_model_unavailable(monkeypatch):
    emb = NGEmbed()
    monkeypatch.setattr(emb, "_ensure_model", lambda: False)
    with pytest.raises(EmbeddingUnavailableError):
        emb.embed("hello")


def test_embed_batch_raises_when_model_unavailable(monkeypatch):
    emb = NGEmbed()
    monkeypatch.setattr(emb, "_ensure_model", lambda: False)
    with pytest.raises(EmbeddingUnavailableError):
        emb.embed_batch(["a", "b"])


def test_empty_batch_returns_empty_without_touching_model(monkeypatch):
    emb = NGEmbed()
    monkeypatch.setattr(emb, "_ensure_model", lambda: (_ for _ in ()).throw(RuntimeError("should not load")))
    assert emb.embed_batch([]) == []


class _Enc:
    def __init__(self, ids):
        self.ids = list(ids)
        self.attention_mask = [1] * len(self.ids)


class _FakeTok:
    """Deterministic tokenizer: one token per character. No truncation method."""
    def encode(self, text):
        return _Enc(list(range(len(text))))
    def decode(self, ids, skip_special_tokens=True):
        # Tests identify windows by token_count / embedding, not decode fidelity.
        return "x" * len(list(ids))
    def encode_batch(self, texts):
        return [self.encode(t) for t in texts]
    def enable_padding(self, **k):
        return None


def _one_hot(dim, idx):
    v = np.zeros(dim, dtype=np.float32)
    v[idx % dim] = 1.0
    return v


def test_tokenizer_truncation_is_not_enabled_on_loaded_instance(monkeypatch):
    emb = NGEmbed()
    called = {"trunc": False}
    tok = _FakeTok()
    def _boom(**k):
        called["trunc"] = True
        raise AssertionError("enable_truncation must not be called")
    tok.enable_truncation = _boom  # type: ignore
    # Force the load path to install our tokenizer. Easier: set attrs after fake load.
    monkeypatch.setattr(emb, "_ensure_model", lambda: True)
    emb._tokenizer = tok
    emb._model_loaded = True
    # The production _ensure_model body must not call enable_truncation.
    src = inspect.getsource(NGEmbed._ensure_model)
    assert "enable_truncation" not in src


def test_short_input_matches_single_window_primitive(monkeypatch):
    emb = NGEmbed()
    monkeypatch.setattr(emb, "_ensure_model", lambda: True)
    emb._tokenizer = _FakeTok()
    emb._model_loaded = True
    sentinel = np.arange(768, dtype=np.float32) / 768.0
    monkeypatch.setattr(emb, "_onnx_embed", lambda text, normalize=False, is_query=False: sentinel.copy())
    out = emb.embed("short")  # 5 tokens under 512
    assert out.shape == (768,)
    assert np.array_equal(out, sentinel)


def test_long_input_pooled_differs_from_first_window_and_is_unit(monkeypatch):
    emb = NGEmbed()
    monkeypatch.setattr(emb, "_ensure_model", lambda: True)
    emb._tokenizer = _FakeTok()
    emb._model_loaded = True
    dim = 768
    def fake_onnx(text, normalize=False, is_query=False, _ids=None):
        n = len(text)
        return _one_hot(dim, n)
    monkeypatch.setattr(emb, "_onnx_embed", fake_onnx)
    # 600-char input → 600 tokens → windows 512 and 152 (start 448)
    text = "a" * 600
    we = emb.embed_windows(text)
    assert we.token_count == 600
    assert len(we.windows) >= 2
    pooled = we.pooled
    first = we.windows[0].embedding
    assert pooled.shape == (768,)
    assert not np.allclose(pooled, first), "pooling must not equal the first window (truncation)"
    assert abs(float(np.linalg.norm(pooled)) - 1.0) < 1e-5
    # Tail window influences the pool: a 512-token prefix must differ.
    prefix = emb.embed_windows("a" * 512).pooled
    assert not np.allclose(pooled, prefix), "tokens past 512 must influence the forest vector"


def test_embed_windows_empty_on_short_input(monkeypatch):
    emb = NGEmbed()
    monkeypatch.setattr(emb, "_ensure_model", lambda: True)
    emb._tokenizer = _FakeTok()
    emb._model_loaded = True
    monkeypatch.setattr(emb, "_onnx_embed", lambda *a, **k: np.ones(768, dtype=np.float32))
    we = emb.embed_windows("hello")
    assert we.windows == ()
