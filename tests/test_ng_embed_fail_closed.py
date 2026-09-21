# tests/test_ng_embed_fail_closed.py
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
