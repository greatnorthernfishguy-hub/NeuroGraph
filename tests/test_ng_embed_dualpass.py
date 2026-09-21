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
