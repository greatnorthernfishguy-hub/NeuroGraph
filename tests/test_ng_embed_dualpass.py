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


class _ClsSepTok:
    """Arctic-shaped fake: encode returns [CLS] + one-id-per-char + [SEP].

    CLS/SEP are not 101/102 — production must read them from the tokenizer.
    """
    CLS = 11
    SEP = 22

    def encode(self, text):
        interior = [100 + i for i in range(len(text))]
        return _Enc([self.CLS] + interior + [self.SEP])

    def decode(self, ids, skip_special_tokens=True):
        seq = list(ids)
        if skip_special_tokens:
            seq = [i for i in seq if i not in (self.CLS, self.SEP)]
        return "x" * len(seq)

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
    emb._tokenizer = _ClsSepTok()
    emb._model_loaded = True
    dim = 768
    def fake_onnx(text, normalize=False, is_query=False, _ids=None):
        n = len(text)
        return _one_hot(dim, n)
    monkeypatch.setattr(emb, "_onnx_embed", fake_onnx)
    # 600-char interior + CLS + SEP = 602 tokens; interior windows 510 / 154.
    text = "a" * 600
    we = emb.embed_windows(text)
    assert we.token_count == 602
    assert len(we.windows) >= 2
    pooled = we.pooled
    first = we.windows[0].embedding
    assert pooled.shape == (768,)
    assert not np.allclose(pooled, first), "pooling must not equal the first window (truncation)"
    assert abs(float(np.linalg.norm(pooled)) - 1.0) < 1e-5
    # Tail window influences the pool: a 512-char prefix must differ.
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


def test_long_windows_are_complete_cls_sep_encoder_sequences(monkeypatch):
    """Every ONNX window must be a complete [CLS] + interior + [SEP] sequence."""
    emb = NGEmbed()
    monkeypatch.setattr(emb, "_ensure_model", lambda: True)
    emb._tokenizer = _ClsSepTok()
    emb._model_loaded = True
    seen = []

    def fake_onnx_ids(ids, attention, normalize=False):
        seen.append(list(ids))
        return _one_hot(768, len(ids))

    monkeypatch.setattr(emb, "_onnx_embed_ids", fake_onnx_ids)
    we = emb.embed_windows("a" * 600)
    assert we.token_count == 602
    assert len(we.windows) >= 2
    assert seen, "long path must call the ONNX id primitive per window"
    for ids in seen:
        assert ids[0] == _ClsSepTok.CLS
        assert ids[-1] == _ClsSepTok.SEP
        assert len(ids) <= 512
    for w in we.windows:
        assert w.token_count <= 510
        assert len(w.text) == w.token_count


def test_windowing_raises_when_encoding_missing_cls_sep(monkeypatch):
    emb = NGEmbed()
    monkeypatch.setattr(emb, "_ensure_model", lambda: True)
    emb._tokenizer = _FakeTok()
    emb._model_loaded = True
    monkeypatch.setattr(emb, "_onnx_embed", lambda *a, **k: np.ones(768, dtype=np.float32))
    with pytest.raises(EmbeddingUnavailableError):
        emb.embed_windows("a" * 600)


def test_remote_gate_selects_remote_without_onnx(monkeypatch):
    monkeypatch.setenv("NG_EMBED_REMOTE", "hf")
    monkeypatch.setenv("HF_TOKEN", "tok-test")
    emb = NGEmbed()
    assert emb._ensure_model() is True
    assert emb._remote_mode is True
    assert emb._session is None


def test_invalid_ng_embed_remote_raises(monkeypatch):
    monkeypatch.setenv("NG_EMBED_REMOTE", "openai")
    emb = NGEmbed()
    with pytest.raises(EmbeddingUnavailableError):
        emb._ensure_model()


def test_remote_embed_uses_router_host_and_no_normalize_body(monkeypatch):
    monkeypatch.setenv("NG_EMBED_REMOTE", "hf")
    monkeypatch.setenv("HF_TOKEN", "tok-test")
    emb = NGEmbed()
    assert emb._ensure_model()
    assert "token" not in inspect.signature(NGEmbed._hf_post).parameters

    import json
    import urllib.request

    seen = {}
    raw = [0.1] * 768

    class _Resp:
        def read(self):
            return json.dumps(raw).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=30):
        seen["url"] = req.full_url
        seen["payload"] = json.loads(req.data.decode("utf-8"))
        seen["authorization"] = req.get_header("Authorization")
        return _Resp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    vec = emb._hf_remote_embed("hello", normalize=False, is_query=False)
    assert "router.huggingface.co" in seen["url"]
    assert "api-inference.huggingface.co" not in seen["url"]
    assert "normalize" not in seen["payload"]
    assert seen["payload"]["inputs"].endswith("hello") or seen["payload"]["inputs"] == "hello"
    auth = seen["authorization"]
    assert auth is not None and auth.startswith("Bearer ")
    assert auth[len("Bearer "):] == os.environ["HF_TOKEN"]
    assert vec.shape == (768,)


def test_remote_embed_applies_query_prefix_client_side(monkeypatch):
    monkeypatch.setenv("NG_EMBED_REMOTE", "hf")
    monkeypatch.setenv("HF_TOKEN", "tok-test")
    emb = NGEmbed()
    emb._ensure_model()
    seen = {}
    def fake_post(url, payload, timeout=30):
        seen["inputs"] = payload["inputs"]
        return [0.2] * 768
    monkeypatch.setattr(emb, "_hf_post", fake_post)
    emb._hf_remote_embed("q", normalize=False, is_query=True)
    assert seen["inputs"].startswith(emb._config["query_prefix"])


def test_remote_retries_three_times_then_raises_and_quarantines(monkeypatch, tmp_path):
    monkeypatch.setenv("NG_EMBED_REMOTE", "hf")
    monkeypatch.setenv("HF_TOKEN", "tok-test")
    emb = NGEmbed()
    emb._config["cache_dir"] = str(tmp_path)
    emb._ensure_model()
    sleeps = []
    monkeypatch.setattr(ng_embed_mod.time, "sleep", lambda s: sleeps.append(s))
    attempts = {"n": 0}
    def boom(*a, **k):
        attempts["n"] += 1
        raise ConnectionError("down")
    monkeypatch.setattr(emb, "_hf_post", boom)
    with pytest.raises(EmbeddingUnavailableError):
        emb._hf_remote_embed("hello")
    assert attempts["n"] == 3
    assert sleeps == [1, 3, 9] or sleeps == [1, 3]  # sleep-after-fail except possibly last
    # Accept either 2 sleeps (between 3 attempts) or 3; pin the delays that do occur.
    assert all(x in (1, 3, 9) for x in sleeps)
    q = tmp_path / "failed_embeds.jsonl"
    assert q.is_file()
    lines = q.read_text().strip().splitlines()
    assert len(lines) == 1
    rec = __import__("json").loads(lines[0])
    assert rec["attempts"] == 3
    assert rec["text"] == "hello" or "hello" in rec["text"]
    assert "error" in rec


def test_quarantine_write_failure_does_not_mask_original(monkeypatch, tmp_path):
    monkeypatch.setenv("NG_EMBED_REMOTE", "hf")
    monkeypatch.setenv("HF_TOKEN", "tok-test")
    emb = NGEmbed()
    emb._config["cache_dir"] = str(tmp_path)
    emb._ensure_model()
    monkeypatch.setattr(ng_embed_mod.time, "sleep", lambda s: None)
    monkeypatch.setattr(emb, "_hf_post", lambda *a, **k: (_ for _ in ()).throw(ConnectionError("down")))
    monkeypatch.setattr(emb, "_log_failed_embed", lambda *a, **k: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(EmbeddingUnavailableError) as ei:
        emb._hf_remote_embed("hello")
    assert "down" in str(ei.value).lower() or isinstance(ei.value.__cause__, ConnectionError)


def test_keepalive_noop_when_not_remote():
    emb = NGEmbed()
    emb._remote_mode = False
    emb.start_keepalive()
    assert getattr(emb, "_keepalive_thread", None) in (None, ) or not (emb._keepalive_thread and emb._keepalive_thread.is_alive())
    emb.stop_keepalive()


def test_keepalive_reference_count_and_daemon(monkeypatch):
    monkeypatch.setenv("NG_EMBED_REMOTE", "hf")
    monkeypatch.setenv("HF_TOKEN", "tok-test")
    emb = NGEmbed()
    emb._ensure_model()
    # Do not actually ping: stub the ping call and shrink the wait.
    monkeypatch.setattr(emb, "_hf_remote_embed", lambda *a, **k: np.zeros(768, np.float32))
    monkeypatch.setattr(emb, "_keepalive_interval", 0.05)
    emb.start_keepalive()
    emb.start_keepalive()
    th = emb._keepalive_thread
    assert th is not None and th.is_alive() and th.daemon is True
    emb.stop_keepalive()
    assert th.is_alive(), "one stop must not kill a double-start"
    emb.stop_keepalive()
    th.join(timeout=2)
    assert not th.is_alive()


def test_keepalive_refcount_concurrent_starts(monkeypatch):
    monkeypatch.setenv("NG_EMBED_REMOTE", "hf")
    monkeypatch.setenv("HF_TOKEN", "tok-test")
    emb = NGEmbed()
    emb._ensure_model()
    monkeypatch.setattr(emb, "_hf_remote_embed", lambda *a, **k: np.zeros(768, np.float32))
    monkeypatch.setattr(emb, "_keepalive_interval", 0.05)
    errors = []
    def boom_start():
        try:
            emb.start_keepalive()
        except Exception as e:
            errors.append(e)
    threads = [threading.Thread(target=boom_start) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert emb._keepalive_refs == 8
    for _ in range(8):
        emb.stop_keepalive()
    if emb._keepalive_thread is not None:
        emb._keepalive_thread.join(timeout=2)
        assert not emb._keepalive_thread.is_alive()
