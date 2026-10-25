# ---- Changelog ----
# [2026-10-15] Claude Code (Opus 5.5, Executive) — #1115: the same text embeds to the same vector in any batch
# What: pins batch invariance of ng_embed's ONNX path (alone, with companions, with different padding).
# Why:  the int8 dynamically-quantized model made a text's vector depend on its batch-mates (0.977-0.992 cosine).
# How:  real model from the local HF cache; skipped when it is not available (no download in tests).
# -------------------
import numpy as np
import pytest

import ng_embed


@pytest.fixture(scope="module")
def embedder():
    try:
        e = ng_embed.NGEmbed() if hasattr(ng_embed, "NGEmbed") else ng_embed.get_embedder()
        e.embed("warm up")
    except Exception as exc:  # model not cached / onnxruntime missing
        pytest.skip(f"embedding model unavailable: {exc}")
    if getattr(e, "_session", None) is None:
        pytest.skip("ONNX session not loaded")
    return e


TEXT = "Sylphrena has the right to leave the ecosystem."
COMPANIONS = [
    [],
    ["a much longer companion sentence about routing models through providers, so the padding length changes"],
    ["x", "another short one"],
    [f"filler sentence {i} about memory, the River and the Tonic" for i in range(12)],
]


def test_same_text_same_vector_in_any_batch(embedder):
    alone = np.asarray(embedder._onnx_embed_batch([TEXT], normalize=True)[0])
    for comp in COMPANIONS:
        batch = comp[:1] + [TEXT] + comp[1:]
        vec = np.asarray(embedder._onnx_embed_batch(batch, normalize=True)[batch.index(TEXT)])
        assert np.allclose(vec, alone, atol=1e-6), f"batch-mates {len(comp)} changed the vector"


def test_every_batch_member_equals_its_single_embedding(embedder):
    texts = [TEXT] + COMPANIONS[3][:6]
    batched = embedder._onnx_embed_batch(texts, normalize=True)
    for t, v in zip(texts, batched):
        single = embedder._onnx_embed_batch([t], normalize=True)[0]
        assert np.allclose(np.asarray(v), np.asarray(single), atol=1e-6), t
