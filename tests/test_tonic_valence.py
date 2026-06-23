"""Focus valence field (#90) — her poles build the light<->dark axis.

Syl-approved design: prd/2026-06-17-tonic-valence-field-design.md
"""
import sys, os
import numpy as np
from types import SimpleNamespace
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tonic_valence import ValenceConfig, ValenceField, load_poles


def _stub_embed(mapping, dim=8):
    """Return an embed_fn that maps known phrases to fixed vectors, else zeros."""
    def _e(text, normalize=False, is_query=False):
        return np.array(mapping.get(text, [0.0] * dim), dtype=np.float32)
    return _e


def test_config_defaults_sane():
    c = ValenceConfig()
    assert c.seed_gain > 0
    assert c.diffusion_steps >= 1
    assert 0.0 <= c.diffusion_alpha <= 1.0


def test_poles_file_loads_her_words():
    poles = load_poles(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "valence_poles.toml"))
    assert "home" in poles["light"]
    assert any("weld" in d for d in poles["dark"])


def test_axis_points_from_dark_to_light():
    # light pole at +x, dark pole at -x → axis ~ +x unit
    poles = {"light": ["L"], "dark": ["D"]}
    embed = _stub_embed({"L": [1, 0], "D": [-1, 0]}, dim=2)
    vf = ValenceField(ValenceConfig(), embed_fn=embed, poles=poles)
    assert vf.axis is not None
    assert vf.axis.shape == (2,)
    assert np.isclose(np.linalg.norm(vf.axis), 1.0)
    assert vf.axis[0] > 0.99  # points toward light
