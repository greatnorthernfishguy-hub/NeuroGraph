# ---- Changelog ----
# 2026-03-25 Claude Code — Initial creation
# What: Tests for parameterized function library
# -------------------

"""Tests for lenia.functions — kernel shapes and growth functions."""

import numpy as np
import pytest

from lenia.functions import evaluate, validate_spec


class TestFunctions:
    def test_gaussian_peak(self):
        spec = {"type": "gaussian", "params": {"center": 0.0, "sigma": 1.0, "amplitude": 1.0}}
        assert abs(evaluate(spec, 0.0) - 1.0) < 1e-10

    def test_gaussian_decay(self):
        spec = {"type": "gaussian", "params": {"center": 0.0, "sigma": 0.3, "amplitude": 1.0}}
        at_center = evaluate(spec, 0.0)
        at_edge = evaluate(spec, 1.0)
        assert at_center > at_edge

    def test_gaussian_vectorized(self):
        spec = {"type": "gaussian", "params": {"center": 0.0, "sigma": 1.0, "amplitude": 2.0}}
        x = np.array([0.0, 0.5, 1.0, 2.0])
        result = evaluate(spec, x)
        assert result.shape == (4,)
        assert abs(result[0] - 2.0) < 1e-10

    def test_bell_center(self):
        spec = {"type": "bell", "params": {"center": 0.5, "sigma": 0.15}}
        # At center: 2*exp(0) - 1 = 1.0
        assert abs(evaluate(spec, 0.5) - 1.0) < 1e-10

    def test_bell_negative_far(self):
        spec = {"type": "bell", "params": {"center": 0.5, "sigma": 0.15}}
        # Far from center should be negative
        assert evaluate(spec, 5.0) < 0.0

    def test_sigmoid_midpoint(self):
        spec = {"type": "sigmoid", "params": {"midpoint": 0.5, "steepness": 10.0, "amplitude": 1.0}}
        assert abs(evaluate(spec, 0.5) - 0.5) < 1e-10

    def test_step(self):
        spec = {"type": "step", "params": {"low": 0.3, "high": 0.7, "amplitude": 2.0}}
        assert abs(evaluate(spec, 0.5) - 2.0) < 1e-10
        assert abs(evaluate(spec, 0.1)) < 1e-10
        assert abs(evaluate(spec, 0.9)) < 1e-10

    def test_polynomial(self):
        spec = {"type": "polynomial", "params": {"coeffs": [0.0, 0.0, 1.0], "max_val": 10.0}}
        # x^2
        assert abs(evaluate(spec, 3.0) - 9.0) < 1e-10

    def test_composite(self):
        spec = {
            "type": "composite",
            "params": {
                "components": [
                    {"spec": {"type": "gaussian", "params": {"center": 0.0, "sigma": 1.0, "amplitude": 1.0}}, "weight": 0.5},
                    {"spec": {"type": "step", "params": {"low": -1.0, "high": 1.0, "amplitude": 1.0}}, "weight": 0.5},
                ]
            },
        }
        result = evaluate(spec, 0.0)
        # gaussian(0)=1.0 * 0.5 + step(0)=1.0 * 0.5 = 1.0
        assert abs(result - 1.0) < 1e-10

    def test_validate_spec_valid(self):
        assert validate_spec({"type": "gaussian", "params": {"center": 0.0}})
        assert validate_spec({"type": "bell", "params": {}})

    def test_validate_spec_invalid(self):
        assert not validate_spec({"type": "unknown", "params": {}})
        assert not validate_spec({"params": {}})
        assert not validate_spec("not a dict")

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError):
            evaluate({"type": "nonexistent", "params": {}}, 0.0)
