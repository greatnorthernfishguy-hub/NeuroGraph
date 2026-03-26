# ---- Changelog ----
# 2026-03-25 Claude Code — Initial creation
# What: Parameterized function library for kernels and growth functions
# Why: Both kernel shapes and growth functions are stored as {type, params} specs
# How: Single evaluate() dispatches by type. Supports vectorized evaluation.
# PRD: ~/docs/prd/Lenia_FlowGraph_Design_v0.1.md §3
# -------------------

"""Parameterized function library.

Kernel shapes and growth functions are stored as FunctionSpec dicts:
    {"type": "gaussian", "params": {"center": 0.0, "sigma": 0.3, "amplitude": 1.0}}

evaluate() dispatches by type. All functions accept and return numpy arrays
for vectorized evaluation (no Python loops over entities needed).
"""

from typing import Any, Dict, List, Union

import numpy as np

# Type alias for function specifications
FunctionSpec = Dict[str, Any]


def evaluate(spec: FunctionSpec, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Evaluate a parameterized function at x.

    Args:
        spec: {"type": str, "params": dict} function specification.
        x: Input value(s). Scalar or numpy array.

    Returns:
        Function output. Same shape as x.
    """
    fn_type = spec["type"]
    params = spec["params"]

    if fn_type == "gaussian":
        return _gaussian(x, **params)
    elif fn_type == "bell":
        return _bell(x, **params)
    elif fn_type == "sigmoid":
        return _sigmoid(x, **params)
    elif fn_type == "step":
        return _step(x, **params)
    elif fn_type == "polynomial":
        return _polynomial(x, **params)
    elif fn_type == "composite":
        return _composite(x, **params)
    else:
        raise ValueError(f"Unknown function type: {fn_type}")


def _gaussian(
    x: Union[float, np.ndarray],
    center: float = 0.0,
    sigma: float = 1.0,
    amplitude: float = 1.0,
) -> Union[float, np.ndarray]:
    """Standard Gaussian: amplitude * exp(-((x - center)^2) / (2 * sigma^2))"""
    return amplitude * np.exp(-((x - center) ** 2) / (2.0 * sigma ** 2))


def _bell(
    x: Union[float, np.ndarray],
    center: float = 0.5,
    sigma: float = 0.15,
) -> Union[float, np.ndarray]:
    """Lenia bell curve: 2 * exp(-((x - center)^2) / (2 * sigma^2)) - 1

    Returns values in [-1, 1]. Positive near center, negative far from it.
    This is the classic Lenia growth function shape.
    """
    return 2.0 * np.exp(-((x - center) ** 2) / (2.0 * sigma ** 2)) - 1.0


def _sigmoid(
    x: Union[float, np.ndarray],
    midpoint: float = 0.5,
    steepness: float = 10.0,
    amplitude: float = 1.0,
) -> Union[float, np.ndarray]:
    """Sigmoid: amplitude / (1 + exp(-steepness * (x - midpoint)))"""
    return amplitude / (1.0 + np.exp(-steepness * (x - midpoint)))


def _step(
    x: Union[float, np.ndarray],
    low: float = 0.0,
    high: float = 1.0,
    amplitude: float = 1.0,
) -> Union[float, np.ndarray]:
    """Step function: amplitude if low <= x <= high, else 0."""
    x_arr = np.asarray(x)
    result = np.where((x_arr >= low) & (x_arr <= high), amplitude, 0.0)
    return float(result) if np.ndim(x) == 0 else result


def _polynomial(
    x: Union[float, np.ndarray],
    coeffs: List[float] = None,
    max_val: float = 1.0,
) -> Union[float, np.ndarray]:
    """Polynomial: sum(coeffs[i] * x^i), clamped to [0, max_val]."""
    if coeffs is None:
        coeffs = [0.0, 1.0]
    x_arr = np.asarray(x, dtype=np.float64)
    result = np.zeros_like(x_arr)
    for i, c in enumerate(coeffs):
        result = result + c * (x_arr ** i)
    return np.clip(result, 0.0, max_val)


def _composite(
    x: Union[float, np.ndarray],
    components: List[Dict[str, Any]] = None,
) -> Union[float, np.ndarray]:
    """Weighted sum of other functions.

    components: [{"spec": FunctionSpec, "weight": float}, ...]
    """
    if components is None:
        return np.zeros_like(np.asarray(x, dtype=np.float64))
    result = np.zeros_like(np.asarray(x, dtype=np.float64))
    for comp in components:
        result = result + comp["weight"] * evaluate(comp["spec"], x)
    return result


def validate_spec(spec: FunctionSpec) -> bool:
    """Check that a function spec is well-formed."""
    if not isinstance(spec, dict):
        return False
    if "type" not in spec or "params" not in spec:
        return False
    valid_types = {"gaussian", "bell", "sigmoid", "step", "polynomial", "composite"}
    return spec["type"] in valid_types
