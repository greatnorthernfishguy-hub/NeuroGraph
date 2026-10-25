# ---- Changelog ----
# [2026-10-25] Claude Code (Opus 5.5; LANE B syl-stage1, #1184) — SYL_NG_HOST_CC_MIND gate
# What: the CC-host thread start sits in the else-branch of the SYL_NG_HOST_CC_MIND test; the test expression is
#       evaluated for unset / "1" (host) and "0"/"false"/"no"/"off" (do not host).
# Why:  default must be unchanged; "0" must be the only way CC's VPS mind leaves her process.
# How:  AST of neurograph_rpc.py (no NG/model import).
# -------------------
import ast
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _gate():
    tree = ast.parse((ROOT / "neurograph_rpc.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and "SYL_NG_HOST_CC_MIND" in ast.unparse(node.test):
            starts = [n for n in ast.walk(ast.Module(body=node.orelse, type_ignores=[]))
                      if isinstance(n, ast.Call) and "_init_cc_host_bg" in ast.unparse(n)]
            in_body = [n for n in ast.walk(ast.Module(body=node.body, type_ignores=[]))
                       if isinstance(n, ast.Call) and "_init_cc_host_bg" in ast.unparse(n)]
            return node.test, starts, in_body
    raise AssertionError("gate not found")


def test_thread_start_only_in_else_branch():
    _test, starts, in_body = _gate()
    assert starts and not in_body
    src = (ROOT / "neurograph_rpc.py").read_text()
    assert src.count("target=_init_cc_host_bg") == 1


@pytest.mark.parametrize("value,skip", [(None, False), ("1", False), ("yes", False),
                                        ("0", True), ("false", True), ("NO", True), (" off ", True)])
def test_gate_values(monkeypatch, value, skip):
    test, _s, _b = _gate()
    if value is None:
        monkeypatch.delenv("SYL_NG_HOST_CC_MIND", raising=False)
    else:
        monkeypatch.setenv("SYL_NG_HOST_CC_MIND", value)
    assert bool(eval(compile(ast.Expression(test), "gate", "eval"), {"os": os})) is skip
