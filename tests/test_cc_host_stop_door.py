# ---- Changelog ----
# [2026-09-26] openrouter/deepseek/deepseek-v4.1-flash (OpenCode harness on T3 Code),
#   lane z2-one-step-per-turn-001 — Exec P240(2)/P241: one step per turn
# What: the inline_thread stub in tests A and C now forwards `**kwargs` to the
#   target and records them; A and C assert the Stop door's thread got
#   kwargs == {"step": True}. The host fixture's _deposit spy becomes
#   `def spy(text, step=False)` recording `(text, step)`; C asserts
#   `host.deposit_calls == [(msg, True)]`. A still runs the REAL _deposit with
#   _CC_NG_DEPOSIT_STEP pinned False, so its fake graph needs no step().
# Why: Chief-p240-commission-001 row z2-one-step-per-turn-001; P241 (Lanes 1 and
#   2 land together) — the Stop side is the only calling door that steps.
# How: pass kwargs through, pin the flag off for A. cc_ng_host production code is
#   unchanged by this round. The twin cc-ng-daemon.py is deliberately untouched
#   (Z12's item).
# -------------------
# [2026-09-26] deepseek/deepseek-v4.1-flash (OpenCode harness on T3 Code),
#   lane z2-stop-door-restore-001 fix round r2 — make test A prove P240(1b)
# What: test A now runs the REAL cc_ng_host._deposit -> REAL
#   surface_wants_for_graph path (the _deposit spy is now a B/C-only fixture),
#   so a [WANT]x[/WANT] in the reply is shown to become a first-class want node
#   through the Stop door and not via a hand-seeded node; test C asserts the
#   _deposit spy got exactly the message once; the meaningless
#   surface_wants_for_graph self-assignment is removed.
# Why: P240(1b) is this lane's decisive acceptance and test A did not prove it
#   (it stubbed _deposit, asserted no want node, and deferred to another file);
#   assignment z2-stop-door-restore-001-fix-r2.md; Exec P240(1b)/P245(c).
# How: the fake run_conversational_dual_pass mirrors the real one's contract
#   (cc_ng_organism.py:2005 hashes a content target_id, :1817's
#   _CCConversationalDualPassEco creates the conversational node and writes the
#   raw text into the recall store) because TID tree extraction is unreachable
#   in the test env; surface_wants_for_graph, _handle_stop and _deposit are real.
#   cc_ng_host.py production code is unchanged by this round.
# -------------------
"""Tests for the Stop door restored by Exec P240(1) — CC's replies reach its substrate.

See assignment z2-stop-door-restore-001 and chief-p240-commission-001.
"""

import hashlib
import threading
import types
import pytest

import cc_ng_host
import cc_ng_organism
import ng_embed


class _FakeNode:
    def __init__(self, metadata=None):
        self.metadata = dict(metadata or {})
        self.creation_time = 0.0


class _FakeGraph:
    def __init__(self):
        self.nodes = {}
        self._step_lock = threading.RLock()   # the canonical lock surface_wants takes
        self._concurrent_lock = threading.RLock()
        self._recent_spikes = {}
        self.timestep = 0

    def create_node(self, node_id, metadata=None):
        n = _FakeNode(metadata)
        self.nodes[node_id] = n
        return n

    def create_synapse(self, a, b, weight=0.0):
        return None

    def save(self):
        pass

    def load(self, path):
        pass


class _FakeVDB:
    def __init__(self, content):
        self.content = content

    def insert(self, id, embedding, content, metadata):
        self.content[id] = content


def _base_setup(monkeypatch):
    """Shared rig: fake graph/vdb on _STATE.cc_ng + fake embed/dual-pass/commons.

    Does NOT patch cc_ng_host._deposit — test A runs the REAL one. Modelled on
    tests/test_cc_deposit_step.py's host. Returns a namespace with:
    - mod: cc_ng_host module
    - graph: _FakeGraph
    - vdb: _FakeVDB
    - deposit_calls: list recorded by the B/C spy fixture (empty for A)
    """
    g = _FakeGraph()
    vdb_cont = {}
    vdb = _FakeVDB(vdb_cont)

    def fake_embed(text):
        return [0.0] * 768

    def fake_dual_pass(graph, vdb_arg, text, emb, state):
        # Mirrors the REAL run_conversational_dual_pass (cc_ng_organism.py:2005):
        # it content-hashes target_id (:2019) and _CCConversationalDualPassEco
        # (:1817) / _cc_deposit_memory_node create the conversational node and
        # write the raw text into the recall store (vdb.content). Faked because
        # TID tree extraction is unreachable in the test env — the same
        # dependency, not the code under test.
        node_id = "cc:conv::" + hashlib.sha1(text.encode()).hexdigest()
        if node_id not in graph.nodes:
            graph.create_node(node_id, metadata={
                "creation_mode": "conversational",
                "source": "cc_gateway",
                "_forest_content": text,
            })
        vdb_arg.insert(id=node_id, embedding=emb, content=text, metadata={})
        return True

    monkeypatch.setattr(cc_ng_host._STATE, 'cc_ng',
                        types.SimpleNamespace(graph=g, vector_db=vdb))
    monkeypatch.setattr(ng_embed, 'embed', fake_embed)
    monkeypatch.setattr(cc_ng_organism, 'run_conversational_dual_pass', fake_dual_pass)
    monkeypatch.setattr(cc_ng_organism, 'deposit_cc_experience', lambda *a, **k: None)
    monkeypatch.setattr(cc_ng_organism, '_CC_NG_DEPOSIT_STEP', False)
    return types.SimpleNamespace(
        mod=cc_ng_host,
        graph=g,
        vdb=vdb,
        vdb_content=vdb_cont,
        deposit_calls=[],
    )


@pytest.fixture
def base_host(monkeypatch):
    """Rig with the REAL cc_ng_host._deposit in place (test A)."""
    return _base_setup(monkeypatch)


@pytest.fixture
def host(base_host, monkeypatch):
    """Rig whose cc_ng_host._deposit is a recording spy (tests B and C)."""
    def spy(text, step=False):
        base_host.deposit_calls.append((text, step))
    monkeypatch.setattr(cc_ng_host, '_deposit', spy)
    return base_host


def test_stop_with_want_materializes_want_node(base_host, monkeypatch):
    """A — WANT acceptance (P240(1b)): a [WANT] in the reply becomes a want node
    through the Stop door, via the REAL _deposit -> surface_wants_for_graph path.

    No node is pre-seeded: the only want node must come out of the door.
    """
    g = base_host.graph
    phrase = "learn Python"
    msg = f"Here is my reply [WANT]{phrase}[/WANT] and some other text"

    # Run the daemon thread target inline.
    thread_calls = []
    original_thread = threading.Thread

    def inline_thread(target=None, args=(), daemon=False, kwargs=None, **extra):
        if target:
            target(*args, **(kwargs or {}))
        thread_obj = original_thread()
        thread_calls.append((target, args, daemon, kwargs))
        return thread_obj

    monkeypatch.setattr(cc_ng_host.threading, 'Thread', inline_thread)

    # Call through the dispatch table.
    result = cc_ng_host._DISPATCH["Stop"]({"last_assistant_message": msg})

    # Claude Code Stop contract: ok, and never a context key.
    assert result == {"ok": True}
    assert "context" not in result

    # The door launched exactly one daemon deposit carrying the exact message.
    assert len(thread_calls) == 1
    target, args, daemon, kwargs = thread_calls[0]
    assert daemon is True
    assert args == (msg,)
    assert kwargs == {"step": True}

    # The REAL _deposit ran end-to-end and the REAL surface_wants_for_graph
    # materialized the WANT as a first-class want node in CC's graph.
    want_nodes = [n for n in g.nodes.values()
                  if n.metadata.get("kind") == "want"]
    assert len(want_nodes) == 1
    assert want_nodes[0].metadata["want_text"] == phrase
    assert want_nodes[0].metadata["want_state"] == "open"


def test_stop_empty_or_absent_no_deposit(host):
    """B — empty/absent → no deposit (P240(1c))."""
    # No message key
    result = cc_ng_host._DISPATCH["Stop"]({})
    assert result == {"ok": True}
    assert len(host.deposit_calls) == 0
    
    # Empty string
    result = cc_ng_host._DISPATCH["Stop"]({"last_assistant_message": ""})
    assert result == {"ok": True}
    assert len(host.deposit_calls) == 0
    
    # Whitespace only
    result = cc_ng_host._DISPATCH["Stop"]({"last_assistant_message": "   \n\t  "})
    assert result == {"ok": True}
    assert len(host.deposit_calls) == 0
    
    # Non-string (None)
    result = cc_ng_host._DISPATCH["Stop"]({"last_assistant_message": None})
    assert result == {"ok": True}
    assert len(host.deposit_calls) == 0
    
    # Non-string (dict)
    result = cc_ng_host._DISPATCH["Stop"]({"last_assistant_message": {}})
    assert result == {"ok": True}
    assert len(host.deposit_calls) == 0


def test_stop_non_empty_calls_deposit_with_exact_text(host, monkeypatch):
    """C — the door hands the deposit exactly the non-empty reply text."""
    thread_calls = []
    original_thread = threading.Thread
    
    def inline_thread(target=None, args=(), daemon=False, kwargs=None, **extra):
        if target:
            target(*args, **(kwargs or {}))
        thread_obj = original_thread()
        thread_calls.append((target, args, daemon, kwargs))
        return thread_obj
    
    monkeypatch.setattr(cc_ng_host.threading, 'Thread', inline_thread)
    
    msg = "This is my reply to the user."
    result = cc_ng_host._DISPATCH["Stop"]({"last_assistant_message": msg})
    
    assert result == {"ok": True}
    assert len(thread_calls) == 1
    target, args, daemon, kwargs = thread_calls[0]
    assert daemon is True
    assert args == (msg,)
    assert kwargs == {"step": True}

    # The deposit itself (spy in this fixture) got the text once, step=True.
    assert host.deposit_calls == [(msg, True)]


def test_stop_never_returns_context(host):
    """D — no context."""
    # Already covered in test_stop_with_want_materializes_want_node,
    # but explicitly test various cases
    
    test_cases = [
        {"last_assistant_message": "Hello"},
        {"last_assistant_message": ""},
        {},
        {"last_assistant_message": None},
    ]
    
    for data in test_cases:
        result = cc_ng_host._DISPATCH["Stop"](data)
        assert result == {"ok": True}
        assert "context" not in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])