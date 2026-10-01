#!/usr/bin/env python3
# ---- Changelog ----
# [2026-10-01] Claude Sonnet 5.5 (Z12 builder, lane model-guard-910, dispatch #14031) — #910 M8: the no-model-load tripwire RECORDS its hits
# What: test_resolver_loads_no_model_and_touches_no_network: the _ensure_model tripwire and the socket.connect block now append to
#   a `hits` list BEFORE raising, and the test asserts `hits == []` after the resolver call (in addition to every assertion it
#   already made). TEST-ONLY: no source file, no other test, and no other assertion changed; the test count is unchanged (53).
# Why: build-002-addendum-001 caveat 2 (Chief-003 ruling #14031): the resolver's own `except Exception` swallows the tripwire's
#   AssertionError, so a resolver that loaded the model inside its OWN inner try/except, discarded the failure and still
#   returned the id (M8b) would have survived this test; the old M8 was killed only because its failure made the resolver
#   return None.
# How: record-then-raise; the list is inspected by the test, which a swallowing `except` cannot hide.
# [2026-10-01] Claude Sonnet 5.5 (Z12 builder, lane model-guard-910, dispatch #13848) — #949: the resolver READS, it never CREATES
# What: new section (f): two FRESH-PROCESS tests (a resolver call leaves NGEmbed._instance None; a later non-default config is not
#   pinned out by a header read), an existing-instance test, an id-unchanged test (== what get_instance() would report), a
#   no-get_instance/no-reset_instance tripwire, and a raising path with no instance. Four existing tests encoded "the resolver
#   builds the singleton" (they patched NGEmbed.get_instance / NGEmbed.stats) and were re-pointed at `NGEmbed._instance` with the
#   same intent and the same or stronger assertions: test_unresolvable_local_id_via_the_real_resolver_aborts,
#   test_sender_refuses_when_the_real_resolver_fails, test_resolver_never_returns_unknown_or_blank (now also covers the no-instance
#   default constant), test_resolver_is_none_not_unknown_when_the_embedder_cannot_be_built (renamed ..._cannot_be_read).
# Why: Exec P518 / le-055 O1 / row #949: get_instance() created the process-wide singleton as a side effect of a header read.
# How: the fresh-process tests load the cc_topology_export UNDER TEST by path (so Z12_EXPORT_UNDER_TEST drives them against the
#   5df0fb7a copy for the fail-first). No assertion was weakened.
# [2026-10-01] Claude Sonnet 5.5 (Z12 builder, lane model-guard-910, dispatch #13145) — #910 embedding-model guard tests
# What: drives the REAL cc_topology_export (both exporters) and the REAL cc_topology_merge.merge_cc_topology into a REAL
#   neuro_foundation.Graph + SimpleVectorDB, with the REAL ng_embed.NGEmbed singleton as the model-id source. Cases:
#   (a) the reproducers that prove the guard was dead (a real conduit whose header names a DIFFERENT model LANDS on the base;
#   the base exporter stamps "unknown"); (b) the fail-closed receiver (mismatch / missing / None / blank / "unknown" / non-str
#   wire id / unresolvable local id / "unknown" expectation all abort; a matching id lands; expected_embedding_model
#   overrides); (c) the sender refuses to export without a resolvable id (both exporters, nothing written); (d) the ONE
#   resolver (configured id of a non-default NGEmbed, never "unknown", no model load, no network); (e) an abort leaves the
#   conduit byte-identical, nothing landed, the membership ack untouched, and logs the reason at ERROR.
# Why: Exec Packet 492 / #910. Both ends read ng_embed.MODEL_NAME/_MODEL_NAME (which ng_embed does not define), always
#   resolved "unknown", and the receiver's `"unknown" not in (wire, local)` waiver made FatherGraph Finding 6 dead code.
#   Every existing fixture passed an explicit id ("test-model") to BOTH ends, which is why nothing noticed.
# How: no fakes for the graph, the exporters or the merge. NGEmbed is real; no model is loaded (a spy on _ensure_model
#   fails the test if the resolver reaches it, and socket.connect is blocked). Z12_EXPORT_UNDER_TEST / Z12_MERGE_UNDER_TEST
#   (paths) load a different cc_topology_export.py / cc_topology_merge.py as the module under test, so this SAME file runs
#   unchanged against the BASE copies (git show ac5800fd:<file>) and against mutants. The scratch exporter is registered as
#   sys.modules['cc_topology_export'] BEFORE the merge loads, so a scratch merge binds to the scratch exporter.
#   P379/#770: the printed preamble names every module under test.
# -------------------
"""Which cases FAIL on the BASE copies and on each mutant: see the return file."""
import hashlib
import importlib.util
import json
import logging
import os
import socket
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

_WORKTREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_WORKTREE))


def _load_by_path(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


from neuro_foundation import Graph  # noqa: E402
from universal_ingestor import SimpleVectorDB  # noqa: E402
import cc_ng_organism as cno  # noqa: E402
import ng_embed  # noqa: E402
from ng_embed import NGEmbed  # noqa: E402

_EXPORT_UNDER_TEST = os.environ.get("Z12_EXPORT_UNDER_TEST")
if _EXPORT_UNDER_TEST:
    # Must happen BEFORE the merge is imported: the merge does `from cc_topology_export import ...`.
    tex = _load_by_path(Path(_EXPORT_UNDER_TEST).resolve(), "cc_topology_export")
else:
    import cc_topology_export as tex  # noqa: E402

_MERGE_UNDER_TEST = os.environ.get("Z12_MERGE_UNDER_TEST")
if _MERGE_UNDER_TEST:
    tmg = _load_by_path(Path(_MERGE_UNDER_TEST).resolve(), "cc_topology_merge_under_test")
else:
    import cc_topology_merge as tmg  # noqa: E402

print("[P379/#770 preamble] worktree root          ->", _WORKTREE)
print("[P379/#770 preamble] export under test      ->", Path(tex.__file__).resolve())
print("[P379/#770 preamble] merge under test       ->", Path(tmg.__file__).resolve())
for _m in (cno, ng_embed):
    print("[P379/#770 preamble] %-20s ->" % _m.__name__, Path(_m.__file__).resolve())

DIM = 768
DEFAULT_ID = "Snowflake/snowflake-arctic-embed-m-v1.5"   # ng_embed._DEFAULT_CONFIG["model_id"], read at ng_embed.py :221


def test_modules_under_test_resolve_inside_the_worktree():
    if _EXPORT_UNDER_TEST or _MERGE_UNDER_TEST:
        pytest.skip("Z12_*_UNDER_TEST is set: the module under test is deliberately elsewhere")
    for mod in (tex, tmg, cno, ng_embed):
        assert _WORKTREE in Path(mod.__file__).resolve().parents, f"{mod.__name__} resolves outside {_WORKTREE}"


# ---------------------------------------------------------------- rig

@pytest.fixture(autouse=True)
def _fresh_embedder():
    """The REAL NGEmbed singleton, reset around every test so a non-default config cannot leak."""
    NGEmbed.reset_instance()
    yield
    NGEmbed.reset_instance()


_SEED = [0]


class _BrokenInstance:
    """Stands in for an existing NGEmbed instance whose `stats` read raises (#949: the resolver READS `_instance`, it
    never calls get_instance(), so this is how the real resolver is made to fail)."""

    @property
    def stats(self):
        raise RuntimeError("SECRET-DETAIL-must-not-reach-the-log")


class _StatsInstance:
    """Stands in for an existing NGEmbed instance that reports a given (possibly unusable) model_id."""

    def __init__(self, model_id):
        self._model_id = model_id

    @property
    def stats(self):
        return {"model_id": self._model_id}


def _emb():
    _SEED[0] += 1
    return np.random.default_rng(_SEED[0]).normal(size=DIM).astype(np.float32)


def _sender():
    """Two bound CC pairs: a real graph + vector db the real exporters can read."""
    g, v = Graph(), SimpleVectorDB()
    for i, tag in enumerate(("a", "b")):
        f, t = f"cc:conv::{tag}", f"cc:conv::{tag}::tree::x"
        for j, (nid, mode) in enumerate(((f, "conversational"), (t, "tree"))):
            meta = {"cc": True, "creation_mode": mode, "_forest_content": "c-" + nid}
            g.create_node(node_id=nid, metadata=meta).creation_time = 10 * i + j
            v.insert(id=nid, embedding=_emb(), content="content for " + nid, metadata=meta)
        g.create_synapse(f, t, weight=0.4, delay=2)
    return g, v


def _export(g, v, tmp_path, name="topo.conduit", **kw):
    """The whole-graph exporter, model id NOT passed unless the test says so (the real resolve path)."""
    out = str(tmp_path / name)
    kw.setdefault("machine_id", "vps")
    tex.export_cc_topology(g, v, out, **kw)
    return out


def _export_frame(g, v, tmp_path, name="frame.conduit", **kw):
    out = str(tmp_path / name)
    kw.setdefault("machine_id", "vps")
    kw.setdefault("skip_resource_gate", True)
    return out, tex.export_cc_topology_frame(g, v, out, **kw)


def _frames(path):
    return list(tex.read_topology_frames(open(path, "rb").read()))


def _header_model(path):
    return _frames(path)[0].get("embedding_model")


_MISSING = object()


def _rewrite_header_model(path, tmp_path, value, name="edited.conduit"):
    """Hand-edit ONLY the header's embedding_model (`_MISSING` drops the key); every other frame is byte-for-byte re-framed."""
    frames = _frames(path)
    hdr = dict(frames[0])
    if value is _MISSING:
        hdr.pop("embedding_model", None)
    else:
        hdr["embedding_model"] = value
    out = str(tmp_path / name)
    with open(out, "wb") as fh:
        fh.write(tex._frame(hdr))
        for fr in frames[1:]:
            fh.write(tex._frame(fr))
    return out


def _merge(rg, rv, path, tmp_path, **kw):
    kw.setdefault("local_machine_id", "laptop")
    kw.setdefault("membership_path", str(tmp_path / "membership.txt"))
    kw.setdefault("idle_steps", 0)
    return tmg.merge_cc_topology(rg, rv, path, **kw)


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _graph_state(g, v):
    return (sorted(g.nodes), sorted(g.synapses), sorted(g.hyperedges), sorted(v.embeddings), sorted(v.content))


# ================================================================ (a) the reproducers: the guard was dead

def test_REPRODUCER_real_conduit_with_a_foreign_header_model_must_not_land(tmp_path):
    """A REAL conduit (real exporter, real resolve), header hand-edited to a DIFFERENT model id, merged with NO
    expected_embedding_model (the daemon's call: handle_merge_topology passes none). On the BASE both ends resolve
    "unknown" and the waiver lets it LAND; on the fix the receiver aborts and nothing lands."""
    sg, sv = _sender()
    real = _export(sg, sv, tmp_path)
    foreign = _rewrite_header_model(real, tmp_path, "some-other-lab/embedder-v9")
    rg, rv = Graph(), SimpleVectorDB()
    with pytest.raises(tmg.TopologyMergeAbort, match="embedding model mismatch"):
        _merge(rg, rv, foreign, tmp_path)
    assert len(rg.nodes) == 0, "a conduit from a different embedding model landed"


def test_REPRODUCER_the_exporter_stamps_the_configured_id_not_unknown(tmp_path):
    """The base exporter's header says "unknown" (the VPS conduit header does today); the fix stamps the real id."""
    sg, sv = _sender()
    path = _export(sg, sv, tmp_path)
    model = _header_model(path)
    assert model != "unknown", "the exporter stamped the dead-guard sentinel"
    assert model == NGEmbed.get_instance().stats["model_id"] == DEFAULT_ID


def test_REPRODUCER_the_frame_exporter_stamps_the_configured_id_not_unknown(tmp_path):
    sg, sv = _sender()
    path, stats = _export_frame(sg, sv, tmp_path)
    assert stats["exported_nodes"] > 0
    assert _header_model(path) == stats["embedding_model"] == DEFAULT_ID != "unknown"


def test_real_conduit_with_no_model_passed_anywhere_round_trips(tmp_path):
    """Real exporter + real receiver, the id resolved by the ONE resolver on both ends, no expected_embedding_model:
    the matching case lands (pins that the fix does not just abort everything)."""
    sg, sv = _sender()
    path = _export(sg, sv, tmp_path)
    rg, rv = Graph(), SimpleVectorDB()
    st = _merge(rg, rv, path, tmp_path)
    assert st["status"] == "ok" and st["embedding_model"] == DEFAULT_ID
    assert st["absorbed_nodes"] == 4 and len(rg.nodes) == 4


def test_matching_non_default_configured_model_lands(tmp_path):
    """A NGEmbed built with a non-default model_id: the sender stamps it, the receiver resolves the same one."""
    NGEmbed.get_instance({"model_id": "acme/custom-embedder-m"})
    sg, sv = _sender()
    path = _export(sg, sv, tmp_path)
    assert _header_model(path) == "acme/custom-embedder-m"
    rg, rv = Graph(), SimpleVectorDB()
    assert _merge(rg, rv, path, tmp_path)["absorbed_nodes"] == 4


def test_receiver_resolving_a_different_configured_model_aborts(tmp_path):
    """Sender stamped the default id; the receiver's live embedder is configured differently -> mismatch, not landed."""
    sg, sv = _sender()
    path = _export(sg, sv, tmp_path)                       # stamps DEFAULT_ID
    NGEmbed.reset_instance()
    NGEmbed.get_instance({"model_id": "acme/custom-embedder-m"})
    rg, rv = Graph(), SimpleVectorDB()
    with pytest.raises(tmg.TopologyMergeAbort, match="embedding model mismatch"):
        _merge(rg, rv, path, tmp_path)
    assert len(rg.nodes) == 0


# ================================================================ (b) the fail-closed receiver

def test_explicit_mismatch_aborts_and_names_both_ids(tmp_path):
    sg, sv = _sender()
    path = _export(sg, sv, tmp_path, embedding_model="model-A")
    rg, rv = Graph(), SimpleVectorDB()
    with pytest.raises(tmg.TopologyMergeAbort) as ei:
        _merge(rg, rv, path, tmp_path, expected_embedding_model="model-B")
    assert "model-A" in str(ei.value) and "model-B" in str(ei.value)
    assert len(rg.nodes) == 0


@pytest.mark.parametrize("wire", [None, "", "   ", "unknown", "UNKNOWN", " Unknown ", 123, ["x"], _MISSING],
                         ids=["None", "empty", "blank", "unknown", "UNKNOWN", "padded-unknown", "int", "list", "key-absent"])
def test_unstamped_or_unknown_wire_id_aborts_even_when_local_is_known(tmp_path, wire):
    sg, sv = _sender()
    path = _rewrite_header_model(_export(sg, sv, tmp_path), tmp_path, wire)
    rg, rv = Graph(), SimpleVectorDB()
    with pytest.raises(tmg.TopologyMergeAbort, match="unverifiable"):
        _merge(rg, rv, path, tmp_path, expected_embedding_model="test-model")
    assert len(rg.nodes) == 0


def test_unknown_on_both_sides_aborts_no_waiver(tmp_path):
    """The exact state of the real wire before this fix: header "unknown", receiver "unknown". It must not land."""
    sg, sv = _sender()
    path = _rewrite_header_model(_export(sg, sv, tmp_path), tmp_path, "unknown")
    rg, rv = Graph(), SimpleVectorDB()
    with pytest.raises(tmg.TopologyMergeAbort):
        _merge(rg, rv, path, tmp_path, expected_embedding_model="unknown")
    assert len(rg.nodes) == 0


def test_unresolvable_local_id_aborts_even_with_a_valid_wire_id(tmp_path, monkeypatch):
    """The resolver itself is patched to None (raising=False: on the base the attribute does not exist, and the test must
    fail on BEHAVIOUR -- the frame landed -- not on an AttributeError)."""
    sg, sv = _sender()
    path = _export(sg, sv, tmp_path, embedding_model="model-A")
    monkeypatch.setattr(tmg, "resolve_embedding_model", lambda: None, raising=False)
    monkeypatch.setattr(tmg, "_local_embedding_model", lambda: None, raising=False)
    rg, rv = Graph(), SimpleVectorDB()
    with pytest.raises(tmg.TopologyMergeAbort, match="local_model_unresolvable"):
        _merge(rg, rv, path, tmp_path)
    assert len(rg.nodes) == 0


def test_unresolvable_local_id_via_the_real_resolver_aborts(tmp_path, monkeypatch):
    """No patching of the receiver: the REAL resolver fails because the existing embedder instance cannot report its id.
    (#949: was a `get_instance` patch -- it encoded "the resolver builds the singleton"; the resolver now READS `_instance`.)"""
    sg, sv = _sender()
    path = _export(sg, sv, tmp_path, embedding_model="model-A")

    monkeypatch.setattr(NGEmbed, "_instance", _BrokenInstance())
    rg, rv = Graph(), SimpleVectorDB()
    with pytest.raises(tmg.TopologyMergeAbort, match="local_model_unresolvable"):
        _merge(rg, rv, path, tmp_path)
    assert len(rg.nodes) == 0


def test_expected_embedding_model_overrides_the_live_resolver(tmp_path):
    """An explicit expectation wins over the local resolve in BOTH directions."""
    sg, sv = _sender()
    path = _export(sg, sv, tmp_path, embedding_model="model-A")     # live embedder says DEFAULT_ID
    rg, rv = Graph(), SimpleVectorDB()
    assert _merge(rg, rv, path, tmp_path, expected_embedding_model="model-A")["absorbed_nodes"] == 4
    default_path = _export(sg, sv, tmp_path, name="d.conduit")      # stamps DEFAULT_ID == live
    rg2, rv2 = Graph(), SimpleVectorDB()
    with pytest.raises(tmg.TopologyMergeAbort, match="mismatch"):
        _merge(rg2, rv2, default_path, tmp_path, expected_embedding_model="model-A")
    assert len(rg2.nodes) == 0


# ================================================================ (c) the sender refuses

def _no_files(tmp_path, out):
    assert not os.path.exists(out), "a conduit was written"
    assert not os.path.exists(out + ".partial"), "a partial conduit was left behind"


def test_sender_refuses_whole_graph_export_when_the_model_id_is_unresolvable(tmp_path, monkeypatch):
    sg, sv = _sender()
    monkeypatch.setattr(tex, "resolve_embedding_model", lambda: None, raising=False)
    out = str(tmp_path / "x.conduit")
    with pytest.raises(ValueError, match="embedding model"):
        tex.export_cc_topology(sg, sv, out, machine_id="vps")
    _no_files(tmp_path, out)


def test_sender_refuses_frame_export_when_the_model_id_is_unresolvable(tmp_path, monkeypatch):
    sg, sv = _sender()
    monkeypatch.setattr(tex, "resolve_embedding_model", lambda: None, raising=False)
    out = str(tmp_path / "f.conduit")
    with pytest.raises(ValueError, match="embedding model"):
        tex.export_cc_topology_frame(sg, sv, out, machine_id="vps", skip_resource_gate=True)
    _no_files(tmp_path, out)


def test_sender_refuses_when_the_real_resolver_fails(tmp_path, monkeypatch):
    # #949: was a `get_instance` patch (encoded "the resolver builds the singleton"); the resolver now READS `_instance`.
    sg, sv = _sender()
    monkeypatch.setattr(NGEmbed, "_instance", _BrokenInstance())
    out = str(tmp_path / "x.conduit")
    with pytest.raises(ValueError) as ei:
        tex.export_cc_topology(sg, sv, out, machine_id="vps")
    assert "SECRET-DETAIL" not in str(ei.value), "str(exc) of the underlying failure leaked into the refusal"
    _no_files(tmp_path, out)


@pytest.mark.parametrize("explicit", ["", "   ", "unknown", "Unknown"])
def test_sender_refuses_an_explicit_blank_or_unknown_id(tmp_path, explicit):
    sg, sv = _sender()
    out = str(tmp_path / "x.conduit")
    with pytest.raises(ValueError, match="embedding model"):
        tex.export_cc_topology(sg, sv, out, machine_id="vps", embedding_model=explicit)
    _no_files(tmp_path, out)
    out2 = str(tmp_path / "f.conduit")
    with pytest.raises(ValueError, match="embedding model"):
        tex.export_cc_topology_frame(sg, sv, out2, machine_id="vps", embedding_model=explicit, skip_resource_gate=True)
    _no_files(tmp_path, out2)


def test_sender_honours_an_explicit_real_id(tmp_path):
    sg, sv = _sender()
    path = _export(sg, sv, tmp_path, embedding_model="explicit/model-1")
    assert _header_model(path) == "explicit/model-1"


# ================================================================ (d) the ONE resolver

def test_resolver_is_shared_not_copied():
    """Receiver and sender use the SAME function object (LAW 3): no per-side copy."""
    assert tmg.resolve_embedding_model is tex.resolve_embedding_model
    assert tmg.is_resolved_model_id is tex.is_resolved_model_id
    assert not hasattr(tmg, "_local_embedding_model"), "the dead receiver-side copy came back"


def test_resolver_returns_the_instances_configured_id():
    assert tex.resolve_embedding_model() == DEFAULT_ID
    NGEmbed.reset_instance()
    NGEmbed.get_instance({"model_id": "acme/custom-embedder-m"})
    assert tex.resolve_embedding_model() == "acme/custom-embedder-m"


def test_resolver_loads_no_model_and_touches_no_network(monkeypatch):
    """How "no model load" is shown, not asserted: _ensure_model (the ONLY place ng_embed loads ONNX / the tokenizer / hits
    the hub) is replaced with a tripwire, socket.connect is blocked, and the instance must still report model_loaded False.

    The tripwires RECORD each hit in `hits` BEFORE they raise, and the test asserts `hits == []` afterwards. The raise alone is
    not enough: the resolver wraps its work in `try/except Exception`, which would swallow the tripwire's AssertionError, so a
    resolver that attempted a load inside its OWN inner try/except, discarded the failure and still returned the id would
    otherwise pass this test (build-002-addendum-001 caveat 2, mutant M8b)."""
    hits = []

    def tripwire(self, *a, **k):
        hits.append("NGEmbed._ensure_model")
        raise AssertionError("resolver reached NGEmbed._ensure_model (a model load)")

    def no_net(self, *a, **k):
        hits.append("socket.socket.connect")
        raise AssertionError("resolver touched the network")

    monkeypatch.setattr(NGEmbed, "_ensure_model", tripwire)
    monkeypatch.setattr(socket.socket, "connect", no_net)
    assert tex.resolve_embedding_model() == DEFAULT_ID
    assert hits == [], "the resolver reached a model load / the network (swallowed or not): %r" % (hits,)
    assert NGEmbed.get_instance().stats["model_loaded"] is False


@pytest.mark.parametrize("bad", ["unknown", "UNKNOWN", "", "   ", None, 7])
def test_resolver_never_returns_unknown_or_blank(monkeypatch, bad):
    """Whatever the embedder reports, an unusable id is None -- never the string "unknown". Both sources: an existing
    instance's stats, and (no instance) the class default constant. (#949: the old version patched `NGEmbed.stats`, which
    only mattered because the resolver used to create the instance.)"""
    monkeypatch.setattr(NGEmbed, "_instance", _StatsInstance(bad))
    assert tex.resolve_embedding_model() is None
    monkeypatch.setattr(NGEmbed, "_instance", None)
    monkeypatch.setitem(ng_embed._DEFAULT_CONFIG, "model_id", bad)
    assert tex.resolve_embedding_model() is None


def test_resolver_is_none_not_unknown_when_the_embedder_cannot_be_read(monkeypatch, caplog):
    monkeypatch.setattr(NGEmbed, "_instance", _BrokenInstance())
    with caplog.at_level(logging.DEBUG):
        got = tex.resolve_embedding_model()
    assert got is None
    assert "RuntimeError" in caplog.text, "the exception CLASS should be logged"
    assert "SECRET-DETAIL" not in caplog.text, "str(exc) reached the log"


def test_is_resolved_model_id_truth_table():
    for ok in ("Snowflake/snowflake-arctic-embed-m-v1.5", "m", "test-model", "unknownish"):
        assert tex.is_resolved_model_id(ok), ok
    for bad in (None, "", " ", "unknown", "UNKNOWN", "\tunknown\n", 0, 1, [], {}, b"x"):
        assert not tex.is_resolved_model_id(bad), repr(bad)


# ================================================================ (e) an abort leaves everything where it was

_ABORT_CASES = [
    ("model_mismatch", "model-A", dict(expected_embedding_model="model-B")),
    ("wire_model_unverifiable", "unknown", dict(expected_embedding_model="model-B")),
    ("wire_model_unverifiable", None, dict(expected_embedding_model="model-B")),
    ("local_model_unresolvable", "model-A", dict(expected_embedding_model="unknown")),
]


@pytest.mark.parametrize("reason,wire,kw", _ABORT_CASES, ids=[f"{r}-{w!r}" for r, w, _ in _ABORT_CASES])
def test_abort_leaves_conduit_byte_identical_nothing_landed_ack_untouched(tmp_path, caplog, reason, wire, kw):
    sg, sv = _sender()
    path = _rewrite_header_model(_export(sg, sv, tmp_path), tmp_path, wire)
    membership = tmp_path / "membership.txt"
    membership.write_bytes(b"cc:conv::prior-ack\n")                 # a prior ack the abort must not advance
    rg, rv = Graph(), SimpleVectorDB()
    before_sha, before_ack, before_graph = _sha(path), membership.read_bytes(), _graph_state(rg, rv)
    with caplog.at_level(logging.ERROR):
        with pytest.raises(tmg.TopologyMergeAbort):
            tmg.merge_cc_topology(rg, rv, path, local_machine_id="laptop", membership_path=str(membership),
                                  idle_steps=0, **kw)
    assert os.path.exists(path) and _sha(path) == before_sha, "the conduit was consumed or altered"
    assert membership.read_bytes() == before_ack, "the membership ack advanced on an abort"
    assert _graph_state(rg, rv) == before_graph, "something landed on an abort"
    assert not os.path.exists(str(membership) + ".partial")
    assert f"reason={reason}" in caplog.text, "the abort was not logged loudly with its reason code"
    assert any(r.levelno == logging.ERROR for r in caplog.records)


def test_a_passing_merge_still_advances_the_ack(tmp_path):
    """The other half of the above: the ack file IS written when the guard passes (so the previous test is not vacuous)."""
    sg, sv = _sender()
    path = _export(sg, sv, tmp_path)
    membership = tmp_path / "membership.txt"
    membership.write_bytes(b"cc:conv::prior-ack\n")
    rg, rv = Graph(), SimpleVectorDB()
    _merge(rg, rv, path, tmp_path, membership_path=str(membership))
    assert membership.read_bytes() != b"cc:conv::prior-ack\n"
    assert b"cc:conv::a" in membership.read_bytes()


def test_the_abort_message_caps_an_untrusted_wire_id(tmp_path):
    sg, sv = _sender()
    path = _rewrite_header_model(_export(sg, sv, tmp_path), tmp_path, "X" * 5000)
    rg, rv = Graph(), SimpleVectorDB()
    with pytest.raises(tmg.TopologyMergeAbort) as ei:
        _merge(rg, rv, path, tmp_path, expected_embedding_model="model-B")
    assert len(str(ei.value)) < 600, "an untrusted wire id was interpolated unbounded"


# ================================================================ (f) #949: the resolver READS, it never CREATES

def _fresh_process(tmp_path, body):
    """Run `body` in a FRESH python process (PYTHONPATH = the worktree, a scratch HOME, HF_HUB_OFFLINE=1) with the
    cc_topology_export under test loaded by path as `tex` and ng_embed imported; `body` fills the dict `out`, which is
    returned. A fresh process is the only honest place to ask "did the resolver leave NGEmbed._instance unset?": in this
    pytest process the autouse fixture has already touched the singleton."""
    code = textwrap.dedent("""
        import importlib.util, json, sys
        spec = importlib.util.spec_from_file_location("cc_topology_export", sys.argv[1])
        tex = importlib.util.module_from_spec(spec)
        sys.modules["cc_topology_export"] = tex
        spec.loader.exec_module(tex)
        import ng_embed
        out = {}
    """) + textwrap.dedent(body) + "\nprint('RESULT=' + json.dumps(out))\n"
    home = tmp_path / "fresh-home"
    home.mkdir(exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(_WORKTREE), HOME=str(home), HF_HUB_OFFLINE="1")
    proc = subprocess.run([sys.executable, "-c", code, str(Path(tex.__file__).resolve())],
                          env=env, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, "fresh process failed: " + proc.stderr[-800:]
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT=")][-1]
    return json.loads(line[len("RESULT="):])


def test_FRESH_PROCESS_a_resolver_call_leaves_the_singleton_uncreated(tmp_path):
    """The #949 defect, in a process where nothing else has touched NGEmbed: on 5df0fb7a the resolver's get_instance()
    CREATES the instance (this fails); on the fix `NGEmbed._instance` is still None and the default id is reported."""
    out = _fresh_process(tmp_path, """
        out["before"] = ng_embed.NGEmbed._instance is None
        out["resolved"] = tex.resolve_embedding_model()
        out["instance_is_none_after"] = ng_embed.NGEmbed._instance is None
    """)
    assert out["before"] is True, "the fresh process was not fresh"
    assert out["resolved"] == DEFAULT_ID
    assert out["instance_is_none_after"] is True, "the resolver CREATED the process-wide NGEmbed singleton"


def test_FRESH_PROCESS_a_later_non_default_config_is_not_pinned_out_by_a_header_read(tmp_path):
    """The consequence of the same defect: a header read first, then a non-default config passed to get_instance(). On
    5df0fb7a the resolver already created the singleton with the DEFAULT config (first-creation-wins), so the later config
    is silently ignored; on the fix it takes effect."""
    out = _fresh_process(tmp_path, """
        tex.resolve_embedding_model()
        out["later"] = ng_embed.NGEmbed.get_instance({"model_id": "later/non-default"}).stats["model_id"]
    """)
    assert out["later"] == "later/non-default", "a header read pinned the default config (first-creation-wins)"


def test_resolver_in_process_leaves_no_instance_behind():
    assert NGEmbed._instance is None            # the autouse fixture reset it
    assert tex.resolve_embedding_model() == DEFAULT_ID
    assert NGEmbed._instance is None, "the resolver created the singleton"


def test_resolver_reports_an_existing_instances_configured_id_and_leaves_it_alone():
    inst = NGEmbed.get_instance({"model_id": "acme/other-embedder-x"})
    assert tex.resolve_embedding_model() == "acme/other-embedder-x"
    assert NGEmbed._instance is inst, "the resolver replaced or reset the existing instance"
    assert inst.stats["model_id"] == "acme/other-embedder-x"


def test_resolver_returns_the_same_id_get_instance_would_have_reported():
    """The returned model_id is UNCHANGED by #949: for the default config it is exactly what get_instance() would create
    and report; for an existing non-default instance it is that instance's id."""
    no_instance = tex.resolve_embedding_model()
    assert NGEmbed._instance is None
    assert no_instance == ng_embed._DEFAULT_CONFIG["model_id"] == DEFAULT_ID
    assert no_instance == NGEmbed().stats["model_id"], "differs from a default-config NGEmbed (direct, non-singleton)"
    assert no_instance == NGEmbed.get_instance().stats["model_id"], "differs from what get_instance() would have reported"
    assert tex.resolve_embedding_model() == no_instance                      # same answer once the instance exists
    NGEmbed.reset_instance()
    NGEmbed.get_instance({"model_id": "acme/other-embedder-x"})
    assert tex.resolve_embedding_model() == NGEmbed.get_instance().stats["model_id"] == "acme/other-embedder-x"


def test_resolver_never_calls_get_instance_or_reset_instance(monkeypatch):
    """No create and no reset, whether or not an instance exists (tripwires, both states)."""
    def no_get(cls, config=None):
        raise AssertionError("resolver called NGEmbed.get_instance() (a CREATE)")

    def no_reset(cls):
        raise AssertionError("resolver called NGEmbed.reset_instance()")

    monkeypatch.setattr(NGEmbed, "get_instance", classmethod(no_get))
    monkeypatch.setattr(NGEmbed, "reset_instance", classmethod(no_reset))
    assert tex.resolve_embedding_model() == DEFAULT_ID                       # no instance
    monkeypatch.setattr(NGEmbed, "_instance", _StatsInstance("acme/other-embedder-x"))
    assert tex.resolve_embedding_model() == "acme/other-embedder-x"          # an instance


def test_resolver_returns_none_when_the_default_constant_cannot_be_read(monkeypatch, caplog):
    """A raising path with NO instance: the class default constant is missing -> None, class name only in the log."""
    monkeypatch.delitem(ng_embed._DEFAULT_CONFIG, "model_id")
    with caplog.at_level(logging.DEBUG):
        got = tex.resolve_embedding_model()
    assert got is None
    assert "KeyError" in caplog.text and "unknown" not in caplog.text.lower()

