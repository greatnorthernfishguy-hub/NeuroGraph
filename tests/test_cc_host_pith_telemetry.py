# ---- Changelog ----
# [2026-09-12] Codex — hosted D4 telemetry durability and lifecycle coverage.
# What: exercise the real hosted snapshot/config functions against temp files.
# Why: VPS acceptance counters otherwise disappear whenever the gateway restarts.
# How: stub organism metrics, disposable files/processes, and static call-site checks;
#   no live graph, model, socket, service, or checkpoint is opened.
# -------------------
"""D4 acceptance telemetry contract for the VPS-hosted CC process."""

import importlib.util
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import types

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOST_PATH = os.path.join(ROOT, "cc_ng_host.py")
sys.path.insert(0, ROOT)

import cc_ng_organism as real_organism


def _load_host():
    spec = importlib.util.spec_from_file_location(
        "cc_ng_host_telemetry_under_test", HOST_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stub_organism(start=0, step=7, include_config=False):
    stub = types.ModuleType("cc_ng_organism")
    counter = {"n": start}

    class Metrics:
        @staticmethod
        def snapshot():
            counter["n"] += step
            return {
                "ranked_kept": counter["n"],
                "l1_assemblies": counter["n"],
            }

    stub._PITH_METRICS = Metrics()
    if include_config:
        stub.pith_effective_config = lambda: {
            "env": {"CC_PITH_PREFETCH_MAX": None},
            "resolved": {"CC_PITH_PREFETCH_MAX": 15},
            "authority": {"CC_PITH_PREFETCH_MAX": "tonic_engine"},
        }
    return stub


@pytest.fixture
def host(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "cc_ng_organism", _stub_organism())
    module = _load_host()
    module.PITH_SNAPSHOT_PATH = str(tmp_path / "pith_metrics.jsonl")
    return module


def _records(path):
    if not os.path.exists(path):
        return []
    with open(path) as stream:
        return [json.loads(line) for line in stream if line.strip()]


def test_gate_allowlist_matches_canonical_configuration_keys():
    host = _load_host()
    assert set(host.PITH_SNAPSHOT_GATE_KEYS) == set(real_organism._PITH_CONFIG_KEYS)


def test_gates_never_capture_the_whole_environment(host, monkeypatch):
    monkeypatch.setenv("CC_PITH_ENABLED", "1")
    monkeypatch.setenv("SECRET_TOKEN_MUST_NOT_APPEAR", "sentinel-value")
    gates = host._pith_gate_values()
    assert gates["CC_PITH_ENABLED"] == "1"
    assert "SECRET_TOKEN_MUST_NOT_APPEAR" not in gates
    assert "sentinel-value" not in repr(gates)


def test_unset_gate_remains_visible(host, monkeypatch):
    monkeypatch.delenv("CC_PITH_PREFETCH_WARM_ENABLED", raising=False)
    assert host._pith_gate_values()["CC_PITH_PREFETCH_WARM_ENABLED"] is None


def test_snapshot_is_complete_append_only_and_differenceable(host):
    host._STATE.stats["started_at"] = 123.5
    for reason in ("start", "interval", "shutdown"):
        assert host._pith_snapshot(reason) is True
    records = _records(host.PITH_SNAPSHOT_PATH)
    assert [record["reason"] for record in records] == [
        "start",
        "interval",
        "shutdown",
    ]
    assert all(
        key in records[0]
        for key in (
            "ts",
            "iso",
            "pid",
            "window_id",
            "window_started_ts",
            "gates",
            "config",
            "config_error",
            "counters",
        )
    )
    assert records[0]["window_started_ts"] == 123.5
    values = [record["counters"]["l1_assemblies"] for record in records]
    assert values == [7, 14, 21]


def test_window_id_is_stable_within_one_host_lifetime(host):
    host._pith_snapshot("start")
    host._pith_snapshot("interval")
    assert {record["window_id"] for record in _records(host.PITH_SNAPSHOT_PATH)} == {
        host.PITH_WINDOW_ID
    }


def test_window_id_differs_between_host_lifetimes():
    assert _load_host().PITH_WINDOW_ID != _load_host().PITH_WINDOW_ID


def test_snapshot_records_resolved_configuration(host):
    sys.modules["cc_ng_organism"].pith_effective_config = lambda: {
        "env": {"CC_PITH_PREFETCH_MAX": None},
        "resolved": {"CC_PITH_PREFETCH_MAX": 15},
        "authority": {"CC_PITH_PREFETCH_MAX": "tonic_engine"},
    }
    host._pith_snapshot("start")
    record = _records(host.PITH_SNAPSHOT_PATH)[0]
    assert record["config"]["resolved"]["CC_PITH_PREFETCH_MAX"] == 15
    assert record["config_error"] is None


def test_unavailable_configuration_has_stable_code_and_no_exception_leak(
    host, caplog
):
    def fail():
        raise RuntimeError("token=SENTINEL-LEAK-VALUE")

    sys.modules["cc_ng_organism"].pith_effective_config = fail
    with caplog.at_level(logging.WARNING):
        host._pith_snapshot("start")
        host._pith_snapshot("interval")
    raw = open(host.PITH_SNAPSHOT_PATH).read()
    assert "SENTINEL-LEAK-VALUE" not in raw
    assert all(record["config_error"] == "resolve_failed" for record in _records(host.PITH_SNAPSHOT_PATH))
    warnings = [record for record in caplog.records if "config UNAVAILABLE" in record.message]
    assert len(warnings) == 1


def test_snapshot_never_raises_on_bad_path(host):
    host.PITH_SNAPSHOT_PATH = "/proc/nonexistent-dir/pith.jsonl"
    assert host._pith_snapshot("shutdown") is False


def test_retention_rotates_instead_of_growing_unbounded(host):
    host.PITH_SNAPSHOT_MAX_BYTES = 800
    for _ in range(40):
        host._pith_snapshot("interval")
    assert os.path.exists(host.PITH_SNAPSHOT_PATH + ".1")
    assert os.path.getsize(host.PITH_SNAPSHOT_PATH) <= 1600


def test_concurrent_snapshot_writers_leave_complete_json_records(host):
    threads = [
        threading.Thread(
            target=lambda: [host._pith_snapshot("concurrent") for _ in range(20)]
        )
        for _ in range(6)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20)
    assert not any(thread.is_alive() for thread in threads)
    records = _records(host.PITH_SNAPSHOT_PATH)
    assert len(records) == 120
    assert all(record["reason"] == "concurrent" for record in records)


def test_snapshot_interval_has_safety_floor(host):
    assert host.PITH_SNAPSHOT_INTERVAL_SECS >= 30.0


def test_fsynced_samples_survive_sigkill():
    tmp = tempfile.mkdtemp(prefix="cc_host_d4_sigkill_")
    try:
        output = os.path.join(tmp, "pith_metrics.jsonl")
        driver = (
            "import importlib.util,sys,types,time\n"
            "stub=types.ModuleType('cc_ng_organism')\n"
            "c={'n':0}\n"
            "class M:\n"
            " @staticmethod\n"
            " def snapshot():\n"
            "  c['n']+=7; return {'l1_assemblies':c['n']}\n"
            "stub._PITH_METRICS=M()\n"
            "sys.modules['cc_ng_organism']=stub\n"
            "spec=importlib.util.spec_from_file_location('h',%r)\n"
            "h=importlib.util.module_from_spec(spec); spec.loader.exec_module(h)\n"
            "h.PITH_SNAPSHOT_PATH=%r\n"
            "while True:\n"
            " h._pith_snapshot('interval'); time.sleep(0.2)\n"
        ) % (HOST_PATH, output)
        process = subprocess.Popen(
            [sys.executable, "-c", driver],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            time.sleep(1.2)
            process.send_signal(signal.SIGKILL)
            process.wait(timeout=15)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
        records = _records(output)
        assert len(records) >= 3
        values = [record["counters"]["l1_assemblies"] for record in records]
        assert values == sorted(values)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_host_lifecycle_wires_periodic_and_final_samples():
    with open(HOST_PATH) as stream:
        source = stream.read()
    assert 'name="cc-pith-telemetry"' in source
    assert '_pith_snapshot("shutdown")' in source
    assert "CC Pith config RESOLVED" in source
    assert "gates at startup (raw env)" in source
    assert "os.fsync" in source
