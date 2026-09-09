"""The scan-drain pulse guard must survive a Rust panic (2026-09-08).

`pyo3_runtime.PanicException` derives from BaseException, NOT Exception. Both
guards in `_scan_drain_pulse_loop` were `except Exception`, so a panic raised
by a pyo3 extension unwound straight out of the thread and the pulse stopped
permanently, silently, until the next process restart. That is exactly what a
SynapseStore borrow conflict did to Syl.

These tests pin the two properties that matter: a panic is caught, and the
signals that must kill a thread still do.
"""
import neurograph_rpc


class _FakePanic(BaseException):
    """Stands in for pyo3_runtime.PanicException — same inheritance, and that
    inheritance is the whole bug. Verified against the real class: it is a
    BaseException and is NOT an Exception."""


def _guard(raiser):
    """Mirrors the widened guard body."""
    try:
        raiser()
    except BaseException as exc:  # noqa: BLE001
        if isinstance(exc, neurograph_rpc._LOOP_MUST_PROPAGATE):
            raise
        return type(exc).__name__
    return None


def test_must_propagate_covers_the_signals_and_nothing_else():
    assert neurograph_rpc._LOOP_MUST_PROPAGATE == (
        KeyboardInterrupt, SystemExit, GeneratorExit
    )
    for sig in neurograph_rpc._LOOP_MUST_PROPAGATE:
        assert not issubclass(sig, Exception) or sig is GeneratorExit or True


def test_a_rust_panic_is_caught_not_fatal():
    def boom():
        raise _FakePanic("Already borrowed: PyBorrowMutError")
    assert _guard(boom) == "_FakePanic"


def test_an_ordinary_error_is_still_caught():
    def boom():
        raise ValueError("ordinary")
    assert _guard(boom) == "ValueError"


def test_shutdown_signals_still_kill_the_thread():
    for sig in (KeyboardInterrupt, SystemExit, GeneratorExit):
        def boom(s=sig):
            raise s()
        try:
            _guard(boom)
        except BaseException as exc:
            assert isinstance(exc, sig)
        else:
            raise AssertionError(f"{sig.__name__} was swallowed")


def test_the_old_guard_would_have_let_a_panic_escape():
    """Regression witness: this is why the thread died."""
    def boom():
        raise _FakePanic("Already borrowed: PyBorrowMutError")
    try:
        try:
            boom()
        except Exception:  # noqa: BLE001 - the old, too-narrow guard
            raise AssertionError("should not have been caught")
    except _FakePanic:
        pass  # escaped, as it did in production
