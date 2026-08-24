"""Tests for deferred (post-import-hook) activation of instrumentations.

The behaviour under test is timing: an instrumentation must be applied no
earlier than ``Netra.init()`` and no later than the first import of the library
it patches.  Most tests drive :func:`register_lazy_instrumentations` with
synthetic trigger modules so they exercise the real wrapt machinery without
depending on which LLM libraries happen to be installed.
"""

import importlib
import importlib.util
import logging
import subprocess
import sys
import textwrap
import threading
from typing import Callable, Generator, List

import pytest
import wrapt
import wrapt.importer

from netra.instrumentation.activation import Activation, apply_traceloop_instrumentation
from netra.instrumentation.instruments import DEFAULT_INSTRUMENTS, InstrumentSet, _Origin
from netra.instrumentation.lazy import register_lazy_instrumentations
from netra.instrumentation.registry import CUSTOM_INSTRUMENTORS
from netra.instrumentation.selection import (
    TRACELOOP_INSTRUMENTS_REPLACED_BY_NETRA,
    partition_by_origin,
    select_instrumentations,
)
from netra.instrumentation.triggers import INSTRUMENT_TRIGGERS

pytestmark = pytest.mark.unit


@pytest.fixture  # type: ignore[misc]
def probe_package(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> Generator[Callable[[str], str], None, None]:
    """Create importable throwaway modules and undo every trace of them.

    Yields a factory taking a bare module name and returning that name after
    writing the module to an importable directory.  On teardown the modules,
    their wrapt post-import hooks and the trigger overrides are all removed, so
    tests cannot leak activation state into one another.
    """
    root = tmp_path  # type: ignore[assignment]
    monkeypatch.syspath_prepend(str(root))
    created: List[str] = []

    def make(name: str) -> str:
        (root / f"{name}.py").write_text("VALUE = 1\n")  # type: ignore[operator]
        created.append(name)
        return name

    yield make

    for name in created:
        sys.modules.pop(name, None)
        wrapt.importer._post_import_hooks.pop(name, None)
    importlib.invalidate_caches()


def _register(triggers: dict, activations: List[Activation], monkeypatch: pytest.MonkeyPatch) -> None:
    """Register *activations* against a trigger table containing only *triggers*."""
    monkeypatch.setattr("netra.instrumentation.lazy._TRIGGERS_BY_NAME", triggers)
    register_lazy_instrumentations(activations)


def test_instrumentation_is_not_applied_until_library_is_imported(
    probe_package: Callable[[str], str], monkeypatch: pytest.MonkeyPatch
) -> None:
    module_name = probe_package("netra_probe_deferred")
    calls: List[str] = []

    _register(
        {"PROBE": (module_name,)},
        [Activation("PROBE", lambda: calls.append("ran"))],
        monkeypatch,
    )

    assert calls == [], "activation ran before the trigger module was imported"

    importlib.import_module(module_name)

    assert calls == ["ran"]


def test_activation_fires_immediately_when_library_already_imported(
    probe_package: Callable[[str], str], monkeypatch: pytest.MonkeyPatch
) -> None:
    module_name = probe_package("netra_probe_preimported")
    importlib.import_module(module_name)
    calls: List[str] = []

    _register(
        {"PROBE": (module_name,)},
        [Activation("PROBE", lambda: calls.append("ran"))],
        monkeypatch,
    )

    # Synchronously during registration — the client may well have imported
    # their LLM library before calling Netra.init().
    assert calls == ["ran"]


def test_activation_runs_once_when_two_triggers_map_to_one_instrument(
    probe_package: Callable[[str], str], monkeypatch: pytest.MonkeyPatch
) -> None:
    first = probe_package("netra_probe_trigger_one")
    second = probe_package("netra_probe_trigger_two")
    calls: List[str] = []

    _register(
        {"PROBE": (first, second)},
        [Activation("PROBE", lambda: calls.append("ran"))],
        monkeypatch,
    )

    importlib.import_module(first)
    importlib.import_module(second)

    assert calls == ["ran"], "an instrument reachable from two triggers activated twice"


def test_instrument_without_trigger_mapping_is_activated_eagerly(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: List[str] = []

    _register({}, [Activation("UNMAPPED", lambda: calls.append("ran"))], monkeypatch)

    # An incomplete trigger table must cost startup latency, never telemetry.
    assert calls == ["ran"]


def test_activation_failure_is_logged_and_does_not_break_the_import(
    probe_package: Callable[[str], str],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    module_name = probe_package("netra_probe_failing")

    def explode() -> None:
        raise RuntimeError("instrumentor blew up")

    _register({"PROBE": (module_name,)}, [Activation("PROBE", explode)], monkeypatch)

    with caplog.at_level(logging.ERROR, logger="netra.instrumentation.activation"):
        module = importlib.import_module(module_name)

    assert module.VALUE == 1, "a failing instrumentor broke the client's own import"
    assert "PROBE" in caplog.text
    assert "instrumentor blew up" in caplog.text


def test_a_failing_activation_does_not_prevent_the_next_one(
    probe_package: Callable[[str], str], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    module_name = probe_package("netra_probe_partial_failure")
    calls: List[str] = []

    def explode() -> None:
        raise RuntimeError("boom")

    _register(
        {"FIRST": (module_name,), "SECOND": (module_name,)},
        [Activation("FIRST", explode), Activation("SECOND", lambda: calls.append("ran"))],
        monkeypatch,
    )

    with caplog.at_level(logging.ERROR, logger="netra.instrumentation.activation"):
        importlib.import_module(module_name)

    assert calls == ["ran"]


@pytest.mark.thread_safety  # type: ignore[misc]
def test_concurrent_imports_of_a_trigger_activate_exactly_once(
    probe_package: Callable[[str], str], monkeypatch: pytest.MonkeyPatch
) -> None:
    module_name = probe_package("netra_probe_threads")
    calls: List[str] = []
    calls_lock = threading.Lock()
    barrier = threading.Barrier(8)

    def record() -> None:
        with calls_lock:
            calls.append("ran")

    _register({"PROBE": (module_name,)}, [Activation("PROBE", record)], monkeypatch)

    errors: List[BaseException] = []

    def worker() -> None:
        try:
            barrier.wait(timeout=10)
            importlib.import_module(module_name)
        except BaseException as exc:  # noqa: BLE001 - re-raised via assertion below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        # A deadlock here is the failure this test exists to catch: hooks run
        # while an import lock is held, and activation itself imports.
        thread.join(timeout=30)
        assert not thread.is_alive(), "importing a trigger module concurrently deadlocked"

    assert errors == []
    assert calls == ["ran"]


def test_every_default_instrument_has_a_trigger() -> None:
    missing = sorted(member.name for member in DEFAULT_INSTRUMENTS if member not in INSTRUMENT_TRIGGERS)

    assert missing == [], (
        f"No INSTRUMENT_TRIGGERS entry for {', '.join(missing)}. "
        "Without one the instrumentation falls back to eager activation, which "
        "costs the startup latency lazy activation exists to remove."
    )


@pytest.mark.parametrize(  # type: ignore[misc]
    "trigger",
    sorted({trigger for triggers in INSTRUMENT_TRIGGERS.values() for trigger in triggers}),
)
def test_trigger_module_is_a_real_module_when_installed(trigger: str) -> None:
    try:
        spec = importlib.util.find_spec(trigger)
    except (ImportError, ValueError):
        pytest.skip(f"{trigger} is not importable in this environment")

    if spec is None:
        pytest.skip(f"{trigger} is not installed in this environment")

    # A namespace package has no loader.  Hooking one fires while the real
    # subpackage is still mid-import, so the trigger must name the real module
    # (livekit.agents, not livekit).
    assert spec.loader is not None, f"{trigger} is a namespace package; the trigger must name the real module"


def test_no_trigger_is_the_parent_namespace_of_another_trigger() -> None:
    triggers = {trigger for values in INSTRUMENT_TRIGGERS.values() for trigger in values}
    overlapping = sorted(
        trigger
        for trigger in triggers
        if any(other != trigger and other.startswith(f"{trigger}.") for other in triggers)
    )

    assert overlapping == [], f"{overlapping} shadow a more specific trigger and may fire mid-import"


def test_partition_by_origin_splits_traceloop_and_custom_instruments() -> None:
    members = frozenset(member for member in InstrumentSet if member is not InstrumentSet.ALL)

    traceloop_names, custom = partition_by_origin(members)

    assert traceloop_names == {member.name for member in members if member.origin is _Origin.TRACELOOP}
    assert custom == {member for member in members if member.origin is _Origin.CUSTOM}


def test_partition_by_origin_skips_the_all_sentinel() -> None:
    traceloop_names, custom = partition_by_origin(frozenset({InstrumentSet.ALL}))

    assert (traceloop_names, custom) == (set(), set())


def test_every_registered_instrumentor_belongs_to_the_custom_family() -> None:
    # A traceloop-origin member listed here would never be activated: selection
    # routes it to traceloop, which knows nothing about this registry.
    misfiled = sorted(instrument.name for instrument in CUSTOM_INSTRUMENTORS if instrument.origin is not _Origin.CUSTOM)

    assert misfiled == []


def test_every_registered_instrumentor_names_a_distinct_candidate_order() -> None:
    # Candidates are tried in order and the first installed one wins, so a
    # duplicated distribution gate would make the later candidate unreachable.
    for instrument, specs in CUSTOM_INSTRUMENTORS.items():
        gates = [spec.required_distributions for spec in specs]

        assert len(gates) == len(set(gates)), f"{instrument.name} has an unreachable instrumentor candidate"


def test_traceloop_warnings_do_not_reach_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    # Activating one instrument at a time makes traceloop's "no valid
    # instruments set" warning routine, and it would print into whatever the
    # client was doing when they imported their library.
    apply_traceloop_instrumentation("ALEPHALPHA", should_enrich_metrics=True, base64_image_uploader=None)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_unknown_traceloop_instrument_is_logged_and_skipped(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        apply_traceloop_instrumentation("NOT_A_REAL_INSTRUMENT", should_enrich_metrics=True, base64_image_uploader=None)

    assert "NOT_A_REAL_INSTRUMENT" in caplog.text


def _run_in_subprocess(body: str) -> subprocess.CompletedProcess:
    """Run *body* in a clean interpreter; sys.modules is dirty in-process."""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)],
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_import_netra_does_not_import_traceloop_or_pandas() -> None:
    result = _run_in_subprocess(
        """
        import sys
        import netra
        loaded = [name for name in ("traceloop.sdk", "pandas", "numpy", "aiohttp") if name in sys.modules]
        assert not loaded, f"import netra pulled in {loaded}"
        print("ok")
        """
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_init_does_not_import_the_libraries_it_instruments() -> None:
    result = _run_in_subprocess(
        """
        import sys
        from netra import Netra

        Netra.init(app_name="lazy-init-test")

        loaded = [name for name in ("traceloop.sdk", "pandas", "transformers", "langchain_core") if name in sys.modules]
        assert not loaded, f"Netra.init() pulled in {loaded}"
        print("ok")
        """
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_eager_kill_switch_restores_up_front_instrumentation() -> None:
    result = _run_in_subprocess(
        """
        import os
        import sys

        os.environ["NETRA_EAGER_INSTRUMENTATION"] = "true"
        from netra import Netra

        Netra.init(app_name="eager-init-test")

        assert "traceloop.sdk" in sys.modules, "NETRA_EAGER_INSTRUMENTATION did not restore eager activation"
        print("ok")
        """
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


@pytest.mark.integration  # type: ignore[misc]
def test_openai_is_patched_when_imported_after_init() -> None:
    pytest.importorskip("openai", reason="openai is not installed in this environment")

    result = _run_in_subprocess(
        """
        import sys
        import wrapt
        from netra import Netra

        Netra.init(app_name="lazy-openai-test")
        assert "openai" not in sys.modules, "Netra.init() imported openai eagerly"

        import openai
        from openai.resources.chat.completions import Completions

        assert isinstance(Completions.create, wrapt.BoundFunctionWrapper), "openai was not patched on import"
        print("ok")
        """
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def _enabled_traceloop_names(
    instruments: object = None,
    block_instruments: object = None,
) -> frozenset:
    """Names of the traceloop instruments ``init_instrumentations`` would enable."""
    return select_instrumentations(instruments, block_instruments).traceloop_instrument_names


def test_default_instruments_enable_their_traceloop_members() -> None:
    expected = {
        member.name for member in DEFAULT_INSTRUMENTS if member.origin is _Origin.TRACELOOP
    } - TRACELOOP_INSTRUMENTS_REPLACED_BY_NETRA

    assert _enabled_traceloop_names() == expected


def test_default_instruments_enable_their_netra_owned_members() -> None:
    expected = {member for member in DEFAULT_INSTRUMENTS if member.origin is _Origin.CUSTOM}

    assert select_instrumentations(None, None).custom_instruments == expected


def test_requesting_only_traceloop_instruments_enables_no_netra_instrumentation() -> None:
    assert select_instrumentations({InstrumentSet.ANTHROPIC}, None).custom_instruments == frozenset()


def test_blocking_all_enables_nothing_from_either_family() -> None:
    selection = select_instrumentations({InstrumentSet.ALL}, {InstrumentSet.ALL})

    assert (selection.traceloop_instrument_names, selection.custom_instruments) == (frozenset(), frozenset())


def test_blocking_one_netra_instrument_removes_only_that_one() -> None:
    baseline = select_instrumentations(None, None).custom_instruments

    selection = select_instrumentations(None, {InstrumentSet.OPENAI})

    assert selection.custom_instruments == baseline - {InstrumentSet.OPENAI}


def test_blocking_all_enables_no_traceloop_instrument() -> None:
    assert _enabled_traceloop_names(block_instruments={InstrumentSet.ALL}) == set()
    assert _enabled_traceloop_names({InstrumentSet.ALL}, {InstrumentSet.ALL}) == set()


def test_blocking_one_instrument_removes_only_that_one() -> None:
    baseline = _enabled_traceloop_names()

    assert _enabled_traceloop_names(block_instruments={InstrumentSet.LANGCHAIN}) == baseline - {"LANGCHAIN"}


def test_requesting_only_custom_instruments_enables_no_traceloop_instrument() -> None:
    assert _enabled_traceloop_names({InstrumentSet.OPENAI}) == set()


def test_netra_owned_instrumentations_are_never_delegated_to_traceloop() -> None:
    # Netra ships its own OpenAI/Groq/... instrumentors; traceloop's versions
    # would double-instrument the same call sites.
    enabled = _enabled_traceloop_names({InstrumentSet.ALL})

    assert enabled.isdisjoint(TRACELOOP_INSTRUMENTS_REPLACED_BY_NETRA)
