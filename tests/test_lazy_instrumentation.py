"""Tests for deferred (post-import-hook) activation of instrumentations.

The behaviour under test is timing: an instrumentation must be applied no
earlier than ``Netra.init()`` and no later than the first import of the library
it patches.  Most tests drive :func:`register_lazy_instrumentations` with
synthetic trigger modules so they exercise the real wrapt machinery without
depending on which LLM libraries happen to be installed.
"""

import ast
import dataclasses
import importlib
import importlib.util
import io
import logging
import pathlib
import re
import subprocess
import sys
import textwrap
import threading
from typing import Callable, Generator, List, Mapping

import pytest
import wrapt
import wrapt.importer

from netra.instrumentation.instruments import ALL_INSTRUMENTS, DEFAULT_INSTRUMENTS, InstrumentSet, _Origin
from netra.instrumentation.wiring import triggers
from netra.instrumentation.wiring.activation import (
    Activation,
    apply_traceloop_instrumentation,
    build_activations,
    is_distribution_installed,
)
from netra.instrumentation.wiring.deferral import _LEDGER, register_lazy_instrumentations
from netra.instrumentation.wiring.registry import CUSTOM_INSTRUMENTORS, InstrumentorSpec
from netra.instrumentation.wiring.selection import partition_by_origin, select_instrumentations
from netra.instrumentation.wiring.triggers import INSTRUMENT_TRIGGERS, INTENTIONALLY_EAGER_INSTRUMENTS

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)  # type: ignore[misc]
def reset_activation_ledger() -> Generator[None, None, None]:
    """Clear the process-wide ledger between tests.

    The ledger is module scope so the exactly-once invariant holds per process
    (see ``netra.instrumentation.wiring.deferral``).  Tests re-register the same synthetic
    instrument names repeatedly, so without this the second test to use a name
    would find it already claimed.
    """
    _LEDGER.reset()
    yield
    _LEDGER.reset()


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
    monkeypatch.setattr("netra.instrumentation.wiring.deferral._TRIGGERS_BY_NAME", triggers)
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

    with caplog.at_level(logging.ERROR, logger="netra.instrumentation.wiring.activation"):
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

    with caplog.at_level(logging.ERROR, logger="netra.instrumentation.wiring.activation"):
        importlib.import_module(module_name)

    assert calls == ["ran"]


@pytest.mark.thread_safety  # type: ignore[misc]
def test_concurrent_imports_of_a_trigger_activate_exactly_once(
    probe_package: Callable[[str], str], monkeypatch: pytest.MonkeyPatch
) -> None:
    module_name = probe_package("netra_probe_threads")
    dependency_name = probe_package("netra_probe_activation_dependency")
    calls: List[str] = []
    calls_lock = threading.Lock()
    barrier = threading.Barrier(8)

    def record() -> None:
        # Importing from inside the activation is the point of this test: the
        # hook already runs under the trigger module's import lock, so a real
        # instrumentor importing the library it patches is the re-entrant case
        # that could deadlock.
        importlib.import_module(dependency_name)
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


@pytest.mark.thread_safety  # type: ignore[misc]
def test_concurrent_activations_leave_stdout_and_stderr_intact(
    probe_package: Callable[[str], str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # contextlib.redirect_stdout saves the displaced stream per instance, so
    # two threads entering and leaving out of order restore each other's
    # buffers and sys.stdout stays a discarded StringIO for the whole process.
    from netra.instrumentation.wiring.activation import _suppressed_output

    first = probe_package("netra_probe_stdout_one")
    second = probe_package("netra_probe_stdout_two")
    real_stdout, real_stderr = sys.stdout, sys.stderr
    both_inside = threading.Barrier(2)

    def suppress_while(hold_seconds: float) -> Callable[[], None]:
        def run() -> None:
            with _suppressed_output():
                both_inside.wait(timeout=10)
                # Forces the exits to interleave rather than nest cleanly.
                threading.Event().wait(hold_seconds)

        return run

    _register(
        {"FIRST": (first,), "SECOND": (second,)},
        [Activation("FIRST", suppress_while(0.0)), Activation("SECOND", suppress_while(0.3))],
        monkeypatch,
    )

    threads = [
        threading.Thread(target=lambda: importlib.import_module(first)),
        threading.Thread(target=lambda: importlib.import_module(second)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert sys.stdout is real_stdout, "sys.stdout was left pointing at a discarded buffer"
    assert sys.stderr is real_stderr, "sys.stderr was left pointing at a discarded buffer"


def test_suppressed_output_restores_streams_after_a_failure() -> None:
    from netra.instrumentation.wiring.activation import _suppressed_output

    real_stdout, real_stderr = sys.stdout, sys.stderr

    with pytest.raises(RuntimeError):
        with _suppressed_output():
            raise RuntimeError("instrumentor blew up mid-activation")

    assert (sys.stdout, sys.stderr) == (real_stdout, real_stderr)


def test_suppressed_output_swallows_writes_inside_the_block() -> None:
    from netra.instrumentation.wiring.activation import _suppressed_output

    with _suppressed_output():
        print("traceloop warning that must not reach the client")
        print("and one on stderr", file=sys.stderr)
        captured = sys.stdout

    assert isinstance(captured, io.StringIO)
    assert "must not reach the client" in captured.getvalue()


def test_registering_twice_does_not_activate_twice(
    probe_package: Callable[[str], str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The ledger is process-wide, so a second registration is a no-op rather
    # than a second application. The traceloop path has no equivalent of
    # is_instrumented_by_opentelemetry to catch a repeat.
    module_name = probe_package("netra_probe_double_register")
    calls: List[str] = []

    _register({"PROBE": (module_name,)}, [Activation("PROBE", lambda: calls.append("ran"))], monkeypatch)
    register_lazy_instrumentations([Activation("PROBE", lambda: calls.append("ran"))])

    importlib.import_module(module_name)

    assert calls == ["ran"]


def test_every_default_instrument_has_a_trigger() -> None:
    missing = sorted(member.name for member in DEFAULT_INSTRUMENTS if member not in INSTRUMENT_TRIGGERS)

    assert missing == [], (
        f"No INSTRUMENT_TRIGGERS entry for {', '.join(missing)}. "
        "Without one the instrumentation falls back to eager activation, which "
        "costs the startup latency lazy activation exists to remove."
    )


def test_every_implemented_instrument_is_lazy_or_deliberately_eager() -> None:
    # DEFAULT_INSTRUMENTS is covered above; this covers InstrumentSet.ALL,
    # where a missing trigger drags traceloop (~620 ms) back into init().
    implemented = {
        member for member in ALL_INSTRUMENTS if member in CUSTOM_INSTRUMENTORS or member.origin is _Origin.TRACELOOP
    }
    missing = sorted(
        member.name
        for member in implemented
        if member not in INSTRUMENT_TRIGGERS and member not in INTENTIONALLY_EAGER_INSTRUMENTS
    )

    assert missing == [], (
        f"No INSTRUMENT_TRIGGERS entry for {', '.join(missing)}, and not listed in "
        "INTENTIONALLY_EAGER_INSTRUMENTS. Add a trigger module, or record the "
        "exemption so an eager activation is a decision rather than drift."
    )


def test_no_trigger_names_an_instrument_that_can_never_activate() -> None:
    # A trigger row for an instrument with no implementation reads as support
    # the SDK does not have: build_activations never emits an activation for
    # it, so the hook would fire against nothing.
    unreachable = sorted(
        member.name
        for member in INSTRUMENT_TRIGGERS
        if member.origin is _Origin.CUSTOM and member not in CUSTOM_INSTRUMENTORS
    )

    assert unreachable == [], f"{unreachable} have trigger modules but no CUSTOM_INSTRUMENTORS entry"


@pytest.mark.parametrize(  # type: ignore[misc]
    "instrument",
    sorted(CUSTOM_INSTRUMENTORS, key=lambda member: member.name),
)
def test_registered_instrumentor_module_resolves(instrument: InstrumentSet) -> None:
    # InstrumentorSpec.module is a string, so a wrong path is not a NameError at
    # import time -- it surfaces as an ImportError inside run_activation, which
    # deliberately swallows it so a broken instrumentor cannot break the client's
    # import. The instrumentation then silently produces no telemetry. Nothing
    # else in the suite would notice, so check the paths resolve.
    unresolvable = []
    for spec in CUSTOM_INSTRUMENTORS[instrument]:
        if not all(is_distribution_installed(dist) for dist in spec.required_distributions):
            continue  # candidate for a library this environment does not have
        try:
            if importlib.util.find_spec(spec.module) is None:
                unresolvable.append(spec.module)
        except (ImportError, ModuleNotFoundError, ValueError):
            unresolvable.append(spec.module)

    assert unresolvable == [], (
        f"{instrument.name} names {unresolvable}, which does not resolve to a module. "
        "Activation would fail silently and the instrumentation would emit nothing."
    )


@pytest.mark.parametrize(  # type: ignore[misc]
    "instrument",
    sorted(CUSTOM_INSTRUMENTORS, key=lambda member: member.name),
)
def test_registered_instrumentor_class_exists_in_its_module(instrument: InstrumentSet) -> None:
    # Same failure mode one level down: the module resolves but the class name
    # is stale, so getattr fails inside the suppressed activation path.
    missing = []
    for spec in CUSTOM_INSTRUMENTORS[instrument]:
        if not all(is_distribution_installed(dist) for dist in spec.required_distributions):
            continue
        try:
            module = importlib.import_module(spec.module)
        except Exception:
            pytest.skip(f"{spec.module} is not importable in this environment")
        if not hasattr(module, spec.class_name):
            missing.append(f"{spec.module}.{spec.class_name}")

    assert missing == [], f"{instrument.name} names {missing}, which do not exist."


@pytest.mark.parametrize(  # type: ignore[misc]
    "instrument",
    sorted(CUSTOM_INSTRUMENTORS, key=lambda member: member.name),
)
def test_registered_gates_name_distributions_not_modules(instrument: InstrumentSet) -> None:
    # is_distribution_installed matches installed distribution metadata, so a
    # gate naming an import path instead of a distribution can never match and
    # silently disables its instrumentor in every environment. Stdlib-backed
    # instrumentors use an empty gate instead.
    stdlib_module_names = set(sys.stdlib_module_names)

    offenders = [
        name
        for spec in CUSTOM_INSTRUMENTORS[instrument]
        for name in spec.required_distributions
        if name.split(".")[0] in stdlib_module_names and not is_distribution_installed(name)
    ]

    assert offenders == [], (
        f"{instrument.name} is gated on {offenders}, which name standard-library "
        "modules rather than installed distributions. Use () to always apply."
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


def test_no_instrument_appears_twice_in_the_trigger_table() -> None:
    # A repeated key is invisible at runtime — the later row silently wins — and
    # pyflakes' F601 only fires when the repeated values *differ*, so a copy
    # with identical triggers passes lint.  Parsing the source is the only way
    # to see the rows the dict literal threw away.
    source = pathlib.Path(triggers.__file__).read_text(encoding="utf-8")
    table = next(
        node.value
        for node in ast.parse(source).body
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "INSTRUMENT_TRIGGERS"
    )

    keys = [ast.unparse(key) for key in table.keys if key is not None]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})

    assert duplicates == [], f"{duplicates} appear twice in INSTRUMENT_TRIGGERS; the later row wins silently"


def test_no_trigger_is_the_parent_namespace_of_another_trigger() -> None:
    triggers = {trigger for values in INSTRUMENT_TRIGGERS.values() for trigger in values}
    overlapping = sorted(
        trigger
        for trigger in triggers
        if any(other != trigger and other.startswith(f"{trigger}.") for other in triggers)
    )

    assert overlapping == [], f"{overlapping} shadow a more specific trigger and may fire mid-import"


def test_naming_an_unimplemented_instrument_warns(caplog: pytest.LogCaptureFixture) -> None:
    # PYRAMID is selectable but ships no instrumentor, so enabling it does
    # nothing.  A caller who typed its name should not have to raise the log
    # level to find that out.
    with caplog.at_level(logging.WARNING, logger="netra.instrumentation.wiring.activation"):
        build_activations(select_instrumentations({InstrumentSet.PYRAMID}, None), should_enrich_metrics=True)

    assert "PYRAMID" in caplog.text


def test_expanding_all_does_not_warn_about_unimplemented_instruments(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # InstrumentSet.ALL sweeps in six unimplemented members every time; warning
    # about them would make the warning above worthless noise.
    with caplog.at_level(logging.WARNING, logger="netra.instrumentation.wiring.activation"):
        build_activations(select_instrumentations({InstrumentSet.ALL}, None), should_enrich_metrics=True)

    assert "No instrumentor registered" not in caplog.text


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


def test_instrumentor_spec_declares_no_mapping_field_default() -> None:
    # Python 3.11's dataclasses rejects any ``default=`` whose type is
    # unhashable, and ``mappingproxy`` only became hashable in 3.12 - so a
    # bare mapping default here makes ``import netra`` fail outright on 3.11.
    offenders = [
        spec_field.name
        for spec_field in dataclasses.fields(InstrumentorSpec)
        if spec_field.default is not dataclasses.MISSING and isinstance(spec_field.default, Mapping)
    ]

    assert offenders == [], f"use default_factory for {offenders}"


def test_traceloop_warnings_do_not_reach_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    # Activating one instrument at a time makes traceloop's "no valid
    # instruments set" warning routine, and it would print into whatever the
    # client was doing when they imported their library.
    apply_traceloop_instrumentation("ALEPHALPHA", should_enrich_metrics=True)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_unknown_traceloop_instrument_is_logged_and_skipped(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        apply_traceloop_instrumentation("NOT_A_REAL_INSTRUMENT", should_enrich_metrics=True)

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
    expected = {member.name for member in DEFAULT_INSTRUMENTS if member.origin is _Origin.TRACELOOP}

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


def test_blocking_a_traceloop_instrument_never_enables_another() -> None:
    # Regression: an explicit request naming only Netra-backed instruments used
    # to fall through to "every installed traceloop instrument" as soon as a
    # block list was present, so blocking Anthropic enabled langchain, bedrock,
    # vertexai and the rest.
    selection = select_instrumentations({InstrumentSet.OPENAI}, {InstrumentSet.ANTHROPIC})

    assert selection.traceloop_instrument_names == frozenset()
    assert selection.custom_instruments == frozenset({InstrumentSet.OPENAI})


def test_blocking_alongside_a_traceloop_request_keeps_only_that_request() -> None:
    selection = select_instrumentations({InstrumentSet.ANTHROPIC, InstrumentSet.LANGCHAIN}, {InstrumentSet.LANGCHAIN})

    assert selection.traceloop_instrument_names == frozenset({"ANTHROPIC"})


def test_selection_never_imports_traceloop() -> None:
    result = _run_in_subprocess(
        """
        import sys
        from netra.instrumentation.instruments import InstrumentSet
        from netra.instrumentation.wiring.selection import select_instrumentations

        for requested, blocked in (
            (None, None),
            ({InstrumentSet.ALL}, None),
            ({InstrumentSet.OPENAI}, {InstrumentSet.ANTHROPIC}),
            (None, {InstrumentSet.LANGCHAIN}),
        ):
            select_instrumentations(requested, blocked)

        assert "traceloop.sdk" not in sys.modules, "selection imported traceloop"
        print("ok")
        """
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_netra_owned_instrumentations_are_never_delegated_to_traceloop() -> None:
    # Netra ships its own OpenAI/Groq/... instrumentors; traceloop's versions
    # would double-instrument the same call sites.  Checked against the
    # registry rather than a hand-kept list of names, which could only ever
    # agree with itself.
    enabled = _enabled_traceloop_names({InstrumentSet.ALL})
    netra_owned = {instrument.name for instrument in CUSTOM_INSTRUMENTORS}

    assert enabled.isdisjoint(netra_owned)


# The OTel scope name of every instrumentor reaches the backend on each span and
# dashboards key off it, so it is a wire contract. It used to be `__name__`,
# which meant moving the vendor packages under `libraries/` silently rewrote all
# 24 of them. These pin it so the next move cannot.

_LIBRARIES_DIR = pathlib.Path(__file__).parent.parent / "netra" / "instrumentation" / "libraries"


def _tracer_name_constant(package: pathlib.Path) -> str | None:
    """Read the literal assigned to ``_TRACER_NAME`` in *package*, without importing it."""
    for source_file in sorted(package.rglob("*.py")):
        tree = ast.parse(source_file.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                if any(isinstance(t, ast.Name) and t.id == "_TRACER_NAME" for t in node.targets):
                    return str(node.value.value)
    return None


@pytest.mark.parametrize(  # type: ignore[misc]
    "package",
    sorted((p for p in _LIBRARIES_DIR.iterdir() if p.is_dir() and not p.name.startswith("_")), key=lambda p: p.name),
    ids=lambda p: p.name,
)
def test_exported_scope_name_is_pinned_to_the_library_not_the_file_path(package: pathlib.Path) -> None:
    scope = _tracer_name_constant(package)
    if scope is None:
        pytest.skip(f"{package.name} creates no tracer of its own")

    assert scope == f"netra.instrumentation.{package.name}", (
        f"{package.name} exports scope {scope!r}. The contract is "
        f"'netra.instrumentation.{package.name}' regardless of where the package sits on disk -- "
        "changing it breaks every backend query and dashboard filtering on scope name."
    )


def test_no_instrumentor_derives_its_scope_name_from_its_module_path() -> None:
    # get_tracer(__name__) is how the scope name became coupled to the directory
    # layout in the first place.
    offenders = [
        str(source_file.relative_to(_LIBRARIES_DIR.parent.parent.parent))
        for source_file in sorted(_LIBRARIES_DIR.rglob("*.py"))
        if re.search(r"get_(?:tracer|meter)\(\s*\n?\s*__name__", source_file.read_text())
    ]

    assert offenders == [], (
        f"{offenders} pass __name__ to get_tracer/get_meter. Use the package's _TRACER_NAME "
        "constant so the exported scope survives the file being moved."
    )
