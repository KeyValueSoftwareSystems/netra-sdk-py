"""One subpackage per instrumented library.

Each directory here patches exactly one third-party library and follows the
same layout: ``__init__.py`` holds the ``BaseInstrumentor`` subclass wiring the
``wrapt`` patches, ``wrappers.py`` the wrapper factories (``chat_wrapper(tracer)``
style, sync plus ``a``-prefixed async pairs), ``utils.py`` the attribute
extraction, and ``version.py`` a single ``__version__`` pinned to the
instrumented library.

Nothing here is imported at ``Netra.init()`` time. Instrumenting a library means
importing it, so ``wiring.deferral`` holds each one behind a post-import hook
on the library it patches -- see ``netra.instrumentation.wiring`` for the
machinery and ``wiring.registry`` for the table that names these modules as
*strings*.

The exported OpenTelemetry scope name of each instrumentor is pinned to
``netra.instrumentation.<library>`` in its ``_TRACER_NAME`` constant. It is
deliberately not derived from ``__name__``: the scope name is a wire contract,
and this directory has moved once already.
"""
