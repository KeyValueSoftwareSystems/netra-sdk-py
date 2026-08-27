"""Deciding which instrumentations run, and applying them at the right moment.

``Netra.init()`` calls :func:`netra.instrumentation.init_instrumentations` once,
which walks these five modules in order:

* ``selection``  — requested/blocked sets to the instrumentations to enable
* ``registry``   — how to build each instrumentor Netra provides itself
* ``triggers``   — which library import each instrumentation waits on
* ``activation`` — applying one instrumentation, whoever implements it
* ``deferral``   — holding each one until its library is imported

The split exists because *what* to instrument and *when* to instrument it have
different failure modes. Getting the first wrong disables telemetry loudly; the
second, silently. Each module's own docstring is the authoritative spec for its
half -- read those before changing anything here.
"""
