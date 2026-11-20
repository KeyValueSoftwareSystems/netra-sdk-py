"""OpenTelemetry Google GenAI API instrumentation"""

import os
import sys
import threading
from typing import Any, Collection, Optional

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.trace import Tracer, get_tracer
from wrapt import ObjectProxy, register_post_import_hook, wrap_function_wrapper

from netra.instrumentation.libraries.google_genai.version import __version__
from netra.instrumentation.libraries.google_genai.wrappers import (
    acontent_stream_wrapper,
    acontent_wrapper,
    adecisions_wrapper,
    aimages_wrapper,
    avideos_wrapper,
    content_stream_wrapper,
    content_wrapper,
    decisions_wrapper,
    images_wrapper,
    videos_wrapper,
)
from netra.netra_logger import get_logger

logger = get_logger(__name__)

_instruments = ("google-genai >= 0.1.0",)


# The exported OpenTelemetry scope name is a wire contract -- dashboards and
# backend queries key off it -- so it is pinned here rather than derived from
# ``__name__``, which would silently change the moment this file moves.
_TRACER_NAME = "netra.instrumentation.google_genai"

# The Decisions API (``client.decisions``) only exists in newer google-genai releases.
# Its module is imported lazily by google-genai on first ``client.decisions`` access and
# costs ~0.5 s to import, so it is wrapped from a post-import hook rather than imported
# here: processes that never use decisions never pay for it, and older google-genai
# releases without the module are simply never patched. ``client.decisions`` is a thin
# subclass whose ``create`` calls this generated resource's ``create``, so wrapping the
# base classes covers sync, async and ``with_raw_response`` calls alike.
_DECISIONS_MODULE = "google.genai._gaos.decisions"
_DECISIONS_CLASSES = ("Decisions", "AsyncDecisions")

# wrapt offers no way to withdraw a post-import hook, so it is registered once per
# process and consults the tracer below: None while uninstrumented, so an import
# after ``_uninstrument`` patches nothing.
_decisions_lock = threading.Lock()
_decisions_hook_registered = False
_decisions_tracer: Optional[Tracer] = None


class NetraGoogleGenAiInstrumentor(BaseInstrumentor):  # type: ignore
    """Custom Google GenAI instrumentor for Netra SDK."""

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        """Instrument Google GenAI client methods"""

        try:
            tracer_provider = kwargs.get("tracer_provider")
            tracer = get_tracer(_TRACER_NAME, __version__, tracer_provider)
        except Exception as e:
            logger.error(f"Failed to initialize tracer: {e}")
            return

        try:
            wrap_function_wrapper(
                "google.genai.models",
                "Models.generate_content",
                content_wrapper(tracer),
            )
            wrap_function_wrapper(
                "google.genai.models",
                "AsyncModels.generate_content",
                acontent_wrapper(tracer),
            )
        except Exception as e:
            logger.error(f"Failed to instrument generate_content: {e}")

        try:
            wrap_function_wrapper(
                "google.genai.models",
                "Models.generate_content_stream",
                content_stream_wrapper(tracer),
            )
            wrap_function_wrapper(
                "google.genai.models",
                "AsyncModels.generate_content_stream",
                acontent_stream_wrapper(tracer),
            )
        except Exception as e:
            logger.error(f"Failed to instrument generate_content_stream: {e}")

        try:
            wrap_function_wrapper(
                "google.genai.models",
                "Models.generate_images",
                images_wrapper(tracer),
            )
            wrap_function_wrapper(
                "google.genai.models",
                "AsyncModels.generate_images",
                aimages_wrapper(tracer),
            )
        except Exception as e:
            logger.error(f"Failed to instrument generate_images: {e}")

        try:
            wrap_function_wrapper(
                "google.genai.models",
                "Models.generate_videos",
                videos_wrapper(tracer),
            )
            wrap_function_wrapper(
                "google.genai.models",
                "AsyncModels.generate_videos",
                avideos_wrapper(tracer),
            )
        except Exception as e:
            logger.error(f"Failed to instrument generate_videos: {e}")

        _enable_decisions(tracer)

    def _uninstrument(self, **kwargs: Any) -> None:
        try:
            unwrap("google.genai.models", "Models.generate_content")
            unwrap("google.genai.models", "AsyncModels.generate_content")
        except (AttributeError, ModuleNotFoundError):
            logger.error("Failed to uninstrument generate_content")

        try:
            unwrap("google.genai.models", "Models.generate_content_stream")
            unwrap("google.genai.models", "AsyncModels.generate_content_stream")
        except (AttributeError, ModuleNotFoundError):
            logger.error("Failed to uninstrument generate_content_stream")

        try:
            unwrap("google.genai.models", "Models.generate_images")
            unwrap("google.genai.models", "AsyncModels.generate_images")
        except (AttributeError, ModuleNotFoundError):
            logger.error("Failed to uninstrument generate_images")

        try:
            unwrap("google.genai.models", "Models.generate_videos")
            unwrap("google.genai.models", "AsyncModels.generate_videos")
        except (AttributeError, ModuleNotFoundError):
            logger.error("Failed to uninstrument generate_videos")

        _disable_decisions()


def _enable_decisions(tracer: Tracer) -> None:
    """Patch ``decisions.create`` now if its module is loaded, else as soon as it is imported."""
    global _decisions_hook_registered, _decisions_tracer
    with _decisions_lock:
        _decisions_tracer = tracer
        register = not _decisions_hook_registered
        _decisions_hook_registered = True
    if register:
        # Fires immediately (synchronously) if the module is already imported.
        register_post_import_hook(_wrap_decisions, _DECISIONS_MODULE)
    elif (module := sys.modules.get(_DECISIONS_MODULE)) is not None:
        # Re-instrumented after an uninstrument: the hook already fired and won't fire again.
        _wrap_decisions(module)


def _wrap_decisions(module: Any) -> None:
    """Post-import hook: wrap ``Decisions.create`` and ``AsyncDecisions.create`` on ``module``."""
    tracer = _decisions_tracer
    if tracer is None:
        return
    wrappers = {"Decisions": decisions_wrapper(tracer), "AsyncDecisions": adecisions_wrapper(tracer)}
    for class_name in _DECISIONS_CLASSES:
        try:
            cls = getattr(module, class_name)
            if isinstance(cls.__dict__.get("create"), ObjectProxy):
                continue
            wrap_function_wrapper(cls, "create", wrappers[class_name])
        except Exception as e:
            logger.error(f"Failed to instrument {class_name}.create: {e}")


def _disable_decisions() -> None:
    """Unpatch ``decisions.create`` if it was patched, and stop patching future imports."""
    global _decisions_tracer
    with _decisions_lock:
        _decisions_tracer = None
    module = sys.modules.get(_DECISIONS_MODULE)
    if module is None:
        return
    for class_name in _DECISIONS_CLASSES:
        try:
            unwrap(getattr(module, class_name), "create")
        except Exception as e:
            logger.error(f"Failed to uninstrument {class_name}.create: {e}")
