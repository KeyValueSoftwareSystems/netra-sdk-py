"""The logger every module of the Netra SDK uses.

Get one per module with ``logger = get_logger(__name__)``.

``debug``, ``info``, ``warning``, ``error`` and ``exception`` follow ``debug_mode``: outside it they
are silenced (``configure_package_logging`` sets the ``netra`` package logger to CRITICAL), because
most of what the SDK logs is internal detail a developer cannot act on.

``notice`` is the exception. It is for problems the developer has to fix: a missing endpoint, a
rejected export, an invalid argument to a public method, a failed API call. It is shown whether or
not ``debug_mode`` is on, every time it is called. Notices go through the ``netra.notice`` logger
and propagate to the root logger, so they flow through the application's own logging setup; when
the application has configured none, Python's last-resort handler prints them to stderr.
To silence them: ``logging.getLogger("netra.notice").setLevel(logging.CRITICAL)``.
"""

import logging
from enum import IntEnum
from typing import Any, Mapping, Optional

NOTICE_LOGGER_NAME = "netra.notice"


class NoticeLevel(IntEnum):
    """Severities a notice can carry. Nothing below WARNING: the notice logger would drop it."""

    WARNING = logging.WARNING
    ERROR = logging.ERROR


_notice_logger = logging.getLogger(NOTICE_LOGGER_NAME)
# Without an explicit level this logger would inherit the CRITICAL level that silences the rest of
# the "netra" package outside debug mode. Only set it when unset, so a level chosen by the
# application before importing netra is kept.
if _notice_logger.level == logging.NOTSET:
    _notice_logger.setLevel(logging.WARNING)

# Each method below calls logging.Logger.log itself, so one extra frame sits between the SDK call
# site and the logging module; skipping it keeps a record's filename/lineno at the call site.
_WRAPPER_FRAMES = 1

# Any: the logging module accepts a bool, an exception instance or an exc_info tuple here, and its
# own stubs type the parameter loosely too.
_ExcInfo = Any


class NetraLogger:
    """Per-module SDK logger: stdlib-compatible level methods plus an always-visible ``notice``."""

    __slots__ = ("_logger",)

    def __init__(self, name: str) -> None:
        """Wrap the standard library logger called ``name``.

        Args:
            name: Logger name, normally the calling module's ``__name__`` (e.g.
                ``"netra.tracer"``). Names under ``netra`` inherit the package's debug_mode level.

        Returns:
            None.
        """
        self._logger = logging.getLogger(name)

    @property
    def name(self) -> str:
        """The wrapped logger's name.

        Returns:
            The name passed to the constructor.
        """
        return self._logger.name

    def debug(
        self,
        msg: str,
        *args: object,
        exc_info: _ExcInfo = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Optional[Mapping[str, object]] = None,
    ) -> None:
        """Log at DEBUG. Shown only when debug_mode is on.

        Args:
            msg: Message, %-format style; formatted with ``args`` only if the record is emitted.
            *args: Values substituted into ``msg``.
            exc_info: Exception information to attach: ``True`` for the exception being handled,
                an exception instance, or an ``(type, value, traceback)`` tuple.
            stack_info: Attach the current call stack to the record.
            stacklevel: How many frames above the caller to attribute the record to; 1 is the
                caller of this method.
            extra: Additional attributes set on the ``LogRecord``.

        Returns:
            None.
        """
        self._logger.log(
            logging.DEBUG,
            msg,
            *args,
            exc_info=exc_info,
            stack_info=stack_info,
            stacklevel=stacklevel + _WRAPPER_FRAMES,
            extra=extra,
        )

    def info(
        self,
        msg: str,
        *args: object,
        exc_info: _ExcInfo = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Optional[Mapping[str, object]] = None,
    ) -> None:
        """Log at INFO. Shown only when debug_mode is on.

        Args:
            msg: Message, %-format style; formatted with ``args`` only if the record is emitted.
            *args: Values substituted into ``msg``.
            exc_info: Exception information to attach: ``True`` for the exception being handled,
                an exception instance, or an ``(type, value, traceback)`` tuple.
            stack_info: Attach the current call stack to the record.
            stacklevel: How many frames above the caller to attribute the record to; 1 is the
                caller of this method.
            extra: Additional attributes set on the ``LogRecord``.

        Returns:
            None.
        """
        self._logger.log(
            logging.INFO,
            msg,
            *args,
            exc_info=exc_info,
            stack_info=stack_info,
            stacklevel=stacklevel + _WRAPPER_FRAMES,
            extra=extra,
        )

    def warning(
        self,
        msg: str,
        *args: object,
        exc_info: _ExcInfo = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Optional[Mapping[str, object]] = None,
    ) -> None:
        """Log at WARNING. Shown only when debug_mode is on.

        Args:
            msg: Message, %-format style; formatted with ``args`` only if the record is emitted.
            *args: Values substituted into ``msg``.
            exc_info: Exception information to attach: ``True`` for the exception being handled,
                an exception instance, or an ``(type, value, traceback)`` tuple.
            stack_info: Attach the current call stack to the record.
            stacklevel: How many frames above the caller to attribute the record to; 1 is the
                caller of this method.
            extra: Additional attributes set on the ``LogRecord``.

        Returns:
            None.
        """
        self._logger.log(
            logging.WARNING,
            msg,
            *args,
            exc_info=exc_info,
            stack_info=stack_info,
            stacklevel=stacklevel + _WRAPPER_FRAMES,
            extra=extra,
        )

    def error(
        self,
        msg: str,
        *args: object,
        exc_info: _ExcInfo = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Optional[Mapping[str, object]] = None,
    ) -> None:
        """Log at ERROR. Shown only when debug_mode is on.

        Args:
            msg: Message, %-format style; formatted with ``args`` only if the record is emitted.
            *args: Values substituted into ``msg``.
            exc_info: Exception information to attach: ``True`` for the exception being handled,
                an exception instance, or an ``(type, value, traceback)`` tuple.
            stack_info: Attach the current call stack to the record.
            stacklevel: How many frames above the caller to attribute the record to; 1 is the
                caller of this method.
            extra: Additional attributes set on the ``LogRecord``.

        Returns:
            None.
        """
        self._logger.log(
            logging.ERROR,
            msg,
            *args,
            exc_info=exc_info,
            stack_info=stack_info,
            stacklevel=stacklevel + _WRAPPER_FRAMES,
            extra=extra,
        )

    def exception(
        self,
        msg: str,
        *args: object,
        exc_info: _ExcInfo = True,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Optional[Mapping[str, object]] = None,
    ) -> None:
        """Log at ERROR with the current exception's traceback. Shown only when debug_mode is on.

        Call it from an ``except`` block: ``exc_info`` defaults to ``True``.

        Args:
            msg: Message, %-format style; formatted with ``args`` only if the record is emitted.
            *args: Values substituted into ``msg``.
            exc_info: Exception information to attach: ``True`` for the exception being handled,
                an exception instance, or an ``(type, value, traceback)`` tuple.
            stack_info: Attach the current call stack to the record.
            stacklevel: How many frames above the caller to attribute the record to; 1 is the
                caller of this method.
            extra: Additional attributes set on the ``LogRecord``.

        Returns:
            None.
        """
        self._logger.log(
            logging.ERROR,
            msg,
            *args,
            exc_info=exc_info,
            stack_info=stack_info,
            stacklevel=stacklevel + _WRAPPER_FRAMES,
            extra=extra,
        )

    def log(
        self,
        level: int,
        msg: str,
        *args: object,
        exc_info: _ExcInfo = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Optional[Mapping[str, object]] = None,
    ) -> None:
        """Log at an explicit ``level``. Shown only when debug_mode is on.

        Args:
            level: A ``logging`` level, e.g. ``logging.WARNING``.
            msg: Message, %-format style; formatted with ``args`` only if the record is emitted.
            *args: Values substituted into ``msg``.
            exc_info: Exception information to attach: ``True`` for the exception being handled,
                an exception instance, or an ``(type, value, traceback)`` tuple.
            stack_info: Attach the current call stack to the record.
            stacklevel: How many frames above the caller to attribute the record to; 1 is the
                caller of this method.
            extra: Additional attributes set on the ``LogRecord``.

        Returns:
            None.
        """
        self._logger.log(
            level,
            msg,
            *args,
            exc_info=exc_info,
            stack_info=stack_info,
            stacklevel=stacklevel + _WRAPPER_FRAMES,
            extra=extra,
        )

    def notice(
        self,
        msg: str,
        *args: object,
        level: NoticeLevel = NoticeLevel.WARNING,
        exc_info: _ExcInfo = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Optional[Mapping[str, object]] = None,
    ) -> None:
        """Log a problem the developer has to fix; shown whether or not debug_mode is on.

        The record goes to the ``netra.notice`` logger, not this logger's own name, so an application
        can route or silence notices separately. Every call is logged: repeats are not collapsed.

        Args:
            msg: Message, %-format style; formatted with ``args`` only if the record is emitted.
            *args: Values substituted into ``msg``.
            exc_info: Exception information to attach: ``True`` for the exception being handled,
                an exception instance, or an ``(type, value, traceback)`` tuple.
            stack_info: Attach the current call stack to the record.
            stacklevel: How many frames above the caller to attribute the record to; 1 is the
                caller of this method.
            extra: Additional attributes set on the ``LogRecord``.
            level: ``NoticeLevel.WARNING`` (default) for something to fix, ``NoticeLevel.ERROR``
                for a failure such as a rejected API call.

        Returns:
            None.
        """
        _notice_logger.log(
            level,
            msg,
            *args,
            exc_info=exc_info,
            stack_info=stack_info,
            stacklevel=stacklevel + _WRAPPER_FRAMES,
            extra=extra,
        )


def get_logger(name: str) -> NetraLogger:
    """Return the SDK logger for a module.

    Args:
        name: Logger name; pass the calling module's ``__name__``.

    Returns:
        A ``NetraLogger`` wrapping ``logging.getLogger(name)``. Loggers are cheap wrappers: calling
        this twice with one name returns two objects that log through the same stdlib logger.
    """
    return NetraLogger(name)
