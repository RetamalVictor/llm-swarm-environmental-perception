"""Logging configuration for the CLI.

One stream handler on the root logger, forced to UTF-8: Windows consoles
default to legacy code pages, and a lossy or crashing log write must never
take down a benchmark run.
"""

from __future__ import annotations

import io
import logging
import sys
from typing import IO

_LEVELS = {0: logging.WARNING, 1: logging.INFO}


def _utf8_stream(stream: IO[str]) -> IO[str]:
    """Return ``stream`` emitting UTF-8 with replacement for unencodable text.

    Streams exposing ``reconfigure`` (regular console/file streams) are
    reconfigured in place; otherwise the underlying byte ``buffer`` is wrapped
    in a fresh UTF-8 :class:`io.TextIOWrapper`. Purely textual streams with
    neither (e.g. ``io.StringIO``) are returned unchanged — they have no byte
    layer, so there is nothing to re-encode.
    """
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="replace")
        return stream
    buffer = getattr(stream, "buffer", None)
    if buffer is None:
        return stream
    return io.TextIOWrapper(buffer, encoding="utf-8", errors="replace", line_buffering=True)


def setup_logging(verbosity: int, stream: IO[str] | None = None) -> None:
    """Configure root logging for one CLI process.

    Replaces any existing root handlers, so repeated calls (and tests) never
    stack duplicate handlers.

    Args:
        verbosity: Count of ``-v`` flags — 0 → WARNING, 1 → INFO, 2+ → DEBUG.
        stream: Destination stream; defaults to ``sys.stderr``.
    """
    handler = logging.StreamHandler(_utf8_stream(stream if stream is not None else sys.stderr))
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(_LEVELS.get(verbosity, logging.DEBUG))
