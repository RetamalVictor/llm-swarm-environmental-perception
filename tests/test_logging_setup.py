"""Tests for logging_setup: verbosity mapping, UTF-8 forcing, CLI wiring."""

import io
import logging

import pytest

from swarm_perception.logging_setup import setup_logging


@pytest.fixture(autouse=True)
def restore_root_logger():
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    yield
    root.handlers[:], root.level = saved_handlers, saved_level


@pytest.mark.parametrize(
    ("verbosity", "level"),
    [
        (0, logging.WARNING),
        (1, logging.INFO),
        (2, logging.DEBUG),
        (5, logging.DEBUG),  # anything past -vv stays DEBUG
    ],
)
def test_verbosity_maps_to_level(verbosity: int, level: int) -> None:
    setup_logging(verbosity, stream=io.StringIO())
    assert logging.getLogger().level == level


def test_repeated_setup_does_not_stack_handlers() -> None:
    setup_logging(0, stream=io.StringIO())
    setup_logging(1, stream=io.StringIO())
    assert len(logging.getLogger().handlers) == 1


def test_stream_is_reconfigured_to_utf8() -> None:
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="ascii", errors="strict")

    setup_logging(0, stream=stream)
    logging.getLogger("t").warning("snowman ☃")

    handler_stream = logging.getLogger().handlers[0].stream  # type: ignore[attr-defined]
    assert handler_stream.encoding == "utf-8"
    assert handler_stream.errors == "replace"
    handler_stream.flush()
    assert "snowman ☃".encode() in raw.getvalue()


def test_stream_without_reconfigure_is_wrapped_via_buffer() -> None:
    class BufferOnlyStream:
        """Text stream exposing only a byte buffer (no ``reconfigure``)."""

        def __init__(self) -> None:
            self.buffer = io.BytesIO()

    setup_logging(0, stream=BufferOnlyStream())  # type: ignore[arg-type]
    logging.getLogger("t").warning("wrapped ☃")

    handler_stream = logging.getLogger().handlers[0].stream  # type: ignore[attr-defined]
    assert isinstance(handler_stream, io.TextIOWrapper)
    assert handler_stream.encoding == "utf-8"
    assert handler_stream.errors == "replace"


def test_cli_verbose_flags_change_log_level(tmp_path) -> None:
    from swarm_perception.cli import main

    # The missing-config error path exits after logging is configured, so the
    # root level shows what each flag set without running a simulation.
    missing = str(tmp_path / "nope.yaml")
    for argv, level in [
        ([missing], logging.WARNING),
        (["-v", missing], logging.INFO),
        (["-vv", missing], logging.DEBUG),
    ]:
        with pytest.raises(SystemExit):
            main(argv)
        assert logging.getLogger().level == level
