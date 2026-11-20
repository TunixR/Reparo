"""Log events captured by pytest's ``caplog``."""

import logging

import pytest


def logged_events(
    caplog: pytest.LogCaptureFixture, name: str
) -> list[logging.LogRecord]:
    """The captured records of the ``name`` log event, in order."""
    return [
        record
        for record in caplog.records
        if vars(record).get("otel.event.name") == name
    ]
