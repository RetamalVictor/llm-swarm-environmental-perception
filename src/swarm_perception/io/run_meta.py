"""Reproducibility metadata helpers for :class:`~swarm_perception.io.run_logger.RunLogger`.

Wall-clock timestamps and environment facts (git SHA, package version) are
allowed ONLY in ``run_metadata.json`` — never in event data (D10) — so the
helpers producing them live here, away from the event path.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path


def git_sha() -> str:
    """Current git commit SHA, or ``"unknown"`` outside a repo / without git."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except OSError:
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip() or "unknown"


def package_version() -> str:
    """Installed ``swarm-perception`` version, or ``"unknown"``."""
    try:
        return importlib_metadata.version("swarm-perception")
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


def utc_now_iso() -> str:
    """Current UTC time, second precision (metadata only, never events)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
