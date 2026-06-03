"""Shared project paths used by runtime, logging, and experiments."""

from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SRC_DIR.parent

LOG_DIR = PROJECT_ROOT / "logs"
OUTPUT_DIR = PROJECT_ROOT / "output"
ASSETS_DIR = PROJECT_ROOT / "src" / "assets"  # regular runtime assets
# ASSETS_DIR = PROJECT_ROOT / "experiments" / "environments"  # alt experiments assets
CONFIG_PATH = PROJECT_ROOT / "configs"

# Ensure runtime output destinations exist before simulation starts.
LOG_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)