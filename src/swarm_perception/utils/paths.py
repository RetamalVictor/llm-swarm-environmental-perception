"""Shared project paths used by runtime, logging, and experiments."""

from pathlib import Path

# paths.py lives at src/swarm_perception/utils/paths.py, so parents[2] is src/.
SRC_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = SRC_DIR.parent

LOG_DIR = PROJECT_ROOT / "logs"
OUTPUT_DIR = PROJECT_ROOT / "output"
ASSETS_DIR = SRC_DIR / "assets"  # regular runtime assets (src/assets)
# ASSETS_DIR = PROJECT_ROOT / "experiments" / "environments"  # alt experiments assets
CONFIG_PATH = PROJECT_ROOT / "configs"

# Ensure runtime output destinations exist before simulation starts.
LOG_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)