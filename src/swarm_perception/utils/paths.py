"""Shared project paths used by runtime, logging, and experiments."""

from pathlib import Path

# paths.py lives at src/swarm_perception/utils/paths.py, so parents[2] is src/.
SRC_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = SRC_DIR.parent

LOG_DIR = PROJECT_ROOT / "logs"
OUTPUT_DIR = PROJECT_ROOT / "output"
ASSETS_DIR = SRC_DIR / "assets"
CONFIG_PATH = PROJECT_ROOT / "configs"

# No import-time side effects: output directories are created where the run
# directory is created (ObservationLogger / future run logger), never on import.