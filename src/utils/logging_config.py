"""Central logging setup for simulation and experiment scripts."""

import logging
import logging.config

from utils.paths import LOG_DIR
LOG_DIR.mkdir(exist_ok=True)

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,

    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
        "detailed": {
            "format": "%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s"
        },
    },

    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "level": "INFO",
        },
        "file_info": {
            "class": "logging.FileHandler",
            "formatter": "standard",
            "filename": str(LOG_DIR / "app.log"),
            "level": "INFO",
        },
        "file_error": {
            "class": "logging.FileHandler",
            "formatter": "detailed",
            "filename": str(LOG_DIR / "error.log"),
            "level": "ERROR",
        },
    },

    "root": {
        "handlers": ["console", "file_info", "file_error"],
        "level": "INFO",
    },
}
def setup_logging():
    """Configure root logging handlers for console, info, and error logs."""
    logging.config.dictConfig(LOGGING_CONFIG)
