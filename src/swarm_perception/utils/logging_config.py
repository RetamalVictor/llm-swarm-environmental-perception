"""Central logging setup for simulation and experiment scripts."""

import logging
import logging.config

from swarm_perception.utils.paths import LOG_DIR

LOG_DIR.mkdir(exist_ok=True)


class CategoryFilter(logging.Filter):
    """Map logger names to short console category labels."""

    _CATEGORIES = {
        "swarm.config": "CONFIG",
        "swarm.sim": "SIM",
        "swarm.comm": "COMM",
        "swarm.llm": "LLM",
    }

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name in self._CATEGORIES:
            record.category = self._CATEGORIES[record.name]
        elif record.name.startswith("llm."):
            record.category = "LLM"
        elif record.name.endswith("config") or "config" in record.name:
            record.category = "CONFIG"
        else:
            record.category = "SIM"
        return True


LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "filters": {
        "category": {
            "()": CategoryFilter,
        },
    },
    "formatters": {
        "console": {
            "format": "%(asctime)s │ %(levelname)-5s │ %(category)-6s │ %(message)s",
            "datefmt": "%H:%M:%S",
        },
        "file": {
            "format": "%(asctime)s │ %(levelname)-5s │ %(category)-6s │ %(name)s │ %(message)s",
        },
        "detailed": {
            "format": "%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "console",
            "level": "INFO",
            "filters": ["category"],
        },
        "file_info": {
            "class": "logging.FileHandler",
            "formatter": "file",
            "filename": str(LOG_DIR / "app.log"),
            "level": "DEBUG",
            "filters": ["category"],
        },
        "file_error": {
            "class": "logging.FileHandler",
            "formatter": "detailed",
            "filename": str(LOG_DIR / "error.log"),
            "level": "ERROR",
        },
    },
    "loggers": {
        "swarm.comm": {
            "level": "DEBUG",
            "handlers": ["file_info"],
            "propagate": False,
        },
    },
    "root": {
        "handlers": ["console", "file_info", "file_error"],
        "level": "INFO",
    },
}


def setup_logging() -> None:
    """Configure root logging handlers for console, info, and error logs."""
    logging.config.dictConfig(LOGGING_CONFIG)
